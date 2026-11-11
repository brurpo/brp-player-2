//! Effects that cannot run live: AI models and the shader effects stacked before them.
//!
//! To the user these are ordinary entries of the effect stack. The stack is split at
//! its last AI entry: everything up to it runs here, on a worker thread, as a list of
//! stages (groups of shader effects, AI models); shader effects after it run live in
//! the renderer on the result.
//!
//! Everything runs on the window's own GPU device and stays there: decoded frames are
//! uploaded once and converted by the effect chain, then processed as GPU tensors
//! (our Vulkan network runner for the models, compute kernels for motion, the
//! temporal window and motion blur). The result is a GPU picture the window draws
//! directly. Nothing comes back to the CPU.
//!
//! The window asks for the picture at (frame, sub-position); until the result is ready
//! the decoded frame is shown. Only the latest request is processed (scrubbing skips
//! stale work), and results are cached per stack.
//!
//! Time is counted in ticks of 1/subs frame (subs = product of the interpolation
//! factors). Each stage outputs pictures on its own grid: source frames every `subs`
//! ticks, finer after each interpolation.

use brp_ai::gpu::{Matting, Ops, Rife, Upscaler};
use brp_ai::gpu_temporal::Temporal;
use brp_ai::rife::RifeModel;
use brp_ai::upscale::UpscaleModel;
use brp_ai::vk::Tensor;
use brp_media::{ColorInfo, Source, VideoFrame};
use brp_render::compute::{Buffer, Compute, Picture, Recorder};
use brp_render::offline_fx::{FxInput, OfflineEffects};
use brp_render::usershader::{Param, ParamKind, ShaderFile};
use brp_render::{EffectSpec, GpuContext, Space, Transfer};
use parking_lot::{Condvar, Mutex};
use std::collections::{HashMap, VecDeque};
use std::sync::Arc;

// ---- AI effects as stack entries -------------------------------------------------

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum AiKind {
    Upscale,
    Interpolate,
    MotionBlur,
    Matting,
    /// Our denoiser (`brp_ai::denoise`): before upscaling, or alone.
    Denoise,
    /// NAFNet (REDS): motion and focus blur removed (`brp_ai::restore`).
    Deblur,
    /// SCUNet: noise, blocking and compression artefacts removed (`brp_ai::restore`).
    CleanUp,
    /// Video Depth Anything / Depth Anything V2 and an effect from the depth
    /// (`brp_ai::depth`).
    Depth,
    /// The camera's shake removed along the motion (`brp_ai::stabilise`).
    Stabilise,
    /// Detail that comes and goes from frame to frame averaged along the motion
    /// (`Temporal::steady_detail`).
    Steady,
}

const UPSCALE_MODELS: [UpscaleModel; 11] = [UpscaleModel::Fast2x, UpscaleModel::Fast3x, UpscaleModel::Fast4x, UpscaleModel::General2x, UpscaleModel::General4x, UpscaleModel::Anime4x, UpscaleModel::Natural2x, UpscaleModel::GeneralV3, UpscaleModel::Swin4x, UpscaleModel::Dat4x, UpscaleModel::DatReal4x];
const INTERP_FACTORS: [u8; 4] = [2, 3, 4, 8];

impl AiKind {
    pub const ALL: [AiKind; 10] = [AiKind::Denoise, AiKind::CleanUp, AiKind::Deblur, AiKind::Steady, AiKind::Stabilise, AiKind::Upscale, AiKind::Interpolate, AiKind::MotionBlur, AiKind::Matting, AiKind::Depth];

    /// Stable name for presets and the command line.
    pub fn id(self) -> &'static str {
        match self {
            AiKind::Upscale => "upscale",
            AiKind::Interpolate => "interpolate",
            AiKind::MotionBlur => "motion_blur",
            AiKind::Matting => "matting",
            AiKind::Denoise => "denoise",
            AiKind::Deblur => "deblur",
            AiKind::CleanUp => "cleanup",
            AiKind::Depth => "depth",
            AiKind::Stabilise => "stabilise",
            AiKind::Steady => "steady",
        }
    }

    pub fn from_id(s: &str) -> Option<AiKind> {
        Self::ALL.into_iter().find(|k| k.id() == s)
    }

    /// Whether parameter `i` applies with these values (the panel hides it if not).
    pub fn param_shown(self, i: usize, values: &[f64]) -> bool {
        let v = |j: usize| values.get(j).copied().unwrap_or(0.0);
        match (self, i) {
            // Denoise: only the model that has it.
            (AiKind::Upscale, 4) => UPSCALE_MODELS.get(v(0).round() as usize) == Some(&UpscaleModel::GeneralV3),
            // Strength, Window and Grain smoothing: only with Temporal on.
            (AiKind::Upscale, 3 | 5 | 6) => v(2) >= 0.5,
            // Depth: Focus for lens blur and the key; Amount for all but the map.
            (AiKind::Depth, 2) => matches!(v(1).round() as i64, 1 | 3),
            (AiKind::Depth, 3) => v(1).round() as i64 != 0,
            // Matting: steadiness for BiRefNet only.
            (AiKind::Matting, 2) => v(1) < 0.5,
            _ => true,
        }
    }

    /// Colour space the model works in (DESIGN.md §7a). All of today's models were
    /// trained on display pictures: they see the picture as the SDR viewer shows it,
    /// HDR mapped there and back invertibly. A model trained on linear light (the
    /// planned learned denoiser) says `Space::Linear`.
    pub fn space(self) -> brp_render::Space {
        brp_render::Space::Display
    }

    /// i18n key of the name shown in the stack.
    pub fn title_key(self) -> &'static str {
        match self {
            AiKind::Upscale => "fx_ai_upscale",
            AiKind::Interpolate => "fx_ai_interp",
            AiKind::MotionBlur => "fx_ai_blur",
            AiKind::Matting => "fx_ai_matting",
            AiKind::Denoise => "fx_ai_denoise",
            AiKind::Deblur => "fx_ai_deblur",
            AiKind::CleanUp => "fx_ai_cleanup",
            AiKind::Depth => "fx_ai_depth",
            AiKind::Stabilise => "fx_ai_stabilise",
            AiKind::Steady => "fx_ai_steady",
        }
    }

    /// Parameters in the same form as a shader's, so the panel shows them the same way.
    /// "(options: a|b)" in the description makes an int a list.
    pub fn file(self) -> Arc<ShaderFile> {
        fn p(name: &str, desc: &str, kind: ParamKind, default: f64, min: f64, max: f64) -> Param {
            Param { name: name.into(), desc: desc.into(), kind, default, min: Some(min), max: Some(max) }
        }
        let params = match self {
            AiKind::Upscale => vec![
                p("Quality", "Fast or best (best is slower); Natural is fast and made for real-world footage (Natural 4× noise removal is adjustable); Fast and Anime are made for drawn pictures; Transformer 4× gives the finest detail on real-world footage, Transformer Restore 4× cleans noise and compression while staying true to the picture (SwinIR invents crisper edges); Transformer HQ 4× is for clean footage only (it keeps noise and compression as they are); the transformers are far slower than the others (options: Fast 2×|Fast 3×|Fast 4×|Best 2×|Best 4×|Anime 4×|Natural 2×|Natural 4×|Transformer 4×|Transformer HQ 4×|Transformer Restore 4×)", ParamKind::Int, 0.0, 0.0, 10.0),
                p("Upscale", "On: the picture gets the new resolution. Off: it is scaled back to its size, with the recovered detail.", ParamKind::Int, 0.0, 0.0, 1.0),
                p("Temporal", "Steadier detail, no shimmering: each frame is first merged with its neighbours (2 before, 2 after) where they agree, then upscaled (options: Off|On)", ParamKind::Int, 1.0, 0.0, 1.0),
                p("Strength", "How much of the neighbours is merged in (lower keeps more of each frame's own grain and detail)", ParamKind::Float, 1.0, 0.0, 1.0),
                p("Denoise", "Natural 4× only: noise removal (lower keeps more grain and fine texture, higher is cleaner but can look smoothed)", ParamKind::Float, 0.5, 0.0, 1.0),
                p("Window", "Pictures merged each way (Temporal): more steadies noise and grain further, at some cost in speed", ParamKind::Int, 2.0, 1.0, 5.0),
                p("Grain smoothing", "Smooths the grain left after merging (Temporal), by how strong the grain measures: the upscalers otherwise draw faint grain as texture that changes every frame. 0 is off; clean video is left almost untouched", ParamKind::Float, 1.0, 0.0, 2.0),
                p("Motion", "How the motion between frames is found. Built-in: fast, and lines frames up best, grain included. SEA-RAFT (AI): follows the real motion far more accurately, small moving things included; about 40 ms per frame at 1080p. WAFT (AI): the most accurate, particles and fine detail included, but slow: about 0.7 s per frame at 1080p (options: Built-in|SEA-RAFT (AI)|WAFT (AI))", ParamKind::Int, 0.0, 0.0, 2.0),
            ],
            AiKind::Interpolate => vec![
                p("Slowdown", "How many times slower: new frames are made between the existing ones, and the clip gets that many times longer (options: 2×|3×|4×|8×)", ParamKind::Int, 0.0, 0.0, 3.0),
                p("Quality", "(options: Best|Fast)", ParamKind::Int, 0.0, 0.0, 1.0),
                p("Motion", "How the new frames are made. Built-in, SEA-RAFT (AI), WAFT (AI): the frames are moved along that motion, with frame interpolation (RIFE) only where it fails: sharper, grain and fine texture kept, small fast things placed where they are. SEA-RAFT follows real motion far more accurately (about 40 ms per frame at 1080p); WAFT the most accurate, slow (about 0.7 s). RIFE only: the frame interpolation network alone (options: Built-in|SEA-RAFT (AI)|WAFT (AI)|RIFE only)", ParamKind::Int, 0.0, 0.0, 3.0),
            ],
            AiKind::MotionBlur => vec![
                p("Quality", "Fast: blur along the motion. Best: the in-between moments are made and averaged, as a real shutter (much slower) (options: Fast|Best)", ParamKind::Int, 0.0, 0.0, 1.0),
                p("Shutter", "Shutter angle in degrees (180 = open half of the frame time)", ParamKind::Float, 180.0, 0.0, 360.0),
                p("Samples", "Samples along the motion (more: smoother); Best makes one in-between picture per 4 samples", ParamKind::Int, 32.0, 4.0, 64.0),
                p("Motion", "How the motion between frames is found (Best moves the frames along it to make its in-between pictures, with frame interpolation only where the motion fails). Built-in: fast, and lines frames up best, grain included. SEA-RAFT (AI): follows the real motion far more accurately, small moving things included; about 40 ms per frame at 1080p. WAFT (AI): the most accurate, particles and fine detail included, but slow: about 0.7 s per frame at 1080p (options: Built-in|SEA-RAFT (AI)|WAFT (AI))", ParamKind::Int, 0.0, 0.0, 2.0),
            ],
            AiKind::Denoise => vec![
                p("Window", "Pictures merged each way, moved onto this one along the motion (more: cleaner and steadier, a little slower)", ParamKind::Int, 3.0, 1.0, 5.0),
                p("Temporal", "How much of the merged neighbours is used (0: this picture alone)", ParamKind::Float, 1.0, 0.0, 1.0),
                p("Spatial", "Noise removal within the picture, against the noise measured on it (0: off; higher removes more, and more fine texture with it). Clean video is left almost untouched", ParamKind::Float, 1.0, 0.0, 3.0),
                p("Colour", "Colour noise removal, relative to Spatial", ParamKind::Float, 1.0, 0.0, 3.0),
                p("Motion", "How the motion between frames is found. Built-in: fast, and lines frames up best, grain included. SEA-RAFT (AI): follows the real motion far more accurately, small moving things included; about 40 ms per frame at 1080p. WAFT (AI): the most accurate, particles and fine detail included, but slow: about 0.7 s per frame at 1080p (options: Built-in|SEA-RAFT (AI)|WAFT (AI))", ParamKind::Int, 0.0, 0.0, 2.0),
                p("Impulses", "Removes isolated pixels far brighter or darker than those around them: fireflies in 3D renders, salt and pepper, hot or dead pixels (0: off; higher catches fainter ones). A small highlight also in the neighbouring pictures, such as a star or a glint, is kept", ParamKind::Float, 0.5, 0.0, 1.0),
            ],
            AiKind::Deblur => vec![p(
                "Strength",
                "How much of the deblurred picture is used (NAFNet, trained on video frames with motion blur and compression: removes motion and focus blur). Heavy: about 0.3 s per 1080p frame on a fast card",
                ParamKind::Float,
                1.0,
                0.0,
                1.0,
            )],
            AiKind::CleanUp => vec![p(
                "Strength",
                "How much of the cleaned picture is used (SCUNet: removes noise, blocking and compression artefacts from web and low-bitrate video; smooths a little fine texture). About 0.25 s per 1080p frame on a fast card",
                ParamKind::Float,
                1.0,
                0.0,
                1.0,
            )],
            AiKind::Depth => vec![
                p("Model", "Video: steady from frame to frame (Video Depth Anything; each frame looks back at earlier ones). Per frame: each frame alone, a little faster, but the depth flickers (options: Video (steady)|Per frame)", ParamKind::Int, 0.0, 0.0, 1.0),
                p("Output", "Depth map: near is white. Lens blur: what is nearer or farther than the focus is blurred, as a wide-open lens. Fog: distance fades to grey. Depth key: what is behind the focus becomes transparent (options: Depth map|Lens blur|Fog|Depth key)", ParamKind::Int, 1.0, 0.0, 3.0),
                p("Focus", "Depth in focus (lens blur) or where the key cuts (depth key): 0 is the farthest part of the picture, 1 the nearest", ParamKind::Float, 0.8, 0.0, 1.0),
                p("Amount", "Lens blur: size of the blur. Fog: density. Depth key: softness of the edge", ParamKind::Float, 0.5, 0.0, 1.0),
            ],
            AiKind::Steady => vec![
                p("Frames", "Frames each way whose detail is averaged with this one's, moved onto it along the motion (more: steadier, a little softer and slower)", ParamKind::Int, 4.0, 1.0, 8.0),
                p("Detail size", "Size in pixels of the detail that is steadied: everything finer than this is averaged over the frames, the broader picture stays this frame's. Raise it when coarser things still shimmer (thin lines, signs); higher softens more", ParamKind::Float, 4.0, 0.5, 8.0),
                p("Keep sharpness", "How much of the detail strength lost to averaging is brought back (0: none, softest and steadiest; 1: all of it, measured per area)", ParamKind::Float, 1.0, 0.0, 1.0),
                p("Strength", "Mix with the original picture", ParamKind::Float, 1.0, 0.0, 1.0),
                p("Motion", "How the motion between frames is found. Built-in: fast. SEA-RAFT (AI): more accurate on fine detail, about 40 ms per frame at 1080p. WAFT (AI): the most accurate, about 0.7 s per frame (options: Built-in|SEA-RAFT (AI)|WAFT (AI))", ParamKind::Int, 0.0, 0.0, 2.0),
            ],
            AiKind::Stabilise => vec![
                p("Smoothness", "Frames each way the camera's path is smoothed over: more removes slower wobble too, but follows deliberate moves later", ParamKind::Int, 15.0, 3.0, 60.0),
                p("Zoom", "Zoom in, in percent, so the edges the correction uncovers stay outside the picture (0: none; what is uncovered repeats the edge)", ParamKind::Float, 5.0, 0.0, 25.0),
                p("Motion", "How the motion between frames is found. Built-in: fast, and enough for the camera's movement. SEA-RAFT (AI): more robust with large moving things, about 40 ms per frame at 1080p. WAFT (AI): the most accurate, about 0.7 s per frame (options: Built-in|SEA-RAFT (AI)|WAFT (AI))", ParamKind::Int, 0.0, 0.0, 2.0),
            ],
            AiKind::Matting => vec![
                p("Output", "What replaces the background (options: Transparent|Green screen|Blue screen|Matte (alpha))", ParamKind::Int, 0.0, 0.0, 3.0),
                p("Model", "BiRefNet: finer edges (hair, glass, see-through things) on any subject. RVM: made for people on video, faster, softer edges (options: BiRefNet|RVM)", ParamKind::Int, 0.0, 0.0, 1.0),
                p("Steadiness", "BiRefNet finds each frame's matte on its own; this averages it with this many frames each way, moved along the motion, where they match (0: off). Steadies shimmering edges; more is steadier but softer on fast movement", ParamKind::Int, 2.0, 0.0, 4.0),
            ],
        };
        Arc::new(ShaderFile { name: format!("ai:{}", self.id()), params, ..Default::default() })
    }
}

/// One step of the non-live part of the stack.
#[derive(Clone)]
pub enum Stage {
    /// Consecutive shader effects (enabled ones only).
    Shaders(Vec<EffectSpec>),
    Ai { kind: AiKind, params: Vec<f64> },
}

impl Stage {
    fn interp_factor(&self) -> i64 {
        match self {
            Stage::Ai { kind: AiKind::Interpolate, params } => INTERP_FACTORS[(params[0] as usize).min(3)] as i64,
            _ => 1,
        }
    }

    fn hash(&self, h: &mut impl std::hash::Hasher) {
        use std::hash::Hash;
        match self {
            Stage::Shaders(specs) => {
                0u8.hash(h);
                for s in specs {
                    (s.id, &s.file.name, s.file.passes.len()).hash(h);
                    s.params.iter().for_each(|v| v.to_bits().hash(h));
                }
            }
            Stage::Ai { kind, params } => {
                1u8.hash(h);
                kind.hash(h);
                params.iter().for_each(|v| v.to_bits().hash(h));
            }
        }
    }
}

/// Temporal upscaling: pictures merged each way (the Window parameter, 1..5).
/// Where an effect's motion between frames comes from (its "Motion" setting).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Motion {
    /// Our Lucas-Kanade motion (`gpu_temporal`): about 3 ms per 1080p pair; it lines
    /// frames up best, grain included (what merging needs).
    Builtin = 0,
    /// SEA-RAFT (`brp_ai::flow`): about 40 ms per pair; far more accurate real motion,
    /// small moving things included.
    SeaRaft = 1,
    /// WAFT (`brp_ai::flow`): the most accurate (particles too), about 0.7 s per pair.
    Waft = 2,
}

impl Motion {
    pub(crate) fn from_param(v: Option<&f64>) -> Motion {
        match v.map(|v| v.round() as i64) {
            Some(1) => Motion::SeaRaft,
            Some(2) => Motion::Waft,
            _ => Motion::Builtin,
        }
    }
}

/// Stabilisation: frames each way its path is smoothed over.
fn stab_radius(params: &[f64]) -> i64 {
    params.first().map_or(15, |v| v.round() as i64).clamp(3, 60)
}

/// Depth: earlier pictures run after a jump so the video model has a history.
const DEPTH_REPLAY: i64 = 4;

/// Steady Detail: frames each way.
fn steady_radius(params: &[f64]) -> i64 {
    params.first().map_or(4, |v| v.round() as i64).clamp(1, 8)
}

/// BiRefNet matting: neighbours each way its alpha is steadied with.
fn matte_radius(params: &[f64]) -> i64 {
    params.get(2).map_or(2, |v| v.round() as i64).clamp(0, 4)
}

fn window_radius(params: &[f64]) -> i64 {
    params.get(5).map_or(2, |v| v.round() as i64).clamp(1, 5)
}

pub(crate) fn rife_model(v: f64) -> RifeModel {
    if v >= 0.5 { RifeModel::V4_25Lite } else { RifeModel::V4_26 }
}

/// Most pictures per frame (indices of AI pictures reserve 64 per frame).
const MAX_SUBS: i64 = 64;

/// How far around a frame the stack reads source frames.
pub(crate) struct Reach {
    pub interp: i64,
    pub blur: i64,
    pub matte: i64,
    /// Temporal upscaling and denoising look their window radius each way (stages
    /// add up).
    pub window: i64,
    /// Deinterlacing reads each frame's neighbours.
    pub deint: i64,
    /// Depth: earlier pictures replayed after a jump (as matting).
    pub depth: i64,
    /// Stabilisation: its smoothing radius each way (stages add up).
    pub stab: i64,
}

impl Reach {
    pub fn of(stages: &[Stage], deint: bool) -> Reach {
        let count = |k: AiKind| stages.iter().filter(|s| matches!(s, Stage::Ai { kind, .. } if *kind == k)).count() as i64;
        let window: i64 = stages
            .iter()
            .filter_map(|s| match s {
                Stage::Ai { kind: AiKind::Upscale, params } if params.get(2).is_none_or(|&v| v >= 0.5) => Some(window_radius(params)),
                Stage::Ai { kind: AiKind::Denoise, params } => Some(params.first().map_or(3, |v| v.round() as i64).clamp(1, 5)),
                Stage::Ai { kind: AiKind::Matting, params } if params.get(1).is_none_or(|&v| v < 0.5) => Some(matte_radius(params)),
                _ => None,
            })
            .sum();
        let stab: i64 = stages
            .iter()
            .filter_map(|s| match s {
                Stage::Ai { kind: AiKind::Stabilise, params } => Some(stab_radius(params)),
                Stage::Ai { kind: AiKind::Steady, params } => Some(steady_radius(params)),
                _ => None,
            })
            .sum();
        // RVM replays earlier pictures after a jump; BiRefNet needs none.
        let rvm = stages.iter().filter(|s| matches!(s, Stage::Ai { kind: AiKind::Matting, params } if params.get(1).is_some_and(|&v| v >= 0.5))).count() as i64;
        Reach { interp: count(AiKind::Interpolate), blur: count(AiKind::MotionBlur), matte: rvm, window, deint: deint as i64, depth: count(AiKind::Depth), stab }
    }

    /// Source frames read before and after a frame.
    pub fn frames(&self) -> (i64, i64) {
        (self.blur + self.window + 4 * self.matte + DEPTH_REPLAY * self.depth + self.stab + self.deint, self.interp + self.blur + self.window + self.stab + self.deint)
    }
}

/// Pictures per source frame for a stack: fields (field-rate deinterlacing) times
/// the slow-motion factors.
pub(crate) fn subs_of(stages: &[Stage], deint: Option<SourceDeint>) -> i64 {
    let fields = if deint.is_some_and(|d| d.fields) { 2 } else { 1 };
    (fields * stages.iter().map(Stage::interp_factor).product::<i64>().min(MAX_SUBS)).min(MAX_SUBS)
}

/// How long the stack makes the clip (slow motion).
pub(crate) fn slowdown_of(stages: &[Stage]) -> i64 {
    stages.iter().map(Stage::interp_factor).product::<i64>().min(MAX_SUBS)
}

/// History pictures stateful shader effects in the stack need after a jump.
pub(crate) fn shader_warmup(stages: &[Stage]) -> i64 {
    stages
        .iter()
        .filter_map(|s| match s {
            Stage::Shaders(specs) => specs.iter().filter(|e| e.file.is_stateful()).map(|e| e.file.warmup.unwrap_or(2) as i64).max(),
            _ => None,
        })
        .max()
        .unwrap_or(0)
}

/// Colour space of each stage's output (index 0: the source picture; DESIGN.md §7a).
/// A picture is stored already in the space the next stage works in (neighbours with
/// the same space share one conversion; a group of shader effects takes any space and
/// converts lazily inside). The output of the last AI stage, and of the whole stack,
/// is in the source's encoding: there the window's live shader chain takes over, and
/// export converts at the same point, so both stay identical.
pub(crate) fn stage_spaces(stages: &[Stage], color: &ColorInfo) -> Vec<brp_render::Space> {
    use brp_render::Space;
    let n = stages.len();
    let last_ai = stages.iter().rposition(|s| matches!(s, Stage::Ai { .. })).map(|i| i + 1);
    let mut spaces = Vec::with_capacity(n + 1);
    for s in 0..=n {
        // What stage s makes by itself (from its input's space).
        let own = match s.checked_sub(1).map(|i| &stages[i]) {
            None => Space::Source,
            Some(Stage::Ai { kind, .. }) => kind.space(),
            Some(Stage::Shaders(specs)) => brp_render::effects::out_space(specs, spaces[s - 1], None, color),
        };
        let space = if s == n || Some(s) == last_ai {
            Space::Source
        } else {
            match &stages[s] {
                Stage::Ai { kind, .. } => kind.space(),
                Stage::Shaders(_) => own,
            }
        };
        spaces.push(space);
    }
    spaces
}

/// Identity of a stack's results (what the caches are keyed by).
/// `source`: which pictures (the window's file, and its EXR layer; 0 for export,
/// which has one source per worker).
fn stack_key(stages: &[Stage], transfer: Transfer, deint: Option<SourceDeint>, source: u64) -> u64 {
    use std::hash::{Hash, Hasher};
    let mut h = std::collections::hash_map::DefaultHasher::new();
    source.hash(&mut h);
    transfer.hash(&mut h);
    stages.iter().for_each(|s| s.hash(&mut h));
    deint.map(|d| (d.method as u8, d.tff, d.fields)).hash(&mut h);
    h.finish()
}


// ---- window side -----------------------------------------------------------------

/// Position of a picture: source frame and sub-position (0..subs).
type Pos = (i64, u8);

pub(crate) struct Job {
    key: u64,
    /// Which pictures (see `stack_key`).
    source: u64,
    stages: Arc<Vec<Stage>>,
    /// Colour space of each stage's output (0 = the source picture), see `stage_spaces`.
    spaces: Vec<brp_render::Space>,
    transfer: Transfer,
    subs: i64,
    pos: Pos,
    /// Decoded source frames around the position.
    frames: Vec<Arc<VideoFrame>>,
    last_frame: i64,
    color: ColorInfo,
    /// The window's GPU device (pictures are made where they are shown).
    ctx: Arc<GpuContext>,
    deint: Option<SourceDeint>,
}

type SourcePicture = (Arc<VideoFrame>, Option<Arc<VideoFrame>>, Option<Arc<VideoFrame>>, Option<brp_render::DeintParams>);

impl Job {
    /// A job outside the window's queue (export): the picture at `tick`, from the
    /// decoded `frames` around it.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn new(stages: Arc<Vec<Stage>>, transfer: Transfer, deint: Option<SourceDeint>, tick: i64, frames: Vec<Arc<VideoFrame>>, last_frame: i64, color: ColorInfo, ctx: Arc<GpuContext>) -> Job {
        let subs = subs_of(&stages, deint);
        Job { key: stack_key(&stages, transfer, deint, 0), source: 0, spaces: stage_spaces(&stages, &color), pos: (tick.div_euclid(subs), tick.rem_euclid(subs) as u8), stages, transfer, subs, frames, last_frame, color, ctx, deint }
    }

    /// Source pictures per frame: 2 when each field becomes a picture.
    fn fields(&self) -> i64 {
        if self.deint.is_some_and(|d| d.fields) { 2 } else { 1 }
    }

    /// The source frame at `tick` with its neighbours and how to deinterlace it (the
    /// field: the second one in the second half of the frame with field rate).
    fn source_at(&self, tick: i64) -> Result<SourcePicture, MissingFrame> {
        let n = tick / self.subs;
        let find = |i: i64| self.frames.iter().find(|f| f.index() == i).cloned();
        let f = find(n).ok_or(MissingFrame(n))?;
        let Some(d) = self.deint else { return Ok((f, None, None, None)) };
        // Neighbours inside the clip must be there (at its ends the frame stands in).
        let (prev, next) = (find(n - 1), find(n + 1));
        if n > 0 && prev.is_none() {
            return Err(MissingFrame(n - 1));
        }
        if n < self.last_frame && next.is_none() {
            return Err(MissingFrame(n + 1));
        }
        let second_field = d.fields && (tick % self.subs) >= self.subs / 2;
        let params = brp_render::DeintParams { method: d.method, tff: d.tff.unwrap_or(f.tff || !f.interlaced), second_field, light: false };
        Ok((f, prev, next, Some(params)))
    }
}

enum Outcome {
    Done(Arc<Picture>),
    /// A source frame the stack needs is not decoded (yet).
    Missing,
    Failed(String),
}

struct Done {
    key: u64,
    pos: Pos,
    outcome: Outcome,
    took: std::time::Duration,
    /// Source frames the job had.
    decoded: usize,
    /// The finished picture's pixels, read back for the RAM cache.
    ram: Option<Arc<RamPicture>>,
    /// Put back on the GPU from the RAM cache (not computed).
    restored: bool,
}

/// A finished picture in system RAM: RGBA16F, rows packed (as `Picture` holds it).
pub(crate) struct RamPicture {
    w: u32,
    h: u32,
    alpha: bool,
    data: Vec<u8>,
}

/// A picture to put back on the GPU from the RAM cache.
struct Restore {
    key: u64,
    pos: Pos,
    pic: Arc<RamPicture>,
    ctx: Arc<GpuContext>,
}

struct Shared {
    /// Pictures to make, most urgent first (replaced by the window as it moves on).
    queue: Mutex<VecDeque<Job>>,
    /// Picture being made.
    current: Mutex<Option<(u64, Pos)>>,
    wake: Condvar,
    done: Mutex<Vec<Done>>,
    /// Pictures to upload from the RAM cache, in order (replaced like `queue`); their
    /// own thread, so a long computation does not hold them up.
    restores: Mutex<VecDeque<Restore>>,
    /// Picture being uploaded.
    restoring: Mutex<Option<(u64, Pos)>>,
    restore_wake: Condvar,
    /// Set when the window closes: the worker drops its models and ends.
    quit: std::sync::atomic::AtomicBool,
}

/// Deinterlacing of the source, done before the whole stack.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SourceDeint {
    pub method: brp_render::DeintMethod,
    /// Field order detected from the content (else each frame's flag).
    pub tff: Option<bool>,
    /// Each field becomes a picture (field rate: 50i -> 50p).
    pub fields: bool,
}

/// How the window is playing, for working ahead of the playhead.
#[derive(Clone, Copy, Debug)]
pub struct Playback {
    /// Seconds between pictures on screen.
    pub interval: f64,
    /// 1 forward, -1 backward.
    pub dir: i8,
}

pub struct Ai {
    stages: Arc<Vec<Stage>>,
    transfer: Transfer,
    deint: Option<SourceDeint>,
    /// The window's source (file and EXR layer): part of the key.
    source: u64,
    key: u64,
    subs: i64,
    shared: Arc<Shared>,
    cache: HashMap<(u64, Pos), Arc<Picture>>,
    order: VecDeque<(u64, Pos)>,
    cache_bytes: usize,
    /// Finished pictures in system RAM by (stack key, position), with when each was
    /// last used: kept across stack changes (undo, or the same settings again, finds
    /// them), least recently used dropped over `mem::fx_cache_budget`.
    ram: HashMap<(u64, Pos), (Arc<RamPicture>, u64)>,
    ram_bytes: usize,
    ram_clock: u64,
    /// Size of the last picture made (limits how far ahead to work).
    last_bytes: usize,
    /// Pictures whose source frames were missing, with how many were decoded then
    /// (asked again only once more are).
    missing: HashMap<(u64, Pos), usize>,
    /// Typical time to make a picture (seconds).
    cost: f64,
    /// Last picture shown, held during playback until the next one is ready.
    held: Option<(u64, Pos, Arc<Picture>)>,
    pub error: Option<String>,
    /// The window's GPU device.
    pub ctx: Option<Arc<GpuContext>>,
    threads: Vec<std::thread::JoinHandle<()>>,
}

/// Finished pictures kept (GPU memory, RGBA16F).
const CACHE_BYTES: usize = 1 << 30;
/// Most pictures queued ahead of the playhead.
const AHEAD: usize = 24;
/// The RAM cache stops growing when less than this is free.
const RAM_RESERVE: u64 = 1536 << 20;

fn picture_bytes(p: &Picture) -> usize {
    p.w as usize * p.h as usize * 8
}

impl Ai {
    pub fn new() -> Self {
        let shared = Arc::new(Shared {
            queue: Mutex::new(VecDeque::new()),
            current: Mutex::new(None),
            wake: Condvar::new(),
            done: Mutex::new(Vec::new()),
            restores: Mutex::new(VecDeque::new()),
            restoring: Mutex::new(None),
            restore_wake: Condvar::new(),
            quit: false.into(),
        });
        let s = shared.clone();
        let worker = std::thread::Builder::new().name("brp-ai".into()).spawn(move || worker(s)).expect("ai thread");
        let s = shared.clone();
        let restorer = std::thread::Builder::new().name("brp-ai-restore".into()).spawn(move || restorer(s)).expect("ai restore thread");
        Ai {
            stages: Arc::new(Vec::new()),
            deint: None,
            transfer: Transfer::Sdr,
            source: 0,
            key: 0,
            subs: 1,
            shared,
            cache: HashMap::new(),
            order: VecDeque::new(),
            cache_bytes: 0,
            ram: HashMap::new(),
            ram_bytes: 0,
            ram_clock: 0,
            last_bytes: 0,
            missing: HashMap::new(),
            cost: 0.1,
            held: None,
            error: None,
            ctx: None,
            threads: vec![worker, restorer],
        }
    }

    /// Another source (a file opened, another EXR layer): new pictures everywhere.
    pub fn set_source(&mut self, id: u64) {
        self.source = id;
        self.rekey();
    }

    /// New non-live part of the stack. Returns true if the pictures per frame changed.
    pub fn set_stages(&mut self, stages: Vec<Stage>, transfer: Transfer) -> bool {
        self.stages = Arc::new(stages);
        self.transfer = transfer;
        self.rekey()
    }

    /// Deinterlacing of the source, done before the whole stack (None: progressive,
    /// or off). Returns true if the pictures per frame changed.
    pub fn set_source_deint(&mut self, deint: Option<SourceDeint>) -> bool {
        if deint == self.deint {
            return false;
        }
        self.deint = deint;
        self.rekey()
    }

    /// New key (cache identity) and pictures per frame after a setting changed.
    fn rekey(&mut self) -> bool {
        let key = stack_key(&self.stages, self.transfer, self.deint, self.source);
        let subs = subs_of(&self.stages, self.deint);
        let changed = subs != self.subs;
        if key != self.key {
            // The held picture stays: while paused it is shown until the new one is
            // ready (see `get`).
            self.error = None;
            self.shared.queue.lock().clear();
            self.shared.restores.lock().clear();
        }
        self.key = key;
        self.subs = subs;
        changed
    }

    fn slowdown_factor(&self) -> i64 {
        self.stages.iter().map(Stage::interp_factor).product::<i64>().min(MAX_SUBS)
    }

    /// How many times slower the clip plays (slow motion; fields do not slow it).
    pub fn slowdown(&self) -> u8 {
        if self.active() { self.slowdown_factor() as u8 } else { 1 }
    }

    pub fn active(&self) -> bool {
        !self.stages.is_empty()
    }

    /// Pictures per source frame (interpolation).
    pub fn subs(&self) -> u8 {
        if self.active() { self.subs as u8 } else { 1 }
    }

    /// Something is being computed (keep redrawing to pick it up).
    pub fn busy(&self) -> bool {
        self.shared.current.lock().is_some()
            || !self.shared.queue.lock().is_empty()
            || self.shared.restoring.lock().is_some()
            || !self.shared.restores.lock().is_empty()
    }

    /// Forget everything (another file: positions mean other pictures).
    pub fn clear(&mut self) {
        self.clear_gpu();
        self.ram.clear();
        self.ram_bytes = 0;
    }

    /// Forget the GPU side (another GPU). The RAM cache stays: its pictures go to
    /// whichever device shows them.
    pub fn clear_gpu(&mut self) {
        self.cache.clear();
        self.order.clear();
        self.cache_bytes = 0;
        self.missing.clear();
        self.held = None;
        self.error = None;
        self.shared.queue.lock().clear();
        self.shared.restores.lock().clear();
    }

    /// Frames with a processed picture of the current stack in RAM (timeline).
    pub fn ram_cached_frames(&self) -> Vec<i64> {
        if !self.active() {
            return Vec::new();
        }
        let mut v: Vec<i64> = self.ram.keys().filter(|(k, _)| *k == self.key).map(|(_, p)| p.0).collect();
        v.sort_unstable();
        v.dedup();
        v
    }

    /// Mark a RAM picture as just used (least recently used go first).
    fn touch(&mut self, k: (u64, Pos)) {
        if let Some(e) = self.ram.get_mut(&k) {
            self.ram_clock += 1;
            e.1 = self.ram_clock;
        }
    }

    /// Keep a finished picture in RAM; drop the least recently used over the budget
    /// (or when the system runs low on memory).
    fn keep_ram(&mut self, k: (u64, Pos), pic: Arc<RamPicture>) {
        let bytes = pic.data.len();
        self.ram_clock += 1;
        if let Some((old, _)) = self.ram.insert(k, (pic, self.ram_clock)) {
            self.ram_bytes -= old.data.len();
        }
        self.ram_bytes += bytes;
        let low = brp_media::mem::avail_ram().is_some_and(|a| a < RAM_RESERVE);
        let budget = if low { self.ram_bytes.saturating_sub(bytes) } else { brp_media::mem::fx_cache_budget() };
        if self.ram_bytes <= budget {
            return;
        }
        let mut by_age: Vec<((u64, Pos), u64)> = self.ram.iter().map(|(k, (_, t))| (*k, *t)).collect();
        by_age.sort_unstable_by_key(|e| e.1);
        for (old, _) in by_age {
            if self.ram_bytes <= budget * 9 / 10 || old == k {
                break;
            }
            if let Some((p, _)) = self.ram.remove(&old) {
                self.ram_bytes -= p.data.len();
            }
        }
    }

    fn collect(&mut self) {
        let key = self.key;
        let done: Vec<Done> = self.shared.done.lock().drain(..).collect();
        for d in done {
            if let Some(r) = d.ram {
                self.keep_ram((d.key, d.pos), r);
            }
            match d.outcome {
                Outcome::Failed(e) if d.restored => {
                    // Computed again instead.
                    log::warn!("AI: picture from RAM: {e}");
                    if let Some((p, _)) = self.ram.remove(&(d.key, d.pos)) {
                        self.ram_bytes -= p.data.len();
                    }
                }
                Outcome::Done(p) => {
                    if !d.restored {
                        self.cost = self.cost * 0.7 + d.took.as_secs_f64() * 0.3;
                    }
                    let bytes = picture_bytes(&p);
                    self.last_bytes = bytes;
                    if let Some(old) = self.cache.insert((d.key, d.pos), p) {
                        self.cache_bytes -= picture_bytes(&old);
                    } else {
                        self.order.push_back((d.key, d.pos));
                    }
                    self.cache_bytes += bytes;
                    while self.cache_bytes > CACHE_BYTES && self.order.len() > 2 {
                        if let Some(old) = self.order.pop_front() {
                            if let Some(p) = self.cache.remove(&old) {
                                self.cache_bytes -= picture_bytes(&p);
                            }
                        }
                    }
                    if d.key == key {
                        self.error = None;
                    }
                }
                Outcome::Failed(e) if d.key == key => {
                    log::error!("AI: {e}");
                    self.error = Some(e);
                }
                Outcome::Missing => {
                    self.missing.insert((d.key, d.pos), d.decoded);
                }
                Outcome::Failed(_) => {}
            }
        }
    }

    /// Playback waits for each processed picture (every picture is shown, in order,
    /// even if that is slower than real time). Not after an error: then the decoded
    /// frames play.
    pub fn paces(&self) -> bool {
        self.active() && self.error.is_none()
    }

    /// The processed picture at `pos` is ready.
    pub fn ready(&mut self, pos: Pos) -> bool {
        self.collect();
        self.cache.contains_key(&(self.key, pos))
    }

    /// The picture after `pos` in playing order.
    pub fn advance(&self, (n, sub): Pos, dir: i8) -> Pos {
        let subs = self.subs as u8;
        if dir >= 0 {
            if sub + 1 < subs { (n, sub + 1) } else { (n + 1, 0) }
        } else if sub > 0 {
            (n, sub - 1)
        } else {
            (n - 1, subs - 1)
        }
    }

    /// A job for `pos` if the source frames it needs are decoded. Earlier frames for
    /// warming up matting are included when decoded.
    fn job(&self, src: &Source, pos: Pos) -> Option<Job> {
        let ctx = self.ctx.clone()?;
        let (n, sub) = pos;
        let last = src.frame_count() - 1;
        if n < 0 || n > last {
            return None;
        }
        let r = Reach::of(&self.stages, self.deint.is_some());
        let (interp, blur, window, d) = (r.interp, r.blur, r.window, r.deint);
        let need_next = (sub > 0 && interp > 0) || blur > 0;
        let (back, fwd) = r.frames();
        let lo = (n - back).max(0);
        let hi = (n + fwd).min(last);
        let frames: Vec<Arc<VideoFrame>> = (lo..=hi).filter_map(|i| src.get(i)).collect();
        let has = |i: i64| i < 0 || i > last || frames.iter().any(|f| f.index() == i);
        let reach = window + d;
        if !has(n) || (need_next && !has(n + 1 + d)) || (blur > 0 && !has(n - 1 - d)) || !(n - reach..=n + reach).all(has) {
            log::trace!("AI: job {pos:?} waits for frames {:?}", (n - reach.max(back)..=n + reach.max(fwd)).filter(|&i| !has(i)).collect::<Vec<_>>());
            return None;
        }
        if self.missing.get(&(self.key, pos)).is_some_and(|&c| c >= frames.len()) {
            return None;
        }
        Some(Job {
            key: self.key,
            source: self.source,
            spaces: stage_spaces(&self.stages, &src.info.color),
            stages: self.stages.clone(),
            transfer: self.transfer,
            subs: self.subs,
            pos,
            frames,
            last_frame: last,
            color: src.info.color,
            ctx,
            deint: self.deint,
        })
    }

    /// The processed picture to show at `pos`, if there is one. Asks for the pictures
    /// needed next: `pos` itself when paused; while playing, `pos` and the pictures
    /// after it, in order (playback waits for each, see `paces`). While playing, the
    /// last processed picture stays on screen if the exact one is not there yet; while
    /// paused, the last one made for this frame (before a setting changed) does.
    pub fn get(&mut self, src: &Source, pos: Pos, playback: Option<Playback>) -> Option<Arc<Picture>> {
        self.collect();
        let key = self.key;
        let exact = self.cache.get(&(key, pos)).cloned();
        if exact.is_some() {
            self.touch((key, pos));
        }
        if self.error.is_none() {
            let current = *self.shared.current.lock();
            let restoring = *self.shared.restoring.lock();
            let mut jobs = VecDeque::new();
            let mut restores = VecDeque::new();
            let todo = |p: Pos| !self.cache.contains_key(&(key, p)) && current != Some((key, p)) && restoring != Some((key, p));
            // A picture kept in RAM is uploaded again, not computed.
            let from_ram = |p: Pos| -> Option<Restore> {
                let (pic, _) = self.ram.get(&(key, p))?;
                Some(Restore { key, pos: p, pic: pic.clone(), ctx: self.ctx.clone()? })
            };
            match playback {
                None => {
                    if todo(pos) {
                        match from_ram(pos) {
                            Some(r) => restores.push_back(r),
                            None => jobs.extend(self.job(src, pos)),
                        }
                    }
                }
                Some(pb) => {
                    // Playback waits for every picture: make them in order from the
                    // playhead (the decoder reads ahead to feed this). No further
                    // than half the cache holds, so pictures are not dropped before
                    // they are shown.
                    let ahead = (CACHE_BYTES / 2 / self.last_bytes.max(1)).clamp(2, AHEAD);
                    let mut p = pos;
                    // Not decoded yet: neither are the ones after it (those kept in
                    // RAM are still uploaded).
                    let mut blocked = false;
                    for _ in 0..=ahead {
                        if todo(p) {
                            match from_ram(p) {
                                Some(r) => restores.push_back(r),
                                None if blocked => {}
                                None => match self.job(src, p) {
                                    Some(j) => jobs.push_back(j),
                                    None => blocked = true,
                                },
                            }
                        }
                        p = self.advance(p, pb.dir);
                    }
                }
            }
            let notify = !jobs.is_empty();
            *self.shared.queue.lock() = jobs;
            if notify {
                self.shared.wake.notify_one();
            }
            let notify = !restores.is_empty();
            *self.shared.restores.lock() = restores;
            if notify {
                self.shared.restore_wake.notify_one();
            }
        }
        match (exact, playback) {
            (Some(f), _) => {
                self.held = Some((key, pos, f.clone()));
                Some(f)
            }
            (None, Some(pb)) => {
                // Hold the last picture made for this stack if it is recent (within a
                // second of playing) and on the right side of the playhead.
                let (k, p, f) = self.held.as_ref()?;
                let dist = ((pos.0 - p.0) * self.subs + pos.1 as i64 - p.1 as i64) * pb.dir as i64;
                let recent = dist >= 0 && (dist as f64) * pb.interval < 1.0;
                (*k == key && recent).then(|| f.clone())
            }
            // Paused, after a setting changed: the last picture made for this frame
            // stays until the new one is ready (not the unprocessed frame).
            (None, None) => {
                let (_, p, f) = self.held.as_ref()?;
                (*p == pos && self.error.is_none()).then(|| f.clone())
            }
        }
    }
}

impl Drop for Ai {
    /// Stops the worker; if it is making a picture it finishes it first.
    fn drop(&mut self) {
        // Under the queue's lock, so a worker about to wait cannot miss it.
        {
            let mut q = self.shared.queue.lock();
            q.clear();
            self.shared.quit.store(true, std::sync::atomic::Ordering::Relaxed);
            self.shared.wake.notify_all();
        }
        {
            let mut q = self.shared.restores.lock();
            q.clear();
            self.shared.restore_wake.notify_all();
        }
        for t in self.threads.drain(..) {
            let _ = t.join();
        }
    }
}

// ---- worker ----------------------------------------------------------------------

/// Which finished picture an export asks for (positions in ticks of 1/subs frame).
#[derive(Clone, Copy, Debug)]
pub(crate) enum Sample {
    /// A picture of the stack's own grid.
    Tick(i64),
    /// Mixed from the two pictures around a fractional tick (frame blending).
    Blend(f64),
    /// Made in between along the motion, with frame interpolation where it fails (as
    /// Slow Motion makes its pictures); `motion` None: frame interpolation alone.
    Motion { tau: f64, model: RifeModel, motion: Option<Motion> },
}

/// A source frame the stack needs is not decoded.
#[derive(Debug)]
pub(crate) struct MissingFrame(pub i64);

impl std::fmt::Display for MissingFrame {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "frame {} not decoded", self.0)
    }
}

impl std::error::Error for MissingFrame {}

struct Gpu {
    ctx: Arc<GpuContext>,
    compute: Arc<Compute>,
    ops: Arc<Ops>,
    tm: Temporal,
    dn: brp_ai::denoise::Denoiser,
}

struct DepthState {
    net: brp_ai::depth::Depth,
    last: Option<i64>,
}

/// Camera movements kept (4 numbers each).
const SIMS: usize = 16384;

struct MattingState {
    net: Matting,
    /// Tick of the last picture fed to the network (its state follows that picture).
    last: Option<i64>,
}

/// Id of the runner that converts source frames to RGB (no effects).
const MERGE: usize = usize::MAX;

/// For each stage, a hash of what its output depends on: the stages up to it, and what
/// sets the source pictures and their numbering (colour transfer, deinterlacing,
/// pictures per frame).
/// What the source pictures (stage 0) depend on: their colour, deinterlacing and the
/// tick grid (pictures per frame: tick t is frame t / subs, so a picture kept across
/// a change of subs would be another frame's).
fn source_key(job: &Job) -> u64 {
    use std::hash::{Hash, Hasher};
    let mut h = std::collections::hash_map::DefaultHasher::new();
    job.source.hash(&mut h);
    job.transfer.hash(&mut h);
    job.deint.map(|d| (d.method as u8, d.tff, d.fields)).hash(&mut h);
    job.subs.hash(&mut h);
    h.finish()
}

fn stage_keys(job: &Job) -> Vec<u64> {
    use std::hash::{Hash, Hasher};
    let mut h = std::collections::hash_map::DefaultHasher::new();
    job.transfer.hash(&mut h);
    job.deint.map(|d| (d.method as u8, d.tff, d.fields)).hash(&mut h);
    job.subs.hash(&mut h);
    job.stages.iter().map(|s| {
        s.hash(&mut h);
        h.finish()
    }).collect()
}

#[derive(Default)]
pub(crate) struct Worker {
    key: u64,
    /// Per stage: what its output depends on (see `stage_keys`).
    stage_keys: Vec<u64>,
    /// What the source pictures depend on (see `source_key`).
    source_key: u64,
    gpu: Option<Gpu>,
    /// Outputs by (stage, tick); stage 0 is the source, stage i the output of stages[i-1].
    pics: HashMap<(usize, i64), Tensor>,
    order: VecDeque<(usize, i64)>,
    bytes: usize,
    upscaler: Option<Upscaler>,
    rife: Option<Rife>,
    matting: HashMap<usize, MattingState>,
    /// Motion between pictures of a stage's input: (stage, from tick, to tick).
    /// (and the method: `Motion` as u8).
    motion: HashMap<(usize, i64, i64, u8), Tensor>,
    motion_order: VecDeque<(usize, i64, i64, u8)>,
    /// Export output: scaling and packing into the encoder's format.
    packer: Option<brp_ai::pack::Packer>,
    /// SEA-RAFT, loaded when a stage first asks for it.
    searaft: Option<brp_ai::flow::SeaRaft>,
    /// WAFT, the same.
    waft: Option<brp_ai::flow::Waft>,
    /// Deblur and clean-up networks, loaded when first asked for.
    deblur: Option<brp_ai::restore::Restorer>,
    /// BiRefNet matting, loaded when first asked for, and the alphas it found per
    /// (stage input, tick), kept for the neighbours' steadying.
    birefnet: Option<brp_ai::birefnet::BiRefNet>,
    alphas: HashMap<(usize, i64), Tensor>,
    alphas_order: VecDeque<(usize, i64)>,
    added_alphas: Vec<(usize, i64)>,
    cleanup: Option<brp_ai::restore::Restorer>,
    /// Depth: per stage input, the model (its frame history) and the last picture fed.
    depth: HashMap<usize, DepthState>,
    depth_fx: Option<brp_ai::depth::DepthFx>,
    /// Stabilisation: the warp, and the camera's movement from each picture of a
    /// stage's input to the next ((stage, from tick, motion as u8); None across a cut).
    stab: Option<brp_ai::stabilise::Stabiliser>,
    sims: HashMap<(usize, i64, u8), Option<brp_ai::stabilise::Sim>>,
    sims_order: VecDeque<(usize, i64, u8)>,
    /// Shader stages: (stage, its own key) -> runner.
    fx: HashMap<usize, (u64, OfflineEffects)>,
    /// Source frames to RGB.
    merge: Option<(Transfer, OfflineEffects)>,
    /// This job: the shader runners started, and the pictures and motions added
    /// (removed again if the job fails before its work ran).
    begun: Vec<usize>,
    added: Vec<(usize, i64)>,
    added_motion: Vec<(usize, i64, i64, u8)>,
    /// Host-visible buffer finished pictures are copied into for the RAM cache.
    readback: Option<Buffer>,
}

/// Stage outputs kept (GPU memory; tensors are planar f32).
const PIC_BUDGET: usize = 1536 << 20;
/// BiRefNet alphas kept (1 channel each).
const ALPHAS: usize = 32;
/// A neighbour's alpha is kept within the current alpha's range over the 5x5 around
/// the pixel (`MATTE_RANGE`), widened by `MATTE_MARGIN`. Measured on a 61-frame
/// 1080p clip (tools/airef/matte_stability.py): a margin of 0.1 pulled faint ghosts
/// (up to 10% alpha) into the background beside fast hands; none with 0.
const MATTE_MARGIN: f32 = 0.0;
const MATTE_RANGE: usize = 2;
/// Motions kept.
const MOTIONS: usize = 32;

fn worker(shared: Arc<Shared>) {
    let mut w = Worker::default();
    loop {
        let job = {
            let mut q = shared.queue.lock();
            loop {
                if shared.quit.load(std::sync::atomic::Ordering::Relaxed) {
                    // Models are dropped here (with `w`), while the GPU driver is loaded.
                    return;
                }
                if let Some(j) = q.pop_front() {
                    // Marked busy before the queue lock is released: the window never
                    // sees "idle" between taking a job and starting it.
                    *shared.current.lock() = Some((j.key, j.pos));
                    break j;
                }
                shared.wake.wait(&mut q);
            }
        };
        let t = std::time::Instant::now();
        // A panic in the work must not kill this thread: "current" would stay set, the
        // window would wait (spinner) forever and no picture would come again. It
        // becomes the job's error (shown on the effect), and the state starts afresh.
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| w.process(&job)));
        let mut ram = None;
        let outcome = match result {
            Ok(Ok((p, r))) => {
                ram = r.map(Arc::new);
                Outcome::Done(p)
            }
            Ok(Err(e)) if e.is::<MissingFrame>() => {
                log::debug!("AI: {e}");
                Outcome::Missing
            }
            Ok(Err(e)) => Outcome::Failed(format!("{e:#}")),
            Err(panic) => {
                let msg = panic.downcast_ref::<String>().cloned().or_else(|| panic.downcast_ref::<&str>().map(|s| s.to_string())).unwrap_or_default();
                log::error!("AI: internal error at frame {}: {msg}", job.pos.0);
                w = Worker::default();
                Outcome::Failed(format!("internal error: {msg}"))
            }
        };
        let took = t.elapsed();
        log::debug!("AI: frame {} sub {} in {:?}", job.pos.0, job.pos.1, took);
        shared.done.lock().push(Done { key: job.key, pos: job.pos, outcome, took, decoded: job.frames.len(), ram, restored: false });
        *shared.current.lock() = None;
    }
}

/// Puts pictures kept in RAM back on the GPU (a staging buffer and one copy each,
/// a few milliseconds at 1080p), beside the worker.
fn restorer(shared: Arc<Shared>) {
    let mut compute: Option<Arc<Compute>> = None;
    let mut staging: Option<Buffer> = None;
    loop {
        let r = {
            let mut q = shared.restores.lock();
            loop {
                if shared.quit.load(std::sync::atomic::Ordering::Relaxed) {
                    return;
                }
                if let Some(r) = q.pop_front() {
                    *shared.restoring.lock() = Some((r.key, r.pos));
                    break r;
                }
                shared.restore_wake.wait(&mut q);
            }
        };
        let t = std::time::Instant::now();
        if compute.as_ref().is_none_or(|c| !Arc::ptr_eq(&c.ctx, &r.ctx)) {
            staging = None;
            compute = Some(Compute::new(r.ctx.clone()));
        }
        let c = compute.as_ref().unwrap();
        let outcome = match upload_picture(c, &mut staging, &r.pic) {
            Ok(p) => Outcome::Done(p),
            Err(e) => Outcome::Failed(format!("{e:#}")),
        };
        let took = t.elapsed();
        log::trace!("AI: frame {} sub {} from RAM in {:?}", r.pos.0, r.pos.1, took);
        shared.done.lock().push(Done { key: r.key, pos: r.pos, outcome, took, decoded: 0, ram: None, restored: true });
        *shared.restoring.lock() = None;
    }
}

/// A picture made from its pixels in RAM (through `staging`, grown as needed; waits).
fn upload_picture(c: &Compute, staging: &mut Option<Buffer>, pic: &RamPicture) -> anyhow::Result<Arc<Picture>> {
    let bytes = pic.data.len() as u64;
    if staging.as_ref().is_none_or(|b| b.size < bytes) {
        *staging = None;
        *staging = Some(c.staging(bytes, "ai restore")?);
    }
    let buf = staging.as_mut().unwrap();
    buf.mapped().ok_or_else(|| anyhow::anyhow!("staging not mapped"))?[..pic.data.len()].copy_from_slice(&pic.data);
    let mut rec = Recorder::new(c)?;
    let out = Picture::new(c, &mut rec, pic.w, pic.h, pic.alpha)?;
    anyhow::ensure!(out.byte_size() == bytes, "picture size");
    rec.copy_buffer_to_picture(buf, &out);
    rec.submit_wait()?;
    Ok(out)
}

/// `input` through a shader runner (started for this job if not yet), as a tensor of
/// `channels`.
/// `spaces`: the colour space of `input` and the one the output must be in.
#[allow(clippy::too_many_arguments)]
fn run_fx(fx: &mut OfflineEffects, begun: &mut Vec<usize>, id: usize, gpu: &Gpu, rec: &mut Recorder, job: &Job, input: FxInput<'_>, seq: i64, target: (u32, u32), channels: usize, spaces: (Space, Space)) -> anyhow::Result<Tensor> {
    if !begun.contains(&id) {
        let frames: Vec<&VideoFrame> = job.frames.iter().map(|f| &**f).collect();
        fx.begin(&frames)?;
        begun.push(id);
    }
    let out = fx.record(rec.command_buffer(), input, job.color, seq, target, spaces.0, Some(spaces.1))?;
    rec.barrier();
    gpu.tm.from_image(rec, out.view, out.w as usize, out.h as usize, channels)
}

fn channels_of(f: &VideoFrame) -> usize {
    if f.layout.has_alpha() { 4 } else { 3 }
}

impl Worker {
    fn gpu(&self) -> &Gpu {
        self.gpu.as_ref().expect("GPU set up")
    }

    /// Done with a tensor in this recording: its buffer is reused by later work (or
    /// kept until the work ran, when it is still held elsewhere).
    fn put(&self, rec: &mut Recorder, t: Tensor) {
        self.gpu().ops.recycle(rec, t);
    }

    /// Ticks between pictures at the output of stage `s` (0 = source).
    fn grid(job: &Job, s: usize) -> i64 {
        let f: i64 = job.stages[..s].iter().map(Stage::interp_factor).product();
        (job.subs / (job.fields() * f)).max(1)
    }

    fn remember(&mut self, rec: &mut Recorder, k: (usize, i64), t: Tensor) {
        self.bytes += t.len() * 4;
        self.pics.insert(k, t);
        self.order.push_back(k);
        self.added.push(k);
        while self.bytes > PIC_BUDGET && self.order.len() > 4 {
            if let Some(old) = self.order.pop_front() {
                if let Some(t) = self.pics.remove(&old) {
                    self.bytes -= t.len() * 4;
                    self.gpu().ops.recycle(rec, t);
                }
            }
        }
    }

    /// The picture for the window, and its pixels read back for the RAM cache (copied
    /// in the same submission; None if that could not be set up).
    fn process(&mut self, job: &Job) -> anyhow::Result<(Arc<Picture>, Option<RamPicture>)> {
        let (p, out, read) = self.run(job, |w, rec| {
            let (pic, out) = w.record_job(rec, job)?;
            let bytes = pic.byte_size();
            if w.readback.as_ref().is_none_or(|b| b.size < bytes) {
                w.readback = None;
                match w.gpu().compute.readback(bytes, "ai ram copy") {
                    Ok(b) => w.readback = Some(b),
                    Err(e) => log::debug!("AI: no read-back for the RAM cache: {e:#}"),
                }
            }
            let read = match &w.readback {
                Some(b) => {
                    rec.copy_picture_to_buffer(&pic, b);
                    true
                }
                None => false,
            };
            Ok((pic, out, read))
        })?;
        let ram = if read {
            let n = p.byte_size() as usize;
            self.readback.as_mut().and_then(|b| b.mapped()).map(|m| RamPicture { w: p.w, h: p.h, alpha: p.alpha, data: m[..n].to_vec() })
        } else {
            None
        };
        if let Some(dir) = std::env::var_os("BRP_AI_DUMP") {
            // Debug: the picture as a file (read back from the GPU).
            let path = std::path::Path::new(&dir).join(format!("ai_{}_{}.ppm", job.pos.0, job.pos.1));
            if let Err(e) = out.to_image(&self.gpu().compute).and_then(|img| dump_ppm(&path, &img)) {
                log::warn!("AI: dump {}: {e:#}", path.display());
            }
        }
        Ok((p, ram))
    }

    /// Export: the picture `sample` asks for, scaled, converted and packed as `spec`
    /// says (given the picture's width, height and channels), read back for the
    /// encoder. The same stages and caches as playback, so it matches the viewer.
    /// With two samples (interlaced output) they are the two fields' moments, in time
    /// order, woven into one frame.
    /// `ocio`: OpenColorIO's output transform, applied to each picture before it is
    /// packed (DESIGN.md §7d).
    pub(crate) fn export(&mut self, job: &Job, samples: &[Sample], ocio: Option<&Arc<brp_render::OcioCompute>>, spec: &mut dyn FnMut((usize, usize, usize)) -> anyhow::Result<brp_ai::pack::OutputSpec>) -> anyhow::Result<Vec<u8>> {
        let mut total = 0;
        self.run(job, |w, rec| {
            rec.label = format!("export {}:{}", job.pos.0, job.pos.1);
            let mut ts = Vec::with_capacity(samples.len());
            for s in samples {
                let t = w.sample(rec, job, *s)?;
                let t = match ocio {
                    Some(o) => {
                        let gpu = w.gpu();
                        let pic = gpu.tm.to_picture(rec, &t)?;
                        let out = Picture::new(&gpu.compute, rec, t.w as u32, t.h as u32, t.c > 3)?;
                        rec.keep(pic.clone());
                        rec.keep(out.clone());
                        o.record(rec, pic.view, out.view, t.w as u32, t.h as u32)?;
                        let back = gpu.tm.from_image(rec, out.view, t.w, t.h, t.c)?;
                        w.put(rec, t);
                        back
                    }
                    None => t,
                };
                ts.push(t);
            }
            let s = spec((ts[0].w, ts[0].h, ts[0].c))?;
            if w.packer.is_none() {
                w.packer = Some(brp_ai::pack::Packer::new(&w.gpu().ops)?);
            }
            let refs: Vec<&Tensor> = ts.iter().collect();
            w.packer.as_mut().unwrap().record(rec, &refs, &s)?;
            total = s.layout.total;
            for t in ts {
                w.put(rec, t);
            }
            Ok(())
        })?;
        self.packer.as_mut().unwrap().output(total)
    }

    /// The picture `sample` asks for as the window gets it (an RGBA16F picture),
    /// read back to the CPU: the reference export output is checked against.
    pub(crate) fn reference(&mut self, job: &Job, sample: Sample) -> anyhow::Result<brp_ai::Image> {
        let t = self.run(job, |w, rec| {
            let t = w.sample(rec, job, sample)?;
            let gpu = w.gpu();
            let pic = gpu.tm.to_picture(rec, &t)?;
            rec.keep(pic.clone());
            let back = gpu.tm.from_image(rec, pic.view, t.w, t.h, t.c)?;
            w.put(rec, t);
            Ok(back)
        })?;
        t.to_image(&self.gpu().compute)
    }

    /// Export: make the picture at `tick` without output (history for stateful
    /// effects before the first exported picture).
    pub(crate) fn warm(&mut self, job: &Job, tick: i64) -> anyhow::Result<()> {
        self.run(job, |w, rec| {
            let t = w.eval(rec, job, job.stages.len(), tick)?;
            w.put(rec, t);
            Ok(())
        })
    }

    /// The finished picture `sample` asks for.
    fn sample(&mut self, rec: &mut Recorder, job: &Job, sample: Sample) -> anyhow::Result<Tensor> {
        let last = job.stages.len();
        match sample {
            Sample::Tick(t) => self.eval(rec, job, last, t),
            Sample::Blend(tau) => {
                let a = tau.floor() as i64;
                let f = (tau - a as f64) as f32;
                if f < 1e-4 {
                    return self.eval(rec, job, last, a);
                }
                let (pa, pb) = (self.eval(rec, job, last, a)?, self.eval(rec, job, last, a + 1)?);
                let out = self.gpu().ops.mix(rec, &pa, &pb, f)?;
                self.put(rec, pa);
                self.put(rec, pb);
                Ok(out)
            }
            Sample::Motion { tau, model, motion } => self.picture_at(rec, job, last, tau, model, motion),
        }
    }

    /// Set up for `job`, record `body`, submit and wait; on failure, forget what the
    /// recording was to make.
    fn run<T>(&mut self, job: &Job, body: impl FnOnce(&mut Self, &mut Recorder) -> anyhow::Result<T>) -> anyhow::Result<T> {
        if self.gpu.as_ref().is_none_or(|g| !Arc::ptr_eq(&g.ctx, &job.ctx)) {
            *self = Worker::default();
            let compute = Compute::new(job.ctx.clone());
            let ops = Ops::new(&compute)?;
            let tm = Temporal::new(&ops)?;
            let dn = brp_ai::denoise::Denoiser::new(&ops)?;
            self.gpu = Some(Gpu { ctx: job.ctx.clone(), compute, ops, tm, dn });
        }
        if job.key != self.key {
            // A setting changed. The output of stage i depends only on the stages up to
            // it: keep the source pictures (stage 0) and the outputs (with their motion
            // and matting) of the stages before the first one that changed, so changing
            // the last effect does not redo the ones before it.
            let keys = stage_keys(job);
            let same = self.stage_keys.iter().zip(&keys).take_while(|(a, b)| a == b).count();
            // Nothing is kept when the source pictures change (deinterlacing, or the tick
            // grid: the interlace detection settling at the start doubled and then
            // restored the pictures per frame, and frame n showed frame n / 2).
            let src = source_key(job);
            let src_same = src == self.source_key;
            self.source_key = src;
            if !src_same {
                // The shader runners keep uploaded frames by frame number: start them
                // again for other pictures.
                self.fx.clear();
                self.merge = None;
            }
            let keep = |i: usize| src_same && i <= same;
            self.pics.retain(|k, _| keep(k.0));
            self.order.retain(|k| keep(k.0));
            self.bytes = self.pics.values().map(|t| t.len() * 4).sum();
            self.matting.retain(|i, _| keep(*i));
            self.alphas.retain(|k, _| keep(k.0));
            self.alphas_order.retain(|k| keep(k.0));
            self.depth.retain(|i, _| keep(*i));
            self.sims.retain(|k, _| keep(k.0));
            self.sims_order.retain(|k| keep(k.0));
            self.motion.retain(|k, _| keep(k.0));
            self.motion_order.retain(|k| keep(k.0));
            self.stage_keys = keys;
            self.key = job.key;
        }
        let compute = self.gpu().compute.clone();
        let mut rec = Recorder::new(&compute)?;
        rec.label = format!("ai {}:{}", job.pos.0, job.pos.1);
        self.begun.clear();
        self.added.clear();
        self.added_motion.clear();
        self.added_alphas.clear();
        let t0 = std::time::Instant::now();
        let made = match body(self, &mut rec) {
            Ok(v) => {
                let recorded = t0.elapsed();
                rec.submit_wait().map(|t| {
                    log::trace!("AI: recorded in {recorded:?}, GPU {t:?}, total {:?}", t0.elapsed());
                    v
                })
            }
            Err(e) => {
                drop(rec);
                Err(e)
            }
        };
        // Work that did not run uploaded nothing: the runners forget the frames it was
        // to upload (else the next job reads never-written images: garbage pictures,
        // often after a jump, when jobs stop at a frame no longer decoded).
        let failed = made.is_err();
        for id in std::mem::take(&mut self.begun) {
            if id == MERGE {
                if let Some((_, fx)) = &mut self.merge {
                    if failed {
                        fx.abort();
                    }
                    fx.end();
                }
            } else if let Some((_, fx)) = self.fx.get_mut(&id) {
                if failed {
                    fx.abort();
                }
                fx.end();
            }
        }
        if made.is_err() {
            // Nothing recorded ran: forget what it was to make.
            for k in std::mem::take(&mut self.added) {
                if let Some(t) = self.pics.remove(&k) {
                    self.bytes -= t.len() * 4;
                }
            }
            self.order.retain(|k| self.pics.contains_key(k));
            for k in std::mem::take(&mut self.added_motion) {
                self.motion.remove(&k);
            }
            self.motion_order.retain(|k| self.motion.contains_key(k));
            self.matting.clear();
            for k in std::mem::take(&mut self.added_alphas) {
                self.alphas.remove(&k);
            }
            self.alphas_order.retain(|k| self.alphas.contains_key(k));
            // The depth history was to be written by work that did not run.
            for d in self.depth.values_mut() {
                d.net.reset();
                d.last = None;
            }
        }
        made
    }

    fn record_job(&mut self, rec: &mut Recorder, job: &Job) -> anyhow::Result<(Arc<Picture>, Tensor)> {
        let (n, sub) = job.pos;
        let tick = n * job.subs + sub as i64;
        let out = self.eval(rec, job, job.stages.len(), tick)?;
        let pic = self.gpu().tm.to_picture(rec, &out)?;
        Ok((pic, out))
    }

    /// The picture after `s` stages at `tick`.
    fn eval(&mut self, rec: &mut Recorder, job: &Job, s: usize, tick: i64) -> anyhow::Result<Tensor> {
        let tick = tick.clamp(0, job.last_frame * job.subs);
        if let Some(p) = self.pics.get(&(s, tick)) {
            return Ok(p.clone());
        }
        let t = if s == 0 { self.source(rec, job, tick)? } else { self.run_stage(rec, job, s, tick)? };
        // A model's picture, in the space the next stage wants (shader stages and the
        // source convert inside their chain).
        let t = match s.checked_sub(1).map(|i| &job.stages[i]) {
            Some(Stage::Ai { kind, .. }) => self.gpu().tm.convert_space(rec, t, kind.space(), job.spaces[s], &job.color)?,
            _ => t,
        };
        self.remember(rec, (s, tick), t.clone());
        Ok(t)
    }

    /// Source frame `n` in RGB.
    /// The source picture at `tick` (a frame, or a field with field rate), in RGB,
    /// deinterlaced first when the source is interlaced.
    fn source(&mut self, rec: &mut Recorder, job: &Job, tick: i64) -> anyhow::Result<Tensor> {
        let (f, prev, next, deint) = job.source_at(tick)?;
        if self.merge.as_ref().is_none_or(|(t, _)| *t != job.transfer) {
            self.merge = None;
            let mut fx = OfflineEffects::new(job.ctx.clone())?;
            let errors = fx.set_effects(&[], job.transfer);
            anyhow::ensure!(errors.is_empty(), "{}", errors.join("\n"));
            self.merge = Some((job.transfer, fx));
        }
        let gpu = self.gpu.as_ref().unwrap();
        let fx = &mut self.merge.as_mut().unwrap().1;
        let input = FxInput::Frame { frame: &f, prev: prev.as_deref(), next: next.as_deref(), deint };
        run_fx(fx, &mut self.begun, MERGE, gpu, rec, job, input, tick / Self::grid(job, 0), (f.width(), f.height()), channels_of(&f), (Space::Source, job.spaces[0]))
    }

    /// The picture at fractional `tau` ticks from the output of stage `s`, in-between
    /// pictures made by frame interpolation. With `motion`, the two pictures are moved
    /// along that motion instead (`Temporal::interpolate`), frame interpolation's
    /// picture kept only where the motion proves wrong: sharper, and exact where the
    /// motion is (particles test 34.5 dB against RIFE's 33.2 with SEA-RAFT or WAFT).
    fn picture_at(&mut self, rec: &mut Recorder, job: &Job, s: usize, tau: f64, model: RifeModel, motion: Option<Motion>) -> anyhow::Result<Tensor> {
        let g = Self::grid(job, s);
        let tau = tau.clamp(0.0, (job.last_frame * job.subs) as f64);
        let a = (tau / g as f64).floor() as i64 * g;
        let t = ((tau - a as f64) / g as f64) as f32;
        if t < 1e-3 || a + g > job.last_frame * job.subs {
            return self.eval(rec, job, s, a);
        }
        if t > 0.999 {
            return self.eval(rec, job, s, a + g);
        }
        let (pa, pb) = (self.eval(rec, job, s, a)?, self.eval(rec, job, s, a + g)?);
        if self.rife.as_ref().is_none_or(|r| r.model() != model) {
            self.rife = None;
            self.rife = Some(Rife::new(&self.gpu().ops, &brp_ai::models_dir(), model)?);
        }
        let mid = self.rife.as_ref().unwrap().record(rec, &pa, &pb, t)?;
        // Across a scene cut, the nearer real picture instead of a blend of two shots.
        let shot = self.same_shot(rec, job, s, a, a + g, motion.unwrap_or(Motion::Builtin))?;
        let out = match motion {
            Some(m) => {
                let (fab, fba) = (self.motion(rec, job, s, a, a + g, m)?, self.motion(rec, job, s, a + g, a, m)?);
                let out = self.gpu().tm.interpolate(rec, &pa, &pb, &fab, &fba, t, &shot, Some(&mid))?;
                self.put(rec, fab);
                self.put(rec, fba);
                out
            }
            None => self.gpu().tm.shot_pick(rec, &mid, &pa, &pb, t, &shot)?,
        };
        for x in [mid, shot, pa, pb] {
            self.put(rec, x);
        }
        Ok(out)
    }

    fn run_stage(&mut self, rec: &mut Recorder, job: &Job, s: usize, tick: i64) -> anyhow::Result<Tensor> {
        let input = s - 1;
        let g = Self::grid(job, input);
        let stage = job.stages[input].clone();
        match &stage {
            Stage::Shaders(specs) => self.run_shaders(rec, job, input, specs, tick, g),
            Stage::Ai { kind, params } => match kind {
                AiKind::Upscale => {
                    let img = self.eval(rec, job, input, tick)?;
                    let (w, h) = (img.w, img.h);
                    let model = UPSCALE_MODELS[(params[0] as usize).min(UPSCALE_MODELS.len() - 1)];
                    let denoise = params.get(4).copied().unwrap_or(0.5).clamp(0.0, 1.0) as f32;
                    if self.upscaler.as_ref().is_none_or(|u| u.model() != model || (model == UpscaleModel::GeneralV3 && (u.denoise() - denoise).abs() > 1e-3)) {
                        self.upscaler = None;
                        self.upscaler = Some(Upscaler::with_denoise(&self.gpu().ops, &brp_ai::models_dir(), model, denoise)?);
                    }
                    let temporal = params.get(2).is_none_or(|&v| v >= 0.5);
                    let src = if temporal {
                        let strength = params.get(3).copied().unwrap_or(1.0).clamp(0.0, 1.0) as f32;
                        let smooth = params.get(6).copied().unwrap_or(1.0).clamp(0.0, 2.0) as f32;
                        let fused = self.window(rec, job, input, tick, g, &img, strength, window_radius(params), smooth, Motion::from_param(params.get(7)))?;
                        self.put(rec, img);
                        fused
                    } else {
                        img
                    };
                    let up = self.upscaler.as_ref().unwrap().record(rec, &src)?;
                    self.put(rec, src);
                    if params[1] >= 0.5 {
                        Ok(up)
                    } else {
                        // Back to the size before, with the recovered detail.
                        let small = self.gpu().ops.resize(rec, &up, w, h)?;
                        self.put(rec, up);
                        Ok(small)
                    }
                }
                AiKind::Interpolate => {
                    let motion = (params.get(2).map(|v| v.round() as i64) != Some(3)).then(|| Motion::from_param(params.get(2)));
                    self.picture_at(rec, job, input, tick as f64, rife_model(params[1]), motion)
                }
                AiKind::MotionBlur => {
                    let best = params[0] >= 0.5;
                    let shutter = (params[1] / 360.0).clamp(0.0, 1.0);
                    let samples = (params.get(2).copied().unwrap_or(32.0) as usize).clamp(4, 64);
                    let cur = self.eval(rec, job, input, tick)?;
                    if shutter <= 0.0 {
                        return Ok(cur);
                    }
                    if best {
                        // The in-between moments over the shutter interval centred on the
                        // frame, moved along the chosen motion (frame interpolation where
                        // it fails), and averaged.
                        self.put(rec, cur);
                        let m = Motion::from_param(params.get(3));
                        let k = (samples / 4).clamp(2, 16);
                        let mut acc: Option<Tensor> = None;
                        for j in 0..k {
                            let tau = tick as f64 + (-shutter * 0.5 + shutter * (j as f64 + 0.5) / k as f64) * g as f64;
                            let p = self.picture_at(rec, job, input, tau, RifeModel::V4_26, Some(m))?;
                            acc = Some(match acc.take() {
                                None => p,
                                Some(a) => {
                                    let m = self.gpu().ops.mix(rec, &a, &p, 1.0 / (j + 1) as f32)?;
                                    self.put(rec, a);
                                    self.put(rec, p);
                                    m
                                }
                            });
                        }
                        return Ok(acc.unwrap());
                    }
                    // Fast: motion to the previous and next pictures (cached, shared with
                    // the temporal window), then blur along it.
                    let end = job.last_frame * job.subs;
                    let m = Motion::from_param(params.get(3));
                    let back = if tick - g >= 0 { Some(self.motion(rec, job, input, tick, tick - g, m)?) } else { None };
                    let fwd = if tick + g <= end { Some(self.motion(rec, job, input, tick, tick + g, m)?) } else { None };
                    // A neighbour across a scene cut is left out (its motion means nothing).
                    let bshot = if back.is_some() { Some(self.same_shot(rec, job, input, tick, tick - g, m)?) } else { None };
                    let fshot = if fwd.is_some() { Some(self.same_shot(rec, job, input, tick, tick + g, m)?) } else { None };
                    let out = self.gpu().tm.motion_blur(rec, &cur, back.as_ref(), fwd.as_ref(), [bshot.as_ref(), fshot.as_ref()], shutter as f32, samples)?;
                    for t in [back, fwd, bshot, fshot].into_iter().flatten() {
                        self.put(rec, t);
                    }
                    self.put(rec, cur);
                    Ok(out)
                }
                AiKind::Matting if params.get(1).copied().unwrap_or(0.0) < 0.5 => self.run_birefnet(rec, job, input, params[0] as usize, matte_radius(params), tick, g),
                AiKind::Matting => self.run_matting(rec, job, input, params[0] as usize, tick, g),
                AiKind::Deblur | AiKind::CleanUp => {
                    use brp_ai::restore::{RestoreModel, Restorer};
                    let img = self.eval(rec, job, input, tick)?;
                    let model = if *kind == AiKind::Deblur { RestoreModel::Deblur } else { RestoreModel::CleanUp };
                    let models = brp_ai::models_dir();
                    if !Restorer::available(&models, model) {
                        anyhow::bail!("{model:?} model not found in {}", models.display());
                    }
                    let slot = if model == RestoreModel::Deblur { &mut self.deblur } else { &mut self.cleanup };
                    if slot.is_none() {
                        *slot = Some(Restorer::new(&self.gpu.as_ref().unwrap().ops, &models, model)?);
                    }
                    let r = if model == RestoreModel::Deblur { self.deblur.as_ref() } else { self.cleanup.as_ref() };
                    let out = r.unwrap().record(rec, &img, params.first().copied().unwrap_or(1.0) as f32);
                    self.put(rec, img);
                    out
                }
                AiKind::Depth => self.run_depth(rec, job, input, params, tick, g),
                AiKind::Stabilise => self.run_stabilise(rec, job, input, params, tick, g),
                AiKind::Steady => {
                    let img = self.eval(rec, job, input, tick)?;
                    let sides = self.neighbours(rec, job, input, tick, g, steady_radius(params), Motion::from_param(params.get(4)));
                    let out = sides.and_then(|sides| {
                        let refs: [Vec<(&Tensor, &Tensor)>; 2] = sides.each_ref().map(|v| v.iter().map(|(p, m)| (p, m)).collect());
                        let v = |i: usize, d: f64| params.get(i).copied().unwrap_or(d) as f32;
                        // Keep sharpness 0..1: a gain of at most 1..1.5 (more is never needed).
                        let keep = 0.5 * v(2, 1.0).clamp(0.0, 1.0);
                        let out = self.gpu().tm.steady_detail(rec, &img, [&refs[0], &refs[1]], v(1, 4.0).clamp(0.5, 8.0), keep, v(3, 1.0), brp_ai::gpu_temporal::DECAY);
                        drop(refs);
                        self.put_neighbours(rec, sides);
                        Ok(out?)
                    });
                    self.put(rec, img);
                    out
                }
                AiKind::Denoise => {
                    let img = self.eval(rec, job, input, tick)?;
                    let out = self.denoise(rec, job, input, tick, g, &img, params);
                    self.put(rec, img);
                    out
                }
            },
        }
    }

    /// Motion from picture `from` to picture `to` of stage `s`'s output (where each
    /// pixel of `from` is in `to`), cached: playing on, a frame's window shares most
    /// of its motion with the previous frame's.
    /// Whether the pictures at `from` and `to` (output of stage `s`) are of one shot:
    /// see `Temporal::same_shot` (a 1-value tensor on the GPU).
    fn same_shot(&mut self, rec: &mut Recorder, job: &Job, s: usize, from: i64, to: i64, m: Motion) -> anyhow::Result<Tensor> {
        let flow = self.motion(rec, job, s, from, to, m)?;
        let (a, b) = (self.eval(rec, job, s, from)?, self.eval(rec, job, s, to)?);
        let shot = self.gpu().tm.same_shot(rec, &a, &b, &flow)?;
        for x in [flow, a, b] {
            self.put(rec, x);
        }
        Ok(shot)
    }

    fn motion(&mut self, rec: &mut Recorder, job: &Job, s: usize, from: i64, to: i64, m: Motion) -> anyhow::Result<Tensor> {
        let k = (s, from, to, m as u8);
        if let Some(f) = self.motion.get(&k) {
            return Ok(f.clone());
        }
        let (a, b) = (self.eval(rec, job, s, from)?, self.eval(rec, job, s, to)?);
        let f = self.flow_between(rec, &a, &b, m)?;
        self.put(rec, a);
        self.put(rec, b);
        self.motion.insert(k, f.clone());
        self.motion_order.push_back(k);
        self.added_motion.push(k);
        while self.motion_order.len() > MOTIONS {
            if let Some(old) = self.motion_order.pop_front() {
                if let Some(t) = self.motion.remove(&old) {
                    self.gpu().ops.recycle(rec, t);
                }
            }
        }
        Ok(f)
    }

    /// The motion from picture `a` to `b` (where each pixel of `a` is in `b`), not cached.
    fn flow_between(&mut self, rec: &mut Recorder, a: &Tensor, b: &Tensor, m: Motion) -> anyhow::Result<Tensor> {
        Ok(match m {
            Motion::Builtin => self.gpu().tm.motion(rec, b, a)?,
            Motion::SeaRaft => {
                if self.searaft.is_none() {
                    self.searaft = Some(brp_ai::flow::SeaRaft::new(&self.gpu().ops, &brp_ai::models_dir())?);
                }
                self.searaft.as_ref().unwrap().flow(rec, a, b)?
            }
            Motion::Waft => {
                if self.waft.is_none() {
                    self.waft = Some(brp_ai::flow::Waft::new(&self.gpu().ops, &brp_ai::models_dir())?);
                }
                self.waft.as_ref().unwrap().flow(rec, a, b)?
            }
        })
    }

    /// Depth of picture `tick` of stage `s` and an effect from it. The video model
    /// continues from the previous picture when this one follows it closely and is of
    /// the same shot; otherwise it starts again, after replaying a few earlier pictures
    /// (as matting) so the first ones have a history.
    #[allow(clippy::too_many_arguments)]
    fn run_depth(&mut self, rec: &mut Recorder, job: &Job, s: usize, params: &[f64], tick: i64, g: i64) -> anyhow::Result<Tensor> {
        use brp_ai::depth::{Depth, DepthFx, DepthModel, DepthOutput};
        let model = if params.first().copied().unwrap_or(0.0) >= 0.5 { DepthModel::Frame } else { DepthModel::Video };
        let models = brp_ai::models_dir();
        if !Depth::available(&models, model) {
            anyhow::bail!("depth model not found in {}", models.display());
        }
        if self.depth.get(&s).is_none_or(|d| d.net.model != model) {
            self.depth.remove(&s);
            let net = Depth::new(&self.gpu().ops, &models, model)?;
            self.depth.insert(s, DepthState { net, last: None });
        }
        if model == DepthModel::Video {
            let mut continues = self.depth[&s].last.is_some_and(|l| l < tick && tick - l <= 8 * g);
            if continues && tick >= g {
                // A cut since the last picture: the history is of another shot.
                let shot = self.same_shot(rec, job, s, tick - g, tick, Motion::Builtin)?;
                rec.sync()?;
                continues = shot.to_image(&self.gpu().compute)?.data[0] >= brp_ai::gpu_temporal::CUT;
                self.put(rec, shot);
            }
            if !continues {
                self.depth.get_mut(&s).unwrap().net.reset();
                for back in (1..=DEPTH_REPLAY).rev() {
                    let t = tick - back * g;
                    if t < 0 {
                        continue;
                    }
                    let Ok(prev) = self.eval(rec, job, s, t) else { continue };
                    let d = self.depth[&s].net.record(rec, &prev)?;
                    self.put(rec, d);
                    self.put(rec, prev);
                }
            }
        }
        let img = self.eval(rec, job, s, tick)?;
        let st = self.depth.get_mut(&s).unwrap();
        let d = st.net.record(rec, &img)?;
        st.last = Some(tick);
        if self.depth_fx.is_none() {
            self.depth_fx = Some(DepthFx::new(&self.gpu().ops)?);
        }
        let out = match params.get(1).map_or(1, |v| v.round() as i64) {
            0 => DepthOutput::Map,
            2 => DepthOutput::Fog,
            3 => DepthOutput::Key,
            _ => DepthOutput::LensBlur,
        };
        let focus = params.get(2).copied().unwrap_or(0.8) as f32;
        let amount = params.get(3).copied().unwrap_or(0.5) as f32;
        let o = self.depth_fx.as_ref().unwrap().apply(rec, &img, &d, out, focus, amount)?;
        self.put(rec, d);
        self.put(rec, img);
        Ok(o)
    }

    /// Picture `tick` of stage `s` as seen by a steady camera: the camera's movement
    /// from each picture to the next over the smoothing window (fitted once each, from
    /// a small grid of the motion read back, and kept), smoothed, and the picture
    /// warped onto it.
    #[allow(clippy::too_many_arguments)]
    fn run_stabilise(&mut self, rec: &mut Recorder, job: &Job, s: usize, params: &[f64], tick: i64, g: i64) -> anyhow::Result<Tensor> {
        use brp_ai::stabilise::{fit, steady, Stabiliser};
        let radius = stab_radius(params);
        let zoom = 1.0 + params.get(1).copied().unwrap_or(5.0).clamp(0.0, 25.0) as f32 / 100.0;
        let m = Motion::from_param(params.get(2));
        let end = job.last_frame * job.subs;
        if self.stab.is_none() {
            self.stab = Some(Stabiliser::new(&self.gpu().ops)?);
        }
        let mut pending = Vec::new();
        for d in -radius..radius {
            let from = tick + d * g;
            if from < 0 || from + g > end || self.sims.contains_key(&(s, from, m as u8)) {
                continue;
            }
            let (a, b) = (self.eval(rec, job, s, from)?, self.eval(rec, job, s, from + g)?);
            let flow = self.flow_between(rec, &a, &b, m)?;
            let grid = self.stab.as_ref().unwrap().grid(rec, &flow)?;
            let shot = self.gpu().tm.same_shot(rec, &a, &b, &flow)?;
            let unit = a.w as f64 / flow.w as f64;
            pending.push((from, grid, shot, unit, (a.w, a.h)));
            for t in [a, b, flow] {
                self.put(rec, t);
            }
        }
        if !pending.is_empty() {
            rec.sync()?;
            let compute = self.gpu().compute.clone();
            for (from, grid, shot, unit, (w, h)) in pending {
                let cut = shot.to_image(&compute)?.data[0] < brp_ai::gpu_temporal::CUT;
                let sim = if cut { None } else { fit(&grid.to_image(&compute)?, w, h, unit) };
                let k = (s, from, m as u8);
                self.sims.insert(k, sim);
                self.sims_order.push_back(k);
                self.put(rec, grid);
                self.put(rec, shot);
            }
            while self.sims_order.len() > SIMS {
                if let Some(old) = self.sims_order.pop_front() {
                    self.sims.remove(&old);
                }
            }
        }
        let steps: Vec<_> = (0..2 * radius).map(|i| self.sims.get(&(s, tick + (i - radius) * g, m as u8)).copied().flatten()).collect();
        let c = steady(&steps, radius as usize);
        let img = self.eval(rec, job, s, tick)?;
        let out = self.stab.as_ref().unwrap().warp(rec, &img, c, zoom);
        self.put(rec, img);
        out
    }

    /// The neighbours of picture `tick` of stage `s` (`radius` each way, `g` apart),
    /// each with the motion to it from the one a step nearer (the steps are cached
    /// and serve the next pictures too); a side stops at the clip's ends. Give them
    /// back with `put_neighbours`.
    #[allow(clippy::too_many_arguments)]
    fn neighbours(&mut self, rec: &mut Recorder, job: &Job, s: usize, tick: i64, g: i64, radius: i64, m: Motion) -> anyhow::Result<[Vec<(Tensor, Tensor)>; 2]> {
        let end = job.last_frame * job.subs;
        let mut sides: [Vec<(Tensor, Tensor)>; 2] = [Vec::new(), Vec::new()];
        for (i, side) in [-1i64, 1].into_iter().enumerate() {
            for d in 1..=radius {
                let (from, to) = (tick + (d - 1) * side * g, tick + d * side * g);
                if to < 0 || to > end {
                    break;
                }
                let step = self.motion(rec, job, s, from, to, m)?;
                let pic = self.eval(rec, job, s, to)?;
                sides[i].push((pic, step));
            }
        }
        Ok(sides)
    }

    fn put_neighbours(&mut self, rec: &mut Recorder, sides: [Vec<(Tensor, Tensor)>; 2]) {
        for (p, m) in sides.into_iter().flatten() {
            self.put(rec, p);
            self.put(rec, m);
        }
    }

    /// Picture `tick` of stage `s` merged with its neighbours, moved onto it along
    /// the motion, where they agree with it. Scene cuts are found on the GPU: a side
    /// stops before one.
    #[allow(clippy::too_many_arguments)]
    fn window(&mut self, rec: &mut Recorder, job: &Job, s: usize, tick: i64, g: i64, cur: &Tensor, strength: f32, radius: i64, smooth: f32, m: Motion) -> anyhow::Result<Tensor> {
        let sides = self.neighbours(rec, job, s, tick, g, radius, m)?;
        let refs: [Vec<(&Tensor, &Tensor)>; 2] = sides.each_ref().map(|v| v.iter().map(|(p, m)| (p, m)).collect());
        let out = self.gpu().tm.window(rec, cur, [&refs[0], &refs[1]], strength, brp_ai::gpu_temporal::DECAY, smooth);
        drop(refs);
        self.put_neighbours(rec, sides);
        Ok(out?)
    }

    /// Picture `tick` of stage `s` denoised (`brp_ai::denoise`).
    #[allow(clippy::too_many_arguments)]
    fn denoise(&mut self, rec: &mut Recorder, job: &Job, s: usize, tick: i64, g: i64, cur: &Tensor, params: &[f64]) -> anyhow::Result<Tensor> {
        let radius = params.first().map_or(3, |v| v.round() as i64).clamp(1, 5);
        let set = brp_ai::denoise::DenoiseSettings {
            temporal: params.get(1).copied().unwrap_or(1.0).clamp(0.0, 1.0) as f32,
            spatial: params.get(2).copied().unwrap_or(1.0).clamp(0.0, 3.0) as f32,
            chroma: params.get(3).copied().unwrap_or(1.0).clamp(0.0, 3.0) as f32,
            impulse: params.get(5).copied().unwrap_or(0.5).clamp(0.0, 1.0) as f32,
            ..Default::default()
        };
        let sides = self.neighbours(rec, job, s, tick, g, radius, Motion::from_param(params.get(4)))?;
        let refs: [Vec<(&Tensor, &Tensor)>; 2] = sides.each_ref().map(|v| v.iter().map(|(p, m)| (p, m)).collect());
        let gpu = self.gpu();
        let out = gpu.dn.record(rec, &gpu.tm, cur, [&refs[0], &refs[1]], &set);
        drop(refs);
        self.put_neighbours(rec, sides);
        Ok(out?)
    }

    /// BiRefNet's alpha of picture `tick` of stage input `s` (`img`), found once.
    fn birefnet_alpha(&mut self, rec: &mut Recorder, s: usize, tick: i64, img: &Tensor) -> anyhow::Result<Tensor> {
        if let Some(a) = self.alphas.get(&(s, tick)) {
            return Ok(a.clone());
        }
        if self.birefnet.is_none() {
            let models = brp_ai::models_dir();
            if !brp_ai::birefnet::BiRefNet::available(&models) {
                anyhow::bail!("BiRefNet model not found in {}", models.display());
            }
            self.birefnet = Some(brp_ai::birefnet::BiRefNet::new(&self.gpu().ops, &models)?);
        }
        let a = self.birefnet.as_ref().unwrap().alpha(rec, img)?;
        self.alphas.insert((s, tick), a.clone());
        self.alphas_order.push_back((s, tick));
        self.added_alphas.push((s, tick));
        while self.alphas_order.len() > ALPHAS {
            if let Some(k) = self.alphas_order.pop_front() {
                self.alphas.remove(&k);
            }
        }
        Ok(a)
    }

    /// Matting with BiRefNet: each picture's alpha on its own, then steadied with
    /// `radius` neighbours each way (0: not).
    #[allow(clippy::too_many_arguments)]
    fn run_birefnet(&mut self, rec: &mut Recorder, job: &Job, input: usize, output: usize, radius: i64, tick: i64, g: i64) -> anyhow::Result<Tensor> {
        let img = self.eval(rec, job, input, tick)?;
        let a = self.birefnet_alpha(rec, input, tick, &img)?;
        let alpha = if radius > 0 {
            let sides = self.neighbours(rec, job, input, tick, g, radius, Motion::from_param(None))?;
            let mut alphas: [Vec<Tensor>; 2] = [Vec::new(), Vec::new()];
            for (i, side) in [-1i64, 1].into_iter().enumerate() {
                for (d, (pic, _)) in sides[i].iter().enumerate() {
                    let t = tick + (d as i64 + 1) * side * g;
                    let na = self.birefnet_alpha(rec, input, t, pic)?;
                    alphas[i].push(na);
                }
            }
            let refs: [Vec<(&Tensor, &Tensor, &Tensor)>; 2] = std::array::from_fn(|i| sides[i].iter().zip(&alphas[i]).map(|((p, m), a)| (p, a, m)).collect());
            let out = self.gpu().tm.alpha_window(rec, &img, &a, [&refs[0], &refs[1]], brp_ai::gpu_temporal::DECAY, MATTE_MARGIN, MATTE_RANGE);
            drop(refs);
            self.put_neighbours(rec, sides);
            out?
        } else {
            a
        };
        let m = self.birefnet.as_ref().unwrap().with_alpha(rec, &img, &alpha)?;
        let out = self.gpu().tm.matte(rec, &m, &img, output.min(3))?;
        self.put(rec, m);
        self.put(rec, alpha);
        self.put(rec, img);
        Ok(out)
    }

    #[allow(clippy::too_many_arguments)]
    fn run_matting(&mut self, rec: &mut Recorder, job: &Job, input: usize, output: usize, tick: i64, g: i64) -> anyhow::Result<Tensor> {
        let img = self.eval(rec, job, input, tick)?;
        let models = brp_ai::models_dir();
        if !Matting::available(&models) {
            anyhow::bail!("background removal model not found in {}", models.display());
        }
        if self.matting.get(&input).is_none_or(|m| m.net.size() != (img.w, img.h)) {
            self.matting.remove(&input);
            let net = Matting::new(&self.gpu().ops, &models, img.w, img.h)?;
            self.matting.insert(input, MattingState { net, last: None });
        }
        // The network is recurrent: after a jump, replay a few earlier pictures so it
        // has settled on the subject. A small gap (playback skipping pictures) keeps
        // the state; it adapts in a frame.
        let continues = self.matting[&input].last.is_some_and(|l| l < tick && tick - l <= 8 * g);
        if !continues {
            self.matting.get_mut(&input).unwrap().net.reset();
            for back in (1..=4).rev() {
                let t = tick - back * g;
                if t < 0 {
                    continue;
                }
                let Ok(prev) = self.eval(rec, job, input, t) else { continue };
                let m = self.matting.get_mut(&input).unwrap().net.record(rec, &prev)?;
                self.put(rec, m);
                self.put(rec, prev);
            }
        }
        let st = self.matting.get_mut(&input).unwrap();
        // Foreground RGB + alpha.
        let m = st.net.record(rec, &img)?;
        st.last = Some(tick);
        let out = self.gpu().tm.matte(rec, &m, &img, output.min(3))?;
        self.put(rec, m);
        self.put(rec, img);
        Ok(out)
    }

    #[allow(clippy::too_many_arguments)]
    fn run_shaders(&mut self, rec: &mut Recorder, job: &Job, input: usize, specs: &[EffectSpec], tick: i64, g: i64) -> anyhow::Result<Tensor> {
        use std::hash::{Hash, Hasher};
        let mut h = std::collections::hash_map::DefaultHasher::new();
        Stage::Shaders(specs.to_vec()).hash(&mut h);
        job.transfer.hash(&mut h);
        let skey = h.finish();
        match self.fx.get_mut(&input) {
            Some((k, _)) if *k == skey => {}
            // New settings on the same runner: it recompiles only what needs it (a
            // shader with its parameters built in), not every pass.
            Some((k, fx)) => {
                let errors = fx.set_effects(specs, job.transfer);
                anyhow::ensure!(errors.is_empty(), "{}", errors.join("\n"));
                *k = skey;
            }
            None => {
                let mut fx = OfflineEffects::new(job.ctx.clone())?;
                let errors = fx.set_effects(specs, job.transfer);
                anyhow::ensure!(errors.is_empty(), "{}", errors.join("\n"));
                self.fx.insert(input, (skey, fx));
            }
        }
        let seq = tick / g;
        if input == 0 {
            // From the source: the decoded frame itself (the chain does the YUV
            // conversion), deinterlaced first when interlaced.
            let (f, prev, next, deint) = job.source_at(tick)?;
            let gpu = self.gpu.as_ref().unwrap();
            let fx = &mut self.fx.get_mut(&input).unwrap().1;
            let src = FxInput::Frame { frame: &f, prev: prev.as_deref(), next: next.as_deref(), deint };
            return run_fx(fx, &mut self.begun, input, gpu, rec, job, src, seq, (f.width(), f.height()), channels_of(&f), (Space::Source, job.spaces[input + 1]));
        }
        let img = self.eval(rec, job, input, tick)?;
        let gpu = self.gpu.as_ref().unwrap();
        let pic = gpu.tm.to_picture(rec, &img)?;
        rec.keep(pic.clone());
        let fx = &mut self.fx.get_mut(&input).unwrap().1;
        let out = run_fx(fx, &mut self.begun, input, gpu, rec, job, FxInput::Picture(&pic), seq, (pic.w, pic.h), img.c, (job.spaces[input], job.spaces[input + 1]));
        gpu.ops.recycle(rec, img);
        out
    }
}

/// A picture as an 8-bit PPM (colour only), for debugging.
fn dump_ppm(path: &std::path::Path, img: &brp_ai::Image) -> anyhow::Result<()> {
    let n = img.w * img.h;
    let mut out = format!("P6\n{} {}\n255\n", img.w, img.h).into_bytes();
    for i in 0..n {
        for c in 0..3 {
            out.push((img.data[c.min(img.channels - 1) * n + i].clamp(0.0, 1.0) * 255.0).round() as u8);
        }
    }
    std::fs::write(path, out)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use brp_media::{Matrix, Primaries, Transfer as Tf};
    use brp_render::Space;

    fn shaders(text: &str) -> Stage {
        let file = Arc::new(brp_render::usershader::parse::parse("t.glsl", text).unwrap());
        Stage::Shaders(vec![EffectSpec { id: 1, file, params: Vec::new(), enabled: true }])
    }

    fn ai(kind: AiKind) -> Stage {
        Stage::Ai { kind, params: kind.file().params.iter().map(|p| p.default).collect() }
    }

    #[test]
    fn stage_spaces_convert_lazily_and_end_in_source() {
        let pq = ColorInfo { matrix: Matrix::Bt2020Ncl, full_range: false, transfer: Tf::Pq, primaries: Primaries::Bt2020 };
        let linear = "// @brp space linear\n//!HOOK MAIN\n//!BIND HOOKED\nvec4 hook() { return HOOKED_tex(HOOKED_pos); }\n";
        // Two models side by side share one conversion; the last model's output is
        // in the source's encoding (where the window's live chain takes over).
        let s = stage_spaces(&[ai(AiKind::Denoise), ai(AiKind::Upscale)], &pq);
        assert_eq!(s, vec![Space::Display, Space::Display, Space::Source]);
        // Shaders in linear before a model: their chain hands the model display.
        let s = stage_spaces(&[shaders(linear), ai(AiKind::Denoise), shaders(linear)], &pq);
        assert_eq!(s, vec![Space::Source, Space::Display, Space::Source, Space::Source]);
        // SDR Rec.709: display is the source itself (no conversion anywhere).
        let sdr = ColorInfo { matrix: Matrix::Bt709, full_range: false, transfer: Tf::Sdr, primaries: Primaries::Bt709 };
        let sc = brp_render::workspace::SourceColor::of(&sdr);
        assert!(stage_spaces(&[ai(AiKind::Denoise)], &sdr).iter().all(|&x| sc.same(x, Space::Source)));
    }
}
