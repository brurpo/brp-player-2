//! Export and batch export.
//!
//! An export runs the whole effect stack through the same worker the player uses
//! (`ai::Worker`: shader effects, AI stages, deinterlacing, the same caches), so
//! every exported picture is the picture the viewer shows. Frames are decoded by
//! their own frame-exact `Source`, pictures are made in order, and the GPU scales
//! them to the output size and writes them straight into the encoder's pixel
//! format (`brp_ai::pack`). Only those bytes go to the CPU, to an encoder thread
//! (`brp_media::encode::Writer`).
//!
//! Time: the stack makes `subs` pictures per source frame (fields times slow-motion
//! factors) at `stack rate` = frame rate x fields. The output either takes every
//! picture at its own rate (retime: a different rate changes the speed, audio
//! follows with or without pitch), or keeps the timing and picks, blends or makes
//! (motion interpolation) the picture at each output frame's time (conform).

use crate::ai::{self, AiKind, Job, Motion, Sample, SourceDeint, Stage, Worker};
use crate::player::DeintMode;
use anyhow::Context as _;
use brp_ai::pack::{ColorConv, OutputSpec};
use brp_media::encode::{self, AudioConfig, AudioSource, PackLayout, RateControl, VideoConfig, Writer};
use brp_media::{ColorInfo, Matrix, Primaries, Request, Source, Transfer, VideoFrame};
use brp_render::GpuContext;
use parking_lot::Mutex;
use serde_json::{json, Value};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

// ---- formats ------------------------------------------------------------------------

/// A container (or image sequence).
pub struct Container {
    pub id: &'static str,
    pub label: &'static str,
    pub muxer: &'static str,
    pub ext: &'static str,
    /// Default audio encoder ("" none).
    pub audio: &'static str,
}

pub const CONTAINERS: &[Container] = &[
    Container { id: "mp4", label: "MP4", muxer: "mp4", ext: "mp4", audio: "aac" },
    Container { id: "mov", label: "QuickTime (MOV)", muxer: "mov", ext: "mov", audio: "pcm_s24le" },
    Container { id: "mkv", label: "Matroska (MKV)", muxer: "matroska", ext: "mkv", audio: "flac" },
    Container { id: "mxf", label: "MXF", muxer: "mxf", ext: "mxf", audio: "pcm_s24le" },
    Container { id: "webm", label: "WebM", muxer: "webm", ext: "webm", audio: "libopus" },
    Container { id: "avi", label: "AVI", muxer: "avi", ext: "avi", audio: "pcm_s16le" },
    Container { id: "gif", label: "Animated GIF", muxer: "gif", ext: "gif", audio: "" },
    Container { id: "png", label: "PNG sequence", muxer: "image2", ext: "png", audio: "" },
    Container { id: "jpg", label: "JPEG sequence", muxer: "image2", ext: "jpg", audio: "" },
    Container { id: "tif", label: "TIFF sequence", muxer: "image2", ext: "tif", audio: "" },
    Container { id: "exr", label: "OpenEXR sequence", muxer: "image2", ext: "exr", audio: "" },
    Container { id: "dpx", label: "DPX sequence", muxer: "image2", ext: "dpx", audio: "" },
];

pub fn container(id: &str) -> &'static Container {
    CONTAINERS.iter().find(|c| c.id == id).unwrap_or(&CONTAINERS[0])
}

/// A codec as offered in the panel: its encoders (software first) and defaults.
pub struct Family {
    pub id: &'static str,
    pub label: &'static str,
    pub software: &'static [&'static str],
    pub hardware: &'static [&'static str],
    /// Pixel format (opaque), and with alpha (None: no alpha).
    pub pix: &'static str,
    pub pix_alpha: Option<&'static str>,
    /// (value, label) of the `profile` option; the first is the default.
    pub profiles: &'static [(&'static str, &'static str)],
    /// Constant quality: (min, max, default); lower is better.
    pub quality: Option<(f32, f32, f32)>,
    /// Offer an average bit rate (kbit/s default).
    pub bitrate: Option<u32>,
    pub containers: &'static [&'static str],
    /// Encoder options always set.
    pub options: &'static [(&'static str, &'static str)],
    /// Width and height must be multiples of this (CineForm: the width only).
    pub align: u32,
}

pub const FAMILIES: &[Family] = &[
    Family {
        id: "h264",
        label: "H.264 / AVC",
        software: &["libx264", "libopenh264"],
        hardware: &["h264_nvenc", "h264_amf", "h264_qsv", "h264_vaapi", "h264_videotoolbox", "h264_mf"],
        pix: "yuv420p",
        pix_alpha: None,
        profiles: &[],
        quality: Some((0.0, 51.0, 18.0)),
        bitrate: Some(20000),
        containers: &["mp4", "mov", "mkv", "avi"],
        options: &[],
        align: 2,
    },
    Family {
        id: "hevc",
        label: "H.265 / HEVC (10-bit)",
        software: &["libx265", "libkvazaar"],
        hardware: &["hevc_nvenc", "hevc_amf", "hevc_qsv", "hevc_vaapi", "hevc_videotoolbox", "hevc_mf"],
        pix: "yuv420p10le",
        pix_alpha: None,
        profiles: &[],
        quality: Some((0.0, 51.0, 20.0)),
        bitrate: Some(15000),
        containers: &["mp4", "mov", "mkv"],
        options: &[],
        align: 2,
    },
    Family {
        id: "av1",
        label: "AV1 (10-bit)",
        software: &["libsvtav1", "libaom-av1", "librav1e"],
        hardware: &["av1_nvenc", "av1_amf", "av1_qsv", "av1_vaapi"],
        pix: "yuv420p10le",
        pix_alpha: None,
        profiles: &[],
        quality: Some((0.0, 63.0, 28.0)),
        bitrate: Some(10000),
        containers: &["mp4", "mkv", "webm"],
        options: &[],
        align: 2,
    },
    Family {
        id: "vp9",
        label: "VP9",
        software: &["libvpx-vp9"],
        hardware: &["vp9_qsv", "vp9_vaapi"],
        pix: "yuv420p",
        pix_alpha: Some("yuva420p"),
        profiles: &[],
        quality: Some((0.0, 63.0, 30.0)),
        bitrate: Some(10000),
        containers: &["webm", "mkv", "mp4"],
        options: &[("row-mt", "1")],
        align: 2,
    },
    Family {
        id: "prores",
        label: "Apple ProRes",
        software: &["prores_ks", "prores_aw", "prores"],
        hardware: &["prores_gpu"],
        pix: "yuv422p10le",
        pix_alpha: Some("yuva444p10le"),
        profiles: &[("hq", "422 HQ"), ("standard", "422"), ("lt", "422 LT"), ("proxy", "422 Proxy"), ("4444", "4444"), ("4444xq", "4444 XQ")],
        quality: None,
        bitrate: None,
        containers: &["mov", "mkv"],
        options: &[("vendor", "apl0")],
        align: 2,
    },
    Family {
        id: "dnxhr",
        label: "Avid DNxHR / DNxHD",
        software: &["dnxhd"],
        hardware: &["dnxhd_gpu"],
        pix: "yuv422p",
        pix_alpha: None,
        profiles: &[("dnxhr_hq", "HQ (8-bit)"), ("dnxhr_hqx", "HQX (10-bit)"), ("dnxhr_sq", "SQ (8-bit)"), ("dnxhr_lb", "LB (8-bit)"), ("dnxhr_444", "444 (10-bit)"), ("dnxhd_hqx", "DNxHD 10-bit (1080 or 720)"), ("dnxhd_hq", "DNxHD HQ (1080 or 720)"), ("dnxhd_sq", "DNxHD SQ (1080 or 720)"), ("dnxhd_lb", "DNxHD 36 (1080p)")],
        quality: None,
        bitrate: None,
        containers: &["mov", "mxf", "mkv"],
        options: &[],
        align: 2,
    },
    Family {
        id: "xdcam",
        label: "XDCAM HD422 (MPEG-2, 50 Mbit/s)",
        software: &["mpeg2video"],
        hardware: &[],
        pix: "yuv422p",
        pix_alpha: None,
        profiles: &[],
        quality: None,
        bitrate: None,
        containers: &["mxf", "mov"],
        options: &[("bf", "2"), ("intra_vlc", "1"), ("non_linear_quant", "1"), ("qmin", "1"), ("qmax", "28"), ("dc", "10"), ("profile", "0"), ("level", "2"), ("bufsize", "17825792")],
        align: 2,
    },
    Family {
        id: "ffv1",
        label: "FFV1 (lossless)",
        software: &["ffv1"],
        hardware: &[],
        pix: "yuv444p10le",
        pix_alpha: Some("yuva444p10le"),
        profiles: &[],
        quality: None,
        bitrate: None,
        containers: &["mkv", "avi"],
        options: &[("level", "3"), ("slicecrc", "1"), ("g", "1")],
        align: 1,
    },
    Family {
        id: "utvideo",
        label: "Ut Video (lossless)",
        software: &["utvideo"],
        hardware: &[],
        pix: "yuv422p",
        pix_alpha: Some("gbrap"),
        profiles: &[],
        quality: None,
        bitrate: None,
        containers: &["avi", "mov", "mkv"],
        options: &[],
        align: 2,
    },
    Family {
        id: "cfhd",
        label: "GoPro CineForm",
        software: &["cfhd"],
        hardware: &[],
        pix: "yuv422p10le",
        pix_alpha: Some("gbrap12le"),
        profiles: &[],
        quality: None,
        bitrate: None,
        containers: &["mov", "mkv", "avi"],
        options: &[],
        align: 16,
    },
    Family {
        id: "qtrle",
        label: "QuickTime Animation",
        software: &["qtrle"],
        hardware: &[],
        pix: "rgb24",
        pix_alpha: Some("argb"),
        profiles: &[],
        quality: None,
        bitrate: None,
        containers: &["mov"],
        options: &[],
        align: 1,
    },
    Family {
        id: "hap",
        label: "HAP",
        software: &["hap"],
        hardware: &[],
        pix: "rgba",
        pix_alpha: Some("rgba"),
        profiles: &[],
        quality: None,
        bitrate: None,
        containers: &["mov"],
        options: &[],
        align: 4,
    },
    Family {
        id: "dxv",
        label: "Resolume DXV3",
        software: &["dxv"],
        hardware: &["dxv_gpu"],
        pix: "rgba",
        pix_alpha: Some("rgba"),
        profiles: &[],
        quality: None,
        bitrate: None,
        containers: &["mov"],
        options: &[],
        align: 4,
    },
    Family {
        id: "mjpeg",
        label: "Motion JPEG",
        software: &["mjpeg"],
        hardware: &["mjpeg_qsv", "mjpeg_vaapi"],
        pix: "yuvj422p",
        pix_alpha: None,
        profiles: &[],
        quality: Some((1.0, 31.0, 2.0)),
        bitrate: None,
        containers: &["mov", "avi", "mkv"],
        options: &[],
        align: 2,
    },
    Family { id: "gif", label: "GIF", software: &["gif"], hardware: &[], pix: "pal8", pix_alpha: None, profiles: &[], quality: None, bitrate: None, containers: &["gif"], options: &[], align: 1 },
    Family { id: "png", label: "PNG", software: &["png"], hardware: &[], pix: "rgb24", pix_alpha: Some("rgba"), profiles: &[], quality: None, bitrate: None, containers: &["png"], options: &[], align: 1 },
    Family { id: "jpeg", label: "JPEG", software: &["mjpeg"], hardware: &[], pix: "yuvj444p", pix_alpha: None, profiles: &[], quality: Some((1.0, 31.0, 2.0)), bitrate: None, containers: &["jpg"], options: &[], align: 2 },
    Family { id: "tiff", label: "TIFF", software: &["tiff"], hardware: &[], pix: "rgb24", pix_alpha: Some("rgba"), profiles: &[], quality: None, bitrate: None, containers: &["tif"], options: &[("compression_algo", "deflate")], align: 1 },
    Family { id: "exr", label: "OpenEXR (half float, linear)", software: &["exr"], hardware: &[], pix: "gbrpf32le", pix_alpha: Some("gbrapf32le"), profiles: &[], quality: None, bitrate: None, containers: &["exr"], options: &[("format", "half"), ("compression", "zip16")], align: 1 },
    Family { id: "dpx", label: "DPX (10-bit)", software: &["dpx"], hardware: &[], pix: "gbrp10le", pix_alpha: Some("rgba64le"), profiles: &[], quality: None, bitrate: None, containers: &["dpx"], options: &[], align: 1 },
];

/// Any encoder of the bundled FFmpeg ("Other" in the codec list).
pub const OTHER: &str = "other";

pub fn family(id: &str) -> Option<&'static Family> {
    FAMILIES.iter().find(|f| f.id == id)
}

// ---- our GPU encoders ---------------------------------------------------------------------

/// Our own encoders on the GPU (`brp_ai::gpuenc`), for codecs no GPU encodes in
/// hardware. They are listed with the families' hardware encoders (so "GPU
/// acceleration" offers them) and stand in for the software encoder named next to
/// them, which takes over when the GPU path fails.
pub const GPU_ENCODERS: &[(&str, &str)] = &[("prores_gpu", "prores_ks"), ("dnxhd_gpu", "dnxhd"), ("dxv_gpu", "dxv")];

/// Ours that work in this build.
const GPU_READY: &[&str] = &["prores_gpu", "dnxhd_gpu", "dxv_gpu"];

pub fn is_ours(name: &str) -> bool {
    GPU_ENCODERS.iter().any(|(n, _)| *n == name)
}

fn ours_ready(name: &str) -> bool {
    GPU_READY.contains(&name) && std::env::var_os("BRP_NO_GPU_ENCODERS").is_none()
}

/// The FFmpeg encoder an encoder name stands for (ours: the software one).
pub fn ffmpeg_encoder_of(name: &str) -> &str {
    GPU_ENCODERS.iter().find(|(n, _)| *n == name).map_or(name, |(_, s)| s)
}

/// An error of our GPU encoder: the export is made again with the CPU encoder.
#[derive(Debug)]
struct GpuEncodeFailed;

impl std::fmt::Display for GpuEncodeFailed {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("GPU encoder failed")
    }
}

fn encoder_exists(name: &str) -> bool {
    if is_ours(name) {
        return ours_ready(name);
    }
    static CACHE: std::sync::OnceLock<Mutex<std::collections::HashMap<String, bool>>> = std::sync::OnceLock::new();
    let m = CACHE.get_or_init(Default::default);
    *m.lock().entry(name.to_string()).or_insert_with(|| encode::encoder(name).is_some())
}

/// DXV with alpha (DXT5) needs FFmpeg built with CVExplorer's patch: its `dxv`
/// encoder then has a `format` option with "dxt5".
pub fn dxv_alpha_available() -> bool {
    static V: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *V.get_or_init(|| encode::encoder_option_values("dxv", "format").iter().any(|v| v == "dxt5"))
}

// ---- hardware encoders ----------------------------------------------------------------

/// Hardware encoders that open on this machine (probed once, in the background:
/// opening NVENC or AMF takes a moment).
static HW: std::sync::OnceLock<Mutex<Option<Vec<String>>>> = std::sync::OnceLock::new();

pub fn probe_hardware() {
    let cell = HW.get_or_init(|| Mutex::new(None));
    if cell.lock().is_some() {
        return;
    }
    *cell.lock() = Some(Vec::new());
    std::thread::Builder::new()
        .name("brp-hw-probe".into())
        .spawn(|| {
            let mut ok = Vec::new();
            // Encoders that do not open here explain why at length: not shown.
            brp_media::encode::quiet_log(true);
            for f in FAMILIES {
                for e in f.hardware {
                    let Some(info) = encode::encoder(e) else { continue };
                    let pix = ["nv12", "yuv420p", "p010le"].into_iter().find(|p| info.pix_fmts.iter().any(|x| x == p)).or(info.pix_fmts.first().map(String::as_str)).unwrap_or("nv12").to_string();
                    match encode::probe_encoder(e, &pix) {
                        Ok(()) => ok.push(e.to_string()),
                        Err(err) => log::debug!("export: {e} unavailable: {err:#}"),
                    }
                }
            }
            brp_media::encode::quiet_log(false);
            log::info!("export: hardware encoders: {ok:?}");
            *HW.get().unwrap().lock() = Some(ok);
            HW_DONE.store(true, Ordering::Relaxed);
        })
        .ok();
}

static HW_DONE: AtomicBool = AtomicBool::new(false);

/// The hardware probe has finished.
pub fn hardware_probed() -> bool {
    HW_DONE.load(Ordering::Relaxed)
}

/// Hardware encoders known to work (FFmpeg's: empty while probing), and ours.
pub fn hardware_ok() -> Vec<String> {
    let mut v = HW.get().and_then(|m| m.lock().clone()).unwrap_or_default();
    v.extend(GPU_ENCODERS.iter().filter(|(n, _)| ours_ready(n)).map(|(n, _)| n.to_string()));
    v
}

/// The encoder to use: an explicit name, "gpu" (this GPU's vendor first, then any
/// that works, else software) or "" (software).
pub fn pick_encoder(f: &Family, choice: &str, gpu_name: &str) -> Option<String> {
    let sw = f.software.iter().find(|e| encoder_exists(e)).map(|s| s.to_string());
    match choice {
        "" => sw.or_else(|| f.hardware.iter().find(|e| encoder_exists(e)).map(|s| s.to_string())),
        "gpu" => {
            let ok = hardware_ok();
            let g = gpu_name.to_ascii_lowercase();
            let vendor = if g.contains("nvidia") || g.contains("geforce") || g.contains("rtx") {
                "_nvenc"
            } else if g.contains("amd") || g.contains("radeon") {
                "_amf"
            } else if g.contains("intel") {
                "_qsv"
            } else {
                "_none"
            };
            f.hardware.iter().find(|e| e.ends_with(vendor) && ok.iter().any(|o| o == *e)).or_else(|| f.hardware.iter().find(|e| ok.iter().any(|o| o == *e))).map(|s| s.to_string()).or(sw)
        }
        name => Some(name.to_string()),
    }
}

// ---- settings ---------------------------------------------------------------------------

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Fit {
    /// Whole picture inside, bars around it.
    Fit,
    /// Fills the frame, cropped.
    Fill,
    Stretch,
}

#[derive(Clone, Debug, PartialEq)]
pub enum Size {
    /// The size the effects make (the source's, or the upscaled one).
    Source,
    /// Width x height; 0 for one of them: from the picture's aspect.
    Fixed(u32, u32),
    /// Percent of the effects' size.
    Scale(f32),
}

/// How a frame rate different from the stack's is reached.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Conform {
    /// Every picture becomes a frame: the speed changes.
    Retime,
    /// Same timing: the picture on screen at each frame's time (repeats or drops).
    Nearest,
    /// Same timing: the two nearest pictures mixed.
    Blend,
    /// Same timing: new pictures made along the motion (as Slow Motion makes them).
    Motion,
}

impl Conform {
    pub const ALL: [Conform; 4] = [Conform::Nearest, Conform::Blend, Conform::Motion, Conform::Retime];
    fn id(self) -> &'static str {
        match self {
            Conform::Retime => "retime",
            Conform::Nearest => "nearest",
            Conform::Blend => "blend",
            Conform::Motion => "motion",
        }
    }
    pub fn from_id(s: &str) -> Option<Conform> {
        Self::ALL.into_iter().find(|c| c.id() == s)
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct ExportSettings {
    pub name: String,
    pub container: String,
    /// Codec family id, or `OTHER` (then `encoder` names it).
    pub codec: String,
    /// "" software, "gpu" the GPU's encoder, or an encoder name.
    pub encoder: String,
    pub profile: String,
    /// "" automatic.
    pub pix_fmt: String,
    /// Constant quality (None: average bit rate when the codec offers one).
    pub quality: Option<f32>,
    pub kbps: u32,
    pub alpha: bool,
    /// Audio encoder ("" none) and bit rate.
    pub audio: String,
    pub audio_kbps: u32,
    /// When the speed changes: keep the pitch (else it shifts with the speed).
    pub keep_pitch: bool,
    pub size: Size,
    pub fit: Fit,
    /// Output frame rate (None: the stack's).
    pub fps: Option<(i32, i32)>,
    pub conform: Conform,
    /// HDR sources: tone map to SDR BT.709 (as the viewer shows it).
    pub tonemap: bool,
    /// Interlaced output at `fps` frames (2 x fps fields) per second.
    pub interlaced: bool,
    /// Interlaced: top field first (else bottom field first).
    pub tff: bool,
    /// Extra encoder options, "key=value" separated by spaces.
    pub extra: String,
}

impl Default for ExportSettings {
    fn default() -> Self {
        ExportSettings {
            name: String::new(),
            container: "mp4".into(),
            codec: "h264".into(),
            encoder: String::new(),
            profile: String::new(),
            pix_fmt: String::new(),
            quality: Some(18.0),
            kbps: 20000,
            alpha: false,
            audio: "aac".into(),
            audio_kbps: 320,
            keep_pitch: true,
            size: Size::Source,
            fit: Fit::Fit,
            fps: None,
            conform: Conform::Motion,
            tonemap: false,
            interlaced: false,
            tff: true,
            extra: String::new(),
        }
    }
}

impl ExportSettings {
    pub fn family(&self) -> Option<&'static Family> {
        family(&self.codec)
    }

    pub fn to_json(&self) -> Value {
        json!({
            "name": self.name,
            "container": self.container,
            "codec": self.codec,
            "encoder": self.encoder,
            "profile": self.profile,
            "pix_fmt": self.pix_fmt,
            "quality": self.quality,
            "kbps": self.kbps,
            "alpha": self.alpha,
            "audio": self.audio,
            "audio_kbps": self.audio_kbps,
            "keep_pitch": self.keep_pitch,
            "size": match &self.size {
                Size::Source => json!("source"),
                Size::Fixed(w, h) => json!([w, h]),
                Size::Scale(p) => json!({ "scale": p }),
            },
            "fit": match self.fit { Fit::Fit => "fit", Fit::Fill => "fill", Fit::Stretch => "stretch" },
            "fps": self.fps.map(|(n, d)| format!("{n}/{d}")),
            "conform": self.conform.id(),
            "interlaced": self.interlaced,
            "tff": self.tff,
            "tonemap": self.tonemap,
            "extra": self.extra,
        })
    }

    pub fn from_json(v: &Value) -> ExportSettings {
        let d = ExportSettings::default();
        let s = |k: &str, def: &str| v.get(k).and_then(Value::as_str).unwrap_or(def).to_string();
        let b = |k: &str, def: bool| v.get(k).and_then(Value::as_bool).unwrap_or(def);
        let u = |k: &str, def: u32| v.get(k).and_then(Value::as_u64).map_or(def, |x| x as u32);
        let size = match v.get("size") {
            Some(Value::Array(a)) if a.len() == 2 => Size::Fixed(a[0].as_u64().unwrap_or(0) as u32, a[1].as_u64().unwrap_or(0) as u32),
            Some(Value::Object(o)) => Size::Scale(o.get("scale").and_then(Value::as_f64).unwrap_or(100.0) as f32),
            _ => Size::Source,
        };
        ExportSettings {
            name: s("name", ""),
            container: s("container", &d.container),
            codec: s("codec", &d.codec),
            encoder: s("encoder", ""),
            profile: s("profile", ""),
            pix_fmt: s("pix_fmt", ""),
            quality: match v.get("quality") {
                Some(Value::Null) => None,
                Some(q) => q.as_f64().map(|q| q as f32),
                None => d.quality,
            },
            kbps: u("kbps", d.kbps),
            alpha: b("alpha", false),
            audio: s("audio", &d.audio),
            audio_kbps: u("audio_kbps", d.audio_kbps),
            keep_pitch: b("keep_pitch", true),
            size,
            fit: match s("fit", "fit").as_str() {
                "fill" => Fit::Fill,
                "stretch" => Fit::Stretch,
                _ => Fit::Fit,
            },
            fps: v.get("fps").and_then(Value::as_str).and_then(parse_rate),
            conform: Conform::from_id(&s("conform", "motion")).unwrap_or(Conform::Motion),
            interlaced: b("interlaced", false),
            tff: b("tff", true),
            tonemap: b("tonemap", false),
            extra: s("extra", ""),
        }
    }

    /// The codec's own default after the container or codec changed.
    pub fn apply_family_defaults(&mut self) {
        let Some(f) = self.family() else { return };
        self.profile = f.profiles.first().map(|p| p.0.to_string()).unwrap_or_default();
        self.quality = f.quality.map(|q| q.2);
        if let Some(k) = f.bitrate {
            self.kbps = k;
        }
        self.pix_fmt.clear();
        if !self.encoder.is_empty() && self.encoder != "gpu" && !f.software.contains(&self.encoder.as_str()) && !f.hardware.contains(&self.encoder.as_str()) {
            self.encoder.clear();
        }
        self.audio = container(&self.container).audio.to_string();
        if f.pix_alpha.is_none() {
            self.alpha = false;
        }
    }

    /// The codec can keep HDR (10 bits or more).
    /// The codec can store interlaced video (field-aware coding, or lossless).
    pub fn interlace_possible(&self) -> bool {
        matches!(self.codec.as_str(), "h264" | "xdcam" | "prores" | "ffv1" | OTHER) || (self.codec == "dnxhr" && self.profile.starts_with("dnxhd_"))
    }

    pub fn hdr_possible(&self) -> bool {
        matches!(self.codec.as_str(), "hevc" | "av1" | "vp9" | "prores" | "dnxhr" | "ffv1" | "cfhd" | "dpx" | OTHER)
    }

    /// A hardware encoder for this codec works here.
    pub fn gpu_possible(&self) -> bool {
        let ok = hardware_ok();
        self.family().is_some_and(|f| f.hardware.iter().any(|h| ok.iter().any(|o| o == h)))
    }

    /// The GPU encoder for this codec is one of ours (they code interlaced video).
    pub fn gpu_is_ours(&self) -> bool {
        self.family().is_some_and(|f| f.hardware.iter().any(|h| is_ours(h) && ours_ready(h)))
    }

    /// Named quality levels (label key, value) for codecs with a quality scale.
    pub fn quality_levels(&self) -> Vec<(&'static str, f32)> {
        match self.family().and_then(|f| f.quality) {
            Some((_, max, _)) if max >= 60.0 => vec![("q_draft", 40.0), ("q_good", 32.0), ("q_high", 26.0), ("q_best", 18.0)],
            Some((_, max, _)) if max >= 50.0 => vec![("q_draft", 28.0), ("q_good", 23.0), ("q_high", 18.0), ("q_best", 12.0)],
            Some(_) => vec![("q_draft", 8.0), ("q_good", 5.0), ("q_high", 3.0), ("q_best", 2.0)],
            None => Vec::new(),
        }
    }

    /// The output can hold alpha with this codec.
    pub fn alpha_possible(&self) -> bool {
        match self.codec.as_str() {
            "dxv" => dxv_alpha_available() || ours_ready("dxv_gpu"),
            "prores" => true,
            OTHER => encode::encoder(&self.encoder).is_some_and(|e| e.pix_fmts.iter().any(|p| encode::pix_fmt_info(p).is_some_and(|i| i.alpha))),
            _ => self.family().is_some_and(|f| f.pix_alpha.is_some()),
        }
    }
}

/// "29.97p", "59.94i", "50i", "25" -> (frame rate, interlaced). An "i" rate counts
/// fields: 59.94i is 29.97 frames of two fields.
pub fn parse_rate_spec(s: &str) -> Option<((i32, i32), bool)> {
    let s = s.trim();
    if let Some(f) = s.strip_suffix(['i', 'I']) {
        let (n, d) = parse_rate(f)?;
        let (n, d) = if n % 2 == 0 { (n / 2, d) } else { (n, d * 2) };
        return Some(((n, d), true));
    }
    Some((parse_rate(s.strip_suffix(['p', 'P']).unwrap_or(s))?, false))
}

/// A frame rate for the panel: "29.97p", or "59.94i" (fields) when interlaced.
pub fn rate_spec_label(r: (i32, i32), interlaced: bool) -> String {
    if interlaced { format!("{}i", rate_label((r.0 * 2, r.1))) } else { format!("{}p", rate_label(r)) }
}

/// "25", "29.97", "30000/1001" -> (num, den). NTSC rates become exact fractions.
pub fn parse_rate(s: &str) -> Option<(i32, i32)> {
    let s = s.trim();
    if let Some((n, d)) = s.split_once('/') {
        let (n, d) = (n.trim().parse().ok()?, d.trim().parse().ok()?);
        return (n > 0 && d > 0).then_some((n, d));
    }
    let f: f64 = s.parse().ok()?;
    (f > 0.0 && f < 1000.0).then(|| rate_of(f))
}

/// The rate a file means: 2997/100 is 29.97, written 30000/1001 (muxers such as MXF
/// take only the standard fractions); other rates stay as they are.
pub fn standard_rate(q: (i32, i32)) -> (i32, i32) {
    let f = rate_f(q);
    let s = rate_of(f);
    if s.1 == 1001 || s.1 == 1 { s } else { q }
}

/// A frame rate as an exact fraction: 23.976 -> 24000/1001, 25 -> 25/1.
pub fn rate_of(f: f64) -> (i32, i32) {
    let ntsc = (f * 1.001).round();
    if (f - ntsc / 1.001).abs() < 0.002 && (f - f.round()).abs() > 0.001 {
        return ((ntsc * 1000.0) as i32, 1001);
    }
    if (f - f.round()).abs() < 0.0005 {
        return (f.round() as i32, 1);
    }
    ((f * 1000.0).round() as i32, 1000)
}

pub fn rate_f(r: (i32, i32)) -> f64 {
    r.0 as f64 / r.1.max(1) as f64
}

pub fn rate_label(r: (i32, i32)) -> String {
    let f = rate_f(r);
    if r.1 == 1 { format!("{}", r.0) } else { format!("{:.3}", f).trim_end_matches('0').trim_end_matches('.').to_string() }
}

fn gcd(a: i64, b: i64) -> i64 {
    if b == 0 { a.abs().max(1) } else { gcd(b, a % b) }
}

/// Ready-made settings.
pub fn builtin_presets() -> Vec<ExportSettings> {
    let mut v = Vec::new();
    let mut p = |name: &str, f: &dyn Fn(&mut ExportSettings)| {
        let mut s = ExportSettings { name: name.into(), ..Default::default() };
        f(&mut s);
        v.push(s);
    };
    p("H.264 · MP4 — delivery", &|_s| {});
    p("H.264 · MP4 — GPU encoder", &|s| {
        s.encoder = "gpu".into();
        s.quality = Some(20.0);
    });
    p("H.264 · MP4 — web (smaller)", &|s| {
        s.quality = Some(23.0);
        s.audio_kbps = 192;
    });
    p("HEVC 10-bit · MP4 (keeps HDR)", &|s| {
        s.codec = "hevc".into();
        s.quality = Some(20.0);
    });
    p("AV1 10-bit · WebM", &|s| {
        s.container = "webm".into();
        s.codec = "av1".into();
        s.quality = Some(28.0);
        s.audio = "libopus".into();
        s.audio_kbps = 192;
    });
    p("ProRes 422 HQ · MOV", &|s| {
        s.container = "mov".into();
        s.codec = "prores".into();
        s.profile = "hq".into();
        s.quality = None;
        s.audio = "pcm_s24le".into();
    });
    p("ProRes 4444 · MOV (alpha)", &|s| {
        s.container = "mov".into();
        s.codec = "prores".into();
        s.profile = "4444".into();
        s.quality = None;
        s.alpha = true;
        s.audio = "pcm_s24le".into();
    });
    p("DNxHR HQX 10-bit · MXF", &|s| {
        s.container = "mxf".into();
        s.codec = "dnxhr".into();
        s.profile = "dnxhr_hqx".into();
        s.quality = None;
        s.audio = "pcm_s24le".into();
    });
    p("DNxHR HQ · MOV", &|s| {
        s.container = "mov".into();
        s.codec = "dnxhr".into();
        s.profile = "dnxhr_hq".into();
        s.quality = None;
        s.audio = "pcm_s24le".into();
    });
    p("XDCAM HD422 · MXF", &|s| {
        s.container = "mxf".into();
        s.codec = "xdcam".into();
        s.quality = None;
        s.audio = "pcm_s24le".into();
    });
    p("FFV1 lossless · MKV (alpha)", &|s| {
        s.container = "mkv".into();
        s.codec = "ffv1".into();
        s.quality = None;
        s.alpha = true;
        s.audio = "flac".into();
    });
    p("VP9 · WebM (alpha)", &|s| {
        s.container = "webm".into();
        s.codec = "vp9".into();
        s.quality = Some(30.0);
        s.alpha = true;
        s.audio = "libopus".into();
        s.audio_kbps = 192;
    });
    p("QuickTime Animation · MOV (alpha)", &|s| {
        s.container = "mov".into();
        s.codec = "qtrle".into();
        s.quality = None;
        s.alpha = true;
        s.audio = "pcm_s16le".into();
    });
    p("HAP Alpha · MOV", &|s| {
        s.container = "mov".into();
        s.codec = "hap".into();
        s.quality = None;
        s.alpha = true;
        s.audio = "pcm_s16le".into();
    });
    p("DXV3 · MOV", &|s| {
        s.container = "mov".into();
        s.codec = "dxv".into();
        s.quality = None;
        s.audio = "pcm_s16le".into();
    });
    if dxv_alpha_available() || ours_ready("dxv_gpu") {
        p("DXV3 alpha · MOV", &|s| {
            s.container = "mov".into();
            s.codec = "dxv".into();
            s.quality = None;
            s.alpha = true;
            s.audio = "pcm_s16le".into();
        });
    }
    p("PNG sequence (alpha)", &|s| {
        s.container = "png".into();
        s.codec = "png".into();
        s.quality = None;
        s.alpha = true;
        s.audio.clear();
    });
    p("OpenEXR sequence (linear)", &|s| {
        s.container = "exr".into();
        s.codec = "exr".into();
        s.quality = None;
        s.alpha = true;
        s.audio.clear();
    });
    p("GIF — 480 px, 15 fps", &|s| {
        s.container = "gif".into();
        s.codec = "gif".into();
        s.quality = None;
        s.audio.clear();
        s.size = Size::Fixed(480, 0);
        s.fps = Some((15, 1));
        s.conform = Conform::Nearest;
    });
    v
}

// ---- the job ----------------------------------------------------------------------------

/// Deinterlacing of an export's source.
#[derive(Clone, Copy, Debug)]
pub enum DeintChoice {
    /// As decided for the file open in the player.
    Fixed(Option<SourceDeint>),
    /// Decide from the file (batch items): the player's mode and field-rate setting.
    Detect { mode: DeintMode, field_rate: bool },
}

#[derive(Clone)]
pub struct ExportJob {
    pub input: String,
    /// Output file (image sequences: the first file's name; numbers are added).
    pub output: String,
    pub settings: ExportSettings,
    /// The whole enabled effect stack.
    pub stack: Vec<Stage>,
    /// Source frames (None: the whole file).
    pub range: Option<(i64, i64)>,
    pub deint: DeintChoice,
    /// EXR: the layer to export (by name; None: the default one).
    pub layer: Option<String>,
    /// OpenColorIO on: the export goes through OCIO's output for its format (DESIGN.md §7d).
    pub ocio: Option<OcioJob>,
}

/// What an export needs of OCIO: the configs, the Color panel's choices, and the
/// settings their "(Default)"s come from.
#[derive(Clone)]
pub struct OcioJob {
    pub lib: Arc<brp_ocio::Library>,
    pub choices: crate::color::OcioChoices,
    pub settings: crate::settings::Settings,
}

/// Codecs that write still images (sRGB with OCIO's Auto output).
const IMAGE_CODECS: &[&str] = &["png", "jpeg", "tiff", "gif", "webp", "bmp", "tga", "qoi"];

#[derive(Clone, Debug, Default, PartialEq)]
pub enum State {
    #[default]
    Starting,
    Indexing,
    Running,
    Finishing,
    Done,
    Failed(String),
    Cancelled,
}

impl State {
    pub fn ended(&self) -> bool {
        matches!(self, State::Done | State::Failed(_) | State::Cancelled)
    }
}

#[derive(Clone, Debug, Default)]
pub struct Progress {
    pub state: State,
    pub done: i64,
    pub total: i64,
    /// Pictures per second (recent).
    pub fps: f64,
    pub elapsed: f64,
    pub size: (u32, u32),
    pub rate: (i32, i32),
    pub encoder: String,
    /// Files written (an image sequence: the pattern).
    pub output: String,
}

impl Progress {
    pub fn fraction(&self) -> f32 {
        if self.total > 0 { (self.done as f32 / self.total as f32).clamp(0.0, 1.0) } else { 0.0 }
    }

    pub fn eta(&self) -> Option<f64> {
        (self.fps > 0.0 && self.total > self.done).then(|| (self.total - self.done) as f64 / self.fps)
    }
}

pub struct Running {
    pub progress: Arc<Mutex<Progress>>,
    cancel: Arc<AtomicBool>,
    thread: Option<std::thread::JoinHandle<()>>,
}

impl Running {
    pub fn cancel(&self) {
        self.cancel.store(true, Ordering::Relaxed);
    }

    pub fn snapshot(&self) -> Progress {
        self.progress.lock().clone()
    }

    pub fn ended(&self) -> bool {
        self.progress.lock().state.ended()
    }
}

impl Drop for Running {
    fn drop(&mut self) {
        self.cancel();
        if let Some(t) = self.thread.take() {
            let _ = t.join();
        }
    }
}

/// Start an export on its own thread, on `ctx`'s GPU.
pub fn start(job: ExportJob, ctx: Arc<GpuContext>) -> Running {
    let progress = Arc::new(Mutex::new(Progress::default()));
    let cancel = Arc::new(AtomicBool::new(false));
    let (p, c) = (progress.clone(), cancel.clone());
    let thread = std::thread::Builder::new()
        .name("brp-export".into())
        .spawn(move || {
            let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| match run(&job, &ctx, &p, &c) {
                // Our GPU encoder failed: the same export with the CPU encoder.
                Err(e) if e.downcast_ref::<GpuEncodeFailed>().is_some() && !c.load(Ordering::Relaxed) => {
                    log::warn!("export {}: {e:#}; exporting again with the CPU encoder", job.input);
                    let mut j = job.clone();
                    j.settings.encoder = String::new();
                    *p.lock() = Progress::default();
                    run(&j, &ctx, &p, &c)
                }
                r => r,
            }));
            let state = match r {
                Ok(Ok(())) => State::Done,
                Ok(Err(_)) if c.load(Ordering::Relaxed) => State::Cancelled,
                Ok(Err(e)) => State::Failed(format!("{e:#}")),
                Err(panic) => State::Failed(panic.downcast_ref::<String>().cloned().or_else(|| panic.downcast_ref::<&str>().map(|s| s.to_string())).unwrap_or_else(|| "internal error".into())),
            };
            match &state {
                State::Failed(e) => log::error!("export {}: {e}", job.input),
                State::Cancelled => log::info!("export {}: cancelled", job.input),
                _ => log::info!("export {} -> {}: done", job.input, job.output),
            }
            p.lock().state = state;
        })
        .expect("export thread");
    Running { progress, cancel, thread: Some(thread) }
}

fn transfer_of(c: &ColorInfo) -> brp_render::Transfer {
    match c.transfer {
        Transfer::Pq => brp_render::Transfer::Pq,
        Transfer::Hlg => brp_render::Transfer::Hlg,
        Transfer::Linear => brp_render::Transfer::Linear,
        Transfer::Sdr => brp_render::Transfer::Sdr,
    }
}

fn kr_kb(m: Matrix) -> (f32, f32) {
    match m {
        Matrix::Bt601 => (0.299, 0.114),
        Matrix::Bt709 | Matrix::Rgb => (0.2126, 0.0722),
        Matrix::Bt2020Ncl => (0.2627, 0.0593),
        Matrix::Smpte240 => (0.212, 0.087),
    }
}

/// For an image sequence: `dir/name.png` -> `dir/name_%06d.png`.
pub fn sequence_pattern(output: &str) -> String {
    let p = Path::new(output);
    let stem = p.file_stem().map(|s| s.to_string_lossy().into_owned()).unwrap_or_else(|| "frame".into());
    let ext = p.extension().map(|s| s.to_string_lossy().into_owned()).unwrap_or_default();
    p.with_file_name(format!("{stem}_%06d.{ext}")).to_string_lossy().into_owned()
}

/// Decide deinterlacing as the player does for a file it opens: the method of the
/// mode; Auto deinterlaces when picture analysis (or, failing that, the decoder's
/// flags) finds the file interlaced.
fn detect_deint(src: &Source, mode: DeintMode, field_rate: bool, cancel: &AtomicBool) -> Option<SourceDeint> {
    use brp_render::DeintMethod;
    let method = match mode {
        DeintMode::Off => return None,
        DeintMode::Best => Some(DeintMethod::NnediAdaptive),
        DeintMode::Nnedi => Some(DeintMethod::Nnedi),
        DeintMode::Bwdif => Some(DeintMethod::Bwdif),
        DeintMode::Auto => None,
    };
    // Picture analysis runs on frames as they decode (triples around every fifth
    // frame): decode the start of the file until it decides.
    let last = src.frame_count() - 1;
    let upto = last.min(125);
    src.request(Request { frame: 0, playing: true, dir: 1, ahead: upto + 1, fill: false });
    let t = Instant::now();
    while src.field_detection().1.is_none() && t.elapsed() < Duration::from_secs(20) && !cancel.load(Ordering::Relaxed) {
        if src.get(upto).is_some() {
            break;
        }
        std::thread::sleep(Duration::from_millis(5));
    }
    let detected = src.field_detection().1;
    let interlaced = detected.map(|(i, _)| i).or(src.interlaced()).unwrap_or(false);
    let method = method.or_else(|| interlaced.then_some(DeintMethod::NnediAdaptive))?;
    Some(SourceDeint { method, tff: detected.map(|(_, tff)| tff), fields: field_rate })
}

/// The decoded frames `lo..=hi`, waiting for the decoder (exact frames only).
fn frames(src: &Source, lo: i64, hi: i64, cancel: &AtomicBool) -> anyhow::Result<Vec<Arc<VideoFrame>>> {
    src.request(Request { frame: lo, playing: true, dir: 1, ahead: hi - lo + 12, fill: false });
    let mut out = Vec::with_capacity((hi - lo + 1) as usize);
    for i in lo..=hi {
        let t = Instant::now();
        loop {
            if let Some(f) = src.get(i) {
                out.push(f);
                break;
            }
            if cancel.load(Ordering::Relaxed) {
                anyhow::bail!("cancelled");
            }
            if t.elapsed() > Duration::from_secs(60) {
                anyhow::bail!("frame {i} could not be decoded");
            }
            std::thread::sleep(Duration::from_micros(500));
        }
    }
    Ok(out)
}

/// The output picture: canvas size, where the effects' picture goes on it, and the
/// sample aspect ratio.
#[derive(Clone, Copy, Debug)]
pub struct Geometry {
    pub w: u32,
    pub h: u32,
    pub content: (i32, i32, u32, u32),
    pub sar: (i32, i32),
}

/// Output geometry for a picture of `pw` x `ph` shown at aspect `dar` (display
/// width / height), with `align` the size multiple the format needs.
pub fn geometry(s: &ExportSettings, pw: u32, ph: u32, dar: f64, align: (u32, u32)) -> Geometry {
    let (ax, ay) = (align.0.max(1), align.1.max(1));
    let alx = |v: u32| (v / ax * ax).max(ax);
    let aly = |v: u32| (v / ay * ay).max(ay);
    let dar = if dar > 0.0 { dar } else { pw as f64 / ph.max(1) as f64 };
    match s.size {
        Size::Source | Size::Scale(_) => {
            let k = if let Size::Scale(p) = s.size { (p as f64 / 100.0).clamp(0.01, 16.0) } else { 1.0 };
            let (tw, th) = (((pw as f64 * k).round() as u32).max(1), ((ph as f64 * k).round() as u32).max(1));
            let (w, h) = (alx(tw), aly(th));
            // Rounded down to the format's multiple: crop the few pixels, centred.
            let content = if k == 1.0 { (-((pw - w.min(pw)) as i32 / 2), -((ph - h.min(ph)) as i32 / 2), pw, ph) } else { (-((tw - w.min(tw)) as i32 / 2), -((th - h.min(th)) as i32 / 2), tw, th) };
            // Non-square pixels stay as they were.
            let sar = dar * ph as f64 / pw as f64;
            let sar = if (sar - 1.0).abs() < 1e-3 {
                (1, 1)
            } else {
                fraction(sar, 1000)
            };
            Geometry { w, h, content, sar }
        }
        Size::Fixed(fw, fh) => {
            let (tw, th) = match (fw, fh) {
                (0, 0) => (pw, ph),
                (w, 0) => (w, ((w as f64 / dar).round() as u32).max(1)),
                (0, h) => (((h as f64 * dar).round() as u32).max(1), h),
                (w, h) => (w, h),
            };
            let (w, h) = (alx(tw), aly(th));
            let r = w as f64 / h as f64;
            let (cw, ch) = match s.fit {
                // One side from the aspect: only rounding differs, fill the frame.
                _ if fw == 0 || fh == 0 => (w, h),
                Fit::Stretch => (w, h),
                Fit::Fit => if r > dar { (((h as f64 * dar).round() as u32).max(1), h) } else { (w, ((w as f64 / dar).round() as u32).max(1)) },
                Fit::Fill => if r > dar { (w, ((w as f64 / dar).round() as u32).max(1)) } else { (((h as f64 * dar).round() as u32).max(1), h) },
            };
            Geometry { w, h, content: ((w as i32 - cw as i32) / 2, (h as i32 - ch as i32) / 2, cw, ch), sar: (1, 1) }
        }
    }
}

/// Everything decided about an export once its source is open.
pub struct Plan {
    pub encoder: String,
    pub pix_fmt: String,
    pub filter: Option<(String, String)>,
    pub rate: (i32, i32),
    /// Stack pictures per source frame, and slow-motion factor.
    pub subs: i64,
    pub slow: i64,
    /// Output frames.
    pub total: i64,
    pub range: (i64, i64),
    /// Width and height multiples the format needs.
    pub align: (u32, u32),
    pub alpha: bool,
    pub conv: ColorConv,
    pub color: ColorInfo,
    pub options: Vec<(String, String)>,
    pub rate_control: RateControl,
    pub muxer: &'static str,
    pub path: String,
    pub sequence: bool,
    /// How the output rate is reached (Retime when it is the stack's own).
    pub conform: Conform,
    /// Pictures per second sampled from the stack: the frame rate, or the field rate
    /// when interlaced.
    pub picture_rate: (i32, i32),
    /// Interlaced: Some(top field first).
    pub field_order: Option<bool>,
    /// Encoded on the GPU by our encoder (`encoder` is then one of `GPU_ENCODERS`).
    pub gpu: Option<brp_ai::gpuenc::Codec>,
    pub precoded: Option<encode::Precoded>,
    /// OpenColorIO's output transform, when on.
    pub ocio: Option<crate::color::ExportOcio>,
    /// The output size the codec requires (DNxHD).
    pub force_size: Option<(u32, u32)>,
}

fn plan(job: &ExportJob, src: &Source, deint: Option<SourceDeint>, gpu_name: &str) -> anyhow::Result<Plan> {
    let s = &job.settings;
    let info = &src.info;
    let c = container(&s.container);
    let fam = s.family();
    let interlaced_out = s.interlaced && s.fps.is_some();
    let mut encoder = match (s.codec.as_str(), fam) {
        (OTHER, _) => s.encoder.clone(),
        (_, Some(f)) => {
            let mut e = pick_encoder(f, &s.encoder, gpu_name).ok_or_else(|| anyhow::anyhow!("no encoder for {} in this FFmpeg", f.label))?;
            // Interlaced: FFmpeg's hardware encoders do not offer field coding (ours do).
            if interlaced_out && !is_ours(&e) && encode::encoder(&e).is_some_and(|i| i.hardware) {
                e = pick_encoder(f, "", gpu_name).ok_or_else(|| anyhow::anyhow!("no encoder for {}", f.label))?;
            }
            e
        }
        _ => anyhow::bail!("unknown codec {}", s.codec),
    };
    // DXV3 with alpha: ours writes it; FFmpeg only with CVExplorer's patch.
    if s.codec == "dxv" && s.alpha && !dxv_alpha_available() && ours_ready("dxv_gpu") {
        encoder = "dxv_gpu".into();
    }
    if is_ours(&encoder) && !ours_ready(&encoder) {
        encoder = ffmpeg_encoder_of(&encoder).to_string();
    }
    let enc = encode::encoder(ffmpeg_encoder_of(&encoder)).ok_or_else(|| anyhow::anyhow!("this FFmpeg has no encoder {encoder}"))?;
    let alpha = s.alpha && s.alpha_possible();
    // HDR sources stay HDR when asked and the codec can hold it (10 bits or more);
    // otherwise they are tone mapped to SDR.
    // OpenColorIO (DESIGN.md §7d): the output is OCIO's for this format: linear for EXR,
    // sRGB for still images, PQ/HLG for HDR kept HDR, Rec.709 for other video.
    let ocio = match &job.ocio {
        Some(o) => {
            use brp_ocio::auto::Output;
            let kind = if s.codec == "exr" {
                Output::Linear
            } else if IMAGE_CODECS.contains(&s.codec.as_str()) {
                Output::Image
            } else if info.color.is_hdr() && !s.tonemap && s.hdr_possible() {
                if info.color.transfer == Transfer::Hlg { Output::Hlg } else { Output::Pq }
            } else {
                Output::SdrVideo
            };
            Some(crate::color::export_ocio(&o.lib, &o.choices, &o.settings, info, kind).map_err(crate::color::friendly)?)
        }
        None => None,
    };
    let keep_hdr = match &ocio {
        Some(o) => matches!(o.encoding, brp_ocio::auto::Encoding::Pq | brp_ocio::auto::Encoding::Hlg) && s.hdr_possible(),
        None => info.color.is_hdr() && !s.tonemap && s.hdr_possible(),
    };
    let mut options: Vec<(String, String)> = fam.map(|f| f.options.iter().map(|(k, v)| (k.to_string(), v.to_string())).collect()).unwrap_or_default();
    let mut profile = s.profile.clone();
    // Pixel format: chosen, or the codec's (alpha variant when asked), mapped onto
    // what this encoder takes.
    let mut pix = if !s.pix_fmt.is_empty() {
        s.pix_fmt.clone()
    } else {
        match (fam, s.codec.as_str()) {
            (Some(f), "prores") => {
                if alpha {
                    if !profile.starts_with("4444") {
                        profile = "4444".into();
                    }
                    f.pix_alpha.unwrap().to_string()
                } else if profile.starts_with("4444") {
                    "yuv444p10le".into()
                } else {
                    f.pix.into()
                }
            }
            (Some(_), "dnxhr") if keep_hdr && !matches!(profile.as_str(), "dnxhr_hqx" | "dnxhr_444" | "dnxhd_hqx") => {
                profile = "dnxhr_hqx".into();
                "yuv422p10le".into()
            }
            (Some(_), "vp9") if keep_hdr && !alpha => "yuv420p10le".into(),
            (Some(_), "dnxhr") => match profile.as_str() {
                "dnxhr_hqx" | "dnxhd_hqx" => "yuv422p10le".into(),
                "dnxhr_444" => "yuv444p10le".into(),
                _ => "yuv422p".into(),
            },
            (Some(f), _) => if alpha { f.pix_alpha.unwrap_or(f.pix) } else { f.pix }.to_string(),
            (None, _) => {
                let want_alpha = alpha;
                enc.pix_fmts.iter().find(|p| encode::pix_fmt_info(p).is_some_and(|i| i.alpha == want_alpha && !i.hw)).or(enc.pix_fmts.first()).cloned().unwrap_or_else(|| "yuv420p".into())
            }
        }
    };
    if s.codec == "hap" && alpha {
        options.push(("format".into(), "hap_alpha".into()));
    }
    if s.codec == "dxv" {
        options.push(("format".into(), if alpha { "dxt5" } else { "dxt1" }.into()));
    }
    if !enc.pix_fmts.is_empty() && !enc.pix_fmts.contains(&pix) && s.codec != "gif" {
        let alt = match pix.as_str() {
            "yuv420p" => "nv12",
            "yuv420p10le" => "p010le",
            "nv12" => "yuv420p",
            "p010le" => "yuv420p10le",
            _ => "",
        };
        if enc.pix_fmts.iter().any(|p| p == alt) {
            pix = alt.into();
        } else {
            let want = encode::pix_fmt_info(&pix);
            let pick = enc.pix_fmts.iter().filter_map(|p| encode::pix_fmt_info(p)).filter(|i| !i.hw).min_by_key(|i| {
                let w = want.as_ref();
                (w.is_some_and(|w| w.alpha != i.alpha) as i32) * 100 + (w.map_or(0, |w| (w.bits as i32 - i.bits as i32).abs())) + (w.is_some_and(|w| w.rgb != i.rgb) as i32) * 10
            });
            pix = pick.map(|i| i.name).ok_or_else(|| anyhow::anyhow!("{encoder} takes no pixel format we can write"))?;
        }
    }
    if !profile.is_empty() && fam.is_some_and(|f| !f.profiles.is_empty()) {
        // DNxHD: FFmpeg picks the CID from size, depth, scan and bit rate (set below).
        let v = if profile.starts_with("dnxhd_") { "dnxhd".to_string() } else { profile.clone() };
        options.push(("profile".into(), v));
    }
    for kv in s.extra.split_whitespace() {
        if let Some((k, v)) = kv.split_once('=') {
            options.push((k.trim_start_matches('-').into(), v.into()));
        }
    }
    let filter = (s.codec == "gif").then(|| ("split[a][b];[a]palettegen=stats_mode=single:max_colors=256[p];[b][p]paletteuse=new=1:dither=sierra2_4a".to_string(), "rgb24".to_string()));
    let mut rate_control = match (s.codec.as_str(), fam) {
        ("xdcam", _) => RateControl::Cbr(50000),
        (_, Some(f)) if f.quality.is_some() || f.bitrate.is_some() => match s.quality {
            Some(q) if f.quality.is_some() => RateControl::Quality(q),
            _ if f.bitrate.is_some() => RateControl::Bitrate(s.kbps.max(100)),
            _ => RateControl::Default,
        },
        (OTHER, _) => match s.quality {
            Some(q) => RateControl::Quality(q),
            None if s.kbps > 0 => RateControl::Bitrate(s.kbps),
            None => RateControl::Default,
        },
        _ => RateControl::Default,
    };

    // Timing.
    let stages = &job.stack;
    let subs = ai::subs_of(stages, deint);
    let slow = ai::slowdown_of(stages);
    let fields = if deint.is_some_and(|d| d.fields) { 2 } else { 1 };
    let last = src.frame_count() - 1;
    let (a, b) = job.range.unwrap_or((0, last));
    let range = (a.clamp(0, last), b.clamp(0, last).max(a.clamp(0, last)));
    let q = standard_rate(info.frame_rate_q);
    let stack_rate = {
        let (n, d) = (q.0 as i64 * fields, q.1.max(1) as i64);
        let g = gcd(n, d);
        ((n / g) as i32, (d / g) as i32)
    };
    let rate = s.fps.unwrap_or(stack_rate);
    // Interlaced: each frame is two fields, each a picture of its own moment.
    let interlaced = s.interlaced && s.fps.is_some();
    if interlaced && !s.interlace_possible() {
        anyhow::bail!("{} cannot store interlaced video", fam.map_or(s.codec.as_str(), |f| f.label));
    }
    let picture_rate = if interlaced {
        let (n, d) = (rate.0 as i64 * 2, rate.1 as i64);
        let g = gcd(n, d);
        ((n / g) as i32, (d / g) as i32)
    } else {
        rate
    };
    let conform = if (rate_f(picture_rate) - rate_f(stack_rate)).abs() < 1e-9 { Conform::Retime } else { s.conform };
    let pictures = match conform {
        Conform::Retime => (range.1 - range.0 + 1) * subs,
        _ => {
            let secs = (src.frame_end_time(range.1) - src.frame_time(range.0)) * slow as f64;
            ((secs * rate_f(picture_rate)) + 1e-6).round().max(1.0) as i64
        }
    };
    let total = if interlaced { (pictures + 1) / 2 } else { pictures };

    // DNxHD: fixed sizes (1080 or 720; 1440x1080 too), and FFmpeg's encoder is
    // asked by bit rate, which follows from the CID and the frame rate.
    let mut force_size = None;
    if s.codec == "dnxhr" && profile.starts_with("dnxhd_") {
        let (w, h) = match s.size {
            Size::Fixed(1280, 720) => (1280, 720),
            Size::Fixed(1440, 1080) => (1440, 1080),
            _ => (1920, 1080),
        };
        let cid = brp_ai::gpuenc::dnxhd::resolve_cid(&profile, w, h, interlaced)
            .and_then(brp_ai::gpuenc::dnxhd::cid)
            .ok_or_else(|| anyhow::anyhow!("DNxHD has no {profile} profile for {w}x{h}{}", if interlaced { " interlaced" } else { "" }))?;
        let mbps = brp_ai::gpuenc::dnxhd::hd_bit_rate(cid, rate_f(rate)).ok_or_else(|| anyhow::anyhow!("DNxHD CID {}: no bit rate", cid.cid))?;
        rate_control = RateControl::Bitrate(mbps * 1000);
        force_size = Some((w, h));
    }

    // Colour.
    let src_c = info.color;
    let float_out = encoder == "exr";
    let sdr = ColorInfo { matrix: Matrix::Bt709, full_range: false, transfer: Transfer::Sdr, primaries: Primaries::Bt709 };
    let (conv, mut color) = if let Some(o) = &ocio {
        // OCIO made the output's encoding: tag the file with it.
        use brp_ocio::auto::Encoding;
        let c = match o.encoding {
            Encoding::Pq => ColorInfo { matrix: Matrix::Bt2020Ncl, full_range: false, transfer: Transfer::Pq, primaries: Primaries::Bt2020 },
            Encoding::Hlg => ColorInfo { matrix: Matrix::Bt2020Ncl, full_range: false, transfer: Transfer::Hlg, primaries: Primaries::Bt2020 },
            Encoding::Linear => ColorInfo { matrix: Matrix::Rgb, full_range: true, transfer: Transfer::Linear, primaries: Primaries::Bt709 },
            Encoding::Sdr => sdr,
        };
        (ColorConv::None, c)
    } else if float_out {
        let conv = if src_c.transfer == Transfer::Linear { ColorConv::None } else { ColorConv::ToLinear { from: src_c.transfer, bt2020: src_c.primaries == Primaries::Bt2020 } };
        (conv, ColorInfo { transfer: Transfer::Linear, primaries: if src_c.is_hdr() { Primaries::Bt709 } else { src_c.primaries }, ..src_c })
    } else if src_c.is_hdr() && !keep_hdr {
        (ColorConv::HdrToSdr { pq: src_c.transfer == Transfer::Pq, bt2020: src_c.primaries == Primaries::Bt2020, peak: 1000.0 / 203.0 }, sdr)
    } else if src_c.transfer == Transfer::Linear {
        (ColorConv::LinearToGamma, sdr)
    } else {
        (ColorConv::None, src_c)
    };
    // YUV output from an RGB source: BT.709 (2020 for HDR), limited range; full range
    // only where the format is (yuvj*) or the YUV source was.
    let src_rgb = info.pix_fmt.starts_with("rgb") || info.pix_fmt.starts_with("bgr") || info.pix_fmt.starts_with("gbr") || info.pix_fmt.starts_with("argb") || info.pix_fmt.starts_with("abgr") || src_c.matrix == Matrix::Rgb;
    if src_rgb && color.matrix == Matrix::Rgb {
        color.matrix = if color.primaries == Primaries::Bt2020 { Matrix::Bt2020Ncl } else { Matrix::Bt709 };
    }
    if src_rgb {
        color.full_range = false;
    }
    if pix.starts_with("yuvj") {
        color.full_range = true;
    }

    let sequence = encode::is_image_sequence(c.muxer);
    let path = if sequence { sequence_pattern(&job.output) } else { job.output.clone() };
    let a = fam.map_or(2, |f| f.align);
    let sub = encode::pix_fmt_info(&pix).map_or((1, 1), |i| (1u32 << i.cw, 1u32 << i.ch));
    let align = (a.max(sub.0), if s.codec == "cfhd" { sub.1 } else { a.max(sub.1) });
    let field_order = interlaced.then_some(s.tff);
    let (gpu, precoded) = if is_ours(&encoder) { gpu_codec(&encoder, &profile, &pix, &color, field_order, alpha)? } else { (None, None) };
    Ok(Plan { encoder, pix_fmt: pix, filter, rate, subs, slow, total, range, align, alpha, conv, color, options, rate_control, muxer: c.muxer, path, sequence, conform, picture_rate, field_order, gpu, precoded, force_size, ocio })
}

/// Our GPU encoder's settings for a plan, and what the muxer needs to know.
fn gpu_codec(encoder: &str, profile: &str, pix: &str, color: &ColorInfo, field_order: Option<bool>, alpha: bool) -> anyhow::Result<(Option<brp_ai::gpuenc::Codec>, Option<encode::Precoded>)> {
    use brp_ai::gpuenc::{self, Codec};
    let info = encode::pix_fmt_info(pix).ok_or_else(|| anyhow::anyhow!("pixel format {pix}"))?;
    let codes = encode::color_codes(color, !info.rgb);
    Ok(match encoder {
        "prores_gpu" => {
            let p = gpuenc::prores::Settings::profile_of(profile).unwrap_or(if info.cw == 0 { 4 } else { 3 });
            let s = gpuenc::prores::Settings { profile: p, alpha: info.alpha, interlaced: field_order, color: codes, vendor: *b"apl0" };
            let pc = encode::Precoded {
                codec: "prores".into(),
                tag: Some(gpuenc::prores::tag(p)),
                profile: p as i32,
                bits_per_coded_sample: if info.alpha { 32 } else { 0 },
                bits_per_raw_sample: 10,
                bit_rate: 0,
                intra_only: true,
            };
            (Some(Codec::ProRes(s)), Some(pc))
        }
        "dnxhd_gpu" => {
            // FFmpeg's profile numbers: DNxHD 0, DNxHR LB 1, SQ 2, HQ 3, HQX 4, 444 5.
            let fp = match profile {
                "dnxhr_lb" => 1,
                "dnxhr_sq" => 2,
                "dnxhr_hq" => 3,
                "dnxhr_hqx" => 4,
                "dnxhr_444" => 5,
                _ => 0,
            };
            let pc = encode::Precoded {
                codec: "dnxhd".into(),
                tag: Some(if fp > 0 { *b"AVdh" } else { *b"AVdn" }),
                profile: fp,
                bits_per_coded_sample: 0,
                bits_per_raw_sample: info.bits as i32,
                bit_rate: 0,
                intra_only: true,
            };
            let _ = codes;
            (Some(Codec::Dnx(gpuenc::dnxhd::Settings { profile: profile.to_string(), interlaced: field_order })), Some(pc))
        }
        "dxv_gpu" => {
            let pc = encode::Precoded { codec: "dxv".into(), tag: Some(*b"DXD3"), profile: -99, bits_per_coded_sample: 0, bits_per_raw_sample: 0, bit_rate: 0, intra_only: true };
            (Some(Codec::Dxv(gpuenc::dxv::Settings { alpha })), Some(pc))
        }
        other => anyhow::bail!("GPU encoder {other} is not available"),
    })
}

/// Which stack picture each output frame is, and the source frames it reads.
struct Timing<'a> {
    src: &'a Source,
    subs: i64,
    slow: i64,
    range: (i64, i64),
    first_tick: i64,
    last_tick: i64,
    last: i64,
    rate: (i32, i32),
    conform: Conform,
    t_in: f64,
    back: i64,
    fwd: i64,
    /// The stack's Slow Motion settings (motion conform makes its pictures alike).
    interp: Option<Vec<f64>>,
}

impl<'a> Timing<'a> {
    fn new(src: &'a Source, p: &Plan, stack: &[Stage], back: i64, fwd: i64) -> Timing<'a> {
        let (in_f, out_f) = p.range;
        Timing {
            src,
            subs: p.subs,
            slow: p.slow,
            range: p.range,
            first_tick: in_f * p.subs,
            last_tick: out_f * p.subs + p.subs - 1,
            last: src.frame_count() - 1,
            rate: p.picture_rate,
            conform: p.conform,
            t_in: src.frame_time(in_f),
            back,
            fwd,
            interp: stack.iter().rev().find_map(|s| match s {
                Stage::Ai { kind: AiKind::Interpolate, params } => Some(params.clone()),
                _ => None,
            }),
        }
    }

    /// The stack's position (ticks, fractional) at output frame `i`'s time: the
    /// source time through the frame index (exact with variable frame rates), slowed
    /// by slow motion.
    fn tau(&self, i: i64) -> f64 {
        let (in_f, out_f) = self.range;
        let secs = self.t_in + i as f64 / rate_f(self.rate) / self.slow as f64 + 1e-9;
        let n = self.src.frame_at_time(secs).clamp(in_f, out_f);
        let (a, b) = (self.src.frame_time(n), self.src.frame_end_time(n));
        let frac = ((secs - a) / (b - a).max(1e-9)).clamp(0.0, 1.0 - 1e-9);
        ((n as f64 + frac) * self.subs as f64).clamp(self.first_tick as f64, self.last_tick as f64)
    }

    fn sample(&self, i: i64) -> Sample {
        match self.conform {
            Conform::Retime => Sample::Tick((self.first_tick + i).min(self.last_tick)),
            Conform::Nearest => Sample::Tick(((self.tau(i) + 1e-6).floor() as i64).min(self.last_tick)),
            Conform::Blend => Sample::Blend(self.tau(i)),
            Conform::Motion => {
                // As the stack's own Slow Motion does it (its settings if there is
                // one): along the motion, frame interpolation where it fails.
                let (model, motion) = match &self.interp {
                    Some(pr) => (ai::rife_model(pr.get(1).copied().unwrap_or(0.0)), (pr.get(2).map(|v| v.round() as i64) != Some(3)).then(|| Motion::from_param(pr.get(2)))),
                    None => (ai::rife_model(0.0), Some(Motion::Builtin)),
                };
                Sample::Motion { tau: self.tau(i), model, motion }
            }
        }
    }

    /// Source frames a sample reads.
    fn span(&self, s: Sample) -> (i64, i64) {
        let (a, b) = match s {
            Sample::Tick(t) => (t, t),
            Sample::Blend(t) | Sample::Motion { tau: t, .. } => (t.floor() as i64, t.floor() as i64 + 1),
        };
        let (fa, fb) = (a.div_euclid(self.subs), b.min(self.last * self.subs).div_euclid(self.subs));
        ((fa - self.back).max(0), (fb + self.fwd + 1).min(self.last))
    }
}

fn run(job: &ExportJob, ctx: &Arc<GpuContext>, progress: &Mutex<Progress>, cancel: &AtomicBool) -> anyhow::Result<()> {
    let t0 = Instant::now();
    let src = match &job.layer {
        Some(l) => Source::open_exr(&job.input, Some(l))?,
        None => Source::open(&job.input)?,
    };
    // A window of frames around the picture being made, not the viewer's big cache.
    let fb = src.info.width as usize * src.info.height as usize * 8;
    src.set_cache_limit((fb * 48).max(256 << 20));
    progress.lock().state = State::Indexing;
    while !src.index_ready() {
        if cancel.load(Ordering::Relaxed) {
            anyhow::bail!("cancelled");
        }
        std::thread::sleep(Duration::from_millis(5));
    }
    let deint = match job.deint {
        DeintChoice::Fixed(d) => d,
        DeintChoice::Detect { mode, field_rate } => detect_deint(&src, mode, field_rate, cancel),
    };
    let p = plan(job, &src, deint, &ctx.device_name)?;
    log::info!(
        "export {} -> {}: {} {} {}/{} fps, {} frames (stack: {} stages, {} pictures/frame, deinterlace {:?})",
        job.input,
        p.path,
        p.encoder,
        p.pix_fmt,
        p.rate.0,
        p.rate.1,
        p.total,
        job.stack.len(),
        p.subs,
        deint.map(|d| (d.method, d.fields))
    );
    {
        let mut g = progress.lock();
        g.state = State::Running;
        g.total = p.total;
        g.rate = p.rate;
        g.encoder = p.encoder.clone();
        g.output = p.path.clone();
    }
    if let Some(dir) = Path::new(&p.path).parent() {
        if !dir.as_os_str().is_empty() {
            std::fs::create_dir_all(dir)?;
        }
    }

    let stages = Arc::new(job.stack.clone());
    let transfer = transfer_of(&src.info.color);
    let last = src.frame_count() - 1;
    let reach = ai::Reach::of(&stages, deint.is_some());
    let (back, fwd) = reach.frames();
    let mut worker = Worker::default();
    // OpenColorIO's output transform on the GPU, when on.
    let ocio = p.ocio.as_ref().map(|o| brp_render::OcioCompute::new(ctx, &o.shader)).transpose()?;
    if let Some(o) = &p.ocio {
        log::info!("export: OCIO {}", o.label);
    }
    let tm = Timing::new(&src, &p, &job.stack, back, fwd);
    let (in_f, out_f) = p.range;
    let first_tick = tm.first_tick;
    let sample_of = |i: i64| tm.sample(i);
    let span = |s: Sample| tm.span(s);
    let job_for = |tick: i64, frames: Vec<Arc<VideoFrame>>| Job::new(stages.clone(), transfer, deint, tick, frames, last, src.info.color, ctx.clone());

    // Stateful shader effects: their history before the first picture, as after a
    // jump in the player.
    let warm = ai::shader_warmup(&stages);
    for t in (first_tick - warm).max(0)..first_tick {
        let (lo, hi) = span(Sample::Tick(t));
        let fr = frames(&src, lo, hi, cancel)?;
        worker.warm(&job_for(t, fr), t)?;
    }

    let dar = src.info.dar;
    let rotation = src.info.rotation;
    let mut geom: Option<(Geometry, PackLayout)> = None;
    let mut enc: Option<(crossbeam_channel::Sender<Vec<u8>>, std::thread::JoinHandle<anyhow::Result<()>>)> = None;
    let audio = (!p.sequence && !job.settings.audio.is_empty() && encode::audio_encoder_exists(&job.settings.audio) && encode::has_audio(&job.input)).then(|| {
        let start = src.info.start_secs + src.frame_time(in_f);
        let end = src.info.start_secs + src.frame_end_time(out_f);
        let length = p.total as f64 / rate_f(p.rate);
        AudioSource { path: job.input.clone(), start, end, tempo: (end - start) / length.max(1e-9), keep_pitch: job.settings.keep_pitch, length }
    });
    let mut last_report = Instant::now();
    let mut fps_t = (Instant::now(), 0i64);
    for i in 0..p.total {
        if cancel.load(Ordering::Relaxed) {
            break;
        }
        // One picture, or the two fields' moments (in time order).
        let samples: Vec<Sample> = if p.field_order.is_some() { vec![sample_of(2 * i), sample_of(2 * i + 1)] } else { vec![sample_of(i)] };
        let sample = samples[0];
        let (lo, hi) = samples.iter().map(|s| span(*s)).fold((i64::MAX, i64::MIN), |a, b| (a.0.min(b.0), a.1.max(b.1)));
        let fr = frames(&src, lo, hi, cancel)?;
        let tick = match sample {
            Sample::Tick(t) => t,
            Sample::Blend(t) | Sample::Motion { tau: t, .. } => t.floor() as i64,
        };
        let job_i = job_for(tick, fr);
        let mut made: Option<(Geometry, PackLayout)> = None;
        let bytes = worker.export(&job_i, &samples, ocio.as_ref(), &mut |(w, h, _c)| {
            let (g, l) = match &geom {
                Some((g, l)) => (*g, l.clone()),
                None => {
                    // Rotated sources are encoded upright as decoded, with the
                    // rotation in the file: a fixed size is meant as displayed.
                    let mut st = job.settings.clone();
                    if let Some((a, b)) = p.force_size {
                        st.size = Size::Fixed(a, b);
                    }
                    let rot = rotation % 180 == 90 && !p.sequence;
                    if let (true, Size::Fixed(a, b)) = (rot, &st.size) {
                        st.size = Size::Fixed(*b, *a);
                    }
                    let g = geometry(&st, w as u32, h as u32, dar, p.align);
                    let l = encode::delivery_layout(&p.pix_fmt, p.filter.as_ref().map(|f| f.1.as_str()), g.w, g.h).ok_or_else(|| anyhow::anyhow!("cannot write pixel format {}", p.pix_fmt))?;
                    (g, l)
                }
            };
            let spec = OutputSpec { content: g.content, conv: p.conv, alpha: p.alpha && l.alpha, kr_kb: kr_kb(p.color.matrix), full_range: p.color.full_range, as_shown: true, layout: l.clone(), tff: p.field_order.unwrap_or(true), encode: p.gpu.clone() };
            made = Some((g, l));
            Ok(spec)
        });
        // A failure of our GPU encoder: the caller exports again with the CPU encoder.
        let bytes = if p.gpu.is_some() { bytes.context(GpuEncodeFailed)? } else { bytes? };
        if geom.is_none() {
            let (g, l) = made.take().unwrap();
            let v = VideoConfig {
                encoder: p.encoder.clone(),
                pix_fmt: p.pix_fmt.clone(),
                width: g.w,
                height: g.h,
                rate: p.rate,
                sar: g.sar,
                rate_control: p.rate_control.clone(),
                options: p.options.clone(),
                gop: if job.settings.codec == "xdcam" { Some(12) } else { None },
                color: Some(p.color),
                rotation: if p.sequence { 0 } else { rotation },
                filter: p.filter.clone(),
                field_order: p.field_order,
                precoded: p.precoded.clone(),
            };
            let ac = AudioConfig { encoder: job.settings.audio.clone(), kbps: if job.settings.audio.starts_with("pcm") || job.settings.audio == "flac" || job.settings.audio == "alac" { 0 } else { job.settings.audio_kbps }, sample_rate: if p.muxer == "mxf" { 48000 } else { 0 } };
            let writer = Writer::create(&p.path, Some(p.muxer), &v, audio.as_ref().map(|a| (&ac, a)), &[])?;
            anyhow::ensure!(writer.is_precoded() || writer.layout().total == l.total, "internal: picture layout mismatch");
            progress.lock().size = (g.w, g.h);
            let (tx, rx) = crossbeam_channel::bounded::<Vec<u8>>(3);
            let handle = std::thread::Builder::new().name("brp-encode".into()).spawn(move || -> anyhow::Result<()> {
                let mut w = writer;
                for pic in rx {
                    w.write(&pic)?;
                }
                w.finish()
            })?;
            enc = Some((tx, handle));
            geom = Some((g, l));
        }
        let (tx, _) = enc.as_ref().unwrap();
        if tx.send(bytes).is_err() {
            // The encoder stopped: its error says why.
            let (_, h) = enc.take().unwrap();
            return Err(h.join().map_err(|_| anyhow::anyhow!("encoder thread panicked"))?.err().unwrap_or_else(|| anyhow::anyhow!("encoder stopped")));
        }
        if last_report.elapsed() > Duration::from_millis(100) || i + 1 == p.total {
            let el = fps_t.0.elapsed().as_secs_f64();
            let mut g = progress.lock();
            if el > 0.5 {
                let f = (i + 1 - fps_t.1) as f64 / el;
                g.fps = if g.fps > 0.0 { g.fps * 0.6 + f * 0.4 } else { f };
                fps_t = (Instant::now(), i + 1);
            }
            g.done = i + 1;
            g.elapsed = t0.elapsed().as_secs_f64();
            last_report = Instant::now();
        }
    }
    let cancelled = cancel.load(Ordering::Relaxed);
    progress.lock().state = State::Finishing;
    let Some((tx, h)) = enc else {
        anyhow::bail!(if cancelled { "cancelled" } else { "nothing to export" });
    };
    drop(tx);
    let r = h.join().map_err(|_| anyhow::anyhow!("encoder thread panicked"))?;
    if cancelled {
        remove_output(&p.path, p.sequence, p.total);
        anyhow::bail!("cancelled");
    }
    if let Err(e) = r {
        remove_output(&p.path, p.sequence, p.total);
        return Err(e);
    }
    let mut g = progress.lock();
    g.elapsed = t0.elapsed().as_secs_f64();
    g.done = p.total;
    Ok(())
}

/// Remove what a cancelled or failed export wrote.
fn remove_output(path: &str, sequence: bool, total: i64) {
    if sequence {
        for i in 1..=total + 1 {
            let _ = std::fs::remove_file(path.replace("%06d", &format!("{i:06}")));
        }
    } else {
        let _ = std::fs::remove_file(path);
    }
}

/// A free output name: `dir/stem_export.ext`, numbered if taken.
pub fn default_output(input: &str, dir: Option<&Path>, s: &ExportSettings) -> String {
    let p = Path::new(input);
    let stem = p.file_stem().map(|s| s.to_string_lossy().into_owned()).unwrap_or_else(|| "export".into());
    let dir: PathBuf = dir.map(Path::to_path_buf).or_else(|| std::path::absolute(p).ok().and_then(|a| a.parent().map(Path::to_path_buf))).unwrap_or_default();
    let ext = container(&s.container).ext;
    let seq = encode::is_image_sequence(container(&s.container).muxer);
    for n in 1.. {
        let name = if n == 1 { format!("{stem}_export.{ext}") } else { format!("{stem}_export {n}.{ext}") };
        let cand = dir.join(&name);
        let taken = if seq { Path::new(&sequence_pattern(&cand.to_string_lossy()).replace("%06d", "000001")).exists() } else { cand.exists() };
        if !taken && cand.to_string_lossy() != input {
            return cand.to_string_lossy().into_owned();
        }
    }
    unreachable!()
}

// ---- batch --------------------------------------------------------------------------------

#[derive(Clone, Debug, PartialEq)]
pub enum ItemState {
    Waiting,
    Running,
    Done,
    Failed(String),
    Cancelled,
}

#[derive(Clone, Debug)]
pub struct BatchItem {
    /// Stays the same while the item is in the list (selection).
    pub id: u64,
    pub path: String,
    /// Its own export settings (the batch's defaults when added; edited in the
    /// panel for the selected files).
    pub settings: ExportSettings,
    pub state: ItemState,
    pub output: Option<String>,
    pub progress: f32,
    pub took: Option<f64>,
}

impl BatchItem {
    pub fn new(id: u64, path: String, settings: ExportSettings) -> BatchItem {
        BatchItem { id, path, settings, state: ItemState::Waiting, output: None, progress: 0.0, took: None }
    }
}

/// Files exported one after another. An item that fails is marked and the queue
/// goes on.
#[derive(Default)]
pub struct Batch {
    pub items: Vec<BatchItem>,
    /// Output folder (None: next to each source).
    pub out_dir: Option<PathBuf>,
    /// Running: the item and its export.
    running: Option<(usize, Running)>,
    /// The queue is going (stopped by Cancel).
    pub active: bool,
    /// Settings new files get.
    pub defaults: ExportSettings,
    next_id: u64,
    /// Stack and deinterlacing the queue was started with.
    snapshot: Option<(Vec<Stage>, DeintMode, bool)>,
    /// OpenColorIO the queue was started with (None: off).
    ocio: Option<OcioJob>,
}

impl Batch {
    pub fn add(&mut self, path: String) {
        if !self.items.iter().any(|i| i.path == path && i.state == ItemState::Waiting) {
            self.next_id += 1;
            self.items.push(BatchItem::new(self.next_id, path, self.defaults.clone()));
        }
    }

    pub fn running(&self) -> Option<(usize, Progress)> {
        self.running.as_ref().map(|(i, r)| (*i, r.snapshot()))
    }

    /// Start (or resume) the queue: each file with its own settings, through this
    /// effect stack.
    pub fn start(&mut self, stack: Vec<Stage>, mode: DeintMode, field_rate: bool, ocio: Option<OcioJob>) {
        self.ocio = ocio;
        for it in &mut self.items {
            if matches!(it.state, ItemState::Cancelled) {
                it.state = ItemState::Waiting;
            }
        }
        self.snapshot = Some((stack, mode, field_rate));
        self.active = true;
    }

    /// Stop: cancel the running item; the others stay waiting.
    pub fn cancel(&mut self) {
        self.active = false;
        if let Some((_, r)) = &self.running {
            r.cancel();
        }
    }

    /// Cancel the running item only; the queue goes on.
    pub fn skip(&mut self) {
        if let Some((_, r)) = &self.running {
            r.cancel();
        }
    }

    /// Move on: collect a finished item, start the next. Returns true while busy.
    pub fn poll(&mut self, ctx: Option<&Arc<GpuContext>>) -> bool {
        if let Some((i, r)) = &self.running {
            let p = r.snapshot();
            let i = *i;
            if let Some(it) = self.items.get_mut(i) {
                it.progress = p.fraction();
            }
            if !p.state.ended() {
                return true;
            }
            if let Some(it) = self.items.get_mut(i) {
                it.took = Some(p.elapsed);
                it.state = match p.state {
                    State::Done => ItemState::Done,
                    State::Cancelled => ItemState::Cancelled,
                    State::Failed(e) => ItemState::Failed(e),
                    _ => ItemState::Failed("?".into()),
                };
            }
            self.running = None;
        }
        if !self.active {
            return false;
        }
        let Some(ctx) = ctx else { return false };
        let Some(next) = self.items.iter().position(|i| i.state == ItemState::Waiting) else {
            self.active = false;
            return false;
        };
        let Some((stack, mode, field_rate)) = self.snapshot.clone() else { return false };
        let it = &mut self.items[next];
        let s = it.settings.clone();
        let output = default_output(&it.path, self.out_dir.as_deref(), &s);
        it.output = Some(output.clone());
        it.state = ItemState::Running;
        it.progress = 0.0;
        let job = ExportJob { input: it.path.clone(), output, settings: s, stack, range: None, deint: DeintChoice::Detect { mode, field_rate }, layer: None, ocio: self.ocio.clone() };
        self.running = Some((next, start(job, ctx.clone())));
        true
    }

    /// Waiting files the settings panel edits: the selected ones, or with none
    /// selected every waiting file.
    fn targets<'a>(&'a self, sel: &'a std::collections::HashSet<u64>) -> impl Iterator<Item = &'a BatchItem> + 'a {
        self.items.iter().filter(move |i| i.state == ItemState::Waiting && (sel.is_empty() || sel.contains(&i.id)))
    }

    /// The settings the panel shows: the first file it edits, or (no waiting file)
    /// those files get when added.
    pub fn panel_settings(&self, sel: &std::collections::HashSet<u64>) -> ExportSettings {
        self.targets(sel).next().map(|i| i.settings.clone()).unwrap_or_else(|| self.defaults.clone())
    }

    /// The files the panel edits do not all have the same settings.
    pub fn panel_mixed(&self, sel: &std::collections::HashSet<u64>) -> bool {
        let mut it = self.targets(sel);
        let first = it.next().map(|i| &i.settings);
        first.is_some_and(|f| it.any(|i| &i.settings != f))
    }

    /// Settings changed in the panel: they go to the files it edits, and with none
    /// selected also to files added later.
    pub fn apply_panel(&mut self, sel: &std::collections::HashSet<u64>, st: &ExportSettings) {
        for it in self.items.iter_mut().filter(|i| i.state == ItemState::Waiting && (sel.is_empty() || sel.contains(&i.id))) {
            it.settings = st.clone();
        }
        if sel.is_empty() {
            self.defaults = st.clone();
        }
    }

    pub fn remove_finished(&mut self) {
        self.items.retain(|i| matches!(i.state, ItemState::Waiting | ItemState::Running));
        self.reindex();
    }

    pub fn remove(&mut self, idx: usize) {
        if self.running.as_ref().is_some_and(|(i, _)| *i == idx) {
            return;
        }
        if idx < self.items.len() {
            self.items.remove(idx);
            if let Some((i, _)) = &mut self.running {
                if *i > idx {
                    *i -= 1;
                }
            }
        }
    }

    fn reindex(&mut self) {
        if let Some((i, _)) = &mut self.running {
            if let Some(n) = self.items.iter().position(|it| it.state == ItemState::Running) {
                *i = n;
            }
        }
    }
}

// ---- command line -------------------------------------------------------------------------

/// Settings from a preset name (built-in or saved; case-insensitive, a part of the
/// name is enough).
pub fn find_preset(name: &str, saved: &[ExportSettings]) -> Option<ExportSettings> {
    let n = name.to_lowercase();
    let all: Vec<ExportSettings> = saved.iter().cloned().chain(builtin_presets()).collect();
    all.iter().find(|p| p.name.to_lowercase() == n).or_else(|| all.iter().find(|p| p.name.to_lowercase().contains(&n))).cloned()
}

/// Settings from command-line options over a starting point.
pub fn settings_from_args(args: &[String], saved: &[ExportSettings]) -> anyhow::Result<ExportSettings> {
    let get = |k: &str| args.iter().find_map(|a| a.strip_prefix(&format!("--{k}=")).map(str::to_string));
    let mut s = match get("preset") {
        Some(n) => find_preset(&n, saved).ok_or_else(|| anyhow::anyhow!("no preset like '{n}'"))?,
        None => ExportSettings::default(),
    };
    if let Some(c) = get("container") {
        s.container = c;
        if get("codec").is_none() {
            if let Some(f) = FAMILIES.iter().find(|f| f.containers.contains(&s.container.as_str())) {
                s.codec = f.id.into();
            }
        }
        s.apply_family_defaults();
    }
    if let Some(c) = get("codec") {
        s.codec = c;
        s.apply_family_defaults();
    }
    if let Some(e) = get("encoder") {
        s.encoder = e;
    }
    if let Some(p) = get("profile") {
        s.profile = p;
    }
    if let Some(p) = get("pix-fmt") {
        s.pix_fmt = p;
    }
    if let Some(q) = get("quality") {
        s.quality = q.parse().ok();
    }
    if let Some(k) = get("kbps") {
        s.kbps = k.parse()?;
        s.quality = None;
    }
    if args.iter().any(|a| a == "--alpha") {
        s.alpha = true;
    }
    if args.iter().any(|a| a == "--no-alpha") {
        s.alpha = false;
    }
    if let Some(a) = get("audio") {
        s.audio = if a == "none" { String::new() } else { a };
    }
    if let Some(p) = get("pitch") {
        s.keep_pitch = p != "shift";
    }
    if let Some(sz) = get("size") {
        s.size = if sz == "source" {
            Size::Source
        } else if let Some(p) = sz.strip_suffix('%') {
            Size::Scale(p.parse()?)
        } else {
            let (w, h) = sz.split_once('x').ok_or_else(|| anyhow::anyhow!("--size=WxH"))?;
            Size::Fixed(w.parse().unwrap_or(0), h.parse().unwrap_or(0))
        };
    }
    if let Some(f) = get("fit") {
        s.fit = match f.as_str() {
            "fill" => Fit::Fill,
            "stretch" => Fit::Stretch,
            _ => Fit::Fit,
        };
    }
    if let Some(r) = get("fps") {
        if r == "source" {
            s.fps = None;
            s.interlaced = false;
        } else {
            let (q, i) = parse_rate_spec(&r).ok_or_else(|| anyhow::anyhow!("bad --fps (25, 29.97p, 59.94i, 50i, 30000/1001)"))?;
            s.fps = Some(q);
            s.interlaced = i;
        }
    }
    if args.iter().any(|a| a == "--bff") {
        s.tff = false;
    }
    if let Some(c) = get("conform") {
        s.conform = Conform::from_id(&c).ok_or_else(|| anyhow::anyhow!("--conform=nearest|blend|motion|retime"))?;
    }
    if args.iter().any(|a| a == "--tonemap") {
        s.tonemap = true;
    }
    if let Some(x) = get("options") {
        s.extra = x;
    }
    Ok(s)
}

/// `brp-player <file> --export=<out> [options]` or `brp-player <files...>
/// --batch-dir=<folder> [options]`: export without a window. Returns the exit code.
pub fn cli(args: &[String], saved: &[ExportSettings], gpu: Option<String>) -> i32 {
    match cli_inner(args, saved, gpu) {
        Ok(()) => 0,
        Err(e) => {
            eprintln!("export failed: {e:#}");
            1
        }
    }
}

fn cli_inner(args: &[String], saved: &[ExportSettings], gpu: Option<String>) -> anyhow::Result<()> {
    let get = |k: &str| args.iter().find_map(|a| a.strip_prefix(&format!("--{k}=")).map(str::to_string));
    let inputs: Vec<String> = args.iter().skip(1).filter(|a| !a.starts_with("--")).cloned().collect();
    anyhow::ensure!(!inputs.is_empty(), "no input file");
    let settings = settings_from_args(args, saved)?;
    let mut fx = crate::effects::Effects::default();
    if let Some(list) = get("fx") {
        fx.add_from_arg(&list);
    }
    let stack = fx.all_stages();
    let mode = match get("deint").as_deref() {
        Some("off") => DeintMode::Off,
        Some("best") => DeintMode::Best,
        Some("nnedi") => DeintMode::Nnedi,
        Some("bwdif") => DeintMode::Bwdif,
        _ => DeintMode::Auto,
    };
    let field_rate = get("field-rate").is_none_or(|v| v != "0");
    let range = get("range").and_then(|r| {
        let (a, b) = r.split_once('-')?;
        Some((a.parse().ok()?, b.parse().ok()?))
    });
    let ctx = Arc::new(GpuContext::headless(brp_render::GpuInstance::new()?, gpu.as_deref())?);
    brp_media::gpudec::set_factory(Some(brp_render::gpudec::factory(&ctx)));
    eprintln!("GPU: {}", ctx.device_name);
    if settings.encoder == "gpu" {
        // The choice needs the probe's answer.
        probe_hardware();
        let t = Instant::now();
        while !hardware_probed() && t.elapsed() < Duration::from_secs(10) {
            std::thread::sleep(Duration::from_millis(50));
        }
    }
    let report = |p: &Progress| {
        let eta = p.eta().map_or(String::new(), |e| format!(", {e:.0} s left"));
        eprintln!("  {:?} {}/{} ({:.0}%), {:.1} fps{eta}", p.state, p.done, p.total, p.fraction() * 100.0, p.fps);
    };
    if let Some(out) = get("export") {
        let job = ExportJob { input: inputs[0].clone(), output: out, settings, stack, range, deint: DeintChoice::Detect { mode, field_rate }, layer: get("layer"), ocio: ocio_from_args(args) };
        let r = start(job, ctx.clone());
        let mut last = Instant::now();
        // BRP_TEST_CANCEL_AFTER=secs: cancel then (checks that a cancelled export leaves no file).
        let cancel_after = std::env::var("BRP_TEST_CANCEL_AFTER").ok().and_then(|s| s.parse::<f64>().ok());
        let t_start = Instant::now();
        while !r.ended() {
            if cancel_after.is_some_and(|c| t_start.elapsed().as_secs_f64() > c) {
                r.cancel();
            }
            std::thread::sleep(Duration::from_millis(50));
            if last.elapsed() > Duration::from_secs(1) {
                report(&r.snapshot());
                last = Instant::now();
            }
        }
        let p = r.snapshot();
        report(&p);
        if p.state == State::Cancelled {
            anyhow::bail!("cancelled");
        }
        if let State::Failed(e) = p.state {
            anyhow::bail!("{e}");
        }
        eprintln!("wrote {} ({}x{}, {} frames at {} fps, {} in {:.1} s)", p.output, p.size.0, p.size.1, p.total, rate_label(p.rate), p.encoder, p.elapsed);
        return Ok(());
    }
    let dir = get("batch-dir").ok_or_else(|| anyhow::anyhow!("--export=<file> or --batch-dir=<folder>"))?;
    let mut b = Batch { out_dir: Some(PathBuf::from(dir)), defaults: settings, ..Default::default() };
    for i in inputs {
        b.add(i);
    }
    b.start(stack, mode, field_rate, ocio_from_args(args));
    let mut last = Instant::now();
    while b.poll(Some(&ctx)) {
        std::thread::sleep(Duration::from_millis(50));
        if last.elapsed() > Duration::from_secs(1) {
            if let Some((i, p)) = b.running() {
                eprint!("[{}/{}] ", i + 1, b.items.len());
                report(&p);
            }
            last = Instant::now();
        }
    }
    let mut failed = 0;
    for it in &b.items {
        eprintln!("{:?}: {} -> {}", it.state, it.path, it.output.as_deref().unwrap_or("-"));
        failed += matches!(it.state, ItemState::Failed(_)) as i32;
    }
    anyhow::ensure!(failed == 0, "{failed} item(s) failed");
    Ok(())
}

/// `brp-player <source> --verify=<exported> [the export's options] [--frames=a,b,c]`:
/// checks an export against the player. Each listed output frame is decoded from
/// the exported file and compared with the picture the player makes for that
/// moment (same stack, same worker, read back as the window gets it). Also checks
/// frame count, frame rate and size against the plan. Exit code 1 on a mismatch.
pub fn verify(args: &[String], saved: &[ExportSettings], gpu: Option<String>) -> i32 {
    match verify_inner(args, saved, gpu) {
        Ok(true) => 0,
        Ok(false) => 1,
        Err(e) => {
            eprintln!("verify failed: {e:#}");
            1
        }
    }
}

fn open_indexed(path: &str) -> anyhow::Result<Source> {
    let s = Source::open(path)?;
    let t = Instant::now();
    while !s.index_ready() && t.elapsed() < Duration::from_secs(60) {
        std::thread::sleep(Duration::from_millis(5));
    }
    Ok(s)
}

fn verify_inner(args: &[String], saved: &[ExportSettings], gpu: Option<String>) -> anyhow::Result<bool> {
    let get = |k: &str| args.iter().find_map(|a| a.strip_prefix(&format!("--{k}=")).map(str::to_string));
    let input = args.iter().skip(1).find(|a| !a.starts_with("--")).cloned().ok_or_else(|| anyhow::anyhow!("no source file"))?;
    let exported = get("verify").unwrap();
    let settings = settings_from_args(args, saved)?;
    let mut fx = crate::effects::Effects::default();
    if let Some(list) = get("fx") {
        fx.add_from_arg(&list);
    }
    let stack = fx.all_stages();
    let mode = match get("deint").as_deref() {
        Some("off") => DeintMode::Off,
        Some("best") => DeintMode::Best,
        Some("nnedi") => DeintMode::Nnedi,
        Some("bwdif") => DeintMode::Bwdif,
        _ => DeintMode::Auto,
    };
    let field_rate = get("field-rate").is_none_or(|v| v != "0");
    let range = get("range").and_then(|r| {
        let (a, b) = r.split_once('-')?;
        Some((a.parse().ok()?, b.parse().ok()?))
    });
    let ctx = Arc::new(GpuContext::headless(brp_render::GpuInstance::new()?, gpu.as_deref())?);
    brp_media::gpudec::set_factory(Some(brp_render::gpudec::factory(&ctx)));
    let never = AtomicBool::new(false);
    let layer = get("layer");
    let src = match &layer {
        Some(l) => Source::open_exr(&input, Some(l))?,
        None => open_indexed(&input)?,
    };
    src.set_cache_limit(1 << 30);
    let deint = detect_deint(&src, mode, field_rate, &never);
    let job = ExportJob { input: input.clone(), output: exported.clone(), settings: settings.clone(), stack: stack.clone(), range, deint: DeintChoice::Fixed(deint), layer, ocio: ocio_from_args(args) };
    let p = plan(&job, &src, deint, &ctx.device_name)?;
    let out_path = if p.sequence { sequence_pattern(&exported) } else { exported.clone() };
    let out = open_indexed(&out_path)?;
    out.set_cache_limit(1 << 30);
    let mut ok = true;
    let n_out = out.frame_count();
    eprintln!("plan: {} frames at {} fps ({}/{}), {:?}; deinterlace {:?}", p.total, rate_label(p.rate), p.rate.0, p.rate.1, p.conform, deint.map(|d| (d.method, d.fields)));
    eprintln!("file: {} frames at {}/{} fps, {}x{} {} {}", n_out, out.info.frame_rate_q.0, out.info.frame_rate_q.1, out.info.width, out.info.height, out.info.codec, out.info.pix_fmt);
    if n_out != p.total {
        eprintln!("MISMATCH: frame count {n_out}, expected {}", p.total);
        ok = false;
    }
    if !p.sequence && (rate_f(out.info.frame_rate_q) / rate_f(p.rate) - 1.0).abs() > 1e-4 {
        eprintln!("MISMATCH: frame rate {:?}, expected {:?}", out.info.frame_rate_q, p.rate);
        ok = false;
    }
    let stages = Arc::new(stack);
    let (back, fwd) = ai::Reach::of(&stages, deint.is_some()).frames();
    let tm = Timing::new(&src, &p, &stages, back, fwd);
    let last = src.frame_count() - 1;
    let list: Vec<i64> = match get("frames") {
        Some(l) => l.split(',').filter_map(|x| x.trim().parse().ok()).collect(),
        None => vec![0, 1, p.total / 3, p.total / 2, p.total - 2, p.total - 1],
    };
    let mut worker = Worker::default();
    let mut viewer = Worker::default();
    // Stateful shaders and the recurrent matting network depend on the pictures
    // before: make every picture in order from the start, as the export (and
    // playback) did, and compare the listed ones.
    let warm = ai::shader_warmup(&stages);
    let sequential = warm > 0 || stages.iter().any(|s| matches!(s, Stage::Ai { kind: AiKind::Matting | AiKind::Depth | AiKind::DenoiseAi, .. }));
    let tol = get("tolerance").and_then(|t| t.parse::<f64>().ok()).unwrap_or(1.5);
    let transfer = transfer_of(&src.info.color);
    let job_at = |s: Sample| -> anyhow::Result<Job> {
        let (lo, hi) = tm.span(s);
        let tick = match s {
            Sample::Tick(t) => t,
            Sample::Blend(t) | Sample::Motion { tau: t, .. } => t.floor() as i64,
        };
        Ok(Job::new(stages.clone(), transfer, deint, tick, frames(&src, lo, hi, &never)?, last, src.info.color, ctx.clone()))
    };
    let list: Vec<i64> = list.into_iter().filter(|&i| i >= 0 && i < p.total.min(n_out)).collect();
    let order: Vec<i64> = if sequential {
        for t in (tm.first_tick - warm).max(0)..tm.first_tick {
            worker.warm(&job_at(Sample::Tick(t))?, t)?;
        }
        (0..=list.iter().copied().max().unwrap_or(0)).collect()
    } else {
        list.clone()
    };
    for i in order {
        // Interlaced: the two fields' pictures, woven (first in time on the top lines
        // when top field first).
        let (s, r) = match p.field_order {
            None => {
                let s = tm.sample(i);
                (s, worker.reference(&job_at(s)?, s)?)
            }
            Some(tff) => {
                let (sa, sb) = (tm.sample(2 * i), tm.sample(2 * i + 1));
                let a = worker.reference(&job_at(sa)?, sa)?;
                let b = worker.reference(&job_at(sb)?, sb)?;
                let mut w = a.clone();
                let n = a.w * a.h;
                for c in 0..a.channels {
                    for y in 0..a.h {
                        if (y % 2 == 0) != tff {
                            w.data[c * n + y * a.w..][..a.w].copy_from_slice(&b.data[c * n + y * a.w..][..a.w]);
                        }
                    }
                }
                (sa, w)
            }
        };
        if !list.contains(&i) {
            continue;
        }
        // The player's picture in the export's encoding when the export converts it
        // (linear EXR to gamma 2.2, SDR to linear EXR), as `pack.rs` does.
        let mut r = r;
        let n = r.w * r.h;
        // OCIO: the same conversion, on the CPU (checks the GPU's shader too).
        if let Some(o) = &p.ocio {
            let mut px: Vec<f32> = (0..n).flat_map(|k| [r.data[k], r.data[n + k], r.data[2 * n + k]]).collect();
            o.cpu.apply(&mut px);
            for k in 0..n {
                for c in 0..3 {
                    r.data[c * n + k] = px[k * 3 + c];
                }
            }
        }
        match p.conv {
            ColorConv::LinearToGamma => r.data[..3 * n].iter_mut().for_each(|v| *v = v.clamp(0.0, 1.0).powf(1.0 / 2.2)),
            ColorConv::ToLinear { from: Transfer::Sdr, .. } => r.data[..3 * n].iter_mut().for_each(|v| *v = v.max(0.0).powf(2.4)),
            ColorConv::None => {}
            c => eprintln!("note: the export converts colour ({c:?}); compared without that conversion"),
        }
        // The exported frame as the player would show that file (its own YUV to RGB).
        let f = frames(&out, i, i, &never)?;
        let e = viewer.reference(&Job::new(Arc::new(Vec::new()), transfer_of(&out.info.color), None, i, f, out.frame_count() - 1, out.info.color, ctx.clone()), Sample::Tick(i))?;
        if (e.w, e.h) != (r.w, r.h) {
            eprintln!("frame {i}: exported {}x{}, the player's picture {}x{} (scaled export: compared after scaling is not done)", e.w, e.h, r.w, r.h);
            continue;
        }
        let n = r.w * r.h;
        let ch = 3 + (e.channels > 3 && r.channels > 3) as usize;
        let (mut max, mut sum, mut sq) = (0f64, 0f64, 0f64);
        for c in 0..ch {
            for k in 0..n {
                let mut rv = r.data[c.min(r.channels - 1) * n + k].clamp(0.0, 1.0) as f64;
                let mut ev = e.data[c * n + k] as f64;
                // With alpha, colours count as seen (weighted by their alpha): under
                // transparent pixels premultiplied codecs (DXV3) keep no colour.
                if ch == 4 && c < 3 {
                    rv *= r.data[3 * n + k].clamp(0.0, 1.0) as f64;
                    ev *= e.data[3 * n + k] as f64;
                }
                let d = ((rv - ev) * 255.0).abs();
                max = max.max(d);
                sum += d;
                sq += d * d;
            }
        }
        let cnt = (n * ch) as f64;
        let mse = sq / cnt;
        let psnr = if mse > 0.0 { 10.0 * (255.0f64 * 255.0 / mse).log10() } else { f64::INFINITY };
        let good = sum / cnt <= tol;
        ok &= good;
        eprintln!("frame {i:>5} = {s:?}: mean |d| {:.3}, max {:.2} (8-bit levels), PSNR {psnr:.1} dB {}", sum / cnt, max, if good { "ok" } else { "MISMATCH" });
    }
    eprintln!("{}", if ok { "verify: OK" } else { "verify: FAILED" });
    Ok(ok)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn presets_round_trip() {
        for p in builtin_presets() {
            assert_eq!(ExportSettings::from_json(&p.to_json()), p, "{}", p.name);
        }
        let mut s = ExportSettings { name: "Mine".into(), size: Size::Fixed(1280, 0), fit: Fit::Fill, fps: Some((25, 1)), conform: Conform::Motion, quality: None, kbps: 8000, ..Default::default() };
        assert_eq!(ExportSettings::from_json(&s.to_json()), s);
        s.size = Size::Scale(50.0);
        assert_eq!(ExportSettings::from_json(&s.to_json()), s);
    }

    #[test]
    fn presets_saved_with_settings() {
        let dir = std::env::temp_dir().join(format!("brp_export_test_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        // Settings live in %APPDATA%\BRP Player (this test process only).
        unsafe { std::env::set_var("APPDATA", &dir) };
        let mut st = crate::settings::Settings::load();
        let p = ExportSettings { name: "Delivery 25p".into(), fps: Some((25, 1)), conform: Conform::Blend, ..Default::default() };
        st.export_presets = vec![p.to_json()];
        st.export_last = Some(p.to_json());
        st.batch_dir = Some("D:/out".into());
        st.save();
        let back = crate::settings::Settings::load();
        assert_eq!(back.export_presets.len(), 1);
        assert_eq!(ExportSettings::from_json(&back.export_presets[0]), p);
        assert_eq!(back.batch_dir.as_deref(), Some("D:/out"));
        let saved: Vec<ExportSettings> = back.export_presets.iter().map(ExportSettings::from_json).collect();
        assert_eq!(find_preset("delivery 25", &saved), Some(p));
        assert!(find_preset("ProRes 422", &saved).is_some());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn rates() {
        assert_eq!(parse_rate("23.976"), Some((24000, 1001)));
        assert_eq!(parse_rate("29.97"), Some((30000, 1001)));
        assert_eq!(parse_rate("59.94"), Some((60000, 1001)));
        assert_eq!(parse_rate("25"), Some((25, 1)));
        assert_eq!(parse_rate("30000/1001"), Some((30000, 1001)));
        assert_eq!(standard_rate((2997, 100)), (30000, 1001));
        assert_eq!(standard_rate((25, 1)), (25, 1));
        assert_eq!(standard_rate((12500, 1000)), (12500, 1000));
    }

    #[test]
    fn geometry_modes() {
        let mut s = ExportSettings { size: Size::Fixed(600, 600), ..Default::default() };
        let dar = 16.0 / 9.0;
        let g = geometry(&s, 1920, 1080, dar, (2, 2));
        assert_eq!((g.w, g.h, g.content), (600, 600, (0, 131, 600, 338)));
        s.fit = Fit::Fill;
        let g = geometry(&s, 1920, 1080, dar, (2, 2));
        assert_eq!((g.w, g.h, g.content), (600, 600, (-233, 0, 1067, 600)));
        s.fit = Fit::Stretch;
        assert_eq!(geometry(&s, 1920, 1080, dar, (2, 2)).content, (0, 0, 600, 600));
        // One side from the aspect, rounded to even: fills the frame.
        s.size = Size::Fixed(1001, 0);
        let g = geometry(&s, 1920, 1080, dar, (2, 2));
        assert_eq!((g.w, g.h, g.content), (1000, 562, (0, 0, 1000, 562)));
        // Same size, odd source, 4:2:0: a pixel cropped, centred.
        s.size = Size::Source;
        let g = geometry(&s, 1921, 1081, 1921.0 / 1081.0, (2, 2));
        assert_eq!((g.w, g.h, g.content), (1920, 1080, (0, 0, 1921, 1081)));
        // CineForm: width multiple of 16 only.
        let g = geometry(&s, 1000, 563, 1000.0 / 563.0, (16, 1));
        assert_eq!((g.w, g.h), (992, 563));
        // Anamorphic PAL 16:9 keeps its pixel aspect.
        let g = geometry(&s, 720, 576, 16.0 / 9.0, (2, 2));
        assert_eq!(g.sar, (64, 45));
    }
}

/// The fraction nearest `x` with a denominator up to `max_den` (continued fractions).
fn fraction(x: f64, max_den: i64) -> (i32, i32) {
    let (mut h0, mut h1, mut k0, mut k1) = (0i64, 1i64, 1i64, 0i64);
    let mut v = x;
    loop {
        let a = v.floor();
        let (h2, k2) = (a as i64 * h1 + h0, a as i64 * k1 + k0);
        if k2 > max_den {
            break;
        }
        (h0, h1, k0, k1) = (h1, h2, k1, k2);
        let f = v - a;
        if f < 1e-9 || ((h1 as f64 / k1 as f64) - x).abs() < 1e-7 {
            break;
        }
        v = 1.0 / f;
    }
    (h1.max(1) as i32, k1.max(1) as i32)
}

#[cfg(test)]
mod batch_tests {
    use super::*;
    use std::collections::HashSet;

    /// The bug: a file added first, then the panel set to 59.94i with nothing
    /// selected. The file must get it (it kept its old settings before).
    #[test]
    fn panel_without_selection_edits_every_file() {
        let mut b = Batch::default();
        b.add("a.mov".into());
        b.add("b.mov".into());
        let none = HashSet::new();
        let mut st = b.panel_settings(&none);
        assert_eq!(st.fps, None);
        st.fps = Some((30000, 1001));
        st.interlaced = true;
        st.conform = Conform::Motion;
        b.apply_panel(&none, &st);
        assert!(b.items.iter().all(|i| i.settings.fps == Some((30000, 1001)) && i.settings.interlaced));
        // Files added later get it too.
        b.add("c.mov".into());
        assert!(b.items[2].settings.interlaced);
        assert!(!b.panel_mixed(&none));
        // A selection edits only those.
        let sel: HashSet<u64> = [b.items[0].id].into();
        let mut one = b.panel_settings(&sel);
        one.interlaced = false;
        one.fps = Some((25, 1));
        b.apply_panel(&sel, &one);
        assert_eq!(b.items[0].settings.fps, Some((25, 1)));
        assert!(b.items[1].settings.interlaced && b.items[2].settings.interlaced);
        assert!(b.panel_mixed(&none));
        // Finished files are not changed.
        b.items[1].state = ItemState::Done;
        b.apply_panel(&none, &one);
        assert!(b.items[1].settings.interlaced);
    }
}

/// Command line: `--ocio` turns OpenColorIO on (each choice the settings' default, else
/// Auto); `--ocio-input=<id>`, `--ocio-display=<id>`, `--ocio-view=<name>`,
/// `--ocio-output=display:<id>|space:<id>` pick one; `--ocio-config=<file>` adds a config of one's own (ids as "maya:ACEScg", "ocio:ACEScg").
fn ocio_from_args(args: &[String]) -> Option<OcioJob> {
    use crate::color::Pick;
    let get = |k: &str| args.iter().find_map(|a| a.strip_prefix(&format!("--{k}=")).map(str::to_string));
    if !args.iter().any(|a| a == "--ocio" || a.starts_with("--ocio-")) {
        return None;
    }
    let settings = crate::settings::Settings::load();
    // The user's config: --ocio-config=<file>, else the one set in the player.
    let custom = get("ocio-config").or(settings.ocio_custom.clone()).map(std::path::PathBuf::from);
    let lib = Arc::new(brp_ocio::Library::load(&crate::color::config_dir(), custom.as_deref()));
    let mut ch = crate::color::Ocio::new(&settings).choices();
    if let Some(e) = get("ocio-input").and_then(|v| brp_ocio::Entry::from_id(&v)) {
        ch.input = Pick::Manual(e);
    }
    if let Some(e) = get("ocio-display").and_then(|v| brp_ocio::Entry::from_id(&v)) {
        ch.display = Pick::Manual(e);
    }
    if let Some(v) = get("ocio-view") {
        ch.view = Pick::Manual(v);
    }
    if let Some(t) = get("ocio-output").and_then(|v| crate::color::output_from_id(&v)) {
        ch.output = Pick::Manual(t);
    }
    Some(OcioJob { lib, choices: ch, settings })
}
