//! The AI models on the renderer's GPU (through our Vulkan runner, `vk`), and the
//! small picture operations around them. Everything takes and returns GPU tensors
//! (planar f32 `[c][h][w]`, colour 0..1) and records into a `Recorder`: the caller
//! submits once, nothing goes through the CPU.

use crate::matting;
use crate::rife::RifeModel;
use crate::upscale::UpscaleModel;
use crate::vk::{Net, Pool, PushTensor, Tensor};
use anyhow::{bail, Context, Result};
use brp_render::compute::{Compute, Kernel, Push, Recorder, PRELUDE};
use std::path::Path;
use std::sync::Arc;

/// Copy a region, with the source outside its picture either edge-replicated or
/// zero: out[c][y][x] = in[ic0 + c][iy0 + y][ix0 + x], optionally clamped to 0..1,
/// written at (oc0, oy0, ox0) of the output.
const CROP: &str = r#"
layout(local_size_x = 16, local_size_y = 8, local_size_z = 1) in;
layout(push_constant) uniform P {
    F32 a; F32 o;
    int C, H, W, iH, iW, ic0, iy0, ix0, oH, oW, oc0, oy0, ox0, replicate, clamp01;
} p;
void main() {
    int x = int(gl_GlobalInvocationID.x), y = int(gl_GlobalInvocationID.y), c = int(gl_GlobalInvocationID.z);
    if (x >= p.W || y >= p.H || c >= p.C) return;
    int sy = p.iy0 + y, sx = p.ix0 + x;
    float v;
    if (sy < 0 || sy >= p.iH || sx < 0 || sx >= p.iW) {
        v = p.replicate != 0 ? p.a.v[((p.ic0 + c) * p.iH + clamp(sy, 0, p.iH - 1)) * p.iW + clamp(sx, 0, p.iW - 1)] : 0.0;
    } else {
        v = p.a.v[((p.ic0 + c) * p.iH + sy) * p.iW + sx];
    }
    if (p.clamp01 != 0) v = clamp(v, 0.0, 1.0);
    p.o.v[((p.oc0 + c) * p.oH + p.oy0 + y) * p.oW + p.ox0 + x] = v;
}
"#;

const FILL: &str = r#"
layout(local_size_x = 256) in;
layout(push_constant) uniform P { F32 o; int n; float v; } p;
void main() {
    int i = int(gl_GlobalInvocationID.x + gl_GlobalInvocationID.y * gl_NumWorkGroups.x * gl_WorkGroupSize.x);
    if (i < p.n) p.o.v[i] = p.v;
}
"#;

/// pixel_unshuffle(2): out channel c*4 + dy*2 + dx at (x, y) = in channel c at (2x+dx, 2y+dy).
const FOLD2: &str = r#"
layout(local_size_x = 16, local_size_y = 8, local_size_z = 1) in;
layout(push_constant) uniform P { F32 a; F32 o; int C, OH, OW; } p;
void main() {
    int x = int(gl_GlobalInvocationID.x), y = int(gl_GlobalInvocationID.y), oc = int(gl_GlobalInvocationID.z);
    if (x >= p.OW || y >= p.OH || oc >= p.C * 4) return;
    int c = oc / 4, dy = (oc / 2) % 2, dx = oc % 2;
    int W = p.OW * 2, H = p.OH * 2;
    p.o.v[(oc * p.OH + y) * p.OW + x] = p.a.v[(c * H + 2 * y + dy) * W + 2 * x + dx];
}
"#;

/// Resize planes: bilinear (pixel centres aligned) or, when shrinking by an integer
/// factor k > 1, the k x k average.
const RESIZE: &str = r#"
layout(local_size_x = 16, local_size_y = 8, local_size_z = 1) in;
layout(push_constant) uniform P { F32 a; F32 o; int ic0, oc0, C, H, W, OH, OW, box_k; } p;
void main() {
    int x = int(gl_GlobalInvocationID.x), y = int(gl_GlobalInvocationID.y), c = int(gl_GlobalInvocationID.z);
    if (x >= p.OW || y >= p.OH || c >= p.C) return;
    int base = (p.ic0 + c) * p.H * p.W;
    float r = 0.0;
    if (p.box_k > 1) {
        for (int j = 0; j < p.box_k; j++)
            for (int i = 0; i < p.box_k; i++)
                r += p.a.v[base + (y * p.box_k + j) * p.W + x * p.box_k + i];
        r /= float(p.box_k * p.box_k);
    } else {
        float fx = clamp((float(x) + 0.5) * float(p.W) / float(p.OW) - 0.5, 0.0, float(p.W - 1));
        float fy = clamp((float(y) + 0.5) * float(p.H) / float(p.OH) - 0.5, 0.0, float(p.H - 1));
        int x0 = int(fx), y0 = int(fy);
        int x1 = min(x0 + 1, p.W - 1), y1 = min(y0 + 1, p.H - 1);
        float ax = fx - float(x0), ay = fy - float(y0);
        r = mix(mix(p.a.v[base + y0 * p.W + x0], p.a.v[base + y0 * p.W + x1], ax), mix(p.a.v[base + y1 * p.W + x0], p.a.v[base + y1 * p.W + x1], ax), ay);
    }
    p.o.v[((p.oc0 + c) * p.OH + y) * p.OW + x] = r;
}
"#;

/// out = a + t * (b - a) (alpha cross-fade between two pictures).
const MIX: &str = r#"
layout(local_size_x = 256) in;
layout(push_constant) uniform P { F32 a; F32 b; F32 o; int n; float t; } p;
void main() {
    int i = int(gl_GlobalInvocationID.x + gl_GlobalInvocationID.y * gl_NumWorkGroups.x * gl_WorkGroupSize.x);
    if (i < p.n) p.o.v[i] = mix(p.a.v[i], p.b.v[i], p.t);
}
"#;

/// Small picture operations on tensors.
pub struct Ops {
    pub compute: Arc<Compute>,
    pub pool: Arc<Pool>,
    crop: Arc<Kernel>,
    fill: Arc<Kernel>,
    fold2: Arc<Kernel>,
    resize: Arc<Kernel>,
    mix: Arc<Kernel>,
}

fn flat(k: &Kernel, n: usize) -> [u32; 3] {
    let groups = n.div_ceil(k.local[0] as usize).max(1);
    let x = groups.min(65535);
    [x as u32, groups.div_ceil(x) as u32, 1]
}

impl Ops {
    pub fn new(compute: &Arc<Compute>) -> Result<Arc<Ops>> {
        let k = |name: &str, body: &str| compute.kernel(&format!("ops_{name}"), &format!("{PRELUDE}{body}"), 0);
        Ok(Arc::new(Ops {
            compute: compute.clone(),
            pool: Default::default(),
            crop: k("crop", CROP)?,
            fill: k("fill", FILL)?,
            fold2: k("fold2", FOLD2)?,
            resize: k("resize", RESIZE)?,
            mix: k("mix", MIX)?,
        }))
    }

    pub fn tensor(&self, c: usize, h: usize, w: usize) -> Result<Tensor> {
        self.pool.tensor(&self.compute, c, h, w)
    }

    /// A tensor filled with `v`.
    pub fn filled(&self, rec: &mut Recorder, c: usize, h: usize, w: usize, v: f32) -> Result<Tensor> {
        let t = self.tensor(c, h, w)?;
        rec.dispatch(&self.fill, flat(&self.fill, t.len()), &Push::new().t(&t).i(t.len() as i32).f(v))?;
        Ok(t)
    }

    /// Region copy (see CROP). `from` = (channel, y, x) in `src` (may be outside it),
    /// `size` = (c, h, w), `to` = (channel, y, x) in `dst`.
    #[allow(clippy::too_many_arguments)]
    pub fn copy(&self, rec: &mut Recorder, src: &Tensor, from: (usize, isize, isize), size: (usize, usize, usize), dst: &Tensor, to: (usize, usize, usize), replicate: bool, clamp01: bool) -> Result<()> {
        let k = &self.crop;
        let push = Push::new()
            .t(&src)
            .t(&dst)
            .i(size.0 as i32)
            .i(size.1 as i32)
            .i(size.2 as i32)
            .i(src.h as i32)
            .i(src.w as i32)
            .i(from.0 as i32)
            .i(from.1 as i32)
            .i(from.2 as i32)
            .i(dst.h as i32)
            .i(dst.w as i32)
            .i(to.0 as i32)
            .i(to.1 as i32)
            .i(to.2 as i32)
            .i(replicate as i32)
            .i(clamp01 as i32);
        rec.dispatch(k, Recorder::groups(k, [size.2 as u32, size.1 as u32, size.0 as u32]), &push)
    }

    /// A new tensor: `c` channels of `src` from channel `c0`, the region at (x, y)
    /// of size w x h (outside: edge-replicated or zero).
    #[allow(clippy::too_many_arguments)]
    pub fn crop(&self, rec: &mut Recorder, src: &Tensor, c0: usize, c: usize, x: isize, y: isize, w: usize, h: usize, replicate: bool) -> Result<Tensor> {
        let out = self.tensor(c, h, w)?;
        self.copy(rec, src, (c0, y, x), (c, h, w), &out, (0, 0, 0), replicate, false)?;
        Ok(out)
    }

    pub fn fold2(&self, rec: &mut Recorder, src: &Tensor) -> Result<Tensor> {
        let (oh, ow) = (src.h / 2, src.w / 2);
        let out = self.tensor(src.c * 4, oh, ow)?;
        let k = &self.fold2;
        rec.dispatch(k, Recorder::groups(k, [ow as u32, oh as u32, (src.c * 4) as u32]), &Push::new().t(&src).t(&out).i(src.c as i32).i(oh as i32).i(ow as i32))?;
        Ok(out)
    }

    /// Channels `c0..c0+n` of `src` resized into `dst` (at channel `dc0`): bilinear,
    /// or the box average when `dst` is `src` / k exactly.
    pub fn resize_into(&self, rec: &mut Recorder, src: &Tensor, c0: usize, n: usize, dst: &Tensor, dc0: usize) -> Result<()> {
        let box_k = if dst.w > 0 && src.w % dst.w == 0 && src.h % dst.h.max(1) == 0 && src.w / dst.w == src.h / dst.h.max(1) { src.w / dst.w } else { 1 };
        let k = &self.resize;
        let push = Push::new().t(&src).t(&dst).i(c0 as i32).i(dc0 as i32).i(n as i32).i(src.h as i32).i(src.w as i32).i(dst.h as i32).i(dst.w as i32).i(box_k as i32);
        rec.dispatch(k, Recorder::groups(k, [dst.w as u32, dst.h as u32, n as u32]), &push)
    }

    pub fn resize(&self, rec: &mut Recorder, src: &Tensor, w: usize, h: usize) -> Result<Tensor> {
        let out = self.tensor(src.c, h, w)?;
        self.resize_into(rec, src, 0, src.c, &out, 0)?;
        Ok(out)
    }

    /// a + t (b - a), same shapes.
    pub fn mix(&self, rec: &mut Recorder, a: &Tensor, b: &Tensor, t: f32) -> Result<Tensor> {
        let out = self.tensor(a.c, a.h, a.w)?;
        rec.dispatch(&self.mix, flat(&self.mix, a.len()), &Push::new().t(&a).t(&b).t(&out).i(a.len() as i32).f(t))?;
        Ok(out)
    }

    /// Give a tensor's buffer back for reuse by later work of this recording.
    pub fn recycle(&self, rec: &mut Recorder, t: Tensor) {
        if Arc::strong_count(&t.buf) == 1 {
            self.pool.put(t.buf);
        } else {
            rec.keep(t.buf);
        }
    }
}

fn load_net(compute: &Arc<Compute>, param: &Path, bin: &Path) -> Result<Net> {
    let text = std::fs::read_to_string(param).with_context(|| format!("reading {}", param.display()))?;
    let data = std::fs::read(bin).with_context(|| format!("reading {}", bin.display()))?;
    Net::load(compute, &text, &data)
}

// ---- upscaling -----------------------------------------------------------------------

pub struct Upscaler {
    ops: Arc<Ops>,
    net: Net,
    model: UpscaleModel,
    denoise: f32,
    /// Tile size chosen for a picture size: (w, h, tile).
    tile: std::sync::Mutex<Option<(usize, usize, usize)>>,
}

/// Context around each tile (input pixels). The large networks see far (300+
/// layers of 3x3): with 10 px a tiled picture differed from the whole one by 32.5
/// dB along edges, with 32 px 48 dB (5-13% more work). The compact ones are fine
/// with 10 (45 dB).
fn pad(model: UpscaleModel) -> usize {
    match model {
        UpscaleModel::General4x | UpscaleModel::General2x | UpscaleModel::Anime4x | UpscaleModel::Swin4x | UpscaleModel::Dat4x | UpscaleModel::DatReal4x => 32,
        _ => 10,
    }
}

impl Upscaler {
    pub fn new(ops: &Arc<Ops>, models: &Path, model: UpscaleModel) -> Result<Self> {
        Self::with_denoise(ops, models, model, 1.0)
    }

    /// `denoise` 0..1 (GeneralV3 only): the weights are mixed with those of the model
    /// trained without denoising, as Real-ESRGAN's `-dn` does: lower keeps more grain
    /// and fine texture, higher is cleaner but can look smoothed.
    pub fn with_denoise(ops: &Arc<Ops>, models: &Path, model: UpscaleModel, denoise: f32) -> Result<Self> {
        let dir = models.join(model.dir());
        let stem = model.file_stem();
        let param = dir.join(format!("{stem}.param"));
        let mut net = if model == UpscaleModel::GeneralV3 && denoise < 0.999 {
            let text = std::fs::read_to_string(&param).with_context(|| format!("reading {}", param.display()))?;
            let a = std::fs::read(dir.join(format!("{stem}.bin")))?;
            let b = std::fs::read(dir.join("realesr-general-wdn-x4v3.bin")).context("realesr-general-wdn-x4v3.bin (no-denoise weights)")?;
            anyhow::ensure!(a.len() == b.len() && a.len() % 4 == 0, "denoise weights do not match");
            // Both files are raw float32 in the same order (tags are 0 in both).
            let t = denoise.clamp(0.0, 1.0);
            let mixed: Vec<u8> = a
                .chunks_exact(4)
                .zip(b.chunks_exact(4))
                .flat_map(|(x, y)| {
                    let (x, y) = (f32::from_le_bytes(x.try_into().unwrap()), f32::from_le_bytes(y.try_into().unwrap()));
                    (t * x + (1.0 - t) * y).to_le_bytes()
                })
                .collect();
            Net::load(&ops.compute, &text, &mixed).with_context(|| format!("upscaling model {stem}"))?
        } else {
            load_net(&ops.compute, &param, &dir.join(format!("{stem}.bin"))).with_context(|| format!("upscaling model {stem}"))?
        };
        // Activations in f16 (as ncnn ran these models): half the memory traffic.
        net.set_half(true);
        // The large networks gain from f16 sums; the compact ones lose a little.
        // (SwinIR's sums go past f16's range: never.)
        net.set_fp16_sums(matches!(model, UpscaleModel::General4x | UpscaleModel::General2x | UpscaleModel::Anime4x));
        Ok(Upscaler { ops: ops.clone(), net, model, denoise, tile: std::sync::Mutex::new(None) })
    }

    pub fn model(&self) -> UpscaleModel {
        self.model
    }

    pub fn denoise(&self) -> f32 {
        self.denoise
    }

    /// Tile size (input pixels): the largest that fits the memory budget, chosen once
    /// per picture size (so the buffers stay the same). Larger tiles are faster (less
    /// overlap, fuller GPU): best 2x 470 -> 370 ms from 384 to 768 px. Memory per
    /// tile pixel was measured (allocator report, equal tiles, f16 activations) and
    /// given margin.
    fn tile(&self, img: &Tensor) -> usize {
        if let Some(t) = std::env::var("BRP_UPSCALE_TILE").ok().and_then(|v| v.parse().ok()) {
            return t;
        }
        let mut chosen = self.tile.lock().unwrap();
        if let Some((w, h, t)) = *chosen {
            if (w, h) == (img.w, img.h) {
                return t;
            }
        }
        let pad = pad(self.model);
        let per_px: u64 = match self.model {
            UpscaleModel::General4x | UpscaleModel::Anime4x => 8000,
            UpscaleModel::General2x => 1600,
            // SwinIR: 8.6 KB measured (most at the output size: the 4x upsampling).
            UpscaleModel::Swin4x => 10000,
            UpscaleModel::Dat4x | UpscaleModel::DatReal4x => 10000,
            _ => 600,
        };
        // The output picture and the model's own buffers come off the budget first.
        let output = (img.c * img.w * img.h) as u64 * (self.model.scale() * self.model.scale()) as u64 * 4;
        let total = ai_memory_budget(&self.ops.compute, self.net.pool.held());
        let budget = total.saturating_sub(output + (256 << 20));
        let s = self.model.scale() as u64;
        let t = [1024usize, 896, 768, 640, 512, 384, 256]
            .into_iter()
            .find(|&t| {
                let side = (t + 2 * pad) as u64;
                // The largest tensor (64 channels at the output size) stays well under
                // 2^30 elements: 32-bit index arithmetic.
                per_px * side * side <= budget && 64 * side * side * s * s < 1 << 30
            })
            .unwrap_or(256);
        log::info!("upscale {:?}: tiles of {t} px for {}x{} (memory budget {} MB)", self.model, img.w, img.h, total >> 20);
        // SwinIR runs 1.2 us per network input pixel on an RTX 5070 Ti (0.4 s for a
        // 512 px tile), DAT about 1.9 us: their tiles are split into submissions of
        // about 0.1 s there, so cards several times slower stay under Windows' 2 s
        // limit too.
        if matches!(self.model, UpscaleModel::Swin4x | UpscaleModel::Dat4x | UpscaleModel::DatReal4x) {
            let side = (t + 2 * pad) as f64;
            let tile_s = side * side * if self.model != UpscaleModel::Swin4x { 1.9e-6 } else { 1.2e-6 };
            let n = (self.net.layer_count() as f64 * 0.1 / tile_s).max(8.0) as usize;
            self.net.set_flush_every(n);
        }
        *chosen = Some((img.w, img.h, t));
        t
    }

    /// `img` (3 or 4 channels) upscaled by the model's factor; alpha resized.
    pub fn record(&self, rec: &mut Recorder, img: &Tensor) -> Result<Tensor> {
        let pad = pad(self.model);
        let ops = &self.ops;
        let s = self.model.scale();
        let tile = self.tile(img);
        let out = ops.tensor(img.c, img.h * s, img.w * s)?;
        // Equal tiles (1920 = 3 x 640, not 768 + 768 + 384), and every network input
        // the same size (edge-replicated past the picture): one set of buffers serves
        // all tiles.
        let split = |n: usize| -> usize {
            let k = n.div_ceil(tile);
            n.div_ceil(k).next_multiple_of(self.model.align())
        };
        let (step_x, step_y) = (split(img.w), split(img.h));
        for ty in (0..img.h).step_by(step_y) {
            for tx in (0..img.w).step_by(step_x) {
                let (tw, th) = (step_x.min(img.w - tx), step_y.min(img.h - ty));
                let (mut iw, mut ih) = (step_x + 2 * pad, step_y + 2 * pad);
                if self.model.folded() {
                    (iw, ih) = ((iw + 1) & !1, (ih + 1) & !1);
                }
                let mut input = ops.crop(rec, img, 0, 3, tx as isize - pad as isize, ty as isize - pad as isize, iw, ih, true)?;
                if self.model.folded() {
                    let f = ops.fold2(rec, &input)?;
                    ops.recycle(rec, std::mem::replace(&mut input, f));
                }
                let res = self.net.record(rec, &[("data", &input)], &["output"])?.remove(0);
                ops.copy(rec, &res, (0, (pad * s) as isize, (pad * s) as isize), (3, th * s, tw * s), &out, (0, ty * s, tx * s), false, true)?;
                ops.recycle(rec, input);
                self.net.pool.put(res.buf);
                // Each tile in its own submission: none runs long (see `Recorder::flush`).
                if tx + step_x < img.w || ty + step_y < img.h {
                    rec.flush()?;
                }
            }
        }
        if img.c == 4 {
            ops.resize_into(rec, img, 3, 1, &out, 3)?;
        }
        Ok(out)
    }
}

/// GPU memory the upscaler may use (its buffers and the output picture): a third of
/// the GPU's own memory, at most 5 GB, and at most half of what is free right now
/// (other programs' use counted, as the driver reports it: VK_EXT_memory_budget)
/// plus what it already holds. The player's picture caches come on top. Running
/// short of GPU memory makes Windows page it, and a GPU stalled for 2 seconds is
/// reset ("device lost"). `BRP_AI_MEMORY_MB` sets it.
pub fn ai_memory_budget(compute: &Compute, held: u64) -> u64 {
    if let Some(mb) = std::env::var("BRP_AI_MEMORY_MB").ok().and_then(|v| v.parse::<u64>().ok()) {
        return mb << 20;
    }
    let cap = (compute.ctx.vram() / 3).min(5 << 30);
    match compute.ctx.vram_free() {
        Some(free) => cap.min((free + held) / 2),
        None => cap,
    }
}

// ---- frame interpolation ---------------------------------------------------------------

pub struct Rife {
    ops: Arc<Ops>,
    net: Net,
    model: RifeModel,
}

impl Rife {
    pub fn new(ops: &Arc<Ops>, models: &Path, model: RifeModel) -> Result<Self> {
        let dir = models.join("rife").join(model.dir_name());
        let net = load_net(&ops.compute, &dir.join("flownet.param"), &dir.join("flownet.bin")).with_context(|| format!("interpolation model {}", model.dir_name()))?;
        Ok(Rife { ops: ops.clone(), net, model })
    }

    pub fn model(&self) -> RifeModel {
        self.model
    }

    /// The picture at time `t` (0 = `a`, 1 = `b`) between two of the same size.
    pub fn record(&self, rec: &mut Recorder, a: &Tensor, b: &Tensor, t: f32) -> Result<Tensor> {
        if (a.w, a.h) != (b.w, b.h) {
            bail!("interpolation: pictures differ in size");
        }
        let ops = &self.ops;
        let pad = self.model.padding();
        let (wp, hp) = (a.w.div_ceil(pad) * pad, a.h.div_ceil(pad) * pad);
        let in0 = ops.crop(rec, a, 0, 3, 0, 0, wp, hp, false)?;
        let in1 = ops.crop(rec, b, 0, 3, 0, 0, wp, hp, false)?;
        let ts = ops.filled(rec, 1, hp, wp, t)?;
        let res = self.net.record(rec, &[("in0", &in0), ("in1", &in1), ("in2", &ts)], &["out0"])?.remove(0);
        let out = ops.tensor(a.c, a.h, a.w)?;
        ops.copy(rec, &res, (0, 0, 0), (3, a.h, a.w), &out, (0, 0, 0), false, true)?;
        if a.c == 4 {
            // Alpha: plain cross-fade.
            let fa = ops.crop(rec, a, 3, 1, 0, 0, a.w, a.h, false)?;
            let fb = ops.crop(rec, b, 3, 1, 0, 0, a.w, a.h, false)?;
            let m = ops.mix(rec, &fa, &fb, t)?;
            ops.copy(rec, &m, (0, 0, 0), (1, a.h, a.w), &out, (3, 0, 0), false, false)?;
            for x in [fa, fb, m] {
                ops.recycle(rec, x);
            }
        }
        for x in [in0, in1, ts] {
            ops.recycle(rec, x);
        }
        self.net.pool.put(res.buf);
        Ok(out)
    }
}

// ---- background removal -------------------------------------------------------------------

pub struct Matting {
    ops: Arc<Ops>,
    net: Net,
    size: (usize, usize),
    state: Option<[Tensor; 4]>,
}

impl Matting {
    pub fn new(ops: &Arc<Ops>, models: &Path, w: usize, h: usize) -> Result<Self> {
        let (param, bin) = matting::files(models);
        let text = std::fs::read_to_string(&param).with_context(|| format!("reading {}", param.display()))?;
        let graph = matting::graph_for(&text, w, h)?;
        let data = std::fs::read(&bin).with_context(|| format!("reading {}", bin.display()))?;
        let net = Net::load(&ops.compute, &graph, &data).context("background removal model")?;
        Ok(Matting { ops: ops.clone(), net, size: (w, h), state: None })
    }

    pub fn available(models: &Path) -> bool {
        let (p, b) = matting::files(models);
        p.exists() && b.exists()
    }

    pub fn size(&self) -> (usize, usize) {
        self.size
    }

    /// Forget the previous frames (after a jump).
    pub fn reset(&mut self) {
        self.state = None;
    }

    /// Matte the next frame of the sequence: foreground colour (edges
    /// decontaminated) and alpha, as 4 channels.
    pub fn record(&mut self, rec: &mut Recorder, img: &Tensor) -> Result<Tensor> {
        let (w, h) = self.size;
        if (img.w, img.h) != (w, h) {
            bail!("matting: picture is {}x{}, set up for {w}x{h}", img.w, img.h);
        }
        let ops = &self.ops;
        let (lw, lh) = matting::low_res(w, h);
        let state = match self.state.take() {
            Some(s) => s,
            None => [
                ops.filled(rec, 16, lh / 2, lw / 2, 0.0)?,
                ops.filled(rec, 20, lh / 4, lw / 4, 0.0)?,
                ops.filled(rec, 40, lh / 8, lw / 8, 0.0)?,
                ops.filled(rec, 64, lh / 16, lw / 16, 0.0)?,
            ],
        };
        let src = if img.c == 3 { img.clone() } else { ops.crop(rec, img, 0, 3, 0, 0, w, h, false)? };
        let mut outs = self
            .net
            .record(rec, &[("in0", &src), ("in1", &state[0]), ("in2", &state[1]), ("in3", &state[2]), ("in4", &state[3])], &["out0", "out1", "out2", "out3", "out4", "out5"])?;
        // The new state replaces the old once this recording has run.
        for s in state {
            rec.keep(s.buf);
        }
        let rest: Vec<Tensor> = outs.drain(2..).collect();
        self.state = Some(rest.try_into().map_err(|_| anyhow::anyhow!("matting outputs"))?);
        let out = ops.tensor(4, h, w)?;
        ops.copy(rec, &outs[0], (0, 0, 0), (3, h, w), &out, (0, 0, 0), false, false)?;
        ops.copy(rec, &outs[1], (0, 0, 0), (1, h, w), &out, (3, 0, 0), false, false)?;
        for t in outs {
            rec.keep(t.buf);
        }
        if img.c != 3 {
            ops.recycle(rec, src);
        }
        Ok(out)
    }
}
