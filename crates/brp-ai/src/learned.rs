//! Our learned video denoiser (tools/denoise_train, `StableDenoiser`): each picture
//! is merged with the one before, the one after and its own previous output, each
//! moved onto it by a learned motion stage, then cleaned by a small U-Net.
//!
//! Three networks in ncnn's format, written by `tools/denoise_train/export_player.py`
//! into `models/brpdn/`, run by our runner; everything between them is the kernels
//! here, the same arithmetic as `model.py`:
//!
//! 1. The picture in PQ (the model's working space), its sides padded to 16.
//! 2. Motion to each neighbour: a pyramid of luma down to `motion_min` px, the `flow`
//!    network refining the motion `motion_iters` times per level (same weights at
//!    every level), finest level at 1/`motion_base` of the size.
//! 3. The neighbours and the previous output moved onto the picture (bilinear).
//! 4. Cut test: a neighbour whose moved picture differs by more than `cut_thr` on
//!    average (16 px block means) is another shot: the picture itself takes its
//!    place, and the previous output is dropped when it is the one before.
//! 5. Mismatch maps (how far each moved picture is from this one, over 4 and 16 px
//!    blocks) and the 22 input channels.
//! 6. `fuse`: per pixel, weights for the four pictures; their mix is the base.
//! 7. `unet`: base + correction = the denoised picture. It is also the state handed
//!    to the next picture. A model with a memory (`state` N in config.txt) also
//!    writes N channels at half size, kept within -1..1, which the next picture's
//!    `fuse` and `unet` read moved along the motion like the previous output; the state is
//!    then the picture's 3 planes followed by the memory.
//!
//! Blind: there is nothing to set but the output mix (`strength`). Checked against
//! PyTorch by `examples/learnedcheck.rs`.

use crate::gpu::Ops;
use crate::gpu_temporal::COMMON;
use crate::vk::{Net, PushTensor, Tensor};
use anyhow::{ensure, Context, Result};
use brp_render::compute::{Kernel, Push, Recorder, PRELUDE};
use std::path::Path;
use std::sync::Arc;

/// PQ as the training code has it (linear 1.0 = 203 nits; clipped to 0..10000 nits),
/// and sRGB: SDR video was linearised with the sRGB curve in training.
const COLOUR: &str = r#"
const float M1 = 0.1593017578125, M2 = 78.84375, C1 = 0.8359375, C2 = 18.8515625, C3 = 18.6875, WHITE = 0.0203;
float to_pq(float l) { float y = pow(clamp(l * WHITE, 0.0, 1.0), M1); return pow((C1 + C2 * y) / (1.0 + C3 * y), M2); }
float from_pq(float v) { float q = pow(clamp(v, 0.0, 1.0), 1.0 / M2); return pow(max(q - C1, 0.0) / (C2 - C3 * q), 1.0 / M1) / WHITE; }
float srgb_eotf(float v) { v = max(v, 0.0); return v <= 0.04045 ? v / 12.92 : pow((v + 0.055) / 1.055, 2.4); }
float srgb_oetf(float x) { x = max(x, 0.0); return x <= 0.0031308 ? 12.92 * x : 1.055 * pow(x, 1.0 / 2.4) - 0.055; }
"#;

/// The picture (linear light, 3+ channels) in PQ, edges repeated out to PW x PH.
/// `sdr`: the linear values are code values ^ 2.4 (the player's SDR decoding); the
/// model saw SDR through the sRGB curve.
const PREP: &str = r#"
G3
layout(push_constant) uniform P { F32 a; F32 o; int W, H, PW, PH, sdr; } p;
void main() { ID3 if (x >= p.PW || y >= p.PH || c >= 3) return;
    float v = p.a.v[(c * p.H + min(y, p.H - 1)) * p.W + min(x, p.W - 1)];
    if (p.sdr != 0) v = srgb_eotf(pow(max(v, 0.0), 1.0 / 2.4));
    p.o.v[(c * p.PH + y) * p.PW + x] = to_pq(v);
}
"#;

/// Means of k x k blocks (whole blocks only) of the luma (C = 3) or of one plane.
const POOL: &str = r#"
G3
layout(push_constant) uniform P { F32 a; F32 o; int C, W, H, k; } p;
void main() { ID3 int OW = p.W / p.k, OH = p.H / p.k; if (x >= OW || y >= OH || c > 0) return;
    int n = p.W * p.H;
    float s = 0.0;
    for (int j = 0; j < p.k; j++) for (int i = 0; i < p.k; i++) {
        int q = (y * p.k + j) * p.W + x * p.k + i;
        s += p.C == 3 ? 0.299 * p.a.v[q] + 0.587 * p.a.v[n + q] + 0.114 * p.a.v[2 * n + q] : p.a.v[q];
    }
    p.o.v[y * OW + x] = s / float(p.k * p.k);
}
"#;

/// The flow network's input at a level: this luma, the neighbour's moved by the flow
/// so far, the flow (`zero`: no flow yet).
const FLOW_IN: &str = r#"
G3
layout(push_constant) uniform P { F32 la; F32 lb; F32 f; F32 o; int W, H, zero; } p;
void main() { ID3 if (x >= p.W || y >= p.H || c > 0) return;
    int n = p.W * p.H, i = y * p.W + x;
    vec2 f = p.zero != 0 ? vec2(0.0) : vec2(p.f.v[i], p.f.v[n + i]);
    p.o.v[i] = p.la.v[i];
    p.o.v[n + i] = bilc(p.lb, 0, p.W, p.H, float(x) + f.x, float(y) + f.y);
    p.o.v[2 * n + i] = f.x;
    p.o.v[3 * n + i] = f.y;
}
"#;

const ADD: &str = r#"
G1
layout(push_constant) uniform P { F32 f; F32 d; int n; } p;
void main() { ID1 if (i >= p.n) return; p.f.v[i] += p.d.v[i]; }
"#;

/// Bilinear resize (pixel centres aligned as PyTorch's `align_corners=False`), x `mul`.
const RESIZE: &str = r#"
G3
layout(push_constant) uniform P { F32 a; F32 o; int C, W, H, OW, OH; float mul; } p;
void main() { ID3 if (x >= p.OW || y >= p.OH || c >= p.C) return;
    float sx = (float(x) + 0.5) * float(p.W) / float(p.OW) - 0.5, sy = (float(y) + 0.5) * float(p.H) / float(p.OH) - 0.5;
    p.o.v[(c * p.OH + y) * p.OW + x] = p.mul * bilc(p.a, c * p.W * p.H, p.W, p.H, sx, sy);
}
"#;

/// `img` (3 channels) sampled at p + flow(p), edges repeated; the flow is at 1/k of
/// the size, in its own pixels.
const WARP: &str = r#"
G3
layout(push_constant) uniform P { F32 img; F32 f; F32 o; int W, H, FW, FH; float k; } p;
void main() { ID3 if (x >= p.W || y >= p.H || c > 0) return;
    float sx = (float(x) + 0.5) / p.k - 0.5, sy = (float(y) + 0.5) / p.k - 0.5;
    float px = float(x) + p.k * bilc(p.f, 0, p.FW, p.FH, sx, sy), py = float(y) + p.k * bilc(p.f, p.FW * p.FH, p.FW, p.FH, sx, sy);
    int n = p.W * p.H;
    for (int ch = 0; ch < 3; ch++) p.o.v[ch * n + y * p.W + x] = bilc(p.img, ch * n, p.W, p.H, px, py);
}
"#;

/// The memory (C planes at half size, after the 3 picture planes of the state) moved
/// onto this picture: sampled at p + flow / 2, the flow (at 1/k of the picture, in
/// its own pixels) taken as the mean over the 2 x 2 pixels a memory pixel covers, as
/// model.py does. Zeros without a previous picture (`zero`, or flag 2 off: a cut).
const MEM: &str = r#"
G3
layout(push_constant) uniform P { F32 st; F32 f; F32 fl; F32 o; int C, W, H, FW, FH, zero; float k; } p;
void main() { ID3 int w = p.W / 2, h = p.H / 2; if (x >= w || y >= h || c > 0) return;
    int n = w * h, i = y * w + x;
    if (p.zero != 0 || p.fl.v[2] < 0.5) { for (int ch = 0; ch < p.C; ch++) p.o.v[ch * n + i] = 0.0; return; }
    vec2 f = vec2(0.0);
    for (int j = 0; j < 2; j++) for (int q = 0; q < 2; q++) {
        float sx = (float(2 * x + q) + 0.5) / p.k - 0.5, sy = (float(2 * y + j) + 0.5) / p.k - 0.5;
        f += vec2(bilc(p.f, 0, p.FW, p.FH, sx, sy), bilc(p.f, p.FW * p.FH, p.FW, p.FH, sx, sy));
    }
    f *= p.k * 0.125;
    int base = 3 * p.W * p.H;
    for (int ch = 0; ch < p.C; ch++) p.o.v[ch * n + i] = bilc(p.st, base + ch * n, w, h, float(x) + f.x, float(y) + f.y);
}
"#;

/// The state for the next picture: the denoised picture's planes (n values), then
/// the memory (m values) kept within -1..1.
const PACK: &str = r#"
G1
layout(push_constant) uniform P { F32 den; F32 mem; F32 o; int n, m; } p;
void main() { ID1 if (i >= p.n + p.m) return; p.o.v[i] = i < p.n ? p.den.v[i] : clamp(p.mem.v[i - p.n], -1.0, 1.0); }
"#;

/// |mean of a - mean of b| over k x k blocks, averaged over the 3 channels.
/// `same` >= 0: 0 when that flag is set (b is replaced by a). `valid` >= 0: b counts
/// as zero when that flag is off (no previous output).
const BLOCKS: &str = r#"
G3
layout(push_constant) uniform P { F32 a; F32 b; F32 fl; F32 o; int W, H, k, same, valid; } p;
void main() { ID3 int OW = p.W / p.k, OH = p.H / p.k; if (x >= OW || y >= OH || c > 0) return;
    if (p.same >= 0 && p.fl.v[p.same] > 0.5) { p.o.v[y * OW + x] = 0.0; return; }
    bool zero = p.valid >= 0 && p.fl.v[p.valid] < 0.5;
    int n = p.W * p.H;
    vec3 s = vec3(0.0);
    for (int j = 0; j < p.k; j++) for (int i = 0; i < p.k; i++) {
        int q = (y * p.k + j) * p.W + x * p.k + i;
        vec3 d = vec3(p.a.v[q], p.a.v[n + q], p.a.v[2 * n + q]);
        if (!zero) d -= vec3(p.b.v[q], p.b.v[n + q], p.b.v[2 * n + q]);
        s += d;
    }
    s = abs(s) / float(p.k * p.k);
    p.o.v[y * OW + x] = (s.x + s.y + s.z) / 3.0;
}
"#;

/// The flags: [before usable, after usable, previous output usable, 1 (strength),
/// cut before, cut after], from what the caller knows and the cut test (mean of the
/// 16 px block differences above `thr`).
const FLAGS: &str = r#"
layout(local_size_x = 1) in;
layout(push_constant) uniform P { F32 db; F32 da; F32 o; int n; float thr, vb, va, pv; } p;
void main() {
    float sb = 0.0, sa = 0.0;
    for (int i = 0; i < p.n; i++) { sb += p.db.v[i]; sa += p.da.v[i]; }
    float cb = (p.vb > 0.5 && sb / float(p.n) > p.thr) ? 1.0 : 0.0, ca = (p.va > 0.5 && sa / float(p.n) > p.thr) ? 1.0 : 0.0;
    p.o.v[0] = p.vb * (1.0 - cb);
    p.o.v[1] = p.va * (1.0 - ca);
    p.o.v[2] = p.pv * (1.0 - cb);
    p.o.v[3] = 1.0;
    p.o.v[4] = cb;
    p.o.v[5] = ca;
}
"#;

/// The 22 channels both networks read: picture, moved before / after / previous
/// output (a cut neighbour replaced by the picture, no previous output: zeros), the
/// 4 flags, the mismatch maps (block differences upsampled, x 8).
const ASSEMBLE: &str = r#"
G3
layout(push_constant) uniform P { F32 cur; F32 wb; F32 wa; F32 wp; F32 fl; F32 d4; F32 d16; F32 o; int W, H; } p;
void main() { ID3 if (x >= p.W || y >= p.H || c >= 22) return;
    int n = p.W * p.H, i = y * p.W + x;
    float v;
    if (c < 3) v = p.cur.v[c * n + i];
    else if (c < 6) v = p.fl.v[4] > 0.5 ? p.cur.v[(c - 3) * n + i] : p.wb.v[(c - 3) * n + i];
    else if (c < 9) v = p.fl.v[5] > 0.5 ? p.cur.v[(c - 6) * n + i] : p.wa.v[(c - 6) * n + i];
    else if (c < 12) v = p.fl.v[2] > 0.5 ? p.wp.v[(c - 9) * n + i] : 0.0;
    else if (c < 16) v = p.fl.v[c - 12];
    else {
        int j = c - 16, pic = j / 2, k = (j & 1) == 0 ? 4 : 16, w = p.W / k, h = p.H / k;
        float sx = (float(x) + 0.5) / float(k) - 0.5, sy = (float(y) + 0.5) / float(k) - 0.5;
        v = 8.0 * ((j & 1) == 0 ? bilc(p.d4, pic * w * h, w, h, sx, sy) : bilc(p.d16, pic * w * h, w, h, sx, sy));
    }
    p.o.v[c * n + i] = v;
}
"#;

/// The pre-merge: the four pictures mixed by the softmax of the network's weights
/// (slots without a picture get none).
const BLEND: &str = r#"
G3
layout(push_constant) uniform P { F32 xs; F32 lg; F32 o; int W, H; } p;
void main() { ID3 if (x >= p.W || y >= p.H || c > 0) return;
    int n = p.W * p.H, i = y * p.W + x;
    float l[4];
    float m = -1e30;
    for (int k = 0; k < 4; k++) {
        l[k] = p.lg.v[k * n + i];
        if (k > 0 && p.xs.v[(11 + k) * n + i] < 0.5) l[k] = -1e4;
        m = max(m, l[k]);
    }
    float s = 0.0;
    for (int k = 0; k < 4; k++) { l[k] = exp(l[k] - m); s += l[k]; }
    for (int ch = 0; ch < 3; ch++) {
        float v = 0.0;
        for (int k = 0; k < 4; k++) v += l[k] * p.xs.v[(3 * k + ch) * n + i];
        p.o.v[ch * n + i] = v / s;
    }
}
"#;

/// Back to linear light at the picture's size, mixed with the input by `s` (in PQ);
/// channels past 3 (alpha) as they came.
const FINISH: &str = r#"
G3
layout(push_constant) uniform P { F32 den; F32 cur; F32 src; F32 o; int C, W, H, PW, PH, sdr; float s; } p;
void main() { ID3 if (x >= p.W || y >= p.H || c >= p.C) return;
    int i = (c * p.H + y) * p.W + x;
    if (c >= 3) { p.o.v[i] = p.src.v[i]; return; }
    int q = (c * p.PH + y) * p.PW + x;
    float v = from_pq(mix(p.cur.v[q], p.den.v[q], p.s));
    if (p.sdr != 0) v = pow(srgb_oetf(v), 2.4);
    p.o.v[i] = v;
}
"#;

pub struct Learned {
    ops: Arc<Ops>,
    flow: Net,
    fuse: Net,
    unet: Net,
    /// Pyramid levels are added until the coarsest side is at most this.
    min_side: usize,
    iters: usize,
    /// The finest motion level is 1/base of the picture.
    base: usize,
    cut_thr: f32,
    /// Channels of the memory carried beside the previous output (0: none).
    mem: usize,
    /// The checkpoint the files were written from (for logs).
    pub source: String,
    prep: Arc<Kernel>,
    pool: Arc<Kernel>,
    flow_in: Arc<Kernel>,
    add: Arc<Kernel>,
    resize: Arc<Kernel>,
    warp: Arc<Kernel>,
    blocks: Arc<Kernel>,
    flags: Arc<Kernel>,
    assemble: Arc<Kernel>,
    blend: Arc<Kernel>,
    finish: Arc<Kernel>,
    mem_warp: Arc<Kernel>,
    pack: Arc<Kernel>,
}

fn g(k: &Kernel, w: usize, h: usize, c: usize) -> [u32; 3] {
    Recorder::groups(k, [w as u32, h as u32, c as u32])
}

impl Learned {
    pub fn available(models: &Path) -> bool {
        models.join("brpdn").join("unet.param").exists()
    }

    pub fn new(ops: &Arc<Ops>, models: &Path) -> Result<Self> {
        let dir = models.join("brpdn");
        let net = |name: &str, half: bool| -> Result<Net> {
            let base = dir.join(name);
            let param = std::fs::read_to_string(base.with_extension("param")).with_context(|| format!("{}.param", base.display()))?;
            let mut net = Net::load(&ops.compute, &param, &std::fs::read(base.with_extension("bin"))?)?;
            net.set_half(half);
            net.share_pool(&ops.pool);
            Ok(net)
        };
        let cfg = std::fs::read_to_string(dir.join("config.txt")).with_context(|| format!("{}", dir.join("config.txt").display()))?;
        let value = |key: &str| cfg.lines().find_map(|l| l.strip_prefix(key).map(str::trim).filter(|_| l[key.len()..].starts_with(' ')));
        let int = |key: &str| -> Result<usize> { value(key).and_then(|v| v.parse().ok()).with_context(|| format!("brpdn/config.txt: {key}")) };
        // BRP_DN_F32: the two picture networks in f32 too (checks; the motion stage always is).
        let half = std::env::var_os("BRP_DN_F32").is_none();
        let k = |name: &str, body: &str| ops.compute.kernel(&format!("learned_{name}"), &format!("{PRELUDE}{COMMON}{COLOUR}{body}"), 0);
        Ok(Learned {
            ops: ops.clone(),
            flow: net("flow", false)?,
            fuse: net("fuse", half)?,
            unet: net("unet", half)?,
            min_side: int("motion_min")?,
            iters: int("motion_iters")?,
            base: int("motion_base")?,
            cut_thr: value("cut_thr").and_then(|v| v.parse().ok()).context("brpdn/config.txt: cut_thr")?,
            mem: value("state").and_then(|v| v.parse().ok()).unwrap_or(0),
            source: value("source").unwrap_or("").to_string(),
            prep: k("prep", PREP)?,
            pool: k("pool", POOL)?,
            flow_in: k("flow_in", FLOW_IN)?,
            add: k("add", ADD)?,
            resize: k("resize", RESIZE)?,
            warp: k("warp", WARP)?,
            blocks: k("blocks", BLOCKS)?,
            flags: k("flags", FLAGS)?,
            assemble: k("assemble", ASSEMBLE)?,
            blend: k("blend", BLEND)?,
            finish: k("finish", FINISH)?,
            mem_warp: k("mem", MEM)?,
            pack: k("pack", PACK)?,
        })
    }

    /// `img` in the model's space: PQ, sides padded to a multiple of 16.
    fn encode(&self, rec: &mut Recorder, img: &Tensor, sdr: bool) -> Result<Tensor> {
        let (pw, ph) = (img.w.div_ceil(16) * 16, img.h.div_ceil(16) * 16);
        let out = self.ops.tensor(3, ph, pw)?;
        let push = Push::new().t(img).t(&out).i(img.w as i32).i(img.h as i32).i(pw as i32).i(ph as i32).i(sdr as i32);
        rec.dispatch(&self.prep, g(&self.prep, pw, ph, 3), &push)?;
        Ok(out)
    }

    fn pooled(&self, rec: &mut Recorder, a: &Tensor, k: usize) -> Result<Tensor> {
        let out = self.ops.tensor(1, a.h / k, a.w / k)?;
        rec.dispatch(&self.pool, g(&self.pool, out.w, out.h, 1), &Push::new().t(a).t(&out).i(a.c as i32).i(a.w as i32).i(a.h as i32).i(k as i32))?;
        Ok(out)
    }

    /// Where each pixel of `a` is in `b` (both PQ pictures), at 1/base of the size, in
    /// that level's pixels.
    fn motion(&self, rec: &mut Recorder, a: &Tensor, b: &Tensor) -> Result<Tensor> {
        let ops = &self.ops;
        let mut pa = vec![self.pooled(rec, a, self.base)?];
        let mut pb = vec![self.pooled(rec, b, self.base)?];
        while pa.last().is_some_and(|t| t.w.min(t.h) > self.min_side) {
            let (na, nb) = (self.pooled(rec, pa.last().unwrap(), 2)?, self.pooled(rec, pb.last().unwrap(), 2)?);
            pa.push(na);
            pb.push(nb);
        }
        let mut flow: Option<Tensor> = None;
        for (la, lb) in pa.iter().rev().zip(pb.iter().rev()) {
            let (w, h) = (la.w, la.h);
            if let Some(f) = flow.take_if(|f| (f.w, f.h) != (w, h)) {
                let up = ops.tensor(2, h, w)?;
                let push = Push::new().t(&f).t(&up).i(2).i(f.w as i32).i(f.h as i32).i(w as i32).i(h as i32).f(2.0);
                rec.dispatch(&self.resize, g(&self.resize, w, h, 2), &push)?;
                ops.recycle(rec, f);
                flow = Some(up);
            }
            for _ in 0..self.iters {
                let x = ops.tensor(4, h, w)?;
                let push = Push::new().t(la).t(lb).t(flow.as_ref().unwrap_or(la)).t(&x).i(w as i32).i(h as i32).i(flow.is_none() as i32);
                rec.dispatch(&self.flow_in, g(&self.flow_in, w, h, 1), &push)?;
                let d = self.flow.record(rec, &[("x", &x)], &["out"])?.remove(0);
                ensure!(!d.half && (d.c, d.h, d.w) == (2, h, w), "flow network output");
                ops.recycle(rec, x);
                match &flow {
                    None => flow = Some(d),
                    Some(f) => {
                        rec.dispatch(&self.add, Recorder::groups(&self.add, [f.len() as u32, 1, 1]), &Push::new().t(f).t(&d).i(f.len() as i32))?;
                        ops.recycle(rec, d);
                    }
                }
            }
        }
        for t in pa.into_iter().chain(pb) {
            ops.recycle(rec, t);
        }
        flow.context("no motion level")
    }

    fn warped(&self, rec: &mut Recorder, img: &Tensor, flow: &Tensor) -> Result<Tensor> {
        let out = self.ops.tensor(3, img.h, img.w)?;
        let push = Push::new().t(img).t(flow).t(&out).i(img.w as i32).i(img.h as i32).i(flow.w as i32).i(flow.h as i32).f(self.base as f32);
        rec.dispatch(&self.warp, g(&self.warp, img.w, img.h, 1), &push)?;
        Ok(out)
    }

    #[allow(clippy::too_many_arguments)]
    fn block_diff(&self, rec: &mut Recorder, a: &Tensor, b: &Tensor, fl: &Tensor, out: &Tensor, k: usize, same: i32, valid: i32) -> Result<()> {
        let push = Push::new().t(a).t(b).t(fl).t(out).i(a.w as i32).i(a.h as i32).i(k as i32).i(same).i(valid);
        rec.dispatch(&self.blocks, g(&self.blocks, a.w / k, a.h / k, 1), &push)
    }

    /// `cur` denoised: (the picture, in `cur`'s space and size, mixed with it by
    /// `strength`; the state to hand to the next picture as `prev`).
    ///
    /// Pictures are linear light in the source's primaries, 1.0 = SDR white, 3+
    /// channels (`sdr`: code values ^ 2.4, the player's SDR decoding). `before` /
    /// `after`: the pictures one step away, None where there is none (the clip's
    /// ends) or to use the picture alone. `prev`: the state from the picture before
    /// (None at a start: then give no `before` either on that picture, as in
    /// training, where a clip's first frame has neither). `cuts`: the cut test.
    #[allow(clippy::too_many_arguments)]
    pub fn record(&self, rec: &mut Recorder, cur: &Tensor, before: Option<&Tensor>, after: Option<&Tensor>, prev: Option<&Tensor>, sdr: bool, strength: f32, cuts: bool) -> Result<(Tensor, Tensor)> {
        let ops = &self.ops;
        ensure!(cur.c >= 3 && !cur.half, "denoise: {} channels", cur.c);
        rec.label = "denoise (learned): motion".into();
        let x = self.encode(rec, cur, sdr)?;
        let (pw, ph) = (x.w, x.h);
        ensure!(self.mem % 4 == 0, "denoise: memory of {} channels", self.mem);
        let prev = prev.filter(|p| (p.c, p.h, p.w) == (3 + self.mem / 4, ph, pw));
        // A missing neighbour is the picture itself, flagged off (as in training).
        let xb = before.map(|b| self.encode(rec, b, sdr)).transpose()?;
        let xa = after.map(|a| self.encode(rec, a, sdr)).transpose()?;
        let fb = self.motion(rec, &x, xb.as_ref().unwrap_or(&x))?;
        let fa = self.motion(rec, &x, xa.as_ref().unwrap_or(&x))?;
        let wb = self.warped(rec, xb.as_ref().unwrap_or(&x), &fb)?;
        let wa = self.warped(rec, xa.as_ref().unwrap_or(&x), &fa)?;
        // The previous output lines up with the picture before: the same motion.
        let wp = prev.map(|p| self.warped(rec, p, &fb)).transpose()?;
        for t in [fa].into_iter().chain(xb).chain(xa) {
            ops.recycle(rec, t);
        }

        rec.label = "denoise (learned): inputs".into();
        let fl = ops.tensor(1, 1, 8)?;
        let (bw, bh) = (pw / 16, ph / 16);
        let test = ops.tensor(2, bh, bw)?;
        self.block_diff(rec, &x, &wb, &fl, &test.view(0, 1), 16, -1, -1)?;
        self.block_diff(rec, &x, &wa, &fl, &test.view(1, 1), 16, -1, -1)?;
        let thr = if cuts { self.cut_thr } else { f32::MAX };
        let on = |v: bool| if v { 1.0f32 } else { 0.0 };
        let push = Push::new().t(&test.view(0, 1)).t(&test.view(1, 1)).t(&fl).i((bw * bh) as i32).f(thr).f(on(before.is_some())).f(on(after.is_some())).f(on(prev.is_some()));
        rec.dispatch(&self.flags, [1, 1, 1], &push)?;
        ops.recycle(rec, test);
        // The memory moved like the previous output (none: zeros).
        let h = if self.mem > 0 {
            let h = ops.tensor(self.mem, ph / 2, pw / 2)?;
            let push = Push::new().t(prev.unwrap_or(&x)).t(&fb).t(&fl).t(&h).i(self.mem as i32).i(pw as i32).i(ph as i32).i(fb.w as i32).i(fb.h as i32).i(prev.is_none() as i32).f(self.base as f32);
            rec.dispatch(&self.mem_warp, g(&self.mem_warp, pw / 2, ph / 2, 1), &push)?;
            Some(h)
        } else {
            None
        };
        ops.recycle(rec, fb);
        let (d4, d16) = (ops.tensor(3, ph / 4, pw / 4)?, ops.tensor(3, bh, bw)?);
        for (i, (w, same, valid)) in [(&wb, 4, -1), (&wa, 5, -1), (wp.as_ref().unwrap_or(&x), -1, 2)].into_iter().enumerate() {
            self.block_diff(rec, &x, w, &fl, &d4.view(i, 1), 4, same, valid)?;
            self.block_diff(rec, &x, w, &fl, &d16.view(i, 1), 16, same, valid)?;
        }
        let xs = ops.tensor(22, ph, pw)?;
        let push = Push::new().t(&x).t(&wb).t(&wa).t(wp.as_ref().unwrap_or(&x)).t(&fl).t(&d4).t(&d16).t(&xs).i(pw as i32).i(ph as i32);
        rec.dispatch(&self.assemble, g(&self.assemble, pw, ph, 22), &push)?;
        for t in [wb, wa, fl, d4, d16].into_iter().chain(wp) {
            ops.recycle(rec, t);
        }

        rec.label = "denoise (learned): pre-merge".into();
        let lg = match &h {
            Some(h) => self.fuse.record(rec, &[("x", &xs), ("h", h)], &["out"])?,
            None => self.fuse.record(rec, &[("x", &xs)], &["out"])?,
        }
        .remove(0);
        ensure!(!lg.half && (lg.c, lg.h, lg.w) == (4, ph, pw), "pre-merge network output");
        let base = ops.tensor(3, ph, pw)?;
        rec.dispatch(&self.blend, g(&self.blend, pw, ph, 1), &Push::new().t(&xs).t(&lg).t(&base).i(pw as i32).i(ph as i32))?;
        ops.recycle(rec, lg);

        rec.label = "denoise (learned): merge".into();
        let mut made = match &h {
            Some(h) => self.unet.record(rec, &[("x", &xs), ("base", &base), ("h", h)], &["out", "state"])?,
            None => self.unet.record(rec, &[("x", &xs), ("base", &base)], &["out"])?,
        };
        let den = made.remove(0);
        ensure!(!den.half && (den.c, den.h, den.w) == (3, ph, pw), "merge network output");
        ops.recycle(rec, xs);
        ops.recycle(rec, base);
        // The state: the picture, and behind it the new memory.
        let state = match made.pop() {
            Some(m) => {
                ensure!(!m.half && (m.c, m.h, m.w) == (self.mem, ph / 2, pw / 2), "merge network memory");
                let st = ops.tensor(3 + self.mem / 4, ph, pw)?;
                rec.dispatch(&self.pack, Recorder::groups(&self.pack, [st.len() as u32, 1, 1]), &Push::new().t(&den).t(&m).t(&st).i(den.len() as i32).i(m.len() as i32))?;
                ops.recycle(rec, m);
                Some(st)
            }
            None => None,
        };
        if let Some(h) = h {
            ops.recycle(rec, h);
        }

        let out = ops.tensor(cur.c, cur.h, cur.w)?;
        let push = Push::new().t(&den).t(&x).t(cur).t(&out).i(cur.c as i32).i(cur.w as i32).i(cur.h as i32).i(pw as i32).i(ph as i32).i(sdr as i32).f(strength.clamp(0.0, 1.0));
        rec.dispatch(&self.finish, g(&self.finish, cur.w, cur.h, cur.c), &push)?;
        ops.recycle(rec, x);
        rec.label = "-".into();
        Ok(match state {
            Some(st) => {
                ops.recycle(rec, den);
                (out, st)
            }
            None => (out, den),
        })
    }
}
