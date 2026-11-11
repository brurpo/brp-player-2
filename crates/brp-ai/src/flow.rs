//! Learned optical flow on our runner, as an alternative to the Lucas-Kanade motion of
//! `gpu_temporal` (see DESIGN.md, "AI optical flow").
//!
//! SEA-RAFT (princeton-vl/SEA-RAFT, BSD-3-Clause; Wang, Lipson and Deng, ECCV 2024),
//! the "M" model fine-tuned on Spring, run as its authors run it on 1080p: at half size,
//! 4 refinement steps. Three graphs (written by `pth2ncnn searaft`): features of one
//! picture (cached per frame by the caller if wanted), the context of the pair, and one
//! refinement step. The rest is kernels here:
//! - the correlation lookup: per pixel and pyramid level, 9x9 bilinear samples of the
//!   correlation between its features and the other picture's around the current
//!   flow. The correlation is linear in the features, so it is computed on the fly from
//!   them (the dot products on the 10x10 integer grid the samples need, then blended)
//!   instead of SEA-RAFT's all-pairs volume, the pyramid levels from the other picture's
//!   features halved (bilinear, as F.interpolate at 0.5);
//! - the convex upsampling of the flow to the network's input size (x8), and the
//!   resize to the picture's (x2).

use crate::gpu::Ops;
use crate::vk::{Net, PushTensor, Tensor};
use anyhow::{Context, Result};
use brp_render::compute::{Kernel, Push, Recorder, PRELUDE};
use std::path::Path;
use std::sync::Arc;

/// Bilinear resize to half size (floor), as F.interpolate(scale_factor=0.5,
/// mode="bilinear", align_corners=False).
const HALF_SIZE: &str = r#"
layout(local_size_x = 16, local_size_y = 8, local_size_z = 1) in;
layout(push_constant) uniform P { F32 a; F32 o; int C, H, W, OH, OW; } p;
void main() {
    int x = int(gl_GlobalInvocationID.x), y = int(gl_GlobalInvocationID.y), c = int(gl_GlobalInvocationID.z);
    if (x >= p.OW || y >= p.OH || c >= p.C) return;
    float fx = max((float(x) + 0.5) * float(p.W) / float(p.OW) - 0.5, 0.0);
    float fy = max((float(y) + 0.5) * float(p.H) / float(p.OH) - 0.5, 0.0);
    int x0 = min(int(fx), p.W - 1), y0 = min(int(fy), p.H - 1);
    int x1 = min(x0 + 1, p.W - 1), y1 = min(y0 + 1, p.H - 1);
    float ax = fx - float(x0), ay = fy - float(y0);
    int b = c * p.H * p.W;
    float t = mix(p.a.v[b + y0 * p.W + x0], p.a.v[b + y0 * p.W + x1], ax);
    float u = mix(p.a.v[b + y1 * p.W + x0], p.a.v[b + y1 * p.W + x1], ax);
    p.o.v[(c * p.OH + y) * p.OW + x] = mix(t, u, ay);
}
"#;

/// The correlation lookup for one pyramid level: out[level 81 + a 9 + b] at each pixel
/// = the correlation (dot product / sqrt(C)) of its features f1 with the level's
/// features f2 sampled at ((x + flow_x) / 2^l + a - R, (y + flow_y) / 2^l + b - R),
/// bilinear, zero outside (grid_sample, align_corners=True). SEA-RAFT's lookup adds
/// the grid's row offset to x and its column offset to y; so does this.
const CORR: &str = r#"
#extension GL_EXT_shader_explicit_arithmetic_types_float16 : require
layout(buffer_reference, std430, buffer_reference_align = 8) buffer H4 { f16vec4 v[]; };
layout(local_size_x = 64) in;
layout(push_constant) uniform P { H4 f1; H4 f2; F32 flow; F32 o; int C, H, W, H2, W2, level, R; } p;
#define G 10
void main() {
    int i = int(gl_GlobalInvocationID.x);
    if (i >= p.H * p.W) return;
    int y = i / p.W, x = i - y * p.W, n = p.H * p.W, C4 = p.C / 4;
    float s = 1.0 / float(1 << p.level);
    float cx = (float(x) + p.flow.v[i]) * s, cy = (float(y) + p.flow.v[n + i]) * s;
    float bx = floor(cx), by = floor(cy);
    float dx = cx - bx, dy = cy - by;
    int x0 = int(bx) - p.R, y0 = int(by) - p.R;
    // Features pixel-major (f16, a pixel's channels side by side: FEAT_T), so each
    // position's dot product reads a few cache lines.
    float d[G * G];
    for (int v = 0; v < G; v++)
        for (int u = 0; u < G; u++) {
            int yy = y0 + v, xx = x0 + u;
            float acc = 0.0;
            if (yy >= 0 && yy < p.H2 && xx >= 0 && xx < p.W2) {
                int b1 = i * C4, b2 = (yy * p.W2 + xx) * C4;
                for (int c = 0; c < C4; c++) acc += dot(vec4(p.f1.v[b1 + c]), vec4(p.f2.v[b2 + c]));
            }
            d[v * G + u] = acc;
        }
    float k = inversesqrt(float(p.C));
    int S = 2 * p.R + 1;
    for (int a = 0; a < S; a++)
        for (int b = 0; b < S; b++) {
            // x offset a - R, y offset b - R: grid cell (u = a, v = b).
            float val = (1.0 - dy) * ((1.0 - dx) * d[b * G + a] + dx * d[b * G + a + 1])
                      + dy * ((1.0 - dx) * d[(b + 1) * G + a] + dx * d[(b + 1) * G + a + 1]);
            p.o.v[((p.level * S + a) * S + b) * n + i] = val * k;
        }
}
"#;

/// Features planar f32 [C][n] to pixel-major f16 [n][C] (for CORR).
const FEAT_T: &str = r#"
#extension GL_EXT_shader_explicit_arithmetic_types_float16 : require
layout(buffer_reference, scalar, buffer_reference_align = 2) buffer H1 { float16_t v[]; };
layout(local_size_x = 64, local_size_y = 4, local_size_z = 1) in;
layout(push_constant) uniform P { F32 a; H1 o; int C, n; } p;
void main() {
    int i = int(gl_GlobalInvocationID.x), c = int(gl_GlobalInvocationID.y);
    if (i >= p.n || c >= p.C) return;
    p.o.v[i * p.C + c] = float16_t(p.a.v[c * p.n + i]);
}
"#;

/// Convex upsampling by F (SEA-RAFT's upsample_data): each output pixel is a softmax
/// blend, by its 9 weights of `mask`, of F x the flow at the 3x3 low-resolution
/// neighbours (zero outside).
const CONVEX: &str = r#"
layout(local_size_x = 16, local_size_y = 8, local_size_z = 1) in;
layout(push_constant) uniform P { F32 flow; F32 mask; F32 o; int H, W, F; } p;
void main() {
    int ox = int(gl_GlobalInvocationID.x), oy = int(gl_GlobalInvocationID.y);
    int OW = p.W * p.F, OH = p.H * p.F;
    if (ox >= OW || oy >= OH) return;
    int X = ox / p.F, Y = oy / p.F, i = oy - Y * p.F, j = ox - X * p.F;
    int n = p.H * p.W, at = Y * p.W + X;
    float m[9];
    float mx = -1e30;
    for (int k = 0; k < 9; k++) { m[k] = p.mask.v[(k * p.F * p.F + i * p.F + j) * n + at]; mx = max(mx, m[k]); }
    float sum = 0.0;
    for (int k = 0; k < 9; k++) { m[k] = exp(m[k] - mx); sum += m[k]; }
    vec2 acc = vec2(0.0);
    for (int k = 0; k < 9; k++) {
        int yy = Y + k / 3 - 1, xx = X + k % 3 - 1;
        if (yy < 0 || yy >= p.H || xx < 0 || xx >= p.W) continue;
        acc += m[k] * vec2(p.flow.v[yy * p.W + xx], p.flow.v[n + yy * p.W + xx]);
    }
    acc *= float(p.F) / sum;
    p.o.v[oy * OW + ox] = acc.x;
    p.o.v[OH * OW + oy * OW + ox] = acc.y;
}
"#;

/// The flow at the network's (padded, reduced) size to the picture's: the padding cut
/// (offset pl, pt; valid size VW x VH) and resized by K (bilinear, align_corners=False,
/// clamped at the valid edges), vectors scaled by K.
const TO_FULL: &str = r#"
layout(local_size_x = 16, local_size_y = 8, local_size_z = 1) in;
layout(push_constant) uniform P { F32 a; F32 o; int AW, AH, pl, pt, VW, VH, OW, OH; float K; } p;
void main() {
    int x = int(gl_GlobalInvocationID.x), y = int(gl_GlobalInvocationID.y), c = int(gl_GlobalInvocationID.z);
    if (x >= p.OW || y >= p.OH || c >= 2) return;
    float fx = max((float(x) + 0.5) * float(p.VW) / float(p.OW) - 0.5, 0.0);
    float fy = max((float(y) + 0.5) * float(p.VH) / float(p.OH) - 0.5, 0.0);
    int x0 = min(int(fx), p.VW - 1), y0 = min(int(fy), p.VH - 1);
    int x1 = min(x0 + 1, p.VW - 1), y1 = min(y0 + 1, p.VH - 1);
    float ax = fx - float(x0), ay = fy - float(y0);
    int b = c * p.AH * p.AW;
    #define AT(xx, yy) p.a.v[b + (yy + p.pt) * p.AW + xx + p.pl]
    float t = mix(AT(x0, y0), AT(x1, y0), ax), u = mix(AT(x0, y1), AT(x1, y1), ax);
    p.o.v[(c * p.OH + y) * p.OW + x] = mix(t, u, ay) * p.K;
}
"#;

pub struct SeaRaft {
    ops: Arc<Ops>,
    feat: Net,
    ctx: Net,
    upd: Net,
    half_size: Arc<Kernel>,
    corr: Arc<Kernel>,
    convex: Arc<Kernel>,
    feat_t: Arc<Kernel>,
    to_full: Arc<Kernel>,
    /// Refinement steps.
    pub iters: usize,
}

const LEVELS: usize = 4;
const RADIUS: i32 = 4;

impl SeaRaft {
    pub fn new(ops: &Arc<Ops>, models: &Path) -> Result<Self> {
        let c = &ops.compute;
        let dir = models.join("flow");
        let load = |part: &str| -> Result<Net> {
            let stem = dir.join(format!("searaft_m_{part}"));
            let param = std::fs::read_to_string(stem.with_extension("param")).with_context(|| format!("{}.param", stem.display()))?;
            let bin = std::fs::read(stem.with_extension("bin"))?;
            let mut net = Net::load(c, &param, &bin)?;
            net.set_half(true);
            // One pool with the other parts (they run one after another).
            net.share_pool(&ops.pool);
            Ok(net)
        };
        let k = |name: &str, body: &str| c.kernel(&format!("flow_{name}"), &format!("{PRELUDE}{body}"), 0);
        Ok(SeaRaft {
            ops: ops.clone(),
            feat: load("feat")?,
            ctx: load("ctx")?,
            upd: load("upd")?,
            half_size: k("half_size", HALF_SIZE)?,
            corr: k("corr", CORR)?,
            convex: k("convex", CONVEX)?,
            feat_t: k("feat_t", FEAT_T)?,
            to_full: k("to_full", TO_FULL)?,
            iters: 4,
        })
    }

    pub fn available(models: &Path) -> bool {
        models.join("flow").join("searaft_m_upd.param").exists()
    }

    fn half(&self, rec: &mut Recorder, a: &Tensor) -> Result<Tensor> {
        let (oh, ow) = (a.h / 2, a.w / 2);
        let out = self.ops.tensor(a.c, oh, ow)?;
        rec.label = "flow: half size".into();
        let push = Push::new().t(a).t(&out).i(a.c as i32).i(a.h as i32).i(a.w as i32).i(oh as i32).i(ow as i32);
        rec.dispatch(&self.half_size, Recorder::groups(&self.half_size, [ow as u32, oh as u32, a.c as u32]), &push)?;
        Ok(out)
    }

    /// `f` (planar f32) as pixel-major f16, for the correlation.
    fn transposed(&self, rec: &mut Recorder, f: &Tensor) -> Result<Tensor> {
        anyhow::ensure!(f.c % 4 == 0, "feature channels must be a multiple of 4");
        rec.label = "flow: features to pixel-major".into();
        let out = self.ops.tensor(f.c.div_ceil(2), f.h, f.w)?;
        let n = f.h * f.w;
        rec.dispatch(&self.feat_t, Recorder::groups(&self.feat_t, [n as u32, f.c as u32, 1]), &Push::new().t(f).t(&out).i(f.c as i32).i(n as i32))?;
        Ok(out)
    }

    /// The flow from `a` to `b` (pictures 3 channels 0..1, same size): 2 channels, in
    /// pixels of the picture, at its size.
    pub fn flow(&self, rec: &mut Recorder, a: &Tensor, b: &Tensor) -> Result<Tensor> {
        let ops = &self.ops;
        let (w, h) = (a.w, a.h);
        // Half size, padded to a multiple of 8 (replicated, centred).
        let (ha, hb) = (self.half(rec, &a.view(0, 3))?, self.half(rec, &b.view(0, 3))?);
        let (vw, vh) = (ha.w, ha.h);
        let (pw, ph) = (vw.div_ceil(8) * 8, vh.div_ceil(8) * 8);
        let (pl, pt) = ((pw - vw) / 2, (ph - vh) / 2);
        let pa = ops.crop(rec, &ha, 0, 3, -(pl as isize), -(pt as isize), pw, ph, true)?;
        let pb = ops.crop(rec, &hb, 0, 3, -(pl as isize), -(pt as isize), pw, ph, true)?;
        ops.recycle(rec, ha);
        ops.recycle(rec, hb);
        let f1 = self.feat.record(rec, &[("img", &pa)], &["fmap"])?.remove(0);
        let f2 = self.feat.record(rec, &[("img", &pb)], &["fmap"])?.remove(0);
        let mut c = self.ctx.record(rec, &[("img1", &pa), ("img2", &pb)], &["net", "context", "flow0"])?;
        let (mut net, context, mut flow) = (c.remove(0), c.remove(0), c.remove(0));
        // The pyramid of the second picture's features.
        let mut pyr = vec![f2];
        for _ in 1..LEVELS {
            let next = self.half(rec, pyr.last().unwrap())?;
            pyr.push(next);
        }
        let pyr_t: Vec<Tensor> = pyr.iter().map(|t| self.transposed(rec, t)).collect::<Result<_>>()?;
        let f1t = self.transposed(rec, &f1)?;
        let (lw, lh, fc) = (f1.w, f1.h, f1.c);
        // Everything made here goes back to the pool (never dropped: a dropped buffer
        // is freed before the recorded work runs).
        for t in pyr {
            ops.recycle(rec, t);
        }
        ops.recycle(rec, f1);
        let pyr = pyr_t;
        let s = 2 * RADIUS as usize + 1;
        let mut mask = None;
        for _ in 0..self.iters {
            let corr = ops.tensor(LEVELS * s * s, lh, lw)?;
            rec.label = "flow: correlation".into();
            for (l, f2) in pyr.iter().enumerate() {
                let push = Push::new().t(&f1t).t(f2).t(&flow).t(&corr).i(fc as i32).i(lh as i32).i(lw as i32).i(f2.h as i32).i(f2.w as i32).i(l as i32).i(RADIUS);
                rec.dispatch(&self.corr, Recorder::groups(&self.corr, [(lw * lh) as u32, 1, 1]), &push)?;
            }
            let mut o = self.upd.record(rec, &[("net", &net), ("context", &context), ("corr", &corr), ("flow", &flow)], &["net_out", "flow_out", "mask"])?;
            ops.recycle(rec, corr);
            ops.recycle(rec, std::mem::replace(&mut net, o.remove(0)));
            ops.recycle(rec, std::mem::replace(&mut flow, o.remove(0)));
            if let Some(m) = mask.replace(o.remove(0)) {
                ops.recycle(rec, m);
            }
        }
        let mask = mask.context("no refinement step")?;
        rec.label = "flow: upsampling".into();
        let up = ops.tensor(2, lh * 8, lw * 8)?;
        let push = Push::new().t(&flow).t(&mask).t(&up).i(lh as i32).i(lw as i32).i(8);
        rec.dispatch(&self.convex, Recorder::groups(&self.convex, [(lw * 8) as u32, (lh * 8) as u32, 1]), &push)?;
        let out = ops.tensor(2, h, w)?;
        let k = w as f32 / vw as f32;
        let push = Push::new().t(&up).t(&out).i(pw as i32).i(ph as i32).i(pl as i32).i(pt as i32).i(vw as i32).i(vh as i32).i(w as i32).i(h as i32).f(k);
        rec.dispatch(&self.to_full, Recorder::groups(&self.to_full, [w as u32, h as u32, 2]), &push)?;
        for t in [pa, pb, up, net, context, flow, mask, f1t].into_iter().chain(pyr) {
            ops.recycle(rec, t);
        }
        Ok(out)
    }
}

// ---- WAFT -------------------------------------------------------------------------

/// Pixels 0..1 to ImageNet-normalized values, placed at (pl, pt) in a zero-padded
/// picture (WAFT pads after normalizing, with zeros).
const NORM_PAD: &str = r#"
layout(local_size_x = 16, local_size_y = 8, local_size_z = 1) in;
layout(push_constant) uniform P { F32 a; F32 o; int H, W, PH, PW, pl, pt; } p;
void main() {
    int x = int(gl_GlobalInvocationID.x), y = int(gl_GlobalInvocationID.y), c = int(gl_GlobalInvocationID.z);
    if (x >= p.PW || y >= p.PH || c >= 3) return;
    const vec3 mean = vec3(0.485, 0.456, 0.406), sd = vec3(0.229, 0.224, 0.225);
    int sx = x - p.pl, sy = y - p.pt;
    float v = 0.0;
    if (sx >= 0 && sx < p.W && sy >= 0 && sy < p.H) v = (p.a.v[(c * p.H + sy) * p.W + sx] - mean[c]) / sd[c];
    p.o.v[(c * p.PH + y) * p.PW + x] = v;
}
"#;

/// DINOv2's token sequence [C][1 + N]: the class token (its position embedding added:
/// `cls`) first, then the patch tokens with theirs (`pos`, [C][N]).
pub(crate) const SEQ_BUILD: &str = r#"
layout(local_size_x = 64, local_size_y = 4, local_size_z = 1) in;
layout(push_constant) uniform P { F32 tok; F32 cls; F32 pos; F32 o; int C, N; } p;
void main() {
    int t = int(gl_GlobalInvocationID.x), c = int(gl_GlobalInvocationID.y);
    if (t > p.N || c >= p.C) return;
    p.o.v[c * (p.N + 1) + t] = t == 0 ? p.cls.v[c] : p.tok.v[c * p.N + t - 1] + p.pos.v[c * p.N + t - 1];
}
"#;

/// The patch tokens of a sequence [C][1 + N] (class token dropped) as [C][N].
pub(crate) const SEQ_STRIP: &str = r#"
layout(local_size_x = 64, local_size_y = 4, local_size_z = 1) in;
layout(push_constant) uniform P { F32 a; F32 o; int C, N; } p;
void main() {
    int t = int(gl_GlobalInvocationID.x), c = int(gl_GlobalInvocationID.y);
    if (t >= p.N || c >= p.C) return;
    p.o.v[c * p.N + t] = p.a.v[c * (p.N + 1) + 1 + t];
}
"#;

/// Features sampled at (x + flow_x, y + flow_y), bilinear, zero outside (grid_sample,
/// align_corners=True, zero padding).
const WARP_FEAT: &str = r#"
layout(local_size_x = 16, local_size_y = 8, local_size_z = 1) in;
layout(push_constant) uniform P { F32 a; F32 flow; F32 o; int C, H, W; } p;
void main() {
    int x = int(gl_GlobalInvocationID.x), y = int(gl_GlobalInvocationID.y), c = int(gl_GlobalInvocationID.z);
    if (x >= p.W || y >= p.H || c >= p.C) return;
    int n = p.H * p.W, i = y * p.W + x;
    float fx = float(x) + p.flow.v[i], fy = float(y) + p.flow.v[n + i];
    float bx = floor(fx), by = floor(fy), ax = fx - bx, ay = fy - by;
    int x0 = int(bx), y0 = int(by);
    float v = 0.0;
    for (int k = 0; k < 4; k++) {
        int xx = x0 + (k & 1), yy = y0 + (k >> 1);
        if (xx < 0 || xx >= p.W || yy < 0 || yy >= p.H) continue;
        float wgt = ((k & 1) != 0 ? ax : 1.0 - ax) * ((k >> 1) != 0 ? ay : 1.0 - ay);
        v += wgt * p.a.v[(c * p.H + yy) * p.W + xx];
    }
    p.o.v[(c * p.H + y) * p.W + x] = v;
}
"#;

/// PyTorch's bicubic resize (mode="bicubic", align_corners=False, antialias=False)
/// with scale factors given, of planar [c][h][w] to [c][oh][ow]: source position
/// (dst + 0.5) / scale - 0.5, cubic A = -0.75, taps clamped to the edges.
pub(crate) fn bicubic(src: &[f32], c: usize, h: usize, w: usize, scale: (f64, f64), oh: usize, ow: usize) -> Vec<f32> {
    const A: f32 = -0.75;
    let cc1 = |x: f32| ((A + 2.0) * x - (A + 3.0)) * x * x + 1.0;
    let cc2 = |x: f32| ((A * x - 5.0 * A) * x + 8.0 * A) * x - 4.0 * A;
    let taps = |dst: usize, s: f64, size: usize| -> ([usize; 4], [f32; 4]) {
        let real = ((dst as f32 + 0.5) * (1.0 / s) as f32) - 0.5;
        let i0 = real.floor();
        let t = real - i0;
        let idx = |k: i64| (i0 as i64 - 1 + k).clamp(0, size as i64 - 1) as usize;
        ([idx(0), idx(1), idx(2), idx(3)], [cc2(t + 1.0), cc1(t), cc1(1.0 - t), cc2(2.0 - t)])
    };
    let mut out = vec![0f32; c * oh * ow];
    for y in 0..oh {
        let (iy, wy) = taps(y, scale.0, h);
        for x in 0..ow {
            let (ix, wx) = taps(x, scale.1, w);
            for ch in 0..c {
                let base = ch * h * w;
                let mut v = 0.0;
                for j in 0..4 {
                    let row: f32 = (0..4).map(|i| wx[i] * src[base + iy[j] * w + ix[i]]).sum();
                    v += wy[j] * row;
                }
                out[(ch * oh + y) * ow + x] = v;
            }
        }
    }
    out
}

/// Token-major [n][c] to planar [c][n].
pub(crate) fn planar(tokens: &[f32], n: usize, c: usize) -> Vec<f32> {
    let mut out = vec![0f32; n * c];
    for t in 0..n {
        for ch in 0..c {
            out[ch * n + t] = tokens[t * c + ch];
        }
    }
    out
}

/// WAFT a1 (princeton-vl/WAFT, BSD-3-Clause; Wang and Deng, ICLR 2026), with Depth
/// Anything V2 ViT-S features (Apache-2.0): the "WAFT (AI)" motion option. Pictures
/// are normalized and padded to a multiple of 112; each gets DINOv2 features (12
/// transformer blocks over 14 px patches, a DPT head) and ResNet-18 features at half
/// size; then 5 refinement steps, each a 12-block transformer over 8 px patches of the
/// half-size features with the other picture's moved by the flow, and the flow
/// upsampled x2 (convex). Graphs from `pth2ncnn waft`.
pub struct Waft {
    ops: Arc<Ops>,
    dpatch: Net,
    dblocks: Net,
    ddpt: Net,
    feat: Net,
    hidden: Net,
    refine: Net,
    norm_pad: Arc<Kernel>,
    seq_build: Arc<Kernel>,
    seq_strip: Arc<Kernel>,
    warp: Arc<Kernel>,
    convex: Arc<Kernel>,
    to_full: Arc<Kernel>,
    /// DINOv2's class token + its position embedding [384], its patch position
    /// embedding (37 x 37, planar), the refinement ViT's (8 x 8).
    cls: Vec<f32>,
    dpos: Vec<f32>,
    rpos: Vec<f32>,
    /// Interpolated position embeddings per padded picture size: (DINOv2 class
    /// vector, DINOv2 patches [C][N], refinement [C][n]).
    pos: std::sync::Mutex<std::collections::HashMap<(usize, usize), Arc<(Tensor, Tensor, Tensor)>>>,
    pub iters: usize,
}

const WAFT_DIM: usize = 384;

impl Waft {
    pub fn new(ops: &Arc<Ops>, models: &Path) -> Result<Self> {
        let c = &ops.compute;
        let dir = models.join("flow");
        let load = |part: &str| -> Result<Net> {
            let stem = dir.join(format!("waft_a1_{part}"));
            let param = std::fs::read_to_string(stem.with_extension("param")).with_context(|| format!("{}.param", stem.display()))?;
            let mut net = Net::load(c, &param, &std::fs::read(stem.with_extension("bin"))?)?;
            net.set_half(true);
            // One pool with the other parts (they run one after another).
            net.share_pool(&ops.pool);
            // At 1080p a whole pair is about a second of GPU work: submitted in pieces,
            // so none runs near Windows' 2 s limit (on slower cards too).
            net.set_flush_every(40);
            Ok(net)
        };
        let k = |name: &str, body: &str| c.kernel(&format!("flow_{name}"), &format!("{PRELUDE}{body}"), 0);
        let raw: Vec<f32> = std::fs::read(dir.join("waft_a1_pos.bin"))?.chunks_exact(4).map(|b| f32::from_le_bytes(b.try_into().unwrap())).collect();
        let d = WAFT_DIM;
        anyhow::ensure!(raw.len() == d + 1370 * d + 64 * d, "waft_a1_pos.bin: {} values", raw.len());
        let mut cls = raw[..d].to_vec();
        cls.iter_mut().zip(&raw[d..2 * d]).for_each(|(a, b)| *a += b);
        Ok(Waft {
            ops: ops.clone(),
            dpatch: load("dpatch")?,
            dblocks: load("dblocks")?,
            ddpt: load("ddpt")?,
            feat: load("feat")?,
            hidden: load("hidden")?,
            refine: load("refine")?,
            norm_pad: k("norm_pad", NORM_PAD)?,
            seq_build: k("seq_build", SEQ_BUILD)?,
            seq_strip: k("seq_strip", SEQ_STRIP)?,
            warp: k("warp_feat", WARP_FEAT)?,
            convex: k("convex", CONVEX)?,
            to_full: k("to_full", TO_FULL)?,
            cls,
            dpos: planar(&raw[2 * d..2 * d + 1369 * d], 1369, d),
            rpos: planar(&raw[d + 1370 * d..], 64, d),
            pos: Default::default(),
            iters: 5,
        })
    }

    pub fn available(models: &Path) -> bool {
        models.join("flow").join("waft_a1_refine.param").exists()
    }

    /// Position embeddings for a padded picture of ph x pw (computed once per size).
    fn pos(&self, ph: usize, pw: usize) -> Result<Arc<(Tensor, Tensor, Tensor)>> {
        if let Some(p) = self.pos.lock().unwrap().get(&(ph, pw)) {
            return Ok(p.clone());
        }
        let c = &self.ops.compute;
        let d = WAFT_DIM;
        // DINOv2: 37 x 37 to (ph / 14) x (pw / 14), scale factors with its +0.1.
        let (h0, w0) = (ph / 14, pw / 14);
        let dp = bicubic(&self.dpos, d, 37, 37, ((h0 as f64 + 0.1) / 37.0, (w0 as f64 + 0.1) / 37.0), h0, w0);
        // The refinement ViT: 8 x 8 to its patch grid at half size (8 px patches).
        let (h1, w1) = (ph / 2 / 8, pw / 2 / 8);
        let rp = bicubic(&self.rpos, d, 8, 8, (h1 as f64 / 8.0, w1 as f64 / 8.0), h1, w1);
        let t = |data: Vec<f32>, h: usize, w: usize| Tensor::from_image(c, &crate::Image { w, h, channels: d, data });
        let p = Arc::new((t(self.cls.clone(), 1, 1)?, t(dp, h0, w0)?, t(rp, h1, w1)?));
        self.pos.lock().unwrap().insert((ph, pw), p.clone());
        Ok(p)
    }

    /// Depth Anything V2 features of a normalized, padded picture: [32][ph][pw].
    fn da_features(&self, rec: &mut Recorder, img: &Tensor, pos: &(Tensor, Tensor, Tensor)) -> Result<Tensor> {
        let ops = &self.ops;
        let tok = self.dpatch.record(rec, &[("img", img)], &["tokens"])?.remove(0);
        let (d, n) = (WAFT_DIM, tok.h * tok.w);
        let seq = ops.tensor(d, 1, n + 1)?;
        rec.label = "waft: token sequence".into();
        let push = Push::new().t(&tok).t(&pos.0).t(&pos.1).t(&seq).i(d as i32).i(n as i32);
        rec.dispatch(&self.seq_build, Recorder::groups(&self.seq_build, [(n + 1) as u32, d as u32, 1]), &push)?;
        let (th, tw) = (tok.h, tok.w);
        ops.recycle(rec, tok);
        let layers = self.dblocks.record(rec, &[("x", &seq)], &["l2", "l5", "l8", "l11"])?;
        ops.recycle(rec, seq);
        let mut maps = Vec::new();
        for l in layers {
            let m = ops.tensor(d, th, tw)?;
            rec.label = "waft: strip class token".into();
            rec.dispatch(&self.seq_strip, Recorder::groups(&self.seq_strip, [n as u32, d as u32, 1]), &Push::new().t(&l).t(&m).i(d as i32).i(n as i32))?;
            ops.recycle(rec, l);
            maps.push(m);
        }
        let da = self.ddpt.record(rec, &[("l2", &maps[0]), ("l5", &maps[1]), ("l8", &maps[2]), ("l11", &maps[3]), ("img", img)], &["da"])?.remove(0);
        for m in maps {
            ops.recycle(rec, m);
        }
        Ok(da)
    }

    /// The flow from `a` to `b` (pictures 3+ channels 0..1, same size): 2 channels in
    /// pixels, at the pictures' size.
    pub fn flow(&self, rec: &mut Recorder, a: &Tensor, b: &Tensor) -> Result<Tensor> {
        let ops = &self.ops;
        let (w, h) = (a.w, a.h);
        // WAFT's Padder: up to a multiple of 112 with at least 8 more, centred.
        let (ph, pw) = (((h + 8) / 112 + 1) * 112, ((w + 8) / 112 + 1) * 112);
        let (pl, pt) = ((pw - w) / 2, (ph - h) / 2);
        let pos = self.pos(ph, pw)?;
        let mut fm = Vec::new();
        for src in [a, b] {
            let img = ops.tensor(3, ph, pw)?;
            rec.label = "waft: normalize".into();
            let push = Push::new().t(src).t(&img).i(h as i32).i(w as i32).i(ph as i32).i(pw as i32).i(pl as i32).i(pt as i32);
            rec.dispatch(&self.norm_pad, Recorder::groups(&self.norm_pad, [pw as u32, ph as u32, 3]), &push)?;
            let da = self.da_features(rec, &img, &pos)?;
            fm.push(self.feat.record(rec, &[("da", &da), ("img", &img)], &["fmap"])?.remove(0));
            rec.flush()?;
            ops.recycle(rec, img);
            ops.recycle(rec, da);
        }
        let (f1, f2) = (&fm[0], &fm[1]);
        let mut net = self.hidden.record(rec, &[("fmap1", f1), ("fmap2", f2)], &["net"])?.remove(0);
        let mut flow = ops.filled(rec, 2, f1.h, f1.w, 0.0)?;
        let mut mask = None;
        for _ in 0..self.iters {
            let warped = ops.tensor(f2.c, f2.h, f2.w)?;
            rec.label = "waft: warp features".into();
            let push = Push::new().t(f2).t(&flow).t(&warped).i(f2.c as i32).i(f2.h as i32).i(f2.w as i32);
            rec.dispatch(&self.warp, Recorder::groups(&self.warp, [f2.w as u32, f2.h as u32, f2.c as u32]), &push)?;
            let mut o = self.refine.record(rec, &[("fmap1", f1), ("warp", &warped), ("net", &net), ("flow", &flow), ("pos", &pos.2)], &["net_out", "flow_out", "mask"])?;
            ops.recycle(rec, warped);
            ops.recycle(rec, std::mem::replace(&mut net, o.remove(0)));
            ops.recycle(rec, std::mem::replace(&mut flow, o.remove(0)));
            if let Some(m) = mask.replace(o.remove(0)) {
                ops.recycle(rec, m);
            }
            rec.flush()?;
        }
        let mask = mask.context("no refinement step")?;
        let (lw, lh) = (f1.w, f1.h);
        rec.label = "waft: upsampling".into();
        let up = ops.tensor(2, lh * 2, lw * 2)?;
        let push = Push::new().t(&flow).t(&mask).t(&up).i(lh as i32).i(lw as i32).i(2);
        rec.dispatch(&self.convex, Recorder::groups(&self.convex, [(lw * 2) as u32, (lh * 2) as u32, 1]), &push)?;
        let out = ops.tensor(2, h, w)?;
        let push = Push::new().t(&up).t(&out).i(pw as i32).i(ph as i32).i(pl as i32).i(pt as i32).i(w as i32).i(h as i32).i(w as i32).i(h as i32).f(1.0);
        rec.dispatch(&self.to_full, Recorder::groups(&self.to_full, [w as u32, h as u32, 2]), &push)?;
        for t in [up, net, flow, mask].into_iter().chain(fm) {
            ops.recycle(rec, t);
        }
        Ok(out)
    }
}
