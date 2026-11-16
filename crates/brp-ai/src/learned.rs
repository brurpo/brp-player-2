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
//! 8. With `acc 1` in config.txt the pre-merge chooses from a fifth picture: the
//!    accumulated one, a running mix of input pictures only (the pre-merge's own mix
//!    without the previous output), moved along the motion. `fuse` then reads 27
//!    channels (the 22, that picture, its two mismatch maps) and gives 5 weights; the
//!    state is the output's 3 planes, the accumulated picture's 3, then the memory.
//!
//! 9. With `flow_match 1` the motion towards the picture before is corrected before it
//!    is used: the picture is matched against the previous result (clean, and at the
//!    picture before's time) for 25 offsets within 1 px around the estimate
//!    (tools/denoise_train/flow_match.py, kernels MATCH_*). No learned part.
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

/// The motion mix's input at the motion's level: this luma, the neighbour's moved by
/// the estimate and by the prior, prior - estimate. The prior is minus the motion in
/// `st` (same size): the previous picture's motion towards this one, read where the
/// estimate says the pixel came from (`same` = 0), or this picture's motion towards
/// the other neighbour, at the pixel itself (`same` = 1). `scale`: the prior times
/// this (the picture before is that many frames back, model.py `flows` gap).
const MIX_IN: &str = r#"
G3
layout(push_constant) uniform P { F32 la; F32 lb; F32 f; F32 st; F32 o; int W, H, same; float scale; } p;
void main() { ID3 if (x >= p.W || y >= p.H || c > 0) return;
    int n = p.W * p.H, i = y * p.W + x;
    vec2 f = vec2(p.f.v[i], p.f.v[n + i]);
    float sx = float(x) + f.x, sy = float(y) + f.y;
    vec2 pr = -p.scale * (p.same != 0 ? vec2(p.st.v[i], p.st.v[n + i]) : vec2(bilc(p.st, 0, p.W, p.H, sx, sy), bilc(p.st, n, p.W, p.H, sx, sy)));
    p.o.v[i] = p.la.v[i];
    p.o.v[n + i] = bilc(p.lb, 0, p.W, p.H, sx, sy);
    p.o.v[2 * n + i] = bilc(p.lb, 0, p.W, p.H, float(x) + pr.x, float(y) + pr.y);
    p.o.v[3 * n + i] = pr.x - f.x;
    p.o.v[4 * n + i] = pr.y - f.y;
}
"#;

/// The mixed motion: estimate + share x (prior - estimate) + correction, from the mix
/// network's output `d` (share logit, correction) and its input `xi` (channels 3, 4).
const MIX_OUT: &str = r#"
G3
layout(push_constant) uniform P { F32 f; F32 xi; F32 d; int W, H; } p;
void main() { ID3 if (x >= p.W || y >= p.H || c > 0) return;
    int n = p.W * p.H, i = y * p.W + x;
    float s = 1.0 / (1.0 + exp(-p.d.v[i]));
    p.f.v[i] += s * p.xi.v[3 * n + i] + p.d.v[n + i];
    p.f.v[n + i] += s * p.xi.v[4 * n + i] + p.d.v[2 * n + i];
}
"#;

/// flow_match.py, step 1: the mean of the 3 channels minus its 5 x 5 mean (edges
/// repeated).
const MATCH_NORM: &str = r#"
G3
layout(push_constant) uniform P { F32 a; F32 o; int W, H; } p;
float lum(int x, int y) { int n = p.W * p.H, q = clamp(y, 0, p.H - 1) * p.W + clamp(x, 0, p.W - 1); return (p.a.v[q] + p.a.v[n + q] + p.a.v[2 * n + q]) / 3.0; }
void main() { ID3 if (x >= p.W || y >= p.H || c > 0) return;
    float s = 0.0;
    for (int j = -2; j <= 2; j++) for (int i = -2; i <= 2; i++) s += lum(x + i, y + j);
    p.o.v[y * p.W + x] = lum(x, y) - s / 25.0;
}
"#;

/// flow_match.py, step 2: plane c of 25 is the offset (c % 5, c / 5) x 0.5 - 1 px. The
/// squared difference between this picture's normalised luma `a` and the reference's
/// `b` at p + flow + offset, averaged over 4 x 4 blocks. The flow is at 1/k of the
/// size, in its own pixels.
const MATCH_COST: &str = r#"
G3
layout(push_constant) uniform P { F32 a; F32 b; F32 f; F32 o; int W, H, FW, FH; float k; } p;
void main() { ID3 int OW = p.W / 4, OH = p.H / 4; if (x >= OW || y >= OH || c >= 25) return;
    vec2 d = vec2(float(c % 5), float(c / 5)) * 0.5 - 1.0;
    float s = 0.0;
    for (int j = 0; j < 4; j++) for (int i = 0; i < 4; i++) {
        int X = 4 * x + i, Y = 4 * y + j;
        float sx = (float(X) + 0.5) / p.k - 0.5, sy = (float(Y) + 0.5) / p.k - 0.5;
        float px = float(X) + p.k * bilc(p.f, 0, p.FW, p.FH, sx, sy) + d.x, py = float(Y) + p.k * bilc(p.f, p.FW * p.FH, p.FW, p.FH, sx, sy) + d.y;
        float e = p.a.v[Y * p.W + X] - bilc(p.b, 0, p.W, p.H, px, py);
        s += e * e;
    }
    p.o.v[(c * OH + y) * OW + x] = s / 16.0;
}
"#;

/// Mean over (2r + 1)^2 pixels of each of C planes, edges repeated.
const MATCH_BOX: &str = r#"
G3
layout(push_constant) uniform P { F32 a; F32 o; int C, W, H, r; } p;
void main() { ID3 if (x >= p.W || y >= p.H || c >= p.C) return;
    int n = p.W * p.H;
    float s = 0.0;
    for (int j = -p.r; j <= p.r; j++) for (int i = -p.r; i <= p.r; i++) s += p.a.v[c * n + clamp(y + j, 0, p.H - 1) * p.W + clamp(x + i, 0, p.W - 1)];
    float k = float(2 * p.r + 1);
    p.o.v[c * n + y * p.W + x] = s / (k * k);
}
"#;

/// flow_match.py, step 4: per block the offset of the lowest cost (the first one, as
/// torch.argmin), refined by a parabola along x and y, times a share that grows with
/// how much lower that cost is than the cost of no correction (plane 12).
const MATCH_PICK: &str = r#"
G3
layout(push_constant) uniform P { F32 cost; F32 o; int W, H; float soft; } p;
void main() { ID3 if (x >= p.W || y >= p.H || c > 0) return;
    int n = p.W * p.H, q = y * p.W + x;
    int best = 0;
    float c0 = p.cost.v[q];
    for (int k = 1; k < 25; k++) { float v = p.cost.v[k * n + q]; if (v < c0) { c0 = v; best = k; } }
    int ix = best % 5, iy = best / 5;
    float sx = 0.0, sy = 0.0;
    if (ix > 0 && ix < 4) { float cm = p.cost.v[(best - 1) * n + q], cp = p.cost.v[(best + 1) * n + q]; sx = clamp(0.5 * (cm - cp) / max(cm + cp - 2.0 * c0, 1e-12), -0.5, 0.5); }
    if (iy > 0 && iy < 4) { float cm = p.cost.v[(best - 5) * n + q], cp = p.cost.v[(best + 5) * n + q]; sy = clamp(0.5 * (cm - cp) / max(cm + cp - 2.0 * c0, 1e-12), -0.5, 0.5); }
    float none = p.cost.v[12 * n + q];
    float share = clamp((none - c0) / (none * p.soft + 1e-12), 0.0, 1.0);
    p.o.v[q] = ((float(ix) + sx) * 0.5 - 1.0) * share;
    p.o.v[n + q] = ((float(iy) + sy) * 0.5 - 1.0) * share;
}
"#;

/// flow_match.py SOFT.
const MATCH_SOFT: f32 = 0.03;

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

/// The memory (C planes at half size: the planes of the state after the pictures) moved
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
    for (int ch = 0; ch < p.C; ch++) p.o.v[ch * n + i] = bilc(p.st, ch * n, w, h, float(x) + f.x, float(y) + f.y);
}
"#;

/// The state for the next picture: the denoised picture's planes (n values), the
/// accumulated picture's (a values, 0: none), the memory (m values) kept within
/// -1..1, then the motion towards the next picture (f values, 0: none; the rest of
/// its plane is not used).
const PACK: &str = r#"
G1
layout(push_constant) uniform P { F32 den; F32 acc; F32 mem; F32 flow; F32 o; int n, a, m, f; } p;
void main() { ID1 int e = p.n + p.a + p.m; if (i >= e + p.f) return;
    p.o.v[i] = i < p.n ? p.den.v[i] : i < p.n + p.a ? p.acc.v[i - p.n] : i < e ? clamp(p.mem.v[i - p.n - p.a], -1.0, 1.0) : p.flow.v[i - e]; }
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
/// 4 flags, the mismatch maps (block differences upsampled, x 8). With the
/// accumulated picture (C = 27) the pre-merge also reads it (zeros without a
/// previous picture) and its two mismatch maps.
const ASSEMBLE: &str = r#"
G3
layout(push_constant) uniform P { F32 cur; F32 wb; F32 wa; F32 wp; F32 wc; F32 fl; F32 d4; F32 d16; F32 o; int W, H, C; } p;
void main() { ID3 if (x >= p.W || y >= p.H || c >= p.C) return;
    int n = p.W * p.H, i = y * p.W + x;
    float v;
    if (c < 3) v = p.cur.v[c * n + i];
    else if (c < 6) v = p.fl.v[4] > 0.5 ? p.cur.v[(c - 3) * n + i] : p.wb.v[(c - 3) * n + i];
    else if (c < 9) v = p.fl.v[5] > 0.5 ? p.cur.v[(c - 6) * n + i] : p.wa.v[(c - 6) * n + i];
    else if (c < 12) v = p.fl.v[2] > 0.5 ? p.wp.v[(c - 9) * n + i] : 0.0;
    else if (c < 16) v = p.fl.v[c - 12];
    else if (c >= 22 && c < 25) v = p.fl.v[2] > 0.5 ? p.wc.v[(c - 22) * n + i] : 0.0;
    else {
        int j = c < 22 ? c - 16 : c - 19, pic = j / 2, k = (j & 1) == 0 ? 4 : 16, w = p.W / k, h = p.H / k;
        float sx = (float(x) + 0.5) / float(k) - 0.5, sy = (float(y) + 0.5) / float(k) - 0.5;
        v = 8.0 * ((j & 1) == 0 ? bilc(p.d4, pic * w * h, w, h, sx, sy) : bilc(p.d16, pic * w * h, w, h, sx, sy));
    }
    p.o.v[c * n + i] = v;
}
"#;

/// The pre-merge: the N pictures (4, or 5 with the accumulated one, which shares the
/// previous output's flag) mixed by the softmax of the network's weights (slots
/// without a picture get none). With 5, also the accumulated picture's next value:
/// the same mix without the previous output.
const BLEND: &str = r#"
G3
layout(push_constant) uniform P { F32 xs; F32 lg; F32 o; F32 acc; int W, H, N; } p;
void main() { ID3 if (x >= p.W || y >= p.H || c > 0) return;
    int n = p.W * p.H, i = y * p.W + x;
    float l[5];
    float m = -1e30;
    for (int k = 0; k < p.N; k++) {
        l[k] = p.lg.v[k * n + i];
        if (k > 0 && p.xs.v[(11 + min(k, 3)) * n + i] < 0.5) l[k] = -1e4;
        m = max(m, l[k]);
    }
    float s = 0.0;
    for (int k = 0; k < p.N; k++) { l[k] = exp(l[k] - m); s += l[k]; }
    for (int ch = 0; ch < 3; ch++) {
        float v = 0.0, a = 0.0;
        for (int k = 0; k < p.N; k++) {
            float px = p.xs.v[((k < 4 ? 3 * k : 22) + ch) * n + i];
            v += l[k] * px;
            if (k != 3) a += l[k] * px;
        }
        p.o.v[ch * n + i] = v / s;
        if (p.N == 5) p.acc.v[ch * n + i] = a / max(s - l[3], 1e-4 * s);
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
    /// The accumulated picture is carried too (a fifth picture for the pre-merge).
    acc: bool,
    /// The motion towards the picture before is estimated against the previous output
    /// (the state's picture), where there is one, instead of the picture before: that
    /// lines the history itself up with the new picture.
    flow_prev: bool,
    /// The motion mix (model.py `Motion.mix`): each motion estimate is mixed with a
    /// prior, the motion of the picture before carried over; the state then has one
    /// more plane, holding the motion towards the next picture.
    mix: Option<Net>,
    mix_in: Arc<Kernel>,
    mix_out: Arc<Kernel>,
    /// The motion towards the picture before is corrected by matching the picture
    /// against the previous result (flow_match.py).
    flow_match: bool,
    match_norm: Arc<Kernel>,
    match_cost: Arc<Kernel>,
    match_box: Arc<Kernel>,
    match_pick: Arc<Kernel>,
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
        Self::available_in(&models.join("brpdn"))
    }

    /// Whether `dir` holds an exported model.
    pub fn available_in(dir: &Path) -> bool {
        dir.join("unet.param").exists()
    }

    pub fn new(ops: &Arc<Ops>, models: &Path) -> Result<Self> {
        Self::open(ops, &models.join("brpdn"))
    }

    /// The model exported into `dir` (`models/brpdn` is the default one; the player
    /// keeps others beside it, see its `LEARNED_MODELS`).
    pub fn open(ops: &Arc<Ops>, dir: &Path) -> Result<Self> {
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
            acc: value("acc").is_some_and(|v| v == "1"),
            flow_prev: value("flow_prev").is_some_and(|v| v == "1"),
            mix: if value("flow_mix").is_some_and(|v| v == "1") { Some(net("mix", false)?) } else { None },
            mix_in: k("mix_in", MIX_IN)?,
            mix_out: k("mix_out", MIX_OUT)?,
            flow_match: value("flow_match").is_some_and(|v| v == "1"),
            match_norm: k("match_norm", MATCH_NORM)?,
            match_cost: k("match_cost", MATCH_COST)?,
            match_box: k("match_box", MATCH_BOX)?,
            match_pick: k("match_pick", MATCH_PICK)?,
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
    /// that level's pixels; and the two lumas at that level (for the mix).
    fn motion(&self, rec: &mut Recorder, a: &Tensor, b: &Tensor) -> Result<(Tensor, Tensor, Tensor)> {
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
        let (la, lb) = (pa.remove(0), pb.remove(0));
        for t in pa.into_iter().chain(pb) {
            ops.recycle(rec, t);
        }
        Ok((flow.context("no motion level")?, la, lb))
    }

    /// `flow` (an estimate, with its lumas) mixed in place with its prior: minus the
    /// motion in `st`, times `scale` (see MIX_IN).
    #[allow(clippy::too_many_arguments)]
    fn mixed(&self, rec: &mut Recorder, mix: &Net, flow: &Tensor, la: &Tensor, lb: &Tensor, st: &Tensor, same: bool, scale: f32) -> Result<()> {
        let (w, h) = (flow.w, flow.h);
        let x = self.ops.tensor(5, h, w)?;
        rec.dispatch(&self.mix_in, g(&self.mix_in, w, h, 1), &Push::new().t(la).t(lb).t(flow).t(st).t(&x).i(w as i32).i(h as i32).i(same as i32).f(scale))?;
        let d = mix.record(rec, &[("x", &x)], &["out"])?.remove(0);
        ensure!(!d.half && (d.c, d.h, d.w) == (3, h, w), "motion mix network output");
        rec.dispatch(&self.mix_out, g(&self.mix_out, w, h, 1), &Push::new().t(flow).t(&x).t(&d).i(w as i32).i(h as i32))?;
        self.ops.recycle(rec, x);
        self.ops.recycle(rec, d);
        Ok(())
    }

    /// flow_match.py: `flow` (towards the picture before, at 1/base of the size)
    /// corrected in place by matching this picture `x` against the previous result
    /// `reference` (both PQ, 3 planes of the padded size).
    fn matched(&self, rec: &mut Recorder, flow: &Tensor, x: &Tensor, reference: &Tensor) -> Result<()> {
        let ops = &self.ops;
        let (w, h) = (x.w, x.h);
        let (qw, qh) = (w / 4, h / 4);
        let (a, b) = (ops.tensor(1, h, w)?, ops.tensor(1, h, w)?);
        for (src, dst) in [(x, &a), (reference, &b)] {
            rec.dispatch(&self.match_norm, g(&self.match_norm, w, h, 1), &Push::new().t(src).t(dst).i(w as i32).i(h as i32))?;
        }
        let cost = ops.tensor(25, qh, qw)?;
        let push = Push::new().t(&a).t(&b).t(flow).t(&cost).i(w as i32).i(h as i32).i(flow.w as i32).i(flow.h as i32).f(self.base as f32);
        rec.dispatch(&self.match_cost, g(&self.match_cost, qw, qh, 25), &push)?;
        let mean = ops.tensor(25, qh, qw)?;
        rec.dispatch(&self.match_box, g(&self.match_box, qw, qh, 25), &Push::new().t(&cost).t(&mean).i(25).i(qw as i32).i(qh as i32).i(2))?;
        let d = ops.tensor(2, qh, qw)?;
        rec.dispatch(&self.match_pick, g(&self.match_pick, qw, qh, 1), &Push::new().t(&mean).t(&d).i(qw as i32).i(qh as i32).f(MATCH_SOFT))?;
        let ds = ops.tensor(2, qh, qw)?;
        rec.dispatch(&self.match_box, g(&self.match_box, qw, qh, 2), &Push::new().t(&d).t(&ds).i(2).i(qw as i32).i(qh as i32).i(1))?;
        // To the motion's own size and pixels, and added there (the motion is enlarged
        // wherever it is used, the correction with it).
        let dr = ops.tensor(2, flow.h, flow.w)?;
        let push = Push::new().t(&ds).t(&dr).i(2).i(qw as i32).i(qh as i32).i(flow.w as i32).i(flow.h as i32).f(1.0 / self.base as f32);
        rec.dispatch(&self.resize, g(&self.resize, flow.w, flow.h, 2), &push)?;
        rec.dispatch(&self.add, Recorder::groups(&self.add, [flow.len() as u32, 1, 1]), &Push::new().t(flow).t(&dr).i(flow.len() as i32))?;
        for t in [a, b, cost, mean, d, ds, dr] {
            ops.recycle(rec, t);
        }
        Ok(())
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
    /// `gap`: how many frames back `before` (and `prev`) are; 1 unless playback
    /// skipped frames. The carried motion is over one frame, so the prior it gives
    /// is scaled by it.
    #[allow(clippy::too_many_arguments)]
    pub fn record(&self, rec: &mut Recorder, cur: &Tensor, before: Option<&Tensor>, after: Option<&Tensor>, prev: Option<&Tensor>, sdr: bool, strength: f32, cuts: bool, gap: f32) -> Result<(Tensor, Tensor)> {
        let gap = if gap.is_finite() { gap.max(1.0) } else { 1.0 };
        let ops = &self.ops;
        ensure!(cur.c >= 3 && !cur.half, "denoise: {} channels", cur.c);
        rec.label = "denoise (learned): motion".into();
        let x = self.encode(rec, cur, sdr)?;
        let (pw, ph) = (x.w, x.h);
        ensure!(self.mem % 4 == 0, "denoise: memory of {} channels", self.mem);
        ensure!(!self.acc || self.mem > 0, "denoise: accumulated picture without a memory");
        let pics = if self.acc { 6 } else { 3 };
        ensure!(self.mix.is_none() || self.mem > 0, "denoise: motion mix without a memory");
        let planes = pics + self.mem / 4 + self.mix.is_some() as usize;
        let prev = prev.filter(|p| (p.c, p.h, p.w) == (planes, ph, pw));
        // A missing neighbour is the picture itself, flagged off (as in training).
        let xb = before.map(|b| self.encode(rec, b, sdr)).transpose()?;
        let xa = after.map(|a| self.encode(rec, a, sdr)).transpose()?;
        let (fb, la, lb) = match prev.filter(|_| self.flow_prev && before.is_some()) {
            Some(p) => self.motion(rec, &x, &p.view(0, 3))?,
            None => self.motion(rec, &x, xb.as_ref().unwrap_or(&x))?,
        };
        // The prior for the motion towards the picture before: the previous picture's
        // motion towards this one (the state's last plane), negated.
        if let (Some(mix), Some(p), true) = (&self.mix, prev, before.is_some()) {
            ensure!(2 * fb.w * fb.h <= pw * ph, "denoise: motion level larger than a plane");
            self.mixed(rec, mix, &fb, &la, &lb, &p.view(planes - 1, 1), false, gap)?;
        }
        ops.recycle(rec, la);
        ops.recycle(rec, lb);
        let (fa, la, lb) = self.motion(rec, &x, xa.as_ref().unwrap_or(&x))?;
        // And for the motion towards the picture after: minus the motion towards the
        // one before (motion going on as it was).
        if let (Some(mix), true, true) = (&self.mix, before.is_some(), after.is_some()) {
            self.mixed(rec, mix, &fa, &la, &lb, &fb, true, 1.0 / gap)?;
        }
        ops.recycle(rec, la);
        ops.recycle(rec, lb);
        // The motion towards the picture before, corrected against the previous result
        // (after the mix above, which takes it as it was estimated, as model.py does).
        if let (true, Some(p), true) = (self.flow_match, prev, before.is_some()) {
            ensure!(pw % 4 == 0 && ph % 4 == 0, "denoise: padded size");
            self.matched(rec, &fb, &x, &p.view(0, 3))?;
        }
        let wb = self.warped(rec, xb.as_ref().unwrap_or(&x), &fb)?;
        let wa = self.warped(rec, xa.as_ref().unwrap_or(&x), &fa)?;
        // The previous output lines up with the picture before: the same motion.
        let wp = prev.map(|p| self.warped(rec, p, &fb)).transpose()?;
        // The accumulated picture, the same way.
        let wc = prev.filter(|_| self.acc).map(|p| self.warped(rec, &p.view(3, 3), &fb)).transpose()?;
        for t in xb.into_iter().chain(xa) {
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
            let mem = prev.map(|p| p.view(pics, self.mem / 4));
            let push = Push::new().t(mem.as_ref().unwrap_or(&x)).t(&fb).t(&fl).t(&h).i(self.mem as i32).i(pw as i32).i(ph as i32).i(fb.w as i32).i(fb.h as i32).i(prev.is_none() as i32).f(self.base as f32);
            rec.dispatch(&self.mem_warp, g(&self.mem_warp, pw / 2, ph / 2, 1), &push)?;
            Some(h)
        } else {
            None
        };
        ops.recycle(rec, fb);
        let moved = if self.acc { 4 } else { 3 };
        let (d4, d16) = (ops.tensor(moved, ph / 4, pw / 4)?, ops.tensor(moved, bh, bw)?);
        let list = [(&wb, 4, -1), (&wa, 5, -1), (wp.as_ref().unwrap_or(&x), -1, 2), (wc.as_ref().unwrap_or(&x), -1, 2)];
        for (i, (w, same, valid)) in list.into_iter().take(moved).enumerate() {
            self.block_diff(rec, &x, w, &fl, &d4.view(i, 1), 4, same, valid)?;
            self.block_diff(rec, &x, w, &fl, &d16.view(i, 1), 16, same, valid)?;
        }
        let chans = if self.acc { 27 } else { 22 };
        let xs = ops.tensor(chans, ph, pw)?;
        let push = Push::new().t(&x).t(&wb).t(&wa).t(wp.as_ref().unwrap_or(&x)).t(wc.as_ref().unwrap_or(&x)).t(&fl).t(&d4).t(&d16).t(&xs).i(pw as i32).i(ph as i32).i(chans as i32);
        rec.dispatch(&self.assemble, g(&self.assemble, pw, ph, chans), &push)?;
        for t in [wb, wa, fl, d4, d16].into_iter().chain(wp).chain(wc) {
            ops.recycle(rec, t);
        }

        rec.label = "denoise (learned): pre-merge".into();
        let lg = match &h {
            Some(h) => self.fuse.record(rec, &[("x", &xs), ("h", h)], &["out"])?,
            None => self.fuse.record(rec, &[("x", &xs)], &["out"])?,
        }
        .remove(0);
        let n = if self.acc { 5 } else { 4 };
        ensure!(!lg.half && (lg.c, lg.h, lg.w) == (n, ph, pw), "pre-merge network output");
        let base = ops.tensor(3, ph, pw)?;
        let acc = if self.acc { Some(ops.tensor(3, ph, pw)?) } else { None };
        rec.dispatch(&self.blend, g(&self.blend, pw, ph, 1), &Push::new().t(&xs).t(&lg).t(&base).t(acc.as_ref().unwrap_or(&base)).i(pw as i32).i(ph as i32).i(n as i32))?;
        ops.recycle(rec, lg);

        rec.label = "denoise (learned): merge".into();
        // The merge reads the first 22 channels.
        let mut made = {
            let x22 = xs.view(0, 22);
            match &h {
                Some(h) => self.unet.record(rec, &[("x", &x22), ("base", &base), ("h", h)], &["out", "state"])?,
                None => self.unet.record(rec, &[("x", &x22), ("base", &base)], &["out"])?,
            }
        };
        let den = made.remove(0);
        ensure!(!den.half && (den.c, den.h, den.w) == (3, ph, pw), "merge network output");
        ops.recycle(rec, xs);
        ops.recycle(rec, base);
        // The state: the picture, the accumulated one, and behind them the new memory.
        let state = match made.pop() {
            Some(m) => {
                ensure!(!m.half && (m.c, m.h, m.w) == (self.mem, ph / 2, pw / 2), "merge network memory");
                let st = ops.tensor(planes, ph, pw)?;
                let fl = if self.mix.is_some() { fa.len() } else { 0 };
                let push = Push::new().t(&den).t(acc.as_ref().unwrap_or(&den)).t(&m).t(&fa).t(&st).i(den.len() as i32).i(acc.as_ref().map_or(0, |a| a.len()) as i32).i(m.len() as i32).i(fl as i32);
                rec.dispatch(&self.pack, Recorder::groups(&self.pack, [st.len() as u32, 1, 1]), &push)?;
                ops.recycle(rec, m);
                Some(st)
            }
            None => None,
        };
        if let Some(a) = acc {
            ops.recycle(rec, a);
        }
        ops.recycle(rec, fa);
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
