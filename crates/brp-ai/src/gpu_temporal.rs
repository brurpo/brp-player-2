//! The temporal processing of `temporal.rs` on the GPU: motion (pyramidal
//! Lucas-Kanade with candidate testing), the temporal window merge before upscaling,
//! and motion blur from the motion. The merge has the same maths as the CPU version
//! (kept as the reference; its motion is the older, plain one); everything
//! stays on the GPU, including the scene-cut and noise-level decisions (computed by
//! reductions the merge kernel reads).

use crate::gpu::Ops;
use crate::vk::{PushTensor, Tensor};
use anyhow::Result;
use brp_render::compute::{Kernel, Push, Recorder, PRELUDE};
use std::sync::Arc;

/// Structure (smallest eigenvalue of the 7x7 gradient tensor) at which motion counts
/// as half determined.
const STRUCTURE: f32 = 0.02;
const FILL_RADIUS: usize = 12;
/// Share of the grain's own structure kept out of the motion solve (of what 7x7
/// windows of independent grain give); 0.1-1 measured alike, 0.25 best.
const GRAIN_REG: f32 = 0.25;
/// A candidate motion must match this much better than the pixel's own.
const CAND_MARGIN: f32 = 0.97;
/// Least structure a motion step needs above the finest level. Faint remains of a
/// pattern the pyramid blurred away would otherwise drive the motion there (waves of
/// 16 px moved 2.5 px came out 4 px wrong; 3e-4 still let 8 px ones go wrong).
const COARSE_FLOOR: f32 = 1e-3;
const SMOOTH_RADIUS: usize = 8;
/// Last, a wide fill at the motion's size: pixels without detail take the motion of the
/// detail around them, so leftover grain moves with the picture instead of boiling
/// (grain cases 30% better; Natural 4× flicker 0.0023 -> 0.0022); detail keeps its own.
const WIDE_RADIUS: usize = 32;
/// Splatting trust (`interpolate`): colour difference and forward-backward motion
/// mismatch (pixels) that each divide the trust by e, and its floor.
const SPLAT_SC: f32 = 0.05;
const SPLAT_SF: f32 = 1.0;
const SPLAT_ZMIN: f32 = 0.01;
/// Motion change (px) within GR px that divides the trust by e (motion boundaries).
const SPLAT_SG: f32 = 1.0;
/// With a fallback picture: mean colour difference of the two pictures along the
/// motion up to which the splatted picture is kept, and from which the fallback is
/// taken (in between, mixed); checked where each picture has this share of the weight.
const SPLAT_D0: f32 = 0.03;
const SPLAT_D1: f32 = 0.10;
const SPLAT_SHARE: f32 = 0.2;
/// Mean colour difference from the fallback up to which the splatted picture is
/// kept, and from which the fallback is taken.
const SPLAT_E0: f32 = 0.05;
const SPLAT_E1: f32 = 0.15;
/// Wider pictures get their motion at half size.
const MOTION_MAX_W: usize = 1280;
/// Agreement below which a neighbour is taken as another shot.
pub const CUT: f32 = 0.3;
/// Steady Detail: a neighbour's detail is used where its blurred luma is within this of
/// the current picture's (lined up on the same thing).
const STEADY_AGREE: f32 = 0.03;
/// Weight of each picture relative to the one a step nearer: near-equal weights,
/// as for grain the more pictures share the average the better (at 2 each way
/// 0.6 and 0.85 measured the same; at 5, grain beside a moving line fell from
/// 0.0094 to 0.0084 with no ghosting).
pub const DECAY: f32 = 0.85;
const TRUST: f32 = 0.05;
const NOISE: f32 = 0.006;
const CLAMP_SD: f32 = 1.25;
/// The clamp to the current 3x3 is widened by this many typical differences (the
/// noise between the pictures). Without it the clamp held every neighbour to the
/// current picture's local mean, and with it the current picture's coarse noise:
/// on real footage that noise survived any window and changed every frame (curtain
/// of the stage clip, Window 5: change between frames 0.0108 -> 0.0057; 1 gave 0.0078,
/// no clamp 0.0069). No ghosting on the moving line or the fist.
const CLAMP_NOISE: f32 = 2.0;
const PIXEL_KEEP: f32 = 2.5;
const PIXEL_REJECT: f32 = 5.0;
/// Histogram for the typical (median) difference: bins over 0..HIST_MAX.
const HIST_BINS: usize = 512;
const HIST_MAX: f32 = 0.25;

pub(crate) const COMMON: &str = r#"
#define G3 layout(local_size_x = 16, local_size_y = 8, local_size_z = 1) in;
#define G1 layout(local_size_x = 256) in;
#define ID3 int x = int(gl_GlobalInvocationID.x), y = int(gl_GlobalInvocationID.y), c = int(gl_GlobalInvocationID.z);
#define ID1 int i = int(gl_GlobalInvocationID.x + gl_GlobalInvocationID.y * gl_NumWorkGroups.x * gl_WorkGroupSize.x);
const float NAN = uintBitsToFloat(0x7fc00000u);
float bil(F32 p, int base, int W, int H, float x, float y) {
    // Bilinear sample, NaN outside the plane (pixel centres at integers).
    if (x < -0.5 || y < -0.5 || x > float(W) - 0.5 || y > float(H) - 0.5) return NAN;
    x = clamp(x, 0.0, float(W - 1)); y = clamp(y, 0.0, float(H - 1));
    int x0 = int(x), y0 = int(y), x1 = min(x0 + 1, W - 1), y1 = min(y0 + 1, H - 1);
    float ax = x - float(x0), ay = y - float(y0);
    return mix(mix(p.v[base + y0 * W + x0], p.v[base + y0 * W + x1], ax), mix(p.v[base + y1 * W + x0], p.v[base + y1 * W + x1], ax), ay);
}
float bilc(F32 p, int base, int W, int H, float x, float y) {
    // Bilinear sample clamped to the edges.
    x = clamp(x, 0.0, float(W - 1)); y = clamp(y, 0.0, float(H - 1));
    int x0 = int(x), y0 = int(y), x1 = min(x0 + 1, W - 1), y1 = min(y0 + 1, H - 1);
    float ax = x - float(x0), ay = y - float(y0);
    return mix(mix(p.v[base + y0 * W + x0], p.v[base + y0 * W + x1], ax), mix(p.v[base + y1 * W + x0], p.v[base + y1 * W + x1], ax), ay);
}
"#;

const LUMA: &str = r#"
G1
layout(push_constant) uniform P { F32 a; F32 o; int n; } p;
void main() { ID1 if (i >= p.n) return; p.o.v[i] = 0.299 * p.a.v[i] + 0.587 * p.a.v[p.n + i] + 0.114 * p.a.v[2 * p.n + i]; }
"#;

/// Half size with a 5x5 binomial filter (1 4 6 4 1 each way): fine textures must not
/// alias at the coarse levels, where they would seem to move the wrong way and set
/// the motion off by whole periods (a 2x2 average gave 4-6 px errors on 3-5 px
/// patterns, `aicheck lkcheck`).
const HALF: &str = r#"
G3
layout(push_constant) uniform P { F32 a; F32 o; int W, OW, OH, H; } p;
void main() { ID3 if (x >= p.OW || y >= p.OH) return;
    const float k[5] = float[5](1.0, 4.0, 6.0, 4.0, 1.0);
    float s = 0.0;
    for (int j = 0; j < 5; j++) {
        int yy = clamp(2 * y - 2 + j, 0, p.H - 1);
        for (int q = 0; q < 5; q++) s += k[j] * k[q] * p.a.v[yy * p.W + clamp(2 * x - 2 + q, 0, p.W - 1)];
    }
    p.o.v[y * p.OW + x] = s / 256.0; }
"#;

/// Flow to the next finer level: twice the size, twice the motion.
const UPFLOW: &str = r#"
G3
layout(push_constant) uniform P { F32 a; F32 o; int FW, FH, W, H; } p;
void main() { ID3 if (x >= p.W || y >= p.H || c >= 2) return;
    int sx = min(x / 2, p.FW - 1), sy = min(y / 2, p.FH - 1);
    p.o.v[(c * p.H + y) * p.W + x] = p.a.v[(c * p.FH + sy) * p.FW + sx] * 2.0; }
"#;

/// Lucas-Kanade terms: previous luma moved by the flow, its gradients, the error:
/// gx², gx gy, gy², gx e, gy e.
const LK_WARP: &str = r#"
G3
layout(push_constant) uniform P { F32 lp; F32 flow; F32 o; int W, H; } p;
void main() { ID3 if (x >= p.W || y >= p.H) return;
    int i = y * p.W + x, n = p.W * p.H;
    p.o.v[i] = bil(p.lp, 0, p.W, p.H, float(x) + p.flow.v[i], float(y) + p.flow.v[n + i]); }
"#;

const LK_TERMS: &str = r#"
G3
layout(push_constant) uniform P { F32 warped; F32 lc; F32 o; int W, H; } p;
void main() { ID3 if (x >= p.W || y >= p.H) return;
    int i = y * p.W + x, n = p.W * p.H;
    float t[5] = float[5](0.0, 0.0, 0.0, 0.0, 0.0);
    if (x > 0 && y > 0 && x < p.W - 1 && y < p.H - 1) {
        float gx = (p.warped.v[i + 1] - p.warped.v[i - 1]) * 0.5, gy = (p.warped.v[i + p.W] - p.warped.v[i - p.W]) * 0.5;
        float e = p.lc.v[i] - p.warped.v[i];
        if (!isnan(gx) && !isnan(gy) && !isnan(e)) t = float[5](gx * gx, gx * gy, gy * gy, gx * e, gy * e);
    }
    for (int k = 0; k < 5; k++) p.o.v[k * n + i] = t[k]; }
"#;

/// Separable window sum (or mean) over (2r+1) along x or y, edges clamped.
const BOX: &str = r#"
G3
layout(push_constant) uniform P { F32 a; F32 o; int C, W, H, r, vertical, mean; } p;
void main() { ID3 if (x >= p.W || y >= p.H || c >= p.C) return;
    int base = c * p.W * p.H;
    float s = 0.0;
    for (int d = -p.r; d <= p.r; d++) {
        int xx = p.vertical != 0 ? x : clamp(x + d, 0, p.W - 1), yy = p.vertical != 0 ? clamp(y + d, 0, p.H - 1) : y;
        s += p.a.v[base + yy * p.W + xx];
    }
    if (p.mean != 0) s /= float(2 * p.r + 1);
    p.o.v[base + y * p.W + x] = s; }
"#;

/// Candidate motions: the pixel's own and its neighbours' 4 and 12 px away (8
/// directions); keeps the one whose 9x9 patch (every other pixel) matches best. A
/// region set off by a wrong coarse guess, or a thin line LK cannot reach, takes the
/// motion of what is around it when that fits.
const CAND: &str = r#"
G3
layout(push_constant) uniform P { F32 lp; F32 lc; F32 fi; F32 fo; F32 hint; int W, H; float margin; int HW, HH, has_hint; } p;
float cost(int x, int y, vec2 v) {
    float s = 0.0;
    for (int j = -4; j <= 4; j += 2) for (int q = -4; q <= 4; q += 2) {
        int xx = clamp(x + q, 0, p.W - 1), yy = clamp(y + j, 0, p.H - 1);
        float w = bilc(p.lp, 0, p.W, p.H, float(xx) + v.x, float(yy) + v.y);
        s += abs(p.lc.v[yy * p.W + xx] - w);
    }
    return s;
}
void main() { ID3 if (x >= p.W || y >= p.H) return;
    int i = y * p.W + x, n = p.W * p.H;
    vec2 own = vec2(p.fi.v[i], p.fi.v[n + i]);
    vec2 best = own;
    float bc = cost(x, y, own) * p.margin;
    const ivec2 dirs[8] = ivec2[8](ivec2(1, 0), ivec2(-1, 0), ivec2(0, 1), ivec2(0, -1), ivec2(1, 1), ivec2(-1, -1), ivec2(1, -1), ivec2(-1, 1));
    for (int r = 0; r < 2; r++) {
        int d = r == 0 ? 4 : 12;
        for (int k = 0; k < 8; k++) {
            int xx = clamp(x + dirs[k].x * d, 0, p.W - 1), yy = clamp(y + dirs[k].y * d, 0, p.H - 1);
            int j = yy * p.W + xx;
            vec2 v = vec2(p.fi.v[j], p.fi.v[n + j]);
            float c = cost(x, y, v);
            if (c < bc) { bc = c; best = v; }
        }
    }
    if (p.has_hint != 0) {
        // The neighbouring pair's motion (at its own size), here and 4 px away.
        float k = float(p.HW) / float(p.W);
        const ivec2 near[5] = ivec2[5](ivec2(0, 0), ivec2(4, 0), ivec2(-4, 0), ivec2(0, 4), ivec2(0, -4));
        for (int t = 0; t < 5; t++) {
            float hx = (float(x + near[t].x) + 0.5) * k - 0.5, hy = (float(y + near[t].y) + 0.5) * k - 0.5;
            vec2 v = vec2(bilc(p.hint, 0, p.HW, p.HH, hx, hy), bilc(p.hint, p.HW * p.HH, p.HW, p.HH, hx, hy)) / k;
            float c = cost(x, y, v);
            if (c < bc) { bc = c; best = v; }
        }
    }
    p.fo.v[i] = best.x; p.fo.v[n + i] = best.y; }
"#;

/// Histogram of |luma - mean of its 4 neighbours| (the grain level is its median).
const NOISE_HIST: &str = r#"
G3
layout(push_constant) uniform P { F32 l; U32 hist; int W, H, bins; float maxv; } p;
void main() { ID3 if (x < 1 || y < 1 || x >= p.W - 1 || y >= p.H - 1) return;
    int i = y * p.W + x;
    float d = abs(p.l.v[i] - 0.25 * (p.l.v[i - 1] + p.l.v[i + 1] + p.l.v[i - p.W] + p.l.v[i + p.W]));
    int b = min(int(sqrt(min(d / p.maxv, 1.0)) * float(p.bins)), p.bins - 1);
    atomicAdd(p.hist.v[b], 1u); }
"#;

/// Solve for the motion step. The grain adds structure of its own and makes the step
/// random where the picture has no more detail than grain: `lam` (from the measured
/// grain) is added to the system, which keeps the coarser estimate there, and taken
/// from the structure.
const LK_SOLVE: &str = r#"
G3
layout(push_constant) uniform P { F32 sums; F32 flow; F32 structure; F32 noise; int W, H; float lk, floor_reg; } p;
void main() { ID3 if (x >= p.W || y >= p.H) return;
    int i = y * p.W + x, n = p.W * p.H;
    float a = p.sums.v[i], b = p.sums.v[n + i], cc = p.sums.v[2 * n + i], ex = p.sums.v[3 * n + i], ey = p.sums.v[4 * n + i];
    // The structure tensor less the grain's share (each eigenvalue less lam).
    float m = p.noise.v[0], lam = p.lk * m * m;
    float hm = (a + cc) * 0.5, hd = sqrt((a - cc) * (a - cc) * 0.25 + b * b);
    float l1 = hm + hd, r1 = max(l1 - lam, 0.0), r2 = max(hm - hd - lam, 0.0);
    vec2 e1 = abs(b) > 1e-12 ? normalize(vec2(b, l1 - a)) : (a >= cc ? vec2(1.0, 0.0) : vec2(0.0, 1.0));
    vec2 e2 = vec2(-e1.y, e1.x);
    p.structure.v[i] = r1 * e1.x * e1.x + r2 * e2.x * e2.x;
    p.structure.v[n + i] = r1 * e1.x * e1.y + r2 * e2.x * e2.y;
    p.structure.v[2 * n + i] = r1 * e1.y * e1.y + r2 * e2.y * e2.y;
    // Regularised: on a line only the motion across it moves (the rest is kept).
    float reg = lam + 1e-3 * (a + cc) + p.floor_reg;
    a += reg; cc += reg;
    float det = a * cc - b * b;
    if (det < 1e-14) return;
    p.flow.v[i] += clamp((cc * ex - b * ey) / det, -1.0, 1.0);
    p.flow.v[n + i] += clamp((a * ey - b * ex) / det, -1.0, 1.0); }
"#;

/// Motion weighted by its (saturated) structure tensor, for filling / smoothing:
/// tensor (3) and tensor x motion (2). A line only vouches for the motion across it.
const FILL_PREP: &str = r#"
G3
layout(push_constant) uniform P { F32 flow; F32 structure; F32 o; int W, H; float S, floor_w; } p;
void main() { ID3 if (x >= p.W || y >= p.H) return;
    int i = y * p.W + x, n = p.W * p.H;
    float a = p.structure.v[i], b = p.structure.v[n + i], tc = p.structure.v[2 * n + i];
    float k = 1.0 / (a + tc + p.S);
    a = a * k + p.floor_w; b *= k; tc = tc * k + p.floor_w;
    float vx = p.flow.v[i], vy = p.flow.v[n + i];
    p.o.v[i] = a; p.o.v[n + i] = b; p.o.v[2 * n + i] = tc;
    p.o.v[3 * n + i] = a * vx + b * vy; p.o.v[4 * n + i] = b * vx + tc * vy; }
"#;

/// Fill (the pixel's own determined motion weighs as much as `own` neighbours) or
/// smooth (own = 0): the motion that best fits the window's tensors; what nothing
/// determines stays as it was.
const FILL_APPLY: &str = r#"
G3
layout(push_constant) uniform P { F32 flow; F32 structure; F32 sums; int W, H; float S, own; } p;
void main() { ID3 if (x >= p.W || y >= p.H) return;
    int i = y * p.W + x, n = p.W * p.H;
    float vx = p.flow.v[i], vy = p.flow.v[n + i];
    float ma = p.sums.v[i], mb = p.sums.v[n + i], mc = p.sums.v[2 * n + i], rx = p.sums.v[3 * n + i], ry = p.sums.v[4 * n + i];
    if (p.own > 0.0) {
        float a = p.structure.v[i], b = p.structure.v[n + i], tc = p.structure.v[2 * n + i];
        float k = p.own / (a + tc + p.S);
        a *= k; b *= k; tc *= k;
        ma += a; mb += b; mc += tc; rx += a * vx + b * vy; ry += b * vx + tc * vy;
    }
    float e = 1e-3 * (ma + mc) + 1e-6;
    ma += e; mc += e; rx += e * vx; ry += e * vy;
    float det = ma * mc - mb * mb;
    if (det < 1e-14) return;
    p.flow.v[i] = (mc * rx - mb * ry) / det;
    p.flow.v[n + i] = (ma * ry - mb * rx) / det; }
"#;

/// Motion a then b (b may be at another resolution).
const CHAIN: &str = r#"
G3
layout(push_constant) uniform P { F32 a; F32 b; F32 o; int W, H, BW, BH; } p;
void main() { ID3 if (x >= p.W || y >= p.H) return;
    int i = y * p.W + x, n = p.W * p.H, bn = p.BW * p.BH;
    float kb = float(p.BW) / float(p.W);
    float ax = p.a.v[i], ay = p.a.v[n + i];
    float sx = (float(x) + ax + 0.5) * kb - 0.5, sy = (float(y) + ay + 0.5) * kb - 0.5;
    p.o.v[i] = ax + bilc(p.b, 0, p.BW, p.BH, sx, sy) / kb;
    p.o.v[n + i] = ay + bilc(p.b, bn, p.BW, p.BH, sx, sy) / kb; }
"#;

/// A picture moved along a flow (any resolution), Catmull-Rom; NaN outside.
const WARP: &str = r#"
G3
layout(push_constant) uniform P { F32 img; F32 flow; F32 o; int C, W, H, FW, FH; } p;
vec4 cr(float t) { float t2 = t * t, t3 = t2 * t;
    return vec4((-t3 + 2.0 * t2 - t), (3.0 * t3 - 5.0 * t2 + 2.0), (-3.0 * t3 + 4.0 * t2 + t), (t3 - t2)) * 0.5; }
void main() { ID3 if (x >= p.W || y >= p.H || c >= p.C) return;
    float k = float(p.FW) / float(p.W);
    float fx = (float(x) + 0.5) * k - 0.5, fy = (float(y) + 0.5) * k - 0.5;
    float dx = bilc(p.flow, 0, p.FW, p.FH, fx, fy) / k, dy = bilc(p.flow, p.FW * p.FH, p.FW, p.FH, fx, fy) / k;
    float sx = float(x) + dx, sy = float(y) + dy;
    int o = (c * p.H + y) * p.W + x;
    if (sx < -0.5 || sy < -0.5 || sx > float(p.W) - 0.5 || sy > float(p.H) - 0.5) { p.o.v[o] = NAN; return; }
    float bx = floor(sx), by = floor(sy);
    vec4 kx = cr(sx - bx), ky = cr(sy - by);
    int base = c * p.W * p.H;
    float v = 0.0;
    for (int j = 0; j < 4; j++) {
        int yy = clamp(int(by) - 1 + j, 0, p.H - 1);
        float r = 0.0;
        for (int q = 0; q < 4; q++) r += kx[q] * p.img.v[base + yy * p.W + clamp(int(bx) - 1 + q, 0, p.W - 1)];
        v += ky[j] * r;
    }
    p.o.v[o] = v; }
"#;

/// Signed difference per colour channel and "outside" (for the 5x5 agreement).
const AGREE_PREP: &str = r#"
G3
layout(push_constant) uniform P { F32 cur; F32 warped; F32 o; int W, H; } p;
void main() { ID3 if (x >= p.W || y >= p.H) return;
    int i = y * p.W + x, n = p.W * p.H;
    bool inside = !isnan(p.warped.v[i]);
    for (int k = 0; k < 3; k++) p.o.v[k * n + i] = inside ? p.cur.v[k * n + i] - p.warped.v[k * n + i] : 0.0;
    p.o.v[3 * n + i] = inside ? 0.0 : 1.0; }
"#;

const AGREE: &str = r#"
G3
layout(push_constant) uniform P { F32 sums; F32 o; int W, H; float noise, trust; } p;
void main() { ID3 if (x >= p.W || y >= p.H) return;
    int i = y * p.W + x, n = p.W * p.H;
    float cnt = 25.0;
    if (p.sums.v[3 * n + i] > 0.0) { p.o.v[i] = 0.0; return; }
    float e = (abs(p.sums.v[i]) + abs(p.sums.v[n + i]) + abs(p.sums.v[2 * n + i])) / (3.0 * cnt);
    p.o.v[i] = clamp(1.0 - max(e - p.noise, 0.0) / p.trust, 0.0, 1.0); }
"#;

/// Sum of `n` values: per workgroup into partials (pass 1), then one workgroup sums
/// the partials into out[slot] scaled (pass 2).
const SUM1: &str = r#"
G1
layout(push_constant) uniform P { F32 a; F32 part; int n; } p;
shared float s[256];
void main() {
    int lid = int(gl_LocalInvocationID.x), wg = int(gl_WorkGroupID.x + gl_WorkGroupID.y * gl_NumWorkGroups.x);
    int groups = int(gl_NumWorkGroups.x * gl_NumWorkGroups.y);
    float acc = 0.0;
    for (int i = wg * 256 + lid; i < p.n; i += groups * 256) acc += p.a.v[i];
    s[lid] = acc; barrier();
    for (int k = 128; k > 0; k >>= 1) { if (lid < k) s[lid] += s[lid + k]; barrier(); }
    if (lid == 0) p.part.v[wg] = s[0]; }
"#;

const SUM2: &str = r#"
G1
layout(push_constant) uniform P { F32 part; F32 o; int n, slot; float scale; } p;
shared float s[256];
void main() {
    int lid = int(gl_LocalInvocationID.x);
    float acc = 0.0;
    for (int i = lid; i < p.n; i += 256) acc += p.part.v[i];
    s[lid] = acc; barrier();
    for (int k = 128; k > 0; k >>= 1) { if (lid < k) s[lid] += s[lid + k]; barrier(); }
    if (lid == 0) p.o.v[p.slot] = s[0] * p.scale; }
"#;

/// Histogram of the per-pixel colour difference (sqrt scale), for its median.
const HIST: &str = r#"
G3
layout(push_constant) uniform P { F32 cur; F32 warped; U32 hist; int W, H, bins; float maxv; } p;
void main() { ID3 if (x >= p.W || y >= p.H) return;
    int i = y * p.W + x, n = p.W * p.H;
    if (isnan(p.warped.v[i])) return;
    float d = (abs(p.cur.v[i] - p.warped.v[i]) + abs(p.cur.v[n + i] - p.warped.v[n + i]) + abs(p.cur.v[2 * n + i] - p.warped.v[2 * n + i])) / 3.0;
    int b = min(int(sqrt(min(d / p.maxv, 1.0)) * float(p.bins)), p.bins - 1);
    atomicAdd(p.hist.v[b], 1u); }
"#;

const MEDIAN: &str = r#"
layout(local_size_x = 1) in;
layout(push_constant) uniform P { U32 hist; F32 o; int bins, slot; float maxv; } p;
void main() {
    uint total = 0u;
    for (int b = 0; b < p.bins; b++) total += p.hist.v[b];
    uint half_n = total / 2u, acc = 0u;
    float v = 1.0;
    if (total > 0u) for (int b = 0; b < p.bins; b++) {
        acc += p.hist.v[b];
        if (acc > half_n) { float t = (float(b) + 0.5) / float(p.bins); v = t * t * p.maxv; break; }
    }
    p.o.v[p.slot] = max(v, 0.002); }
"#;

/// The window merge, one pass per pixel over every neighbour (any number each way).
/// `nbs`: per slot, the moved neighbour (3 channels; NaN outside) then its agreement
/// map; slot = side * R + distance - 1. Stats: mean agreement per slot [0, 2R),
/// typical difference per slot [2R, 4R). Cut rule: a neighbour is used only if it and
/// every nearer one on its side are present and of the same shot. Weights fall off as
/// decay^(distance - 1); the last usable neighbour of a side also takes the weights of
/// those beyond it (cut or past the clip's end), so a side keeps its share. Each value
/// is clamped to the current 3x3 (variance clipping), widened by the noise.
/// The merged picture: the current one (weight 1) with the neighbours' sums, then
/// `strength` of the way from the current one to that; alpha passes through. With
/// `want_fac`, the noise left as a fraction of one picture's (variance; independent
/// noise: the current picture weighs (1 - s) + s / W, each neighbour s k / W).
const MERGE_ALL: &str = r#"
G3
layout(push_constant) uniform P { F32 cur; F32 nbs; F32 stats; F32 o; F32 fac; int C, W, H, R, present, want_fac; float cut, decay, clamp_sd, keep, reject, clamp_noise, strength; } p;
bool usable(int side, int dd) {
    for (int q = 1; q <= dd; q++) {
        int slot = side * p.R + q - 1;
        if (((p.present >> slot) & 1) == 0 || p.stats.v[slot] < p.cut) return false;
    }
    return true;
}
void main() { ID3 if (x >= p.W || y >= p.H) return;
    int i = y * p.W + x, n = p.W * p.H;
    vec3 u = vec3(p.cur.v[i], p.cur.v[n + i], p.cur.v[2 * n + i]);
    vec3 mean = vec3(0.0), sd = vec3(0.0);
    for (int c = 0; c < 3; c++) {
        float s1 = 0.0, s2 = 0.0, cnt = 0.0;
        for (int yy = max(y - 1, 0); yy <= min(y + 1, p.H - 1); yy++) for (int xx = max(x - 1, 0); xx <= min(x + 1, p.W - 1); xx++) {
            float a = p.cur.v[c * n + yy * p.W + xx];
            s1 += a; s2 += a * a; cnt += 1.0;
        }
        mean[c] = s1 / cnt; sd[c] = sqrt(max(s2 / cnt - mean[c] * mean[c], 0.0));
    }
    vec3 acc = vec3(0.0); float ks = 0.0, k2 = 0.0;
    for (int side = 0; side < 2; side++) for (int d = 1; d <= p.R; d++) {
        if (!usable(side, d)) break;
        int slot = side * p.R + d - 1, b = slot * 4 * n;
        vec3 v = vec3(p.nbs.v[b + i], p.nbs.v[b + n + i], p.nbs.v[b + 2 * n + i]);
        if (isnan(v.x)) continue;
        float w = pow(p.decay, float(d - 1));
        if (!(d < p.R && usable(side, d + 1))) for (int q = d + 1; q <= p.R; q++) w += pow(p.decay, float(q - 1));
        float dd = (abs(u.x - v.x) + abs(u.y - v.y) + abs(u.z - v.z)) / 3.0;
        float m = p.stats.v[2 * p.R + slot];
        float k = p.nbs.v[b + 3 * n + i] * w * clamp((p.reject * m - dd) / ((p.reject - p.keep) * m), 0.0, 1.0);
        if (k <= 0.0) continue;
        for (int c = 0; c < 3; c++) acc[c] += k * clamp(v[c], mean[c] - p.clamp_sd * sd[c] - p.clamp_noise * m, mean[c] + p.clamp_sd * sd[c] + p.clamp_noise * m);
        ks += k; k2 += k * k;
    }
    for (int c = 0; c < 3; c++) {
        float mm = (u[c] + acc[c]) / (1.0 + ks);
        p.o.v[c * n + i] = u[c] + p.strength * (mm - u[c]);
    }
    for (int c = 3; c < p.C; c++) p.o.v[c * n + i] = p.cur.v[c * n + i];
    if (p.want_fac != 0) {
        float wt = 1.0 + ks, a = (1.0 - p.strength) + p.strength / wt;
        p.fac.v[i] = a * a + p.strength * p.strength * k2 / (wt * wt);
    } }
"#;

/// Gaussian blur, one direction (C channels; edges repeat; NaN spreads).
const GAUSS: &str = r#"
G3
layout(push_constant) uniform P { F32 a; F32 o; int C, W, H, R, vertical; float sigma; } p;
void main() { ID3 if (x >= p.W || y >= p.H || c >= p.C) return;
    int n = p.W * p.H;
    float s = 0.0, ws = 0.0;
    for (int d = -p.R; d <= p.R; d++) {
        float w = exp(-float(d * d) / (2.0 * p.sigma * p.sigma));
        int xx = p.vertical != 0 ? x : clamp(x + d, 0, p.W - 1), yy = p.vertical != 0 ? clamp(y + d, 0, p.H - 1) : y;
        s += w * p.a.v[c * n + yy * p.W + xx];
        ws += w;
    }
    p.o.v[c * n + y * p.W + x] = s / ws; }
"#;

/// Steady Detail, start: the current picture's own detail (picture - its blur) into
/// the sums `acc` (3 colour channels, the detail's energy, the weight).
const STEADY_INIT: &str = r#"
G3
layout(push_constant) uniform P { F32 cur; F32 low; F32 acc; int W, H; } p;
void main() { ID3 if (x >= p.W || y >= p.H) return;
    int i = y * p.W + x, n = p.W * p.H;
    float e = 0.0;
    for (int k = 0; k < 3; k++) { float d = p.cur.v[k * n + i] - p.low.v[k * n + i]; p.acc.v[k * n + i] = d; e += d * d; }
    p.acc.v[3 * n + i] = e;
    p.acc.v[4 * n + i] = 1.0; }
"#;

/// Steady Detail: a neighbour moved onto the current picture (`nb`, NaN outside) and
/// its blur (`nlow`): its detail is added where its blur's luma is within `agree` of
/// the current one's (the same thing, lined up), weighted decay^(d - 1). The cut rule
/// of MERGE_ALL (stats: mean agreement per slot).
const STEADY_ACC: &str = r#"
G3
layout(push_constant) uniform P { F32 nb; F32 nlow; F32 low; F32 stats; F32 acc; int W, H, R, side, d, present; float cut, decay, agree; } p;
bool usable(int dd) {
    for (int q = 1; q <= dd; q++) {
        int slot = p.side * p.R + q - 1;
        if (((p.present >> slot) & 1) == 0 || p.stats.v[slot] < p.cut) return false;
    }
    return true;
}
float luma(F32 b, int i, int n) { return 0.299 * b.v[i] + 0.587 * b.v[n + i] + 0.114 * b.v[2 * n + i]; }
void main() { ID3 if (x >= p.W || y >= p.H) return;
    if (!usable(p.d)) return;
    int i = y * p.W + x, n = p.W * p.H;
    if (isnan(p.nb.v[i]) || isnan(p.nlow.v[i])) return;
    if (abs(luma(p.nlow, i, n) - luma(p.low, i, n)) >= p.agree) return;
    float w = pow(p.decay, float(p.d - 1)), e = 0.0;
    for (int k = 0; k < 3; k++) { float dd = p.nb.v[k * n + i] - p.nlow.v[k * n + i]; p.acc.v[k * n + i] += w * dd; e += dd * dd; }
    p.acc.v[3 * n + i] += w * e;
    p.acc.v[4 * n + i] += w; }
"#;

/// Steady Detail: the averaged detail (`hp`, 3 channels) and, for the gain, its
/// energy and the frames' mean energy (`en`, 2 channels).
const STEADY_MEAN: &str = r#"
G3
layout(push_constant) uniform P { F32 acc; F32 hp; F32 en; int W, H; } p;
void main() { ID3 if (x >= p.W || y >= p.H) return;
    int i = y * p.W + x, n = p.W * p.H;
    float ws = p.acc.v[4 * n + i], e = 0.0;
    for (int k = 0; k < 3; k++) { float v = p.acc.v[k * n + i] / ws; p.hp.v[k * n + i] = v; e += v * v; }
    p.en.v[i] = e;
    p.en.v[n + i] = p.acc.v[3 * n + i] / ws; }
"#;

/// Steady Detail, end: the current blur plus the averaged detail, brought back up to
/// the frames' mean detail strength over the box around (`ben`: 31 x 31 box means of
/// `en`; gain at most `gmax`), then `strength` of the way from the picture to that.
/// Alpha passes through.
const STEADY_END: &str = r#"
G3
layout(push_constant) uniform P { F32 cur; F32 low; F32 hp; F32 ben; F32 o; int C, W, H; float gmax, strength; } p;
void main() { ID3 if (x >= p.W || y >= p.H || c >= p.C) return;
    int i = y * p.W + x, n = p.W * p.H;
    float u = p.cur.v[c * n + i];
    if (c >= 3) { p.o.v[c * n + i] = u; return; }
    float g = clamp(sqrt(p.ben.v[n + i] / max(p.ben.v[i], 1e-10)), 1.0, p.gmax);
    float v = p.low.v[c * n + i] + g * p.hp.v[c * n + i];
    p.o.v[c * n + i] = u + p.strength * (v - u); }
"#;

/// An alpha matte steadied along the motion: the current alpha (weight 1) averaged
/// with its neighbours' (`nbs`: per slot the moved alpha, NaN outside, then the
/// picture's agreement map; slot = side * R + distance - 1), each weighted by its
/// picture's agreement and decay^(distance - 1). The same cut rule as MERGE_ALL.
/// A neighbour's alpha is kept within the range of the current alpha over the
/// (2 rad + 1)^2 around the pixel, widened by `margin`: where the current matte is
/// flat, a wrong motion cannot pull in a ghost; near edges (a wide range), the
/// shimmer is averaged out.
const ALPHA_MERGE: &str = r#"
G3
layout(push_constant) uniform P { F32 a; F32 nbs; F32 stats; F32 o; int W, H, R, present; float cut, decay, margin; int rad; } p;
bool usable(int side, int dd) {
    for (int q = 1; q <= dd; q++) {
        int slot = side * p.R + q - 1;
        if (((p.present >> slot) & 1) == 0 || p.stats.v[slot] < p.cut) return false;
    }
    return true;
}
void main() { ID3 if (x >= p.W || y >= p.H) return;
    int i = y * p.W + x, n = p.W * p.H;
    float a0 = p.a.v[i], lo = a0, hi = a0;
    for (int yy = max(y - p.rad, 0); yy <= min(y + p.rad, p.H - 1); yy++) for (int xx = max(x - p.rad, 0); xx <= min(x + p.rad, p.W - 1); xx++) {
        float v = p.a.v[yy * p.W + xx];
        lo = min(lo, v); hi = max(hi, v);
    }
    lo -= p.margin; hi += p.margin;
    float acc = a0, ws = 1.0;
    for (int side = 0; side < 2; side++) for (int d = 1; d <= p.R; d++) {
        if (!usable(side, d)) break;
        int slot = side * p.R + d - 1, b = slot * 2 * n;
        float v = p.nbs.v[b + i];
        if (isnan(v)) continue;
        float k = p.nbs.v[b + n + i] * pow(p.decay, float(d - 1));
        acc += k * clamp(v, lo, hi);
        ws += k;
    }
    p.o.v[i] = clamp(acc / ws, 0.0, 1.0); }
"#;

/// Grain left after the merge, smoothed in place of being drawn as texture: a 5x5
/// bilateral filter whose range follows the grain measured on this picture (the
/// typical difference to the nearest neighbours, stats slots 2R and 3R), scaled by
/// `k`. Differences of grain size are averaged; edges well above it are kept.
const SMOOTH: &str = r#"
G3
layout(push_constant) uniform P { F32 img; F32 stats; F32 o; int C, W, H, R, present; float k; } p;
void main() { ID3 if (x >= p.W || y >= p.H) return;
    int i = y * p.W + x, n = p.W * p.H;
    float m = 0.0, cnt = 0.0;
    for (int s = 0; s < 2; s++) if (((p.present >> (s * p.R)) & 1) != 0) { m += p.stats.v[2 * p.R + s * p.R]; cnt += 1.0; }
    float sr = p.k * (cnt > 0.0 ? m / cnt : 0.0);
    vec3 u = vec3(p.img.v[i], p.img.v[n + i], p.img.v[2 * n + i]);
    if (sr <= 0.0) { for (int c = 0; c < p.C; c++) p.o.v[c * n + i] = p.img.v[c * n + i]; return; }
    vec3 acc = vec3(0.0); float wsum = 0.0;
    for (int dy = -2; dy <= 2; dy++) for (int dx = -2; dx <= 2; dx++) {
        int yy = clamp(y + dy, 0, p.H - 1), xx = clamp(x + dx, 0, p.W - 1), j = yy * p.W + xx;
        vec3 v = vec3(p.img.v[j], p.img.v[n + j], p.img.v[2 * n + j]);
        vec3 d = v - u;
        float r2 = dot(d, d) / 3.0;
        float w = exp(-float(dx * dx + dy * dy) / 4.5 - r2 / (2.0 * sr * sr));
        acc += w * v; wsum += w;
    }
    acc /= wsum;
    p.o.v[i] = acc.x; p.o.v[n + i] = acc.y; p.o.v[2 * n + i] = acc.z;
    for (int c = 3; c < p.C; c++) p.o.v[c * n + i] = p.img.v[c * n + i]; }
"#;

/// Motion blur: blur vector per pixel (half the motion from the previous to the next
/// picture, times the shutter); either flow may be missing.
const MB_VEL: &str = r#"
G3
layout(push_constant) uniform P { F32 back; F32 fwd; F32 o; F32 bshot; F32 fshot; int W, H, BW, BH, FW, FH, have, shots; float shutter, cut; } p;
void main() { ID3 if (x >= p.W || y >= p.H) return;
    int i = y * p.W + x, n = p.W * p.H;
    // A neighbour across a scene cut (its mean agreement under the cut threshold) counts
    // as absent: its motion is meaningless.
    int have = p.have;
    if ((p.shots & 1) != 0 && p.bshot.v[0] < p.cut) have &= ~1;
    if ((p.shots & 2) != 0 && p.fshot.v[0] < p.cut) have &= ~2;
    vec2 b = vec2(0.0), f = vec2(0.0);
    if ((have & 1) != 0) { float k = float(p.BW) / float(p.W); float sx = (float(x) + 0.5) * k - 0.5, sy = (float(y) + 0.5) * k - 0.5;
        b = vec2(bilc(p.back, 0, p.BW, p.BH, sx, sy), bilc(p.back, p.BW * p.BH, p.BW, p.BH, sx, sy)) / k; }
    if ((have & 2) != 0) { float k = float(p.FW) / float(p.W); float sx = (float(x) + 0.5) * k - 0.5, sy = (float(y) + 0.5) * k - 0.5;
        f = vec2(bilc(p.fwd, 0, p.FW, p.FH, sx, sy), bilc(p.fwd, p.FW * p.FH, p.FW, p.FH, sx, sy)) / k; }
    vec2 m = have == 3 ? (f - b) * 0.5 : (have == 2 ? f : (have == 1 ? -b : vec2(0.0)));
    p.o.v[i] = m.x * p.shutter; p.o.v[n + i] = m.y * p.shutter; }
"#;

/// For GAIN: both pictures' colours where the moved one has a value (not where the
/// motion brought nothing: NaN), zero elsewhere, as 6 channels (the other, then `a`).
const GAIN_PREP: &str = r#"
G3
layout(push_constant) uniform P { F32 cur; F32 a; F32 o; int W, H; } p;
void main() { ID3 if (x >= p.W || y >= p.H) return;
    int n = p.W * p.H, i = y * p.W + x;
    bool ok = true;
    for (int c = 0; c < 3; c++) ok = ok && !isnan(p.a.v[c * n + i]) && !isinf(p.a.v[c * n + i]);
    for (int c = 0; c < 3; c++) { p.o.v[c * n + i] = ok ? p.cur.v[c * n + i] : 0.0; p.o.v[(3 + c) * n + i] = ok ? p.a.v[c * n + i] : 0.0; } }
"#;

/// Each colour channel of `a` scaled to the mean of the other picture's (means in m:
/// 0-2 the other, 3-5 `a`): a flash or a fade is not taken for a scene cut.
const GAIN: &str = r#"
G3
layout(push_constant) uniform P { F32 a; F32 m; int W, H; } p;
void main() { ID3 if (x >= p.W || y >= p.H) return;
    int n = p.W * p.H, i = y * p.W + x;
    for (int c = 0; c < 3; c++) p.a.v[c * n + i] *= clamp(p.m.v[c] / max(p.m.v[3 + c], 1e-4), 0.25, 4.0); }
"#;

/// The picture between `a` and `b` (made by frame interpolation: `x`) unless the two are
/// of different shots (stats[0], their mean agreement, under the cut threshold): then
/// the nearer of them, not a blend of two scenes.
const SHOT_PICK: &str = r#"
G3
layout(push_constant) uniform P { F32 x; F32 a; F32 b; F32 shot; F32 o; int C, W, H; float t, cut; } p;
void main() { ID3 if (x >= p.W || y >= p.H) return;
    bool cut = p.shot.v[0] < p.cut;
    for (int c = 0; c < p.C; c++) { int i = (c * p.H + y) * p.W + x;
        p.o.v[i] = cut ? (p.t < 0.5 ? p.a.v[i] : p.b.v[i]) : p.x.v[i]; } }
"#;

/// In-between pictures by splatting along the motion (`Temporal::interpolate`). Every
/// pixel of each picture is pushed forward to where it is at the in-between time,
/// weighted by how much it is to be trusted (softmax splatting): where two land on
/// one spot, the trusted one wins (the visible one: an occluded pixel matches badly
/// in the other picture and its motion fails the forward-backward check).
/// Sums are 32-bit fixed point (integer atomics work on every GPU; float atomic adds
/// do not, e.g. on AMD's integrated GPUs).
const SPLAT_COMMON: &str = r#"
layout(buffer_reference, std430, buffer_reference_align = 4) buffer SplatU { uint v[]; };
// Fixed point: weights (scale WS); motions (scale FS, offset by FOFF so they are
// positive, at most FMAX px; up to 64 full-weight contributions per pixel fit).
#define WS 4194304.0
#define FS 16384.0
#define FOFF 2048.0
#define FMAX 2000.0
// Distance (px) at which the motion is compared for boundaries (SPLAT_Z).
#define GR 3.0
// Flow sample at picture pixel (x, y), in picture pixels (flows may be smaller).
vec2 flow_at(F32 f, int FW, int FH, int W, float x, float y) {
    float k = float(FW) / float(W);
    float fx = (x + 0.5) * k - 0.5, fy = (y + 0.5) * k - 0.5;
    return vec2(bilc(f, 0, FW, FH, fx, fy), bilc(f, FW * FH, FW, FH, fx, fy)) / k;
}
"#;

/// Trust of each pixel of `a` moved by `fab` to `b`: exp(-(colour difference / SC +
/// forward-backward mismatch / SF + motion change within GR px / SG)), at least ZMIN.
const SPLAT_Z: &str = r#"
G3
layout(push_constant) uniform P { F32 a; F32 b; F32 fab; F32 fba; F32 z; int C, W, H, FW, FH; float sc, sf, zmin, sg; } p;
void main() { ID3 if (x >= p.W || y >= p.H || c > 0) return;
    vec2 d = flow_at(p.fab, p.FW, p.FH, p.W, float(x), float(y));
    float sx = float(x) + d.x, sy = float(y) + d.y;
    int n = p.W * p.H, i = y * p.W + x;
    float z = p.zmin;
    if (sx >= -0.5 && sy >= -0.5 && sx <= float(p.W) - 0.5 && sy <= float(p.H) - 0.5) {
        float e = 0.0;
        int cc = min(p.C, 3);
        for (int k = 0; k < cc; k++) e += abs(p.a.v[k * n + i] - bil(p.b, k * n, p.W, p.H, sx, sy));
        e /= float(cc);
        vec2 back = flow_at(p.fba, p.FW, p.FH, p.W, sx, sy);
        float fb = length(d + back);
        // On a motion boundary (the motion changes within GR px). The networks' motion
        // there is a blend of both sides, and the soft edge pixels just outside an
        // object get the background's: left behind as the object moves on, they drew
        // thin outlines of its edges at its other place. Untrusted, so the other
        // picture's clean pixels win.
        float g = 0.0;
        for (int k = 0; k < 4; k++) {
            vec2 o = vec2(k == 0 ? GR : k == 1 ? -GR : 0.0, k == 2 ? GR : k == 3 ? -GR : 0.0);
            g = max(g, length(flow_at(p.fab, p.FW, p.FH, p.W, float(x) + o.x, float(y) + o.y) - d));
        }
        z = max(exp(-(e / p.sc + fb / p.sf + g / p.sg)), p.zmin);
    }
    p.z.v[i] = z; }
"#;

/// Splat each pixel's motion (as from `a` to `b`: `vs` -1 for `b`'s own flow) to
/// where the pixel is at the in-between time (`s` times its flow along), weighted by
/// tw * trust, and the weight into the side's own sum. Sums: x and y motion (offset
/// by FOFF, scale FS), weight from a, from b.
const SPLAT: &str = r#"
G3
layout(push_constant) uniform P { F32 flow; F32 z; F32 acc; int W, H, FW, FH; float s, tw, vs; int side; } p;
void main() { ID3 if (x >= p.W || y >= p.H || c > 0) return;
    int n = p.W * p.H, i = y * p.W + x;
    vec2 f = flow_at(p.flow, p.FW, p.FH, p.W, float(x), float(y));
    if (!(abs(f.x) < FMAX && abs(f.y) < FMAX)) return;
    vec2 d = f * p.s;
    vec2 m = clamp(f * p.vs, -FMAX, FMAX) + FOFF;
    float px = float(x) + d.x, py = float(y) + d.y;
    float x0 = floor(px), y0 = floor(py), ax = px - x0, ay = py - y0;
    float w0 = p.tw * p.z.v[i];
    SplatU acc = SplatU(uint64_t(p.acc));
    for (int k = 0; k < 4; k++) {
        int xx = int(x0) + (k & 1), yy = int(y0) + (k >> 1);
        if (xx < 0 || yy < 0 || xx >= p.W || yy >= p.H) continue;
        float bw = ((k & 1) != 0 ? ax : 1.0 - ax) * ((k >> 1) != 0 ? ay : 1.0 - ay);
        int o = yy * p.W + xx;
        float w = w0 * bw;
        if (w < 1e-6) continue;
        atomicAdd(acc.v[o], uint(w * m.x * FS + 0.5));
        atomicAdd(acc.v[n + o], uint(w * m.y * FS + 0.5));
        atomicAdd(acc.v[(2 + p.side) * n + o], uint(w * WS + 0.5));
    } }
"#;

/// The in-between picture: each pixel's motion from the sums, then both pictures
/// sampled along it (Catmull-Rom), weighted by how much of the motion came from each
/// (what only one of them shows comes from that one). With a fallback picture
/// (`fb` 1: frame interpolation's), where the two samples disagree (the motion is
/// wrong there: small fast things) or nothing landed, the fallback takes over;
/// without, a blend of the two pictures where nothing landed. Across a scene cut,
/// the nearer picture.
const SPLAT_RESOLVE: &str = r#"
G3
layout(push_constant) uniform P { F32 acc; F32 a; F32 b; F32 shot; F32 fbk; F32 o; int C, W, H; float t, cut; int fb; float d0, d1, share, e0, e1; } p;
vec4 cr(float t) { float t2 = t * t, t3 = t2 * t;
    return vec4((-t3 + 2.0 * t2 - t), (3.0 * t3 - 5.0 * t2 + 2.0), (-3.0 * t3 + 4.0 * t2 + t), (t3 - t2)) * 0.5; }
float cubic(F32 img, int base, float sx, float sy) {
    if (sx < -0.5 || sy < -0.5 || sx > float(p.W) - 0.5 || sy > float(p.H) - 0.5) return NAN;
    float bx = floor(sx), by = floor(sy);
    vec4 kx = cr(sx - bx), ky = cr(sy - by);
    float v = 0.0;
    for (int j = 0; j < 4; j++) {
        int yy = clamp(int(by) - 1 + j, 0, p.H - 1);
        float r = 0.0;
        for (int q = 0; q < 4; q++) r += kx[q] * img.v[base + yy * p.W + clamp(int(bx) - 1 + q, 0, p.W - 1)];
        v += ky[j] * r;
    }
    return v;
}
void main() { ID3 if (x >= p.W || y >= p.H || c >= p.C) return;
    int n = p.W * p.H, i = y * p.W + x, j = c * n + i;
    SplatU acc = SplatU(uint64_t(p.acc));
    float wa = float(acc.v[2 * n + i]) / WS, wb = float(acc.v[3 * n + i]) / WS, w = wa + wb;
    float v = p.fb != 0 ? p.fbk.v[j] : mix(p.a.v[j], p.b.v[j], p.t);
    if (p.shot.v[0] < p.cut) { p.o.v[j] = p.t < 0.5 ? p.a.v[j] : p.b.v[j]; return; }
    if (w > 1e-3) {
        vec2 m = vec2(float(acc.v[i]), float(acc.v[n + i])) / FS / w - FOFF;
        vec2 pa = vec2(float(x), float(y)) - p.t * m, pb = vec2(float(x), float(y)) + (1.0 - p.t) * m;
        float va = cubic(p.a, c * n, pa.x, pa.y);
        float vb = cubic(p.b, c * n, pb.x, pb.y);
        if (isnan(va)) wa = 0.0;
        if (isnan(vb)) wb = 0.0;
        if (wa + wb > 0.0) {
            float s = ((wa > 0.0 ? wa * va : 0.0) + (wb > 0.0 ? wb * vb : 0.0)) / (wa + wb);
            float conf = 1.0;
            // Both pictures show this point: their colours along the motion must agree.
            if (p.fb != 0 && min(wa, wb) > p.share * (wa + wb)) {
                float d = 0.0;
                int cc = min(p.C, 3);
                for (int k = 0; k < cc; k++) d += abs(cubic(p.a, k * n, pa.x, pa.y) - cubic(p.b, k * n, pb.x, pb.y));
                d /= float(cc);
                conf = clamp((p.d1 - d) / (p.d1 - p.d0), 0.0, 1.0);
            }
            // Far from the fallback (a thing in one and not the other): the fallback,
            // which places whole objects more reliably than motion that missed them.
            if (p.fb != 0) {
                float d = 0.0;
                int cc = min(p.C, 3);
                for (int k = 0; k < cc; k++) {
                    float ak = cubic(p.a, k * n, pa.x, pa.y), bk = cubic(p.b, k * n, pb.x, pb.y);
                    float sk = ((wa > 0.0 ? wa * ak : 0.0) + (wb > 0.0 ? wb * bk : 0.0)) / (wa + wb);
                    d += abs(sk - p.fbk.v[k * n + i]);
                }
                d /= float(cc);
                conf = min(conf, clamp((p.e1 - d) / (p.e1 - p.e0), 0.0, 1.0));
            }
            v = mix(v, s, conf);
        }
    }
    p.o.v[j] = v; }
"#;

/// Largest vector per 16x16 tile (mode 0) or over each tile's 3x3 neighbours (mode 1).
const MB_TILE: &str = r#"
G3
layout(push_constant) uniform P { F32 a; F32 o; int W, H, TW, TH, mode; } p;
void main() { ID3 if (x >= p.TW || y >= p.TH) return;
    vec2 best = vec2(0.0);
    if (p.mode == 0) {
        for (int yy = y * 16; yy < min(y * 16 + 16, p.H); yy++) for (int xx = x * 16; xx < min(x * 16 + 16, p.W); xx++) {
            vec2 v = vec2(p.a.v[yy * p.W + xx], p.a.v[p.W * p.H + yy * p.W + xx]);
            if (dot(v, v) > dot(best, best)) best = v;
        }
    } else {
        for (int yy = max(y - 1, 0); yy <= min(y + 1, p.TH - 1); yy++) for (int xx = max(x - 1, 0); xx <= min(x + 1, p.TW - 1); xx++) {
            vec2 v = vec2(p.a.v[yy * p.TW + xx], p.a.v[p.TW * p.TH + yy * p.TW + xx]);
            if (dot(v, v) > dot(best, best)) best = v;
        }
    }
    int n = p.TW * p.TH;
    p.o.v[y * p.TW + x] = best.x; p.o.v[n + y * p.TW + x] = best.y; }
"#;

/// The gather (McGuire et al.), as MB.glsl: samples along the neighbourhood's
/// largest motion, weighted by whose motion covers the pixel (faster = nearer).
const MB_GATHER: &str = r#"
G3
layout(push_constant) uniform P { F32 cur; F32 vel; F32 neigh; F32 o; int C, W, H, TW, samples; } p;
float cone(float d, float v) { return clamp(1.0 - d / max(v, 1e-3), 0.0, 1.0); }
float cyl(float d, float v) { return 1.0 - smoothstep(0.95 * v, 1.05 * v + 1e-3, d); }
float front(float a, float b) { return clamp(1.0 + (a - b) * 0.5, 0.0, 1.0); }
void main() { ID3 if (x >= p.W || y >= p.H) return;
    int i = y * p.W + x, n = p.W * p.H, tn = p.TW * ((p.H + 15) / 16);
    int ti = (y / 16) * p.TW + x / 16;
    vec2 vn = vec2(p.neigh.v[ti], p.neigh.v[tn + ti]);
    if (dot(vn, vn) < 0.25) { for (int c2 = 0; c2 < p.C; c2++) p.o.v[c2 * n + i] = p.cur.v[c2 * n + i]; return; }
    vec2 vx = vec2(p.vel.v[i], p.vel.v[n + i]);
    float lx = length(vx);
    float jitter = float(((uint(x) * 73856093u) ^ (uint(y) * 19349663u)) % 1024u) / 1024.0 - 0.5;
    float wsum = 1.0 / max(lx, 1.0);
    vec3 sum = vec3(p.cur.v[i], p.cur.v[n + i], p.cur.v[2 * n + i]) * wsum;
    for (int s = 0; s < p.samples; s++) {
        if (s == (p.samples - 1) / 2) continue;
        float t = -0.5 + (float(s) + jitter * 0.95 + 0.5) / float(p.samples);
        vec2 off = vn * t;
        float d = length(off);
        float px = clamp(float(x) + off.x, 0.0, float(p.W - 1)), py = clamp(float(y) + off.y, 0.0, float(p.H - 1));
        int q = int(floor(py + 0.5)) * p.W + int(floor(px + 0.5));
        float ly = length(vec2(p.vel.v[q], p.vel.v[n + q]));
        float w = front(ly, lx) * cone(d, ly * 0.5) + front(lx, ly) * cone(d, lx * 0.5) + cyl(d, ly * 0.5) * cyl(d, lx * 0.5) * 2.0;
        if (w <= 0.0) continue;
        sum += w * vec3(bilc(p.cur, 0, p.W, p.H, px, py), bilc(p.cur, n, p.W, p.H, px, py), bilc(p.cur, 2 * n, p.W, p.H, px, py));
        wsum += w;
    }
    vec3 r = sum / wsum;
    p.o.v[i] = r.x; p.o.v[n + i] = r.y; p.o.v[2 * n + i] = r.z;
    for (int c2 = 3; c2 < p.C; c2++) p.o.v[c2 * n + i] = p.cur.v[c2 * n + i]; }
"#;

/// Background removal output: 0 transparent (foreground + alpha), 1 green, 2 blue,
/// 3 the matte; existing transparency multiplies in.
const MATTE: &str = r#"
G3
layout(push_constant) uniform P { F32 m; F32 src; F32 o; int W, H, mode, src_alpha, oc; } p;
void main() { ID3 if (x >= p.W || y >= p.H || c >= p.oc) return;
    int i = y * p.W + x, n = p.W * p.H;
    float a = p.m.v[3 * n + i];
    if (p.src_alpha != 0) a *= p.src.v[3 * n + i];
    float v;
    if (p.mode == 0) v = c < 3 ? p.m.v[c * n + i] : a;
    else if (p.mode == 3) v = a;
    else { vec3 bg = p.mode == 1 ? vec3(0.0, 1.0, 0.0) : vec3(0.0, 0.0, 1.0); v = p.m.v[c * n + i] * a + bg[c] * (1.0 - a); }
    p.o.v[c * n + i] = v; }
"#;

/// Storage image (RGBA) <-> tensor.
const IMG_TO_TENSOR: &str = r#"
G3
layout(set = 0, binding = 0, rgba16f) uniform readonly image2D img;
layout(push_constant) uniform P { F32 o; int C, W, H; } p;
void main() { ID3 if (x >= p.W || y >= p.H) return;
    vec4 v = imageLoad(img, ivec2(x, y));
    int i = y * p.W + x, n = p.W * p.H;
    for (int k = 0; k < p.C; k++) p.o.v[k * n + i] = v[k]; }
"#;

const TENSOR_TO_IMG: &str = r#"
G3
layout(set = 0, binding = 0, rgba16f) uniform writeonly image2D img;
layout(push_constant) uniform P { F32 a; int C, W, H; } p;
void main() { ID3 if (x >= p.W || y >= p.H) return;
    int i = y * p.W + x, n = p.W * p.H;
    vec4 v = vec4(p.a.v[i], p.a.v[n + i], p.a.v[2 * n + i], p.C > 3 ? p.a.v[3 * n + i] : 1.0);
    imageStore(img, ivec2(x, y), v); }
"#;

/// A picture from one working colour space to another (DESIGN.md §7a; the maths is
/// brp-render's `space.glsl`, as in the renderer's chain). Alpha is kept.
const SPACE_HEAD: &str = r#"
G3
layout(push_constant) uniform P { F32 a; F32 o; int C, W, H, from, to, transfer, src2020; float peak; } p;
"#;
const SPACE_MAIN: &str = r#"
void main() { ID3 if (x >= p.W || y >= p.H) return;
    int i = y * p.W + x, n = p.W * p.H;
    vec3 v = vec3(p.a.v[i], p.a.v[n + i], p.a.v[2 * n + i]);
    v = brp_convert(p.from, p.to, v, p.transfer, p.src2020, p.peak);
    p.o.v[i] = v.r; p.o.v[n + i] = v.g; p.o.v[2 * n + i] = v.b;
    for (int k = 3; k < p.C; k++) p.o.v[k * n + i] = p.a.v[k * n + i]; }
"#;

pub struct Temporal {
    ops: Arc<Ops>,
    space_k: Arc<Kernel>,
    luma: Arc<Kernel>,
    half: Arc<Kernel>,
    cand: Arc<Kernel>,
    noise_hist: Arc<Kernel>,
    upflow: Arc<Kernel>,
    lk_warp: Arc<Kernel>,
    lk_terms: Arc<Kernel>,
    boxk: Arc<Kernel>,
    lk_solve: Arc<Kernel>,
    fill_prep: Arc<Kernel>,
    fill_apply: Arc<Kernel>,
    chain_k: Arc<Kernel>,
    warp_k: Arc<Kernel>,
    agree_prep: Arc<Kernel>,
    agree_k: Arc<Kernel>,
    sum1: Arc<Kernel>,
    sum2: Arc<Kernel>,
    hist: Arc<Kernel>,
    median: Arc<Kernel>,
    merge_all: Arc<Kernel>,
    gauss: Arc<Kernel>,
    steady_init: Arc<Kernel>,
    steady_acc: Arc<Kernel>,
    steady_mean: Arc<Kernel>,
    steady_end: Arc<Kernel>,
    alpha_merge: Arc<Kernel>,
    smooth_k: Arc<Kernel>,
    mb_vel: Arc<Kernel>,
    shot_pick: Arc<Kernel>,
    splat_z: Arc<Kernel>,
    splat: Arc<Kernel>,
    splat_resolve: Arc<Kernel>,
    gain_k: Arc<Kernel>,
    gain_prep: Arc<Kernel>,
    mb_tile: Arc<Kernel>,
    mb_gather: Arc<Kernel>,
    matte_k: Arc<Kernel>,
    img_to_tensor: Arc<Kernel>,
    tensor_to_img: Arc<Kernel>,
}

fn g3(k: &Kernel, w: usize, h: usize, c: usize) -> [u32; 3] {
    Recorder::groups(k, [w as u32, h as u32, c as u32])
}

fn flat(k: &Kernel, n: usize) -> [u32; 3] {
    let groups = n.div_ceil(k.local[0] as usize).max(1);
    let x = groups.min(65535);
    [x as u32, groups.div_ceil(x) as u32, 1]
}

impl Temporal {
    pub fn new(ops: &Arc<Ops>) -> Result<Temporal> {
        let c = &ops.compute;
        let k = |name: &str, body: &str, images: u32| c.kernel(&format!("tmp_{name}"), &format!("{PRELUDE}{COMMON}{body}"), images);
        Ok(Temporal {
            ops: ops.clone(),
            luma: k("luma", LUMA, 0)?,
            half: k("half", HALF, 0)?,
            cand: k("cand", CAND, 0)?,
            noise_hist: k("noise_hist", NOISE_HIST, 0)?,
            upflow: k("upflow", UPFLOW, 0)?,
            lk_warp: k("lk_warp", LK_WARP, 0)?,
            lk_terms: k("lk_terms", LK_TERMS, 0)?,
            boxk: k("box", BOX, 0)?,
            lk_solve: k("lk_solve", LK_SOLVE, 0)?,
            fill_prep: k("fill_prep", FILL_PREP, 0)?,
            fill_apply: k("fill_apply", FILL_APPLY, 0)?,
            chain_k: k("chain", CHAIN, 0)?,
            warp_k: k("warp", WARP, 0)?,
            agree_prep: k("agree_prep", AGREE_PREP, 0)?,
            agree_k: k("agree", AGREE, 0)?,
            sum1: k("sum1", SUM1, 0)?,
            sum2: k("sum2", SUM2, 0)?,
            hist: k("hist", HIST, 0)?,
            median: k("median", MEDIAN, 0)?,
            merge_all: k("merge_all", MERGE_ALL, 0)?,
            gauss: k("gauss", GAUSS, 0)?,
            steady_init: k("steady_init", STEADY_INIT, 0)?,
            steady_acc: k("steady_acc", STEADY_ACC, 0)?,
            steady_mean: k("steady_mean", STEADY_MEAN, 0)?,
            steady_end: k("steady_end", STEADY_END, 0)?,
            alpha_merge: k("alpha_merge", ALPHA_MERGE, 0)?,
            smooth_k: k("smooth", SMOOTH, 0)?,
            mb_vel: k("mb_vel", MB_VEL, 0)?,
            shot_pick: k("shot_pick", SHOT_PICK, 0)?,
            splat_z: k("splat_z", &format!("{SPLAT_COMMON}{SPLAT_Z}"), 0)?,
            splat: k("splat", &format!("{SPLAT_COMMON}{SPLAT}"), 0)?,
            splat_resolve: k("splat_resolve", &format!("{SPLAT_COMMON}{SPLAT_RESOLVE}"), 0)?,
            gain_k: k("gain", GAIN, 0)?,
            gain_prep: k("gain_prep", GAIN_PREP, 0)?,
            mb_tile: k("mb_tile", MB_TILE, 0)?,
            mb_gather: k("mb_gather", MB_GATHER, 0)?,
            matte_k: k("matte", MATTE, 0)?,
            img_to_tensor: k("img_to_tensor", IMG_TO_TENSOR, 1)?,
            tensor_to_img: k("tensor_to_img", TENSOR_TO_IMG, 1)?,
            space_k: k("space", &format!("{SPACE_HEAD}{}{SPACE_MAIN}", brp_render::workspace::SPACE_GLSL), 0)?,
        })
    }

    fn t(&self, c: usize, h: usize, w: usize) -> Result<Tensor> {
        self.ops.tensor(c, h, w)
    }

    fn recycle(&self, rec: &mut Recorder, t: Tensor) {
        self.ops.recycle(rec, t)
    }

    /// Window sum (or mean) over (2r+1)^2, separable.
    fn box_sum(&self, rec: &mut Recorder, a: &Tensor, r: usize, mean: bool) -> Result<Tensor> {
        let tmp = self.t(a.c, a.h, a.w)?;
        let out = self.t(a.c, a.h, a.w)?;
        for (src, dst, vertical) in [(a, &tmp, 0), (&tmp, &out, 1)] {
            let push = Push::new().t(src).t(dst).i(a.c as i32).i(a.w as i32).i(a.h as i32).i(r as i32).i(vertical).i(mean as i32);
            rec.dispatch(&self.boxk, g3(&self.boxk, a.w, a.h, a.c), &push)?;
        }
        self.recycle(rec, tmp);
        Ok(out)
    }

    fn lk_step(&self, rec: &mut Recorder, flow: &Tensor, lp: &Tensor, lc: &Tensor, noise: &Tensor, lk: [f32; 2]) -> Result<Tensor> {
        let (w, h) = (lp.w, lp.h);
        let warped = self.t(1, h, w)?;
        rec.dispatch(&self.lk_warp, g3(&self.lk_warp, w, h, 1), &Push::new().t(lp).t(flow).t(&warped).i(w as i32).i(h as i32))?;
        let terms = self.t(5, h, w)?;
        rec.dispatch(&self.lk_terms, g3(&self.lk_terms, w, h, 1), &Push::new().t(&warped).t(lc).t(&terms).i(w as i32).i(h as i32))?;
        let sums = self.box_sum(rec, &terms, 3, false)?;
        let structure = self.t(3, h, w)?;
        rec.dispatch(&self.lk_solve, g3(&self.lk_solve, w, h, 1), &Push::new().t(&sums).t(flow).t(&structure).t(noise).i(w as i32).i(h as i32).f(lk[0]).f(lk[1]))?;
        for x in [warped, terms, sums] {
            self.recycle(rec, x);
        }
        Ok(structure)
    }

    /// Candidate motions (see `CAND`): `flow` replaced by the best per pixel.
    fn candidates(&self, rec: &mut Recorder, flow: &mut Tensor, lp: &Tensor, lc: &Tensor, hint: Option<&Tensor>) -> Result<()> {
        let (w, h) = (lp.w, lp.h);
        let o = self.t(2, h, w)?;
        let ht = hint.unwrap_or(flow);
        let push = Push::new().t(lp).t(lc).t(flow).t(&o).t(ht).i(w as i32).i(h as i32).f(CAND_MARGIN).i(ht.w as i32).i(ht.h as i32).i(hint.is_some() as i32);
        rec.dispatch(&self.cand, g3(&self.cand, w, h, 1), &push)?;
        let old = std::mem::replace(flow, o);
        self.recycle(rec, old);
        Ok(())
    }

    fn fill(&self, rec: &mut Recorder, flow: &Tensor, structure: &Tensor, radius: usize, smooth_all: bool) -> Result<()> {
        let (w, h) = (flow.w, flow.h);
        let prep = self.t(5, h, w)?;
        let floor_w = if smooth_all { 1e-3 } else { 0.0 };
        rec.dispatch(&self.fill_prep, g3(&self.fill_prep, w, h, 1), &Push::new().t(flow).t(structure).t(&prep).i(w as i32).i(h as i32).f(STRUCTURE).f(floor_w))?;
        let sums = self.box_sum(rec, &prep, radius, false)?;
        rec.dispatch(&self.fill_apply, g3(&self.fill_apply, w, h, 1), &Push::new().t(flow).t(structure).t(&sums).i(w as i32).i(h as i32).f(STRUCTURE).f(if smooth_all { 0.0 } else { ((2 * radius + 1) * (2 * radius + 1)) as f32 }))?;
        self.recycle(rec, prep);
        self.recycle(rec, sums);
        Ok(())
    }

    /// Motion between two pictures: pixel (x, y) of `cur` is at (x + dx, y + dy) in
    /// `prev`. 2 channels, at their size or half of it (wide pictures).
    pub fn motion(&self, rec: &mut Recorder, prev: &Tensor, cur: &Tensor) -> Result<Tensor> {
        self.motion_hinted(rec, prev, cur, None)
    }

    /// `motion`, also trying `hint` (the motion of a neighbouring pair of pictures, the
    /// same way round) as a candidate everywhere: motion changes little from one pair
    /// to the next, and two estimates beat one where grain hides the detail.
    pub fn motion_hinted(&self, rec: &mut Recorder, prev: &Tensor, cur: &Tensor, hint: Option<&Tensor>) -> Result<Tensor> {
        let (w, h) = (cur.w, cur.h);
        let label = std::mem::replace(&mut rec.label, "motion".into());
        let r = self.motion_inner(rec, prev, cur, hint, w, h);
        rec.label = label;
        r
    }

    fn motion_inner(&self, rec: &mut Recorder, prev: &Tensor, cur: &Tensor, hint: Option<&Tensor>, w: usize, h: usize) -> Result<Tensor> {
        let luma = |img: &Tensor, rec: &mut Recorder| -> Result<Tensor> {
            let o = self.t(1, h, w)?;
            rec.dispatch(&self.luma, flat(&self.luma, w * h), &Push::new().t(img).t(&o).i((w * h) as i32))?;
            Ok(o)
        };
        let mut pyr = vec![(luma(prev, rec)?, luma(cur, rec)?)];
        while pyr.len() < 5 {
            let (lp, _) = pyr.last().unwrap();
            if lp.w < 64 || lp.h < 64 {
                break;
            }
            let (ow, oh) = (lp.w / 2, lp.h / 2);
            let mut next = Vec::new();
            for src in [&pyr.last().unwrap().0, &pyr.last().unwrap().1] {
                let o = self.t(1, oh, ow)?;
                rec.dispatch(&self.half, g3(&self.half, ow, oh, 1), &Push::new().t(src).t(&o).i(src.w as i32).i(ow as i32).i(oh as i32).i(src.h as i32))?;
                next.push(o);
            }
            let b = next.pop().unwrap();
            let a = next.pop().unwrap();
            pyr.push((a, b));
        }
        // Grain level (median high-pass of the current luma), for the solve.
        let noise = self.ops.filled(rec, 1, 1, 1, 0.0)?;
        {
            let l0 = &pyr[0].1;
            let hist = self.ops.filled(rec, HIST_BINS, 1, 1, 0.0)?;
            rec.dispatch(&self.noise_hist, g3(&self.noise_hist, l0.w, l0.h, 1), &Push::new().t(l0).t(&hist).i(l0.w as i32).i(l0.h as i32).i(HIST_BINS as i32).f(HIST_MAX))?;
            rec.dispatch(&self.median, [1, 1, 1], &Push::new().t(&hist).t(&noise).i(HIST_BINS as i32).i(0).f(HIST_MAX))?;
            self.recycle(rec, hist);
        }
        let top = if w > MOTION_MAX_W && pyr.len() > 2 { 1 } else { 0 };
        let fill_level = top + (pyr.len() > top + 1) as usize;
        let (cw, ch) = (pyr.last().unwrap().0.w, pyr.last().unwrap().0.h);
        let mut flow = self.ops.filled(rec, 2, ch, cw, 0.0)?;
        for level in (top..pyr.len()).rev() {
            let (lp, lc) = (&pyr[level].0, &pyr[level].1);
            if (flow.w, flow.h) != (lp.w, lp.h) {
                let up = self.t(2, lp.h, lp.w)?;
                rec.dispatch(&self.upflow, g3(&self.upflow, lp.w, lp.h, 2), &Push::new().t(&flow).t(&up).i(flow.w as i32).i(flow.h as i32).i(lp.w as i32).i(lp.h as i32))?;
                let old = std::mem::replace(&mut flow, up);
                self.recycle(rec, old);
            }
            let finest = level == top;
            // 7x7 window, grain variance from the median (0.754 sigma), less per halving.
            let lk = GRAIN_REG * 49.0 / (0.754f32 * 0.754) * 0.0748f32.powi(level as i32);
            let lk = [lk, if finest { 1e-9 } else { COARSE_FLOOR }];
            // Twice: a good motion spreads 4-12 px per pass.
            for _ in 0..2 {
                self.candidates(rec, &mut flow, lp, lc, hint)?;
            }
            let iterations = if finest && fill_level > top { 1 } else { 3 };
            let mut structure = None;
            for _ in 0..iterations {
                if let Some(s) = structure.take() {
                    self.recycle(rec, s);
                }
                structure = Some(self.lk_step(rec, &flow, lp, lc, &noise, lk)?);
            }
            let mut structure = structure.unwrap();
            if level == fill_level {
                self.fill(rec, &flow, &structure, FILL_RADIUS >> level, false)?;
                if finest {
                    let s2 = self.lk_step(rec, &flow, lp, lc, &noise, lk)?;
                    self.recycle(rec, std::mem::replace(&mut structure, s2));
                }
            }
            if finest {
                self.fill(rec, &flow, &structure, SMOOTH_RADIUS >> top, true)?;
                self.fill(rec, &flow, &structure, WIDE_RADIUS, false)?;
            }
            self.recycle(rec, structure);
        }
        for (a, b) in pyr {
            self.recycle(rec, a);
            self.recycle(rec, b);
        }
        self.recycle(rec, noise);
        Ok(flow)
    }

    /// Motion `a` then `b` (b may be at another resolution).
    pub fn chain(&self, rec: &mut Recorder, a: &Tensor, b: &Tensor) -> Result<Tensor> {
        let o = self.t(2, a.h, a.w)?;
        rec.dispatch(&self.chain_k, g3(&self.chain_k, a.w, a.h, 1), &Push::new().t(a).t(b).t(&o).i(a.w as i32).i(a.h as i32).i(b.w as i32).i(b.h as i32))?;
        Ok(o)
    }

    /// `img`'s colour moved along `flow` into `dst` (3 channels; NaN outside).
    pub fn warp_into(&self, rec: &mut Recorder, img: &Tensor, flow: &Tensor, dst: &Tensor) -> Result<()> {
        self.warp_channels_into(rec, img, 3, flow, dst)
    }

    /// The first `c` channels of `img` moved along `flow` into `dst` (NaN outside).
    pub fn warp_channels_into(&self, rec: &mut Recorder, img: &Tensor, c: usize, flow: &Tensor, dst: &Tensor) -> Result<()> {
        let push = Push::new().t(img).t(flow).t(dst).i(c as i32).i(img.w as i32).i(img.h as i32).i(flow.w as i32).i(flow.h as i32);
        rec.dispatch(&self.warp_k, g3(&self.warp_k, img.w, img.h, c), &push)
    }

    /// The first `c` channels of `a` blurred (Gaussian, `sigma` pixels).
    pub fn gauss(&self, rec: &mut Recorder, a: &Tensor, c: usize, sigma: f32) -> Result<Tensor> {
        let r = (3.0 * sigma).ceil().max(1.0) as i32;
        let tmp = self.t(c, a.h, a.w)?;
        let out = self.t(c, a.h, a.w)?;
        for (src, dst, v) in [(a, &tmp, 0), (&tmp, &out, 1)] {
            let push = Push::new().t(src).t(dst).i(c as i32).i(a.w as i32).i(a.h as i32).i(r).i(v).f(sigma);
            rec.dispatch(&self.gauss, g3(&self.gauss, a.w, a.h, c), &push)?;
        }
        self.recycle(rec, tmp);
        Ok(out)
    }

    /// Steady Detail: detail that comes and goes from frame to frame (aliasing,
    /// compression, sharpening that changes per frame) averaged with the neighbours'
    /// along the motion; the broad picture (blurred by `sigma`) stays this frame's.
    /// `sides` as `window`. `keep` (0..1): how far the detail lost to averaging is
    /// brought back (a gain up to 1 + keep, towards the frames' mean detail strength
    /// over a 31 x 31 box; it rarely needs more than 1.5); `strength`: mix with the
    /// picture.
    #[allow(clippy::too_many_arguments)]
    pub fn steady_detail(&self, rec: &mut Recorder, cur: &Tensor, sides: [&[(&Tensor, &Tensor)]; 2], sigma: f32, keep: f32, strength: f32, decay: f32) -> Result<Tensor> {
        let (w, h) = (cur.w, cur.h);
        let r = sides[0].len().max(sides[1].len());
        let mut flows: [Vec<Tensor>; 2] = [Vec::new(), Vec::new()];
        for (s, list) in sides.iter().enumerate() {
            for (d, (_, step)) in list.iter().enumerate() {
                let f = if d == 0 { (*step).clone() } else { self.chain(rec, &flows[s][d - 1], step)? };
                flows[s].push(f);
            }
        }
        rec.label = "steady detail".into();
        let low = self.gauss(rec, cur, 3, sigma)?;
        let acc = self.t(5, h, w)?;
        rec.dispatch(&self.steady_init, g3(&self.steady_init, w, h, 1), &Push::new().t(cur).t(&low).t(&acc).i(w as i32).i(h as i32))?;
        let stats = self.ops.filled(rec, 2 * r.max(1), 1, 1, 0.0)?;
        let warped = self.t(3, h, w)?;
        let agree = self.t(1, h, w)?;
        let mut present = 0;
        for (s, list) in sides.iter().enumerate() {
            for (d, _) in list.iter().enumerate() {
                present |= 1 << (s * r + d);
            }
        }
        for (s, list) in sides.iter().enumerate() {
            for (d, (img, _)) in list.iter().enumerate() {
                let slot = s * r + d;
                self.warp_into(rec, img, &flows[s][d], &warped)?;
                self.agreement_into(rec, cur, &warped, &agree, &stats, slot)?;
                let nlow = self.gauss(rec, &warped, 3, sigma)?;
                let push = Push::new().t(&warped).t(&nlow).t(&low).t(&stats).t(&acc).i(w as i32).i(h as i32).i(r as i32).i(s as i32).i(d as i32 + 1).i(present).f(CUT).f(decay).f(STEADY_AGREE);
                rec.dispatch(&self.steady_acc, g3(&self.steady_acc, w, h, 1), &push)?;
                self.recycle(rec, nlow);
            }
        }
        let hp = self.t(3, h, w)?;
        let en = self.t(2, h, w)?;
        rec.dispatch(&self.steady_mean, g3(&self.steady_mean, w, h, 1), &Push::new().t(&acc).t(&hp).t(&en).i(w as i32).i(h as i32))?;
        let ben = self.box_sum(rec, &en, 15, true)?;
        let out = self.t(cur.c, h, w)?;
        let push = Push::new().t(cur).t(&low).t(&hp).t(&ben).t(&out).i(cur.c as i32).i(w as i32).i(h as i32).f(1.0 + keep.clamp(0.0, 1.0)).f(strength.clamp(0.0, 1.0));
        rec.dispatch(&self.steady_end, g3(&self.steady_end, w, h, cur.c), &push)?;
        for x in [low, acc, stats, warped, agree, hp, en, ben] {
            self.recycle(rec, x);
        }
        for list in flows {
            for f in list.into_iter().skip(1) {
                self.recycle(rec, f);
            }
        }
        Ok(out)
    }

    /// The alpha matte `alpha` of `cur` steadied with its neighbours' (ALPHA_MERGE).
    /// `sides`: before and after, each the pictures 1, 2, ... away with their alphas
    /// and the motion to each from the one a step nearer (as `window`).
    #[allow(clippy::too_many_arguments)]
    pub fn alpha_window(&self, rec: &mut Recorder, cur: &Tensor, alpha: &Tensor, sides: [&[(&Tensor, &Tensor, &Tensor)]; 2], decay: f32, margin: f32, rad: usize) -> Result<Tensor> {
        let (w, h) = (cur.w, cur.h);
        let r = sides[0].len().max(sides[1].len());
        let out = self.t(1, h, w)?;
        if r == 0 {
            self.ops.copy(rec, alpha, (0, 0, 0), (1, h, w), &out, (0, 0, 0), false, false)?;
            return Ok(out);
        }
        let mut flows: [Vec<Tensor>; 2] = [Vec::new(), Vec::new()];
        for (s, list) in sides.iter().enumerate() {
            for (d, (_, _, step)) in list.iter().enumerate() {
                let f = if d == 0 { (*step).clone() } else { self.chain(rec, &flows[s][d - 1], step)? };
                flows[s].push(f);
            }
        }
        rec.label = "matte: steady".into();
        let stats = self.ops.filled(rec, 4 * r, 1, 1, 0.0)?;
        let nbs = self.t(2 * 2 * r, h, w)?;
        let warped = self.t(3, h, w)?;
        let mut present = 0;
        for (s, list) in sides.iter().enumerate() {
            for (d, (img, a, _)) in list.iter().enumerate() {
                let slot = s * r + d;
                self.warp_into(rec, img, &flows[s][d], &warped)?;
                self.agreement_into(rec, cur, &warped, &nbs.view(slot * 2 + 1, 1), &stats, slot)?;
                self.warp_channels_into(rec, a, 1, &flows[s][d], &nbs.view(slot * 2, 1))?;
                present |= 1 << slot;
            }
        }
        let push = Push::new().t(alpha).t(&nbs).t(&stats).t(&out).i(w as i32).i(h as i32).i(r as i32).i(present).f(CUT).f(decay).f(margin).i(rad as i32);
        rec.dispatch(&self.alpha_merge, g3(&self.alpha_merge, w, h, 1), &push)?;
        for x in [stats, nbs, warped] {
            self.recycle(rec, x);
        }
        for list in flows {
            for f in list.into_iter().skip(1) {
                self.recycle(rec, f);
            }
        }
        Ok(out)
    }

    /// Agreement of a moved neighbour with `cur` into `dst` (1 channel), and its mean
    /// into stats[slot].
    pub fn agreement_into(&self, rec: &mut Recorder, cur: &Tensor, warped: &Tensor, dst: &Tensor, stats: &Tensor, slot: usize) -> Result<()> {
        let (w, h) = (cur.w, cur.h);
        let prep = self.t(4, h, w)?;
        rec.dispatch(&self.agree_prep, g3(&self.agree_prep, w, h, 1), &Push::new().t(cur).t(warped).t(&prep).i(w as i32).i(h as i32))?;
        let sums = self.box_sum(rec, &prep, 2, false)?;
        rec.dispatch(&self.agree_k, g3(&self.agree_k, w, h, 1), &Push::new().t(&sums).t(dst).i(w as i32).i(h as i32).f(NOISE).f(TRUST))?;
        self.recycle(rec, prep);
        self.recycle(rec, sums);
        self.mean_into(rec, dst, w * h, stats, slot)
    }

    fn mean_into(&self, rec: &mut Recorder, a: &Tensor, n: usize, out: &Tensor, slot: usize) -> Result<()> {
        let groups = n.div_ceil(256).clamp(1, 1024);
        let part = self.t(groups, 1, 1)?;
        let g = [groups.min(65535) as u32, groups.div_ceil(65535) as u32, 1];
        rec.dispatch(&self.sum1, g, &Push::new().t(a).t(&part).i(n as i32))?;
        rec.dispatch(&self.sum2, [1, 1, 1], &Push::new().t(&part).t(out).i(groups as i32).i(slot as i32).f(1.0 / n.max(1) as f32))?;
        self.recycle(rec, part);
        Ok(())
    }

    /// The typical per-pixel colour difference of a moved neighbour (its median)
    /// into stats[slot].
    pub fn typical_into(&self, rec: &mut Recorder, cur: &Tensor, warped: &Tensor, stats: &Tensor, slot: usize) -> Result<()> {
        let (w, h) = (cur.w, cur.h);
        let hist = self.ops.filled(rec, HIST_BINS, 1, 1, 0.0)?;
        let push = Push::new().t(cur).t(warped).t(&hist).i(w as i32).i(h as i32).i(HIST_BINS as i32).f(HIST_MAX);
        rec.dispatch(&self.hist, g3(&self.hist, w, h, 1), &push)?;
        rec.dispatch(&self.median, [1, 1, 1], &Push::new().t(&hist).t(stats).i(HIST_BINS as i32).i(slot as i32).f(HIST_MAX))?;
        self.recycle(rec, hist);
        Ok(())
    }

    /// The merge of `cur` with its neighbours. `sides`: before and after, each the
    /// pictures 1, 2, ... away with the motion to each from the one a step nearer
    /// (from `cur` for the first); a side stops at the clip's ends. `decay`: weight
    /// of each picture relative to the one a step nearer.
    pub fn window(&self, rec: &mut Recorder, cur: &Tensor, sides: [&[(&Tensor, &Tensor)]; 2], strength: f32, decay: f32, smooth: f32) -> Result<Tensor> {
        self.merge(rec, cur, sides, strength, decay, smooth, None)
    }

    /// The merge, and the noise left at each pixel into `fac` (1 channel): its
    /// variance as a fraction of one picture's.
    pub fn window_with_noise(&self, rec: &mut Recorder, cur: &Tensor, sides: [&[(&Tensor, &Tensor)]; 2], strength: f32, decay: f32, fac: &Tensor) -> Result<Tensor> {
        self.merge(rec, cur, sides, strength, decay, 0.0, Some(fac))
    }

    #[allow(clippy::too_many_arguments)]
    fn merge(&self, rec: &mut Recorder, cur: &Tensor, sides: [&[(&Tensor, &Tensor)]; 2], strength: f32, decay: f32, smooth: f32, fac: Option<&Tensor>) -> Result<Tensor> {
        let (w, h) = (cur.w, cur.h);
        let r = sides[0].len().max(sides[1].len());
        let out = self.t(cur.c, h, w)?;
        if r == 0 {
            self.ops.copy(rec, cur, (0, 0, 0), (cur.c, h, w), &out, (0, 0, 0), false, false)?;
            if let Some(f) = fac {
                // Nothing merged: all of one picture's noise.
                let one = self.ops.filled(rec, 1, h, w, 1.0)?;
                self.ops.copy(rec, &one, (0, 0, 0), (1, h, w), f, (0, 0, 0), false, false)?;
                self.recycle(rec, one);
            }
            return Ok(out);
        }
        // Motions from `cur` to each neighbour: steps chained outward.
        let mut flows: [Vec<Tensor>; 2] = [Vec::new(), Vec::new()];
        for (s, list) in sides.iter().enumerate() {
            for (d, (_, step)) in list.iter().enumerate() {
                let f = if d == 0 { (*step).clone() } else { self.chain(rec, &flows[s][d - 1], step)? };
                flows[s].push(f);
            }
        }
        let stats = self.ops.filled(rec, 4 * r, 1, 1, 0.0)?;
        let mut present = 0;
        rec.label = "merge: measure".into();
        // Pass 1: every neighbour moved, its agreement and typical difference (the
        // cut rule needs those of a side's nearer ones; the weights, of its
        // farther ones), all kept for pass 2.
        let nbs = self.t(4 * 2 * r, h, w)?;
        for (s, list) in sides.iter().enumerate() {
            for (d, (img, _)) in list.iter().enumerate() {
                let slot = s * r + d;
                let warped = nbs.view(slot * 4, 3);
                self.warp_into(rec, img, &flows[s][d], &warped)?;
                self.agreement_into(rec, cur, &warped, &nbs.view(slot * 4 + 3, 1), &stats, slot)?;
                self.typical_into(rec, cur, &warped, &stats, 2 * r + slot)?;
                present |= 1 << slot;
            }
        }
        rec.label = "merge: add".into();
        // Pass 2: all added in and finished, one pass per pixel.
        let push = Push::new()
            .t(cur)
            .t(&nbs)
            .t(&stats)
            .t(&out)
            .t(fac.unwrap_or(&out))
            .i(cur.c as i32)
            .i(w as i32)
            .i(h as i32)
            .i(r as i32)
            .i(present)
            .i(fac.is_some() as i32)
            .f(CUT)
            .f(decay)
            .f(CLAMP_SD)
            .f(PIXEL_KEEP)
            .f(PIXEL_REJECT)
            .f(CLAMP_NOISE)
            .f(strength);
        rec.dispatch(&self.merge_all, g3(&self.merge_all, w, h, 1), &push)?;
        self.recycle(rec, nbs);
        let out = self.smooth(rec, out, &stats, r, present, smooth)?;
        self.recycle(rec, stats);
        for list in flows {
            // The first of each side is the caller's.
            for f in list.into_iter().skip(1) {
                self.recycle(rec, f);
            }
        }
        Ok(out)
    }

    /// Leftover grain smoothed (SMOOTH), if `smooth` > 0.
    fn smooth(&self, rec: &mut Recorder, out: Tensor, stats: &Tensor, r: usize, present: i32, smooth: f32) -> Result<Tensor> {
        if smooth <= 0.0 {
            return Ok(out);
        }
        let (c, w, h) = (out.c, out.w, out.h);
        let o2 = self.t(c, h, w)?;
        let push = Push::new().t(&out).t(stats).t(&o2).i(c as i32).i(w as i32).i(h as i32).i(r as i32).i(present).f(smooth);
        rec.dispatch(&self.smooth_k, g3(&self.smooth_k, w, h, 1), &push)?;
        self.recycle(rec, out);
        Ok(o2)
    }

    /// For tests: a neighbour's mean agreement and typical difference (2 values),
    /// as the merge measures them.
    pub fn neighbour_stats(&self, rec: &mut Recorder, cur: &Tensor, img: &Tensor, flow: &Tensor) -> Result<Tensor> {
        let (w, h) = (cur.w, cur.h);
        let stats = self.ops.filled(rec, 2, 1, 1, -1.0)?;
        let warped = self.t(3, h, w)?;
        let agree = self.t(1, h, w)?;
        self.warp_into(rec, img, flow, &warped)?;
        self.agreement_into(rec, cur, &warped, &agree, &stats, 0)?;
        self.typical_into(rec, cur, &warped, &stats, 1)?;
        self.recycle(rec, warped);
        self.recycle(rec, agree);
        Ok(stats)
    }

    /// Whether `other` is of the same shot as `cur`: its mean agreement with `cur` once
    /// moved by `flow` (the motion from `cur` to it), as the merge measures it, into a
    /// 1-value tensor. Under `CUT`: a scene cut lies between them. It stays on the GPU:
    /// the kernels that need it read it (`motion_blur`, `shot_pick`).
    pub fn same_shot(&self, rec: &mut Recorder, cur: &Tensor, other: &Tensor, flow: &Tensor) -> Result<Tensor> {
        let (w, h) = (cur.w, cur.h);
        let warped = self.t(3, h, w)?;
        self.warp_into(rec, other, flow, &warped)?;
        // Brightness and colour matched first (a mean per channel): a flash or a fade
        // changes them, a cut changes what is where.
        // Over the pixels the motion filled only (the same set in both, so the ratio of
        // the means is the gain).
        let both = self.t(6, h, w)?;
        rec.dispatch(&self.gain_prep, g3(&self.gain_prep, w, h, 1), &Push::new().t(cur).t(&warped).t(&both).i(w as i32).i(h as i32))?;
        let means = self.t(6, 1, 1)?;
        for c in 0..6 {
            self.mean_into(rec, &both.view(c, 1), w * h, &means, c)?;
        }
        self.recycle(rec, both);
        rec.dispatch(&self.gain_k, g3(&self.gain_k, w, h, 1), &Push::new().t(&warped).t(&means).i(w as i32).i(h as i32))?;
        self.recycle(rec, means);
        let agree = self.t(1, h, w)?;
        let stats = self.t(1, 1, 1)?;
        self.agreement_into(rec, cur, &warped, &agree, &stats, 0)?;
        self.recycle(rec, warped);
        self.recycle(rec, agree);
        Ok(stats)
    }

    /// `x` (a picture interpolated between `a` and `b` at `t`), or the nearer of `a` and
    /// `b` when `shot` (from `same_shot`) says a cut lies between them.
    pub fn shot_pick(&self, rec: &mut Recorder, x: &Tensor, a: &Tensor, b: &Tensor, t: f32, shot: &Tensor) -> Result<Tensor> {
        let out = self.t(x.c, x.h, x.w)?;
        let push = Push::new().t(x).t(a).t(b).t(shot).t(&out).i(x.c as i32).i(x.w as i32).i(x.h as i32).f(t).f(CUT);
        rec.dispatch(&self.shot_pick, g3(&self.shot_pick, x.w, x.h, 1), &push)?;
        Ok(out)
    }

    /// The picture at time `t` (0..1) between `a` and `b`, from their motions both ways
    /// (`fab`: where each pixel of `a` is in `b`; `fba` the reverse; any size, in
    /// their own pixels as `motion` gives them). `shot`: their `same_shot` measure
    /// (across a cut, the nearer picture). `fallback` (frame interpolation's picture)
    /// takes over where the motion proves wrong. The motions are splatted to the in-between
    /// time (SPLAT), then both pictures sampled along the result: unlike frame
    /// interpolation networks it follows the given motion, small fast things included.
    pub fn interpolate(&self, rec: &mut Recorder, a: &Tensor, b: &Tensor, fab: &Tensor, fba: &Tensor, t: f32, shot: &Tensor, fallback: Option<&Tensor>) -> Result<Tensor> {
        let (w, h, c) = (a.w, a.h, a.c);
        let mut z = Vec::new();
        for (x, y, f, r) in [(a, b, fab, fba), (b, a, fba, fab)] {
            let o = self.t(1, h, w)?;
            let push = Push::new().t(x).t(y).t(f).t(r).t(&o).i(c as i32).i(w as i32).i(h as i32).i(f.w as i32).i(f.h as i32).f(SPLAT_SC).f(SPLAT_SF).f(SPLAT_ZMIN).f(SPLAT_SG);
            rec.dispatch(&self.splat_z, g3(&self.splat_z, w, h, 1), &push)?;
            z.push(o);
        }
        // Sums of the weighted motion (x, y) and of the weights from a and from b.
        let acc = self.ops.filled(rec, 4, h, w, 0.0)?;
        for (side, (f, zz, s, tw, vs)) in [(fab, &z[0], t, 1.0 - t, 1.0), (fba, &z[1], 1.0 - t, t, -1.0)].into_iter().enumerate() {
            let push = Push::new().t(f).t(zz).t(&acc).i(w as i32).i(h as i32).i(f.w as i32).i(f.h as i32).f(s).f(tw).f(vs).i(side as i32);
            rec.dispatch(&self.splat, g3(&self.splat, w, h, 1), &push)?;
        }
        let out = self.t(c, h, w)?;
        let push = Push::new().t(&acc).t(a).t(b).t(shot).t(fallback.unwrap_or(a)).t(&out).i(c as i32).i(w as i32).i(h as i32).f(t).f(CUT).i(fallback.is_some() as i32).f(SPLAT_D0).f(SPLAT_D1).f(SPLAT_SHARE).f(SPLAT_E0).f(SPLAT_E1);
        rec.dispatch(&self.splat_resolve, g3(&self.splat_resolve, w, h, c), &push)?;
        self.recycle(rec, acc);
        for x in z {
            self.recycle(rec, x);
        }
        Ok(out)
    }

    /// Motion blur of `cur` (shutter 0..1 of the frame time) from its motions to the
    /// previous and next pictures. `shots`: for each, its `same_shot` measure if known
    /// (a neighbour across a scene cut is left out).
    pub fn motion_blur(&self, rec: &mut Recorder, cur: &Tensor, back: Option<&Tensor>, fwd: Option<&Tensor>, shots: [Option<&Tensor>; 2], shutter: f32, samples: usize) -> Result<Tensor> {
        let (w, h) = (cur.w, cur.h);
        let vel = self.t(2, h, w)?;
        let have = back.is_some() as i32 | (fwd.is_some() as i32) << 1;
        let (b, f) = (back.unwrap_or(&vel), fwd.unwrap_or(&vel));
        let known = shots[0].is_some() as i32 | (shots[1].is_some() as i32) << 1;
        let push = Push::new().t(b).t(f).t(&vel).t(shots[0].unwrap_or(&vel)).t(shots[1].unwrap_or(&vel)).i(w as i32).i(h as i32).i(b.w as i32).i(b.h as i32).i(f.w as i32).i(f.h as i32).i(have).i(known).f(shutter).f(CUT);
        rec.dispatch(&self.mb_vel, g3(&self.mb_vel, w, h, 1), &push)?;
        let (tw, th) = (w.div_ceil(16), h.div_ceil(16));
        let tiles = self.t(2, th, tw)?;
        let neigh = self.t(2, th, tw)?;
        rec.dispatch(&self.mb_tile, g3(&self.mb_tile, tw, th, 1), &Push::new().t(&vel).t(&tiles).i(w as i32).i(h as i32).i(tw as i32).i(th as i32).i(0))?;
        rec.dispatch(&self.mb_tile, g3(&self.mb_tile, tw, th, 1), &Push::new().t(&tiles).t(&neigh).i(w as i32).i(h as i32).i(tw as i32).i(th as i32).i(1))?;
        let out = self.t(cur.c, h, w)?;
        let push = Push::new().t(cur).t(&vel).t(&neigh).t(&out).i(cur.c as i32).i(w as i32).i(h as i32).i(tw as i32).i(samples.max(2) as i32);
        rec.dispatch(&self.mb_gather, g3(&self.mb_gather, w, h, 1), &push)?;
        for x in [vel, tiles, neigh] {
            self.recycle(rec, x);
        }
        Ok(out)
    }

    /// Background removal output from the network's foreground + alpha (`m`, 4
    /// channels): 0 transparent, 1 green, 2 blue, 3 the matte.
    pub fn matte(&self, rec: &mut Recorder, m: &Tensor, src: &Tensor, mode: usize) -> Result<Tensor> {
        let oc = if mode == 0 { 4 } else { 3 };
        let out = self.t(oc, m.h, m.w)?;
        let push = Push::new().t(m).t(src).t(&out).i(m.w as i32).i(m.h as i32).i(mode as i32).i((src.c == 4) as i32).i(oc as i32);
        rec.dispatch(&self.matte_k, g3(&self.matte_k, m.w, m.h, oc), &push)?;
        Ok(out)
    }

    /// An RGBA storage image (GENERAL layout) as a tensor of `c` channels.
    pub fn from_image(&self, rec: &mut Recorder, view: ash::vk::ImageView, w: usize, h: usize, c: usize) -> Result<Tensor> {
        let out = self.t(c, h, w)?;
        rec.dispatch_images(&self.img_to_tensor, g3(&self.img_to_tensor, w, h, 1), &Push::new().t(&out).i(c as i32).i(w as i32).i(h as i32), &[view])?;
        Ok(out)
    }

    /// `t` (3 or 4 channels) from colour space `from` to `to` for this source; `t`
    /// itself when they are the same picture (DESIGN.md §7a).
    pub fn convert_space(&self, rec: &mut Recorder, t: Tensor, from: brp_render::Space, to: brp_render::Space, color: &brp_media::ColorInfo) -> Result<Tensor> {
        let sc = brp_render::workspace::SourceColor::of(color);
        if sc.same(from, to) || t.c < 3 {
            return Ok(t);
        }
        let out = self.t(t.c, t.h, t.w)?;
        let [f, to_, tr, p2020] = sc.words(from, to);
        let push = Push::new().t(&t).t(&out).i(t.c as i32).i(t.w as i32).i(t.h as i32).i(f).i(to_).i(tr).i(p2020).f(sc.peak);
        rec.dispatch(&self.space_k, g3(&self.space_k, t.w, t.h, 1), &push)?;
        self.recycle(rec, t);
        Ok(out)
    }

    /// A tensor written into a new picture (for display, or the effect chain).
    pub fn to_picture(&self, rec: &mut Recorder, t: &Tensor) -> Result<Arc<brp_render::compute::Picture>> {
        let pic = brp_render::compute::Picture::new(&self.ops.compute, rec, t.w as u32, t.h as u32, t.c > 3)?;
        rec.dispatch_images(&self.tensor_to_img, g3(&self.tensor_to_img, t.w, t.h, 1), &Push::new().t(t).i(t.c as i32).i(t.w as i32).i(t.h as i32), &[pic.view])?;
        Ok(pic)
    }
}
