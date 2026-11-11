//! Our own video denoiser, for before upscaling: the upscalers are stable on clean
//! pictures but draw faint grain as texture that changes every frame.
//!
//! 1. Temporal: the motion-compensated merge of `gpu_temporal` (up to 5 pictures
//!    each way), which also reports the noise left at each pixel (a fraction of one
//!    picture's, from the weights it used).
//! 2. Noise profile, measured on every picture, with nothing to set: where a
//!    neighbour moved onto the picture agrees with it, their difference is noise
//!    (twice its variance). Its 8x8 DCT energies give the noise per frequency, per
//!    brightness level (8 bins) and per colour channel (an orthonormal opponent
//!    space: brightness and two colour differences, so noise stays independent).
//! 3. Spatial: overlapping 8x8 DCT blocks (every 2 pixels) of the merged picture,
//!    each coefficient shrunk by the noise expected there (profile x the noise left
//!    at that place), blocks put back weighted by how little they kept.

use crate::gpu::Ops;
use crate::gpu_temporal::{Temporal, COMMON};
use crate::vk::{PushTensor, Tensor};
use anyhow::Result;
use brp_render::compute::{Kernel, Push, Recorder, PRELUDE};
use brp_render::usershader::NO_OPT;
use std::sync::Arc;

/// Brightness levels of the profile.
const BINS: usize = 8;
/// Agreement (0..1) a neighbour must have over a whole block for its difference to
/// count as noise.
const AGREE_MIN: f32 = 0.9;
/// Blocks a brightness level needs; with fewer it takes the nearest level's noise.
const MIN_BLOCKS: f32 = 16.0;

/// Orthonormal opponent colours, the 8x8 DCT, and a block transform.
const SHARED: &str = r#"
#extension GL_EXT_control_flow_attributes : require
const float S3 = 0.57735027, S2 = 0.70710678, S6 = 0.40824829;
vec3 to_opp(vec3 c) { return vec3((c.r + c.g + c.b) * S3, (c.r - c.b) * S2, (c.r - 2.0 * c.g + c.b) * S6); }
vec3 from_opp(vec3 o) { return vec3(o.x * S3 + o.y * S2 + o.z * S6, o.x * S3 - 2.0 * o.z * S6, o.x * S3 - o.y * S2 + o.z * S6); }
// out[q] = sum_r M(q, r) in[r] along rows, then columns; M = DCT (inv: its transpose).
void dct8x8(inout float a[64], bool inv) {
    float t[64];
    [[unroll]] for (int y = 0; y < 8; y++) [[unroll]] for (int q = 0; q < 8; q++) {
        float s = 0.0;
        [[unroll]] for (int r = 0; r < 8; r++) s += (inv ? D8[r * 8 + q] : D8[q * 8 + r]) * a[y * 8 + r];
        t[y * 8 + q] = s;
    }
    [[unroll]] for (int q = 0; q < 8; q++) [[unroll]] for (int x = 0; x < 8; x++) {
        float s = 0.0;
        [[unroll]] for (int r = 0; r < 8; r++) s += (inv ? D8[r * 8 + q] : D8[q * 8 + r]) * t[r * 8 + x];
        a[q * 8 + x] = s;
    }
}
int bin_of(float ysum) { return clamp(int(ysum / 64.0 / 1.7320508 * float(BINS)), 0, BINS - 1); }
"#;

/// Impulses (salt and pepper, fireflies, hot or dead pixels): a pixel, or one of a
/// pair, far above or below its 8 neighbours. Per channel, the neighbours sorted;
/// the pixel's excess over the 2nd brightest (below the 2nd darkest), so a pair is
/// caught but a line (2 bright neighbours) is not, against a threshold of `a` x the
/// local variation (the spread of the middle six, or, if less, the range of the 4
/// opposite pairs' sums, which a smooth ramp leaves at 0: on a gradient the spread
/// alone hid impulses) plus `b` x the level (at least 1: the picture is in
/// display space, which for linear sources is gamma 2.2 with no ceiling, so
/// fireflies far above 1 are still compared relatively). A caught pixel is kept when
/// the moved neighbour pictures (`wa`, `wb`, nearest each side, NaN outside; bit k
/// of `sides` = side k bound) show it too, of like colour and standing out there (a
/// star, a glint, the end of a thin line): on one side right at its place, or on
/// every side within 1 pixel. Otherwise it is replaced by the neighbours' median,
/// not a blur, ramping in over 1..1.5 x the threshold.
/// `sides` < 0: no neighbour check (cleaning the neighbours themselves). No usable
/// neighbour at a pixel: the stricter rule, a single pixel at twice the threshold.
const IMPULSE: &str = r#"
G3
layout(push_constant) uniform P { F32 img; F32 wa; F32 wb; F32 o; int C, W, H, sides; float a, b; } p;
#define CE(i, j) { float lo = min(s[i], s[j]); s[j] = max(s[i], s[j]); s[i] = lo; }
vec3 px(F32 t, int j, int n) { return vec3(t.v[j], t.v[n + j], t.v[2 * n + j]); }
void main() { ID3 if (x >= p.W || y >= p.H) return;
    int i = y * p.W + x, n = p.W * p.H;
    if (p.W < 3 || p.H < 3) { for (int c = 0; c < p.C; c++) p.o.v[c * n + i] = p.img.v[c * n + i]; return; }
    // Mirrored at the edges: never the pixel itself.
    int xl = x > 0 ? x - 1 : 1, xr = x < p.W - 1 ? x + 1 : p.W - 2;
    int yu = y > 0 ? y - 1 : 1, yd = y < p.H - 1 ? y + 1 : p.H - 2;
    int nb[8] = int[8](yu * p.W + xl, yu * p.W + x, yu * p.W + xr, y * p.W + xl, y * p.W + xr, yd * p.W + xl, yd * p.W + x, yd * p.W + xr);
    vec3 v, med;
    float r1 = 0.0, r2 = 0.0;
    for (int c = 0; c < 3; c++) {
        float s[8];
        [[unroll]] for (int k = 0; k < 8; k++) s[k] = p.img.v[c * n + nb[k]];
        // The means of the 4 opposite pairs agree on any plane: their range is the
        // local variation without the slope (a ramp spreads the 8 themselves).
        float m0 = s[0] + s[7], m1 = s[1] + s[6], m2 = s[2] + s[5], m3 = s[3] + s[4];
        float ramp = max(max(m0, m1), max(m2, m3)) - min(min(m0, m1), min(m2, m3));
        // Batcher's odd-even merge sort, 19 exchanges.
        CE(0,1) CE(2,3) CE(4,5) CE(6,7) CE(0,2) CE(1,3) CE(4,6) CE(5,7) CE(1,2) CE(5,6)
        CE(0,4) CE(1,5) CE(2,6) CE(3,7) CE(2,4) CE(3,5) CE(1,2) CE(3,4) CE(5,6)
        float u = p.img.v[c * n + i];
        v[c] = u;
        med[c] = 0.5 * (s[3] + s[4]);
        float t = p.a * min(s[6] - s[1], ramp) + p.b * max(abs(med[c]), 1.0);
        r2 = max(r2, max(u - s[6], s[1] - u) / t);
        r1 = max(r1, max(u - s[7], s[0] - u) / (2.0 * t));
    }
    float r = r2;
    if (p.sides >= 0 && r2 > 1.0) {
        vec3 d = v - med;
        float dd = dot(d, d);
        // Per side: no picture there (0), there but not showing it (1), showing it
        // within 1 pixel (2), right at the moved place (3).
        int got[2] = int[2](0, 0);
        for (int sd = 0; sd < 2; sd++) {
            if (((p.sides >> sd) & 1) == 0) continue;
            if (isnan(sd == 0 ? p.wa.v[i] : p.wb.v[i])) continue;
            got[sd] = 1;
            for (int dy = -1; dy <= 1 && got[sd] < 3; dy++) for (int dx = -1; dx <= 1 && got[sd] < 3; dx++) {
                int qx = clamp(x + dx, 0, p.W - 1), qy = clamp(y + dy, 0, p.H - 1);
                vec3 e = (sd == 0 ? px(p.wa, qy * p.W + qx, n) : px(p.wb, qy * p.W + qx, n)) - med;
                if (isnan(e.x)) continue;
                // Along the pixel's own deviation (moved pictures are interpolated: a
                // point split over 2x2 comes back at about a third), and of like colour.
                float pr = dot(e, d) / dd;
                if (pr < 0.3 || length(e - pr * d) > 0.5 * pr * sqrt(dd)) continue;
                // And standing out there too: well above the middle of its own 8 (a
                // point smeared over 2x2 still is; the bright side of an edge is not).
                float s[8];
                int k = 0;
                for (int ey = -1; ey <= 1; ey++) for (int ex = -1; ex <= 1; ex++) {
                    if (ex == 0 && ey == 0) continue;
                    int q = clamp(qy + ey, 0, p.H - 1) * p.W + clamp(qx + ex, 0, p.W - 1);
                    vec3 f = sd == 0 ? px(p.wa, q, n) : px(p.wb, q, n);
                    s[k++] = isnan(f.x) ? pr : dot(f - med, d) / dd;
                }
                CE(0,1) CE(2,3) CE(4,5) CE(6,7) CE(0,2) CE(1,3) CE(4,6) CE(5,7) CE(1,2) CE(5,6)
                CE(0,4) CE(1,5) CE(2,6) CE(3,7) CE(2,4) CE(3,5) CE(1,2) CE(3,4) CE(5,6)
                if (pr - s[4] >= 0.3 * pr) got[sd] = (dx == 0 && dy == 0) ? 3 : 2;
            }
        }
        // Kept if seen right at its place on a side, or within 1 pixel on every side
        // there is (a 3x3 search alone also finds another frame's own impulses, about
        // 1 in 12 at 0.3% salt and pepper).
        bool any = got[0] > 0 || got[1] > 0;
        bool keep = got[0] == 3 || got[1] == 3 || (any && got[0] != 1 && got[1] != 1);
        r = any ? (keep ? 0.0 : r2) : r1;
    }
    float al = smoothstep(1.0, 1.5, r);
    for (int c = 0; c < 3; c++) p.o.v[c * n + i] = mix(v[c], med[c], al);
    for (int c = 3; c < p.C; c++) p.o.v[c * n + i] = p.img.v[c * n + i];
}
"#;

/// One thread per whole 8x8 block: the DCT energies of (picture - moved neighbour)
/// in opponent colours, and the block's brightness level (-1: not usable).
const PROFILE_BLOCKS: &str = r#"
G1
layout(push_constant) uniform P { F32 cur; F32 warped; F32 agree; F32 e; F32 meta; int W, H, BW, BH, base; float amin; } p;
void main() { ID1 if (i >= p.BW * p.BH) return;
    int bx = (i % p.BW) * 8, by = (i / p.BW) * 8, n = p.W * p.H, o = p.base + i;
    float a0[64], a1[64], a2[64];
    float ysum = 0.0;
    for (int k = 0; k < 64; k++) {
        int j = (by + k / 8) * p.W + bx + k % 8;
        float wv = p.warped.v[j];
        if (isnan(wv) || p.agree.v[j] < p.amin) { p.meta.v[o] = -1.0; return; }
        vec3 c = vec3(p.cur.v[j], p.cur.v[n + j], p.cur.v[2 * n + j]);
        vec3 d = to_opp(c - vec3(wv, p.warped.v[n + j], p.warped.v[2 * n + j]));
        a0[k] = d.x; a1[k] = d.y; a2[k] = d.z;
        ysum += to_opp(c).x;
    }
    p.meta.v[o] = float(bin_of(ysum));
    dct8x8(a0, false); dct8x8(a1, false); dct8x8(a2, false);
    for (int k = 0; k < 64; k++) {
        p.e.v[(o * 3) * 64 + k] = a0[k] * a0[k];
        p.e.v[(o * 3 + 1) * 64 + k] = a1[k] * a1[k];
        p.e.v[(o * 3 + 2) * 64 + k] = a2[k] * a2[k];
    }
}
"#;

/// One workgroup per (channel, coefficient): mean energy per brightness level over
/// the usable blocks, halved (a difference of two pictures has twice the noise).
const PROFILE_REDUCE: &str = r#"
layout(local_size_x = 256) in;
layout(push_constant) uniform P { F32 e; F32 meta; F32 prof; F32 cnt; int nb; } p;
shared float ss[16 * 256];
void main() {
    int lid = int(gl_LocalInvocationID.x), idx = int(gl_WorkGroupID.x), ch = idx / 64, k = idx % 64;
    float s[BINS], c[BINS];
    [[unroll]] for (int q = 0; q < BINS; q++) { s[q] = 0.0; c[q] = 0.0; }
    for (int b = lid; b < p.nb; b += 256) {
        float m = p.meta.v[b];
        if (m < 0.0) continue;
        float v = p.e.v[(b * 3 + ch) * 64 + k];
        [[unroll]] for (int q = 0; q < BINS; q++) if (int(m) == q) { s[q] += v; c[q] += 1.0; }
    }
    [[unroll]] for (int q = 0; q < BINS; q++) { ss[q * 256 + lid] = s[q]; ss[(BINS + q) * 256 + lid] = c[q]; }
    barrier();
    for (int st = 128; st > 0; st >>= 1) {
        if (lid < st) for (int q = 0; q < 2 * BINS; q++) ss[q * 256 + lid] += ss[q * 256 + lid + st];
        barrier();
    }
    if (lid < BINS) {
        float cn = ss[(BINS + lid) * 256];
        p.prof.v[(ch * BINS + lid) * 64 + k] = cn > 0.0 ? 0.5 * ss[lid * 256] / cn : 0.0;
        if (idx == 0) p.cnt.v[lid] = cn;
    }
}
"#;

/// Brightness levels with too few blocks take the nearest level's noise (none: 0,
/// so nothing is filtered).
const PROFILE_FIX: &str = r#"
layout(local_size_x = 64) in;
layout(push_constant) uniform P { F32 prof; F32 cnt; float minb; } p;
void main() {
    int k = int(gl_LocalInvocationID.x);
    for (int b = 0; b < BINS; b++) {
        if (p.cnt.v[b] >= p.minb) continue;
        int src = -1;
        for (int d = 1; d < BINS && src < 0; d++) {
            if (b - d >= 0 && p.cnt.v[b - d] >= p.minb) src = b - d;
            else if (b + d < BINS && p.cnt.v[b + d] >= p.minb) src = b + d;
        }
        for (int ch = 0; ch < 3; ch++) p.prof.v[(ch * BINS + b) * 64 + k] = src >= 0 ? p.prof.v[(ch * BINS + src) * 64 + k] : 0.0;
    }
}
"#;

/// The spatial filter. A workgroup makes a 16x16 tile from the 11x11 blocks (every
/// 2 pixels) that cover it, one channel at a time: each block's coefficients
/// (except DC) are shrunk against the noise expected there, and every pixel is the
/// weighted mean of its 16 blocks (weight 1 / (1 + sum of gains^2): blocks that
/// kept less are smoother, so more reliable).
const DCT_DENOISE: &str = r#"
layout(local_size_x = 16, local_size_y = 16) in;
layout(push_constant) uniform P { F32 img; F32 fac; F32 prof; F32 o; int C, W, H, mode; float lam_y, lam_c, tau; } p;
#define RW 28
#define NB 11
shared float s_in[3 * RW * RW];
shared float s_blk[NB * NB * 64];
shared float s_w[NB * NB];
void main() {
    int lx = int(gl_LocalInvocationID.x), ly = int(gl_LocalInvocationID.y), lid = ly * 16 + lx;
    int tx0 = int(gl_WorkGroupID.x) * 16, ty0 = int(gl_WorkGroupID.y) * 16;
    int x = tx0 + lx, y = ty0 + ly, n = p.W * p.H;
    for (int e = lid; e < RW * RW; e += 256) {
        int gx = clamp(tx0 - 6 + e % RW, 0, p.W - 1), gy = clamp(ty0 - 6 + e / RW, 0, p.H - 1), j = gy * p.W + gx;
        vec3 o = to_opp(vec3(p.img.v[j], p.img.v[n + j], p.img.v[2 * n + j]));
        s_in[e] = o.x; s_in[RW * RW + e] = o.y; s_in[2 * RW * RW + e] = o.z;
    }
    barrier();
    vec3 res;
    for (int ch = 0; ch < 3; ch++) {
        if (lid < NB * NB) {
            int bxl = 2 * (lid % NB), byl = 2 * (lid / NB);
            float a[64];
            float ysum = 0.0;
            [[unroll]] for (int k = 0; k < 64; k++) {
                int r = (byl + k / 8) * RW + bxl + k % 8;
                a[k] = s_in[ch * RW * RW + r];
                ysum += s_in[r];
            }
            int bin = bin_of(ysum);
            int gcx = clamp(tx0 - 6 + bxl + 4, 0, p.W - 1), gcy = clamp(ty0 - 6 + byl + 4, 0, p.H - 1);
            float lam = (ch == 0 ? p.lam_y : p.lam_c) * p.fac.v[gcy * p.W + gcx];
            dct8x8(a, false);
            float g2 = 0.0;
            [[unroll]] for (int k = 1; k < 64; k++) {
                float nv = lam * p.prof.v[(ch * BINS + bin) * 64 + k];
                float c2 = a[k] * a[k];
                float g = p.mode == 0 ? (c2 > p.tau * p.tau * nv ? 1.0 : 0.0) : c2 / (c2 + nv + 1e-20);
                a[k] *= g;
                g2 += g * g;
            }
            dct8x8(a, true);
            [[unroll]] for (int k = 0; k < 64; k++) s_blk[lid * 64 + k] = a[k];
            s_w[lid] = 1.0 / (1.0 + g2);
        }
        barrier();
        float acc = 0.0, ws = 0.0;
        int rx = lx + 6, ry = ly + 6;
        for (int j = 0; j < 4; j++) for (int q = 0; q < 4; q++) {
            int bx = (lx & ~1) + 2 * q, by = (ly & ~1) + 2 * j;
            int b = (by / 2) * NB + bx / 2;
            float w = s_w[b];
            acc += w * s_blk[b * 64 + (ry - by) * 8 + (rx - bx)];
            ws += w;
        }
        res[ch] = acc / ws;
        barrier();
    }
    if (x >= p.W || y >= p.H) return;
    int i = y * p.W + x;
    vec3 rgb = from_opp(res);
    p.o.v[i] = rgb.r; p.o.v[n + i] = rgb.g; p.o.v[2 * n + i] = rgb.b;
    for (int c = 3; c < p.C; c++) p.o.v[c * n + i] = p.img.v[c * n + i];
}
"#;

/// How strongly to denoise.
#[derive(Clone, Copy, Debug)]
pub struct DenoiseSettings {
    /// The temporal merge, 0..1 (0: the picture alone).
    pub temporal: f32,
    /// Spatial filter strength (x the measured noise variance; 0: off).
    pub spatial: f32,
    /// Colour noise strength relative to `spatial`.
    pub chroma: f32,
    /// Weight of each picture relative to the one a step nearer.
    pub decay: f32,
    /// 0: hard threshold at `tau` noise deviations; 1: Wiener gain.
    pub mode: i32,
    pub tau: f32,
    /// Impulse removal (salt and pepper, fireflies), 0 (off) ..1 (strong).
    pub impulse: f32,
}

impl DenoiseSettings {
    /// The impulse test's (spread factor, level floor) for `impulse`.
    fn impulse_thresholds(&self) -> (f32, f32) {
        let k = self.impulse.clamp(0.0, 1.0);
        (3.0 - 2.0 * k, 0.15 - 0.12 * k)
    }
}

impl Default for DenoiseSettings {
    fn default() -> Self {
        // Hard threshold: on grain it removed more than the Wiener gain (Natural 2x
        // flicker 0.015 against 0.029) and left clean video as untouched.
        DenoiseSettings { temporal: 1.0, spatial: 1.0, chroma: 1.0, decay: crate::gpu_temporal::DECAY, mode: 0, tau: 2.7, impulse: 0.5 }
    }
}

pub struct Denoiser {
    ops: Arc<Ops>,
    blocks_k: Arc<Kernel>,
    reduce_k: Arc<Kernel>,
    fix_k: Arc<Kernel>,
    dct_k: Arc<Kernel>,
    impulse_k: Arc<Kernel>,
}

impl Denoiser {
    pub fn new(ops: &Arc<Ops>) -> Result<Denoiser> {
        let c = &ops.compute;
        // The orthonormal 8-point DCT-II, row u = frequency.
        let d8: Vec<String> = (0..64)
            .map(|i| {
                let (u, x) = ((i / 8) as f64, (i % 8) as f64);
                let a = if u == 0.0 { (1.0f64 / 8.0).sqrt() } else { (2.0f64 / 8.0).sqrt() };
                format!("{:.9}", a * ((2.0 * x + 1.0) * u * std::f64::consts::PI / 16.0).cos())
            })
            .collect();
        let head = format!("{PRELUDE}{COMMON}#define BINS {BINS}\nconst float D8[64] = float[64]({});\n{SHARED}", d8.join(", "));
        let k = |name: &str, body: &str| c.kernel(&format!("dn_{name}"), &format!("{head}{body}"), 0);
        // The DCT kernels without SPIR-V optimization (15-22 s to compile otherwise, same speed).
        let k0 = |name: &str, body: &str| c.kernel(&format!("dn_{name}"), &format!("{head}{body}{NO_OPT}"), 0);
        Ok(Denoiser { ops: ops.clone(), blocks_k: k0("blocks", PROFILE_BLOCKS)?, reduce_k: k("reduce", PROFILE_REDUCE)?, fix_k: k("fix", PROFILE_FIX)?, dct_k: k0("dct", DCT_DENOISE)?, impulse_k: k("impulse", IMPULSE)? })
    }

    /// `cur` denoised. `sides`: as for `Temporal::window` (before and after, the
    /// pictures 1, 2, ... away with the motion to each from the one a step nearer).
    /// The noise profile needs a neighbour; with none, only the merge is done.
    pub fn record(&self, rec: &mut Recorder, tm: &Temporal, cur: &Tensor, sides: [&[(&Tensor, &Tensor)]; 2], s: &DenoiseSettings) -> Result<Tensor> {
        if s.impulse <= 0.0 {
            return self.record_clean(rec, tm, cur, sides, s);
        }
        // Impulses out first: left in, one spreads into a blob through the DCT and
        // lowers the merge's agreement around it. The current picture is checked
        // against its nearest neighbours as they are (a real small highlight is there
        // too); the neighbours, which only feed the merge and the noise profile, are
        // cleaned on their own (a real highlight they lose is still in the current
        // picture, where the merge then just keeps it).
        let ops = &self.ops;
        let (a, b) = s.impulse_thresholds();
        let (w, h) = (cur.w, cur.h);
        let groups = [w.div_ceil(16) as u32, h.div_ceil(8) as u32, 1];
        rec.label = "denoise: impulses (moved neighbours)".into();
        let mut moved: Vec<Tensor> = Vec::new();
        let mut mask = 0;
        for (side, list) in sides.iter().enumerate() {
            if let Some((img, step)) = list.first() {
                let t = ops.tensor(3, h, w)?;
                tm.warp_into(rec, img, step, &t)?;
                moved.push(t);
                mask |= 1 << side;
            }
        }
        let wa = if mask & 1 != 0 { &moved[0] } else { cur };
        let wb = if mask & 2 != 0 { moved.last().unwrap() } else { cur };
        let clean = ops.tensor(cur.c, h, w)?;
        let push = Push::new().t(cur).t(wa).t(wb).t(&clean).i(cur.c as i32).i(w as i32).i(h as i32).i(mask).f(a).f(b);
        rec.label = "denoise: impulses".into();
        rec.dispatch(&self.impulse_k, groups, &push)?;
        let mut nbs: [Vec<Tensor>; 2] = [Vec::new(), Vec::new()];
        rec.label = "denoise: impulses (neighbours)".into();
        for (side, list) in sides.iter().enumerate() {
            for (img, _) in list.iter() {
                let t = ops.tensor(img.c, img.h, img.w)?;
                let push = Push::new().t(img).t(img).t(img).t(&t).i(img.c as i32).i(img.w as i32).i(img.h as i32).i(-1).f(a).f(b);
                let g = [img.w.div_ceil(16) as u32, img.h.div_ceil(8) as u32, 1];
                rec.dispatch(&self.impulse_k, g, &push)?;
                nbs[side].push(t);
            }
        }
        rec.label = "-".into();
        let lists: [Vec<(&Tensor, &Tensor)>; 2] = [0, 1].map(|k| nbs[k].iter().zip(sides[k].iter()).map(|(t, (_, step))| (t, *step)).collect());
        let out = self.record_clean(rec, tm, &clean, [&lists[0], &lists[1]], s);
        drop(lists);
        for t in moved.into_iter().chain(nbs.into_iter().flatten()).chain([clean]) {
            ops.recycle(rec, t);
        }
        out
    }

    fn record_clean(&self, rec: &mut Recorder, tm: &Temporal, cur: &Tensor, sides: [&[(&Tensor, &Tensor)]; 2], s: &DenoiseSettings) -> Result<Tensor> {
        let ops = &self.ops;
        let (w, h) = (cur.w, cur.h);
        let fac = ops.tensor(1, h, w)?;
        let merged = tm.window_with_noise(rec, cur, sides, s.temporal, s.decay, &fac)?;
        let (bw, bh) = (w / 8, h / 8);
        if s.spatial <= 0.0 || (sides[0].is_empty() && sides[1].is_empty()) || bw == 0 || bh == 0 {
            ops.recycle(rec, fac);
            return Ok(merged);
        }
        // Noise profile from the nearest neighbour on each side.
        let nb = bw * bh;
        let e = ops.tensor(1, 1, 2 * nb * 3 * 64)?;
        let meta = ops.filled(rec, 1, 1, 2 * nb, -1.0)?;
        let warped = ops.tensor(3, h, w)?;
        let agree = ops.tensor(1, h, w)?;
        let scratch = ops.filled(rec, 1, 1, 1, 0.0)?;
        for (side, list) in sides.iter().enumerate() {
            let Some((img, step)) = list.first() else { continue };
            tm.warp_into(rec, img, step, &warped)?;
            tm.agreement_into(rec, cur, &warped, &agree, &scratch, 0)?;
            let push = Push::new().t(cur).t(&warped).t(&agree).t(&e).t(&meta).i(w as i32).i(h as i32).i(bw as i32).i(bh as i32).i((side * nb) as i32).f(AGREE_MIN);
            let groups = nb.div_ceil(256);
            rec.label = "denoise: noise profile (blocks)".into();
            rec.dispatch(&self.blocks_k, [groups.min(65535) as u32, groups.div_ceil(65535) as u32, 1], &push)?;
        }
        let prof = ops.tensor(1, 1, 3 * BINS * 64)?;
        let cnt = ops.tensor(1, 1, BINS)?;
        rec.label = "denoise: noise profile (sums)".into();
        rec.dispatch(&self.reduce_k, [3 * 64, 1, 1], &Push::new().t(&e).t(&meta).t(&prof).t(&cnt).i((2 * nb) as i32))?;
        rec.dispatch(&self.fix_k, [1, 1, 1], &Push::new().t(&prof).t(&cnt).f(MIN_BLOCKS))?;
        let out = ops.tensor(cur.c, h, w)?;
        let push = Push::new()
            .t(&merged)
            .t(&fac)
            .t(&prof)
            .t(&out)
            .i(cur.c as i32)
            .i(w as i32)
            .i(h as i32)
            .i(s.mode)
            .f(s.spatial)
            .f(s.spatial * s.chroma)
            .f(s.tau);
        rec.label = "denoise: DCT filter".into();
        rec.dispatch(&self.dct_k, [w.div_ceil(16) as u32, h.div_ceil(16) as u32, 1], &push)?;
        for t in [fac, merged, e, meta, warped, agree, scratch, prof, cnt] {
            ops.recycle(rec, t);
        }
        Ok(out)
    }
}
