//! Apple ProRes (422 Proxy/LT/Standard/HQ, 4444 and 4444 XQ, with alpha) on the GPU.
//!
//! The bitstream, the profiles' quantisation matrices, bit budgets and the
//! per-slice quantiser search follow FFmpeg's `prores_ks` encoder (LGPL; tables
//! from `libavcodec/proresenc_kostya.c`; scans and codebooks shared with the GPU
//! decoders in `brp_render::gpudec::tables`), so files are alike:
//!
//! 1. `est` (a workgroup per slice of 8 macroblocks): 8x8 DCTs of the slice into
//!    shared memory, then the exact coded size and the quantisation error of every
//!    plane at every quantiser of the profile, in parallel; above the budget, the
//!    smallest coarser quantiser that fits (binary search).
//! 2. `trellis` (a thread per slice row): `prores_ks`'s trellis picks each slice's
//!    quantiser so the row stays within its budget at the least error.
//! 3. `offsets` (one workgroup): slice offsets (prefix sum), picture header and
//!    slice index.
//! 4. `enc` (a workgroup per slice): the DCTs again, then each plane's codewords
//!    (one thread per plane, the alpha run-length coder on another) at its offset.
//!
//! Interlaced frames are two field pictures, coded one after the other.

use super::{Codec, Encoder, Output, COMMON, DCT8, HEAD, SAMPLE};
use anyhow::{bail, ensure, Result};
use brp_media::encode::PackLayout;
use brp_render::compute::{Buffer, Compute, Kernel, Push, Recorder, PRELUDE};
use std::sync::Arc;
use brp_render::gpudec::tables::{glsl_array, PRORES_DC_CB, PRORES_FIRST_DC_CB, PRORES_INTERLACED_SCAN, PRORES_LEV_CB, PRORES_PROGRESSIVE_SCAN, PRORES_RUN_CB};

#[derive(Clone, Debug, PartialEq)]
pub struct Settings {
    /// 0 Proxy, 1 LT, 2 Standard, 3 HQ, 4 4444, 5 4444 XQ.
    pub profile: usize,
    /// Code the alpha plane (4444 profiles, `yuva444p10le`).
    pub alpha: bool,
    /// Interlaced: Some(top field first).
    pub interlaced: Option<bool>,
    /// Colour primaries, transfer, matrix (FFmpeg's / ISO 23091 values).
    pub color: (u8, u8, u8),
    pub vendor: [u8; 4],
}

impl Settings {
    /// The profile index of FFmpeg's `profile` option names.
    pub fn profile_of(name: &str) -> Option<usize> {
        Some(match name {
            "proxy" => 0,
            "lt" => 1,
            "standard" => 2,
            "hq" => 3,
            "4444" => 4,
            "4444xq" => 5,
            _ => return None,
        })
    }
}

/// MOV sample entry tag of a profile.
pub fn tag(profile: usize) -> [u8; 4] {
    *[b"apco", b"apcs", b"apcn", b"apch", b"ap4h", b"ap4x"][profile.min(5)]
}

// ---- profile tables (prores_ks) ----------------------------------------------------

const QM_PROXY: [u8; 64] = [
    4, 7, 9, 11, 13, 14, 15, 63, 7, 7, 11, 12, 14, 15, 63, 63, 9, 11, 13, 14, 15, 63, 63, 63, 11, 11, 13, 14, 63, 63, 63, 63, 11, 13, 14, 63, 63, 63, 63, 63, 13, 14, 63, 63, 63, 63, 63, 63, 13, 63, 63, 63,
    63, 63, 63, 63, 63, 63, 63, 63, 63, 63, 63, 63,
];
const QM_PROXY_CHROMA: [u8; 64] = [
    4, 7, 9, 11, 13, 14, 63, 63, 7, 7, 11, 12, 14, 63, 63, 63, 9, 11, 13, 14, 63, 63, 63, 63, 11, 11, 13, 14, 63, 63, 63, 63, 11, 13, 14, 63, 63, 63, 63, 63, 13, 14, 63, 63, 63, 63, 63, 63, 13, 63, 63, 63,
    63, 63, 63, 63, 63, 63, 63, 63, 63, 63, 63, 63,
];
const QM_LT: [u8; 64] = [
    4, 5, 6, 7, 9, 11, 13, 15, 5, 5, 7, 8, 11, 13, 15, 17, 6, 7, 9, 11, 13, 15, 15, 17, 7, 7, 9, 11, 13, 15, 17, 19, 7, 9, 11, 13, 14, 16, 19, 23, 9, 11, 13, 14, 16, 19, 23, 29, 9, 11, 13, 15, 17, 21, 28, 35, 11,
    13, 16, 17, 21, 28, 35, 41,
];
const QM_STANDARD: [u8; 64] = [
    4, 4, 5, 5, 6, 7, 7, 9, 4, 4, 5, 6, 7, 7, 9, 9, 5, 5, 6, 7, 7, 9, 9, 10, 5, 5, 6, 7, 7, 9, 9, 10, 5, 6, 7, 7, 8, 9, 10, 12, 6, 7, 7, 8, 9, 10, 12, 15, 6, 7, 7, 9, 10, 11, 14, 17, 7, 7, 9, 10, 11, 14, 17, 21,
];
const QM_HQ: [u8; 64] = [
    4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 5, 4, 4, 4, 4, 4, 4, 5, 5, 4, 4, 4, 4, 4, 5, 5, 6, 4, 4, 4, 4, 5, 5, 6, 7, 4, 4, 4, 4, 5, 6, 7, 7,
];

struct Profile {
    min_q: u32,
    max_q: u32,
    br: [u32; 4],
    luma: &'static [u8; 64],
    chroma: &'static [u8; 64],
}

const PROFILES: [Profile; 6] = [
    Profile { min_q: 4, max_q: 8, br: [300, 242, 220, 194], luma: &QM_PROXY, chroma: &QM_PROXY_CHROMA },
    Profile { min_q: 1, max_q: 9, br: [720, 560, 490, 440], luma: &QM_LT, chroma: &QM_LT },
    Profile { min_q: 1, max_q: 6, br: [1050, 808, 710, 632], luma: &QM_STANDARD, chroma: &QM_STANDARD },
    Profile { min_q: 1, max_q: 6, br: [1566, 1216, 1070, 950], luma: &QM_HQ, chroma: &QM_HQ },
    Profile { min_q: 1, max_q: 6, br: [2350, 1828, 1600, 1425], luma: &QM_HQ, chroma: &QM_HQ },
    Profile { min_q: 1, max_q: 6, br: [3525, 2742, 2400, 2137], luma: &QM_HQ, chroma: &QM_HQ },
];
const MB_LIMITS: [u32; 4] = [1620, 2700, 6075, 9216];


/// Rounding of the quantiser (FFmpeg truncates: 0). Rounding towards the nearest
/// level, with a small dead zone for the AC coefficients, gives a better picture
/// for the same budget (measured, see DESIGN.md).
const RND_AC: f32 = 0.33;
const RND_DC: f32 = 0.5;

// ---- kernels ---------------------------------------------------------------------------

const KERNELS: &str = r#"
const uint SCORE_LIMIT = 0x3fffffffu;
#define CH_MAX (C444 != 0 ? 2048u : 1024u)
#define BASE_U 2048u
#define BASE_V (2048u + CH_MAX)
#define R_SLICE (NQ * 4u + 4u)

layout(push_constant) uniform P { U32 src; U32 work; U32 o; uint field; uint first; } p;

shared uint cf[3072];
shared uint al[1024];

int getc(uint i) {
    uint w = cf[i >> 1];
    return (i & 1u) == 0u ? (int(w << 16) >> 16) : (int(w) >> 16);
}

void slice_geom(uint s, out uint mbx, out uint mby, out uint mbs) {
    mby = s / SW;
    uint k = s % SW;
    uint full = MBW >> 3;
    if (k < full) { mbx = k * 8u; mbs = 8u; return; }
    uint x = full * 8u, j = k - full;
    mbx = x; mbs = 1u;
    for (int b = 2; b >= 0; b--) {
        uint sz = 1u << uint(b);
        if ((MBW & sz) != 0u) {
            if (j == 0u) { mbx = x; mbs = sz; return; }
            j--;
            x += sz;
        }
    }
}

uint frame_line(uint y) { return INTERLACED != 0 ? y * 2u + p.field : y; }

uint px(uint plane, uint x, uint y) {
    return sample16(p.src, PL_OFF[plane], PL_STRIDE[plane], x, frame_line(y));
}

// The 8x8 block at (bx, by) of a plane (edges repeated), transformed: coefficients
// are 4x the orthonormal DCT, the DC exactly as FFmpeg's integer DCT has it.
void dct_block(uint plane, uint bx, uint by, uint pw, uint dst) {
    float blk[64];
    uint sum = 0u;
    for (uint y = 0u; y < 8u; y++) {
        uint yy = min(by + y, FH - 1u);
        for (uint x = 0u; x < 8u; x++) {
            uint v = px(plane, min(bx + x, pw - 1u), yy);
            sum += v;
            blk[y * 8u + x] = float(v);
        }
    }
    fdct8x8(blk);
    int c[64];
    for (int k = 0; k < 64; k++) c[k] = int(round(blk[k] * 4.0));
    c[0] = int((sum + 1u) >> 1);
    for (int k = 0; k < 64; k += 2) cf[(dst + uint(k)) >> 1] = (uint(c[k]) & 0xffffu) | (uint(c[k + 1]) << 16);
}

// The slice's planes transformed into `cf`, its alpha (16-bit codes) into `al`.
void load_slice(uint mbx, uint mby, uint mbs) {
    uint t = gl_LocalInvocationID.x;
    uint cpb = C444 != 0 ? 4u : 2u;
    uint nl = mbs * 4u, nc = mbs * cpb;
    for (uint b = t; b < nl + 2u * nc; b += 64u) {
        if (b < nl) {
            uint m = b >> 2, sub = b & 3u;
            dct_block(0u, (mbx + m) * 16u + (sub & 1u) * 8u, mby * 16u + (sub >> 1) * 8u, W, b * 64u);
        } else {
            uint pl = b < nl + nc ? 1u : 2u;
            uint cb = b - nl - (pl - 1u) * nc;
            uint base = pl == 1u ? BASE_U : BASE_V;
            if (C444 != 0) {
                uint m = cb >> 2, sub = cb & 3u;
                dct_block(pl, (mbx + m) * 16u + (sub >> 1) * 8u, mby * 16u + (sub & 1u) * 8u, W, base + cb * 64u);
            } else {
                uint m = cb >> 1, sub = cb & 1u;
                dct_block(pl, (mbx + m) * 8u, mby * 16u + sub * 8u, W >> 1, base + cb * 64u);
            }
        }
    }
#if ALPHA
    uint sw = 16u * mbs;
    for (uint i = t; i < 8u * sw; i += 64u) {
        uint v2 = 0u;
        for (uint h = 0u; h < 2u; h++) {
            uint j = i * 2u + h, y = j / sw, x = j % sw;
            uint a = px(3u, min(mbx * 16u + x, W - 1u), min(mby * 16u + y, FH - 1u));
            v2 |= (((a << 6) | (a >> 4)) & 0xffffu) << (16u * h);
        }
        al[i] = v2;
    }
#endif
}

uint geta(uint i) { return (al[i >> 1] >> (16u * (i & 1u))) & 0xffffu; }

// Quantisation in exact integer arithmetic (the size estimate and the writer are
// different shaders and must agree on every level): |c| / step rounded with the
// 16-bit fixed-point offset `rnd`, through the reciprocal m = ceil(2^16 / step).
uint qmul(uint step) { return (65536u + step - 1u) / step; }
int qlev(int c, uint m, uint rnd) {
    int l = int((uint(abs(c)) * m + rnd) >> 16);
    return c < 0 ? -l : l;
}
uint qerr(int c, int l, uint step) { return uint(abs(abs(c) - abs(l) * int(step))); }
uint mkcode(int x) { return uint((x << 1) ^ (x >> 31)); }

uint vlc_len(uint cb, uint val) {
    uint sw = (cb & 3u) + 1u, ro = cb >> 5, eo = (cb >> 2) & 7u, sv = sw << ro;
    if (val >= sv) {
        val -= sv - (1u << eo);
        uint e = uint(findMSB(val));
        return e * 2u - eo + sw + 1u;
    }
    return (val >> ro) + ro + 1u;
}

uint qm(bool chroma, uint k) { return chroma ? QC[k] : QL[k]; }

// DC bits and error of a plane at quantiser q (one thread).
uint est_dc(uint base, uint nblk, bool chroma, uint q, inout uint err) {
    uint step0 = qm(chroma, 0u) * q, m0 = qmul(step0);
    int c0 = getc(base) - 16384;
    int prev = qlev(c0, m0, RND_DC);
    err += qerr(c0, prev, step0);
    uint bits = vlc_len(FIRST_DC_CB, mkcode(prev));
    int sgn = 0;
    uint cbi = 5u;
    for (uint b = 1u; b < nblk; b++) {
        int c = getc(base + b * 64u) - 16384;
        int dc = qlev(c, m0, RND_DC);
        err += qerr(c, dc, step0);
        int delta = dc - prev;
        int ns = delta >> 31;
        delta = (delta ^ sgn) - sgn;
        uint code = mkcode(delta);
        bits += vlc_len(DCCB[cbi], code);
        cbi = min(code, 6u);
        sgn = ns;
        prev = dc;
    }
    return bits;
}

#ifdef CHUNKS
// Exact AC bits of a plane at several quantisers at once (qv[0..nqv)), by the
// whole workgroup: each thread takes a chunk of the coding order (coefficient i
// of every block, i = 1..63), counts the codewords whose context lies inside its
// chunk, and keeps its first two and last nonzero levels; a scan over the chunks
// (a thread per quantiser) gives each chunk the context of the nonzero before it,
// and the chunks' first codewords are counted with it.
const uint NONE = 0xffffffffu;
shared uint ch_f[64 * NQS], ch_s[64 * NQS], ch_l[64 * NQS], ch_r[64 * NQS];
shared uint acc_bits[NQS], acc_err[NQS], qv[NQS];
shared uint nqv;
// AC bits of each thread's chunk at qv[0] (the writer's offsets).
shared uint tbits[64];

void est_planes(uint base, uint nblk, bool chroma) {
    uint t = gl_LocalInvocationID.x;
    uint nq = nqv;
    uint N = 63u * nblk, K = (N + 63u) / 64u;
    uint n0 = min(t * K, N), n1 = min(n0 + K, N);
    if (t < nq) { acc_bits[t] = 0u; acc_err[t] = 0u; }
    barrier();
    for (uint j = 0u; j < nq; j++) {
        uint q = qv[j];
        uint first = NONE, second = NONE, cnt = 0u, bits = 0u, err = 0u;
        uint lpos = 0u, labs = 0u, prun = 0u;
        uint ci = 0xffu, step = 1u, m = 1u;
        for (uint n = n0; n < n1; n++) {
            uint i = n / nblk + 1u, b = n - (i - 1u) * nblk;
            if (i != ci) { ci = i; step = qm(chroma, SCAN[i]) * q; m = qmul(step); }
            int c = getc(base + b * 64u + SCAN[i]);
            int l = qlev(c, m, RND_AC);
            err += qerr(c, l, step);
            if (l != 0) {
                uint a = uint(abs(l));
                if (cnt >= 2u) bits += vlc_len(RUNCB[min(prun, 15u)], n - lpos - 1u) + vlc_len(LEVCB[min(labs, 9u)], a - 1u) + 1u;
                uint myrun = n - lpos - 1u;
                if (cnt == 0u) first = n | (a << 16);
                else if (cnt == 1u) second = n | (a << 16);
                prun = myrun;
                lpos = n;
                labs = a;
                cnt++;
            }
        }
        uint c = t * NQS + j;
        ch_f[c] = first;
        ch_s[c] = second;
        ch_l[c] = cnt > 0u ? (lpos | (labs << 16)) : NONE;
        ch_r[c] = cnt >= 2u ? prun : NONE;
        if (j == 0u) tbits[t] = bits;
        atomicAdd(acc_bits[j], bits);
        atomicAdd(acc_err[j], err);
    }
    barrier();
    if (t < nq) {
        // Context before each chunk: (position of the last nonzero + 1, its level, its run);
        // before the first nonzero: run 4, level 2.
        uint cp1 = 0u, cabs = 2u, crun = 4u;
        for (uint k = 0u; k < 64u; k++) {
            uint c = k * NQS + t;
            uint f = ch_f[c], l = ch_l[c], r = ch_r[c];
            ch_l[c] = cp1 | (cabs << 16);
            ch_r[c] = crun;
            if (f != NONE) {
                if (r != NONE) {
                    cp1 = (l & 0xffffu) + 1u; cabs = l >> 16; crun = r;
                } else {
                    crun = (f & 0xffffu) - cp1;
                    cp1 = (f & 0xffffu) + 1u;
                    cabs = f >> 16;
                }
            }
        }
    }
    barrier();
    for (uint j = 0u; j < nq; j++) {
        uint c = t * NQS + j;
        uint f = ch_f[c];
        if (f == NONE) continue;
        uint cp1 = ch_l[c] & 0xffffu, cabs = ch_l[c] >> 16, crun = ch_r[c];
        uint fpos = f & 0xffffu, fabs = f >> 16;
        uint rf = fpos - cp1;
        uint bits = vlc_len(RUNCB[min(crun, 15u)], rf) + vlc_len(LEVCB[min(cabs, 9u)], fabs - 1u) + 1u;
        uint s = ch_s[c];
        if (s != NONE) bits += vlc_len(RUNCB[min(rf, 15u)], (s & 0xffffu) - fpos - 1u) + vlc_len(LEVCB[min(fabs, 9u)], (s >> 16) - 1u) + 1u;
        if (j == 0u) tbits[t] += bits;
        atomicAdd(acc_bits[j], bits);
    }
    if (t < nq) {
        uint e = 0u;
        uint b = est_dc(base, nblk, chroma, qv[t], e);
        atomicAdd(acc_bits[t], b);
        atomicAdd(acc_err[t], e);
    }
    barrier();
}
#endif

// Alpha: run-length coded 16-bit values (exact bits, byte-aligned).
uint alpha_diff_len(uint cur, uint prev) {
    int d = int((cur - prev) & 0xffffu);
    if (d >= 65536 - 64) d -= 65536;
    return (d < -64 || d > 64 || d == 0) ? 17u : 8u;
}
uint run_len(uint r) { return r == 0u ? 1u : (r < 16u ? 5u : 16u); }
uint est_alpha(uint n) {
    uint prev = 0xffffu, cur = geta(0u);
    uint bits = alpha_diff_len(cur, prev);
    prev = cur;
    uint run = 0u;
    for (uint i = 1u; i < n; i++) {
        cur = geta(i);
        if (cur != prev) {
            bits += run_len(run) + alpha_diff_len(cur, prev);
            prev = cur;
            run = 0u;
        } else run++;
    }
    bits += run_len(run);
    return (bits + 7u) & ~7u;
}
"#;

const EST: &str = r#"
layout(local_size_x = 64, local_size_y = 1, local_size_z = 1) in;
shared uint sbits[3 * NQS], serr[3 * NQS];
shared uint s_abits, s_lo, s_hi, s_probe[3], s_perr[3];

// The three planes at the quantisers in qv: into bits/err slot `slot` (+ plane * NQS).
void est_all(uint mbs, uint slot, bool probe) {
    uint t = gl_LocalInvocationID.x;
    uint cpb = C444 != 0 ? 4u : 2u;
    for (uint pl = 0u; pl < 3u; pl++) {
        uint base = pl == 0u ? 0u : (pl == 1u ? BASE_U : BASE_V);
        est_planes(base, pl == 0u ? mbs * 4u : mbs * cpb, pl > 0u);
        if (probe) {
            if (t == 0u) { s_probe[pl] = (acc_bits[0] + 7u) & ~7u; s_perr[pl] = acc_err[0]; }
        } else if (t < nqv) {
            sbits[pl * NQS + slot + t] = (acc_bits[t] + 7u) & ~7u;
            serr[pl * NQS + slot + t] = acc_err[t];
        }
        barrier();
    }
}

void main() {
    uint s = gl_WorkGroupID.x, t = gl_LocalInvocationID.x;
    uint mbx, mby, mbs;
    slice_geom(s, mbx, mby, mbs);
    load_slice(mbx, mby, mbs);
    if (t < NQS) qv[t] = QMIN + t;
    if (t == 0u) { nqv = NQS; s_lo = QMAX + 1u; s_hi = 128u; s_abits = 0u; }
    barrier();
#if ALPHA
    if (t == 0u) s_abits = est_alpha(mbs * 256u);
    barrier();
#endif
    est_all(mbs, 0u, false);
    uint budget = BPM * mbs;
    uint at_max = sbits[NQS - 1u] + sbits[2u * NQS - 1u] + sbits[3u * NQS - 1u] + s_abits;
    // Coarser than the profile allows, when even its coarsest does not fit.
    bool search = at_max > budget;
    while (search && s_lo < s_hi) {
        uint mid = (s_lo + s_hi) >> 1;
        barrier();
        if (t == 0u) { qv[0] = mid; nqv = 1u; }
        barrier();
        est_all(mbs, 0u, true);
        if (t == 0u) {
            if (s_probe[0] + s_probe[1] + s_probe[2] + s_abits <= budget) s_hi = mid; else s_lo = mid + 1u;
        }
        barrier();
    }
    uint overq = search ? s_lo : QMAX;
    if (search) {
        if (t == 0u) { qv[0] = overq; nqv = 1u; }
        barrier();
        est_all(mbs, 0u, true);
    }
    barrier();
    if (t == 0u) {
        uint r = s * R_SLICE;
        for (uint j = 0u; j < NQ; j++) {
            uint y, u, v, sc;
            if (j < NQS) {
                y = sbits[j]; u = sbits[NQS + j]; v = sbits[2u * NQS + j];
                sc = serr[j] + serr[NQS + j] + serr[2u * NQS + j];
            } else if (!search) {
                y = sbits[NQS - 1u]; u = sbits[2u * NQS - 1u]; v = sbits[3u * NQS - 1u];
                sc = serr[NQS - 1u] + serr[2u * NQS - 1u] + serr[3u * NQS - 1u] + 1u;
            } else {
                y = s_probe[0]; u = s_probe[1]; v = s_probe[2];
                sc = s_perr[0] + s_perr[1] + s_perr[2];
            }
            if (y + u + v + s_abits > 65000u * 8u) sc = SCORE_LIMIT;
            p.work.v[EST_OFF + r + j * 4u + 0u] = y >> 3;
            p.work.v[EST_OFF + r + j * 4u + 1u] = u >> 3;
            p.work.v[EST_OFF + r + j * 4u + 2u] = v >> 3;
            p.work.v[EST_OFF + r + j * 4u + 3u] = min(sc, SCORE_LIMIT);
        }
        p.work.v[EST_OFF + r + NQ * 4u] = s_abits >> 3;
        p.work.v[EST_OFF + r + NQ * 4u + 1u] = overq;
    }
}
"#;

const TRELLIS: &str = r#"
// prores_ks's trellis over a row of slices, a workgroup per row: the NQ x NQ
// transitions of each step in parallel, back pointers in shared memory.
layout(local_size_x = 128, local_size_y = 1, local_size_z = 1) in;
shared uint t_bits[NQ], t_score[NQ], c_bits[NQ * NQ], c_score[NQ * NQ], s_sb[NQ], s_err[NQ];
shared uint back[SW * NQ];
shared uint s_best;
void main() {
    uint row = gl_WorkGroupID.x, t = gl_LocalInvocationID.x;
    if (t < NQ) { t_bits[t] = 0u; t_score[t] = 0u; }
    uint mbcum = 0u;
    for (uint x = 0u; x < SW; x++) {
        uint s = row * SW + x;
        uint mbx, mby, mbs;
        slice_geom(s, mbx, mby, mbs);
        mbcum += mbs;
        uint limit = mbcum * BPM;
        uint r = EST_OFF + s * R_SLICE;
        if (t < NQ) {
            uint abytes = p.work.v[r + NQ * 4u];
            s_sb[t] = (p.work.v[r + t * 4u] + p.work.v[r + t * 4u + 1u] + p.work.v[r + t * 4u + 2u] + abytes) * 8u;
            s_err[t] = p.work.v[r + t * 4u + 3u];
        }
        barrier();
        if (t < NQ * NQ) {
            uint pj = t / NQ, j = t % NQ;
            uint bits = t_bits[pj] + s_sb[j];
            uint err = bits > limit ? SCORE_LIMIT : s_err[j];
            c_bits[t] = bits;
            c_score[t] = (t_score[pj] < SCORE_LIMIT && err < SCORE_LIMIT) ? t_score[pj] + err : SCORE_LIMIT;
        }
        barrier();
        if (t < NQ) {
            // The last previous state with the least score (prores_ks replaces on >=).
            uint bj = 0u, bs = c_score[t];
            for (uint pj = 1u; pj < NQ; pj++) {
                uint sc = c_score[pj * NQ + t];
                if (sc <= bs) { bs = sc; bj = pj; }
            }
            back[x * NQ + t] = bj;
            t_bits[t] = c_bits[bj * NQ + t];
            t_score[t] = bs;
        }
        barrier();
    }
    if (t == 0u) {
        uint e = t_score[0], best = 0u;
        for (uint j = 1u; j < NQ; j++) {
            if (t_score[j] <= e) { e = t_score[j]; best = j; }
        }
        uint j = best;
        for (int x = int(SW) - 1; x >= 0; x--) {
            uint s = row * SW + uint(x);
            uint r = EST_OFF + s * R_SLICE;
            uint q = j < NQS ? QMIN + j : p.work.v[r + NQ * 4u + 1u];
            uint y = p.work.v[r + j * 4u], u = p.work.v[r + j * 4u + 1u], v = p.work.v[r + j * 4u + 2u], a = p.work.v[r + NQ * 4u];
            uint i = INFO_OFF + s * 8u;
            p.work.v[i] = q; p.work.v[i + 1u] = y; p.work.v[i + 2u] = u; p.work.v[i + 3u] = v; p.work.v[i + 4u] = a;
            p.work.v[i + 5u] = SLICE_HDR + y + u + v + a;
            j = back[uint(x) * NQ + j];
        }
    }
}
"#;

const OFFSETS: &str = r#"
layout(local_size_x = 1024, local_size_y = 1, local_size_z = 1) in;
shared uint part[1024];
void main() {
    uint t = gl_LocalInvocationID.x;
    uint per = (SLICES + 1023u) / 1024u;
    uint s0 = t * per, s1 = min(s0 + per, SLICES);
    uint sum = 0u;
    for (uint s = s0; s < s1; s++) sum += p.work.v[INFO_OFF + s * 8u + 5u];
    part[t] = sum;
    uint base = p.first != 0u ? FH_BYTES : p.o.v[0];
    barrier();
    if (t == 0u) {
        uint acc = 0u;
        for (uint i = 0u; i < 1024u; i++) { uint v = part[i]; part[i] = acc; acc += v; }
    }
    barrier();
    uint index = base + 8u;
    uint data = index + 2u * SLICES;
    uint off = data + part[t];
    for (uint s = s0; s < s1; s++) {
        uint sz = p.work.v[INFO_OFF + s * 8u + 5u];
        p.work.v[INFO_OFF + s * 8u + 6u] = off;
        if (off + sz <= CAP) put_be16(p.o, HEAD + index + 2u * s, sz);
        off += sz;
    }
    barrier();
    if (t == 1023u) {
        uint total = off - base;
        bool over = off > CAP;
        if (!over) {
            put_byte(p.o, HEAD + base, 0x40u);
            put_be32(p.o, HEAD + base + 1u, total);
            put_be16(p.o, HEAD + base + 5u, SLICES);
            put_byte(p.o, HEAD + base + 7u, 3u << 4);
        }
        if (over) p.o.v[1] = 1u;
        p.o.v[0] = off;
    }
}
"#;

const ENC: &str = r#"
layout(local_size_x = 64, local_size_y = 1, local_size_z = 1) in;

void vlc_put(inout BitW w, uint cb, uint val) {
    uint sw = (cb & 3u) + 1u, ro = cb >> 5, eo = (cb >> 2) & 7u, sv = sw << ro;
    if (val >= sv) {
        val -= sv - (1u << eo);
        uint e = uint(findMSB(val));
        bw_zeros(p.o, w, e - eo + sw);
        bw_put(p.o, w, e + 1u, val);
    } else {
        bw_zeros(p.o, w, val >> ro);
        bw_put(p.o, w, 1u, 1u);
        bw_put(p.o, w, ro, val);
    }
}

// The DC codewords of a plane from its (byte-aligned) start.
void write_dc(uint base, uint nblk, bool chroma, uint q, uint pos) {
    BitW w = bw_new(HEAD + pos);
    uint step0 = qm(chroma, 0u) * q, m0 = qmul(step0);
    int prev = qlev(getc(base) - 16384, m0, RND_DC);
    vlc_put(w, FIRST_DC_CB, mkcode(prev));
    int sgn = 0;
    uint cbi = 5u;
    for (uint b = 1u; b < nblk; b++) {
        int dc = qlev(getc(base + b * 64u) - 16384, m0, RND_DC);
        int delta = dc - prev;
        int ns = delta >> 31;
        delta = (delta ^ sgn) - sgn;
        uint code = mkcode(delta);
        vlc_put(w, DCCB[cbi], code);
        cbi = min(code, 6u);
        sgn = ns;
        prev = dc;
    }
    bw_flush(p.o, w);
}

// This thread's chunk of AC codewords, from bit `bitpos` of the buffer, with the
// context the chunk scan left in ch_l / ch_r (quantiser slot 0).
void write_chunk(uint base, uint nblk, bool chroma, uint q, uint bitpos) {
    uint t = gl_LocalInvocationID.x;
    uint N = 63u * nblk, K = (N + 63u) / 64u;
    uint n0 = min(t * K, N), n1 = min(n0 + K, N);
    uint c = t * NQS;
    uint lastp1 = ch_l[c] & 0xffffu, plev = ch_l[c] >> 16, prun = ch_r[c];
    BitW w;
    w.acc = 0ul; w.n = bitpos & 7u; w.pos = bitpos >> 3;
    bool any = false;
    uint ci = 0xffu, m = 1u;
    for (uint n = n0; n < n1; n++) {
        uint i = n / nblk + 1u, b = n - (i - 1u) * nblk;
        if (i != ci) { ci = i; m = qmul(qm(chroma, SCAN[i]) * q); }
        int l = qlev(getc(base + b * 64u + SCAN[i]), m, RND_AC);
        if (l != 0) {
            uint a = uint(abs(l));
            uint run = n - lastp1;
            vlc_put(w, RUNCB[min(prun, 15u)], run);
            vlc_put(w, LEVCB[min(plev, 9u)], a - 1u);
            bw_put(p.o, w, 1u, l < 0 ? 1u : 0u);
            prun = run;
            plev = a;
            lastp1 = n + 1u;
            any = true;
        }
    }
    if (any) bw_flush(p.o, w);
}

shared uint s_dcbits, s_pref[64];

// One plane of the slice at quantiser q (qv[0]), byte-aligned at `pos`: `bytes` is
// what the estimate said (a mismatch is flagged).
void enc_plane(uint base, uint nblk, bool chroma, uint q, uint pos, uint bytes) {
    uint t = gl_LocalInvocationID.x;
    est_planes(base, nblk, chroma);
    if (t == 0u) {
        uint e = 0u;
        s_dcbits = est_dc(base, nblk, chroma, q, e);
        uint acc = 0u;
        for (uint i = 0u; i < 64u; i++) { s_pref[i] = acc; acc += tbits[i]; }
        if (((s_dcbits + acc + 7u) >> 3) != bytes) p.o.v[2] = 1u;
    }
    barrier();
    if (t == 0u) write_dc(base, nblk, chroma, q, pos);
    write_chunk(base, nblk, chroma, q, (HEAD + pos) * 8u + s_dcbits + s_pref[t]);
    barrier();
}

void alpha_diff(inout BitW w, uint cur, uint prev) {
    int d = int((cur - prev) & 0xffffu);
    if (d >= 65536 - 64) d -= 65536;
    if (d < -64 || d > 64 || d == 0) {
        bw_put(p.o, w, 1u, 1u);
        bw_put(p.o, w, 16u, uint(d) & 0xffffu);
    } else {
        bw_put(p.o, w, 1u, 0u);
        bw_put(p.o, w, 6u, uint(abs(d) - 1));
        bw_put(p.o, w, 1u, d < 0 ? 1u : 0u);
    }
}
void alpha_run(inout BitW w, uint r) {
    if (r == 0u) { bw_put(p.o, w, 1u, 1u); return; }
    bw_put(p.o, w, 1u, 0u);
    if (r < 16u) bw_put(p.o, w, 4u, r); else bw_put(p.o, w, 15u, r);
}
void enc_alpha(uint n, uint pos) {
    BitW w = bw_new(HEAD + pos);
    uint prev = 0xffffu, cur = geta(0u);
    alpha_diff(w, cur, prev);
    prev = cur;
    uint run = 0u;
    for (uint i = 1u; i < n; i++) {
        cur = geta(i);
        if (cur != prev) {
            alpha_run(w, run);
            alpha_diff(w, cur, prev);
            prev = cur;
            run = 0u;
        } else run++;
    }
    alpha_run(w, run);
    bw_flush(p.o, w);
}

void main() {
    if (p.o.v[1] != 0u) return;
    uint s = gl_WorkGroupID.x, t = gl_LocalInvocationID.x;
    uint mbx, mby, mbs;
    slice_geom(s, mbx, mby, mbs);
    load_slice(mbx, mby, mbs);
    barrier();
    uint i = INFO_OFF + s * 8u;
    uint q = p.work.v[i], y = p.work.v[i + 1u], u = p.work.v[i + 2u], v = p.work.v[i + 3u], off = p.work.v[i + 6u];
    uint cpb = C444 != 0 ? 4u : 2u;
    uint d = off + SLICE_HDR;
    if (t == 0u) {
        put_byte(p.o, HEAD + off, SLICE_HDR << 3);
        put_byte(p.o, HEAD + off + 1u, q);
        put_be16(p.o, HEAD + off + 2u, y);
        put_be16(p.o, HEAD + off + 4u, u);
        if (ALPHA != 0) put_be16(p.o, HEAD + off + 6u, v);
        qv[0] = q;
        nqv = 1u;
    }
    barrier();
    enc_plane(0u, mbs * 4u, false, q, d, y);
    enc_plane(BASE_U, mbs * cpb, true, q, d + y, u);
    enc_plane(BASE_V, mbs * cpb, true, q, d + y + u, v);
#if ALPHA
    if (t == 0u) enc_alpha(mbs * 256u, d + y + u + v);
#endif
}
"#;

// ---- the encoder ----------------------------------------------------------------------

struct Geometry {
    pictures: u32,
    mbh: u32,
    slices: u32,
}

pub struct ProRes {
    compute: Arc<Compute>,
    codec: Codec,
    s: Settings,
    g: Geometry,
    est: Arc<Kernel>,
    trellis: Arc<Kernel>,
    offsets: Arc<Kernel>,
    enc: Arc<Kernel>,
    work: Buffer,
    out: Output,
    header: Vec<u8>,
    layout: PackLayout,
    planes_addr: u64,
    /// Kernel defines (rebuilt when the capacity grows).
    defs: String,
}

/// Bytes of the frame atom and frame header before the pictures.
const FH_BYTES: u32 = 8 + 148;

impl ProRes {
    pub fn new(compute: &Arc<Compute>, s: Settings, layout: &PackLayout) -> Result<ProRes> {
        let c444 = layout.cw == 0;
        let want = match (c444, s.alpha) {
            (false, _) => "yuv422p10le",
            (true, false) => "yuv444p10le",
            (true, true) => "yuva444p10le",
        };
        ensure!(layout.name == want, "GPU ProRes takes {want}, not {}", layout.name);
        ensure!(s.profile < 6, "ProRes profile {}", s.profile);
        ensure!(!c444 || s.profile >= 4, "4:4:4 needs a 4444 profile");
        ensure!(c444 || s.profile < 4, "4444 profiles take 4:4:4");
        let (w, h) = (layout.w, layout.h);
        ensure!(w >= 16 && h >= 16 && w % 2 == 0, "ProRes size {w}x{h}");
        let pictures = if s.interlaced.is_some() { 2 } else { 1 };
        let fh = h / pictures;
        let mbw = w.div_ceil(16);
        let mbh = if pictures == 2 { h.div_ceil(32) } else { h.div_ceil(16) };
        let sw = mbw / 8 + (mbw % 8).count_ones();
        let g = Geometry { pictures, mbh, slices: sw * mbh };
        let pr = &PROFILES[s.profile];
        let mbs_total = mbw * mbh * pictures;
        let i = MB_LIMITS[..3].iter().position(|l| *l >= mbs_total).unwrap_or(3);
        let mut bpm = pr.br[i];
        if s.alpha {
            bpm *= 20;
        }
        let planes = if s.alpha { 4 } else { 3 };
        // Like prores_ks's upper bound, with room to spare (it grows when exceeded).
        let mut cap = (pictures * g.slices + 1) as u64 * (2 + 2 * planes + (8 * bpm as u64) / 8) + 200;
        if s.alpha {
            cap += (pictures * g.slices + 1) as u64 * ((8 * 256 * 18 + 7) / 8);
        }
        let cap = cap * 3 / 2 + (1 << 20);
        let header = frame_header(&s, w, h, c444);
        let nqs = pr.max_q - pr.min_q + 1;
        let nq = nqs + 1;
        let r_slice = nq * 4 + 4;
        let est_off = 0u32;
        let info_off = est_off + g.slices * r_slice;
        let scr_off = info_off + g.slices * 8;
        let work_words = scr_off;
        let mut defs = String::new();
        let d = |defs: &mut String, k: &str, v: u32| defs.push_str(&format!("#define {k} {v}u\n"));
        d(&mut defs, "W", w);
        d(&mut defs, "FH", fh);
        defs.push_str(&format!("#define INTERLACED {}\n#define C444 {}\n#define ALPHA {}\n", (pictures == 2) as u32, c444 as u32, s.alpha as u32));
        d(&mut defs, "MBW", mbw);
        d(&mut defs, "MBH", mbh);
        d(&mut defs, "SW", sw);
        d(&mut defs, "SLICES", g.slices);
        d(&mut defs, "QMIN", pr.min_q);
        d(&mut defs, "QMAX", pr.max_q);
        d(&mut defs, "NQS", nqs);
        d(&mut defs, "NQ", nq);
        d(&mut defs, "BPM", bpm);
        d(&mut defs, "EST_OFF", est_off);
        d(&mut defs, "INFO_OFF", info_off);
        d(&mut defs, "SCR_OFF", scr_off);
        d(&mut defs, "FH_BYTES", FH_BYTES);
        d(&mut defs, "HEAD", HEAD as u32);
        d(&mut defs, "SLICE_HDR", 2 + 2 * (planes as u32 - 1));
        let rnd = |k: &str, d: f32| std::env::var(k).ok().and_then(|v| v.parse().ok()).unwrap_or(d);
        let fx = |v: f32| (v * 65536.0).round() as u32;
        defs.push_str(&format!("const uint RND_AC = {}u;\nconst uint RND_DC = {}u;\n", fx(rnd("BRP_PRORES_RND_AC", RND_AC)), fx(rnd("BRP_PRORES_RND_DC", RND_DC))));
        let mut offs = vec![0i64; 4];
        let mut strides = vec![0i64; 4];
        for (i, pl) in layout.planes.iter().enumerate().take(4) {
            ensure!(pl.step == 1 && pl.bytes == 2, "ProRes input plane {i} not 16-bit planar");
            offs[i] = pl.offset as i64;
            strides[i] = pl.stride as i64;
        }
        defs.push_str(&uarray("PL_OFF", &offs));
        defs.push_str(&uarray("PL_STRIDE", &strides));
        let scan: Vec<i64> = if pictures == 2 { PRORES_INTERLACED_SCAN } else { PRORES_PROGRESSIVE_SCAN }.iter().map(|v| *v as i64).collect();
        defs.push_str(&uarray("SCAN", &scan));
        defs.push_str(&uarray("QL", &pr.luma.iter().map(|v| *v as i64).collect::<Vec<_>>()));
        defs.push_str(&uarray("QC", &pr.chroma.iter().map(|v| *v as i64).collect::<Vec<_>>()));
        // Codebooks shared with the GPU decoders.
        defs.push_str(&glsl_array("DCCB", &PRORES_DC_CB));
        defs.push_str(&glsl_array("RUNCB", &PRORES_RUN_CB));
        defs.push_str(&glsl_array("LEVCB", &PRORES_LEV_CB));
        defs.push_str(&format!("#define FIRST_DC_CB {}u
", PRORES_FIRST_DC_CB));
        defs.push_str(&super::dct_table());
        let work = compute.buffer(work_words as u64 * 4, "prores work")?;
        let out = Output::new(compute, cap)?;
        let [est, trellis, offsets, enc] = kernels(compute, &defs, out.cap)?;
        Ok(ProRes { compute: compute.clone(), codec: Codec::ProRes(s.clone()), s, g, est, trellis, offsets, enc, work, out, header, layout: layout.clone(), planes_addr: 0, defs })
    }

    fn push(&self, field: u32, first: bool) -> Push {
        let mut p = Push::new();
        p.0.extend(self.planes_addr.to_le_bytes());
        p.ptr(&self.work).ptr(&self.out.dev).u(field).u(first as u32)
    }

    fn record_all(&self, rec: &mut Recorder) -> Result<()> {
        let g = &self.g;
        for pic in 0..g.pictures {
            // Field pictures: the first is the top field when top field first.
            let field = pic ^ (self.s.interlaced == Some(false)) as u32;
            let push = self.push(field, pic == 0);
            let label = std::mem::take(&mut rec.label);
            rec.label = "prores est".into();
            rec.dispatch(&self.est, [g.slices, 1, 1], &push)?;
            rec.label = "prores trellis".into();
            rec.dispatch(&self.trellis, [g.mbh, 1, 1], &push)?;
            rec.label = "prores offsets".into();
            rec.dispatch(&self.offsets, [1, 1, 1], &push)?;
            rec.label = "prores enc".into();
            rec.dispatch(&self.enc, [g.slices, 1, 1], &push)?;
            rec.label = label;
        }
        let label = std::mem::replace(&mut rec.label, "packet out".into());
        self.out.record_out(rec)?;
        rec.label = label;
        Ok(())
    }
}

/// The four kernels for these defines and packet capacity.
fn kernels(compute: &Compute, defs: &str, cap: u64) -> Result<[Arc<Kernel>; 4]> {
    let defs = format!("{defs}#define CAP {cap}u\n");
    let k = |name: &str, body: &str| -> Result<Arc<Kernel>> {
        let chunks = if name == "est" || name == "enc" { "#define CHUNKS\n" } else { "" };
        let s = format!("{PRELUDE}{defs}{chunks}{COMMON}{SAMPLE}{DCT8}{KERNELS}{body}");
        compute.kernel(&format!("prores_{name}_{}", super::spec_hash(&s)), &s, 0)
    };
    Ok([k("est", EST)?, k("trellis", TRELLIS)?, k("offsets", OFFSETS)?, k("enc", ENC)?])
}

pub(crate) fn uarray(name: &str, v: &[i64]) -> String {
    let body: Vec<String> = v.iter().map(|x| format!("{x}u")).collect();
    format!("const uint {name}[{}] = uint[]({});\n", v.len(), body.join(","))
}

/// The frame header (after the 8-byte frame atom), as `prores_ks` writes it, with
/// both quantisation matrices.
fn frame_header(s: &Settings, w: u32, h: u32, c444: bool) -> Vec<u8> {
    let pr = &PROFILES[s.profile];
    let mut b = Vec::with_capacity(148);
    b.extend(148u16.to_be_bytes());
    b.extend(((c444 || s.alpha) as u16).to_be_bytes());
    b.extend(s.vendor);
    b.extend((w as u16).to_be_bytes());
    b.extend((h as u16).to_be_bytes());
    let mut flags = (if c444 { 3u8 } else { 2 }) << 6;
    match s.interlaced {
        Some(true) => flags |= 0x04,
        Some(false) => flags |= 0x08,
        None => {}
    }
    b.push(flags);
    b.push(0);
    b.extend([s.color.0, s.color.1, s.color.2]);
    b.push(if s.alpha { 2 } else { 0 });
    b.push(0);
    b.push(0x03);
    b.extend(pr.luma);
    b.extend(pr.chroma);
    debug_assert_eq!(b.len(), 148);
    b
}

impl Encoder for ProRes {
    fn record(&mut self, rec: &mut Recorder, planes: &Buffer, layout: &PackLayout) -> Result<()> {
        ensure!(layout.total == self.layout.total && layout.name == self.layout.name, "ProRes: the picture layout changed");
        self.planes_addr = planes.addr;
        self.record_all(rec)
    }

    fn packet(&mut self) -> Result<Vec<u8>> {
        for _ in 0..4 {
            let (mut pkt, over) = self.out.read()?;
            if !over {
                ensure!(pkt.len() >= FH_BYTES as usize, "ProRes: empty packet");
                let n = pkt.len() as u32;
                pkt[0..4].copy_from_slice(&n.to_be_bytes());
                pkt[4..8].copy_from_slice(b"icpf");
                pkt[8..FH_BYTES as usize].copy_from_slice(&self.header);
                return Ok(pkt);
            }
            // Bigger than the buffer: grow it and encode the same picture again.
            let cap = self.out.cap * 2;
            log::info!("GPU ProRes: packet over {} MB, growing", self.out.cap >> 20);
            self.out = Output::new(&self.compute, cap)?;
            [self.est, self.trellis, self.offsets, self.enc] = kernels(&self.compute, &self.defs, self.out.cap)?;
            let mut rec = Recorder::new(&self.compute)?;
            self.record_all(&mut rec)?;
            rec.submit_wait()?;
        }
        bail!("ProRes: packet does not fit")
    }

    fn codec(&self) -> &Codec {
        &self.codec
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn frame_header_layout() {
        let s = Settings { profile: 3, alpha: false, interlaced: Some(true), color: (1, 1, 1), vendor: *b"apl0" };
        let h = frame_header(&s, 1920, 1080, false);
        assert_eq!(h.len(), 148);
        assert_eq!(u16::from_be_bytes([h[0], h[1]]), 148);
        assert_eq!(&h[4..8], b"apl0");
        assert_eq!(u16::from_be_bytes([h[8], h[9]]), 1920);
        // 4:2:2, top field first.
        assert_eq!(h[12], 0x80 | 0x04);
        assert_eq!(h[19], 0x03);
        assert_eq!(&h[20..84], &QM_HQ[..]);
        assert_eq!(tag(3), *b"apch");
        assert_eq!(Settings::profile_of("4444xq"), Some(5));
    }
}
