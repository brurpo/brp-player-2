//! Avid DNxHD / DNxHR (SMPTE VC-3) on the GPU.
//!
//! Every profile (CID) has a fixed coded frame size; the bitstream, tables and
//! rate control follow FFmpeg's `dnxhd` encoder (LGPL; tables generated from
//! `libavcodec/dnxhddata.c` into `brp_render::gpudec::dnxhd_tables`, shared with the
//! GPU decoders):
//!
//! 1. `dct`: a thread per 8x8 block: the transform, DC (exact), luma variance per
//!    macroblock.
//! 2. `dcbits`: each block's DC difference code (DC prediction runs along a row of
//!    macroblocks and does not depend on the quantiser).
//! 3. `acbits` + `rows`, twice: the coded size of every macroblock at 16
//!    quantisers (a ladder, then the values between the two that bracket the
//!    budget), rows padded to 32 bits as in the stream.
//! 4. `decide`: the smallest quantiser whose frame fits; then, like FFmpeg's "fast"
//!    rate control, the least detailed macroblocks get the next finer quantiser
//!    while the budget lasts (by variance histogram instead of a sort).
//! 5. `final` + `rowscan` + `rowoff`: exact bits of every block at its quantiser,
//!    their offsets in each row (prefix sums) and the rows' offsets (the header's
//!    index).
//! 6. `write`: a thread per block writes its codewords at its bit offset.
//!
//! Interlaced frames (DNxHD 1080i CIDs) are two coding units, one per field.

use super::{Codec, Encoder, Output, COMMON, DCT8, HEAD, SAMPLE};
use anyhow::{bail, ensure, Result};
use brp_media::encode::PackLayout;
use brp_render::compute::{Buffer, Compute, Kernel, Push, Recorder, PRELUDE};
use std::sync::Arc;

#[derive(Clone, Debug, PartialEq)]
pub struct Settings {
    /// FFmpeg profile name (`dnxhr_hq` ...), ours for DNxHD (`dnxhd_hqx`, `dnxhd_hq`,
    /// `dnxhd_sq`, `dnxhd_lb`: the CID follows from the size), or a CID number.
    pub profile: String,
    /// Interlaced (DNxHD interlaced CIDs): Some(top field first).
    pub interlaced: Option<bool>,
}

/// The VC-3 profiles (shared with the GPU decoders: `brp_render::gpudec::dnxhd_tables`).
pub use brp_render::gpudec::dnxhd_tables::{cid, Cid, CIDS};

/// DNxHR CID of FFmpeg's profile name (`dnxhr_hq` ...).
pub fn hr_cid(profile: &str) -> Option<u32> {
    Some(match profile {
        "dnxhr_444" => 1270,
        "dnxhr_hqx" => 1271,
        "dnxhr_hq" => 1272,
        "dnxhr_sq" => 1273,
        "dnxhr_lb" => 1274,
        _ => return None,
    })
}

/// The DNxHD CID for a size, bit depth, scan and class ("hqx" 10-bit, "hq", "sq",
/// "lb"), as offered for 1080 and 720.
pub fn hd_cid(w: u32, h: u32, class: &str, interlaced: bool) -> Option<u32> {
    Some(match (w, h, interlaced, class) {
        (1920, 1080, false, "hqx") => 1235,
        (1920, 1080, false, "hq") => 1238,
        (1920, 1080, false, "sq") => 1237,
        (1920, 1080, false, "lb") => 1253,
        (1920, 1080, true, "hqx") => 1241,
        (1920, 1080, true, "hq") => 1243,
        (1920, 1080, true, _) => 1242,
        (1280, 720, false, "hqx") => 1250,
        (1280, 720, false, "hq") => 1251,
        (1280, 720, false, _) => 1252,
        (1440, 1080, false, _) => 1259,
        (1440, 1080, true, _) => 1244,
        _ => return None,
    })
}

/// Frame size of a CID at a picture size (DNxHR: from the macroblock count).
pub fn frame_size(c: &Cid, w: u32, h: u32) -> u32 {
    if c.frame_size != 0 {
        return c.frame_size;
    }
    let mbs = (h.div_ceil(16) * w.div_ceil(16)) as u64;
    let r = (mbs * c.packet_scale.0 as u64 / c.packet_scale.1 as u64) as u32;
    ((r + 2048) / 4096 * 4096).max(8192)
}

/// The DNxHD bit rate (Mbit/s) FFmpeg's encoder wants for a CID at a frame rate.
pub fn hd_bit_rate(c: &Cid, fps: f64) -> Option<u32> {
    let want = c.frame_size as f64 * 8.0 * fps / 1e6;
    c.bit_rates.iter().copied().min_by(|a, b| (*a as f64 - want).abs().total_cmp(&(*b as f64 - want).abs()))
}

const ZIGZAG: [u8; 64] = [
    0, 1, 8, 16, 9, 2, 3, 10, 17, 24, 32, 25, 18, 11, 4, 5, 12, 19, 26, 33, 40, 48, 41, 34, 27, 20, 13, 6, 7, 14, 21, 28, 35, 42, 49, 56, 57, 50, 43, 36, 29, 22, 15, 23, 30, 37, 44, 51, 58, 59, 52, 45, 38, 31, 39, 46,
    53, 60, 61, 54, 47, 55, 62, 63,
];

/// The first quantiser ladder (then the values between the two bracketing the budget).
const LADDER: [u32; 16] = [1, 2, 3, 4, 6, 8, 11, 16, 23, 32, 45, 64, 128, 256, 512, 1023];

const KERNELS: &str = r#"
#define LIMIT 0xffffffffu
layout(push_constant) uniform P { U32 src; U32 work; U32 o; U32 vlc; uint field; uint pic; uint pass; } p;

uint frame_line(uint y) { return INTERLACED != 0 ? y * 2u + p.field : y; }
uint px(uint plane, uint x, uint y) {
    uint fy = frame_line(y);
    return BITS == 8 ? sample8(p.src, PL_OFF[plane], PL_STRIDE[plane], x, fy) : sample16(p.src, PL_OFF[plane], PL_STRIDE[plane], x, fy);
}
uint lowv(uint v, uint n) { return n == 0u ? 0u : (v & ((1u << n) - 1u)); }

// Coefficients of block b (global index), 4x the orthonormal DCT, packed int16.
int coef(uint b, uint k) {
    uint w = p.work.v[COEF_OFF + b * 32u + (k >> 1)];
    return (k & 1u) == 0u ? (int(w << 16) >> 16) : (int(w) >> 16);
}
uint weight(uint comp, uint i) { return comp == 0u ? WL[i] : WC[i]; }

// The quantised level at zigzag position i (0 < i < 64) at quantiser q.
int level_of(int c, uint q, uint w) {
    int l = int((uint(abs(c)) * MK) / (q * w));
    l = min(l, int(MAXLEV) - 1);
    return c < 0 ? -l : l;
}
// Macroblock luma variance, as a histogram bin (log2 steps of 1/12).
uint mb_var(uint mb) {
    uint s = p.work.v[VAR_OFF + mb * 2u], q = p.work.v[VAR_OFF + mb * 2u + 1u];
    uint m = s >> 8, sq = q >> 8;
    return sq > m * m ? sq - m * m : 0u;
}
uint VAR_BIN(uint mb) { return min(255u, uint(log2(float(mb_var(mb)) + 1.0) * 12.0)); }
uint vlc_entry(int level, uint run) { return p.vlc.v[uint(level * 2 + int(MAXLEV) * 2) | run]; }
uint run_entry(uint run) { return p.vlc.v[RUN_OFF + run]; }

// AC bits of a block (EOB included) at quantiser q.
uint ac_bits(uint b, uint comp, uint q) {
    uint bits = 0u, last = 0u;
    for (uint i = 1u; i < 64u; i++) {
        int l = level_of(coef(b, ZZ[i]), q, weight(comp, i));
        if (l != 0) {
            uint run = i - last - 1u;
            bits += (vlc_entry(l, run > 0u ? 1u : 0u) >> 24) + (run > 0u ? (run_entry(run) >> 24) : 0u);
            last = i;
        }
    }
    return bits + (vlc_entry(0, 0u) >> 24);
}
"#;

const K_CLEAR: &str = r#"
layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;
void main() {
    uint i = gl_GlobalInvocationID.x;
    if (i < CLEAR_N) p.work.v[CLEAR_OFF + i] = 0u;
}
"#;

const K_DCT: &str = r#"
layout(local_size_x = 64, local_size_y = 1, local_size_z = 1) in;
void main() {
    uint b = gl_GlobalInvocationID.x;
    if (b >= NBLOCKS) return;
    uint mb = b / NBLK, i = b % NBLK;
    uint mbx = mb % MBW, mby = mb / MBW;
    uint comp = BCOMP[i];
    uint cw = (C444 != 0 || comp == 0u) ? 16u : 8u;
    uint pw = comp == 0u ? W : CW;
    uint bx = mbx * cw + BDX[i], by = mby * 16u + BDY[i];
    float blk[64];
    uint sum = 0u, sq = 0u;
    for (uint y = 0u; y < 8u; y++) {
        uint yy = min(by + y, FH - 1u);
        for (uint x = 0u; x < 8u; x++) {
            uint v = px(comp, min(bx + x, pw - 1u), yy);
            sum += v;
            sq += v * v;
            blk[y * 8u + x] = float(v);
        }
    }
    fdct8x8(blk);
    for (uint k = 0u; k < 64u; k += 2u) {
        int c0 = int(round(blk[k] * 4.0)), c1 = int(round(blk[k + 1u] * 4.0));
        p.work.v[COEF_OFF + b * 32u + (k >> 1)] = (uint(c0) & 0xffffu) | (uint(c1) << 16);
    }
    p.work.v[DCV_OFF + b] = (sum + 4u) >> 3;
    if (comp == 0u) {
        atomicAdd(p.work.v[VAR_OFF + mb * 2u], sum);
        atomicAdd(p.work.v[VAR_OFF + mb * 2u + 1u], sq);
    }
}
"#;

const K_DCBITS: &str = r#"
layout(local_size_x = 64, local_size_y = 1, local_size_z = 1) in;
void main() {
    uint b = gl_GlobalInvocationID.x;
    if (b >= NBLOCKS) return;
    uint mb = b / NBLK, i = b % NBLK, mbx = mb % MBW;
    uint pb = BPRED[i];
    int prev;
    if (pb < 64u) prev = int(p.work.v[DCV_OFF + mb * NBLK + pb]);
    else if (mbx > 0u) prev = int(p.work.v[DCV_OFF + (mb - 1u) * NBLK + (pb - 64u)]);
    else prev = int(1u << (BITS + 2u));
    int diff = int(p.work.v[DCV_OFF + b]) - prev;
    uint nbits = diff == 0 ? 0u : uint(findMSB(uint(abs(diff)))) + 1u;
    p.work.v[DCB_OFF + b] = DCBITS[nbits] + nbits;
}
"#;

const K_ACBITS: &str = r#"
layout(local_size_x = 64, local_size_y = 1, local_size_z = 1) in;
void main() {
    uint b = gl_GlobalInvocationID.x;
    if (b >= NBLOCKS) return;
    uint mb = b / NBLK, i = b % NBLK, comp = BCOMP[i];
    uint extra = p.work.v[DCB_OFF + b] + (i == 0u ? 12u : 0u);
    uint lst = p.pass == 0u ? QL1_OFF : QL2_OFF;
    uint mbb = p.pass == 0u ? MBB1_OFF : MBB2_OFF;
    // All 16 quantisers in one pass over the coefficients (the list ascends: once a
    // coefficient quantises to 0, it does for the coarser ones too). Same levels and
    // codes as ac_bits().
    uint q[16], bits[16], last[16];
    for (uint qi = 0u; qi < 16u; qi++) { q[qi] = p.work.v[lst + qi]; bits[qi] = 0u; last[qi] = 0u; }
    for (uint k = 1u; k < 64u; k++) {
        int c = coef(b, ZZ[k]);
        uint v = uint(abs(c)) * MK, w = weight(comp, k);
        for (uint qi = 0u; qi < 16u; qi++) {
            uint qw = q[qi] * w;
            if (v < qw) break;
            int l = min(int(v / qw), int(MAXLEV) - 1);
            if (c < 0) l = -l;
            uint run = k - last[qi] - 1u;
            bits[qi] += (vlc_entry(l, run > 0u ? 1u : 0u) >> 24) + (run > 0u ? (run_entry(run) >> 24) : 0u);
            last[qi] = k;
        }
    }
    uint eob = vlc_entry(0, 0u) >> 24;
    for (uint qi = 0u; qi < 16u; qi++) atomicAdd(p.work.v[mbb + mb * 16u + qi], bits[qi] + eob + extra);
}
"#;

const K_ROWS: &str = r#"
layout(local_size_x = 64, local_size_y = 1, local_size_z = 1) in;
void main() {
    uint t = gl_GlobalInvocationID.x;
    if (t >= MBH * 16u) return;
    uint row = t / 16u, qi = t % 16u;
    uint mbb = p.pass == 0u ? MBB1_OFF : MBB2_OFF;
    uint tot = p.pass == 0u ? TOT1_OFF : TOT2_OFF;
    uint bits = 0u;
    for (uint x = 0u; x < MBW; x++) bits += p.work.v[mbb + (row * MBW + x) * 16u + qi];
    atomicAdd(p.work.v[tot + qi], (bits + 31u) & ~31u);
}
"#;

// Pass 0 done: the values between the ladder's two bracketing quantisers.
// Pass 1 done: the frame quantiser, the finer one and the budget left.
const K_DECIDE: &str = r#"
layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;
void main() {
    if (p.pass == 0u) {
        uint i = 0u;
        while (i < 16u && p.work.v[TOT1_OFF + i] > FRAME_BITS) i++;
        uint hi = i < 16u ? p.work.v[QL1_OFF + i] : 1023u;
        uint lo = i > 0u ? p.work.v[QL1_OFF + i - 1u] : 0u;
        // 16 values in (lo, hi], ascending, ending at hi.
        uint span = hi - lo;
        for (uint k = 0u; k < 16u; k++) {
            uint q = span <= 16u ? min(lo + 1u + k, hi) : lo + ((k + 1u) * span + 15u) / 16u;
            p.work.v[QL2_OFF + k] = max(q, 1u);
        }
        p.work.v[ST_OFF + 4u] = i;
    } else {
        uint i = 0u;
        while (i < 15u && p.work.v[TOT2_OFF + i] > FRAME_BITS) i++;
        uint q = p.work.v[QL2_OFF + i];
        uint total = p.work.v[TOT2_OFF + i];
        // The finer quantiser: the one before in this list, else the ladder's.
        uint fq = 0u, fslot = LIMIT;
        if (i > 0u && p.work.v[QL2_OFF + i - 1u] < q) { fq = p.work.v[QL2_OFF + i - 1u]; fslot = 16u + i - 1u; }
        else if (p.work.v[ST_OFF + 4u] > 0u) { fq = p.work.v[QL1_OFF + p.work.v[ST_OFF + 4u] - 1u]; fslot = p.work.v[ST_OFF + 4u] - 1u; }
        p.work.v[ST_OFF + 0u] = q;
        p.work.v[ST_OFF + 1u] = 16u + i;
        p.work.v[ST_OFF + 2u] = fslot;
        p.work.v[ST_OFF + 3u] = fq;
        uint slack = 32u * MBH;
        p.work.v[ST_OFF + 5u] = (fslot != LIMIT && total + slack < FRAME_BITS) ? FRAME_BITS - total - slack : 0u;
        p.work.v[ST_OFF + 6u] = total;
    }
}
"#;

const K_HIST: &str = r#"
layout(local_size_x = 64, local_size_y = 1, local_size_z = 1) in;
uint mb_bits(uint mb, uint slot) { return slot < 16u ? p.work.v[MBB1_OFF + mb * 16u + slot] : p.work.v[MBB2_OFF + mb * 16u + slot - 16u]; }
void main() {
    uint mb = gl_GlobalInvocationID.x;
    if (mb >= NMB || p.work.v[ST_OFF + 2u] == LIMIT) return;
    uint fine = mb_bits(mb, p.work.v[ST_OFF + 2u]), coarse = mb_bits(mb, p.work.v[ST_OFF + 1u]);
    uint extra = fine > coarse ? fine - coarse : 0u;
    atomicAdd(p.work.v[HIST_OFF + VAR_BIN(mb)], extra);
}
"#;

const K_THRESH: &str = r#"
layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;
void main() {
    uint budget = p.work.v[ST_OFF + 5u], acc = 0u, tb = 0u;
    for (; tb < 256u; tb++) {
        uint e = p.work.v[HIST_OFF + tb];
        if (acc + e > budget) break;
        acc += e;
    }
    p.work.v[ST_OFF + 7u] = tb;
}
"#;

const K_FINAL: &str = r#"
layout(local_size_x = 64, local_size_y = 1, local_size_z = 1) in;
void main() {
    uint b = gl_GlobalInvocationID.x;
    if (b >= NBLOCKS) return;
    uint mb = b / NBLK, i = b % NBLK;
    uint q = p.work.v[ST_OFF + 0u];
    if (p.work.v[ST_OFF + 2u] != LIMIT && VAR_BIN(mb) < p.work.v[ST_OFF + 7u]) q = p.work.v[ST_OFF + 3u];
    if (i == 0u) p.work.v[MBQ_OFF + mb] = q;
    p.work.v[BLB_OFF + b] = ac_bits(b, BCOMP[i], q) + p.work.v[DCB_OFF + b] + (i == 0u ? 12u : 0u);
}
"#;

// A workgroup per row: bit offsets of its blocks, and the row's padded bytes.
const K_ROWSCAN: &str = r#"
layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;
shared uint part[256];
void main() {
    uint row = gl_WorkGroupID.x, t = gl_LocalInvocationID.x;
    uint n = MBW * NBLK, b0 = row * n;
    uint per = (n + 255u) / 256u, s0 = min(t * per, n), s1 = min(s0 + per, n);
    uint sum = 0u;
    for (uint s = s0; s < s1; s++) sum += p.work.v[BLB_OFF + b0 + s];
    part[t] = sum;
    barrier();
    if (t == 0u) {
        uint acc = 0u;
        for (uint i = 0u; i < 256u; i++) { uint v = part[i]; part[i] = acc; acc += v; }
        p.work.v[ROWB_OFF + row] = ((acc + 31u) & ~31u) >> 3;
    }
    barrier();
    uint off = part[t];
    for (uint s = s0; s < s1; s++) {
        p.work.v[BLO_OFF + b0 + s] = off;
        off += p.work.v[BLB_OFF + b0 + s];
    }
}
"#;

const K_ROWOFF: &str = r#"
layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;
void main() {
    uint base = p.pic * CU_SIZE, off = 0u;
    for (uint r = 0u; r < MBH; r++) {
        p.work.v[ROWO_OFF + r] = off;
        put_be32(p.o, HEAD + base + 0x170u + 4u * r, off);
        off += p.work.v[ROWB_OFF + r];
    }
    if (DATA_OFFSET + off + 4u > CU_SIZE) p.o.v[2] = 1u;
    p.o.v[0] = (p.pic + 1u) * CU_SIZE;
}
"#;

const K_WRITE: &str = r#"
layout(local_size_x = 64, local_size_y = 1, local_size_z = 1) in;
void main() {
    uint b = gl_GlobalInvocationID.x;
    if (b >= NBLOCKS || p.o.v[2] != 0u) return;
    uint mb = b / NBLK, i = b % NBLK, mbx = mb % MBW, row = mb / MBW;
    uint comp = BCOMP[i];
    uint q = p.work.v[MBQ_OFF + mb];
    uint bit = (HEAD + p.pic * CU_SIZE + DATA_OFFSET + p.work.v[ROWO_OFF + row]) * 8u + p.work.v[BLO_OFF + b];
    BitW w;
    w.acc = 0ul; w.n = bit & 7u; w.pos = bit >> 3;
    if (i == 0u) {
        bw_put(p.o, w, 11u, q);
        bw_put(p.o, w, 1u, ACT);
    }
    // DC difference.
    uint pb = BPRED[i];
    int prev;
    if (pb < 64u) prev = int(p.work.v[DCV_OFF + mb * NBLK + pb]);
    else if (mbx > 0u) prev = int(p.work.v[DCV_OFF + (mb - 1u) * NBLK + (pb - 64u)]);
    else prev = int(1u << (BITS + 2u));
    int diff = int(p.work.v[DCV_OFF + b]) - prev;
    uint nbits = diff == 0 ? 0u : uint(findMSB(uint(abs(diff)))) + 1u;
    if (diff < 0) diff -= 1;
    bw_put(p.o, w, DCBITS[nbits] + nbits, (DCCODES[nbits] << nbits) + lowv(uint(diff), nbits));
    // AC codewords, then EOB.
    uint last = 0u;
    for (uint k = 1u; k < 64u; k++) {
        int l = level_of(coef(b, ZZ[k]), q, weight(comp, k));
        if (l != 0) {
            uint run = k - last - 1u;
            uint e = vlc_entry(l, run > 0u ? 1u : 0u);
            bw_put(p.o, w, e >> 24, e & 0xffffffu);
            if (run > 0u) {
                uint r = run_entry(run);
                bw_put(p.o, w, r >> 24, r & 0xffffffu);
            }
            last = k;
        }
    }
    uint eob = vlc_entry(0, 0u);
    bw_put(p.o, w, eob >> 24, eob & 0xffffffu);
    bw_flush(p.o, w);
}
"#;

struct Geo {
    pictures: u32,
    mbh: u32,
    nblocks: u32,
    nmb: u32,
}

pub struct Dnx {
    codec: Codec,
    s: Settings,
    g: Geo,
    kernels: Vec<(&'static str, Arc<Kernel>)>,
    work: Buffer,
    vlc: Buffer,
    out: Output,
    header: Vec<u8>,
    cu_size: u32,
    layout: PackLayout,
    planes_addr: u64,
    clear_n: u32,
}

impl Dnx {
    pub fn new(compute: &Arc<Compute>, s: Settings, layout: &PackLayout) -> Result<Dnx> {
        let (w, h) = (layout.w, layout.h);
        let n = resolve_cid(&s.profile, w, h, s.interlaced.is_some()).ok_or_else(|| anyhow::anyhow!("DNx profile {} has no CID for {w}x{h}{}", s.profile, if s.interlaced.is_some() { " interlaced" } else { "" }))?;
        let c = cid(n).ok_or_else(|| anyhow::anyhow!("DNx CID {n} is not supported"))?;
        let c444 = layout.cw == 0;
        ensure!(c444 == c.is_444, "DNx CID {}: chroma 4:{}", c.cid, if c444 { "4:4" } else { "2:2" });
        let bits = if c.bit_depth != 0 { c.bit_depth } else { 10 };
        let bits = if c.cid >= 1272 { 8 } else { bits };
        ensure!(layout.bits as u32 == bits && layout.yuv && !layout.alpha, "DNx CID {} takes {bits}-bit Y'CbCr, not {}", c.cid, layout.name);
        if c.width != 0 {
            ensure!(c.width == w && c.height == h, "DNxHD CID {} is {}x{}, not {w}x{h}", c.cid, c.width, c.height);
        } else {
            ensure!(w >= 256 && h >= 120, "DNxHR needs at least 256x120");
        }
        ensure!(s.interlaced.is_some() == c.interlaced, "DNx CID {}: {}", c.cid, if c.interlaced { "interlaced only" } else { "progressive only" });
        let pictures = if c.interlaced { 2 } else { 1 };
        let fh = h / pictures;
        let mbw = w.div_ceil(16);
        let mbh = h.div_ceil(16) / pictures;
        let nblk = if c444 { 12 } else { 8 };
        let nmb = mbw * mbh;
        let nblocks = nmb * nblk;
        let fsize = frame_size(c, w, h);
        let cu_size = if c.frame_size != 0 { c.cu_size } else { fsize };
        let data_offset = if mbh > 68 { 0x170 + mbh * 4 } else { 0x280 };
        let frame_bits = (cu_size - data_offset - 4) * 8;
        // Quantisation: level = |4T| * MK / (q w): 8-bit and 4:4:4/HQX scale 32 T / (q w),
        // 10-bit DNxHD 8 T / (q w) (FFmpeg's encoder, VC-3's p/s).
        let hqx_like = c444 || c.cid == 1271;
        let mk = if bits == 8 || hqx_like { 8 } else { 2 };
        let maxlev = 1u32 << (bits + 2);

        // Work buffer regions (u32 words).
        let off = std::cell::Cell::new(0u32);
        let region = |n: u32| {
            let o = off.get();
            off.set(o + n);
            o
        };
        let coef = region(nblocks * 32);
        let dcv = region(nblocks);
        let dcb = region(nblocks);
        let blb = region(nblocks);
        let blo = region(nblocks);
        let mbq = region(nmb);
        let rowb = region(mbh);
        let rowo = region(mbh);
        let ql1 = region(16);
        let ql2 = region(16);
        // Cleared each picture: from here to the end.
        let clear_off = off.get();
        let var = region(nmb * 2);
        let mbb1 = region(nmb * 16);
        let mbb2 = region(nmb * 16);
        let tot1 = region(16);
        let tot2 = region(16);
        let hist = region(256);
        let st = region(16);
        let clear_n = off.get() - clear_off;
        let words = off.get();

        let mut d = String::new();
        let mut def = |k: &str, v: u32| d.push_str(&format!("#define {k} {v}u\n"));
        def("W", w);
        def("CW", if c444 { w } else { w.div_ceil(2) });
        def("FH", fh);
        def("MBW", mbw);
        def("MBH", mbh);
        def("NMB", nmb);
        def("NBLK", nblk);
        def("NBLOCKS", nblocks);
        def("BITS", bits);
        def("MK", mk);
        def("MAXLEV", maxlev);
        def("RUN_OFF", maxlev * 4);
        def("CU_SIZE", cu_size);
        def("DATA_OFFSET", data_offset);
        def("FRAME_BITS", frame_bits);
        def("HEAD", HEAD as u32);
        def("ACT", (c444) as u32);
        for (k, v) in [("COEF_OFF", coef), ("DCV_OFF", dcv), ("DCB_OFF", dcb), ("BLB_OFF", blb), ("BLO_OFF", blo), ("MBQ_OFF", mbq), ("ROWB_OFF", rowb), ("ROWO_OFF", rowo), ("QL1_OFF", ql1), ("QL2_OFF", ql2), ("VAR_OFF", var), ("MBB1_OFF", mbb1), ("MBB2_OFF", mbb2), ("TOT1_OFF", tot1), ("TOT2_OFF", tot2), ("HIST_OFF", hist), ("ST_OFF", st), ("CLEAR_OFF", clear_off), ("CLEAR_N", clear_n)] {
            def(k, v);
        }
        d.push_str(&format!("#define INTERLACED {}\n#define C444 {}\n", c.interlaced as u32, c444 as u32));
        let mut offs = vec![0i64; 3];
        let mut strides = vec![0i64; 3];
        for (i, pl) in layout.planes.iter().enumerate().take(3) {
            ensure!(pl.step == 1, "DNx input not planar");
            offs[i] = pl.offset as i64;
            strides[i] = pl.stride as i64;
        }
        d.push_str(&super::prores::uarray("PL_OFF", &offs));
        d.push_str(&super::prores::uarray("PL_STRIDE", &strides));
        d.push_str(&super::prores::uarray("ZZ", &ZIGZAG.iter().map(|v| *v as i64).collect::<Vec<_>>()));
        d.push_str(&super::prores::uarray("WL", &c.luma.iter().map(|v| *v as i64).collect::<Vec<_>>()));
        d.push_str(&super::prores::uarray("WC", &c.chroma.iter().map(|v| *v as i64).collect::<Vec<_>>()));
        let mut dcc = c.dc_codes.iter().map(|v| *v as i64).collect::<Vec<_>>();
        let mut dcb_t = c.dc_bits.iter().map(|v| *v as i64).collect::<Vec<_>>();
        dcc.resize(16, 0);
        dcb_t.resize(16, 0);
        d.push_str(&super::prores::uarray("DCCODES", &dcc));
        d.push_str(&super::prores::uarray("DCBITS", &dcb_t));
        // Blocks of a macroblock: component, offset in its plane's macroblock, and
        // the block before it of the same component (64 + i: in the macroblock before).
        let (comp, dx, dy, pred): (Vec<i64>, Vec<i64>, Vec<i64>, Vec<i64>) = if c444 {
            (vec![0, 0, 1, 1, 2, 2, 0, 0, 1, 1, 2, 2], vec![0, 8, 0, 8, 0, 8, 0, 8, 0, 8, 0, 8], vec![0, 0, 0, 0, 0, 0, 8, 8, 8, 8, 8, 8], vec![64 + 7, 0, 64 + 9, 2, 64 + 11, 4, 1, 6, 3, 8, 5, 10])
        } else {
            (vec![0, 0, 1, 2, 0, 0, 1, 2], vec![0, 8, 0, 0, 0, 8, 0, 0], vec![0, 0, 0, 0, 8, 8, 8, 8], vec![64 + 5, 0, 64 + 6, 64 + 7, 1, 4, 2, 3])
        };
        d.push_str(&super::prores::uarray("BCOMP", &comp));
        d.push_str(&super::prores::uarray("BDX", &dx));
        d.push_str(&super::prores::uarray("BDY", &dy));
        d.push_str(&super::prores::uarray("BPRED", &pred));
        d.push_str(&super::dct_table());

        let work = compute.buffer(words as u64 * 4, "dnx work")?;
        // Quantiser ladder (constant).
        {
            let mut init = vec![0u8; 64];
            for (i, q) in LADDER.iter().enumerate() {
                init[i * 4..i * 4 + 4].copy_from_slice(&q.to_le_bytes());
            }
            let up = compute.upload(&init, "dnx ladder")?;
            let mut rec = Recorder::new(compute)?;
            rec.copy(&up, 0, &work, ql1 as u64 * 4, 64);
            rec.submit_wait()?;
        }
        let vlc = compute.upload(&vlc_table(c, bits, maxlev), "dnx vlc")?;
        let out = Output::new(compute, cu_size as u64 * pictures as u64 + 4096)?;
        let mut kernels = Vec::new();
        for (name, body) in [
            ("clear", K_CLEAR),
            ("dct", K_DCT),
            ("dcbits", K_DCBITS),
            ("acbits", K_ACBITS),
            ("rows", K_ROWS),
            ("decide", K_DECIDE),
            ("hist", K_HIST),
            ("thresh", K_THRESH),
            ("final", K_FINAL),
            ("rowscan", K_ROWSCAN),
            ("rowoff", K_ROWOFF),
            ("write", K_WRITE),
        ] {
            let src = format!("{PRELUDE}{d}{COMMON}{SAMPLE}{DCT8}{KERNELS}{body}");
            kernels.push((name, compute.kernel(&format!("dnx_{name}_{}", super::spec_hash(&src)), &src, 0)?));
        }
        let header = header(c, &s, w, h, mbh, data_offset, bits, c444);
        Ok(Dnx { codec: Codec::Dnx(s.clone()), s, g: Geo { pictures, mbh, nblocks, nmb }, kernels, work, vlc, out, header, cu_size, layout: layout.clone(), planes_addr: 0, clear_n })
    }

    fn k(&self, name: &str) -> &Kernel {
        &self.kernels.iter().find(|(n, _)| *n == name).unwrap().1
    }

    fn push(&self, field: u32, pic: u32, pass: u32) -> Push {
        let mut p = Push::new();
        p.0.extend(self.planes_addr.to_le_bytes());
        p.ptr(&self.work).ptr(&self.out.dev).ptr(&self.vlc).u(field).u(pic).u(pass)
    }

    fn record_all(&self, rec: &mut Recorder) -> Result<()> {
        let g = &self.g;
        let label = std::mem::take(&mut rec.label);
        let blocks = |k: &Kernel| Recorder::groups(k, [g.nblocks, 1, 1]);
        for pic in 0..g.pictures {
            let field = pic ^ (self.s.interlaced == Some(false)) as u32;
            let p0 = self.push(field, pic, 0);
            let p1 = self.push(field, pic, 1);
            rec.label = "dnx prepare".into();
            rec.dispatch(self.k("clear"), Recorder::groups(self.k("clear"), [self.clear_n, 1, 1]), &p0)?;
            rec.dispatch(self.k("dct"), blocks(self.k("dct")), &p0)?;
            rec.dispatch(self.k("dcbits"), blocks(self.k("dcbits")), &p0)?;
            rec.label = "dnx rate".into();
            for (pass, push) in [(0, &p0), (1, &p1)] {
                let _ = pass;
                rec.dispatch(self.k("acbits"), blocks(self.k("acbits")), push)?;
                rec.dispatch(self.k("rows"), Recorder::groups(self.k("rows"), [g.mbh * 16, 1, 1]), push)?;
                rec.dispatch(self.k("decide"), [1, 1, 1], push)?;
            }
            rec.dispatch(self.k("hist"), Recorder::groups(self.k("hist"), [g.nmb, 1, 1]), &p1)?;
            rec.dispatch(self.k("thresh"), [1, 1, 1], &p1)?;
            rec.label = "dnx write".into();
            rec.dispatch(self.k("final"), blocks(self.k("final")), &p1)?;
            rec.dispatch(self.k("rowscan"), [g.mbh, 1, 1], &p1)?;
            rec.dispatch(self.k("rowoff"), [1, 1, 1], &p1)?;
            rec.dispatch(self.k("write"), blocks(self.k("write")), &p1)?;
        }
        rec.label = "packet out".into();
        self.out.record_out(rec)?;
        rec.label = label;
        Ok(())
    }
}

/// FFmpeg's VLC table for (level, run flag): code | bits << 24, levels in
/// [-maxlev, maxlev) (index (level * 2 | run) + 2 maxlev), then the run codes.
fn vlc_table(c: &Cid, bits: u32, maxlev: u32) -> Vec<u8> {
    let maxlev = maxlev as i32;
    let mut t = vec![0u32; (maxlev as usize) * 4 + 64];
    for level in -maxlev..maxlev {
        for run in 0..2 {
            let index = ((level * 2) | run) + maxlev * 2;
            let (sign, mut alevel) = (level < 0, level.unsigned_abs());
            let mut offset = 0u32;
            if alevel > 64 {
                offset = (alevel - 1) >> 6;
                alevel -= offset << 6;
            }
            let mut code = 0u32;
            let mut len = 0u32;
            let mut found = false;
            for j in 0..257 {
                let (info, flags) = (c.ac_info[2 * j] as u32, c.ac_info[2 * j + 1]);
                if info >> 1 == alevel && (offset == 0 || flags & 1 != 0) && (run == 0 || flags & 2 != 0) {
                    if alevel != 0 {
                        code = ((c.ac_codes[j] as u32) << 1) | sign as u32;
                        len = c.ac_bits[j] as u32 + 1;
                    } else {
                        code = c.ac_codes[j] as u32;
                        len = c.ac_bits[j] as u32;
                    }
                    found = true;
                    break;
                }
            }
            if !found {
                continue;
            }
            if offset != 0 {
                code = (code << c.index_bits) | offset;
                len += c.index_bits;
            }
            t[index as usize] = code | len << 24;
        }
    }
    let _ = bits;
    let run_off = maxlev as usize * 4;
    for i in 0..62 {
        let run = c.run[i] as usize;
        t[run_off + run] = c.run_codes[i] as u32 | (c.run_bits[i] as u32) << 24;
    }
    t.iter().flat_map(|v| v.to_le_bytes()).collect()
}

/// The coding unit header as FFmpeg writes it (the row index at 0x170 comes from
/// the GPU), for field `field` (0 = first coded).
#[allow(clippy::too_many_arguments)]
fn header(c: &Cid, s: &Settings, w: u32, h: u32, mbh: u32, data_offset: u32, bits: u32, c444: bool) -> Vec<u8> {
    let mut b = vec![0u8; 0x170];
    b[2..4].copy_from_slice(&(data_offset as u16).to_be_bytes());
    b[4] = if (1270..=1274).contains(&c.cid) { 0x03 } else { 0x01 };
    let il = s.interlaced.is_some() as u32;
    b[5] = 0x01; // per field below
    b[6] = 0x80;
    b[7] = 0xa0;
    b[0x18..0x1a].copy_from_slice(&((h >> il) as u16).to_be_bytes());
    b[0x1a..0x1c].copy_from_slice(&(w as u16).to_be_bytes());
    b[0x1d..0x1f].copy_from_slice(&((h >> il) as u16).to_be_bytes());
    b[0x21] = if bits == 10 { 0x58 } else { 0x38 };
    b[0x22] = 0x88 + ((il as u8) << 2);
    b[0x28..0x2c].copy_from_slice(&c.cid.to_be_bytes());
    b[0x2c] = ((il == 0) as u8) << 7 | (c444 as u8) << 6 | c444 as u8;
    b[0x5f] = 0x01;
    b[0x167] = 0x02;
    b[0x16a..0x16c].copy_from_slice(&((mbh * 4 + 4) as u16).to_be_bytes());
    b[0x16c..0x16e].copy_from_slice(&(mbh as u16).to_be_bytes());
    b[0x16f] = 0x10;
    b
}

impl Encoder for Dnx {
    fn record(&mut self, rec: &mut Recorder, planes: &Buffer, layout: &PackLayout) -> Result<()> {
        ensure!(layout.total == self.layout.total && layout.name == self.layout.name, "DNx: the picture layout changed");
        self.planes_addr = planes.addr;
        self.record_all(rec)
    }

    fn packet(&mut self) -> Result<Vec<u8>> {
        let (mut pkt, over) = self.out.read()?;
        if over {
            bail!("DNx: the picture did not fit the profile's frame size");
        }
        let cu = self.cu_size as usize;
        ensure!(pkt.len() == cu * self.g.pictures as usize, "DNx: packet size {} (want {})", pkt.len(), cu * self.g.pictures as usize);
        for pic in 0..self.g.pictures as usize {
            let base = pic * cu;
            pkt[base..base + 0x170].copy_from_slice(&self.header);
            if let Some(tff) = self.s.interlaced {
                // Second field coded second; cur_field 1 is the bottom field.
                let cur_field = (pic as u8) ^ (!tff) as u8;
                pkt[base + 5] = cur_field + 2;
            }
            pkt[base + cu - 4..base + cu].copy_from_slice(&0x600DC0DEu32.to_be_bytes());
        }
        Ok(pkt)
    }

    fn codec(&self) -> &Codec {
        &self.codec
    }
}

/// The CID of a profile name at a size (see [`Settings::profile`]).
pub fn resolve_cid(profile: &str, w: u32, h: u32, interlaced: bool) -> Option<u32> {
    if let Ok(n) = profile.parse() {
        return Some(n);
    }
    if let Some(c) = hr_cid(profile) {
        return Some(c);
    }
    hd_cid(w, h, profile.strip_prefix("dnxhd_")?, interlaced)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cids_and_rates() {
        // DNxHD: CID from size, class and scan; FFmpeg's bit rate from the frame rate.
        assert_eq!(resolve_cid("dnxhd_hq", 1920, 1080, false), Some(1238));
        assert_eq!(resolve_cid("dnxhd_hqx", 1920, 1080, true), Some(1241));
        assert_eq!(resolve_cid("dnxhd_sq", 1280, 720, false), Some(1252));
        assert_eq!(resolve_cid("dnxhd_hq", 1280, 720, true), None);
        assert_eq!(resolve_cid("dnxhr_hqx", 3840, 2160, false), Some(1271));
        let c = cid(1238).unwrap();
        assert_eq!(hd_bit_rate(c, 25.0), Some(185));
        assert_eq!(hd_bit_rate(c, 30000.0 / 1001.0), Some(220));
        assert_eq!(hd_bit_rate(cid(1242).unwrap(), 30000.0 / 1001.0), Some(145));
        // DNxHR frame size as FFmpeg computes it (UHD HQX: 1,646,592 bytes... see ff_dnxhd_get_hr_frame_size).
        let hqx = cid(1271).unwrap();
        let mbs = 2160u64.div_ceil(16) * 3840u64.div_ceil(16);
        let want = ((mbs * 28672 / 255) as u32 + 2048) / 4096 * 4096;
        assert_eq!(frame_size(hqx, 3840, 2160), want);
    }

    #[test]
    fn vlc_table_matches_ffmpeg_rules() {
        // Level 1 without run: the first AC code (ac_info 3), plus the sign bit.
        let c = cid(1235).unwrap();
        let t = vlc_table(c, 10, 4096);
        let word = |i: usize| u32::from_le_bytes(t[i * 4..i * 4 + 4].try_into().unwrap());
        let entry = word(2 * 4096 + 2);
        assert_eq!(entry >> 24, c.ac_bits[0] as u32 + 1);
        assert_eq!(entry & 0xffffff, (c.ac_codes[0] as u32) << 1);
        // EOB (level 0, no run) has no sign bit.
        let eob = word(2 * 4096);
        assert_eq!(eob >> 24, c.ac_bits[c.eob_index as usize] as u32);
    }
}
