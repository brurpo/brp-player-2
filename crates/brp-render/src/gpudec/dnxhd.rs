//! Avid DNxHD and DNxHR (VC-3) on the GPU: 8 and 10-bit, 4:2:2 and 4:4:4,
//! progressive and interlaced (two field coding units per frame).
//!
//! Each macroblock row is its own bitstream (the header lists where each starts),
//! so the GPU decodes them in parallel, one thread per row (`dnx_vlc`): DC
//! differences and run/level codes (Huffman, through lookup tables made from the
//! profile's code tables), dequantised as FFmpeg's `dnxhddec.c` does. Then one thread
//! per 8x8 block runs FFmpeg's integer IDCT (`simple_idct`, put variant) and writes
//! the planes (`dnx_idct`). The arithmetic is FFmpeg's, so pictures are identical.
//!
//! FFmpeg decodes what this does not take: the CIDs missing from the table, alpha,
//! the adaptive colour transform (RGB 4:4:4), MBAFF (CID 1260) and 12-bit (no file
//! to check it against).

use super::dnxhd_tables::{cid, Cid};
use super::{Io, BITS_GLSL};
use crate::compute::{Buffer, Compute, Push, Recorder};
use anyhow::{bail, ensure, Result};
use brp_media::gpudec::{pix, GpuDecoder, GpuPicture};
use brp_media::FrameLayout;
use std::sync::Arc;

/// Parameter words (see the kernels' `prm()` reads).
const P_WEIGHTS: usize = 32; // luma 64, chroma 64
const P_ZIGZAG: usize = 160;
const P_ACINFO: usize = 224; // 514
const P_RUN: usize = 738; // 62
const P_ROWS: usize = 800;

const VLC: &str = r#"
layout(local_size_x = 32) in;
layout(push_constant) uniform P { U32 src; U32 prm; U32 lut; U32 co; U32 err; } p;
uint prm(uint i) { return p.prm.v[i]; }

// The profile's tables (weights, zigzag, AC info, runs: words P_WEIGHTS..P_ROWS),
// and each thread's block being built.
shared uint tab[P_ROWS - P_WEIGHTS];
shared uint blk_sh[32][32];
// First-level code tables (DC, AC, run), 10 bits each.
shared uint sl[3][1024];
uint tb(uint i) { return tab[i - P_WEIGHTS]; }

// Bit reader with a 64-bit cache in registers (one global load per 32 bits).
uint64_t bc;
uint bn, bw;
void br_init(uint pos) {
    bw = pos >> 5;
    bc = 0ul;
    bn = 0u;
    for (int k = 0; k < 2; k++) { bc |= uint64_t(bswap(p.src.v[bw++])) << (32u - bn); bn += 32u; }
    bc <<= pos & 31u;
    bn -= pos & 31u;
}
void refill() {
    if (bn <= 32u) { bc |= uint64_t(bswap(p.src.v[bw++])) << (32u - bn); bn += 32u; }
}
uint show(uint k) { return uint(bc >> (64u - k)); }
void skip(uint k) { bc <<= k; bn -= k; }
uint get(uint k) { refill(); uint v = show(k); skip(k); return v; }

// Next prefix code through a lookup table at word `base` indexed by `maxb` bits:
// entry = symbol | length << 16 (length 0: no such code).
uint vlc(uint kind, uint base, uint maxb, out bool ok) {
    refill();
    uint e = sl[kind][show(10u)];
    if (e == 0xffffffffu) e = p.lut.v[base + show(maxb)];
    uint len = e >> 16;
    ok = len != 0u;
    skip(len);
    return e & 0xffffu;
}

void put(uint l, uint i, int v) {
    uint w = i >> 1, sh = (i & 1u) * 16u;
    blk_sh[l][w] = (blk_sh[l][w] & ~(0xffffu << sh)) | ((uint(v) & 0xffffu) << sh);
}

void main() {
    // One row per workgroup: the lanes load the tables, then one decodes (rows in one
    // warp would diverge at every data-dependent branch).
    uint t = gl_WorkGroupID.x, l = gl_LocalInvocationID.x;
    for (uint i = l; i < P_ROWS - P_WEIGHTS; i += 32u) tab[i] = prm(P_WEIGHTS + i);
    for (uint i = l; i < 3072u; i += 32u) sl[i >> 10][i & 1023u] = p.lut.v[prm(25u) + i];
    barrier();
    uint mbw = prm(0u), mbh = prm(1u), fields = prm(2u), nb = prm(3u);
    if (l != 0u || t >= mbh * fields) return;
    uint bits = prm(4u), index_bits = prm(5u);
    int level_bias = int(prm(6u)), level_shift = int(prm(7u)), dc_shift = int(prm(8u));
    uint eob = prm(9u);
    bool is444 = prm(10u) != 0u;
    bool act = prm(24u) != 0u;
    uint dc_maxb = prm(13u), ac_maxb = prm(14u), run_maxb = prm(15u);
    uint dc_lut = 0u, ac_lut = prm(22u), run_lut = prm(23u);
    br_init(prm(P_ROWS + t) * 8u);
    int dc0 = 1 << (bits + 2u), dc1 = dc0, dc2 = dc0;
    uint blk = t * mbw * nb;
    for (uint x = 0u; x < mbw; x++) {
        int qscale = int(get(11u));
        // With the adaptive colour transform allowed, the macroblock bit set means Y'CbCr
        // (FFmpeg outputs YUV 4:4:4 when every macroblock has it); clear means RGB,
        // which FFmpeg decodes.
        if (get(1u) == 0u && act) { p.err.v[0] = 5u; return; }
        for (uint n = 0u; n < nb; n++, blk++) {
            uint comp;
            if (!is444) comp = (n & 2u) != 0u ? 1u + (n & 1u) : 0u;
            else comp = (n >> 1) % 3u;
            uint wbase = P_WEIGHTS + (comp != 0u ? 64u : 0u);
            for (uint i = 0u; i < 32u; i++) blk_sh[l][i] = 0u;
            bool ok;
            uint len = vlc(0u, dc_lut, dc_maxb, ok);
            if (!ok) { p.err.v[0] = 1u; return; }
            if (len != 0u) {
                uint v = get(len);
                int level = (v >> (len - 1u)) != 0u ? int(v) : int(v) - int((1u << len) - 1u);
                level *= 1 << dc_shift;
                if (comp == 0u) dc0 += level; else if (comp == 1u) dc1 += level; else dc2 += level;
            }
            put(l, 0u, s16(comp == 0u ? dc0 : comp == 1u ? dc1 : dc2));
            int i = 0;
            uint idx = vlc(1u, ac_lut, ac_maxb, ok);
            if (!ok) { p.err.v[0] = 2u; return; }
            while (idx != eob) {
                int level = int(tb(P_ACINFO + 2u * idx));
                uint flags = tb(P_ACINFO + 2u * idx + 1u);
                int sign = -int(get(1u));
                if ((flags & 1u) != 0u) level += int(get(index_bits)) << 7;
                if ((flags & 2u) != 0u) {
                    uint r = vlc(2u, run_lut, run_maxb, ok);
                    if (!ok) { p.err.v[0] = 3u; return; }
                    i += int(tb(P_RUN + r));
                }
                if (++i > 63) { p.err.v[0] = 4u; return; }
                int w = int(tb(wbase + uint(i)));
                int scale = qscale * w;
                level *= scale;
                level += scale >> 1;
                if (level_bias < 32 || w != level_bias) level += level_bias;
                level >>= level_shift;
                put(l, tb(P_ZIGZAG + uint(i)), s16((level ^ sign) - sign));
                idx = vlc(1u, ac_lut, ac_maxb, ok);
                if (!ok) { p.err.v[0] = 2u; return; }
            }
            for (uint k = 0u; k < 32u; k++) p.co.v[blk * 32u + k] = blk_sh[l][k];
        }
    }
}
"#;

const IDCT: &str = r#"
layout(local_size_x = 64) in;
layout(push_constant) uniform P { U32 co; U32 dst; U32 prm; } p;
uint prm(uint i) { return p.prm.v[i]; }
int get(uint i) { return int(p.co.v[i >> 1] << (16u - (i & 1u) * 16u)) >> 16; }

void main() {
    uint b = gl_GlobalInvocationID.x;
    uint mbw = prm(0u), mbh = prm(1u), fields = prm(2u), nb = prm(3u);
    if (b >= mbw * mbh * fields * nb) return;
    uint n = b % nb, mb = b / nb;
    uint x = mb % mbw, rowf = mb / mbw;
    uint field = rowf / mbh, row = rowf % mbh;
    bool is444 = prm(10u) != 0u;
    // Plane and position of block n in its macroblock (dnxhd_decode_macroblock).
    uint plane, bx, by;
    if (!is444) {
        const uint pl[8] = uint[8](0u, 0u, 1u, 2u, 0u, 0u, 1u, 2u);
        plane = pl[n];
        by = n >= 4u ? 8u : 0u;
        bx = plane == 0u ? x * 16u + ((n & 1u) != 0u ? 8u : 0u) : x * 8u;
    } else {
        const uint pl[12] = uint[12](0u, 0u, 1u, 1u, 2u, 2u, 0u, 0u, 1u, 1u, 2u, 2u);
        plane = pl[n];
        by = n >= 6u ? 8u : 0u;
        bx = x * 16u + ((n & 1u) != 0u ? 8u : 0u);
    }
    uint step = fields == 2u ? 2u : 1u;
    uint line0 = (row * 16u + by) * step + (fields == 2u ? prm(11u + field) : 0u);
    uint stride = plane == 0u ? prm(19u) : prm(20u);
    uint base = prm(16u + plane);

    int bl[64];
    for (uint i = 0u; i < 64u; i++) bl[i] = get(b * 64u + i);
    // Rows (idctRowCondDC).
    for (uint r = 0u; r < 8u; r++) {
        uint o = r * 8u;
        if (bl[o+1] == 0 && bl[o+2] == 0 && bl[o+3] == 0 && bl[o+4] == 0 && bl[o+5] == 0 && bl[o+6] == 0 && bl[o+7] == 0) {
            int t = DCK >= 0 ? s16(bl[o] << DCK) : s16((bl[o] + (1 << (-DCK - 1))) >> (-DCK));
            for (uint i = 0u; i < 8u; i++) bl[o + i] = t;
            continue;
        }
        int a0 = W4 * bl[o] + (1 << (RSHIFT - 1));
        int a1 = a0, a2 = a0, a3 = a0;
        a0 += W2 * bl[o+2]; a1 += W6 * bl[o+2]; a2 -= W6 * bl[o+2]; a3 -= W2 * bl[o+2];
        int b0 = W1 * bl[o+1] + W3 * bl[o+3];
        int b1 = W3 * bl[o+1] - W7 * bl[o+3];
        int b2 = W5 * bl[o+1] - W1 * bl[o+3];
        int b3 = W7 * bl[o+1] - W5 * bl[o+3];
        a0 += W4 * bl[o+4] + W6 * bl[o+6];
        a1 += -W4 * bl[o+4] - W2 * bl[o+6];
        a2 += -W4 * bl[o+4] + W2 * bl[o+6];
        a3 += W4 * bl[o+4] - W6 * bl[o+6];
        b0 += W5 * bl[o+5] + W7 * bl[o+7];
        b1 += -W1 * bl[o+5] - W5 * bl[o+7];
        b2 += W7 * bl[o+5] + W3 * bl[o+7];
        b3 += W3 * bl[o+5] - W1 * bl[o+7];
        bl[o+0] = s16((a0 + b0) >> RSHIFT); bl[o+7] = s16((a0 - b0) >> RSHIFT);
        bl[o+1] = s16((a1 + b1) >> RSHIFT); bl[o+6] = s16((a1 - b1) >> RSHIFT);
        bl[o+2] = s16((a2 + b2) >> RSHIFT); bl[o+5] = s16((a2 - b2) >> RSHIFT);
        bl[o+3] = s16((a3 + b3) >> RSHIFT); bl[o+4] = s16((a3 - b3) >> RSHIFT);
    }
    // Columns, straight to clipped samples (idctSparseColPut).
    int outv[64];
    for (uint c = 0u; c < 8u; c++) {
        int a0 = W4 * (bl[c] + ((1 << (CSHIFT - 1)) / W4));
        int a1 = a0, a2 = a0, a3 = a0;
        a0 += W2 * bl[c+16]; a1 += W6 * bl[c+16]; a2 += -W6 * bl[c+16]; a3 += -W2 * bl[c+16];
        int b0 = W1 * bl[c+8] + W3 * bl[c+24];
        int b1 = W3 * bl[c+8] - W7 * bl[c+24];
        int b2 = W5 * bl[c+8] - W1 * bl[c+24];
        int b3 = W7 * bl[c+8] - W5 * bl[c+24];
        a0 += W4 * bl[c+32]; a1 += -W4 * bl[c+32]; a2 += -W4 * bl[c+32]; a3 += W4 * bl[c+32];
        b0 += W5 * bl[c+40]; b1 += -W1 * bl[c+40]; b2 += W7 * bl[c+40]; b3 += W3 * bl[c+40];
        a0 += W6 * bl[c+48]; a1 += -W2 * bl[c+48]; a2 += W2 * bl[c+48]; a3 += -W6 * bl[c+48];
        b0 += W7 * bl[c+56]; b1 += -W5 * bl[c+56]; b2 += W3 * bl[c+56]; b3 += -W1 * bl[c+56];
        outv[c+0]  = clamp((a0 + b0) >> CSHIFT, 0, MAXV);
        outv[c+8]  = clamp((a1 + b1) >> CSHIFT, 0, MAXV);
        outv[c+16] = clamp((a2 + b2) >> CSHIFT, 0, MAXV);
        outv[c+24] = clamp((a3 + b3) >> CSHIFT, 0, MAXV);
        outv[c+32] = clamp((a3 - b3) >> CSHIFT, 0, MAXV);
        outv[c+40] = clamp((a2 - b2) >> CSHIFT, 0, MAXV);
        outv[c+48] = clamp((a1 - b1) >> CSHIFT, 0, MAXV);
        outv[c+56] = clamp((a0 - b0) >> CSHIFT, 0, MAXV);
    }
    for (uint r = 0u; r < 8u; r++) {
        uint o = r * 8u;
        uint at = (base + (line0 + r * step) * stride + bx * BPS) / 4u;
        if (BPS == 1u) {
            p.dst.v[at] = uint(outv[o]) | uint(outv[o+1]) << 8 | uint(outv[o+2]) << 16 | uint(outv[o+3]) << 24;
            p.dst.v[at + 1u] = uint(outv[o+4]) | uint(outv[o+5]) << 8 | uint(outv[o+6]) << 16 | uint(outv[o+7]) << 24;
        } else {
            for (uint j = 0u; j < 4u; j++) p.dst.v[at + j] = uint(outv[o + 2u * j]) | (uint(outv[o + 2u * j + 1u]) << 16);
        }
    }
}
"#;

/// IDCT constants per bit depth (simple_idct_template.c, put variant).
fn idct_defs(bits: u32) -> &'static str {
    match bits {
        8 => "const int W1 = 22725, W2 = 21407, W3 = 19266, W4 = 16383, W5 = 12873, W6 = 8867, W7 = 4520;\n#define RSHIFT 11\n#define CSHIFT 20\n#define DCK 3\n#define MAXV 255\n#define BPS 1u\n",
        // Plain 10-bit put: row shift 12, column 19 (ProRes adds its own extra shift).
        10 => "const int W1 = 22725, W2 = 21407, W3 = 19265, W4 = 16384, W5 = 12873, W6 = 8867, W7 = 4520;
#define RSHIFT 12
#define CSHIFT 19
#define DCK 2
#define MAXV 1023
#define BPS 2u
",
        _ => "const int W1 = 45451, W2 = 42813, W3 = 38531, W4 = 32767, W5 = 25746, W6 = 17734, W7 = 9041;\n#define RSHIFT 16\n#define CSHIFT 17\n#define DCK -1\n#define MAXV 4095\n#define BPS 2u\n",
    }
}

const ZIGZAG: [u8; 64] = [
    0, 1, 8, 16, 9, 2, 3, 10, 17, 24, 32, 25, 18, 11, 4, 5, 12, 19, 26, 33, 40, 48, 41, 34, 27, 20, 13, 6, 7, 14, 21, 28, //
    35, 42, 49, 56, 57, 50, 43, 36, 29, 22, 15, 23, 30, 37, 44, 51, 58, 59, 52, 45, 38, 31, 39, 46, 53, 60, 61, 54, 47, 55, 62, 63,
];

/// Bits of the first-level code tables kept in shared memory.
const SHORT: u32 = 10;

/// Lookup table for a prefix code: 2^maxb entries of symbol | length << 16.
fn lut(codes: &[u32], lens: &[u8]) -> Result<(Vec<u32>, u32)> {
    let maxb = *lens.iter().max().unwrap() as u32;
    ensure!(maxb <= 20, "code of {maxb} bits");
    let mut t = vec![0u32; 1 << maxb];
    for (s, (&c, &l)) in codes.iter().zip(lens).enumerate() {
        if l == 0 {
            continue;
        }
        let shift = maxb - l as u32;
        let start = (c << shift) as usize;
        for e in &mut t[start..start + (1 << shift)] {
            *e = s as u32 | (l as u32) << 16;
        }
    }
    Ok((t, maxb))
}

/// One profile's code tables on the GPU (DC, AC, run tables back to back).
struct Luts {
    cid: u32,
    buf: Buffer,
    maxb: [u32; 3],
    ac_at: u32,
    run_at: u32,
    /// Where the three first-level tables start.
    short_at: u32,
}

fn make_luts(compute: &Compute, c: &Cid, bits: u32) -> Result<Luts> {
    // FFmpeg's DC table has bit depth + 4 entries.
    let n_dc = (bits as usize + 4).min(c.dc_codes.len());
    let dc_codes: Vec<u32> = c.dc_codes[..n_dc].iter().map(|&v| v as u32).collect();
    let (dc, dmax) = lut(&dc_codes, &c.dc_bits[..n_dc])?;
    let ac_codes: Vec<u32> = c.ac_codes.iter().map(|&v| v as u32).collect();
    let (ac, amax) = lut(&ac_codes, c.ac_bits)?;
    let run_codes: Vec<u32> = c.run_codes.iter().map(|&v| v as u32).collect();
    let (run, rmax) = lut(&run_codes, c.run_bits)?;
    // First-level tables of SHORT bits (DC, AC, run) for shared memory: the entry, or
    // !0 for longer codes (looked up in the full table).
    let short = |t: &[u32], maxb: u32| -> Vec<u32> {
        (0..1u32 << SHORT)
            .map(|q| {
                let e = if maxb >= SHORT { t[(q << (maxb - SHORT)) as usize] } else { t[(q >> (SHORT - maxb)) as usize] };
                if e >> 16 > SHORT { !0 } else { e }
            })
            .collect()
    };
    let shorts = [short(&dc, dmax), short(&ac, amax), short(&run, rmax)];
    let mut all = dc;
    let ac_at = all.len() as u32;
    all.extend(ac);
    let run_at = all.len() as u32;
    all.extend(run);
    let short_at = all.len() as u32;
    for s in shorts {
        all.extend(s);
    }
    let bytes: Vec<u8> = all.iter().flat_map(|v| v.to_le_bytes()).collect();
    Ok(Luts { cid: c.cid, buf: compute.upload(&bytes, "dnx tables")?, maxb: [dmax, amax, rmax], ac_at, run_at, short_at })
}

pub struct Decoder {
    io: Option<Io>,
    compute: Arc<Compute>,
    budget: usize,
    luts: Option<Luts>,
}

impl Decoder {
    pub fn new(compute: Arc<Compute>, budget: usize) -> Option<Decoder> {
        Some(Decoder { io: None, compute, budget, luts: None })
    }
}

fn be16(b: &[u8], i: usize) -> u32 {
    u16::from_be_bytes([b[i], b[i + 1]]) as u32
}
fn be32(b: &[u8], i: usize) -> u32 {
    u32::from_be_bytes(b[i..i + 4].try_into().unwrap())
}

/// One coding unit's header (a frame, or one field of an interlaced frame).
struct Unit {
    cid: u32,
    bits: u32,
    width: u32,
    height: u32,
    interlaced: bool,
    cur_field: u32,
    mb_height: u32,
    /// Byte offsets (in the packet) where each macroblock row starts.
    rows: Vec<u32>,
    /// The frame allows the adaptive colour transform.
    act: bool,
}

fn header(pkt: &[u8], at: usize, first: bool, prev_field: u32) -> Result<Unit> {
    let b = &pkt[at..];
    ensure!(b.len() >= 0x280, "short DNx coding unit");
    let prefix_ok = (b[0..4] == [0, 0, 2, 0x80] && (1..=3).contains(&b[4])) || b[0..5] == [0, 0, 3, 0x8C, 3];
    ensure!(prefix_ok, "not a DNx header");
    let interlaced = b[5] & 2 != 0;
    let cur_field = if !interlaced { 0 } else if first { (b[5] & 1) as u32 } else { 1 - prev_field };
    ensure!((b[6] >> 5) & 1 == 0, "MBAFF");
    ensure!(b[7] & 1 == 0, "alpha");
    let act = b[0x2C] & 1 != 0;
    let mut height = be16(b, 0x18);
    let width = be16(b, 0x1a);
    let bits = match b[0x21] >> 5 {
        1 => 8,
        2 => 10,
        3 => 12,
        d => bail!("bit depth code {d}"),
    };
    let cid = be32(b, 0x28);
    // The frame may allow the adaptive colour transform (bit 0 of 0x2C); macroblocks
    // that use it are refused by the kernel (FFmpeg then outputs RGB).
    let mb_width = width.div_ceil(16);
    let mb_height = be16(b, 0x16c);
    ensure!(width > 0 && mb_height > 0 && mb_width > 0, "picture size");
    if height.div_ceil(16) == mb_height && interlaced {
        height <<= 1;
    }
    ensure!((mb_height << interlaced as u32) <= height.div_ceil(16), "macroblock rows");
    let hr = b[0..5] == [0, 0, 3, 0x8C, 3];
    let data_offset = if mb_height > 68 {
        ensure!(hr || b[4] == 3, "too many rows for this header");
        0x170 + (mb_height as usize) * 4
    } else {
        0x280
    };
    ensure!(b.len() >= data_offset, "short header");
    let rows = (0..mb_height as usize)
        .map(|i| {
            let o = be32(b, 0x170 + i * 4) as usize;
            ensure!(data_offset + o < b.len(), "row offset");
            Ok((at + data_offset + o) as u32)
        })
        .collect::<Result<Vec<_>>>()?;
    let _ = mb_width;
    Ok(Unit { cid, bits, width, height, interlaced, cur_field, mb_height, rows, act })
}

impl GpuDecoder for Decoder {
    fn name(&self) -> &str {
        "DNxHD (GPU)"
    }

    fn vram_budget(&self, _held: usize) -> usize {
        self.budget
    }

    fn decode(&mut self, data: &[u8]) -> Result<GpuPicture> {
        let u0 = header(data, 0, true, 0)?;
        let c = cid(u0.cid).ok_or_else(|| anyhow::anyhow!("CID {}", u0.cid))?;
        let mut units = vec![u0];
        if units[0].interlaced {
            let at = c.cu_size as usize;
            ensure!(at > 0 && at < data.len(), "second field");
            let u1 = header(data, at, false, units[0].cur_field)?;
            ensure!(u1.cid == units[0].cid && u1.mb_height == units[0].mb_height, "fields differ");
            units.push(u1);
        }
        let u = &units[0];
        let bits = u.bits;
        let is444 = c.is_444;
        let fields = units.len() as u32;
        let mbw = u.width.div_ceil(16);
        let mbh = u.mb_height;
        let nb = if is444 { 12 } else { 8 };
        // (index bits, level bias, level shift, DC shift) of FFmpeg's block decoders. DNxHR
        // HQX dequantises like 4:4:4 (checked bit-exact on both). 12-bit is left to
        // FFmpeg: no 12-bit file to check against.
        let (index_bits, level_bias, level_shift, dc_shift) = match bits {
            8 => (4, 32, 6, 0),
            10 if is444 || c.cid == 1271 => (6, 32, 6, 0),
            10 => (6, 8, 4, 0),
            _ => bail!("{bits}-bit DNx (not checked on the GPU)"),
        };

        if self.luts.as_ref().is_none_or(|l| l.cid != c.cid) {
            self.luts = Some(make_luts(&self.compute, c, bits)?);
        }
        let luts = self.luts.as_ref().unwrap();

        let bps = if bits == 8 { 1u64 } else { 2 };
        let aw = (mbw * 16) as u64;
        let ah = (mbh * 16 * fields) as u64;
        let cw = if is444 { aw } else { aw / 2 };
        let io = match &mut self.io {
            Some(io) => io,
            None => self.io.insert(Io::new(self.compute.clone())?),
        };
        let out = io.planes(&[(aw * bps, ah), (cw * bps, ah), (cw * bps, ah)])?;
        let nblocks = (mbw * mbh * fields * nb) as u64;
        let coef = self.compute.buffer(nblocks * 128, "dnx coefficients")?;

        let mut prm = vec![0u32; P_ROWS + (mbh * fields) as usize];
        prm[0] = mbw;
        prm[1] = mbh;
        prm[2] = fields;
        prm[3] = nb;
        prm[4] = bits;
        prm[5] = index_bits;
        prm[6] = level_bias;
        prm[7] = level_shift;
        prm[8] = dc_shift;
        prm[9] = c.eob_index;
        prm[10] = is444 as u32;
        prm[11] = units[0].cur_field;
        prm[12] = units.get(1).map_or(0, |u| u.cur_field);
        prm[13..16].copy_from_slice(&luts.maxb);
        let offs = &out.offsets;
        prm[16] = offs[0].0 as u32;
        prm[17] = offs[1].0 as u32;
        prm[18] = offs[2].0 as u32;
        prm[19] = offs[0].1 as u32;
        prm[20] = offs[1].1 as u32;
        prm[22] = luts.ac_at;
        prm[23] = luts.run_at;
        prm[24] = units[0].act as u32;
        prm[25] = luts.short_at;
        for i in 0..64 {
            prm[P_WEIGHTS + i] = c.luma[i] as u32;
            prm[P_WEIGHTS + 64 + i] = c.chroma[i] as u32;
            prm[P_ZIGZAG + i] = ZIGZAG[i] as u32;
        }
        for i in 0..514 {
            prm[P_ACINFO + i] = c.ac_info[i] as u32;
        }
        for i in 0..62 {
            prm[P_RUN + i] = c.run[i] as u32;
        }
        let mut k = P_ROWS;
        for u in &units {
            for &r in &u.rows {
                prm[k] = r;
                k += 1;
            }
        }

        let prm_at = (data.len() + 64).next_multiple_of(256);
        let total = prm_at + prm.len() * 4;
        let st = io.staging(total)?;
        st[..data.len()].copy_from_slice(data);
        st[data.len()..prm_at].fill(0);
        for (i, v) in prm.iter().enumerate() {
            st[prm_at + i * 4..prm_at + i * 4 + 4].copy_from_slice(&v.to_le_bytes());
        }

        let pre = crate::compute::PRELUDE;
        let vlc = self.compute.kernel("dnx_vlc", &format!("{pre}{BITS_GLSL}const uint P_WEIGHTS = {P_WEIGHTS}u, P_ZIGZAG = {P_ZIGZAG}u, P_ACINFO = {P_ACINFO}u, P_RUN = {P_RUN}u, P_ROWS = {P_ROWS}u;\n{VLC}"), 0)?;
        let idct = self.compute.kernel(&format!("dnx_idct_{bits}"), &format!("{pre}{BITS_GLSL}{}{IDCT}", idct_defs(bits)), 0)?;
        let mut rec = Recorder::new(&self.compute)?;

        io.upload(&mut rec, total)?;
        let input = io.input();
        rec.label = "dnx vlc".into();
        rec.dispatch(&vlc, [mbh * fields, 1, 1], &Push::new().ptr(input).ptr_at(input, prm_at as u64).ptr(&luts.buf).ptr(&coef).ptr(io.err()))?;
        rec.label = "dnx idct".into();
        rec.dispatch(&idct, Recorder::groups(&idct, [nblocks as u32, 1, 1]), &Push::new().ptr(&coef).ptr(&out.buf).ptr_at(input, prm_at as u64))?;
        rec.submit_wait()?;
        match io.error() {
            0 => {}
            e => bail!("damaged macroblock row (error {e})"),
        }
        let (pix_fmt, cws) = match (bits, is444) {
            (8, false) => (pix::YUV422P, 1),
            (10, false) => (pix::YUV422P10, 1),
            (12, false) => (pix::YUV422P12, 1),
            (10, true) => (pix::YUV444P10, 0),
            _ => (pix::YUV444P12, 0),
        };
        Ok(GpuPicture {
            planes: Arc::new(out),
            layout: FrameLayout::Yuv { bits: bits as u8, chroma_w_shift: cws, chroma_h_shift: 0, alpha: false },
            pix_fmt,
            width: u.width,
            height: u.height,
            interlaced: u.interlaced,
            // FFmpeg: top field first when the first coding unit is field 0.
            tff: u.interlaced && units[0].cur_field == 0,
        })
    }
}
