//! Apple ProRe        let name = if bits == 12 { "prores_12" } else { "prores_10" }; (422 Proxy/LT/Standard/HQ, 4444, 4444 XQ, with alpha, progressive
//! and interlaced) on the GPU.
//!
//! The CPU reads the frame, picture and slice headers (a few bytes per slice) and
//! uploads the packet. On the GPU one workgroup decodes one slice: four threads read
//! the Y, Cb, Cr and alpha bitstreams (adaptive Rice/exp-Golomb codes, as FFmpeg's
//! `proresdec.c`) into shared memory, then all 64 threads run FFmpeg's integer IDCT
//! (`simple_idct`, ProRes variant) block by block and write the planes. Same
//! arithmetic as FFmpeg, so the pictures are identical: 422 as 10-bit, 4444 as
//! 12-bit, as FFmpeg outputs them.

use super::tables::*;
use super::{Io, BITS_GLSL};
use crate::compute::{Compute, Push, Recorder};
use anyhow::{bail, ensure, Result};
use brp_media::gpudec::{pix, GpuDecoder, GpuPicture};
use brp_media::FrameLayout;
use std::sync::Arc;

const KERNEL: &str = r#"
layout(local_size_x = 64) in;
layout(push_constant) uniform P { U32 src; U32 prm; U32 dst; U32 err; } p;

// Coefficients of one slice (at most 8 macroblocks x 12 blocks), int16 pairs.
shared uint co[3072];

void put(uint i, int v) {
    uint w = i >> 1, sh = (i & 1u) * 16u;
    co[w] = (co[w] & ~(0xffffu << sh)) | ((uint(v) & 0xffffu) << sh);
}
int get(uint i) { return int(co[i >> 1] << (16u - (i & 1u) * 16u)) >> 16; }
uint prm(uint i) { return p.prm.v[i]; }
void fail(uint code) { p.err.v[0] = code; }

// FFmpeg DECODE_CODEWORD.
bool cw(inout uint pos, uint cb, out uint val) {
    uint buf = show32(p.src, pos);
    uint sw = cb & 3u, rice = cb >> 5, ex = (cb >> 2) & 7u;
    uint q = buf == 0u ? 31u : 31u - uint(findMSB(buf));
    if (q > sw) {
        uint bits = ex - sw + (q << 1);
        // proresdec uses the long (64-bit) bit reader: codes up to 31 bits.
        if (bits > 31u) { val = 0u; return false; }
        val = showbits(p.src, pos, bits) - (1u << ex) + ((sw + 1u) << rice);
        pos += bits;
    } else if (rice != 0u) {
        pos += q + 1u;
        val = (q << rice) + showbits(p.src, pos, rice);
        pos += rice;
    } else {
        val = q;
        pos += q + 1u;
    }
    return true;
}

// DC and AC coefficients of `nb` blocks starting at block b0 (decode_dc_coeffs,
// decode_ac_coeffs). `pos`: first bit, `size`: bytes.
bool component(uint b0, uint nb, uint pos, uint size) {
    uint end = pos + size * 8u;
    uint code;
    if (!cw(pos, FIRST_DC_CB, code)) return false;
    int prev = s16(int((code >> 1) ^ (0u - (code & 1u))));
    put(b0 * 64u, prev);
    code = 5u;
    int sign = 0;
    for (uint i = 1u; i < nb; i++) {
        if (!cw(pos, DC_CB[min(code, 6u)], code)) return false;
        if (code != 0u) sign ^= -int(code & 1u); else sign = 0;
        prev = s16(prev + ((int((code + 1u) >> 1) ^ sign) - sign));
        put((b0 + i) * 64u, prev);
    }
    uint run = 4u, level = 2u;
    uint l2 = uint(findMSB(nb));
    uint maxc = 64u << l2, mask = nb - 1u;
    uint ps = mask;
    for (;;) {
        int left = int(end) - int(pos);
        if (left <= 0 || (left < 32 && showbits(p.src, pos, uint(left)) == 0u)) break;
        if (!cw(pos, RUN_CB[min(run, 15u)], run)) return false;
        ps += run + 1u;
        if (ps >= maxc) return false;
        if (!cw(pos, LEV_CB[min(level, 9u)], level)) return false;
        level += 1u;
        uint i = ps >> l2;
        int sg = -int(showbits(p.src, pos, 1u));
        pos += 1u;
        put((b0 + (ps & mask)) * 64u + prm(144u + i), s16((int(level) ^ sg) - sg));
    }
    return true;
}

// One 16-bit sample of a plane (read-modify-write of its 32-bit word: each word
// belongs to one thread).
void put16(uint byte_off, uint v) {
    uint w = byte_off >> 2, sh = (byte_off & 2u) * 8u;
    p.dst.v[w] = (p.dst.v[w] & ~(0xffffu << sh)) | ((v & 0xffffu) << sh);
}

uint getb(inout uint pos, uint n) { uint v = showbits(p.src, pos, n); pos += n; return v; }

// unpack_alpha(): run-length coded alpha, straight into the alpha plane.
void alpha(uint pos, uint size, uint mbc, uint x0, uint row0, uint step, uint stride, uint plane) {
    uint end = pos + size * 8u;
    uint nbits = prm(3u) == 2u ? 16u : 8u;
    uint mask = (1u << nbits) - 1u;
    uint num = mbc * 256u, w = 16u * mbc;
    bool twelve = prm(2u) == 12u;
    uint idx = 0u, aval = mask;
    do {
        do {
            uint val;
            if (getb(pos, 1u) != 0u) {
                val = getb(pos, nbits);
            } else {
                val = getb(pos, nbits == 16u ? 7u : 4u);
                uint sgn = val & 1u;
                val = (val + 2u) >> 1;
                if (sgn != 0u) val = 0u - val;
            }
            aval = (aval + val) & mask;
            uint o = nbits == 16u ? (twelve ? aval >> 4 : aval >> 6) : (twelve ? (aval << 4) | (aval >> 4) : (aval << 2) | (aval >> 6));
            put16(plane + (row0 + (idx / w) * step) * stride + (x0 + idx % w) * 2u, o);
            idx++;
            if (idx >= num) break;
        } while (int(end) - int(pos) > 0 && getb(pos, 1u) != 0u);
        uint val = getb(pos, 4u);
        if (val == 0u) val = getb(pos, 11u);
        if (idx + val > num) val = num - idx;
        for (uint i = 0u; i < val; i++) {
            uint o = nbits == 16u ? (twelve ? aval >> 4 : aval >> 6) : (twelve ? (aval << 4) | (aval >> 4) : (aval << 2) | (aval >> 6));
            put16(plane + (row0 + (idx / w) * step) * stride + (x0 + idx % w) * 2u, o);
            idx++;
        }
    } while (idx < num);
}

// simple_idct row pass (idctRowCondDC with the ProRes extra shift).
void idct_row(inout int b[64], uint r) {
    uint o = r * 8u;
    if (b[o+1] == 0 && b[o+2] == 0 && b[o+3] == 0 && b[o+4] == 0 && b[o+5] == 0 && b[o+6] == 0 && b[o+7] == 0) {
        int t = DCK >= 0 ? s16(b[o] << DCK) : s16((b[o] + (1 << (-DCK - 1))) >> (-DCK));
        for (uint i = 0u; i < 8u; i++) b[o + i] = t;
        return;
    }
    int a0 = W4 * b[o] + (1 << (RSHIFT - 1));
    int a1 = a0, a2 = a0, a3 = a0;
    a0 += W2 * b[o+2]; a1 += W6 * b[o+2]; a2 -= W6 * b[o+2]; a3 -= W2 * b[o+2];
    int b0 = W1 * b[o+1] + W3 * b[o+3];
    int b1 = W3 * b[o+1] - W7 * b[o+3];
    int b2 = W5 * b[o+1] - W1 * b[o+3];
    int b3 = W7 * b[o+1] - W5 * b[o+3];
    a0 += W4 * b[o+4] + W6 * b[o+6];
    a1 += -W4 * b[o+4] - W2 * b[o+6];
    a2 += -W4 * b[o+4] + W2 * b[o+6];
    a3 += W4 * b[o+4] - W6 * b[o+6];
    b0 += W5 * b[o+5] + W7 * b[o+7];
    b1 += -W1 * b[o+5] - W5 * b[o+7];
    b2 += W7 * b[o+5] + W3 * b[o+7];
    b3 += W3 * b[o+5] - W1 * b[o+7];
    b[o+0] = s16((a0 + b0) >> RSHIFT);
    b[o+7] = s16((a0 - b0) >> RSHIFT);
    b[o+1] = s16((a1 + b1) >> RSHIFT);
    b[o+6] = s16((a1 - b1) >> RSHIFT);
    b[o+2] = s16((a2 + b2) >> RSHIFT);
    b[o+5] = s16((a2 - b2) >> RSHIFT);
    b[o+3] = s16((a3 + b3) >> RSHIFT);
    b[o+4] = s16((a3 - b3) >> RSHIFT);
}

// simple_idct column pass (idctSparseCol), with the +8192 DC bias added first.
void idct_col(inout int b[64], uint c) {
    b[c] = s16(b[c] + BIAS);
    int a0 = W4 * (b[c] + ((1 << (CSHIFT - 1)) / W4));
    int a1 = a0, a2 = a0, a3 = a0;
    a0 += W2 * b[c+16]; a1 += W6 * b[c+16]; a2 += -W6 * b[c+16]; a3 += -W2 * b[c+16];
    int b0 = W1 * b[c+8] + W3 * b[c+24];
    int b1 = W3 * b[c+8] - W7 * b[c+24];
    int b2 = W5 * b[c+8] - W1 * b[c+24];
    int b3 = W7 * b[c+8] - W5 * b[c+24];
    a0 += W4 * b[c+32]; a1 += -W4 * b[c+32]; a2 += -W4 * b[c+32]; a3 += W4 * b[c+32];
    b0 += W5 * b[c+40]; b1 += -W1 * b[c+40]; b2 += W7 * b[c+40]; b3 += W3 * b[c+40];
    a0 += W6 * b[c+48]; a1 += -W2 * b[c+48]; a2 += W2 * b[c+48]; a3 += -W6 * b[c+48];
    b0 += W7 * b[c+56]; b1 += -W5 * b[c+56]; b2 += W3 * b[c+56]; b3 += -W1 * b[c+56];
    b[c+0]  = s16((a0 + b0) >> CSHIFT);
    b[c+8]  = s16((a1 + b1) >> CSHIFT);
    b[c+16] = s16((a2 + b2) >> CSHIFT);
    b[c+24] = s16((a3 + b3) >> CSHIFT);
    b[c+32] = s16((a3 - b3) >> CSHIFT);
    b[c+40] = s16((a2 - b2) >> CSHIFT);
    b[c+48] = s16((a1 - b1) >> CSHIFT);
    b[c+56] = s16((a0 - b0) >> CSHIFT);
}

void main() {
    uint s = gl_WorkGroupID.x, t = gl_LocalInvocationID.x;
    uint sb = 208u + s * 8u;
    uint off = prm(sb), ys = prm(sb + 1u), us = prm(sb + 2u), vs = prm(sb + 3u), as_ = prm(sb + 4u);
    int qscale = int(prm(sb + 5u));
    uint mb_x = prm(sb + 6u) & 0xffffu, mb_y = prm(sb + 6u) >> 16;
    uint mbc = prm(sb + 7u) & 0xffu, field = (prm(sb + 7u) >> 8) & 1u;
    uint log2c = prm(1u), maxv = prm(2u) == 12u ? 4091u : 1019u;
    bool ilace = prm(4u) != 0u;
    uint ystride = prm(6u), cstride = prm(7u);
    // Field lines: every other line, the first field on top for top-field-first.
    uint step = ilace ? 2u : 1u;
    uint line_off = ilace ? (((field == 0u) ? 1u : 0u) ^ prm(5u)) : 0u;
    uint ny = mbc * 4u, nc = mbc << log2c;
    bool chroma = us + vs > 0u;

    for (uint i = t; i < 3072u; i += 64u) co[i] = 0u;
    barrier();
    if (t == 0u) {
        if (!component(0u, ny, off * 8u, ys)) fail(1u);
    } else if (t == 1u && chroma) {
        if (!component(ny, nc, (off + ys) * 8u, us)) fail(2u);
    } else if (t == 2u && chroma) {
        if (!component(ny + nc, nc, (off + ys + us) * 8u, vs)) fail(3u);
    } else if (t == 3u && prm(3u) != 0u && as_ > 0u) {
        alpha((off + ys + us + vs) * 8u, as_, mbc, mb_x * 16u, mb_y * 16u * step + line_off, step, ystride, prm(10u));
    }
    barrier();

    uint total = ny + 2u * nc;
    for (uint b = t; b < total; b += 64u) {
        uint c, k;
        if (b < ny) { c = 0u; k = b; } else if (b < ny + nc) { c = 1u; k = b - ny; } else { c = 2u; k = b - ny - nc; }
        uint x, y;
        if (c == 0u) {
            uint mb = k >> 2, sub = k & 3u;
            x = (mb_x + mb) * 16u + (sub & 1u) * 8u;
            y = (sub >> 1) * 8u;
        } else if (log2c == 1u) {
            x = (mb_x + (k >> 1)) * 8u;
            y = (k & 1u) * 8u;
        } else {
            uint kk = k & 3u;
            x = (mb_x + (k >> 2)) * 16u + (kk >> 1) * 8u;
            y = (kk & 1u) * 8u;
        }
        uint stride = c == 0u ? ystride : cstride;
        uint plane = c == 0u ? 0u : prm(c == 1u ? 8u : 9u);
        uint row0 = (mb_y * 16u + y) * step + line_off;
        if (c != 0u && !chroma) {
            uint v = prm(2u) == 12u ? 2044u : 511u;
            for (uint r = 0u; r < 8u; r++)
                for (uint j = 0u; j < 4u; j++)
                    p.dst.v[(plane + (row0 + r * step) * stride + x * 2u) / 4u + j] = v | (v << 16);
            continue;
        }
        uint qm = c == 0u ? 16u : 80u;
        int bl[64];
        for (uint i = 0u; i < 64u; i++)
            bl[i] = s16(get(b * 64u + i) * s16(int(prm(qm + i)) * qscale));
        for (uint r = 0u; r < 8u; r++) idct_row(bl, r);
        for (uint cc = 0u; cc < 8u; cc++) idct_col(bl, cc);
        for (uint r = 0u; r < 8u; r++) {
            uint base = (plane + (row0 + r * step) * stride + x * 2u) / 4u;
            for (uint j = 0u; j < 4u; j++) {
                uint lo = uint(clamp(bl[r * 8u + 2u * j], 4, int(maxv)));
                uint hi = uint(clamp(bl[r * 8u + 2u * j + 1u], 4, int(maxv)));
                p.dst.v[base + j] = lo | (hi << 16);
            }
        }
    }
}
"#;

/// IDCT constants (simple_idct_template.c). 10-bit: row shift 13 + the ProRes extra
/// shift of 2, column shift 18 (note W3 = 19265, W4 = 16384: not the 8-bit values).
/// 12-bit: row shift 16, column shift 17. DCK: the shift of a DC-only row.
const IDCT10: &str = "const int W1 = 22725, W2 = 21407, W3 = 19265, W4 = 16384, W5 = 12873, W6 = 8867, W7 = 4520;
#define RSHIFT 15
#define CSHIFT 18
#define BIAS 8192
#define DCK -1
";
const IDCT12: &str = "const int W1 = 45451, W2 = 42813, W3 = 38531, W4 = 32767, W5 = 25746, W6 = 17734, W7 = 9041;\n#define RSHIFT 16\n#define CSHIFT 17\n#define BIAS 8192\n#define DCK -1\n";

pub struct Decoder {
    io: Option<Io>,
    compute: Arc<Compute>,
    budget: usize,
}

impl Decoder {
    pub fn new(compute: Arc<Compute>, budget: usize) -> Decoder {
        Decoder { io: None, compute, budget }
    }
}

fn be16(b: &[u8], i: usize) -> usize {
    u16::from_be_bytes([b[i], b[i + 1]]) as usize
}

struct Slice {
    off: usize,
    sizes: [usize; 4],
    qscale: u32,
    mb_x: u32,
    mb_y: u32,
    mb_count: u32,
    field: u32,
}

/// What the CPU reads from a frame: geometry, quantisers and the slice table.
struct Frame {
    width: u32,
    height: u32,
    is444: bool,
    alpha: u32,
    frame_type: u32,
    qmat: [[u8; 64]; 2],
    mb_width: u32,
    mb_height: u32,
    slices: Vec<Slice>,
}

fn parse(data: &[u8]) -> Result<Frame> {
    ensure!(data.len() >= 28 && &data[4..8] == b"icpf", "not a ProRes frame");
    let buf = &data[8..];
    let hdr_size = be16(buf, 0);
    ensure!(hdr_size >= 20 && hdr_size <= buf.len(), "frame header size");
    ensure!(be16(buf, 2) <= 1, "ProRes version {}", be16(buf, 2));
    let width = be16(buf, 8) as u32;
    let height = be16(buf, 10) as u32;
    ensure!(width > 0 && height > 0, "picture size");
    let frame_type = ((buf[12] >> 2) & 3) as u32;
    let alpha = (buf[17] & 0xf) as u32;
    ensure!(alpha <= 2, "alpha info {alpha}");
    let is444 = buf[12] & 0xC0 == 0xC0;
    // 422 with alpha (no encoder we know writes it) stays with FFmpeg.
    ensure!(is444 || alpha == 0, "422 with alpha");
    let flags = buf[19];
    let mut qmat = [[4u8; 64]; 2];
    let mut at = 20;
    if flags & 2 != 0 {
        ensure!(buf.len() >= at + 64, "luma matrix");
        qmat[0].copy_from_slice(&buf[at..at + 64]);
        at += 64;
    }
    if flags & 1 != 0 {
        ensure!(buf.len() >= at + 64, "chroma matrix");
        qmat[1].copy_from_slice(&buf[at..at + 64]);
    } else {
        qmat[1] = qmat[0];
    }

    let mb_width = width.div_ceil(16);
    let mb_height = if frame_type != 0 { (height + 31) >> 5 } else { height.div_ceil(16) };
    let mut slices = Vec::new();
    let mut pic = 8 + hdr_size; // offset in `data`
    let pictures = if frame_type != 0 { 2 } else { 1 };
    for field in 0..pictures {
        ensure!(pic < data.len(), "picture missing");
        let b = &data[pic..];
        let ph = (b[0] >> 3) as usize;
        ensure!(ph >= 8 && ph <= b.len(), "picture header size");
        let pic_size = u32::from_be_bytes(b[1..5].try_into().unwrap()) as usize;
        ensure!(pic_size <= b.len(), "picture size");
        let log2w = (b[7] >> 4) as u32;
        ensure!(log2w <= 3 && b[7] & 0xF == 0, "slice size");
        let count = (mb_height * ((mb_width >> log2w) + (mb_width & ((1 << log2w) - 1)).count_ones())) as usize;
        ensure!(ph + count * 2 <= b.len(), "slice table");
        let index = &b[ph..];
        let mut data_at = ph + count * 2;
        let mut smc = 1u32 << log2w;
        let (mut mb_x, mut mb_y) = (0u32, 0u32);
        for i in 0..count {
            let start = data_at;
            data_at += be16(index, i * 2);
            while mb_width - mb_x < smc {
                smc >>= 1;
            }
            let size = data_at - start;
            ensure!(size >= 6 && data_at <= b.len(), "slice data");
            // Slice header.
            let s = &b[start..];
            let sh = (s[0] >> 3) as usize;
            let q = (s[1] as u32).clamp(1, 224);
            let qscale = if q > 128 { (q - 96) << 2 } else { q };
            let y = be16(s, 2);
            let u = be16(s, 4);
            let v = if sh > 7 { be16(s, 6) as isize } else { size as isize - y as isize - u as isize - sh as isize };
            ensure!(v >= 0 && sh + y + u + v as usize <= size, "invalid plane data size");
            let a = size - y - u - v as usize - sh;
            slices.push(Slice { off: pic + start + sh, sizes: [y, u, v as usize, a], qscale, mb_x, mb_y, mb_count: smc, field });
            mb_x += smc;
            if mb_x == mb_width {
                smc = 1 << log2w;
                mb_x = 0;
                mb_y += 1;
            }
        }
        ensure!(mb_x == 0 && mb_y == mb_height, "slice table does not cover the picture");
        pic += pic_size;
    }
    Ok(Frame { width, height, is444, alpha, frame_type, qmat, mb_width, mb_height, slices })
}

impl GpuDecoder for Decoder {
    fn name(&self) -> &str {
        "ProRes (GPU)"
    }

    fn vram_budget(&self, _held: usize) -> usize {
        self.budget
    }

    fn decode(&mut self, data: &[u8]) -> Result<GpuPicture> {
        let f = parse(data)?;
        let bits = if f.is444 { 12 } else { 10 };
        let log2c = if f.is444 { 2 } else { 1 };
        let ilace = f.frame_type != 0;
        // Planes: whole macroblocks (FFmpeg's padded frame), cropped by the size.
        let aw = f.mb_width * 16;
        let ah = if ilace { f.mb_height * 32 } else { f.mb_height * 16 };
        let cw = if f.is444 { aw } else { aw / 2 };
        let mut planes = vec![(aw as u64 * 2, ah as u64), (cw as u64 * 2, ah as u64), (cw as u64 * 2, ah as u64)];
        if f.alpha != 0 {
            planes.push((aw as u64 * 2, ah as u64));
        }

        let io = match &mut self.io {
            Some(io) => io,
            None => self.io.insert(Io::new(self.compute.clone())?),
        };
        let out = io.planes(&planes)?;
        // Staging: the packet, then the parameters.
        let prm_at = (data.len() + 64).next_multiple_of(256);
        let nwords = 208 + f.slices.len() * 8;
        let total = prm_at + nwords * 4;
        let st = io.staging(total)?;
        st[..data.len()].copy_from_slice(data);
        st[data.len()..prm_at].fill(0);
        let mut prm = vec![0u32; nwords];
        let offs = out.offsets.clone();
        prm[1] = log2c;
        prm[2] = bits;
        prm[3] = f.alpha;
        prm[4] = ilace as u32;
        prm[5] = (f.frame_type == 1) as u32;
        prm[6] = offs[0].1 as u32;
        prm[7] = offs[1].1 as u32;
        prm[8] = offs[1].0 as u32;
        prm[9] = offs[2].0 as u32;
        prm[10] = offs.get(3).map_or(0, |o| o.0 as u32);
        for i in 0..64 {
            prm[16 + i] = f.qmat[0][i] as u32;
            prm[80 + i] = f.qmat[1][i] as u32;
            prm[144 + i] = if ilace { PRORES_INTERLACED_SCAN[i] } else { PRORES_PROGRESSIVE_SCAN[i] } as u32;
        }
        for (i, s) in f.slices.iter().enumerate() {
            let w = &mut prm[208 + i * 8..208 + i * 8 + 8];
            w[0] = s.off as u32;
            w[1] = s.sizes[0] as u32;
            w[2] = s.sizes[1] as u32;
            w[3] = s.sizes[2] as u32;
            w[4] = s.sizes[3] as u32;
            w[5] = s.qscale;
            w[6] = s.mb_x | (s.mb_y << 16);
            w[7] = s.mb_count | (s.field << 8);
        }
        for (i, v) in prm.iter().enumerate() {
            st[prm_at + i * 4..prm_at + i * 4 + 4].copy_from_slice(&v.to_le_bytes());
        }

        let name = if bits == 12 { "prores_12" } else { "prores_10" };
        let src = format!(
            "{}{BITS_GLSL}{}{}{}{}{}{KERNEL}",
            crate::compute::PRELUDE,
            if bits == 12 { IDCT12 } else { IDCT10 },
            format_args!("const uint FIRST_DC_CB = {}u;\n", PRORES_FIRST_DC_CB),
            glsl_array("DC_CB", &PRORES_DC_CB),
            glsl_array("RUN_CB", &PRORES_RUN_CB),
            glsl_array("LEV_CB", &PRORES_LEV_CB),
        );
        let kernel = self.compute.kernel(name, &src, 0)?;
        let mut rec = Recorder::new(&self.compute)?;
        rec.label = "prores".into();
        io.upload(&mut rec, total)?;
        let push = Push::new().ptr(io.input()).ptr_at(io.input(), prm_at as u64).ptr(&out.buf).ptr(io.err());
        rec.dispatch(&kernel, [f.slices.len() as u32, 1, 1], &push)?;
        rec.submit_wait()?;
        match io.error() {
            0 => {}
            e => bail!("damaged slice data (component {e})"),
        }
        let (pix_fmt, layout) = match (f.is444, f.alpha != 0) {
            (true, true) => (pix::YUVA444P12, FrameLayout::Yuv { bits: 12, chroma_w_shift: 0, chroma_h_shift: 0, alpha: true }),
            (true, false) => (pix::YUV444P12, FrameLayout::Yuv { bits: 12, chroma_w_shift: 0, chroma_h_shift: 0, alpha: false }),
            _ => (pix::YUV422P10, FrameLayout::Yuv { bits: 10, chroma_w_shift: 1, chroma_h_shift: 0, alpha: false }),
        };
        Ok(GpuPicture {
            planes: Arc::new(out),
            layout,
            pix_fmt,
            width: f.width,
            height: f.height,
            interlaced: ilace,
            tff: f.frame_type == 1,
        })
    }
}
