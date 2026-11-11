//! Resolume DXV3 on the GPU: DXT1 (opaque) and DXT5 with premultiplied alpha
//! (as CVExplorer writes it; FFmpeg's decoder reads it as DXT4).
//!
//! 1. `tex`: a thread per 4x4 block compresses it to BC1 (BC3 for alpha) with the
//!    stb_dxt method FFmpeg uses (principal axis, one least-squares refinement,
//!    optimal single-colour tables; `bc_tables.rs`).
//! 2. `match`: the LZ stage, in parallel: each block looks for the same block (or
//!    the same half, or word) among nearby blocks (to its left, the row above) and
//!    chooses its ops (copy from a distance, or literal).
//! 3. Prefix sums give every op its index (ops are packed 16 to a 32-bit word that
//!    sits in the stream where its first op is) and every block its byte offset.
//! 4. `opwords` places the op words, `write` each block's ops and data. When the LZ
//!    stream would not be smaller, the raw texture is stored (DXV's raw flag).
//!
//! FFmpeg's DXT1 encoder searches the whole frame with a hash table; the
//! neighbourhood search here finds the repeats that matter in video (flat areas)
//! in parallel.

use super::{Codec, Encoder, Output, RadixSort, Scan, COMMON, HEAD};
use anyhow::{bail, ensure, Result};
use brp_media::encode::PackLayout;
use brp_render::compute::{Buffer, Compute, Kernel, Push, Recorder, PRELUDE};
use std::sync::Arc;

#[derive(Clone, Debug, PartialEq)]
pub struct Settings {
    /// DXT5 with premultiplied alpha (else DXT1, opaque).
    pub alpha: bool,
}

const KERNELS: &str = r#"
layout(push_constant) uniform P { U32 src; U32 work; U32 o; uint a; uint b; } p;
uint wtex(uint b, uint k) { return p.work.v[TEX_OFF + b * WPB + k]; }

// ---- BC1 / BC3 (stb_dxt, as FFmpeg's texturedspenc) ----
uint mul8(uint a, uint b) { uint t = a * b + 128u; return (t + (t >> 8)) >> 8; }
uint rgb565(uvec3 c) { return (mul8(c.r, 31u) << 11) | (mul8(c.g, 63u) << 5) | mul8(c.b, 31u); }
ivec3 unpack565(uint v) { return ivec3(int(EXPAND5[(v >> 11) & 31u]), int(EXPAND6[(v >> 5) & 63u]), int(EXPAND5[v & 31u])); }

uint match_colors(uvec3 px[16], uint c0, uint c1) {
    ivec3 col[4];
    col[0] = unpack565(c0);
    col[1] = unpack565(c1);
    col[2] = (2 * col[0] + col[1]) / 3;
    col[3] = (2 * col[1] + col[0]) / 3;
    ivec3 dir = col[0] - col[1];
    int stops[4];
    for (int i = 0; i < 4; i++) stops[i] = col[i].r * dir.r + col[i].g * dir.g + col[i].b * dir.b;
    int c0p = (stops[1] + stops[3]) >> 1, half_p = (stops[3] + stops[2]) >> 1, c3p = (stops[2] + stops[0]) >> 1;
    const uint idx_map[8] = uint[](0u, 2u, 0u, 2u, 3u, 3u, 1u, 1u);
    uint mask = 0u;
    for (int i = 0; i < 16; i++) {
        int dot = int(px[i].r) * dir.r + int(px[i].g) * dir.g + int(px[i].b) * dir.b;
        uint bits = (dot < half_p ? 4u : 0u) | (dot < c0p ? 2u : 0u) | (dot < c3p ? 1u : 0u);
        mask = (mask >> 2) | (idx_map[bits] << 30);
    }
    return mask;
}

void optimize_colors(uvec3 px[16], out uint max16, out uint min16) {
    ivec3 mu = ivec3(px[0]), mn = ivec3(px[0]), mx = ivec3(px[0]);
    for (int i = 0; i < 16; i++) {
        ivec3 v = ivec3(px[i]);
        mu += v;
        mn = min(mn, v);
        mx = max(mx, v);
    }
    mu = (mu + 8) >> 4;
    int cov[6] = int[](0, 0, 0, 0, 0, 0);
    for (int i = 0; i < 16; i++) {
        ivec3 d = ivec3(px[i]) - mu;
        cov[0] += d.r * d.r; cov[1] += d.r * d.g; cov[2] += d.r * d.b;
        cov[3] += d.g * d.g; cov[4] += d.g * d.b; cov[5] += d.b * d.b;
    }
    float cf[6];
    for (int i = 0; i < 6; i++) cf[i] = float(cov[i]) / 255.0;
    vec3 v = vec3(mx - mn);
    for (int it = 0; it < 4; it++) {
        v = vec3(v.r * cf[0] + v.g * cf[1] + v.b * cf[2], v.r * cf[1] + v.g * cf[3] + v.b * cf[4], v.r * cf[2] + v.g * cf[4] + v.b * cf[5]);
    }
    float magn = max(max(abs(v.r), abs(v.g)), abs(v.b));
    ivec3 vi;
    if (magn < 4.0) vi = ivec3(299, 587, 114);
    else vi = ivec3(v * (512.0 / magn));
    int mind = int(px[0].r) * vi.r + int(px[0].g) * vi.g + int(px[0].b) * vi.b, maxd = mind;
    int mini = 0, maxi = 0;
    for (int i = 0; i < 16; i++) {
        int dot = int(px[i].r) * vi.r + int(px[i].g) * vi.g + int(px[i].b) * vi.b;
        if (dot < mind) { mind = dot; mini = i; }
        else if (dot > maxd) { maxd = dot; maxi = i; }
    }
    max16 = rgb565(px[maxi]);
    min16 = rgb565(px[mini]);
}

uint single565(uint r, uint g, uint b, uint which) {
    return (MATCH5[r * 2u + which] << 11) | (MATCH6[g * 2u + which] << 5) | MATCH5[b * 2u + which];
}

bool refine_colors(uvec3 px[16], inout uint max16, inout uint min16, uint mask) {
    uint old_min = min16, old_max = max16;
    if ((mask ^ (mask << 2)) < 4u) {
        uvec3 s = uvec3(8u);
        for (int i = 0; i < 16; i++) s += px[i];
        s >>= 4;
        max16 = single565(s.r, s.g, s.b, 0u);
        min16 = single565(s.r, s.g, s.b, 1u);
    } else {
        const int w1tab[4] = int[](3, 0, 2, 1);
        const int prods[4] = int[](0x090000, 0x000900, 0x040102, 0x010402);
        ivec3 at1 = ivec3(0), at2 = ivec3(0);
        int akku = 0;
        uint cm = mask;
        for (int i = 0; i < 16; i++) {
            int st = int(cm & 3u);
            int w1 = w1tab[st];
            ivec3 c = ivec3(px[i]);
            akku += prods[st];
            at1 += w1 * c;
            at2 += c;
            cm >>= 2;
        }
        at2 = 3 * at2 - at1;
        int xx = akku >> 16, yy = (akku >> 8) & 255, xy = akku & 255;
        float fr = 3.0 * 31.0 / 255.0 / float(xx * yy - xy * xy);
        float fg = fr * 63.0 / 31.0, fb = fr;
        uint mxr = uint(clamp(int(float(at1.r * yy - at2.r * xy) * fr + 0.5), 0, 31));
        uint mxg = uint(clamp(int(float(at1.g * yy - at2.g * xy) * fg + 0.5), 0, 63));
        uint mxb = uint(clamp(int(float(at1.b * yy - at2.b * xy) * fb + 0.5), 0, 31));
        uint mnr = uint(clamp(int(float(at2.r * xx - at1.r * xy) * fr + 0.5), 0, 31));
        uint mng = uint(clamp(int(float(at2.g * xx - at1.g * xy) * fg + 0.5), 0, 63));
        uint mnb = uint(clamp(int(float(at2.b * xx - at1.b * xy) * fb + 0.5), 0, 31));
        max16 = (mxr << 11) | (mxg << 5) | mxb;
        min16 = (mnr << 11) | (mng << 5) | mnb;
    }
    return old_min != min16 || old_max != max16;
}

// The BC1 colour block: (max16 | min16 << 16, indices).
uvec2 compress_color(uvec3 px[16]) {
    bool constant = true;
    for (int i = 1; i < 16; i++) constant = constant && all(equal(px[i], px[0]));
    uint mask, max16, min16;
    if (constant) {
        mask = 0xAAAAAAAAu;
        max16 = single565(px[0].r, px[0].g, px[0].b, 0u);
        min16 = single565(px[0].r, px[0].g, px[0].b, 1u);
    } else {
        optimize_colors(px, max16, min16);
        mask = max16 != min16 ? match_colors(px, max16, min16) : 0u;
        if (refine_colors(px, max16, min16, mask)) mask = max16 != min16 ? match_colors(px, max16, min16) : 0u;
    }
    if (max16 < min16) {
        uint t = max16; max16 = min16; min16 = t;
        mask ^= 0x55555555u;
    }
    return uvec2(max16 | (min16 << 16), mask);
}

// The BC3 alpha block as two words.
uvec2 compress_alpha(uint a[16]) {
    uint mn = a[0], mx = a[0];
    for (int i = 1; i < 16; i++) { mn = min(mn, a[i]); mx = max(mx, a[i]); }
    uint w0 = mx | (mn << 8);
    if (mn == mx) return uvec2(w0, 0u);
    int dist = int(mx - mn), dist4 = dist * 4, dist2 = dist * 2;
    int bias = dist < 8 ? dist - 1 - int(mn) * 7 : dist / 2 + 2 - int(mn) * 7;
    uint64_t bits = 0ul;
    for (int i = 0; i < 16; i++) {
        int alp = int(a[i]) * 7 + bias;
        int ind = 0;
        if (alp >= dist4) { ind = 4; alp -= dist4; }
        if (alp >= dist2) { ind += 2; alp -= dist2; }
        ind += alp >= dist ? 1 : 0;
        ind = -ind & 7;
        ind ^= (2 > ind) ? 1 : 0;
        bits |= uint64_t(ind) << (3 * i);
    }
    return uvec2(w0 | (uint(bits & 0xfffful) << 16), uint(bits >> 16));
}
"#;

const K_TEX: &str = r#"
layout(local_size_x = 64, local_size_y = 1, local_size_z = 1) in;
void main() {
    uint b = gl_GlobalInvocationID.x;
    if (b >= NB) return;
    uint bx = b % BW, by = b / BW;
    uvec3 px[16];
    uint al[16];
    for (uint i = 0u; i < 16u; i++) {
        uint x = bx * 4u + (i & 3u), y = by * 4u + (i >> 2);
        uint v = p.src.v[(PL_OFF + y * PL_STRIDE + x * 4u) >> 2];
        uvec3 c = uvec3(v & 255u, (v >> 8) & 255u, (v >> 16) & 255u);
        uint a = v >> 24;
        if (ALPHA != 0) c = (c * a + 127u) / 255u;
        px[i] = c;
        al[i] = a;
    }
    uvec2 col = compress_color(px);
    if (ALPHA != 0) {
        uvec2 aw = compress_alpha(al);
        p.work.v[TEX_OFF + b * 4u] = aw.x;
        p.work.v[TEX_OFF + b * 4u + 1u] = aw.y;
        p.work.v[TEX_OFF + b * 4u + 2u] = col.x;
        p.work.v[TEX_OFF + b * 4u + 3u] = col.y;
    } else {
        p.work.v[TEX_OFF + b * 2u] = col.x;
        p.work.v[TEX_OFF + b * 2u + 1u] = col.y;
    }
}
"#;

// Ops of a block: up to 4, each opcode | extra bytes << 2 | value << 8 (value: the
// byte or 16-bit number; for literals the first word slot).
const K_MATCH: &str = r#"
// Sorted slots: (first word, words): DXT1 block, colour word, index word; DXT5
// alpha half, colour half, colour word, index word.
int slot_of(uint k0, uint n) {
    if (WPB == 2u) return n == 2u ? 0 : int(k0) + 1;
    if (n == 2u) return k0 == 0u ? 0 : 1;
    return n == 1u && k0 >= 2u ? int(k0) : -1;
}
layout(local_size_x = 64, local_size_y = 1, local_size_z = 1) in;
const uint NONE = 0xffffffffu;
// Distances searched: 64 blocks to the left, the row above (+-8), two rows up (+-2).
#define NCAND 86u
uint cand(uint i) {
    if (i < 64u) return i + 1u;
    if (i < 81u) return BW + i - 72u;
    return 2u * BW + i - 83u;
}
// Cost of a block distance in an x-scaled CHECKPOINT (op1: 1; op2: byte; op3: le16).
uint dist_cost(uint d) { return d == 1u ? 0u : (d <= 257u ? 1u : (d <= 0x102u + 0xffffu ? 2u : NONE)); }
uint dist_op(uint d) {
    if (d == 1u) return 1u;
    if (d <= 257u) return 2u | (1u << 2) | ((d - 2u) << 8);
    return 3u | (2u << 2) | ((d - 0x102u) << 8);
}
// The cheapest distance at which words [k0, k0 + n) of block b repeat: the most
// recent earlier block with the same words (from the sort, when this run of words
// is one of the sorted slots), else the nearest among the neighbours.
uint best(uint b, uint k0, uint n) {
    uint bd = NONE, bc = NONE;
    int slot = slot_of(k0, n);
    if (slot >= 0) {
        uint r = p.work.v[REC_OFF + uint(slot) * NB + b];
        if (r != NONE) {
            uint d = b - r;
            bool same = true;
            for (uint k = 0u; k < n; k++) same = same && wtex(b, k0 + k) == wtex(r, k0 + k);
            if (same && dist_cost(d) != NONE) { bd = d; bc = dist_cost(d); }
        }
    }
    for (uint i = 0u; i < NCAND; i++) {
        uint d = cand(i);
        if (d == 0u || d > b) continue;
        bool same = true;
        for (uint k = 0u; k < n; k++) same = same && wtex(b, k0 + k) == wtex(b - d, k0 + k);
        uint c = dist_cost(d);
        if (same && c < bc) { bc = c; bd = d; }
    }
    return bd;
}
void main() {
    uint b = gl_GlobalInvocationID.x;
    if (b >= NB) return;
    uint ops[4] = uint[](0u, 0u, 0u, 0u);
    uint n = 0u;
    if (b > 0u) {
        if (WPB == 2u) {
            uint d = best(b, 0u, 2u);
            if (d != NONE) { ops[n++] = dist_op(d); }
            else {
                ops[n++] = 0u;
                for (uint k = 0u; k < 2u; k++) {
                    uint dk = best(b, k, 1u);
                    ops[n++] = dk != NONE ? dist_op(dk) : (0u | (4u << 2) | (k << 8));
                }
            }
        } else {
            bool whole = best(b, 0u, 4u) == 1u;
            if (whole) {
                ops[n++] = 0u | (1u << 2);          // long copy of 1 block (byte 0)
            } else {
                // Alpha half: run 0 (previous block), a 16-bit distance, or literal.
                uint da = best(b, 0u, 2u);
                if (da == 1u) ops[n++] = 1u | (1u << 2);
                else if (da != NONE && da - 2u <= 0xffffu) ops[n++] = 2u | (2u << 2) | ((da - 2u) << 8);
                else ops[n++] = 3u | (8u << 2);
                uint d = best(b, 2u, 2u);
                if (d != NONE) { ops[n++] = dist_op(d); }
                else {
                    ops[n++] = 0u;
                    for (uint k = 2u; k < 4u; k++) {
                        uint dk = best(b, k, 1u);
                        ops[n++] = dk != NONE ? dist_op(dk) : (0u | (4u << 2) | (k << 8));
                    }
                }
            }
        }
    }
    uint bytes = 0u;
    for (uint k = 0u; k < 4u; k++) {
        p.work.v[OPS_OFF + b * 4u + k] = ops[k];
        if (k < n) bytes += (ops[k] >> 2) & 15u;
    }
    p.work.v[NOPS_OFF + b] = n;
    p.work.v[OPB_OFF + b] = n;
    p.work.v[XB_OFF + b] = bytes;
}
"#;

// Bytes of each block with the op words it starts (after the op index scan).
const K_BYTES: &str = r#"
layout(local_size_x = 64, local_size_y = 1, local_size_z = 1) in;
void main() {
    uint b = gl_GlobalInvocationID.x;
    if (b >= NB) return;
    uint n = p.work.v[NOPS_OFF + b], g = p.work.v[OPB_OFF + b], words = 0u;
    for (uint k = 0u; k < n; k++) if (((g + k) & 15u) == 0u) words++;
    p.work.v[BOF_OFF + b] = p.work.v[XB_OFF + b] + 4u * words;
}
"#;

const K_FINISH: &str = r#"
layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;
void main() {
    uint lz = FIRST + p.work.v[BSUM_OFF + BTOTAL];
    bool raw = lz >= TEX_BYTES;
    uint size = 12u + (raw ? TEX_BYTES : lz);
    put_le32(p.o, HEAD, TAG);
    put_byte(p.o, HEAD + 4u, 4u);
    put_byte(p.o, HEAD + 6u, raw ? 1u : 0u);
    put_le32(p.o, HEAD + 8u, size - 12u);
    if (size > CAP) p.o.v[1] = 1u;
    p.o.v[0] = size;
    p.o.v[3] = raw ? 1u : 0u;
}
"#;

// Where each op word sits: the byte of the op that starts it.
const K_OPWORDS: &str = r#"
layout(local_size_x = 64, local_size_y = 1, local_size_z = 1) in;
void main() {
    uint b = gl_GlobalInvocationID.x;
    if (b >= NB || p.o.v[3] != 0u || p.o.v[1] != 0u) return;
    uint n = p.work.v[NOPS_OFF + b], g = p.work.v[OPB_OFF + b];
    uint pos = 12u + FIRST + p.work.v[BOF_OFF + b];
    for (uint k = 0u; k < n; k++) {
        uint op = p.work.v[OPS_OFF + b * 4u + k];
        if (((g + k) & 15u) == 0u) { p.work.v[OPW_OFF + ((g + k) >> 4)] = pos; pos += 4u; }
        pos += (op >> 2) & 15u;
    }
}
"#;

const K_WRITE: &str = r#"
layout(local_size_x = 64, local_size_y = 1, local_size_z = 1) in;
void main() {
    uint b = gl_GlobalInvocationID.x;
    if (b >= NB || p.o.v[1] != 0u) return;
    uint base = (HEAD + 12u) >> 2;
    if (p.o.v[3] != 0u || b == 0u) {
        // Raw texture (or the first block, always raw): word-aligned stores.
        for (uint k = 0u; k < WPB; k++) p.o.v[base + b * WPB + k] = wtex(b, k);
        if (p.o.v[3] != 0u || b == 0u) return;
    }
    uint n = p.work.v[NOPS_OFF + b], g = p.work.v[OPB_OFF + b];
    uint pos = 12u + FIRST + p.work.v[BOF_OFF + b];
    for (uint k = 0u; k < n; k++) {
        uint op = p.work.v[OPS_OFF + b * 4u + k];
        uint gi = g + k;
        if ((gi & 15u) == 0u) pos += 4u;
        uint ow = p.work.v[OPW_OFF + (gi >> 4)];
        uint sl = gi & 15u;
        put_byte(p.o, HEAD + ow + (sl >> 2), (op & 3u) << (2u * (sl & 3u)));
        uint xb = (op >> 2) & 15u, val = op >> 8;
        if (xb == 1u) put_byte(p.o, HEAD + pos, val);
        else if (xb == 2u) put_le16(p.o, HEAD + pos, val);
        else if (xb == 4u) put_le32(p.o, HEAD + pos, wtex(b, val));
        else if (xb == 8u) { put_le32(p.o, HEAD + pos, wtex(b, 0u)); put_le32(p.o, HEAD + pos + 4u, wtex(b, 1u)); }
        pos += xb;
    }
}
"#;

pub struct Dxv {
    codec: Codec,
    nb: u32,
    kernels: Vec<(&'static str, Arc<Kernel>)>,
    scan: Scan,
    sort: RadixSort,
    work: Buffer,
    out: Output,
    layout: PackLayout,
    planes_addr: u64,
    offs: [u32; 4],
    sorting: [u32; 5],
    nslots: u32,
}

impl Dxv {
    pub fn new(compute: &Arc<Compute>, s: Settings, layout: &PackLayout) -> Result<Dxv> {
        ensure!(layout.name == "rgba", "GPU DXV takes rgba, not {}", layout.name);
        let (w, h) = (layout.w, layout.h);
        ensure!(w % 4 == 0 && h % 4 == 0, "DXV size {w}x{h} is not a multiple of 4");
        let pl = &layout.planes[0];
        let (bw, bh) = (w / 4, h / 4);
        let nb = bw * bh;
        let wpb = if s.alpha { 4 } else { 2 };
        let tex_bytes = nb * wpb * 4;
        let off = std::cell::Cell::new(0u32);
        let region = |n: u32| {
            let o = off.get();
            off.set(o + n);
            o
        };
        let tex = region(nb * wpb);
        let ops = region(nb * 4);
        let nops = region(nb);
        let opb = region(nb);
        let xb = region(nb);
        let bof = region(nb);
        let osum = region(nb / 1024 + 2);
        let bsum = region(nb / 1024 + 2);
        let opw = region(nb / 4 + 2);
        let nslots = if s.alpha { 4 } else { 3 };
        let keys = region(nb);
        let vals = region(nb);
        let tkeys = region(nb);
        let tvals = region(nb);
        let scratch = region(RadixSort::scratch_words(nb));
        let rec_off = region(nslots * nb);
        let words = off.get();
        let mut d = String::new();
        let mut def = |k: &str, v: u32| d.push_str(&format!("#define {k} {v}u\n"));
        def("BW", bw);
        def("NB", nb);
        def("WPB", wpb);
        def("PL_OFF", pl.offset as u32);
        def("PL_STRIDE", pl.stride);
        def("TEX_BYTES", tex_bytes);
        def("FIRST", wpb * 4);
        def("HEAD", HEAD as u32);
        def("TAG", u32::from_be_bytes(if s.alpha { *b"DXT5" } else { *b"DXT1" }));
        for (k, v) in [("TEX_OFF", tex), ("OPS_OFF", ops), ("NOPS_OFF", nops), ("OPB_OFF", opb), ("XB_OFF", xb), ("BOF_OFF", bof), ("BSUM_OFF", bsum), ("OPW_OFF", opw), ("KEYS_OFF", keys), ("VALS_OFF", vals), ("REC_OFF", rec_off)] {
            def(k, v);
        }
        def("BTOTAL", nb.div_ceil(1024).max(1));
        d.push_str(&format!("#define ALPHA {}\n", s.alpha as u32));
        let _ = bh;
        // Packet: header, then at most the raw texture (or the LZ stream, which is
        // used only when smaller).
        let out = Output::new(compute, 12 + tex_bytes as u64 + 64)?;
        d.push_str(&format!("#define CAP {}u\n", out.cap));
        let mut kernels = Vec::new();
        for (name, body) in [("keys", K_KEYS), ("prev", K_PREV), ("tex", K_TEX), ("match", K_MATCH), ("bytes", K_BYTES), ("finish", K_FINISH), ("opwords", K_OPWORDS), ("write", K_WRITE)] {
            let src = format!("{PRELUDE}{d}{COMMON}{}{KERNELS}{body}", super::bc_tables::BC_TABLES);
            kernels.push((name, compute.kernel(&format!("dxv_{name}_{}", super::spec_hash(&src)), &src, 0)?));
        }
        let work = compute.buffer(words as u64 * 4, "dxv work")?;
        Ok(Dxv { codec: Codec::Dxv(s), nb, kernels, scan: Scan::new(compute)?, sort: RadixSort::new(compute)?, work, out, layout: layout.clone(), planes_addr: 0, offs: [opb, osum, bof, bsum], sorting: [keys, vals, tkeys, tvals, scratch], nslots })
    }

    fn k(&self, name: &str) -> &Kernel {
        &self.kernels.iter().find(|(n, _)| *n == name).unwrap().1
    }

    fn record_all(&self, rec: &mut Recorder) -> Result<()> {
        let mut push = Push::new();
        push.0.extend(self.planes_addr.to_le_bytes());
        let push = push.ptr(&self.work).ptr(&self.out.dev);
        let label = std::mem::replace(&mut rec.label, "dxv".into());
        let blocks = |k: &Kernel| Recorder::groups(k, [self.nb, 1, 1]);
        rec.dispatch(self.k("tex"), blocks(self.k("tex")), &push)?;
        // Most recent earlier occurrence of each slot (sorted by value, stable).
        let [keys, vals, tkeys, tvals, scratch] = self.sorting;
        let slots: &[(u32, u32)] = if self.nslots == 4 { &[(0, 2), (2, 2), (2, 1), (3, 1)] } else { &[(0, 2), (0, 1), (1, 1)] };
        for (i, (k0, n)) in slots.iter().enumerate() {
            let q = push.clone().u(*k0).u(*n);
            rec.dispatch(self.k("keys"), blocks(self.k("keys")), &q)?;
            self.sort.record(rec, &self.work, keys, vals, tkeys, tvals, scratch, self.nb)?;
            let q = push.clone().u(i as u32).u(0);
            rec.dispatch(self.k("prev"), blocks(self.k("prev")), &q)?;
        }
        rec.dispatch(self.k("match"), blocks(self.k("match")), &push)?;
        let [opb, osum, bof, bsum] = self.offs;
        self.scan.record(rec, &self.work, opb, osum, self.nb)?;
        rec.dispatch(self.k("bytes"), blocks(self.k("bytes")), &push)?;
        self.scan.record(rec, &self.work, bof, bsum, self.nb)?;
        rec.dispatch(self.k("finish"), [1, 1, 1], &push)?;
        rec.dispatch(self.k("opwords"), blocks(self.k("opwords")), &push)?;
        rec.dispatch(self.k("write"), blocks(self.k("write")), &push)?;
        rec.label = "packet out".into();
        self.out.record_out(rec)?;
        rec.label = label;
        Ok(())
    }
}

impl Encoder for Dxv {
    fn record(&mut self, rec: &mut Recorder, planes: &Buffer, layout: &PackLayout) -> Result<()> {
        ensure!(layout.total == self.layout.total && layout.name == self.layout.name, "DXV: the picture layout changed");
        self.planes_addr = planes.addr;
        self.record_all(rec)
    }

    fn packet(&mut self) -> Result<Vec<u8>> {
        let (pkt, over) = self.out.read()?;
        if over {
            bail!("DXV: packet larger than the raw texture");
        }
        Ok(pkt)
    }

    fn codec(&self) -> &Codec {
        &self.codec
    }
}

/// Sort keys of a slot: the word itself, or a hash of the words.
const K_KEYS: &str = r#"
layout(local_size_x = 64, local_size_y = 1, local_size_z = 1) in;
void main() {
    uint b = gl_GlobalInvocationID.x;
    if (b >= NB) return;
    uint key = p.work.v[TEX_OFF + b * WPB + p.a];
    for (uint k = 1u; k < p.b; k++) key = (key ^ (key >> 15)) * 0x2c1b3c6du + p.work.v[TEX_OFF + b * WPB + p.a + k] * 0x297a2d39u;
    p.work.v[KEYS_OFF + b] = key;
    p.work.v[VALS_OFF + b] = b;
}
"#;

/// After the sort: each block's most recent earlier block with the same key.
const K_PREV: &str = r#"
layout(local_size_x = 64, local_size_y = 1, local_size_z = 1) in;
void main() {
    uint i = gl_GlobalInvocationID.x;
    if (i >= NB) return;
    uint prev = (i > 0u && p.work.v[KEYS_OFF + i - 1u] == p.work.v[KEYS_OFF + i]) ? p.work.v[VALS_OFF + i - 1u] : 0xffffffffu;
    p.work.v[REC_OFF + p.a * NB + p.work.v[VALS_OFF + i]] = prev;
}
"#;
