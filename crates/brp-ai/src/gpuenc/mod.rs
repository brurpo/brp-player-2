//! Our own encoders on the GPU (Vulkan compute, any vendor) for the intra-frame
//! post-production codecs no GPU has a hardware encoder for: ProRes, DNxHD/DNxHR,
//! Resolume DXV3 and MPEG-2 (XDCAM HD422).
//!
//! They run right after the export packer (`pack.rs`), which has already written
//! the picture in the codec's pixel format into a GPU buffer. Each encoder turns
//! those planes into one finished packet on the GPU: transform, quantisation, rate
//! control and entropy coding, in pieces the format codes independently (ProRes
//! slices, DNx macroblock rows, DXV texture blocks, MPEG-2 slices). Only the packet
//! is copied back, and FFmpeg just muxes it (`brp_media::encode::Precoded`).
//!
//! Packets are written into a zeroed device buffer with byte-granular `atomicOr`s
//! (no 8-bit storage needed), behind a 16-byte header the kernels fill:
//! word 0 = packet bytes, word 1 = overflow flag. A last kernel copies exactly the
//! packet to host memory and zeroes what it copied, ready for the next frame.

pub mod dnxhd;
mod bc_tables;
pub mod dxv;
pub mod prores;

use anyhow::{bail, Context as _, Result};
use brp_media::encode::PackLayout;
use brp_render::compute::{Buffer, Compute, Kernel, Push, Recorder, PRELUDE};
use std::sync::Arc;

/// What to encode: the codec and its settings (sizes come from the packed layout).
#[derive(Clone, Debug, PartialEq)]
pub enum Codec {
    ProRes(prores::Settings),
    Dnx(dnxhd::Settings),
    Dxv(dxv::Settings),
}

/// An encoder on the GPU: records its work after the packer, then gives the packet.
pub trait Encoder: Send {
    /// Record the encoding of the picture in `planes` (laid out as `layout`).
    fn record(&mut self, rec: &mut Recorder, planes: &Buffer, layout: &PackLayout) -> Result<()>;
    /// The packet of the last recording (once it has run).
    fn packet(&mut self) -> Result<Vec<u8>>;
    fn codec(&self) -> &Codec;
}

/// The encoder for `codec`, its kernels compiled.
pub fn create(compute: &Arc<Compute>, codec: &Codec, layout: &PackLayout) -> Result<Box<dyn Encoder>> {
    // Test hook: the caller's fallback to the CPU encoder.
    if std::env::var_os("BRP_TEST_GPUENC_FAIL").is_some() {
        bail!("GPU encoder disabled by BRP_TEST_GPUENC_FAIL");
    }
    Ok(match codec {
        Codec::ProRes(s) => Box::new(prores::ProRes::new(compute, s.clone(), layout)?),
        Codec::Dnx(s) => Box::new(dnxhd::Dnx::new(compute, s.clone(), layout)?),
        Codec::Dxv(s) => Box::new(dxv::Dxv::new(compute, s.clone(), layout)?),
    })
}

/// Bytes before the packet in the output buffer (sizes and flags).
pub const HEAD: u64 = 16;

/// GLSL shared by the encoders: byte and bit writers into a zeroed U32 buffer.
pub const COMMON: &str = r#"
// Byte `b` at byte `pos` of a zeroed buffer.
void put_byte(U32 o, uint pos, uint b) {
    atomicOr(o.v[pos >> 2], (b & 255u) << (8u * (pos & 3u)));
}
void put_be16(U32 o, uint pos, uint v) { put_byte(o, pos, v >> 8); put_byte(o, pos + 1u, v); }
void put_be32(U32 o, uint pos, uint v) { put_be16(o, pos, v >> 16); put_be16(o, pos + 2u, v); }
void put_le32(U32 o, uint pos, uint v) {
    uint sh = 8u * (pos & 3u);
    atomicOr(o.v[pos >> 2], v << sh);
    if (sh != 0u) atomicOr(o.v[(pos >> 2) + 1u], v >> (32u - sh));
}
void put_le16(U32 o, uint pos, uint v) { put_byte(o, pos, v); put_byte(o, pos + 1u, v >> 8); }
uint bswap32(uint w) {
    return (w >> 24) | ((w >> 8) & 0xff00u) | ((w << 8) & 0xff0000u) | (w << 24);
}
// Big-endian bit writer (MSB first), byte-aligned start.
struct BitW { uint64_t acc; uint n; uint pos; };
BitW bw_new(uint pos) { BitW w; w.acc = 0ul; w.n = 0u; w.pos = pos; return w; }
uint lowbits(uint v, uint n) { return n >= 32u ? v : (v & ((1u << n) - 1u)); }
void bw_put(U32 o, inout BitW w, uint n, uint v) {
    if (n == 0u) return;
    w.acc = (w.acc << n) | uint64_t(lowbits(v, n));
    w.n += n;
    if (w.n >= 32u) {
        uint word = uint(w.acc >> (w.n - 32u));
        w.n -= 32u;
        put_le32(o, w.pos, bswap32(word));
        w.pos += 4u;
    }
}
// Up to 64 zero bits (long prefixes).
void bw_zeros(U32 o, inout BitW w, uint n) {
    while (n > 24u) { bw_put(o, w, 24u, 0u); n -= 24u; }
    bw_put(o, w, n, 0u);
}
// Pad with zero bits to a byte boundary and write what is left; returns the end.
uint bw_flush(U32 o, inout BitW w) {
    while (w.n >= 8u) {
        put_byte(o, w.pos, uint(w.acc >> (w.n - 8u)));
        w.pos += 1u;
        w.n -= 8u;
    }
    if (w.n > 0u) {
        put_byte(o, w.pos, uint(w.acc << (8u - w.n)));
        w.pos += 1u;
        w.n = 0u;
    }
    return w.pos;
}
"#;

/// Copies the packet (header word 0 = its bytes) to host memory and zeroes it.
const COPY_OUT: &str = r#"
layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;
layout(push_constant) uniform P { U32 src; U32 dst; uint cap_words; } p;
void main() {
    uint words = min((16u + p.src.v[0] + 3u) / 4u, p.cap_words);
    uint stride = gl_NumWorkGroups.x * 256u;
    for (uint i = gl_GlobalInvocationID.x; i < words; i += stride) {
        p.dst.v[i] = p.src.v[i];
    }
}
"#;

/// Zeroes the packet region after the host copy was recorded (separate pass: the
/// copy above must have read every word first).
const ZERO_OUT: &str = r#"
layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;
layout(push_constant) uniform P { U32 dst; U32 head; uint cap_words; } p;
void main() {
    uint words = min((16u + p.head.v[0] + 3u) / 4u, p.cap_words);
    uint stride = gl_NumWorkGroups.x * 256u;
    for (uint i = gl_GlobalInvocationID.x; i < words; i += stride) p.dst.v[i] = 0u;
}
"#;

/// The packet buffers: device output (zeroed between frames) and host copy.
pub struct Output {
    pub dev: Buffer,
    host: Buffer,
    /// Packet bytes the buffers hold.
    pub cap: u64,
    copy: Arc<Kernel>,
    zero: Arc<Kernel>,
    /// A small buffer keeping the last packet size for the zero pass.
    size_keep: Buffer,
}

impl Output {
    pub fn new(compute: &Arc<Compute>, cap: u64) -> Result<Output> {
        let cap = cap.next_multiple_of(1 << 16);
        let dev = compute.buffer(HEAD + cap, "gpu encoder packet")?;
        // Zero once; afterwards the copy pass leaves it zeroed.
        let mut rec = Recorder::new(compute)?;
        let z = compute.kernel("gpuenc_fill0", &format!("{PRELUDE}{FILL0}"), 0)?;
        let words = ((HEAD + cap) / 4) as u32;
        rec.dispatch(&z, [1024, 1, 1], &Push::new().ptr(&dev).u(words))?;
        rec.submit_wait()?;
        Ok(Output {
            host: compute.readback(HEAD + cap, "gpu encoder packet (host)")?,
            size_keep: compute.buffer(16, "gpu encoder size")?,
            dev,
            cap,
            copy: compute.kernel("gpuenc_copy_out", &format!("{PRELUDE}{COPY_OUT}"), 0)?,
            zero: compute.kernel("gpuenc_zero_out", &format!("{PRELUDE}{ZERO_OUT}"), 0)?,
        })
    }

    /// Record the copy of the finished packet to the host, then clear the device
    /// side (header included) for the next frame.
    pub fn record_out(&self, rec: &mut Recorder) -> Result<()> {
        let cap_words = ((HEAD + self.cap) / 4) as u32;
        rec.dispatch(&self.copy, [512, 1, 1], &Push::new().ptr(&self.dev).ptr(&self.host).u(cap_words))?;
        rec.copy(&self.dev, 0, &self.size_keep, 0, 16);
        // Clears the packet and its header (the size it needs is in `size_keep`).
        rec.dispatch(&self.zero, [512, 1, 1], &Push::new().ptr(&self.dev).ptr(&self.size_keep).u(cap_words))?;
        Ok(())
    }

    /// The packet the last recording made: (bytes, overflowed).
    pub fn read(&mut self) -> Result<(Vec<u8>, bool)> {
        let m = self.host.mapped().context("packet readback not mapped")?;
        let n = u32::from_le_bytes(m[0..4].try_into().unwrap()) as u64;
        let over = u32::from_le_bytes(m[4..8].try_into().unwrap()) != 0;
        if u32::from_le_bytes(m[8..12].try_into().unwrap()) != 0 {
            bail!("GPU encoder: coded sizes differ from the estimate (internal error)");
        }
        if over || n > self.cap {
            return Ok((Vec::new(), true));
        }
        Ok((m[HEAD as usize..(HEAD + n) as usize].to_vec(), false))
    }
}

const FILL0: &str = r#"
layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;
layout(push_constant) uniform P { U32 dst; uint words; } p;
void main() {
    uint stride = gl_NumWorkGroups.x * 256u;
    for (uint i = gl_GlobalInvocationID.x; i < p.words; i += stride) p.dst.v[i] = 0u;
}
"#;

/// GLSL array literal of `name` with `v` (ints).
pub fn glsl_array(ty: &str, name: &str, v: &[i64]) -> String {
    let body: Vec<String> = v.iter().map(|x| x.to_string()).collect();
    format!("const {ty} {name}[{}] = {ty}[]({});\n", v.len(), body.join(","))
}

/// A short stable hash of a kernel's specialisation (its name in the kernel cache).
pub fn spec_hash(s: &str) -> String {
    let mut h: u64 = 0xcbf29ce484222325;
    for b in s.bytes() {
        h ^= b as u64;
        h = h.wrapping_mul(0x100000001b3);
    }
    format!("{h:016x}")
}

/// A sample of a packed plane: the layout's byte offset and stride, 8 or 16 bits.
pub fn plane_defs(prefix: &str, l: &PackLayout, plane: usize) -> Result<String> {
    let Some(p) = l.planes.get(plane) else { bail!("pixel format {} has no plane {plane}", l.name) };
    if p.step != 1 {
        bail!("GPU encoder: plane {plane} of {} is not planar", l.name);
    }
    Ok(format!("#define {prefix}_OFF {}u\n#define {prefix}_STRIDE {}u\n#define {prefix}_BYTES {}u\n", p.offset, p.stride, p.bytes))
}

/// GLSL to read a sample of a plane defined by [`plane_defs`].
pub const SAMPLE: &str = r#"
uint sample16(U32 s, uint off, uint stride, uint x, uint y) {
    uint b = off + y * stride + x * 2u;
    return (s.v[b >> 2] >> (8u * (b & 3u))) & 0xffffu;
}
uint sample8(U32 s, uint off, uint stride, uint x, uint y) {
    uint b = off + y * stride + x;
    return (s.v[b >> 2] >> (8u * (b & 3u))) & 0xffu;
}
"#;

/// Orthonormal 8-point DCT basis, c(u) cos((2x+1)u pi / 16) / 2, as a GLSL table.
pub fn dct_table() -> String {
    let mut v = Vec::with_capacity(64);
    for u in 0..8 {
        for x in 0..8 {
            let c = if u == 0 { (0.125f64).sqrt() } else { 0.5 };
            v.push(c * ((2 * x + 1) as f64 * u as f64 * std::f64::consts::PI / 16.0).cos());
        }
    }
    let body: Vec<String> = v.iter().map(|x| format!("{x:.9}")).collect();
    format!("const float DCTB[64] = float[]({});\n", body.join(","))
}

/// A forward 8x8 DCT of `blk` (raster, in place), orthonormal: the DC is 8 x mean.
pub const DCT8: &str = r#"
void fdct8x8(inout float blk[64]) {
    float t[64];
    for (int y = 0; y < 8; y++)
        for (int u = 0; u < 8; u++) {
            float s = 0.0;
            for (int x = 0; x < 8; x++) s += blk[y * 8 + x] * DCTB[u * 8 + x];
            t[y * 8 + u] = s;
        }
    for (int u = 0; u < 8; u++)
        for (int v = 0; v < 8; v++) {
            float s = 0.0;
            for (int y = 0; y < 8; y++) s += t[y * 8 + u] * DCTB[v * 8 + y];
            blk[v * 8 + u] = s;
        }
}
"#;

// ---- prefix sums ------------------------------------------------------------------------

/// Exclusive prefix sum of `n` u32 values in place (up to 1024 x 65535 values),
/// three passes: per-group sums, their scan (also the total, at `total`), and each
/// group's own scan plus its offset.
pub struct Scan {
    reduce: Arc<Kernel>,
    top: Arc<Kernel>,
    apply: Arc<Kernel>,
}

const SCAN_REDUCE: &str = r#"
layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;
layout(push_constant) uniform P { U32 data; U32 sums; uint n; uint total; } p;
shared uint part[256];
void main() {
    uint g = gl_WorkGroupID.x, t = gl_LocalInvocationID.x;
    uint base = g * 1024u + t * 4u, s = 0u;
    for (uint k = 0u; k < 4u; k++) if (base + k < p.n) s += p.data.v[base + k];
    part[t] = s;
    barrier();
    for (uint w = 128u; w > 0u; w >>= 1) {
        if (t < w) part[t] += part[t + w];
        barrier();
    }
    if (t == 0u) p.sums.v[g] = part[0];
}
"#;

const SCAN_TOP: &str = r#"
layout(local_size_x = 1024, local_size_y = 1, local_size_z = 1) in;
layout(push_constant) uniform P { U32 data; U32 sums; uint n; uint total; } p;
shared uint part[1024];
void main() {
    uint t = gl_LocalInvocationID.x;
    uint groups = (p.n + 1023u) / 1024u;
    uint per = (groups + 1023u) / 1024u, s0 = min(t * per, groups), s1 = min(s0 + per, groups);
    uint s = 0u;
    for (uint i = s0; i < s1; i++) s += p.sums.v[i];
    part[t] = s;
    barrier();
    if (t == 0u) {
        uint acc = 0u;
        for (uint i = 0u; i < 1024u; i++) { uint v = part[i]; part[i] = acc; acc += v; }
        p.sums.v[p.total] = acc;
    }
    barrier();
    uint acc = part[t];
    for (uint i = s0; i < s1; i++) { uint v = p.sums.v[i]; p.sums.v[i] = acc; acc += v; }
}
"#;

const SCAN_APPLY: &str = r#"
layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;
layout(push_constant) uniform P { U32 data; U32 sums; uint n; uint total; } p;
shared uint part[256];
void main() {
    uint g = gl_WorkGroupID.x, t = gl_LocalInvocationID.x;
    uint base = g * 1024u + t * 4u;
    uint v[4];
    uint s = 0u;
    for (uint k = 0u; k < 4u; k++) { v[k] = base + k < p.n ? p.data.v[base + k] : 0u; s += v[k]; }
    part[t] = s;
    barrier();
    // Inclusive Hillis-Steele scan of the 256 thread sums.
    for (uint w = 1u; w < 256u; w <<= 1) {
        uint x = t >= w ? part[t - w] : 0u;
        barrier();
        part[t] += x;
        barrier();
    }
    uint acc = p.sums.v[g] + part[t] - s;
    for (uint k = 0u; k < 4u; k++) {
        if (base + k < p.n) p.data.v[base + k] = acc;
        acc += v[k];
    }
}
"#;

impl Scan {
    pub fn new(compute: &Compute) -> Result<Scan> {
        let k = |name: &str, body: &str| compute.kernel(&format!("gpuenc_scan_{name}"), &format!("{PRELUDE}{body}"), 0);
        Ok(Scan { reduce: k("reduce", SCAN_REDUCE)?, top: k("top", SCAN_TOP)?, apply: k("apply", SCAN_APPLY)? })
    }

    /// Scan `n` values at word `data` of `buf`, using `sums` (at least n/1024 + 1
    /// words, the last one getting the total at word `sums + ceil(n/1024)`).
    pub fn record(&self, rec: &mut Recorder, buf: &Buffer, data: u32, sums: u32, n: u32) -> Result<()> {
        let groups = n.div_ceil(1024).max(1);
        let push = Push::new().ptr_at(buf, data as u64 * 4).ptr_at(buf, sums as u64 * 4).u(n).u(groups);
        rec.dispatch(&self.reduce, [groups, 1, 1], &push)?;
        rec.dispatch(&self.top, [1, 1, 1], &push)?;
        rec.dispatch(&self.apply, [groups, 1, 1], &push)?;
        Ok(())
    }
}

// ---- radix sort -------------------------------------------------------------------------

/// Stable sort of `n` (key, value) u32 pairs by key: four 8-bit passes, each a
/// histogram per 256-element chunk, a scan of the histograms, and a stable scatter
/// (rank among the chunk's earlier elements with the same digit).
pub struct RadixSort {
    hist: Arc<Kernel>,
    scatter: Arc<Kernel>,
    scan: Scan,
}

const SORT_HIST: &str = r#"
layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;
layout(push_constant) uniform P { U32 keys; U32 vals; U32 okeys; U32 ovals; U32 hist; uint n; uint shift; uint groups; } p;
shared uint h[256];
void main() {
    uint g = gl_WorkGroupID.x, t = gl_LocalInvocationID.x, i = g * 256u + t;
    h[t] = 0u;
    barrier();
    if (i < p.n) atomicAdd(h[(p.keys.v[i] >> p.shift) & 255u], 1u);
    barrier();
    p.hist.v[t * p.groups + g] = h[t];
}
"#;

const SORT_SCATTER: &str = r#"
layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;
layout(push_constant) uniform P { U32 keys; U32 vals; U32 okeys; U32 ovals; U32 hist; uint n; uint shift; uint groups; } p;
shared uint dig[256];
void main() {
    uint g = gl_WorkGroupID.x, t = gl_LocalInvocationID.x, i = g * 256u + t;
    uint key = i < p.n ? p.keys.v[i] : 0u;
    uint d = (key >> p.shift) & 255u;
    dig[t] = i < p.n ? d : 0xffffffffu;
    barrier();
    if (i >= p.n) return;
    uint rank = 0u;
    for (uint j = 0u; j < t; j++) rank += dig[j] == d ? 1u : 0u;
    uint dst = p.hist.v[d * p.groups + g] + rank;
    p.okeys.v[dst] = key;
    p.ovals.v[dst] = p.vals.v[i];
}
"#;

impl RadixSort {
    pub fn new(compute: &Compute) -> Result<RadixSort> {
        let k = |name: &str, body: &str| compute.kernel(&format!("gpuenc_sort_{name}"), &format!("{PRELUDE}{body}"), 0);
        Ok(RadixSort { hist: k("hist", SORT_HIST)?, scatter: k("scatter", SORT_SCATTER)?, scan: Scan::new(compute)? })
    }

    /// Words of scratch the sort needs for `n` pairs (histograms and their sums).
    pub fn scratch_words(n: u32) -> u32 {
        let groups = n.div_ceil(256).max(1);
        256 * groups + (256 * groups).div_ceil(1024) + 2
    }

    /// Sort `n` pairs at words `keys`/`vals` of `buf`, using `tmp_keys`/`tmp_vals`
    /// (n words each) and `scratch`: the result is back at `keys`/`vals`.
    #[allow(clippy::too_many_arguments)]
    pub fn record(&self, rec: &mut Recorder, buf: &Buffer, keys: u32, vals: u32, tmp_keys: u32, tmp_vals: u32, scratch: u32, n: u32) -> Result<()> {
        let groups = n.div_ceil(256).max(1);
        let sums = scratch + 256 * groups;
        let at = |w: u32| w as u64 * 4;
        for pass in 0..4u32 {
            let (k, v, ok, ov) = if pass % 2 == 0 { (keys, vals, tmp_keys, tmp_vals) } else { (tmp_keys, tmp_vals, keys, vals) };
            let push = Push::new().ptr_at(buf, at(k)).ptr_at(buf, at(v)).ptr_at(buf, at(ok)).ptr_at(buf, at(ov)).ptr_at(buf, at(scratch)).u(n).u(pass * 8).u(groups);
            rec.dispatch(&self.hist, [groups, 1, 1], &push)?;
            self.scan.record(rec, buf, scratch, sums, 256 * groups)?;
            rec.dispatch(&self.scatter, [groups, 1, 1], &push)?;
        }
        Ok(())
    }
}

/// Workgroups for `n` items of a 1-D kernel.
pub fn groups1(k: &Kernel, n: u32) -> [u32; 3] {
    Recorder::groups(k, [n, 1, 1])
}
