//! Resolume DXV (DXV3: DXT1, and DXT5 with alpha) on the GPU.
//!
//! A DXV frame is DXT texture blocks packed with a simple LZ-style scheme. The
//! unpacking is a sequential byte stream, so it runs on the CPU (straight into the
//! upload buffer, ~1 MB per 1080p frame); expanding the blocks into pixels, the bulk
//! of the work, runs on the GPU, one thread per 4x4 block, with FFmpeg's integer
//! arithmetic (`texturedsp.c`) so the pixels are identical. Hardware BC sampling is
//! not used: vendors round the interpolated colours differently.
//!
//! On the GPU: DXT1 and DXT5 with the 12-byte header (DXV3). FFmpeg takes the
//! high-quality YCoCg variants (YCG6, YG10) and the old headerless DXV2 frames.

use super::{Io, Planes, BITS_GLSL};
use crate::compute::{Compute, Push, Recorder};
use anyhow::{bail, ensure, Result};
use brp_media::gpudec::{pix, GpuDecoder, GpuPicture, StreamDesc};
use brp_media::FrameLayout;
use std::sync::Arc;

const KERNEL: &str = r#"
layout(local_size_x = 64) in;
layout(push_constant) uniform P { U32 src; U32 dst; uint bw; uint bh; uint stride; uint fmt; } p;

uint rgba(uint r, uint g, uint b, uint a) { return (r & 255u) | ((g & 255u) << 8) | ((b & 255u) << 16) | ((a & 255u) << 24); }

// texturedsp.c extract_color()
void extract(uint c0, uint c1, bool dxtn, uint alpha, out uint col[4]) {
    uint t;
    t = (c0 >> 11) * 255u + 16u;            uint r0 = ((t / 32u + t) / 32u) & 255u;
    t = ((c0 & 0x07E0u) >> 5) * 255u + 32u; uint g0 = ((t / 64u + t) / 64u) & 255u;
    t = (c0 & 0x001Fu) * 255u + 16u;        uint b0 = ((t / 32u + t) / 32u) & 255u;
    t = (c1 >> 11) * 255u + 16u;            uint r1 = ((t / 32u + t) / 32u) & 255u;
    t = ((c1 & 0x07E0u) >> 5) * 255u + 32u; uint g1 = ((t / 64u + t) / 64u) & 255u;
    t = (c1 & 0x001Fu) * 255u + 16u;        uint b1 = ((t / 32u + t) / 32u) & 255u;
    uint a = dxtn ? 0u : 255u;
    col[0] = rgba(r0, g0, b0, a);
    col[1] = rgba(r1, g1, b1, a);
    if (dxtn || c0 > c1) {
        col[2] = rgba((2u * r0 + r1) / 3u, (2u * g0 + g1) / 3u, (2u * b0 + b1) / 3u, a);
        col[3] = rgba((2u * r1 + r0) / 3u, (2u * g1 + g0) / 3u, (2u * b1 + b0) / 3u, a);
    } else {
        col[2] = rgba((r0 + r1) / 2u, (g0 + g1) / 2u, (b0 + b1) / 2u, a);
        col[3] = rgba(0u, 0u, 0u, alpha);
    }
}

void main() {
    uint b = gl_GlobalInvocationID.x;
    if (b >= p.bw * p.bh) return;
    uint bx = b % p.bw, by = b / p.bw;
    uint base = by * 4u * p.stride + bx * 4u;
    uint col[4];
    if (p.fmt == 0u) {
        // DXT1 (dxt1_block, opaque).
        uint w0 = p.src.v[b * 2u], code = p.src.v[b * 2u + 1u];
        extract(w0 & 0xffffu, w0 >> 16, false, 255u, col);
        for (uint y = 0u; y < 4u; y++)
            for (uint x = 0u; x < 4u; x++) {
                p.dst.v[base + y * p.stride + x] = col[code & 3u];
                code >>= 2;
            }
    } else {
        // DXT5 read as DXT4 (dxt4_block): premultiplied colours, made straight.
        uint w0 = p.src.v[b * 4u], w1 = p.src.v[b * 4u + 1u], w2 = p.src.v[b * 4u + 2u], code = p.src.v[b * 4u + 3u];
        uint a0 = w0 & 255u, a1 = (w0 >> 8) & 255u;
        uint lo = (w0 >> 16) | ((w1 & 255u) << 16), hi = w1 >> 8;
        extract(w2 & 0xffffu, w2 >> 16, true, 0u, col);
        for (uint i = 0u; i < 16u; i++) {
            uint ac = i < 8u ? (lo >> (3u * i)) & 7u : (hi >> (3u * (i - 8u))) & 7u;
            uint a;
            if (ac == 0u) a = a0;
            else if (ac == 1u) a = a1;
            else if (a0 > a1) a = (((8u - ac) * a0 + (ac - 1u) * a1) / 7u) & 255u;
            else if (ac == 6u) a = 0u;
            else if (ac == 7u) a = 255u;
            else a = (((6u - ac) * a0 + (ac - 1u) * a1) / 5u) & 255u;
            uint c = col[code & 3u];
            code >>= 2;
            uint r = c & 255u, g = (c >> 8) & 255u, bl = (c >> 16) & 255u;
            // premult2straight(): back to straight alpha, c * 255 / a (at most 255);
            // colours with alpha 0 are left as they are.
            if (a != 0u) { r = min(r * 255u / a, 255u); g = min(g * 255u / a, 255u); bl = min(bl * 255u / a, 255u); }
            p.dst.v[base + (i >> 2) * p.stride + (i & 3u)] = rgba(r, g, bl, a);
        }
    }
}
"#;

const TAG_DXT1: u32 = u32::from_be_bytes(*b"DXT1");
const TAG_DXT5: u32 = u32::from_be_bytes(*b"DXT5");

pub struct Decoder {
    io: Option<Io>,
    compute: Arc<Compute>,
    budget: usize,
    /// Picture size, and the block-aligned size FFmpeg decodes (coded size).
    w: u32,
    h: u32,
    cw: u32,
    ch: u32,
    /// The unpacked texture (reused).
    tex: Vec<u8>,
}

impl Decoder {
    pub fn new(compute: Arc<Compute>, budget: usize, s: &StreamDesc) -> Decoder {
        // FFmpeg aligns the size to whole 4x4 blocks.
        Decoder { io: None, compute, budget, w: s.width, h: s.height, cw: s.width.next_multiple_of(4), ch: s.height.next_multiple_of(4), tex: Vec::new() }
    }
}

impl GpuDecoder for Decoder {
    fn name(&self) -> &str {
        "DXV (GPU)"
    }

    fn vram_budget(&self, _held: usize) -> usize {
        self.budget
    }

    fn decode(&mut self, data: &[u8]) -> Result<GpuPicture> {
        ensure!(self.w > 0 && self.h > 0, "no picture size");
        ensure!(data.len() >= 12, "short packet");
        let tag = u32::from_le_bytes(data[..4].try_into().unwrap());
        let dxt5 = match tag {
            TAG_DXT1 => false,
            TAG_DXT5 => true,
            _ => bail!("DXV texture format {:?} (not on the GPU)", String::from_utf8_lossy(&data[..4])),
        };
        // 12-byte header: tag, version major+1, minor, raw flag, unknown, size.
        let raw = data[6] != 0;
        let size = u32::from_le_bytes(data[8..12].try_into().unwrap()) as usize;
        let body = &data[12..];
        ensure!(size == body.len(), "DXV size {size} != {}", body.len());
        let ratio = if dxt5 { 16 } else { 8 };
        let tex_size = (self.cw / 4 * self.ch / 4) as usize * ratio;

        if self.io.is_none() {
            self.io = Some(Io::new(self.compute.clone())?);
        }
        // Unpacked in system memory (back references read it; the upload buffer is
        // write-combined, very slow to read), then copied over.
        let tex = &mut self.tex;
        tex.clear();
        tex.resize(tex_size, 0);
        if raw {
            let n = body.len().min(tex_size);
            tex[..n].copy_from_slice(&body[..n]);
        } else if dxt5 {
            unpack_dxt5(body, tex)?;
        } else {
            unpack_dxt1(body, tex)?;
        }
        let io = self.io.as_mut().unwrap();
        io.staging(tex_size)?[..tex_size].copy_from_slice(tex);

        let kernel = self.compute.kernel("dxv_blocks", &format!("{}{BITS_GLSL}{KERNEL}", crate::compute::PRELUDE), 0)?;
        let planes = io.planes(&[(self.cw as u64 * 4, self.ch as u64)])?;
        let (bw, bh) = (self.cw / 4, self.ch / 4);
        let mut rec = Recorder::new(&self.compute)?;
        rec.label = "dxv".into();
        io.upload(&mut rec, tex_size)?;
        let push = Push::new().ptr(io.input()).ptr(&planes.buf).u(bw).u(bh).u(self.cw).u(dxt5 as u32);
        rec.dispatch(&kernel, Recorder::groups(&kernel, [bw * bh, 1, 1]), &push)?;
        rec.submit_wait()?;
        Ok(picture(planes, self.w, self.h))
    }
}

fn picture(planes: Planes, w: u32, h: u32) -> GpuPicture {
    GpuPicture {
        planes: Arc::new(planes),
        layout: FrameLayout::Rgba8 { offsets: [0, 1, 2], alpha: Some(3) },
        pix_fmt: pix::RGBA,
        width: w,
        height: h,
        interlaced: false,
        tff: false,
    }
}

/// Little-endian reader that, like FFmpeg's bytestream2, gives zeros past the end.
struct Bytes<'a> {
    d: &'a [u8],
    at: usize,
}

impl Bytes<'_> {
    fn left(&self) -> usize {
        self.d.len().saturating_sub(self.at)
    }
    fn u8(&mut self) -> u32 {
        if self.left() < 1 {
            self.at = self.d.len();
            return 0;
        }
        self.at += 1;
        self.d[self.at - 1] as u32
    }
    fn u16(&mut self) -> u32 {
        if self.left() < 2 {
            self.at = self.d.len();
            return 0;
        }
        self.at += 2;
        u16::from_le_bytes([self.d[self.at - 2], self.d[self.at - 1]]) as u32
    }
    fn u32(&mut self) -> u32 {
        if self.left() < 4 {
            self.at = self.d.len();
            return 0;
        }
        self.at += 4;
        u32::from_le_bytes(self.d[self.at - 4..self.at].try_into().unwrap())
    }
}

/// The texture as 32-bit words.
struct Words<'a>(&'a mut [u8]);

impl Words<'_> {
    fn len(&self) -> usize {
        self.0.len() / 4
    }
    fn get(&self, i: usize) -> u32 {
        u32::from_le_bytes(self.0[i * 4..i * 4 + 4].try_into().unwrap())
    }
    fn set(&mut self, i: usize, v: u32) {
        self.0[i * 4..i * 4 + 4].copy_from_slice(&v.to_le_bytes());
    }
    /// Copy word `pos - idx` to `pos` (checked).
    fn back(&mut self, pos: usize, idx: usize) -> Result<()> {
        ensure!(idx <= pos && pos < self.len(), "DXV back reference out of range");
        let v = self.get(pos - idx);
        self.set(pos, v);
        Ok(())
    }
}

/// Op state shared by the unpackers (FFmpeg's CHECKPOINT macro): 2-bit ops read
/// 16 at a time from a 32-bit word.
struct Ops {
    value: u32,
    state: u32,
}

impl Ops {
    /// Next op; for ops 1..3 also the back distance (in words) `idx`.
    fn next(&mut self, g: &mut Bytes, x: usize, pos: usize, idx: &mut usize) -> Result<u32> {
        if self.state == 0 {
            ensure!(g.left() >= 4, "DXV ops past the end");
            self.value = g.u32();
            self.state = 16;
        }
        let op = self.value & 3;
        self.value >>= 2;
        self.state -= 1;
        match op {
            1 => *idx = x,
            2 => {
                *idx = (g.u8() as usize + 2) * x;
                ensure!(*idx <= pos, "DXV idx {} > {pos}", *idx);
            }
            3 => {
                *idx = (g.u16() as usize + 0x102) * x;
                ensure!(*idx <= pos, "DXV idx {} > {pos}", *idx);
            }
            _ => {}
        }
        Ok(op)
    }
}

/// `dxv_decompress_dxt1`.
fn unpack_dxt1(src: &[u8], tex: &mut [u8]) -> Result<()> {
    let mut g = Bytes { d: src, at: 0 };
    let mut t = Words(tex);
    let n = t.len();
    ensure!(n >= 2, "texture too small");
    t.set(0, g.u32());
    t.set(1, g.u32());
    let mut ops = Ops { value: 0, state: 0 };
    let mut idx = 0usize;
    let mut pos = 2usize;
    while pos + 2 <= n {
        let op = ops.next(&mut g, 2, pos, &mut idx)?;
        if op != 0 {
            t.back(pos, idx)?;
            pos += 1;
            t.back(pos, idx)?;
            pos += 1;
        } else {
            for _ in 0..2 {
                let op = ops.next(&mut g, 2, pos, &mut idx)?;
                if op != 0 {
                    t.back(pos, idx)?;
                } else {
                    let v = g.u32();
                    t.set(pos, v);
                }
                pos += 1;
            }
        }
    }
    Ok(())
}

/// `dxv_decompress_dxt5`.
fn unpack_dxt5(src: &[u8], tex: &mut [u8]) -> Result<()> {
    let mut g = Bytes { d: src, at: 0 };
    let mut t = Words(tex);
    let n = t.len();
    ensure!(n >= 4, "texture too small");
    for i in 0..4 {
        let v = g.u32();
        t.set(i, v);
    }
    let mut ops = Ops { value: 0, state: 0 };
    let (mut idx, mut run, mut pos) = (0usize, 0u32, 4usize);
    let copy4 = |t: &mut Words, pos: &mut usize| -> Result<()> {
        t.back(*pos, 4)?;
        *pos += 1;
        t.back(*pos, 4)?;
        *pos += 1;
        Ok(())
    };
    while pos + 2 <= n {
        let op;
        if run > 0 {
            run -= 1;
            copy4(&mut t, &mut pos)?;
        } else {
            ensure!(g.left() > 0, "DXV data ends early");
            if ops.state == 0 {
                ops.value = g.u32();
                ops.state = 16;
            }
            let o = ops.value & 3;
            ops.value >>= 2;
            ops.state -= 1;
            match o {
                0 => {
                    // Long copy of whole blocks from 4 words back.
                    let mut check = g.u8() + 1;
                    if check == 256 {
                        loop {
                            let probe = g.u16();
                            check += probe;
                            if probe != 0xFFFF {
                                break;
                            }
                        }
                    }
                    while check > 0 && pos + 4 <= n {
                        for _ in 0..4 {
                            t.back(pos, 4)?;
                            pos += 1;
                        }
                        check -= 1;
                    }
                    continue;
                }
                1 => {
                    run = g.u8();
                    if run == 255 {
                        loop {
                            let probe = g.u16();
                            run += probe;
                            if probe != 0xFFFF {
                                break;
                            }
                        }
                    }
                    copy4(&mut t, &mut pos)?;
                }
                2 => {
                    // Back distance in whole 4-word blocks, past the last two.
                    let i = 8 + 4 * g.u16() as usize;
                    ensure!(i <= pos && pos - i + 2 <= n, "DXV idx out of range");
                    t.back(pos, i)?;
                    pos += 1;
                    t.back(pos, i)?;
                    pos += 1;
                }
                _ => {
                    let a = g.u32();
                    t.set(pos, a);
                    let b = g.u32();
                    t.set(pos + 1, b);
                    pos += 2;
                }
            }
        }
        op = ops.next(&mut g, 4, pos, &mut idx)?;
        ensure!(pos + 2 <= n, "DXV texture overrun");
        if op != 0 {
            ensure!(idx <= pos && pos - idx + 2 <= n, "DXV idx out of range");
            t.back(pos, idx)?;
            pos += 1;
            t.back(pos, idx)?;
            pos += 1;
        } else {
            for _ in 0..2 {
                let op = ops.next(&mut g, 4, pos, &mut idx)?;
                ensure!(op == 0 || (idx <= pos && pos - idx + 2 <= n), "DXV idx out of range");
                if op != 0 {
                    t.back(pos, idx)?;
                } else {
                    let v = g.u32();
                    t.set(pos, v);
                }
                pos += 1;
            }
        }
    }
    Ok(())
}
