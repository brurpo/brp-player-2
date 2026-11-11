//! Video decoders on the GPU (Vulkan compute) for intra-only production codecs:
//! DXV (Resolume), ProRes, DNxHD/DNxHR. Registered with the media layer through
//! [`factory`]; see `brp_media::gpudec`.
//!
//! Each packet is copied to GPU memory once, decoded there by compute kernels, and the
//! picture stays on the GPU (`Planes`): the renderer copies it into its textures
//! without a CPU round trip. Every decoder matches FFmpeg's output bit for bit (the
//! same integer arithmetic), so frames are identical whichever path made them.
//! Streams or packets a decoder does not take go back to FFmpeg.

pub mod dnxhd;
pub mod dnxhd_tables;
pub mod dxv;
pub mod prores;
pub mod tables;

use crate::compute::{Buffer, Compute, Recorder};
use crate::context::GpuContext;
use anyhow::{Context as _, Result};
use ash::vk;
use brp_media::gpudec::{codec, GpuDecoder, GpuDecoders, GpuPlanes, StreamDesc};
use gpu_allocator::MemoryLocation;
use std::sync::Arc;

/// GPU decoders on one device.
pub struct Factory {
    compute: Arc<Compute>,
    vram_budget: usize,
    /// ProRes and DNx too (their entropy decoding is serial per slice or row: an
    /// integrated GPU is slower at it than the CPU). DXV is fast everywhere.
    entropy: bool,
}

/// The GPU decoders for `ctx`, to register with `brp_media::gpudec::set_factory`.
/// On an integrated GPU only DXV, unless `BRP_GPU_DECODE=all`.
pub fn factory(ctx: &Arc<GpuContext>) -> Arc<dyn GpuDecoders> {
    // At most a quarter of the card's memory for decoded pictures (the rest is for
    // effects, AI and the display). Sources keep only a window around the playhead
    // there; frames farther away stay in system memory compressed (re-decoded here).
    let vram_budget = ((ctx.vram() / 4) as usize).max(512 << 20);
    let discrete = ctx.properties().device_type == vk::PhysicalDeviceType::DISCRETE_GPU;
    let entropy = discrete || std::env::var("BRP_GPU_DECODE").is_ok_and(|v| v == "all");
    Arc::new(Factory { compute: Compute::new(ctx.clone()), vram_budget, entropy })
}

impl GpuDecoders for Factory {
    fn open(&self, s: &StreamDesc) -> Option<Box<dyn GpuDecoder>> {
        let c = self.compute.clone();
        let b = self.vram_budget;
        let d: Box<dyn GpuDecoder> = match s.codec_id {
            codec::DXV => Box::new(dxv::Decoder::new(c.clone(), b, s)),
            codec::PRORES if self.entropy => Box::new(prores::Decoder::new(c.clone(), b)),
            codec::DNXHD if self.entropy => Box::new(dnxhd::Decoder::new(c.clone(), b)?),
            _ => return None,
        };
        Some(Box::new(Paced { d, times: Vec::new(), slow: false, compute: c, free: std::cell::Cell::new(None) }))
    }
}

/// Gives a stream back to FFmpeg when this GPU decodes it too slowly for playback
/// (median of the first frames above `SLOW_MS`).
struct Paced {
    d: Box<dyn GpuDecoder>,
    times: Vec<f64>,
    slow: bool,
    compute: Arc<Compute>,
    /// Free GPU memory as last asked (asked at most every `FREE_EVERY`).
    free: std::cell::Cell<Option<(std::time::Instant, Option<u64>)>>,
}

/// GPU memory left free for effects, AI and other programs before decoded pictures
/// give theirs back.
const VRAM_RESERVE: usize = 1 << 30;
/// Decoded pictures may always keep this much (playback needs a few frames).
const VRAM_FLOOR: usize = 256 << 20;
const FREE_EVERY: std::time::Duration = std::time::Duration::from_millis(250);

const SLOW_MS: f64 = 25.0;
const PACE_FRAMES: usize = 6;

impl GpuDecoder for Paced {
    fn name(&self) -> &str {
        self.d.name()
    }
    fn vram_budget(&self, held: usize) -> usize {
        let cap = self.d.vram_budget(held);
        let now = std::time::Instant::now();
        let free = match self.free.get() {
            Some((t, f)) if now - t < FREE_EVERY => f,
            _ => {
                let f = self.compute.ctx.vram_free();
                self.free.set(Some((now, f)));
                f
            }
        };
        // The system's budget for this process counts other programs' use: when it
        // runs short, the cache shrinks.
        match free {
            Some(free) => cap.min((held + free as usize).saturating_sub(VRAM_RESERVE).max(VRAM_FLOOR)),
            None => cap,
        }
    }
    fn decode(&mut self, data: &[u8]) -> Result<brp_media::gpudec::GpuPicture> {
        anyhow::ensure!(!self.slow, "too slow on this GPU");
        let t = std::time::Instant::now();
        let r = self.d.decode(data);
        if r.is_ok() && self.times.len() < PACE_FRAMES {
            self.times.push(t.elapsed().as_secs_f64() * 1e3);
            if self.times.len() == PACE_FRAMES {
                let mut v = self.times.clone();
                v.sort_by(f64::total_cmp);
                let median = v[PACE_FRAMES / 2];
                if median > SLOW_MS {
                    log::info!("{}: {median:.1} ms per frame on this GPU; FFmpeg decodes this stream", self.d.name());
                    self.slow = true;
                }
            }
        }
        r
    }
}

/// A decoded picture: planes in one GPU buffer.
pub struct Planes {
    pub buf: Buffer,
    offsets: Vec<(u64, u64)>,
    compute: Arc<Compute>,
    bytes: u64,
}

impl Planes {
    /// The device the planes live on (the renderer copies directly only on the same one).
    pub fn device(&self) -> vk::Device {
        self.compute.ctx.device.handle()
    }

    /// Plane `p`: (byte offset, stride in bytes) in `buf`.
    pub fn plane_offsets_of(&self, p: usize) -> (u64, u64) {
        self.offsets[p]
    }
}

impl GpuPlanes for Planes {
    fn plane_offsets(&self) -> &[(u64, u64)] {
        &self.offsets
    }
    fn download(&self) -> Result<Vec<u8>> {
        self.compute.download(&self.buf, 0, self.bytes)
    }
    fn bytes(&self) -> usize {
        self.bytes as usize
    }
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

/// Per-decoder GPU memory reused from packet to packet: the packet (and the
/// decoder's tables for it) go through a host-visible staging buffer into GPU memory.
pub(crate) struct Io {
    pub compute: Arc<Compute>,
    staging: Option<Buffer>,
    input: Option<Buffer>,
    /// Errors reported by kernels (host-visible, one u32).
    err: Buffer,
}

impl Io {
    pub fn new(compute: Arc<Compute>) -> Result<Io> {
        let err = Buffer::new(&compute.ctx, 16, MemoryLocation::GpuToCpu, "decode errors")?;
        Ok(Io { compute, staging: None, input: None, err })
    }

    /// Host memory of at least `bytes` (zeroed tail: bit readers may look past the end).
    pub fn staging(&mut self, bytes: usize) -> Result<&mut [u8]> {
        let need = (bytes as u64 + 64).next_multiple_of(1 << 16);
        if self.staging.as_ref().is_none_or(|b| b.size < need) {
            let size = need + need / 4;
            self.staging = Some(Buffer::new(&self.compute.ctx, size, MemoryLocation::CpuToGpu, "decode staging")?);
            self.input = Some(self.compute.buffer(size, "decode input")?);
        }
        let s = self.staging.as_mut().unwrap().mapped().context("staging not mapped")?;
        Ok(&mut s[..bytes + 64])
    }

    /// The GPU copy of the staging bytes (valid after `upload` in a recording).
    pub fn input(&self) -> &Buffer {
        self.input.as_ref().expect("staging first")
    }

    pub fn err(&self) -> &Buffer {
        &self.err
    }

    /// Record the copy of the first `bytes` of staging into the input buffer, and
    /// clear the error word.
    pub fn upload(&mut self, rec: &mut Recorder, bytes: usize) -> Result<()> {
        self.err.mapped().context("error word not mapped")?[..4].copy_from_slice(&0u32.to_le_bytes());
        let n = (bytes as u64 + 64).next_multiple_of(4);
        rec.copy(self.staging.as_ref().unwrap(), 0, self.input.as_ref().unwrap(), 0, n);
        Ok(())
    }

    /// The error a kernel reported (0: none). Read after the recording ran.
    pub fn error(&mut self) -> u32 {
        self.err.mapped().map_or(0, |m| u32::from_le_bytes(m[..4].try_into().unwrap()))
    }

    /// GPU memory for one picture's planes, each `(stride, rows)` bytes; offsets
    /// aligned for 32-bit access.
    pub fn planes(&self, planes: &[(u64, u64)]) -> Result<Planes> {
        let mut offsets = Vec::with_capacity(planes.len());
        let mut at = 0u64;
        for &(stride, rows) in planes {
            offsets.push((at, stride));
            at = (at + stride * rows).next_multiple_of(256);
        }
        let buf = self.compute.buffer(at, "decoded picture")?;
        Ok(Planes { buf, offsets, compute: self.compute.clone(), bytes: at })
    }
}

/// GLSL shared by the decoders: big-endian bit reading from a byte buffer.
pub(crate) const BITS_GLSL: &str = r#"
uint bswap(uint w) { return (w >> 24) | ((w >> 8) & 0xff00u) | ((w << 8) & 0xff0000u) | (w << 24); }
// 32 bits starting at bit `pos` of the byte stream `s` (MSB first).
uint show32(U32 s, uint pos) {
    uint wi = pos >> 5, sh = pos & 31u;
    uint a = bswap(s.v[wi]);
    if (sh == 0u) return a;
    return (a << sh) | (bswap(s.v[wi + 1u]) >> (32u - sh));
}
// The next `n` bits (1..32) as an unsigned number.
uint showbits(U32 s, uint pos, uint n) { return show32(s, pos) >> (32u - n); }
int s16(int v) { return (v << 16) >> 16; }
"#;
