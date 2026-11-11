//! Decoders that run on the GPU (DXV, ProRes, DNxHD/HR), plugged in by the renderer.
//!
//! The media layer has no GPU of its own: the renderer registers a [`GpuDecoders`]
//! factory once it has a device ([`set_factory`]). A [`Source`](crate::Source) whose
//! stream a GPU decoder accepts sends it the packets instead of FFmpeg; the decoded
//! picture stays in GPU memory ([`GpuPlanes`]) and is copied to the CPU only when
//! something asks for its bytes. Packets the GPU decoder refuses (a profile it does
//! not handle, a damaged slice) go to FFmpeg, so support never shrinks.

use crate::frame::FrameLayout;
use parking_lot::RwLock;
use std::any::Any;
use std::sync::Arc;

/// FFmpeg codec ids of the streams GPU decoders may take.
pub mod codec {
    use ffmpeg_sys_next::AVCodecID as C;
    pub const DXV: i32 = C::AV_CODEC_ID_DXV as i32;
    pub const PRORES: i32 = C::AV_CODEC_ID_PRORES as i32;
    pub const DNXHD: i32 = C::AV_CODEC_ID_DNXHD as i32;
    pub const MPEG2: i32 = C::AV_CODEC_ID_MPEG2VIDEO as i32;
}

/// FFmpeg pixel formats a GPU decoder may produce.
pub mod pix {
    use ffmpeg_sys_next::AVPixelFormat as P;
    pub const RGBA: i32 = P::AV_PIX_FMT_RGBA as i32;
    pub const YUV422P: i32 = P::AV_PIX_FMT_YUV422P as i32;
    pub const YUV444P: i32 = P::AV_PIX_FMT_YUV444P as i32;
    pub const YUV422P10: i32 = P::AV_PIX_FMT_YUV422P10LE as i32;
    pub const YUV444P10: i32 = P::AV_PIX_FMT_YUV444P10LE as i32;
    pub const YUVA422P10: i32 = P::AV_PIX_FMT_YUVA422P10LE as i32;
    pub const YUV422P12: i32 = P::AV_PIX_FMT_YUV422P12LE as i32;
    pub const YUV444P12: i32 = P::AV_PIX_FMT_YUV444P12LE as i32;
    pub const YUVA444P12: i32 = P::AV_PIX_FMT_YUVA444P12LE as i32;
}

/// What a GPU decoder needs to know about a stream.
#[derive(Clone, Debug)]
pub struct StreamDesc {
    /// FFmpeg codec id (`AVCodecID` as an integer).
    pub codec_id: i32,
    /// Container fourcc (e.g. `DXD3`, `apch`), little-endian as FFmpeg stores it.
    pub codec_tag: u32,
    pub width: u32,
    pub height: u32,
    pub extradata: Vec<u8>,
}

/// A decoded picture in GPU memory.
pub trait GpuPlanes: Send + Sync {
    /// Plane `i`: (byte offset, stride in bytes) in the decoder's memory. Planes
    /// hold `FrameLayout` samples (YUV: one sample per texel, u8 or u16 LE; RGBA8:
    /// 4 bytes per pixel), at least `width` x `height` of them.
    fn plane_offsets(&self) -> &[(u64, u64)];
    /// The bytes of all planes (same offsets), read back from the GPU (waits).
    fn download(&self) -> anyhow::Result<Vec<u8>>;
    /// Total bytes the planes take in GPU memory.
    fn bytes(&self) -> usize;
    /// For the renderer to reach its own type (buffer, device).
    fn as_any(&self) -> &dyn Any;
}

/// One decoded picture.
pub struct GpuPicture {
    pub planes: Arc<dyn GpuPlanes>,
    pub layout: FrameLayout,
    /// FFmpeg pixel format of the same picture (for CPU copies and conversions).
    pub pix_fmt: i32,
    pub width: u32,
    pub height: u32,
    pub interlaced: bool,
    pub tff: bool,
}

pub trait GpuDecoder: Send {
    /// Short name for logs and the Info panel (e.g. "ProRes (GPU)").
    fn name(&self) -> &str;
    /// Decode one packet (intra-only codecs: one picture per packet). An error means
    /// this packet is not for the GPU; the caller decodes it with FFmpeg.
    fn decode(&mut self, data: &[u8]) -> anyhow::Result<GpuPicture>;
    /// Most bytes of decoded pictures a source should keep on the GPU, given that it
    /// holds `held` bytes now (less when other programs need the GPU's memory).
    fn vram_budget(&self, held: usize) -> usize;
}

pub trait GpuDecoders: Send + Sync {
    /// A decoder for this stream, or `None` (FFmpeg decodes it).
    fn open(&self, stream: &StreamDesc) -> Option<Box<dyn GpuDecoder>>;
}

static FACTORY: RwLock<Option<Arc<dyn GpuDecoders>>> = RwLock::new(None);

/// Register (or remove) the GPU decoders. Sources opened before this use FFmpeg
/// until their next seek.
pub fn set_factory(f: Option<Arc<dyn GpuDecoders>>) {
    *FACTORY.write() = f;
}

/// The registered factory, unless GPU decoding is off (`BRP_GPU_DECODE=0`).
pub fn factory() -> Option<Arc<dyn GpuDecoders>> {
    if !enabled() {
        return None;
    }
    FACTORY.read().clone()
}

/// GPU decoding is on unless `BRP_GPU_DECODE=0`.
pub fn enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("BRP_GPU_DECODE").map_or(true, |v| v != "0"))
}

/// The stream description and the first `max` video packets of a file (test and
/// benchmark tool: decoders fed without a `Source`).
pub fn read_packets(path: &str, max: usize) -> anyhow::Result<(StreamDesc, Vec<Vec<u8>>)> {
    use crate::ffutil::{Input, Packet};
    use ffmpeg_sys_next as ff;
    let input = Input::open(path)?;
    unsafe {
        let vs = ff::av_find_best_stream(input.ctx, ff::AVMediaType::AVMEDIA_TYPE_VIDEO, -1, -1, std::ptr::null_mut(), 0);
        anyhow::ensure!(vs >= 0, "no video stream");
        let desc = stream_desc(input.stream(vs));
        let pkt = Packet::new();
        let mut out = Vec::new();
        while out.len() < max && ff::av_read_frame(input.ctx, pkt.0) >= 0 {
            let p = &*pkt.0;
            if p.stream_index == vs && !p.data.is_null() {
                out.push(std::slice::from_raw_parts(p.data, p.size as usize).to_vec());
            }
            ff::av_packet_unref(pkt.0);
        }
        Ok((desc, out))
    }
}

/// What GPU decoders are told about a stream.
pub(crate) unsafe fn stream_desc(st: *mut ffmpeg_sys_next::AVStream) -> StreamDesc {
    unsafe {
        let par = &*(*st).codecpar;
        let extradata = if par.extradata.is_null() || par.extradata_size <= 0 {
            Vec::new()
        } else {
            std::slice::from_raw_parts(par.extradata, par.extradata_size as usize).to_vec()
        };
        StreamDesc { codec_id: par.codec_id as i32, codec_tag: par.codec_tag, width: par.width as u32, height: par.height as u32, extradata }
    }
}
