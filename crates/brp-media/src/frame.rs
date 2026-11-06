//! Decoded video frames, kept as reference-counted AVFrames (no copies).

use crate::ffutil;
use ffmpeg_sys_next as ff;

/// How the planes of a frame map to the GPU.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FrameLayout {
    /// One plane per component (Y, U, V[, A]). `bits` is the coded depth; samples wider
    /// than 8 bits are stored as little-endian u16 in the low bits.
    Yuv { bits: u8, chroma_w_shift: u8, chroma_h_shift: u8, alpha: bool },
    /// Packed 4-byte RGB(A). `offsets` gives the byte offset of R, G, B, A in a pixel;
    /// A is `None` for formats with padding instead of alpha.
    Rgba8 { offsets: [u8; 3], alpha: Option<u8> },
    /// Packed RGBA, 16 bits per channel (used for converted frames).
    Rgba16,
}

impl FrameLayout {
    pub fn has_alpha(&self) -> bool {
        match *self {
            FrameLayout::Yuv { alpha, .. } => alpha,
            FrameLayout::Rgba8 { alpha, .. } => alpha.is_some(),
            FrameLayout::Rgba16 => true,
        }
    }

    pub fn is_rgb(&self) -> bool {
        !matches!(self, FrameLayout::Yuv { .. })
    }
}

/// Map an FFmpeg pixel format to a layout the renderer can upload directly.
/// `None` means the frame has to be converted first.
pub fn classify(fmt: i32) -> Option<FrameLayout> {
    unsafe {
        let desc = ff::av_pix_fmt_desc_get(std::mem::transmute::<i32, ff::AVPixelFormat>(fmt));
        if desc.is_null() {
            return None;
        }
        let d = &*desc;
        let flags = d.flags;
        const BAD: u64 = (ff::AV_PIX_FMT_FLAG_BE | ff::AV_PIX_FMT_FLAG_HWACCEL | ff::AV_PIX_FMT_FLAG_PAL | ff::AV_PIX_FMT_FLAG_BITSTREAM) as u64;
        if flags & BAD != 0 {
            return None;
        }
        let rgb = flags & ff::AV_PIX_FMT_FLAG_RGB as u64 != 0;
        let planar = flags & ff::AV_PIX_FMT_FLAG_PLANAR as u64 != 0;
        let n = d.nb_components as usize;

        if !rgb && planar && (n == 3 || n == 4) {
            let bits = d.comp[0].depth as u8;
            if bits == 0 || bits > 16 {
                return None;
            }
            let step = if bits > 8 { 2 } else { 1 };
            for i in 0..n {
                let c = &d.comp[i];
                if c.plane as usize != i || c.step != step || c.shift != 0 || c.offset != 0 || c.depth as u8 != bits {
                    return None;
                }
            }
            return Some(FrameLayout::Yuv {
                bits,
                chroma_w_shift: d.log2_chroma_w,
                chroma_h_shift: d.log2_chroma_h,
                alpha: n == 4,
            });
        }

        if rgb && !planar && (n == 3 || n == 4) {
            let c = &d.comp;
            if (0..n).all(|i| c[i].depth == 8 && c[i].step == 4 && c[i].plane == 0 && c[i].shift == 0) {
                return Some(FrameLayout::Rgba8 {
                    offsets: [c[0].offset as u8, c[1].offset as u8, c[2].offset as u8],
                    alpha: if n == 4 { Some(c[3].offset as u8) } else { None },
                });
            }
        }
        None
    }
}

pub struct VideoFrame {
    ptr: *mut ff::AVFrame,
    index: std::sync::atomic::AtomicI64,
    pub pts: i64,
    pub layout: FrameLayout,
    pub interlaced: bool,
    /// Top field first (from the frame flags; meaningful when interlaced).
    pub tff: bool,
    bytes: usize,
}

unsafe impl Send for VideoFrame {}
unsafe impl Sync for VideoFrame {}

impl VideoFrame {
    /// Takes ownership of `ptr`.
    pub(crate) unsafe fn new(ptr: *mut ff::AVFrame, index: i64, pts: i64, layout: FrameLayout) -> Self {
        let f = unsafe { &*ptr };
        let planes = match layout {
            FrameLayout::Yuv { alpha, .. } => 3 + alpha as usize,
            _ => 1,
        };
        let bytes = (0..planes).map(|i| f.linesize[i].unsigned_abs() as usize * f.height as usize).sum();
        let interlaced = f.flags & ff::AV_FRAME_FLAG_INTERLACED != 0;
        let tff = f.flags & ff::AV_FRAME_FLAG_TOP_FIELD_FIRST != 0;
        VideoFrame { ptr, index: index.into(), pts, layout, interlaced, tff, bytes }
    }

    /// Frame number (presentation order).
    pub fn index(&self) -> i64 {
        self.index.load(std::sync::atomic::Ordering::Relaxed)
    }

    pub(crate) fn set_index(&self, i: i64) {
        self.index.store(i, std::sync::atomic::Ordering::Relaxed)
    }

    pub fn width(&self) -> u32 {
        unsafe { (*self.ptr).width as u32 }
    }

    pub fn height(&self) -> u32 {
        unsafe { (*self.ptr).height as u32 }
    }

    pub fn byte_size(&self) -> usize {
        self.bytes
    }

    pub fn plane_count(&self) -> usize {
        match self.layout {
            FrameLayout::Yuv { alpha, .. } => 3 + alpha as usize,
            _ => 1,
        }
    }

    /// (width, height) in samples of plane `i`.
    pub fn plane_size(&self, i: usize) -> (u32, u32) {
        let (w, h) = (self.width(), self.height());
        match self.layout {
            FrameLayout::Yuv { chroma_w_shift, chroma_h_shift, .. } if i == 1 || i == 2 => (
                (w + (1 << chroma_w_shift) - 1) >> chroma_w_shift,
                (h + (1 << chroma_h_shift) - 1) >> chroma_h_shift,
            ),
            _ => (w, h),
        }
    }

    /// Bytes per sample of plane `i` (1 or 2 for YUV, 4 or 8 for packed RGBA).
    pub fn bytes_per_sample(&self) -> usize {
        match self.layout {
            FrameLayout::Yuv { bits, .. } => if bits > 8 { 2 } else { 1 },
            FrameLayout::Rgba8 { .. } => 4,
            FrameLayout::Rgba16 => 8,
        }
    }

    /// Raw rows of plane `i`: (data, stride in bytes). Stride is always positive.
    pub fn plane(&self, i: usize) -> (&[u8], usize) {
        unsafe {
            let f = &*self.ptr;
            let stride = f.linesize[i] as usize;
            let (_, h) = self.plane_size(i);
            (std::slice::from_raw_parts(f.data[i], stride * h as usize), stride)
        }
    }
}

impl Drop for VideoFrame {
    fn drop(&mut self) {
        unsafe { ff::av_frame_free(&mut self.ptr) };
    }
}

/// Converts frames whose pixel format the renderer cannot take directly.
pub(crate) struct Converter {
    ctx: *mut ff::SwsContext,
    key: (i32, i32, i32),
}

unsafe impl Send for Converter {}

impl Converter {
    pub fn new() -> Self {
        Converter { ctx: std::ptr::null_mut(), key: (0, 0, 0) }
    }

    /// Returns a new RGBA64 frame (caller owns it). Copies timestamps and flags.
    pub unsafe fn to_rgba16(&mut self, src: *mut ff::AVFrame) -> anyhow::Result<*mut ff::AVFrame> {
        unsafe {
            let s = &*src;
            let key = (s.width, s.height, s.format);
            if self.ctx.is_null() || self.key != key {
                ff::sws_freeContext(self.ctx);
                self.ctx = ff::sws_getContext(
                    s.width,
                    s.height,
                    std::mem::transmute::<i32, ff::AVPixelFormat>(s.format),
                    s.width,
                    s.height,
                    ff::AVPixelFormat::AV_PIX_FMT_RGBA64LE,
                    ff::SWS_BILINEAR,
                    std::ptr::null_mut(),
                    std::ptr::null_mut(),
                    std::ptr::null(),
                );
                if self.ctx.is_null() {
                    anyhow::bail!("unsupported pixel format for conversion");
                }
                self.key = key;
            }
            let dst = ff::av_frame_alloc();
            (*dst).width = s.width;
            (*dst).height = s.height;
            (*dst).format = ff::AVPixelFormat::AV_PIX_FMT_RGBA64LE as i32;
            ffutil::check(ff::av_frame_get_buffer(dst, 0), "alloc rgba frame")?;
            ff::av_frame_copy_props(dst, src);
            let r = ff::sws_scale_frame(self.ctx, dst, src);
            if r < 0 {
                let mut d = dst;
                ff::av_frame_free(&mut d);
                ffutil::check(r, "convert frame")?;
            }
            Ok(dst)
        }
    }
}

impl Drop for Converter {
    fn drop(&mut self) {
        unsafe { ff::sws_freeContext(self.ctx) };
    }
}
