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
    /// Packed RGBA, half float (EXR: linear light, values above 1 kept).
    RgbaF16 { alpha: bool },
}

impl FrameLayout {
    pub fn has_alpha(&self) -> bool {
        match *self {
            FrameLayout::Yuv { alpha, .. } => alpha,
            FrameLayout::Rgba8 { alpha, .. } => alpha.is_some(),
            FrameLayout::Rgba16 => true,
            FrameLayout::RgbaF16 { alpha } => alpha,
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
    /// The picture in system memory (null for a GPU picture: see `cpu`).
    ptr: *mut ff::AVFrame,
    /// Decoded on the GPU: the planes stay there.
    gpu: Option<GpuSide>,
    /// CPU copy of a GPU picture, made the first time its bytes are asked for.
    cpu: std::sync::OnceLock<OwnedFrame>,
    w: u32,
    h: u32,
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
        let (w, h) = (f.width as u32, f.height as u32);
        VideoFrame { ptr, gpu: None, cpu: Default::default(), w, h, index: index.into(), pts, layout, interlaced, tff, bytes }
    }

    /// A picture decoded on the GPU (its planes stay there).
    pub(crate) fn from_gpu(pic: crate::gpudec::GpuPicture, index: i64, pts: i64) -> Self {
        let bytes = pic.planes.bytes();
        VideoFrame {
            ptr: std::ptr::null_mut(),
            gpu: Some(GpuSide { planes: pic.planes, pix_fmt: pic.pix_fmt }),
            cpu: Default::default(),
            w: pic.width,
            h: pic.height,
            index: index.into(),
            pts,
            layout: pic.layout,
            interlaced: pic.interlaced,
            tff: pic.tff,
            bytes,
        }
    }

    /// The GPU planes of a picture decoded on the GPU.
    pub fn gpu_planes(&self) -> Option<&std::sync::Arc<dyn crate::gpudec::GpuPlanes>> {
        self.gpu.as_ref().map(|g| &g.planes)
    }

    /// The picture lives in video memory (decoded on the GPU).
    pub fn on_gpu(&self) -> bool {
        self.gpu.is_some()
    }

    /// The picture as an AVFrame in system memory (a GPU picture is read back the
    /// first time).
    fn av(&self) -> *mut ff::AVFrame {
        if !self.ptr.is_null() {
            return self.ptr;
        }
        self.cpu
            .get_or_init(|| {
                let g = self.gpu.as_ref().expect("frame without data");
                match unsafe { self.download(g) } {
                    Ok(f) => OwnedFrame(f),
                    Err(e) => {
                        log::error!("reading a GPU-decoded frame back: {e:#}");
                        OwnedFrame(unsafe { blank(g.pix_fmt, self.w, self.h) })
                    }
                }
            })
            .0
    }

    unsafe fn download(&self, g: &GpuSide) -> anyhow::Result<*mut ff::AVFrame> {
        unsafe {
            log::debug!("frame {}: GPU picture read back to system memory", self.index());
            let bytes = g.planes.download()?;
            let f = blank(g.pix_fmt, self.w, self.h);
            anyhow::ensure!(!(*f).data[0].is_null(), "out of memory");
            for (i, &(off, stride)) in g.planes.plane_offsets().iter().enumerate().take(self.plane_count()) {
                let (pw, ph) = self.plane_size(i);
                let row = pw as usize * self.bytes_per_sample();
                let dst_stride = (*f).linesize[i] as usize;
                for y in 0..ph as usize {
                    let src = off as usize + y * stride as usize;
                    anyhow::ensure!(src + row <= bytes.len(), "GPU plane {i} too small");
                    std::ptr::copy_nonoverlapping(bytes[src..].as_ptr(), (*f).data[i].add(y * dst_stride), row);
                }
            }
            if self.interlaced {
                (*f).flags |= ff::AV_FRAME_FLAG_INTERLACED;
                if self.tff {
                    (*f).flags |= ff::AV_FRAME_FLAG_TOP_FIELD_FIRST;
                }
            }
            (*f).pts = self.pts;
            (*f).best_effort_timestamp = self.pts;
            Ok(f)
        }
    }

    /// Frame number (presentation order).
    pub fn index(&self) -> i64 {
        self.index.load(std::sync::atomic::Ordering::Relaxed)
    }

    pub(crate) fn set_index(&self, i: i64) {
        self.index.store(i, std::sync::atomic::Ordering::Relaxed)
    }

    pub fn width(&self) -> u32 {
        self.w
    }

    pub fn height(&self) -> u32 {
        self.h
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
            FrameLayout::Rgba16 | FrameLayout::RgbaF16 { .. } => 8,
        }
    }

    /// Raw rows of plane `i`: (data, stride in bytes). Stride is always positive.
    pub fn plane(&self, i: usize) -> (&[u8], usize) {
        unsafe {
            let f = &*self.av();
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

struct GpuSide {
    planes: std::sync::Arc<dyn crate::gpudec::GpuPlanes>,
    pix_fmt: i32,
}

/// An AVFrame freed on drop.
struct OwnedFrame(*mut ff::AVFrame);

unsafe impl Send for OwnedFrame {}
unsafe impl Sync for OwnedFrame {}

impl Drop for OwnedFrame {
    fn drop(&mut self) {
        unsafe { ff::av_frame_free(&mut self.0) };
    }
}

/// A new frame of `pix_fmt` with buffers.
unsafe fn blank(pix_fmt: i32, w: u32, h: u32) -> *mut ff::AVFrame {
    unsafe {
        let f = ff::av_frame_alloc();
        (*f).width = w as i32;
        (*f).height = h as i32;
        (*f).format = pix_fmt;
        if ff::av_frame_get_buffer(f, 0) < 0 {
            log::error!("cannot allocate a {w}x{h} frame");
        }
        f
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

/// A picture as planar f32 RGB(A) in 0..1, in the source's transfer (for AI models).
pub struct PlanarRgb {
    pub w: usize,
    pub h: usize,
    /// 3 (RGB) or 4 (RGBA) planes of `w * h`.
    pub channels: usize,
    pub data: Vec<f32>,
}

impl VideoFrame {
    /// Convert to planar float RGB(A) with the stream's matrix and range.
    pub fn to_planar_rgb(&self, color: &crate::ColorInfo) -> anyhow::Result<PlanarRgb> {
        use crate::Matrix;
        unsafe {
            let src = self.av();
            let s = &*src;
            let alpha = self.layout.has_alpha() && !matches!(self.layout, FrameLayout::Rgba8 { alpha: None, .. });
            let dst_fmt = if alpha { ff::AVPixelFormat::AV_PIX_FMT_GBRAPF32LE } else { ff::AVPixelFormat::AV_PIX_FMT_GBRPF32LE };
            // Built through options to enable slice threading (one 1080p frame: ~11 ms
            // on one thread).
            let ctx = ff::sws_alloc_context();
            anyhow::ensure!(!ctx.is_null(), "out of memory");
            let set = |k: &std::ffi::CStr, v: i64| ff::av_opt_set_int(ctx as *mut _, k.as_ptr(), v, 0);
            set(c"srcw", s.width as i64);
            set(c"srch", s.height as i64);
            set(c"src_format", s.format as i64);
            set(c"dstw", s.width as i64);
            set(c"dsth", s.height as i64);
            set(c"dst_format", dst_fmt as i64);
            set(c"sws_flags", (ff::SWS_BICUBIC | ff::SWS_ACCURATE_RND | ff::SWS_FULL_CHR_H_INT) as i64);
            set(c"threads", std::thread::available_parallelism().map_or(4, |n| n.get()).min(16) as i64);
            if ff::sws_init_context(ctx, std::ptr::null_mut(), std::ptr::null_mut()) < 0 {
                ff::sws_freeContext(ctx);
                anyhow::bail!("unsupported pixel format for RGB conversion");
            }
            let cs = match color.matrix {
                Matrix::Bt601 => ff::SWS_CS_ITU601,
                Matrix::Bt2020Ncl => ff::SWS_CS_BT2020,
                Matrix::Smpte240 => ff::SWS_CS_SMPTE240M,
                _ => ff::SWS_CS_ITU709,
            };
            let table = ff::sws_getCoefficients(cs);
            ff::sws_setColorspaceDetails(ctx, table, color.full_range as i32, table, 1, 0, 1 << 16, 1 << 16);
            let dst = ff::av_frame_alloc();
            (*dst).width = s.width;
            (*dst).height = s.height;
            (*dst).format = dst_fmt as i32;
            let r = ff::av_frame_get_buffer(dst, 0);
            let r = if r >= 0 { ff::sws_scale_frame(ctx, dst, src) } else { r };
            ff::sws_freeContext(ctx);
            let mut d = dst;
            if r < 0 {
                ff::av_frame_free(&mut d);
                ffutil::check(r, "convert to RGB")?;
            }
            let (w, h) = (s.width as usize, s.height as usize);
            let channels = if alpha { 4 } else { 3 };
            let mut data = vec![0f32; w * h * channels];
            // GBR(A) plane order -> R, G, B(, A).
            for (out_c, plane) in [(0usize, 2usize), (1, 0), (2, 1), (3, 3)].into_iter().take(channels) {
                let f = &*dst;
                let stride = f.linesize[plane] as usize;
                for y in 0..h {
                    let row = std::slice::from_raw_parts(f.data[plane].add(y * stride) as *const f32, w);
                    data[out_c * w * h + y * w..][..w].copy_from_slice(row);
                }
            }
            ff::av_frame_free(&mut d);
            Ok(PlanarRgb { w, h, channels, data })
        }
    }

    /// A half-float RGBA frame (layout `RgbaF16`, an EXR layer): `data` is R, G, B, A
    /// per pixel, as f16 bits.
    pub fn from_rgba_f16(w: usize, h: usize, data: &[u16], alpha: bool, index: i64, pts: i64) -> anyhow::Result<VideoFrame> {
        anyhow::ensure!(data.len() >= w * h * 4, "RGBA image too small");
        unsafe {
            let f = ff::av_frame_alloc();
            (*f).width = w as i32;
            (*f).height = h as i32;
            (*f).format = ff::AVPixelFormat::AV_PIX_FMT_RGBAF16LE as i32;
            let r = ff::av_frame_get_buffer(f, 0);
            if r < 0 {
                let mut d = f;
                ff::av_frame_free(&mut d);
                ffutil::check(r, "alloc RGBA half-float frame")?;
            }
            let stride = (*f).linesize[0] as usize;
            for y in 0..h {
                let row = std::slice::from_raw_parts_mut((*f).data[0].add(y * stride) as *mut u16, w * 4);
                row.copy_from_slice(&data[y * w * 4..(y + 1) * w * 4]);
            }
            Ok(VideoFrame::new(f, index, pts, FrameLayout::RgbaF16 { alpha }))
        }
    }

    /// A packed RGBA16 frame (layout `Rgba16`) from planar f32 RGB(A); alpha 1 if absent.
    /// `index` is the frame's identity for caches (AI results use their own numbering).
    pub fn from_planar_rgb(img: &PlanarRgb, index: i64, pts: i64) -> anyhow::Result<VideoFrame> {
        Self::from_planar_f32(img.w, img.h, img.channels, &img.data, index, pts)
    }

    /// Like [`from_planar_rgb`], from borrowed planes (`channels` planes of `w * h`).
    pub fn from_planar_f32(w: usize, h: usize, channels: usize, data: &[f32], index: i64, pts: i64) -> anyhow::Result<VideoFrame> {
        anyhow::ensure!(data.len() >= w * h * channels, "planar image too small");
        unsafe {
            let f = ff::av_frame_alloc();
            (*f).width = w as i32;
            (*f).height = h as i32;
            (*f).format = ff::AVPixelFormat::AV_PIX_FMT_RGBA64LE as i32;
            let r = ff::av_frame_get_buffer(f, 0);
            if r < 0 {
                let mut d = f;
                ff::av_frame_free(&mut d);
                ffutil::check(r, "alloc RGBA frame")?;
            }
            let stride = (*f).linesize[0] as usize;
            let n = w * h;
            let q = |v: f32| (v.clamp(0.0, 1.0) * 65535.0 + 0.5) as u16;
            // Rows in parallel (one 1080p frame: ~10 ms on one thread).
            let out = std::slice::from_raw_parts_mut((*f).data[0], stride * h);
            let threads = std::thread::available_parallelism().map_or(4, |n| n.get()).min(16);
            let rows_per = h.div_ceil(threads).max(1);
            std::thread::scope(|sc| {
                for (k, chunk) in out.chunks_mut(stride * rows_per).enumerate() {
                    sc.spawn(move || {
                        for (r, line) in chunk.chunks_mut(stride).enumerate() {
                            let y = k * rows_per + r;
                            let row = std::slice::from_raw_parts_mut(line.as_mut_ptr() as *mut u16, w * 4);
                            for x in 0..w {
                                let i = y * w + x;
                                row[x * 4] = q(data[i]);
                                row[x * 4 + 1] = q(data[n + i]);
                                row[x * 4 + 2] = q(data[2 * n + i]);
                                row[x * 4 + 3] = if channels > 3 { q(data[3 * n + i]) } else { 65535 };
                            }
                        }
                    });
                }
            });
            Ok(VideoFrame::new(f, index, pts, FrameLayout::Rgba16))
        }
    }

}
