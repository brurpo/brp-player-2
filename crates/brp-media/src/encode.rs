//! Encoding and muxing (libavcodec / libavformat / libavfilter): the export side of
//! the media layer.
//!
//! The GPU writes each picture straight into the encoder's pixel format when it can
//! ([`pack_layout`] describes the planes it has to write). Formats it cannot write
//! (big-endian, bit-packed, palette, hardware-only) are written in a wide format and
//! converted by swscale, so every encoder FFmpeg has stays usable. Audio is read
//! from the source file, cut to the exported range, retimed when the export changes
//! speed, and encoded alongside.

use crate::external::{self, External, Msg, Pkt};
use crate::ffutil::{self, cstr, Input, Packet};
use crate::info::{ColorInfo, Matrix, Primaries, Transfer};
use anyhow::{bail, Context as _, Result};
use ffmpeg_sys_next as ff;
use std::ffi::CStr;

// ---- pixel formats the GPU writes ------------------------------------------------

/// One plane of a packed picture.
#[derive(Clone, Debug)]
pub struct PackPlane {
    /// Size in samples (pixels of this plane).
    pub w: u32,
    pub h: u32,
    /// Bytes per row: a multiple of 4.
    pub stride: u32,
    /// Byte offset of the plane in the picture buffer.
    pub offset: u64,
    /// Bytes per sample: 1, 2 or 4.
    pub bytes: u8,
    /// Samples per pixel in this plane (packed formats: 3 or 4).
    pub step: u8,
    /// Component of each sample of a pixel: 0..3 = R/Y, G/U, B/V, A; -1 = padding.
    pub comps: [i8; 4],
    /// Left shift of the value inside its sample (P010: 6).
    pub shift: u8,
    /// Subsampled chroma plane.
    pub chroma: bool,
}

/// How a picture in some pixel format lies in memory, for the GPU packer.
#[derive(Clone, Debug)]
pub struct PackLayout {
    pub pix_fmt: i32,
    pub name: String,
    /// Y'CbCr (else RGB or gray-as-RGB).
    pub yuv: bool,
    /// Float samples (32-bit, or 16-bit half floats).
    pub float: bool,
    /// Bits of the samples (8, 10, 12, 16; 32 for float).
    pub bits: u8,
    /// log2 chroma subsampling.
    pub cw: u8,
    pub ch: u8,
    pub alpha: bool,
    pub w: u32,
    pub h: u32,
    pub planes: Vec<PackPlane>,
    /// Bytes of the whole picture.
    pub total: u64,
}

fn pix_fmt_of(name: &str) -> i32 {
    let c = cstr(name);
    unsafe { ff::av_get_pix_fmt(c.as_ptr()) as i32 }
}

pub fn pix_fmt_name(fmt: i32) -> String {
    unsafe { ffutil::opt_str(ff::av_get_pix_fmt_name(std::mem::transmute::<i32, ff::AVPixelFormat>(fmt))).unwrap_or_default() }
}

/// Planes the GPU writes for `pix_fmt` at `w` x `h`, or None when it cannot write
/// that format directly (it is then converted from a wide format on the CPU).
pub fn pack_layout(pix_fmt: &str, w: u32, h: u32) -> Option<PackLayout> {
    pack_layout_of(pix_fmt_of(pix_fmt), w, h)
}

pub fn pack_layout_of(fmt: i32, w: u32, h: u32) -> Option<PackLayout> {
    if fmt < 0 {
        return None;
    }
    unsafe {
        let desc = ff::av_pix_fmt_desc_get(std::mem::transmute::<i32, ff::AVPixelFormat>(fmt));
        if desc.is_null() {
            return None;
        }
        let d = &*desc;
        let flags = d.flags;
        let bad = (ff::AV_PIX_FMT_FLAG_BE | ff::AV_PIX_FMT_FLAG_HWACCEL | ff::AV_PIX_FMT_FLAG_PAL | ff::AV_PIX_FMT_FLAG_BITSTREAM | ff::AV_PIX_FMT_FLAG_BAYER | ff::AV_PIX_FMT_FLAG_XYZ) as u64;
        if flags & bad != 0 {
            return None;
        }
        let rgb = flags & ff::AV_PIX_FMT_FLAG_RGB as u64 != 0;
        let float = flags & ff::AV_PIX_FMT_FLAG_FLOAT as u64 != 0;
        let alpha = flags & ff::AV_PIX_FMT_FLAG_ALPHA as u64 != 0;
        let n = d.nb_components as usize;
        if n == 0 {
            return None;
        }
        // Gray formats: one component (two with alpha), written as luma.
        let gray = n <= 2;
        let mut planes: Vec<Option<PackPlane>> = vec![None; 4];
        for i in 0..n {
            let c = &d.comp[i];
            let comp = if gray && i == 1 { 3 } else { i as i8 };
            let depth = c.depth as u32;
            let sb: u32 = if float {
                match depth {
                    16 => 2,
                    32 => 4,
                    _ => return None,
                }
            } else if depth + c.shift as u32 <= 8 {
                1
            } else if depth + c.shift as u32 <= 16 {
                2
            } else {
                return None;
            };
            let (step, off) = (c.step as u32, c.offset as u32);
            if step == 0 || step % sb != 0 || off % sb != 0 || step / sb > 4 {
                return None;
            }
            let p = c.plane as usize;
            if p > 3 {
                return None;
            }
            let slot = (off / sb) as usize;
            let chroma = !rgb && !gray && (i == 1 || i == 2);
            let e = planes[p].get_or_insert(PackPlane { w: 0, h: 0, stride: 0, offset: 0, bytes: sb as u8, step: (step / sb) as u8, comps: [-1; 4], shift: c.shift as u8, chroma });
            if e.bytes as u32 != sb || e.step as u32 != step / sb || e.shift != c.shift as u8 || slot >= 4 || e.comps[slot] != -1 {
                return None;
            }
            e.comps[slot] = comp;
            e.chroma |= chroma;
        }
        let (cw, ch) = (d.log2_chroma_w, d.log2_chroma_h);
        let mut out = Vec::new();
        let mut total = 0u64;
        for p in planes.into_iter() {
            let Some(mut p) = p else { break };
            let (pw, ph) = if p.chroma { ((w + (1 << cw) - 1) >> cw, (h + (1 << ch) - 1) >> ch) } else { (w, h) };
            p.w = pw;
            p.h = ph;
            p.stride = (pw * p.step as u32 * p.bytes as u32).div_ceil(4) * 4;
            p.offset = total;
            total += p.stride as u64 * ph as u64;
            out.push(p);
        }
        Some(PackLayout {
            pix_fmt: fmt,
            name: pix_fmt_name(fmt),
            yuv: !rgb,
            float,
            bits: d.comp[0].depth as u8,
            cw,
            ch,
            alpha,
            w,
            h,
            planes: out,
            total,
        })
    }
}

/// What the GPU writes for an encoder taking `pix_fmt` (through a filter taking
/// `filter_fmt`, if any): the format itself, or a wide one converted by swscale.
pub fn delivery_layout(pix_fmt: &str, filter_fmt: Option<&str>, w: u32, h: u32) -> Option<PackLayout> {
    let fmt = filter_fmt.unwrap_or(pix_fmt);
    if let Some(l) = pack_layout(fmt, w, h) {
        return Some(l);
    }
    let i = pix_fmt_info(fmt)?;
    let wide = if i.float {
        if i.alpha { "gbrapf32le" } else { "gbrpf32le" }
    } else if i.rgb {
        if i.alpha { "rgba64le" } else { "rgb48le" }
    } else if i.alpha {
        "yuva444p16le"
    } else {
        "yuv444p16le"
    };
    pack_layout(wide, w, h)
}

/// Pixel format details used to pick formats (alpha, bit depth, subsampling).
#[derive(Clone, Debug)]
pub struct PixFmtInfo {
    pub name: String,
    pub alpha: bool,
    pub rgb: bool,
    pub float: bool,
    pub bits: u8,
    pub cw: u8,
    pub ch: u8,
    pub hw: bool,
}

pub fn pix_fmt_info(name: &str) -> Option<PixFmtInfo> {
    let fmt = pix_fmt_of(name);
    if fmt < 0 {
        return None;
    }
    unsafe {
        let d = ff::av_pix_fmt_desc_get(std::mem::transmute::<i32, ff::AVPixelFormat>(fmt));
        if d.is_null() {
            return None;
        }
        let d = &*d;
        let f = d.flags;
        Some(PixFmtInfo {
            name: name.to_string(),
            alpha: f & ff::AV_PIX_FMT_FLAG_ALPHA as u64 != 0,
            rgb: f & ff::AV_PIX_FMT_FLAG_RGB as u64 != 0,
            float: f & ff::AV_PIX_FMT_FLAG_FLOAT as u64 != 0,
            bits: d.comp[0].depth as u8,
            cw: d.log2_chroma_w,
            ch: d.log2_chroma_h,
            hw: f & ff::AV_PIX_FMT_FLAG_HWACCEL as u64 != 0,
        })
    }
}

// ---- encoders ----------------------------------------------------------------------

#[derive(Clone, Debug)]
pub struct EncoderInfo {
    pub name: String,
    pub long_name: String,
    /// Codec it produces (e.g. "h264").
    pub codec: String,
    pub pix_fmts: Vec<String>,
    /// Runs on a GPU / media engine (NVENC, AMF, QSV, VAAPI, Vulkan, MediaFoundation...).
    pub hardware: bool,
}

unsafe fn supported<T: Copy>(codec: *const ff::AVCodec, cfg: ff::AVCodecConfig, end: impl Fn(&T) -> bool) -> Vec<T> {
    let mut out: *const std::ffi::c_void = std::ptr::null();
    let mut n: i32 = 0;
    unsafe {
        if ff::avcodec_get_supported_config(std::ptr::null(), codec, cfg, 0, &mut out, &mut n) < 0 || out.is_null() {
            return Vec::new();
        }
        let p = out as *const T;
        (0..n.max(0) as usize).map(|i| *p.add(i)).take_while(|v| !end(v)).collect()
    }
}

fn is_hw_name(name: &str) -> bool {
    ["_nvenc", "_amf", "_qsv", "_vaapi", "_vulkan", "_mf", "_d3d12va", "_videotoolbox", "_v4l2m2m", "_mediacodec"].iter().any(|s| name.ends_with(s))
}

unsafe fn info_of(c: *const ff::AVCodec) -> EncoderInfo {
    unsafe {
        let pix: Vec<ff::AVPixelFormat> = supported(c, ff::AVCodecConfig::AV_CODEC_CONFIG_PIX_FORMAT, |p: &ff::AVPixelFormat| *p as i32 == -1);
        let name = ffutil::opt_str((*c).name).unwrap_or_default();
        EncoderInfo {
            hardware: (*c).capabilities as u32 & ff::AV_CODEC_CAP_HARDWARE != 0 || is_hw_name(&name),
            long_name: ffutil::opt_str((*c).long_name).unwrap_or_default(),
            codec: ffutil::opt_str(ff::avcodec_get_name((*c).id)).unwrap_or_default(),
            pix_fmts: pix.into_iter().map(|p| pix_fmt_name(p as i32)).collect(),
            name,
        }
    }
}

/// Every video encoder of the FFmpeg we run with.
pub fn video_encoders() -> Vec<EncoderInfo> {
    let mut out = Vec::new();
    let mut it: *mut std::ffi::c_void = std::ptr::null_mut();
    unsafe {
        loop {
            let c = ff::av_codec_iterate(&mut it);
            if c.is_null() {
                break;
            }
            if ff::av_codec_is_encoder(c) != 0 && (*c).type_ == ff::AVMediaType::AVMEDIA_TYPE_VIDEO {
                out.push(info_of(c));
            }
        }
    }
    out.extend(external::ENCODERS.iter().filter(|n| external::handles(n)).filter_map(|n| external_info(n)));
    out
}

/// An encoder run by the external GPL `ffmpeg.exe` (x264, x265).
fn external_info(name: &str) -> Option<EncoderInfo> {
    let (codec, pix) = external::info(name)?;
    Some(EncoderInfo {
        name: name.to_string(),
        long_name: format!("{name} (external ffmpeg.exe)"),
        codec: codec.to_string(),
        pix_fmts: pix.iter().map(|s| s.to_string()).collect(),
        hardware: false,
    })
}

pub fn encoder(name: &str) -> Option<EncoderInfo> {
    let n = cstr(name);
    unsafe {
        let c = ff::avcodec_find_encoder_by_name(n.as_ptr());
        if !c.is_null() {
            Some(info_of(c))
        } else if external::handles(name) {
            external_info(name)
        } else {
            None
        }
    }
}

/// Audio encoders' names (for the audio codec list).
pub fn audio_encoder_exists(name: &str) -> bool {
    let n = cstr(name);
    unsafe {
        let c = ff::avcodec_find_encoder_by_name(n.as_ptr());
        !c.is_null() && (*c).type_ == ff::AVMediaType::AVMEDIA_TYPE_AUDIO
    }
}

/// Named values of an encoder's private option (e.g. the `dxv` encoder's `format`:
/// "dxt1", and "dxt5" when FFmpeg was built with the alpha patch).
pub fn encoder_option_values(encoder: &str, option: &str) -> Vec<String> {
    let n = cstr(encoder);
    let o = cstr(option);
    let mut out = Vec::new();
    unsafe {
        let c = ff::avcodec_find_encoder_by_name(n.as_ptr());
        if c.is_null() || (*c).priv_class.is_null() {
            return out;
        }
        let class = &(*c).priv_class as *const *const ff::AVClass as *mut std::ffi::c_void;
        let opt = ff::av_opt_find(class, o.as_ptr(), std::ptr::null(), 0, ff::AV_OPT_SEARCH_FAKE_OBJ as i32);
        if opt.is_null() || (*opt).unit.is_null() {
            return out;
        }
        let unit = CStr::from_ptr((*opt).unit).to_owned();
        let mut prev: *const ff::AVOption = std::ptr::null();
        loop {
            prev = ff::av_opt_next(class, prev);
            if prev.is_null() {
                break;
            }
            let p = &*prev;
            if p.type_ == ff::AVOptionType::AV_OPT_TYPE_CONST && !p.unit.is_null() && CStr::from_ptr(p.unit) == unit.as_c_str() {
                if let Some(s) = ffutil::opt_str(p.name) {
                    out.push(s);
                }
            }
        }
    }
    out
}

/// The encoder has a private option of that name.
pub fn encoder_has_option(encoder: &str, option: &str) -> bool {
    let n = cstr(encoder);
    let o = cstr(option);
    unsafe {
        let c = ff::avcodec_find_encoder_by_name(n.as_ptr());
        if c.is_null() || (*c).priv_class.is_null() {
            return false;
        }
        let class = &(*c).priv_class as *const *const ff::AVClass as *mut std::ffi::c_void;
        !ff::av_opt_find(class, o.as_ptr(), std::ptr::null(), 0, ff::AV_OPT_SEARCH_FAKE_OBJ as i32).is_null()
    }
}

/// Whether the encoder really opens here (hardware encoders need the right GPU and
/// driver): opens it on a small picture and closes it.
pub fn probe_encoder(name: &str, pix_fmt: &str) -> Result<()> {
    if external::handles(name) {
        return Ok(());
    }
    let n = cstr(name);
    unsafe {
        let c = ff::avcodec_find_encoder_by_name(n.as_ptr());
        if c.is_null() {
            bail!("no encoder {name}");
        }
        let mut ctx = ff::avcodec_alloc_context3(c);
        (*ctx).width = 640;
        (*ctx).height = 480;
        (*ctx).pix_fmt = std::mem::transmute::<i32, ff::AVPixelFormat>(pix_fmt_of(pix_fmt));
        (*ctx).time_base = ff::AVRational { num: 1, den: 25 };
        (*ctx).framerate = ff::AVRational { num: 25, den: 1 };
        (*ctx).bit_rate = 2_000_000;
        let r = ff::avcodec_open2(ctx, c, std::ptr::null_mut());
        ff::avcodec_free_context(&mut ctx);
        ffutil::check(r, name).map(|_| ())
    }
}

/// Silence FFmpeg's own messages (true) or back to errors only.
pub fn quiet_log(quiet: bool) {
    unsafe { ff::av_log_set_level(if quiet { ff::AV_LOG_QUIET } else { ff::AV_LOG_ERROR }) };
}

/// The muxer can hold this encoder's codec (None: unknown).
pub fn muxer_accepts(muxer: &str, encoder: &str) -> Option<bool> {
    let m = cstr(muxer);
    let e = cstr(encoder);
    unsafe {
        let of = ff::av_guess_format(m.as_ptr(), std::ptr::null(), std::ptr::null());
        let c = ff::avcodec_find_encoder_by_name(e.as_ptr());
        let id = if !c.is_null() {
            (*c).id
        } else if let Some((codec, _)) = external::info(encoder).filter(|_| external::handles(encoder)) {
            let d = ff::avcodec_descriptor_get_by_name(cstr(codec).as_ptr());
            if d.is_null() {
                return None;
            }
            (*d).id
        } else {
            return None;
        };
        if of.is_null() {
            return None;
        }
        match ff::avformat_query_codec(of, id, ff::FF_COMPLIANCE_NORMAL as i32) {
            1 => Some(true),
            0 => Some(false),
            _ => None,
        }
    }
}

// ---- writer -----------------------------------------------------------------------

#[derive(Clone, Debug, PartialEq)]
pub enum RateControl {
    /// The encoder's defaults (or a profile that fixes the rate, e.g. DNxHR).
    Default,
    /// Constant quality: CRF / CQ / QP / global quality, in the encoder's own scale.
    Quality(f32),
    /// Average bit rate in kbit/s.
    Bitrate(u32),
    /// Constant bit rate in kbit/s (broadcast formats).
    Cbr(u32),
}

#[derive(Clone, Debug)]
pub struct VideoConfig {
    pub encoder: String,
    /// Pixel format the encoder gets.
    pub pix_fmt: String,
    pub width: u32,
    pub height: u32,
    /// Frames per second as num/den.
    pub rate: (i32, i32),
    /// Sample aspect ratio (1:1 for square pixels).
    pub sar: (i32, i32),
    pub rate_control: RateControl,
    /// Codec options as on the FFmpeg command line (`profile`, `preset`, `format`...).
    pub options: Vec<(String, String)>,
    pub gop: Option<i32>,
    pub color: Option<ColorInfo>,
    /// Clockwise rotation players should apply (display matrix).
    pub rotation: u32,
    /// A libavfilter graph between the delivered pictures and the encoder, with the
    /// pixel format it takes (GIF: a palette per picture).
    pub filter: Option<(String, String)>,
    /// Interlaced: Some(top field first). Field order and the codec's interlaced
    /// coding are set, and every frame is flagged.
    pub field_order: Option<bool>,
    /// The pictures arrive already encoded (our GPU encoders): FFmpeg only muxes
    /// them, `encoder` is not opened and [`Writer::write`] takes finished packets.
    pub precoded: Option<Precoded>,
}

/// A stream whose packets are encoded elsewhere (by our GPU encoders): what the
/// container needs to know about it.
#[derive(Clone, Debug, Default)]
pub struct Precoded {
    /// FFmpeg codec name ("prores", "dnxhd", "dxv", "mpeg2video").
    pub codec: String,
    /// Four-character code (MOV sample entry), e.g. "apch".
    pub tag: Option<[u8; 4]>,
    /// AVCodecParameters.profile (-99 unknown).
    pub profile: i32,
    pub bits_per_coded_sample: i32,
    pub bits_per_raw_sample: i32,
    /// Bits per second (MXF and some players read it).
    pub bit_rate: i64,
    /// Every packet is a key frame (intra-only codecs).
    pub intra_only: bool,
}

#[derive(Clone, Debug)]
pub struct AudioConfig {
    pub encoder: String,
    pub kbps: u32,
    /// 0: the source's rate (or the nearest the encoder takes).
    pub sample_rate: u32,
}

/// The stretch of a file's audio to export.
#[derive(Clone, Debug)]
pub struct AudioSource {
    pub path: String,
    /// Absolute stream times (seconds) of the range.
    pub start: f64,
    pub end: f64,
    /// Playback speed of the export (2 = twice as fast, 0.5 = slow motion).
    pub tempo: f64,
    /// Keep the pitch when the speed changes (else it shifts, like tape).
    pub keep_pitch: bool,
    /// Length of the export (seconds): the audio is cut or padded to it.
    pub length: f64,
}

/// What a muxer is called and how its files are named.
pub fn is_image_sequence(muxer: &str) -> bool {
    muxer == "image2"
}

pub struct Writer {
    oc: *mut ff::AVFormatContext,
    venc: *mut ff::AVCodecContext,
    vst: i32,
    /// Picture as delivered (the GPU's layout).
    layout: PackLayout,
    deliver: *mut ff::AVFrame,
    /// Conversion to the encoder's format when the GPU cannot write it.
    sws: *mut ff::SwsContext,
    converted: *mut ff::AVFrame,
    vfilter: Option<Graph>,
    pkt: Packet,
    next_pts: i64,
    rate: (i32, i32),
    field_order: Option<bool>,
    audio: Option<AudioPipe>,
    header: bool,
    finished: bool,
    /// Packets come encoded (no encoder of ours is open).
    precoded: Option<Precoded>,
    /// x264/x265 in the external GPL ffmpeg.exe: pictures go to it, packets come back.
    /// The header waits for its codec parameters.
    ext: Option<External>,
    ext_pending: Vec<Pkt>,
    ext_tb: ff::AVRational,
    rotation: u32,
}

unsafe impl Send for Writer {}

struct Graph {
    graph: *mut ff::AVFilterGraph,
    src: *mut ff::AVFilterContext,
    sink: *mut ff::AVFilterContext,
}

impl Drop for Graph {
    fn drop(&mut self) {
        unsafe { ff::avfilter_graph_free(&mut self.graph) };
    }
}

fn col_tags(c: &ColorInfo, yuv: bool) -> (ff::AVColorPrimaries, ff::AVColorTransferCharacteristic, ff::AVColorSpace, ff::AVColorRange) {
    use ff::{AVColorPrimaries as P, AVColorRange as R, AVColorSpace as S, AVColorTransferCharacteristic as T};
    let p = match c.primaries {
        Primaries::Bt709 => P::AVCOL_PRI_BT709,
        Primaries::Bt601 => P::AVCOL_PRI_SMPTE170M,
        Primaries::Bt2020 => P::AVCOL_PRI_BT2020,
        Primaries::Other => P::AVCOL_PRI_UNSPECIFIED,
    };
    let t = match c.transfer {
        Transfer::Sdr => T::AVCOL_TRC_BT709,
        Transfer::Pq => T::AVCOL_TRC_SMPTE2084,
        Transfer::Hlg => T::AVCOL_TRC_ARIB_STD_B67,
        Transfer::Linear => T::AVCOL_TRC_LINEAR,
    };
    let s = if !yuv {
        S::AVCOL_SPC_RGB
    } else {
        match c.matrix {
            Matrix::Bt601 => S::AVCOL_SPC_SMPTE170M,
            Matrix::Bt709 | Matrix::Rgb => S::AVCOL_SPC_BT709,
            Matrix::Bt2020Ncl => S::AVCOL_SPC_BT2020_NCL,
            Matrix::Smpte240 => S::AVCOL_SPC_SMPTE240M,
        }
    };
    let r = if !yuv || c.full_range { R::AVCOL_RANGE_JPEG } else { R::AVCOL_RANGE_MPEG };
    (p, t, s, r)
}

/// The colour tags' codes (ISO/IEC 23091-2, as FFmpeg numbers them): primaries,
/// transfer, matrix. For bitstreams that carry them (ProRes, DNxHR).
pub fn color_codes(c: &ColorInfo, yuv: bool) -> (u8, u8, u8) {
    let (p, t, s, _) = col_tags(c, yuv);
    (p as i32 as u8, t as i32 as u8, s as i32 as u8)
}

unsafe fn dict_set(d: &mut *mut ff::AVDictionary, k: &str, v: &str) {
    let (k, v) = (cstr(k), cstr(v));
    unsafe { ff::av_dict_set(d, k.as_ptr(), v.as_ptr(), 0) };
}

impl Writer {
    /// Open `path` for writing. `muxer` None: chosen from the file name. For image
    /// sequences (`image2`) the path holds a `%06d` pattern.
    pub fn create(path: &str, muxer: Option<&str>, v: &VideoConfig, audio: Option<(&AudioConfig, &AudioSource)>, metadata: &[(String, String)]) -> Result<Writer> {
        unsafe {
            let cpath = cstr(path);
            let cmux = muxer.map(cstr);
            let mut oc: *mut ff::AVFormatContext = std::ptr::null_mut();
            ffutil::check(ff::avformat_alloc_output_context2(&mut oc, std::ptr::null(), cmux.as_ref().map_or(std::ptr::null(), |m| m.as_ptr()), cpath.as_ptr()), "output format")?;
            let mut w = Writer {
                oc,
                venc: std::ptr::null_mut(),
                vst: 0,
                layout: pack_layout("rgba", 1, 1).unwrap(),
                deliver: std::ptr::null_mut(),
                sws: std::ptr::null_mut(),
                converted: std::ptr::null_mut(),
                vfilter: None,
                pkt: Packet::new(),
                next_pts: 0,
                rate: v.rate,
                field_order: v.field_order,
                audio: None,
                header: false,
                finished: false,
                precoded: v.precoded.clone(),
                ext: None,
                ext_pending: Vec::new(),
                ext_tb: ff::AVRational { num: 1, den: 1 },
                rotation: v.rotation,
            };
            if v.precoded.is_some() {
                w.open_precoded(v)?;
            } else if external::handles(&v.encoder) {
                w.open_external(v)?;
            } else {
                w.open_video(v)?;
            }
            if let Some((ac, src)) = audio {
                match AudioPipe::new(oc, ac, src) {
                    Ok(Some(a)) => w.audio = Some(a),
                    Ok(None) => {}
                    Err(e) => return Err(e.context("audio")),
                }
            }
            let fmt = (*oc).oformat;
            if (*fmt).flags & ff::AVFMT_NOFILE as i32 == 0 {
                ffutil::check(ff::avio_open(&mut (*oc).pb, cpath.as_ptr(), ff::AVIO_FLAG_WRITE as i32), "open output file")?;
            }
            for (k, val) in metadata {
                dict_set(&mut (*oc).metadata, k, val);
            }
            // The external encoder's header comes with its first packets.
            if w.ext.is_none() {
                w.write_header()?;
            }
            Ok(w)
        }
    }

    unsafe fn write_header(&mut self) -> Result<()> {
        unsafe {
            let oc = self.oc;
            let fmt = (*oc).oformat;
            let mut opts: *mut ff::AVDictionary = std::ptr::null_mut();
            let mname = ffutil::opt_str((*fmt).name).unwrap_or_default();
            if mname.contains("mp4") || mname.contains("mov") {
                dict_set(&mut opts, "movflags", "+faststart");
            }
            if mname == "gif" {
                dict_set(&mut opts, "loop", "0");
            }
            let r = ff::avformat_write_header(oc, &mut opts);
            ff::av_dict_free(&mut opts);
            ffutil::check(r, "write header")?;
            self.header = true;
            Ok(())
        }
    }

    /// x264/x265 through the external program: the stream gets its parameters
    /// when the program sends them; the pictures are prepared as for our own
    /// encoders and sent raw.
    unsafe fn open_external(&mut self, v: &VideoConfig) -> Result<()> {
        unsafe {
            let st = ff::avformat_new_stream(self.oc, std::ptr::null());
            if st.is_null() {
                bail!("new stream");
            }
            self.vst = (*st).index;
            (*st).time_base = ff::AVRational { num: v.rate.1, den: v.rate.0 };
            (*st).avg_frame_rate = ff::AVRational { num: v.rate.0, den: v.rate.1 };
            (*st).sample_aspect_ratio = ff::AVRational { num: v.sar.0.max(1), den: v.sar.1.max(1) };
            let enc_fmt = pix_fmt_of(&v.pix_fmt);
            if enc_fmt < 0 {
                bail!("unknown pixel format {}", v.pix_fmt);
            }
            let info = pix_fmt_info(&v.pix_fmt);
            let yuv = info.as_ref().is_some_and(|i| !i.rgb);
            let color = v.color.as_ref().map(|c| {
                let (p, t, s, r) = col_tags(c, yuv);
                let n = |x: *const std::ffi::c_char| ffutil::opt_str(x).unwrap_or_else(|| "unknown".into());
                [n(ff::av_color_primaries_name(p)), n(ff::av_color_transfer_name(t)), n(ff::av_color_space_name(s)), n(ff::av_color_range_name(r))]
            });
            let mut options = v.options.clone();
            match &v.rate_control {
                RateControl::Default => {}
                RateControl::Quality(q) => options.push(("crf".into(), format!("{q}"))),
                RateControl::Bitrate(k) => options.push(("b:v".into(), format!("{k}k"))),
                RateControl::Cbr(k) => {
                    for o in ["b:v", "minrate", "maxrate"] {
                        options.push((o.into(), format!("{k}k")));
                    }
                    options.push(("bufsize".into(), format!("{}k", (k / 2).max(1))));
                }
            }
            let job = external::Job {
                encoder: &v.encoder,
                pix_fmt: &v.pix_fmt,
                width: v.width,
                height: v.height,
                rate: v.rate,
                sar: v.sar,
                options,
                gop: v.gop,
                field_order: v.field_order,
                color,
                global_header: (*(*self.oc).oformat).flags & ff::AVFMT_GLOBALHEADER as i32 != 0,
            };
            self.setup_delivery(v, enc_fmt, &info)?;
            self.ext = Some(External::start(&job)?);
            Ok(())
        }
    }

    /// Take what the external program sent: its parameters (then the header is
    /// written) and packets. `block`: until it ends.
    unsafe fn ext_pump(&mut self, block: bool) -> Result<()> {
        unsafe {
            loop {
                let Some(ext) = self.ext.as_ref() else { return Ok(()) };
                let msg = if block {
                    match ext.rx.recv() {
                        Ok(m) => m,
                        Err(_) => return Ok(()),
                    }
                } else {
                    match ext.rx.try_recv() {
                        Ok(m) => m,
                        Err(_) => return Ok(()),
                    }
                };
                match msg {
                    Msg::Params(p) => {
                        let st = *(*self.oc).streams.add(self.vst as usize);
                        let sar = (*st).sample_aspect_ratio;
                        ffutil::check(ff::avcodec_parameters_copy((*st).codecpar, p.0), "stream parameters")?;
                        (*(*st).codecpar).codec_tag = 0;
                        (*(*st).codecpar).sample_aspect_ratio = sar;
                        if self.rotation % 360 != 0 {
                            let par = (*st).codecpar;
                            let sd = ff::av_packet_side_data_new(&mut (*par).coded_side_data, &mut (*par).nb_coded_side_data, ff::AVPacketSideDataType::AV_PKT_DATA_DISPLAYMATRIX, 36, 0);
                            if !sd.is_null() {
                                ff::av_display_rotation_set((*sd).data as *mut i32, -(self.rotation as f64));
                            }
                        }
                        if let Some(tff) = self.field_order {
                            (*(*st).codecpar).field_order = if tff { ff::AVFieldOrder::AV_FIELD_TT } else { ff::AVFieldOrder::AV_FIELD_BB };
                        }
                        self.ext_tb = p.1;
                        self.write_header()?;
                        self.ext_flush(false)?;
                    }
                    Msg::Packet(k) => {
                        self.ext_pending.push(k);
                        self.ext_flush(false)?;
                    }
                    Msg::End(r) => {
                        if let Err(e) = r {
                            let why = self.ext.as_ref().map(|x| x.error_text()).unwrap_or_default();
                            bail!("{e}: {why}");
                        }
                        self.ext_flush(true)?;
                        return Ok(());
                    }
                }
            }
        }
    }

    /// Mux the waiting packets once the header is out. The first packets of a stream
    /// with B-frames come without decode times (libavformat learns the reorder depth
    /// a few packets in): they wait for the first known one and get the times before
    /// it, one frame apart. `end`: no more are coming (unknown times = presentation).
    unsafe fn ext_flush(&mut self, end: bool) -> Result<()> {
        unsafe {
            if !self.header || self.ext_pending.is_empty() {
                return Ok(());
            }
            let none = ffutil::AV_NOPTS_VALUE;
            if let Some(i) = self.ext_pending.iter().position(|k| (*k.0).dts != none) {
                let d = (*self.ext_pending[i].0).dts;
                let step = ((*self.ext_pending[i].0).duration).max(1);
                for (j, k) in self.ext_pending[..i].iter().enumerate() {
                    (*k.0).dts = d - (i - j) as i64 * step;
                }
            } else if end {
                for k in &self.ext_pending {
                    (*k.0).dts = (*k.0).pts;
                }
            } else {
                return Ok(());
            }
            for k in std::mem::take(&mut self.ext_pending) {
                if (*k.0).dts == none {
                    // After the first known time every packet has one; keep the order.
                    (*k.0).dts = (*k.0).pts;
                }
                self.ext_mux(k)?;
            }
            Ok(())
        }
    }

    unsafe fn ext_mux(&mut self, k: Pkt) -> Result<()> {
        unsafe {
            let st = *(*self.oc).streams.add(self.vst as usize);
            ff::av_packet_rescale_ts(k.0, self.ext_tb, (*st).time_base);
            (*k.0).stream_index = self.vst;
            ffutil::check(ff::av_interleaved_write_frame(self.oc, k.0), "write video")?;
            Ok(())
        }
    }

    unsafe fn open_video(&mut self, v: &VideoConfig) -> Result<()> {
        unsafe {
            let en = cstr(&v.encoder);
            let codec = ff::avcodec_find_encoder_by_name(en.as_ptr());
            if codec.is_null() {
                bail!("this FFmpeg has no encoder {}", v.encoder);
            }
            let st = ff::avformat_new_stream(self.oc, std::ptr::null());
            if st.is_null() {
                bail!("new stream");
            }
            self.vst = (*st).index;
            let ctx = ff::avcodec_alloc_context3(codec);
            self.venc = ctx;
            let enc_fmt = pix_fmt_of(&v.pix_fmt);
            if enc_fmt < 0 {
                bail!("unknown pixel format {}", v.pix_fmt);
            }
            (*ctx).width = v.width as i32;
            (*ctx).height = v.height as i32;
            (*ctx).pix_fmt = std::mem::transmute::<i32, ff::AVPixelFormat>(enc_fmt);
            (*ctx).time_base = ff::AVRational { num: v.rate.1, den: v.rate.0 };
            (*ctx).framerate = ff::AVRational { num: v.rate.0, den: v.rate.1 };
            (*ctx).sample_aspect_ratio = ff::AVRational { num: v.sar.0.max(1), den: v.sar.1.max(1) };
            (*ctx).thread_count = 0;
            if let Some(g) = v.gop {
                (*ctx).gop_size = g;
            }
            if let Some(tff) = v.field_order {
                (*ctx).field_order = if tff { ff::AVFieldOrder::AV_FIELD_TT } else { ff::AVFieldOrder::AV_FIELD_BB };
                // Field-aware coding (MPEG-2, H.264, ProRes, DNxHR read these).
                (*ctx).flags |= (ff::AV_CODEC_FLAG_INTERLACED_DCT | ff::AV_CODEC_FLAG_INTERLACED_ME) as i32;
            }
            let info = pix_fmt_info(&v.pix_fmt);
            let yuv = info.as_ref().is_some_and(|i| !i.rgb);
            if let Some(c) = &v.color {
                let (p, t, s, r) = col_tags(c, yuv);
                (*ctx).color_primaries = p;
                (*ctx).color_trc = t;
                (*ctx).colorspace = s;
                (*ctx).color_range = r;
            }
            if (*(*self.oc).oformat).flags & ff::AVFMT_GLOBALHEADER as i32 != 0 {
                (*ctx).flags |= ff::AV_CODEC_FLAG_GLOBAL_HEADER as i32;
            }
            let mut opts: *mut ff::AVDictionary = std::ptr::null_mut();
            let name = v.encoder.as_str();
            match &v.rate_control {
                RateControl::Default => {}
                RateControl::Quality(q) => apply_quality(ctx, &mut opts, name, *q),
                RateControl::Bitrate(k) => {
                    (*ctx).bit_rate = *k as i64 * 1000;
                    if name.ends_with("_nvenc") {
                        dict_set(&mut opts, "rc", "vbr");
                    }
                }
                RateControl::Cbr(k) => {
                    let b = *k as i64 * 1000;
                    (*ctx).bit_rate = b;
                    (*ctx).rc_min_rate = b;
                    (*ctx).rc_max_rate = b;
                    (*ctx).rc_buffer_size = (b / 2).min(i32::MAX as i64) as i32;
                    if name.ends_with("_nvenc") || name.ends_with("_amf") {
                        dict_set(&mut opts, "rc", "cbr");
                    }
                }
            }
            for (k, val) in &v.options {
                dict_set(&mut opts, k, val);
            }
            let r = ff::avcodec_open2(ctx, codec, &mut opts);
            let mut left = Vec::new();
            let mut e: *const ff::AVDictionaryEntry = std::ptr::null();
            loop {
                e = ff::av_dict_get(opts, c"".as_ptr(), e, ff::AV_DICT_IGNORE_SUFFIX as i32);
                if e.is_null() {
                    break;
                }
                left.push(ffutil::opt_str((*e).key).unwrap_or_default());
            }
            ff::av_dict_free(&mut opts);
            ffutil::check(r, &format!("open encoder {}", v.encoder))?;
            if !left.is_empty() {
                log::warn!("export: {} ignored options {:?}", v.encoder, left);
            }
            ffutil::check(ff::avcodec_parameters_from_context((*st).codecpar, ctx), "stream parameters")?;
            (*st).time_base = (*ctx).time_base;
            (*st).avg_frame_rate = (*ctx).framerate;
            (*st).sample_aspect_ratio = (*ctx).sample_aspect_ratio;
            if v.rotation % 360 != 0 {
                let par = (*st).codecpar;
                let sd = ff::av_packet_side_data_new(&mut (*par).coded_side_data, &mut (*par).nb_coded_side_data, ff::AVPacketSideDataType::AV_PKT_DATA_DISPLAYMATRIX, 36, 0);
                if !sd.is_null() {
                    ff::av_display_rotation_set((*sd).data as *mut i32, -(v.rotation as f64));
                }
            }

            self.setup_delivery(v, enc_fmt, &info)?;
            Ok(())
        }
    }

    /// What the GPU delivers, and how it reaches the encoder (swscale for formats
    /// the GPU cannot write, the filter graph when there is one).
    unsafe fn setup_delivery(&mut self, v: &VideoConfig, enc_fmt: i32, info: &Option<PixFmtInfo>) -> Result<()> {
        unsafe {
            let (deliver_fmt, filter_fmt) = match &v.filter {
                Some((_, input)) => (pix_fmt_of(input), Some(pix_fmt_of(input))),
                None => (enc_fmt, None),
            };
            let layout = match pack_layout_of(deliver_fmt, v.width, v.height) {
                Some(l) => l,
                None => {
                    // A wide format the GPU writes, converted by swscale.
                    let l = delivery_layout(&pix_fmt_name(deliver_fmt), None, v.width, v.height).context("wide pixel format")?;
                    self.sws = ff::sws_getContext(
                        v.width as i32,
                        v.height as i32,
                        std::mem::transmute::<i32, ff::AVPixelFormat>(l.pix_fmt),
                        v.width as i32,
                        v.height as i32,
                        std::mem::transmute::<i32, ff::AVPixelFormat>(deliver_fmt),
                        (ff::SWS_BICUBIC | ff::SWS_ACCURATE_RND | ff::SWS_FULL_CHR_H_INT) as i32,
                        std::ptr::null_mut(),
                        std::ptr::null_mut(),
                        std::ptr::null(),
                    );
                    if self.sws.is_null() {
                        bail!("cannot convert to {}", pix_fmt_name(deliver_fmt));
                    }
                    if let Some(c) = &v.color {
                        let cs = match c.matrix {
                            Matrix::Bt601 => ff::SWS_CS_ITU601,
                            Matrix::Bt2020Ncl => ff::SWS_CS_BT2020,
                            Matrix::Smpte240 => ff::SWS_CS_SMPTE240M,
                            _ => ff::SWS_CS_ITU709,
                        };
                        let table = ff::sws_getCoefficients(cs as i32);
                        let full = (!l.yuv || c.full_range) as i32;
                        let dst_full = info.as_ref().is_some_and(|i| i.rgb) as i32 | c.full_range as i32;
                        ff::sws_setColorspaceDetails(self.sws, table, full, table, dst_full, 0, 1 << 16, 1 << 16);
                    }
                    self.converted = ff::av_frame_alloc();
                    (*self.converted).width = v.width as i32;
                    (*self.converted).height = v.height as i32;
                    (*self.converted).format = deliver_fmt;
                    ffutil::check(ff::av_frame_get_buffer(self.converted, 0), "frame buffer")?;
                    l
                }
            };
            self.deliver = ff::av_frame_alloc();
            (*self.deliver).width = v.width as i32;
            (*self.deliver).height = v.height as i32;
            (*self.deliver).format = layout.pix_fmt;
            ffutil::check(ff::av_frame_get_buffer(self.deliver, 0), "frame buffer")?;
            self.layout = layout;
            if let (Some((graph, _)), Some(ffmt)) = (&v.filter, filter_fmt) {
                self.vfilter = Some(video_graph(graph, v.width, v.height, ffmt, v.rate, enc_fmt)?);
            }
            Ok(())
        }
    }

    /// A stream for packets encoded elsewhere: its parameters are set from the
    /// config, no encoder is opened.
    unsafe fn open_precoded(&mut self, v: &VideoConfig) -> Result<()> {
        unsafe {
            let pc = v.precoded.as_ref().unwrap();
            let cn = cstr(&pc.codec);
            let desc = ff::avcodec_descriptor_get_by_name(cn.as_ptr());
            if desc.is_null() {
                bail!("unknown codec {}", pc.codec);
            }
            let st = ff::avformat_new_stream(self.oc, std::ptr::null());
            if st.is_null() {
                bail!("new stream");
            }
            self.vst = (*st).index;
            let par = &mut *(*st).codecpar;
            par.codec_type = ff::AVMediaType::AVMEDIA_TYPE_VIDEO;
            par.codec_id = (*desc).id;
            if let Some(t) = pc.tag {
                par.codec_tag = u32::from_le_bytes(t);
            }
            par.width = v.width as i32;
            par.height = v.height as i32;
            par.format = pix_fmt_of(&v.pix_fmt);
            par.sample_aspect_ratio = ff::AVRational { num: v.sar.0.max(1), den: v.sar.1.max(1) };
            par.profile = pc.profile;
            par.bits_per_coded_sample = pc.bits_per_coded_sample;
            par.bits_per_raw_sample = pc.bits_per_raw_sample;
            par.bit_rate = pc.bit_rate;
            par.field_order = match v.field_order {
                Some(true) => ff::AVFieldOrder::AV_FIELD_TT,
                Some(false) => ff::AVFieldOrder::AV_FIELD_BB,
                None => ff::AVFieldOrder::AV_FIELD_PROGRESSIVE,
            };
            if let Some(c) = &v.color {
                let yuv = pix_fmt_info(&v.pix_fmt).is_some_and(|i| !i.rgb);
                let (p, t, s, r) = col_tags(c, yuv);
                par.color_primaries = p;
                par.color_trc = t;
                par.color_space = s;
                par.color_range = r;
            }
            (*st).time_base = ff::AVRational { num: v.rate.1, den: v.rate.0 };
            (*st).avg_frame_rate = ff::AVRational { num: v.rate.0, den: v.rate.1 };
            (*st).r_frame_rate = (*st).avg_frame_rate;
            (*st).sample_aspect_ratio = par.sample_aspect_ratio;
            if v.rotation % 360 != 0 {
                let sd = ff::av_packet_side_data_new(&mut par.coded_side_data, &mut par.nb_coded_side_data, ff::AVPacketSideDataType::AV_PKT_DATA_DISPLAYMATRIX, 36, 0);
                if !sd.is_null() {
                    ff::av_display_rotation_set((*sd).data as *mut i32, -(v.rotation as f64));
                }
            }
            // Nothing is packed for us: write() takes the packet as it is.
            self.layout = PackLayout { total: 0, ..pack_layout("gray", 1, 1).unwrap() };
            Ok(())
        }
    }

    /// One finished packet (pre-encoded streams).
    unsafe fn write_packet(&mut self, data: &[u8]) -> Result<()> {
        unsafe {
            let pkt = self.pkt.0;
            ff::av_packet_unref(pkt);
            ffutil::check(ff::av_new_packet(pkt, data.len() as i32), "packet")?;
            std::ptr::copy_nonoverlapping(data.as_ptr(), (*pkt).data, data.len());
            (*pkt).pts = self.next_pts;
            (*pkt).dts = self.next_pts;
            (*pkt).duration = 1;
            if self.precoded.as_ref().is_none_or(|p| p.intra_only) {
                (*pkt).flags |= ff::AV_PKT_FLAG_KEY as i32;
            }
            let st = *(*self.oc).streams.add(self.vst as usize);
            ff::av_packet_rescale_ts(pkt, ff::AVRational { num: self.rate.1, den: self.rate.0 }, (*st).time_base);
            (*pkt).stream_index = self.vst;
            ffutil::check(ff::av_interleaved_write_frame(self.oc, pkt), "write video")?;
            self.next_pts += 1;
            let t = self.next_pts as f64 * self.rate.1 as f64 / self.rate.0 as f64;
            if let Some(a) = &mut self.audio {
                a.pump(self.oc, t)?;
            }
            Ok(())
        }
    }

    /// How to lay out each picture given to [`write`].
    pub fn layout(&self) -> &PackLayout {
        &self.layout
    }

    /// Pictures arrive encoded ([`write`] takes packets).
    pub fn is_precoded(&self) -> bool {
        self.precoded.is_some()
    }

    /// One picture, laid out as [`layout`] says (for a pre-encoded stream: one
    /// finished packet).
    pub fn write(&mut self, data: &[u8]) -> Result<()> {
        if self.precoded.is_some() {
            return unsafe { self.write_packet(data) };
        }
        anyhow::ensure!(data.len() as u64 >= self.layout.total, "picture too small");
        unsafe {
            ffutil::check(ff::av_frame_make_writable(self.deliver), "frame writable")?;
            let f = &mut *self.deliver;
            for (i, p) in self.layout.planes.iter().enumerate() {
                let row = p.w as usize * p.step as usize * p.bytes as usize;
                let dst_stride = f.linesize[i] as usize;
                for y in 0..p.h as usize {
                    let src = &data[p.offset as usize + y * p.stride as usize..][..row];
                    std::ptr::copy_nonoverlapping(src.as_ptr(), f.data[i].add(y * dst_stride), row);
                }
            }
            let frame = if !self.sws.is_null() {
                ffutil::check(ff::av_frame_make_writable(self.converted), "frame writable")?;
                ffutil::check(ff::sws_scale_frame(self.sws, self.converted, self.deliver), "convert picture")?;
                self.converted
            } else {
                self.deliver
            };
            (*frame).pts = self.next_pts;
            match self.field_order {
                Some(tff) => {
                    (*frame).flags |= ff::AV_FRAME_FLAG_INTERLACED as i32;
                    if tff {
                        (*frame).flags |= ff::AV_FRAME_FLAG_TOP_FIELD_FIRST as i32;
                    } else {
                        (*frame).flags &= !(ff::AV_FRAME_FLAG_TOP_FIELD_FIRST as i32);
                    }
                }
                None => (*frame).flags &= !((ff::AV_FRAME_FLAG_INTERLACED | ff::AV_FRAME_FLAG_TOP_FIELD_FIRST) as i32),
            }
            self.next_pts += 1;
            if let Some(ext) = &mut self.ext {
                let f = &*frame;
                let fmt = std::mem::transmute::<i32, ff::AVPixelFormat>(f.format);
                let size = ff::av_image_get_buffer_size(fmt, f.width, f.height, 1);
                ffutil::check(size, "picture size")?;
                let mut buf = std::mem::take(&mut ext.buf);
                buf.resize(size as usize, 0);
                ffutil::check(ff::av_image_copy_to_buffer(buf.as_mut_ptr(), size, f.data.as_ptr() as *const *const u8, f.linesize.as_ptr(), fmt, f.width, f.height, 1), "copy picture")?;
                let r = ext.send(&buf);
                ext.buf = buf;
                r?;
                self.ext_pump(false)?;
            } else if let Some(g) = &self.vfilter {
                ffutil::check(ff::av_buffersrc_add_frame_flags(g.src, frame, ff::AV_BUFFERSRC_FLAG_KEEP_REF as i32), "video filter")?;
                self.drain_filter(false)?;
            } else {
                self.encode(frame)?;
            }
            // Audio up to the end of this picture (once the header is out).
            let t = self.next_pts as f64 * self.rate.1 as f64 / self.rate.0 as f64;
            if self.header
                && let Some(a) = &mut self.audio
            {
                a.pump(self.oc, t)?;
            }
        }
        Ok(())
    }

    unsafe fn drain_filter(&mut self, eof: bool) -> Result<()> {
        unsafe {
            let g = self.vfilter.as_ref().unwrap();
            let (src, sink) = (g.src, g.sink);
            if eof {
                ff::av_buffersrc_add_frame_flags(src, std::ptr::null_mut(), 0);
            }
            let out = ff::av_frame_alloc();
            loop {
                let r = ff::av_buffersink_get_frame(sink, out);
                if r == ffutil::AVERROR_EAGAIN || r == ffutil::AVERROR_EOF {
                    break;
                }
                if r < 0 {
                    let mut o = out;
                    ff::av_frame_free(&mut o);
                    ffutil::check(r, "video filter")?;
                }
                let res = self.encode(out);
                ff::av_frame_unref(out);
                if let Err(e) = res {
                    let mut o = out;
                    ff::av_frame_free(&mut o);
                    return Err(e);
                }
            }
            let mut o = out;
            ff::av_frame_free(&mut o);
            Ok(())
        }
    }

    /// Send a frame (null: flush) and write what comes out.
    unsafe fn encode(&mut self, frame: *mut ff::AVFrame) -> Result<()> {
        unsafe {
            let r = ff::avcodec_send_frame(self.venc, frame);
            if r < 0 && r != ffutil::AVERROR_EOF {
                ffutil::check(r, "encode")?;
            }
            loop {
                let r = ff::avcodec_receive_packet(self.venc, self.pkt.0);
                if r == ffutil::AVERROR_EAGAIN || r == ffutil::AVERROR_EOF {
                    return Ok(());
                }
                ffutil::check(r, "encode")?;
                let st = *(*self.oc).streams.add(self.vst as usize);
                ff::av_packet_rescale_ts(self.pkt.0, (*self.venc).time_base, (*st).time_base);
                (*self.pkt.0).stream_index = self.vst;
                ffutil::check(ff::av_interleaved_write_frame(self.oc, self.pkt.0), "write video")?;
            }
        }
    }

    /// Flush the encoders and close the file.
    pub fn finish(mut self) -> Result<()> {
        unsafe {
            if self.vfilter.is_some() {
                self.drain_filter(true)?;
            }
            if let Some(ext) = &mut self.ext {
                ext.close_input();
                self.ext_pump(true)?;
                if let Some(ext) = &mut self.ext {
                    ext.wait()?;
                }
                if !self.header {
                    bail!("the external encoder sent nothing");
                }
            } else if self.precoded.is_none() {
                self.encode(std::ptr::null_mut())?;
            }
            if let Some(a) = &mut self.audio {
                a.pump(self.oc, f64::INFINITY)?;
            }
            ffutil::check(ff::av_write_trailer(self.oc), "write trailer")?;
            self.finished = true;
        }
        Ok(())
    }

    /// Pictures written so far.
    pub fn frames(&self) -> i64 {
        self.next_pts
    }
}

unsafe fn apply_quality(ctx: *mut ff::AVCodecContext, opts: &mut *mut ff::AVDictionary, name: &str, q: f32) {
    let s = format!("{q}");
    let qi = format!("{}", q.round() as i32);
    unsafe {
        match name {
            "libx264" | "libx264rgb" | "libx265" | "libsvtav1" => dict_set(opts, "crf", &s),
            "libvpx-vp9" | "libvpx" | "libaom-av1" => {
                dict_set(opts, "crf", &qi);
                (*ctx).bit_rate = 0;
            }
            "librav1e" => dict_set(opts, "qp", &format!("{}", (q * 4.0).round() as i32)),
            "libkvazaar" => dict_set(opts, "kvazaar-params", &format!("qp={qi}")),
            n if n.ends_with("_nvenc") => {
                dict_set(opts, "rc", "vbr");
                dict_set(opts, "cq", &qi);
                (*ctx).bit_rate = 0;
            }
            n if n.ends_with("_amf") => {
                dict_set(opts, "rc", "cqp");
                for k in ["qp_i", "qp_p", "qp_b"] {
                    dict_set(opts, k, &qi);
                }
            }
            n if n.ends_with("_qsv") || n.ends_with("_vaapi") => {
                (*ctx).global_quality = q.round() as i32;
            }
            n if n.ends_with("_vulkan") || n.ends_with("_d3d12va") => {
                dict_set(opts, "rc_mode", "cqp");
                (*ctx).global_quality = q.round() as i32;
            }
            n if n.ends_with("_mf") => {
                dict_set(opts, "rate_control", "quality");
                dict_set(opts, "quality", &format!("{}", (100.0 - q * 2.0).clamp(0.0, 100.0) as i32));
            }
            "cfhd" => dict_set(opts, "quality", &qi),
            // Quantiser scale (MPEG-2, MJPEG/JPEG, ProRes ks, MPEG-4, WebP...).
            _ => {
                (*ctx).flags |= ff::AV_CODEC_FLAG_QSCALE as i32;
                (*ctx).global_quality = (q * ff::FF_QP2LAMBDA as f32).round() as i32;
            }
        }
    }
}

unsafe fn video_graph(desc: &str, w: u32, h: u32, in_fmt: i32, rate: (i32, i32), out_fmt: i32) -> Result<Graph> {
    unsafe {
        let mut g = Graph { graph: ff::avfilter_graph_alloc(), src: std::ptr::null_mut(), sink: std::ptr::null_mut() };
        let args = format!("video_size={w}x{h}:pix_fmt={in_fmt}:time_base={}/{}:pixel_aspect=1/1", rate.1, rate.0);
        let a = cstr(&args);
        ffutil::check(ff::avfilter_graph_create_filter(&mut g.src, ff::avfilter_get_by_name(c"buffer".as_ptr()), c"in".as_ptr(), a.as_ptr(), std::ptr::null_mut(), g.graph), "buffer filter")?;
        ffutil::check(ff::avfilter_graph_create_filter(&mut g.sink, ff::avfilter_get_by_name(c"buffersink".as_ptr()), c"out".as_ptr(), std::ptr::null(), std::ptr::null_mut(), g.graph), "buffersink filter")?;
        let full = format!("{desc},format={}", pix_fmt_name(out_fmt));
        link(&g, &full)?;
        Ok(g)
    }
}

/// Parse `desc` between the graph's source and sink and configure it.
unsafe fn link(g: &Graph, desc: &str) -> Result<()> {
    unsafe {
        let mut outputs = ff::avfilter_inout_alloc();
        let mut inputs = ff::avfilter_inout_alloc();
        (*outputs).name = ff::av_strdup(c"in".as_ptr());
        (*outputs).filter_ctx = g.src;
        (*outputs).pad_idx = 0;
        (*outputs).next = std::ptr::null_mut();
        (*inputs).name = ff::av_strdup(c"out".as_ptr());
        (*inputs).filter_ctx = g.sink;
        (*inputs).pad_idx = 0;
        (*inputs).next = std::ptr::null_mut();
        let d = cstr(desc);
        let r = ff::avfilter_graph_parse_ptr(g.graph, d.as_ptr(), &mut inputs, &mut outputs, std::ptr::null_mut());
        ff::avfilter_inout_free(&mut inputs);
        ff::avfilter_inout_free(&mut outputs);
        ffutil::check(r, &format!("filter graph '{desc}'"))?;
        ffutil::check(ff::avfilter_graph_config(g.graph, std::ptr::null_mut()), "filter graph config")?;
        Ok(())
    }
}

impl Drop for Writer {
    fn drop(&mut self) {
        unsafe {
            if self.header && !self.finished {
                // Cancelled or failed: close the file so the caller can remove it.
                ff::av_write_trailer(self.oc);
            }
            self.audio = None;
            self.vfilter = None;
            ff::avcodec_free_context(&mut self.venc);
            ff::av_frame_free(&mut self.deliver);
            ff::av_frame_free(&mut self.converted);
            ff::sws_freeContext(self.sws);
            if !self.oc.is_null() {
                if (*(*self.oc).oformat).flags & ff::AVFMT_NOFILE as i32 == 0 {
                    ff::avio_closep(&mut (*self.oc).pb);
                }
                ff::avformat_free_context(self.oc);
                self.oc = std::ptr::null_mut();
            }
        }
    }
}

// ---- audio --------------------------------------------------------------------------

struct AudioPipe {
    input: Input,
    stream: i32,
    dec: *mut ff::AVCodecContext,
    enc: *mut ff::AVCodecContext,
    st: i32,
    graph: Graph,
    frame: *mut ff::AVFrame,
    pkt: Packet,
    out_pkt: Packet,
    /// Samples given to the encoder.
    next_pts: i64,
    rate: i32,
    /// Stop reading the source after this (absolute seconds).
    read_until: f64,
    input_done: bool,
    flushed: bool,
}

impl AudioPipe {
    unsafe fn new(oc: *mut ff::AVFormatContext, cfg: &AudioConfig, src: &AudioSource) -> Result<Option<AudioPipe>> {
        unsafe {
            let input = Input::open(&src.path)?;
            let si = ff::av_find_best_stream(input.ctx, ff::AVMediaType::AVMEDIA_TYPE_AUDIO, -1, -1, std::ptr::null_mut(), 0);
            if si < 0 {
                return Ok(None);
            }
            input.discard_others(si);
            let ist = input.stream(si);
            let par = (*ist).codecpar;
            let dcodec = ff::avcodec_find_decoder((*par).codec_id);
            if dcodec.is_null() {
                bail!("no decoder for the source audio");
            }
            let mut dec = ff::avcodec_alloc_context3(dcodec);
            ffutil::check(ff::avcodec_parameters_to_context(dec, par), "audio decoder")?;
            (*dec).pkt_timebase = (*ist).time_base;
            let r = ff::avcodec_open2(dec, dcodec, std::ptr::null_mut());
            if r < 0 {
                ff::avcodec_free_context(&mut dec);
                ffutil::check(r, "open audio decoder")?;
            }

            let en = cstr(&cfg.encoder);
            let ecodec = ff::avcodec_find_encoder_by_name(en.as_ptr());
            if ecodec.is_null() {
                ff::avcodec_free_context(&mut dec);
                bail!("this FFmpeg has no audio encoder {}", cfg.encoder);
            }
            let enc = ff::avcodec_alloc_context3(ecodec);
            let fmts: Vec<ff::AVSampleFormat> = supported(ecodec, ff::AVCodecConfig::AV_CODEC_CONFIG_SAMPLE_FORMAT, |f: &ff::AVSampleFormat| *f as i32 == -1);
            let rates: Vec<i32> = supported(ecodec, ff::AVCodecConfig::AV_CODEC_CONFIG_SAMPLE_RATE, |r: &i32| *r == 0);
            (*enc).sample_fmt = fmts.first().copied().unwrap_or(ff::AVSampleFormat::AV_SAMPLE_FMT_FLTP);
            let want = if cfg.sample_rate > 0 { cfg.sample_rate as i32 } else { (*par).sample_rate.max(8000) };
            (*enc).sample_rate = if rates.is_empty() || rates.contains(&want) {
                want
            } else if rates.contains(&48000) {
                48000
            } else {
                *rates.iter().min_by_key(|r| (**r - want).abs()).unwrap()
            };
            let ch = (*par).ch_layout.nb_channels.clamp(1, 8);
            ff::av_channel_layout_default(&mut (*enc).ch_layout, ch);
            (*enc).time_base = ff::AVRational { num: 1, den: (*enc).sample_rate };
            if cfg.kbps > 0 {
                (*enc).bit_rate = cfg.kbps as i64 * 1000;
            }
            if (*(*oc).oformat).flags & ff::AVFMT_GLOBALHEADER as i32 != 0 {
                (*enc).flags |= ff::AV_CODEC_FLAG_GLOBAL_HEADER as i32;
            }
            let st = ff::avformat_new_stream(oc, std::ptr::null());
            let r = ff::avcodec_open2(enc, ecodec, std::ptr::null_mut());
            if r < 0 || st.is_null() {
                let mut e = enc;
                ff::avcodec_free_context(&mut e);
                ff::avcodec_free_context(&mut dec);
                ffutil::check(r, &format!("open audio encoder {}", cfg.encoder))?;
                bail!("new stream");
            }
            ffutil::check(ff::avcodec_parameters_from_context((*st).codecpar, enc), "audio stream")?;
            (*st).time_base = (*enc).time_base;

            // Filters: cut to the range (silence where the source has no audio),
            // retime, convert to what the encoder takes, and make the length exact.
            let sr = (*dec).sample_rate.max(1);
            let in_fmt = ffutil::opt_str(ff::av_get_sample_fmt_name((*dec).sample_fmt)).unwrap_or_default();
            let mut lay = [0 as std::ffi::c_char; 128];
            if (*dec).ch_layout.order == ff::AVChannelOrder::AV_CHANNEL_ORDER_UNSPEC {
                ff::av_channel_layout_default(&mut (*dec).ch_layout, (*dec).ch_layout.nb_channels.max(1));
            }
            ff::av_channel_layout_describe(&(*dec).ch_layout, lay.as_mut_ptr(), lay.len());
            let lay_s = CStr::from_ptr(lay.as_ptr()).to_string_lossy().into_owned();
            let tb = (*ist).time_base;
            let args = format!("time_base={}/{}:sample_rate={sr}:sample_fmt={in_fmt}:channel_layout={lay_s}", tb.num, tb.den);
            let mut g = Graph { graph: ff::avfilter_graph_alloc(), src: std::ptr::null_mut(), sink: std::ptr::null_mut() };
            let a = cstr(&args);
            ffutil::check(ff::avfilter_graph_create_filter(&mut g.src, ff::avfilter_get_by_name(c"abuffer".as_ptr()), c"in".as_ptr(), a.as_ptr(), std::ptr::null_mut(), g.graph), "abuffer")?;
            ffutil::check(ff::avfilter_graph_create_filter(&mut g.sink, ff::avfilter_get_by_name(c"abuffersink".as_ptr()), c"out".as_ptr(), std::ptr::null(), std::ptr::null_mut(), g.graph), "abuffersink")?;
            let osr = (*enc).sample_rate;
            let total = (src.length * osr as f64).round().max(0.0) as i64;
            let mut chain = vec![
                format!("atrim=start={:.6}:end={:.6}", src.start, src.end),
                format!("asetpts=PTS-{:.6}/TB", src.start),
                "aresample=async=1:first_pts=0".to_string(),
            ];
            let t = src.tempo;
            if (t - 1.0).abs() > 1e-6 {
                if src.keep_pitch {
                    // atempo takes 0.5..100 per instance.
                    let mut left = t;
                    while left < 0.5 {
                        chain.push("atempo=0.5".into());
                        left /= 0.5;
                    }
                    chain.push(format!("atempo={left:.6}"));
                } else {
                    chain.push(format!("asetrate={}", (sr as f64 * t).round() as i64));
                }
            }
            let out_fmt = ffutil::opt_str(ff::av_get_sample_fmt_name((*enc).sample_fmt)).unwrap_or_default();
            let mut olay = [0 as std::ffi::c_char; 128];
            ff::av_channel_layout_describe(&(*enc).ch_layout, olay.as_mut_ptr(), olay.len());
            let olay = CStr::from_ptr(olay.as_ptr()).to_string_lossy().into_owned();
            chain.push(format!("aresample={osr}"));
            chain.push(format!("aformat=sample_fmts={out_fmt}:sample_rates={osr}:channel_layouts={olay}"));
            chain.push(format!("apad=whole_len={total}"));
            chain.push(format!("atrim=end_sample={total}"));
            link(&g, &chain.join(","))?;
            let caps = (*ecodec).capabilities as u32;
            if (*enc).frame_size > 0 && caps & ff::AV_CODEC_CAP_VARIABLE_FRAME_SIZE == 0 {
                ff::av_buffersink_set_frame_size(g.sink, (*enc).frame_size as u32);
            }
            // Start reading a little before the range.
            let seek = ((src.start - 1.0).max(0.0) * ff::AV_TIME_BASE as f64) as i64;
            if seek > 0 {
                ff::av_seek_frame(input.ctx, -1, seek, ff::AVSEEK_FLAG_BACKWARD as i32);
            }
            Ok(Some(AudioPipe {
                input,
                stream: si,
                dec,
                enc,
                st: (*st).index,
                graph: g,
                frame: ff::av_frame_alloc(),
                pkt: Packet::new(),
                out_pkt: Packet::new(),
                next_pts: 0,
                rate: osr,
                read_until: src.end + 0.5,
                input_done: false,
                flushed: false,
            }))
        }
    }

    /// Encode audio up to `until` seconds of output (infinity: all of it, then flush).
    unsafe fn pump(&mut self, oc: *mut ff::AVFormatContext, until: f64) -> Result<()> {
        unsafe {
            while !self.flushed && (self.next_pts as f64 / self.rate as f64) < until {
                let r = ff::av_buffersink_get_frame(self.graph.sink, self.frame);
                if r >= 0 {
                    (*self.frame).pts = self.next_pts;
                    self.next_pts += (*self.frame).nb_samples as i64;
                    let s = ff::avcodec_send_frame(self.enc, self.frame);
                    ff::av_frame_unref(self.frame);
                    ffutil::check(s, "encode audio")?;
                    self.write_packets(oc)?;
                    continue;
                }
                if r == ffutil::AVERROR_EOF {
                    ff::avcodec_send_frame(self.enc, std::ptr::null());
                    self.write_packets(oc)?;
                    self.flushed = true;
                    break;
                }
                if r != ffutil::AVERROR_EAGAIN {
                    ffutil::check(r, "audio filter")?;
                }
                self.feed()?;
            }
            Ok(())
        }
    }

    /// Decode more source audio into the filters.
    unsafe fn feed(&mut self) -> Result<()> {
        unsafe {
            if self.input_done {
                return Ok(());
            }
            loop {
                let r = ff::avcodec_receive_frame(self.dec, self.frame);
                if r == 0 {
                    let mut pts = (*self.frame).best_effort_timestamp;
                    if pts == ffutil::AV_NOPTS_VALUE {
                        pts = (*self.frame).pts;
                    }
                    (*self.frame).pts = pts;
                    let tb = (*self.input.stream(self.stream)).time_base;
                    let past = pts != ffutil::AV_NOPTS_VALUE && pts as f64 * ffutil::q2d(tb) > self.read_until;
                    ffutil::check(ff::av_buffersrc_add_frame_flags(self.graph.src, self.frame, 0), "audio filter in")?;
                    ff::av_frame_unref(self.frame);
                    if past {
                        self.end_input();
                    }
                    return Ok(());
                }
                if r == ffutil::AVERROR_EOF {
                    self.end_input();
                    return Ok(());
                }
                let rp = ff::av_read_frame(self.input.ctx, self.pkt.0);
                if rp < 0 {
                    ff::avcodec_send_packet(self.dec, std::ptr::null());
                    continue;
                }
                if (*self.pkt.0).stream_index == self.stream {
                    let s = ff::avcodec_send_packet(self.dec, self.pkt.0);
                    if s < 0 && s != ffutil::AVERROR_EAGAIN && s != ffutil::AVERROR_EOF {
                        log::debug!("audio packet: {}", ffutil::err_str(s));
                    }
                }
                ff::av_packet_unref(self.pkt.0);
            }
        }
    }

    unsafe fn end_input(&mut self) {
        self.input_done = true;
        unsafe { ff::av_buffersrc_add_frame_flags(self.graph.src, std::ptr::null_mut(), 0) };
    }

    unsafe fn write_packets(&mut self, oc: *mut ff::AVFormatContext) -> Result<()> {
        unsafe {
            loop {
                let r = ff::avcodec_receive_packet(self.enc, self.out_pkt.0);
                if r == ffutil::AVERROR_EAGAIN || r == ffutil::AVERROR_EOF {
                    return Ok(());
                }
                ffutil::check(r, "encode audio")?;
                let st = *(*oc).streams.add(self.st as usize);
                ff::av_packet_rescale_ts(self.out_pkt.0, (*self.enc).time_base, (*st).time_base);
                (*self.out_pkt.0).stream_index = self.st;
                ffutil::check(ff::av_interleaved_write_frame(oc, self.out_pkt.0), "write audio")?;
            }
        }
    }
}

impl Drop for AudioPipe {
    fn drop(&mut self) {
        unsafe {
            ff::av_frame_free(&mut self.frame);
            ff::avcodec_free_context(&mut self.dec);
            ff::avcodec_free_context(&mut self.enc);
        }
    }
}

/// The source file has an audio stream.
pub fn has_audio(path: &str) -> bool {
    let Ok(input) = Input::open(path) else { return false };
    unsafe { ff::av_find_best_stream(input.ctx, ff::AVMediaType::AVMEDIA_TYPE_AUDIO, -1, -1, std::ptr::null_mut(), 0) >= 0 }
}
