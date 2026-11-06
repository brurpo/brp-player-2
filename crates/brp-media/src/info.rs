//! Static information about an opened media file.

use crate::ffutil::{self, Input};
use ffmpeg_sys_next as ff;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Matrix {
    Bt601,
    Bt709,
    Bt2020Ncl,
    Smpte240,
    Rgb,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Transfer {
    Sdr,
    Pq,
    Hlg,
    Linear,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Primaries {
    Bt709,
    Bt601,
    Bt2020,
    Other,
}

#[derive(Clone, Copy, Debug)]
pub struct ColorInfo {
    pub matrix: Matrix,
    pub full_range: bool,
    pub transfer: Transfer,
    pub primaries: Primaries,
}

impl ColorInfo {
    pub fn is_hdr(&self) -> bool {
        matches!(self.transfer, Transfer::Pq | Transfer::Hlg)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FieldOrder {
    Unknown,
    Progressive,
    TopFirst,
    BottomFirst,
}

#[derive(Clone, Debug)]
pub struct MediaInfo {
    pub path: String,
    pub file_name: String,
    pub container: String,
    pub codec: String,
    pub pix_fmt: String,
    pub width: u32,
    pub height: u32,
    /// Display aspect ratio (width / height) including sample aspect ratio.
    pub dar: f64,
    pub frame_rate: f64,
    pub time_base: (i32, i32),
    pub start_pts: i64,
    /// Absolute time of the first video frame (seconds), to line up with audio.
    pub start_secs: f64,
    pub duration_secs: f64,
    pub frame_count_hint: i64,
    pub color: ColorInfo,
    pub field_order: FieldOrder,
    pub has_alpha: bool,
    pub bit_depth: u8,
    pub audio: Option<String>,
    pub timecode: Option<String>,
    pub video_stream: i32,
    /// Clockwise rotation needed for display: 0, 90, 180 or 270 degrees.
    pub rotation: u32,
}

impl MediaInfo {
    pub fn pts_to_secs(&self, pts: i64) -> f64 {
        (pts - self.start_pts) as f64 * self.time_base.0 as f64 / self.time_base.1.max(1) as f64
    }
}

pub(crate) unsafe fn read_info(path: &str, input: &Input, vs: i32) -> MediaInfo {
    unsafe {
        let st = input.stream(vs);
        let par = (*st).codecpar;
        let fmt = (*input.ctx).iformat;

        let codec = ffutil::opt_str(ff::avcodec_get_name((*par).codec_id)).unwrap_or_default();
        let pix: ff::AVPixelFormat = std::mem::transmute((*par).format);
        let pix_fmt = ffutil::opt_str(ff::av_get_pix_fmt_name(pix)).unwrap_or_else(|| "?".into());
        let desc = ff::av_pix_fmt_desc_get(pix);
        let (has_alpha, bit_depth) = if desc.is_null() {
            (false, 8)
        } else {
            (
                (*desc).flags & ff::AV_PIX_FMT_FLAG_ALPHA as u64 != 0,
                (*desc).comp[0].depth as u8,
            )
        };

        let width = (*par).width.max(0) as u32;
        let height = (*par).height.max(0) as u32;
        let sar = (*par).sample_aspect_ratio;
        let sar = if sar.num > 0 && sar.den > 0 { ffutil::q2d(sar) } else { 1.0 };
        let dar = if height > 0 { width as f64 * sar / height as f64 } else { 16.0 / 9.0 };

        let fr = ff::av_guess_frame_rate(input.ctx, st, std::ptr::null_mut());
        let mut frame_rate = ffutil::q2d(fr);
        if !(frame_rate > 0.0 && frame_rate < 1000.0) {
            frame_rate = 25.0;
        }

        let tb = (*st).time_base;
        let start_pts = if (*st).start_time != ffutil::AV_NOPTS_VALUE { (*st).start_time } else { 0 };
        let duration_secs = if (*st).duration > 0 {
            (*st).duration as f64 * ffutil::q2d(tb)
        } else if (*input.ctx).duration > 0 {
            (*input.ctx).duration as f64 / ff::AV_TIME_BASE as f64
        } else {
            0.0
        };
        let frame_count_hint = if (*st).nb_frames > 0 {
            (*st).nb_frames
        } else {
            (duration_secs * frame_rate).round() as i64
        };

        let rotation = display_rotation(par);
        let color = color_info(par, height);
        let field_order = match (*par).field_order {
            ff::AVFieldOrder::AV_FIELD_PROGRESSIVE => FieldOrder::Progressive,
            ff::AVFieldOrder::AV_FIELD_TT | ff::AVFieldOrder::AV_FIELD_TB => FieldOrder::TopFirst,
            ff::AVFieldOrder::AV_FIELD_BB | ff::AVFieldOrder::AV_FIELD_BT => FieldOrder::BottomFirst,
            _ => FieldOrder::Unknown,
        };

        let mut audio = None;
        let mut timecode = dict_get((*st).metadata, "timecode");
        for i in 0..input.nb_streams() as i32 {
            let s = input.stream(i);
            let p = (*s).codecpar;
            if audio.is_none() && (*p).codec_type == ff::AVMediaType::AVMEDIA_TYPE_AUDIO {
                let name = ffutil::opt_str(ff::avcodec_get_name((*p).codec_id)).unwrap_or_default();
                let ch = (*p).ch_layout.nb_channels;
                audio = Some(format!("{} {}ch {:.1} kHz", name.to_uppercase(), ch, (*p).sample_rate as f64 / 1000.0));
            }
            if timecode.is_none() {
                timecode = dict_get((*s).metadata, "timecode");
            }
        }
        if timecode.is_none() {
            timecode = dict_get((*input.ctx).metadata, "timecode");
        }

        let file_name = std::path::Path::new(path)
            .file_name()
            .map(|s| s.to_string_lossy().into_owned())
            .unwrap_or_default();

        MediaInfo {
            path: path.to_string(),
            file_name,
            container: ffutil::opt_str((*fmt).name).unwrap_or_default(),
            codec,
            pix_fmt,
            width,
            height,
            dar,
            frame_rate,
            time_base: (tb.num, tb.den),
            start_pts,
            start_secs: start_pts as f64 * ffutil::q2d(tb),
            duration_secs,
            frame_count_hint,
            color,
            field_order,
            has_alpha,
            bit_depth,
            audio,
            timecode,
            video_stream: vs,
            rotation,
        }
    }
}

unsafe fn display_rotation(par: *const ff::AVCodecParameters) -> u32 {
    unsafe {
        let sd = ff::av_packet_side_data_get(
            (*par).coded_side_data,
            (*par).nb_coded_side_data,
            ff::AVPacketSideDataType::AV_PKT_DATA_DISPLAYMATRIX,
        );
        if sd.is_null() || (*sd).size < 36 {
            return 0;
        }
        // FFmpeg gives the counter-clockwise angle; display needs the opposite.
        let ccw = ff::av_display_rotation_get((*sd).data as *const i32);
        if ccw.is_nan() {
            return 0;
        }
        let cw = (-ccw).rem_euclid(360.0);
        (((cw / 90.0).round() as u32) % 4) * 90
    }
}

unsafe fn dict_get(dict: *mut ff::AVDictionary, key: &str) -> Option<String> {
    let k = ffutil::cstr(key);
    unsafe {
        let e = ff::av_dict_get(dict, k.as_ptr(), std::ptr::null(), 0);
        if e.is_null() { None } else { ffutil::opt_str((*e).value) }
    }
}

unsafe fn color_info(par: *const ff::AVCodecParameters, height: u32) -> ColorInfo {
    use ff::AVColorSpace as S;
    use ff::AVColorTransferCharacteristic as T;
    use ff::AVColorPrimaries as P;
    unsafe {
        let matrix = match (*par).color_space {
            S::AVCOL_SPC_BT709 => Matrix::Bt709,
            S::AVCOL_SPC_BT470BG | S::AVCOL_SPC_SMPTE170M | S::AVCOL_SPC_FCC => Matrix::Bt601,
            S::AVCOL_SPC_BT2020_NCL | S::AVCOL_SPC_BT2020_CL => Matrix::Bt2020Ncl,
            S::AVCOL_SPC_SMPTE240M => Matrix::Smpte240,
            S::AVCOL_SPC_RGB => Matrix::Rgb,
            _ => {
                if height >= 720 { Matrix::Bt709 } else { Matrix::Bt601 }
            }
        };
        let transfer = match (*par).color_trc {
            T::AVCOL_TRC_SMPTE2084 => Transfer::Pq,
            T::AVCOL_TRC_ARIB_STD_B67 => Transfer::Hlg,
            T::AVCOL_TRC_LINEAR => Transfer::Linear,
            _ => Transfer::Sdr,
        };
        let primaries = match (*par).color_primaries {
            P::AVCOL_PRI_BT709 => Primaries::Bt709,
            P::AVCOL_PRI_BT2020 => Primaries::Bt2020,
            P::AVCOL_PRI_BT470BG | P::AVCOL_PRI_SMPTE170M | P::AVCOL_PRI_SMPTE240M => Primaries::Bt601,
            P::AVCOL_PRI_UNSPECIFIED => {
                if matrix == Matrix::Bt2020Ncl { Primaries::Bt2020 } else { Primaries::Bt709 }
            }
            _ => Primaries::Other,
        };
        let full_range = (*par).color_range == ff::AVColorRange::AVCOL_RANGE_JPEG;
        ColorInfo { matrix, full_range, transfer, primaries }
    }
}
