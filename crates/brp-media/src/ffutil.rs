//! Small helpers around the raw FFmpeg API.

use ffmpeg_sys_next as ff;
use std::ffi::{CStr, CString};

pub const AV_NOPTS_VALUE: i64 = i64::MIN;

const fn fferrtag(a: u8, b: u8, c: u8, d: u8) -> i32 {
    -((a as i32) | ((b as i32) << 8) | ((c as i32) << 16) | ((d as i32) << 24))
}
pub const AVERROR_EOF: i32 = fferrtag(b'E', b'O', b'F', b' ');
pub const AVERROR_EAGAIN: i32 = -ff::EAGAIN;

pub fn err_str(code: i32) -> String {
    let mut buf = [0 as std::ffi::c_char; 256];
    unsafe {
        ff::av_strerror(code, buf.as_mut_ptr(), buf.len());
        CStr::from_ptr(buf.as_ptr()).to_string_lossy().into_owned()
    }
}

pub fn check(code: i32, what: &str) -> anyhow::Result<i32> {
    if code < 0 {
        anyhow::bail!("{what}: {} ({code})", err_str(code));
    }
    Ok(code)
}

pub fn cstr(s: &str) -> CString {
    CString::new(s).unwrap_or_default()
}

pub unsafe fn opt_str(p: *const std::ffi::c_char) -> Option<String> {
    if p.is_null() {
        None
    } else {
        Some(unsafe { CStr::from_ptr(p) }.to_string_lossy().into_owned())
    }
}

pub fn q2d(q: ff::AVRational) -> f64 {
    if q.den == 0 { 0.0 } else { q.num as f64 / q.den as f64 }
}

/// Owned AVFormatContext opened for reading.
pub struct Input {
    pub ctx: *mut ff::AVFormatContext,
}

unsafe impl Send for Input {}

impl Input {
    pub fn open(path: &str) -> anyhow::Result<Self> {
        let cpath = cstr(path);
        let mut ctx: *mut ff::AVFormatContext = std::ptr::null_mut();
        unsafe {
            check(
                ff::avformat_open_input(&mut ctx, cpath.as_ptr(), std::ptr::null(), std::ptr::null_mut()),
                "open input",
            )?;
            let input = Input { ctx };
            check(ff::avformat_find_stream_info(ctx, std::ptr::null_mut()), "find stream info")?;
            Ok(input)
        }
    }

    pub fn stream(&self, index: i32) -> *mut ff::AVStream {
        unsafe { *(*self.ctx).streams.add(index as usize) }
    }

    pub fn nb_streams(&self) -> u32 {
        unsafe { (*self.ctx).nb_streams }
    }

    /// Discard every stream except `keep`, so the demuxer skips their packets where it can.
    pub fn discard_others(&self, keep: i32) {
        for i in 0..self.nb_streams() as i32 {
            if i != keep {
                unsafe { (*self.stream(i)).discard = ff::AVDiscard::AVDISCARD_ALL };
            }
        }
    }
}

impl Drop for Input {
    fn drop(&mut self) {
        unsafe { ff::avformat_close_input(&mut self.ctx) };
    }
}

pub struct Packet(pub *mut ff::AVPacket);
unsafe impl Send for Packet {}
impl Packet {
    pub fn new() -> Self {
        Packet(unsafe { ff::av_packet_alloc() })
    }
}
impl Drop for Packet {
    fn drop(&mut self) {
        unsafe { ff::av_packet_free(&mut self.0) };
    }
}
