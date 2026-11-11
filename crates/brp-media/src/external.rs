//! Encoders that run in a separate `ffmpeg.exe` (x264 and x265).
//!
//! The app links only FFmpeg's LGPL libraries. x264 and x265 are GPL and exist only
//! in FFmpeg's GPL build, so they run in that build's own `ffmpeg.exe`, a separate
//! program: pictures go to it raw through its standard input, it encodes them and
//! sends the packets back as a NUT stream on its standard output. Our writer muxes
//! those packets like any other (audio, metadata and colour tags stay ours).
//!
//! Where the GPL `ffmpeg.exe` is: `BRP_FFMPEG_GPL`, else `<exe dir>/ffmpeg-gpl/ffmpeg.exe`
//! (the app downloads it there).

use crate::ffutil::{self, AVERROR_EOF};
use anyhow::{bail, Result};
use crossbeam_channel::{Receiver, Sender};
use ffmpeg_sys_next as ff;
use std::io::{Read, Write};
use std::path::PathBuf;
use std::process::{Child, ChildStdin, ChildStdout, Command, Stdio};
use std::sync::{Arc, Mutex, OnceLock};

/// Encoders run through the external program.
pub const ENCODERS: [&str; 2] = ["libx264", "libx265"];

/// The GPL `ffmpeg.exe`, if present.
pub fn program() -> Option<PathBuf> {
    static P: OnceLock<Option<PathBuf>> = OnceLock::new();
    P.get_or_init(|| {
        if let Some(p) = std::env::var_os("BRP_FFMPEG_GPL").map(PathBuf::from) {
            return p.is_file().then_some(p);
        }
        let exe = std::env::current_exe().ok()?;
        let p = exe.parent()?.join("ffmpeg-gpl").join(if cfg!(windows) { "ffmpeg.exe" } else { "ffmpeg" });
        p.is_file().then_some(p)
    })
    .clone()
}

/// `name` is encoded by the external program here (it is one of [`ENCODERS`], the
/// linked FFmpeg lacks it and the program is present).
pub fn handles(name: &str) -> bool {
    ENCODERS.contains(&name) && unsafe { ff::avcodec_find_encoder_by_name(ffutil::cstr(name).as_ptr()).is_null() } && program().is_some()
}

/// Codec each external encoder produces, and the pixel formats it takes.
pub fn info(name: &str) -> Option<(&'static str, &'static [&'static str])> {
    match name {
        "libx264" => Some(("h264", &["yuv420p", "yuvj420p", "yuv422p", "yuvj422p", "yuv444p", "yuvj444p", "nv12", "nv16", "nv21", "yuv420p10le", "yuv422p10le", "yuv444p10le", "nv20le", "gray", "gray10le"])),
        "libx265" => Some(("hevc", &["yuv420p", "yuvj420p", "yuv422p", "yuvj422p", "yuv444p", "yuvj444p", "gbrp", "yuv420p10le", "yuv422p10le", "yuv444p10le", "gbrp10le", "yuv420p12le", "yuv422p12le", "yuv444p12le", "gbrp12le", "gray", "gray10le", "gray12le"])),
        _ => None,
    }
}

/// Raw pointers handed between threads (each owned by exactly one side at a time).
pub(crate) struct Params(pub *mut ff::AVCodecParameters, pub ff::AVRational);
unsafe impl Send for Params {}
impl Drop for Params {
    fn drop(&mut self) {
        unsafe { ff::avcodec_parameters_free(&mut self.0) };
    }
}
pub(crate) struct Pkt(pub *mut ff::AVPacket);
unsafe impl Send for Pkt {}
impl Drop for Pkt {
    fn drop(&mut self) {
        unsafe { ff::av_packet_free(&mut self.0) };
    }
}

pub(crate) enum Msg {
    Params(Params),
    Packet(Pkt),
    End(Result<()>),
}

/// What the external program needs to know.
pub(crate) struct Job<'a> {
    pub encoder: &'a str,
    pub pix_fmt: &'a str,
    pub width: u32,
    pub height: u32,
    pub rate: (i32, i32),
    pub sar: (i32, i32),
    /// Codec options as name/value (`-name value` on its command line).
    pub options: Vec<(String, String)>,
    pub gop: Option<i32>,
    /// Some(top field first): interlaced coding.
    pub field_order: Option<bool>,
    /// Colour tags: primaries, transfer, matrix, range (FFmpeg's names).
    pub color: Option<[String; 4]>,
    /// The container needs the codec headers out of band (MP4, MOV, MKV).
    pub global_header: bool,
}

pub(crate) struct External {
    child: Child,
    stdin: Option<ChildStdin>,
    pub rx: Receiver<Msg>,
    stderr: Arc<Mutex<String>>,
    pub buf: Vec<u8>,
}

impl External {
    pub fn start(job: &Job) -> Result<External> {
        let Some(prog) = program() else { bail!("{} needs the GPL ffmpeg.exe (ffmpeg-gpl folder), which is missing", job.encoder) };
        let mut a: Vec<String> = ["-hide_banner", "-nostdin", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", job.pix_fmt].iter().map(|s| s.to_string()).collect();
        a.extend(["-s".into(), format!("{}x{}", job.width, job.height), "-framerate".into(), format!("{}/{}", job.rate.0, job.rate.1), "-i".into(), "pipe:0".into()]);
        a.extend(["-map".into(), "0:v".into(), "-c:v".into(), job.encoder.into(), "-pix_fmt".into(), job.pix_fmt.into()]);
        a.extend(["-sar".into(), format!("{}/{}", job.sar.0.max(1), job.sar.1.max(1))]);
        if let Some(g) = job.gop {
            a.extend(["-g".into(), g.to_string()]);
        }
        if let Some(tff) = job.field_order {
            a.extend(["-flags".into(), "+ildct+ilme".into(), "-field_order".into(), if tff { "tt" } else { "bb" }.into(), "-top".into(), if tff { "1" } else { "0" }.into()]);
        }
        if job.global_header {
            a.extend(["-flags:v".into(), "+global_header".into()]);
        }
        if let Some([p, t, s, r]) = &job.color {
            a.extend(["-color_primaries".into(), p.clone(), "-color_trc".into(), t.clone(), "-colorspace".into(), s.clone(), "-color_range".into(), r.clone()]);
        }
        for (k, v) in &job.options {
            a.extend([format!("-{k}"), v.clone()]);
        }
        a.extend(["-f".into(), "nut".into(), "-flush_packets".into(), "1".into(), "pipe:1".into()]);
        log::info!("export: {} {}", prog.display(), a.join(" "));
        let mut cmd = Command::new(&prog);
        cmd.args(&a).stdin(Stdio::piped()).stdout(Stdio::piped()).stderr(Stdio::piped());
        #[cfg(windows)]
        {
            use std::os::windows::process::CommandExt;
            cmd.creation_flags(0x0800_0000); // CREATE_NO_WINDOW
        }
        let mut child = cmd.spawn().map_err(|e| anyhow::anyhow!("start {}: {e}", prog.display()))?;
        let stdin = child.stdin.take();
        let stdout = child.stdout.take().unwrap();
        let mut err = child.stderr.take().unwrap();
        let stderr = Arc::new(Mutex::new(String::new()));
        let se = stderr.clone();
        std::thread::Builder::new().name("ext-stderr".into()).spawn(move || {
            let mut s = String::new();
            let _ = err.read_to_string(&mut s);
            *se.lock().unwrap() = s;
        })?;
        let (tx, rx) = crossbeam_channel::bounded(256);
        std::thread::Builder::new().name("ext-demux".into()).spawn(move || {
            let r = unsafe { demux(stdout, &tx) };
            let _ = tx.send(Msg::End(r));
        })?;
        Ok(External { child, stdin, rx, stderr, buf: Vec::new() })
    }

    /// One raw picture (the encoder's pixel format, planes packed tightly).
    pub fn send(&mut self, data: &[u8]) -> Result<()> {
        let Some(s) = self.stdin.as_mut() else { bail!("encoder input closed") };
        if let Err(e) = s.write_all(data) {
            bail!("encoder stopped ({e}): {}", self.error_text());
        }
        Ok(())
    }

    /// No more pictures: the program flushes and ends.
    pub fn close_input(&mut self) {
        self.stdin = None;
    }

    pub fn error_text(&self) -> String {
        let s = self.stderr.lock().unwrap().trim().to_string();
        if s.is_empty() { "no message".into() } else { s }
    }

    /// Wait for the program; Err with its message if it failed.
    pub fn wait(&mut self) -> Result<()> {
        self.stdin = None;
        let st = self.child.wait()?;
        // Let the stderr reader finish its last read.
        std::thread::sleep(std::time::Duration::from_millis(20));
        if !st.success() {
            bail!("{}", self.error_text());
        }
        Ok(())
    }
}

impl Drop for External {
    fn drop(&mut self) {
        self.stdin = None;
        if let Ok(None) = self.child.try_wait() {
            let _ = self.child.kill();
        }
        let _ = self.child.wait();
    }
}

unsafe extern "C" fn read_cb(opaque: *mut std::ffi::c_void, buf: *mut u8, size: i32) -> i32 {
    let out = unsafe { &mut *(opaque as *mut ChildStdout) };
    let b = unsafe { std::slice::from_raw_parts_mut(buf, size.max(0) as usize) };
    loop {
        match out.read(b) {
            Ok(0) => return AVERROR_EOF,
            Ok(n) => return n as i32,
            Err(e) if e.kind() == std::io::ErrorKind::Interrupted => continue,
            Err(_) => return AVERROR_EOF,
        }
    }
}

/// Read the NUT stream from the program: its parameters, then every packet.
unsafe fn demux(stdout: ChildStdout, tx: &Sender<Msg>) -> Result<()> {
    unsafe {
        let mut opaque = Box::new(stdout);
        const SIZE: usize = 1 << 16;
        let buf = ff::av_malloc(SIZE) as *mut u8;
        let mut avio = ff::avio_alloc_context(buf, SIZE as i32, 0, &mut *opaque as *mut ChildStdout as *mut _, Some(read_cb), None, None);
        if avio.is_null() {
            bail!("avio");
        }
        let mut ic = ff::avformat_alloc_context();
        (*ic).pb = avio;
        (*ic).flags |= ff::AVFMT_FLAG_CUSTOM_IO as i32;
        let fmt = ff::av_find_input_format(c"nut".as_ptr());
        let r = ff::avformat_open_input(&mut ic, std::ptr::null(), fmt, std::ptr::null_mut());
        let result = (|| -> Result<()> {
            ffutil::check(r, "read encoder output")?;
            // NUT keeps only presentation times; analysing the stream tells libavformat
            // the reorder depth (B-frames), so it rebuilds the decode times.
            ffutil::check(ff::avformat_find_stream_info(ic, std::ptr::null_mut()), "analyse encoder output")?;
            if (*ic).nb_streams < 1 {
                bail!("encoder sent no stream");
            }
            let st = *(*ic).streams;
            let par = ff::avcodec_parameters_alloc();
            ffutil::check(ff::avcodec_parameters_copy(par, (*st).codecpar), "parameters")?;
            if tx.send(Msg::Params(Params(par, (*st).time_base))).is_err() {
                return Ok(());
            }
            loop {
                let pkt = ff::av_packet_alloc();
                let r = ff::av_read_frame(ic, pkt);
                if r < 0 {
                    let mut p = pkt;
                    ff::av_packet_free(&mut p);
                    if r == AVERROR_EOF {
                        return Ok(());
                    }
                    return ffutil::check(r, "read encoder output").map(|_| ());
                }
                if tx.send(Msg::Packet(Pkt(pkt))).is_err() {
                    return Ok(());
                }
            }
        })();
        if r >= 0 {
            ff::avformat_close_input(&mut ic);
        } else if !ic.is_null() {
            ff::avformat_free_context(ic);
        }
        if !avio.is_null() {
            ff::av_freep(&mut (*avio).buffer as *mut *mut u8 as *mut std::ffi::c_void);
            ff::avio_context_free(&mut avio);
        }
        drop(opaque);
        result
    }
}
