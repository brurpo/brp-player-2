//! Audio playback: a worker decodes and resamples into a lock-free ring buffer,
//! the cpal callback plays it and keeps a sample-accurate clock.
//!
//! Times here are absolute stream seconds (pts * time base), so the caller adds the
//! video stream's start time when converting from frame times.

use crate::ffutil::{self, Input, Packet};
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use crossbeam_channel::{Receiver, Sender};
use ffmpeg_sys_next as ff;
use parking_lot::Mutex;
use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};
use std::sync::Arc;
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

/// Names of the available output devices (default first).
pub fn output_devices() -> Vec<String> {
    let host = cpal::default_host();
    let default = host.default_output_device().and_then(|d| d.description().ok()).map(|d| d.name().to_string());
    let mut names: Vec<String> = host
        .output_devices()
        .map(|it| it.filter_map(|d| d.description().ok().map(|x| x.name().to_string())).collect())
        .unwrap_or_default();
    if let Some(d) = default {
        names.retain(|n| *n != d);
        names.insert(0, d);
    }
    names
}

enum Cmd {
    /// Play from this absolute time (seconds).
    Start(f64),
    Stop,
}

#[derive(Default)]
struct Clock {
    /// Absolute time of the first sample after the last start.
    base: f64,
    /// Frames handed to the device before the latest callback.
    frames_before: u64,
    /// Frames handed to the device including the latest callback.
    frames: u64,
    at: Option<Instant>,
    latency: f64,
}

struct Shared {
    volume: AtomicU32,
    muted: AtomicBool,
    /// Callback plays ring samples (otherwise silence).
    active: AtomicBool,
    /// Callback must discard everything in the ring, then clear this.
    flush: AtomicBool,
    clock: Mutex<Clock>,
    cancel: AtomicBool,
}

pub struct Audio {
    shared: Arc<Shared>,
    tx: Option<Sender<Cmd>>,
    worker: Option<JoinHandle<()>>,
    _stream: cpal::Stream,
    pub device_name: String,
    pub rate: u32,
    pub channels: u16,
}

impl Audio {
    /// Opens the file's main audio stream on `device` (or the default device).
    /// Returns `Ok(None)` when the file has no audio.
    pub fn open(path: &str, device: Option<&str>) -> anyhow::Result<Option<Audio>> {
        let input = Input::open(path)?;
        let stream = unsafe {
            ff::av_find_best_stream(input.ctx, ff::AVMediaType::AVMEDIA_TYPE_AUDIO, -1, -1, std::ptr::null_mut(), 0)
        };
        if stream < 0 {
            return Ok(None);
        }
        input.discard_others(stream);
        let dec = unsafe { open_audio_decoder(input.stream(stream))? };

        let host = cpal::default_host();
        let dev = match device {
            Some(name) => host
                .output_devices()?
                .find(|d| d.description().is_ok_and(|x| x.name() == name))
                .or_else(|| host.default_output_device()),
            None => host.default_output_device(),
        }
        .ok_or_else(|| anyhow::anyhow!("no audio output device"))?;
        let device_name = dev.description().map(|d| d.name().to_string()).unwrap_or_default();
        let supported = dev.default_output_config()?;
        let config: cpal::StreamConfig = supported.config();
        let (rate, channels) = (config.sample_rate, config.channels);

        let shared = Arc::new(Shared {
            volume: AtomicU32::new(1.0f32.to_bits()),
            muted: AtomicBool::new(false),
            active: AtomicBool::new(false),
            flush: AtomicBool::new(false),
            clock: Mutex::new(Clock::default()),
            cancel: AtomicBool::new(false),
        });
        // ~0.5 s of audio buffered ahead.
        let (producer, consumer) = rtrb::RingBuffer::<f32>::new(rate as usize * channels as usize / 2);

        let stream_out = build_stream(&dev, &config, supported.sample_format(), consumer, shared.clone())?;
        stream_out.play()?;
        log::info!("audio: {device_name}, {rate} Hz, {channels} ch");

        let (tx, rx) = crossbeam_channel::unbounded();
        let mut w = AudioWorker {
            input,
            dec,
            stream,
            pkt: Packet::new(),
            frame: unsafe { ff::av_frame_alloc() },
            swr: std::ptr::null_mut(),
            swr_key: (0, 0, 0),
            out_rate: rate,
            out_channels: channels,
            producer,
            shared: shared.clone(),
            rx,
            trim_until: None,
            producing: false,
            pending: Vec::new(),
        };
        let worker = std::thread::Builder::new().name("brp-audio".into()).spawn(move || w.run())?;
        Ok(Some(Audio { shared, tx: Some(tx), worker: Some(worker), _stream: stream_out, device_name, rate, channels }))
    }

    /// Start playing from absolute time `secs`.
    pub fn start(&self, secs: f64) {
        if let Some(tx) = &self.tx {
            let _ = tx.send(Cmd::Start(secs));
        }
    }

    pub fn stop(&self) {
        self.shared.active.store(false, Ordering::Release);
        if let Some(tx) = &self.tx {
            let _ = tx.send(Cmd::Stop);
        }
    }

    pub fn set_volume(&self, v: f32) {
        self.shared.volume.store(v.clamp(0.0, 2.0).to_bits(), Ordering::Relaxed);
    }

    pub fn set_muted(&self, m: bool) {
        self.shared.muted.store(m, Ordering::Relaxed);
    }

    /// Absolute time currently heard, once samples are flowing.
    pub fn position(&self) -> Option<f64> {
        if !self.shared.active.load(Ordering::Acquire) {
            return None;
        }
        let c = self.shared.clock.lock();
        let at = c.at?;
        let elapsed = at.elapsed().as_secs_f64();
        let written = (c.frames - c.frames_before) as f64 / self.rate as f64;
        let into = (elapsed - c.latency).min(written + 0.05);
        Some(c.base + c.frames_before as f64 / self.rate as f64 + into)
    }
}

impl Drop for Audio {
    fn drop(&mut self) {
        self.shared.cancel.store(true, Ordering::Relaxed);
        self.tx = None;
        if let Some(w) = self.worker.take() {
            let _ = w.join();
        }
    }
}

fn build_stream(
    dev: &cpal::Device,
    config: &cpal::StreamConfig,
    format: cpal::SampleFormat,
    mut cons: rtrb::Consumer<f32>,
    shared: Arc<Shared>,
) -> anyhow::Result<cpal::Stream> {
    let channels = config.channels as usize;
    let err = |e: cpal::Error| log::warn!("audio stream: {e}");
    // Fills `out` (f32) from the ring; returns frames taken from the ring.
    let mut fill = move |out: &mut [f32], info: &cpal::OutputCallbackInfo| {
        if shared.flush.load(Ordering::Acquire) {
            let n = cons.slots();
            if let Ok(chunk) = cons.read_chunk(n) {
                chunk.commit_all();
            }
            shared.flush.store(false, Ordering::Release);
        }
        let mut taken = 0;
        if shared.active.load(Ordering::Acquire) {
            let avail = cons.slots().min(out.len());
            let avail = avail - avail % channels;
            if let Ok(chunk) = cons.read_chunk(avail) {
                let (a, b) = chunk.as_slices();
                out[..a.len()].copy_from_slice(a);
                out[a.len()..a.len() + b.len()].copy_from_slice(b);
                chunk.commit_all();
                taken = avail;
            }
        }
        let vol = if shared.muted.load(Ordering::Relaxed) { 0.0 } else { f32::from_bits(shared.volume.load(Ordering::Relaxed)) };
        for s in &mut out[..taken] {
            *s *= vol;
        }
        out[taken..].fill(0.0);
        if taken > 0 {
            let ts = info.timestamp();
            let latency = ts.playback.duration_since(ts.callback).as_secs_f64();
            let mut c = shared.clock.lock();
            c.frames_before = c.frames;
            c.frames += (taken / channels) as u64;
            c.at = Some(Instant::now());
            c.latency = latency;
        }
    };
    let stream = match format {
        cpal::SampleFormat::F32 => dev.build_output_stream::<f32, _, _>(config.clone(), move |d, i| fill(d, i), err, None)?,
        cpal::SampleFormat::I16 => {
            let mut tmp = Vec::new();
            dev.build_output_stream::<i16, _, _>(
                config.clone(),
                move |d: &mut [i16], i| {
                    tmp.resize(d.len(), 0.0);
                    fill(&mut tmp, i);
                    for (o, s) in d.iter_mut().zip(&tmp) {
                        *o = (s.clamp(-1.0, 1.0) * i16::MAX as f32) as i16;
                    }
                },
                err,
                None,
            )?
        }
        other => anyhow::bail!("unsupported audio sample format {other:?}"),
    };
    Ok(stream)
}

unsafe fn open_audio_decoder(st: *mut ff::AVStream) -> anyhow::Result<*mut ff::AVCodecContext> {
    unsafe {
        let par = (*st).codecpar;
        let codec = ff::avcodec_find_decoder((*par).codec_id);
        if codec.is_null() {
            anyhow::bail!("no decoder for the audio codec");
        }
        let mut ctx = ff::avcodec_alloc_context3(codec);
        ffutil::check(ff::avcodec_parameters_to_context(ctx, par), "audio decoder params")?;
        (*ctx).pkt_timebase = (*st).time_base;
        let r = ff::avcodec_open2(ctx, codec, std::ptr::null_mut());
        if r < 0 {
            ff::avcodec_free_context(&mut ctx);
            ffutil::check(r, "open audio decoder")?;
        }
        Ok(ctx)
    }
}

struct AudioWorker {
    input: Input,
    dec: *mut ff::AVCodecContext,
    stream: i32,
    pkt: Packet,
    frame: *mut ff::AVFrame,
    swr: *mut ff::SwrContext,
    swr_key: (i32, i32, i32),
    out_rate: u32,
    out_channels: u16,
    producer: rtrb::Producer<f32>,
    shared: Arc<Shared>,
    rx: Receiver<Cmd>,
    /// After a seek: drop samples before this absolute time.
    trim_until: Option<f64>,
    producing: bool,
    /// Converted samples not yet pushed (ring was full).
    pending: Vec<f32>,
}

unsafe impl Send for AudioWorker {}

impl AudioWorker {
    fn run(&mut self) {
        loop {
            if self.shared.cancel.load(Ordering::Relaxed) {
                return;
            }
            let cmd = if self.producing {
                match self.rx.try_recv() {
                    Ok(c) => Some(c),
                    Err(crossbeam_channel::TryRecvError::Empty) => None,
                    Err(_) => return,
                }
            } else {
                match self.rx.recv_timeout(Duration::from_millis(200)) {
                    Ok(c) => Some(c),
                    Err(crossbeam_channel::RecvTimeoutError::Timeout) => None,
                    Err(_) => return,
                }
            };
            match cmd {
                Some(Cmd::Start(t)) => self.start(t),
                Some(Cmd::Stop) => {
                    self.producing = false;
                    self.flush_ring();
                }
                None if self.producing => {
                    if let Err(e) = self.produce() {
                        log::warn!("audio decode: {e}");
                        self.producing = false;
                    }
                }
                None => {}
            }
        }
    }

    /// Ask the callback to empty the ring and wait for it (bounded).
    fn flush_ring(&mut self) {
        self.pending.clear();
        self.shared.flush.store(true, Ordering::Release);
        let t = Instant::now();
        while self.shared.flush.load(Ordering::Acquire) && t.elapsed() < Duration::from_millis(200) {
            std::thread::sleep(Duration::from_millis(1));
        }
    }

    fn start(&mut self, t: f64) {
        self.shared.active.store(false, Ordering::Release);
        self.flush_ring();
        unsafe {
            let tb = (*self.input.stream(self.stream)).time_base;
            let ts = (t / ffutil::q2d(tb)).floor() as i64;
            ff::av_seek_frame(self.input.ctx, self.stream, ts, ff::AVSEEK_FLAG_BACKWARD);
            ff::avcodec_flush_buffers(self.dec);
            if !self.swr.is_null() {
                ff::swr_init(self.swr); // drop resampler history
            }
        }
        *self.shared.clock.lock() = Clock { base: t, ..Clock::default() };
        self.trim_until = Some(t);
        self.producing = true;
        self.shared.active.store(true, Ordering::Release);
    }

    /// Decode one frame and push it (or wait while the ring is full).
    fn produce(&mut self) -> anyhow::Result<()> {
        if !self.pending.is_empty() {
            self.push_pending();
            return Ok(());
        }
        unsafe {
            loop {
                let r = ff::avcodec_receive_frame(self.dec, self.frame);
                if r == 0 {
                    break;
                }
                if r == ffutil::AVERROR_EOF {
                    self.producing = false;
                    return Ok(());
                }
                if r != ffutil::AVERROR_EAGAIN {
                    ffutil::check(r, "receive audio")?;
                }
                let r = ff::av_read_frame(self.input.ctx, self.pkt.0);
                if r < 0 {
                    ff::avcodec_send_packet(self.dec, std::ptr::null());
                    continue;
                }
                if (*self.pkt.0).stream_index == self.stream {
                    ff::avcodec_send_packet(self.dec, self.pkt.0);
                }
                ff::av_packet_unref(self.pkt.0);
            }
            let samples = self.convert()?;
            ff::av_frame_unref(self.frame);
            self.pending = samples;
        }
        self.push_pending();
        Ok(())
    }

    fn push_pending(&mut self) {
        let free = self.producer.slots();
        if free == 0 {
            std::thread::sleep(Duration::from_millis(4));
            return;
        }
        let n = free.min(self.pending.len());
        if let Ok(mut chunk) = self.producer.write_chunk(n) {
            let (a, b) = chunk.as_mut_slices();
            let (x, y) = self.pending[..n].split_at(a.len());
            a.copy_from_slice(x);
            b.copy_from_slice(y);
            chunk.commit_all();
            self.pending.drain(..n);
        }
    }

    /// Resample the decoded frame to interleaved f32 at the device rate, trimming
    /// samples before the seek target.
    unsafe fn convert(&mut self) -> anyhow::Result<Vec<f32>> {
        unsafe {
            let f = &*self.frame;
            let key = (f.format, f.sample_rate, f.ch_layout.nb_channels);
            if self.swr.is_null() || self.swr_key != key {
                ff::swr_free(&mut self.swr);
                let mut out_layout: ff::AVChannelLayout = std::mem::zeroed();
                ff::av_channel_layout_default(&mut out_layout, self.out_channels as i32);
                ffutil::check(
                    ff::swr_alloc_set_opts2(
                        &mut self.swr,
                        &out_layout,
                        ff::AVSampleFormat::AV_SAMPLE_FMT_FLT,
                        self.out_rate as i32,
                        &f.ch_layout,
                        std::mem::transmute::<i32, ff::AVSampleFormat>(f.format),
                        f.sample_rate,
                        0,
                        std::ptr::null_mut(),
                    ),
                    "audio resampler",
                )?;
                ffutil::check(ff::swr_init(self.swr), "audio resampler init")?;
                self.swr_key = key;
            }
            let ch = self.out_channels as usize;
            let cap = (ff::swr_get_delay(self.swr, f.sample_rate as i64)
                + f.nb_samples as i64 * self.out_rate as i64 / f.sample_rate.max(1) as i64
                + 256) as usize;
            let mut out = vec![0f32; cap * ch];
            let mut out_ptr = out.as_mut_ptr() as *mut u8;
            let got = ff::swr_convert(
                self.swr,
                &mut out_ptr,
                cap as i32,
                f.extended_data as *const *const u8,
                f.nb_samples,
            );
            ffutil::check(got, "resample")?;
            out.truncate(got as usize * ch);

            if let Some(target) = self.trim_until {
                let mut pts = f.best_effort_timestamp;
                if pts == ffutil::AV_NOPTS_VALUE {
                    pts = f.pts;
                }
                if pts != ffutil::AV_NOPTS_VALUE {
                    let tb = (*self.input.stream(self.stream)).time_base;
                    let start = pts as f64 * ffutil::q2d(tb);
                    let skip = ((target - start) * self.out_rate as f64).round();
                    if skip >= got as f64 {
                        return Ok(Vec::new()); // whole frame before the target
                    }
                    if skip > 0.0 {
                        out.drain(..skip as usize * ch);
                    } else if skip < 0.0 {
                        // Audio starts after the target: pad with silence so the clock stays exact.
                        let pad = ((-skip) as usize).min(self.out_rate as usize) * ch;
                        out.splice(0..0, std::iter::repeat_n(0.0, pad));
                    }
                }
                self.trim_until = None;
            }
            Ok(out)
        }
    }
}

impl Drop for AudioWorker {
    fn drop(&mut self) {
        unsafe {
            ff::swr_free(&mut self.swr);
            ff::av_frame_free(&mut self.frame);
            ff::avcodec_free_context(&mut self.dec);
        }
    }
}

#[allow(dead_code)]
fn _assert_send() {
    fn is_send<T: Send>() {}
    is_send::<Audio>();
}
