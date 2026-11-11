//! A media source: a decode worker with a frame-exact seek and a decoded-frame cache.
//!
//! The UI asks for frame N with [`Source::request`] and reads frames back with
//! [`Source::get`]. The worker decides what to decode: the requested frame first,
//! then frames ahead of (or behind, when playing backwards) the playhead.
//!
//! Seeking: find the latest keyframe at or before frame N, seek there, decode
//! forward and keep every frame on the way. If the first frame after the seek is
//! already past N (open GOP, demuxer quirks), retry from an earlier keyframe.

use crate::ffutil::{self, Input, Packet};
use crate::frame::{classify, Converter, VideoFrame};
use crate::gpudec::{GpuDecoder, GpuPicture, StreamDesc};
use crate::index::{self, FrameIndex};
use crate::info::{self, MediaInfo};
use crossbeam_channel::{Receiver, Sender};
use ffmpeg_sys_next as ff;
use parking_lot::{Mutex, RwLock};
use std::collections::{HashMap, HashSet};
use std::sync::atomic::{AtomicBool, AtomicU8, Ordering};
use std::sync::Arc;
use std::thread::JoinHandle;
use std::time::Duration;

/// What the viewer wants right now.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Request {
    pub frame: i64,
    pub playing: bool,
    /// +1 forward, -1 backward.
    pub dir: i8,
    /// Frames to decode ahead while playing forward (0 = default). Background effects
    /// work ahead of the playhead and need more.
    pub ahead: i64,
    /// While paused, keep decoding ahead of the playhead in the background, filling
    /// the cache (not while scrubbing: the playhead moves too fast for it to help).
    pub fill: bool,
}

enum Msg {
    Request(Request),
    IndexReady,
}

struct Cache {
    frames: HashMap<i64, Arc<VideoFrame>>,
    bytes: usize,
}

struct Shared {
    cache: Mutex<Cache>,
    index: RwLock<Option<Arc<FrameIndex>>>,
    /// 0 unknown, 1 progressive, 2 interlaced (from decoded frame flags).
    interlaced: AtomicU8,
    /// Content-based interlace votes (idet).
    votes: Mutex<crate::detect::Votes>,
    cancel: AtomicBool,
    /// Cache limit in bytes for this source (0: the global budget).
    limit: std::sync::atomic::AtomicUsize,
    /// Which decoder makes the pictures ("FFmpeg", or the GPU decoder's name).
    decoder: Mutex<String>,
    /// Frames whose compressed packet is kept in RAM (a copy of the worker's keys).
    packet_frames: Mutex<HashSet<i64>>,
}

/// Where a source's frames are cached (frame numbers, unsorted).
#[derive(Default, Clone)]
pub struct CacheMap {
    /// Decoded, ready to show (in video memory for GPU-decoded files).
    pub decoded: Vec<i64>,
    /// In system RAM: compressed packets kept for re-decoding, and decoded frames
    /// that live in RAM (FFmpeg's).
    pub ram: Vec<i64>,
}

pub struct Source {
    pub info: MediaInfo,
    shared: Arc<Shared>,
    tx: Option<Sender<Msg>>,
    worker: Option<JoinHandle<()>>,
    indexer: Option<JoinHandle<()>>,
}

impl Source {
    pub fn open(path: &str) -> anyhow::Result<Source> {
        Self::open_with(path, true)
    }

    /// Like `open`; `gpu: false` always decodes with FFmpeg (reference decodes).
    pub fn open_with(path: &str, gpu: bool) -> anyhow::Result<Source> {
        // EXR files and sequences: our own reader (layers; FFmpeg reads single-layer
        // files only).
        if crate::exr_seq::is_exr(path) {
            return Self::open_exr(path, None);
        }
        // A numbered still image: the whole sequence it belongs to (a single image, or
        // an FFmpeg pattern with %, opens as before).
        let p = std::path::Path::new(path);
        if !path.contains('%') && p.is_file() && crate::sequence::is_image(p) {
            let (files, _) = crate::sequence::files(p, crate::sequence::is_image)?;
            if files.len() > 1 {
                return Self::open_image_sequence(path, files);
            }
        }
        let input = Input::open(path)?;
        let vs = unsafe {
            ff::av_find_best_stream(input.ctx, ff::AVMediaType::AVMEDIA_TYPE_VIDEO, -1, -1, std::ptr::null_mut(), 0)
        };
        if vs < 0 {
            anyhow::bail!("no video stream");
        }
        let info = unsafe { info::read_info(path, &input, vs) };
        input.discard_others(vs);
        let dec = unsafe { open_decoder(input.stream(vs))? };

        let shared = Arc::new(Shared {
            cache: Mutex::new(Cache { frames: HashMap::new(), bytes: 0 }),
            index: RwLock::new(None),
            interlaced: AtomicU8::new(0),
            votes: Mutex::new(Default::default()),
            cancel: AtomicBool::new(false),
            limit: 0.into(),
            decoder: Mutex::new("FFmpeg".into()),
            packet_frames: Mutex::new(HashSet::new()),
        });
        let desc = unsafe { crate::gpudec::stream_desc(input.stream(vs)) };
        let (tx, rx) = crossbeam_channel::unbounded();

        let indexer = {
            let shared = shared.clone();
            let tx = tx.clone();
            let path = path.to_string();
            std::thread::Builder::new().name("brp-index".into()).spawn(move || {
                let t = std::time::Instant::now();
                match index::build(&path, vs, &shared.cancel) {
                    Ok(ix) => {
                        log::info!("index: {} frames, {} keyframes in {:?}", ix.len(), ix.key_pts.len(), t.elapsed());
                        *shared.index.write() = Some(Arc::new(ix));
                        let _ = tx.send(Msg::IndexReady);
                    }
                    Err(e) => log::warn!("index failed: {e}"),
                }
            })?
        };

        let worker = {
            let mut w = Worker {
                input,
                dec,
                stream: vs,
                pkt: Packet::new(),
                frame: unsafe { ff::av_frame_alloc() },
                shared: shared.clone(),
                rx,
                info: info.clone(),
                conv: Converter::new(),
                req: Request { frame: 0, playing: false, dir: 1, ahead: 0, fill: false },
                index: None,
                last: None,
                last_pts: None,
                est_origin: None,
                draining: false,
                eof: false,
                seek: None,
                unreachable: HashSet::new(),
                end_frame: None,
                analyzed: HashSet::new(),
                desc,
                gpu: if gpu { GpuState::Untried } else { GpuState::Off },
                fallback: std::ptr::null_mut(),
                refused: 0,
                packets: HashMap::new(),
                packet_bytes: 0,
                gpu_frame_bytes: 0,
            };
            std::thread::Builder::new().name("brp-decode".into()).spawn(move || w.run())?
        };

        Ok(Source { info, shared, tx: Some(tx), worker: Some(worker), indexer: Some(indexer) })
    }

    pub fn request(&self, r: Request) {
        if let Some(tx) = &self.tx {
            let _ = tx.send(Msg::Request(r));
        }
    }

    /// The exact frame, if decoded.
    pub fn get(&self, frame: i64) -> Option<Arc<VideoFrame>> {
        self.shared.cache.lock().frames.get(&frame).cloned()
    }

    /// The closest decoded frame (used to show something while the exact frame decodes).
    pub fn nearest(&self, frame: i64) -> Option<Arc<VideoFrame>> {
        let c = self.shared.cache.lock();
        c.frames.values().min_by_key(|f| (f.index() - frame).abs()).cloned()
    }

    pub fn index_ready(&self) -> bool {
        self.shared.index.read().is_some()
    }

    pub fn frame_count(&self) -> i64 {
        match &*self.shared.index.read() {
            Some(ix) if !ix.is_empty() => ix.len() as i64,
            _ => self.info.frame_count_hint.max(1),
        }
    }

    /// Seconds from the start for a frame number (exact once the index is ready).
    pub fn frame_time(&self, frame: i64) -> f64 {
        match &*self.shared.index.read() {
            Some(ix) => ix.pts_of(frame).map(|p| self.info.pts_to_secs(p)).unwrap_or(0.0),
            None => frame as f64 / self.info.frame_rate,
        }
    }

    /// Time at which `frame` stops being shown (start of the next frame, or one
    /// nominal frame duration for the last frame).
    pub fn frame_end_time(&self, frame: i64) -> f64 {
        let n = self.frame_count();
        if frame + 1 < n {
            self.frame_time(frame + 1)
        } else {
            self.frame_time(frame) + 1.0 / self.info.frame_rate.max(1.0)
        }
    }

    /// Frame on screen at `secs` from the start (the last frame starting at or before it).
    pub fn frame_at_time(&self, secs: f64) -> i64 {
        match &*self.shared.index.read() {
            Some(ix) if !ix.is_empty() => {
                let tb = self.info.time_base;
                let pts = self.info.start_pts + (secs * tb.1 as f64 / tb.0.max(1) as f64).floor() as i64;
                (ix.pts.partition_point(|&p| p <= pts) as i64 - 1).max(0)
            }
            _ => (secs * self.info.frame_rate).floor() as i64,
        }
    }

    pub fn interlaced(&self) -> Option<bool> {
        match self.shared.interlaced.load(Ordering::Relaxed) {
            1 => Some(false),
            2 => Some(true),
            _ => None,
        }
    }

    /// Interlace votes from picture content, and the decision once confident:
    /// (interlaced, top field first).
    pub fn field_detection(&self) -> (crate::detect::Votes, Option<(bool, bool)>) {
        let v = *self.shared.votes.lock();
        (v, v.decision())
    }

    /// Keep at most `bytes` of decoded frames for this source (0: the global
    /// budget). Export uses a small window instead of the viewer's large cache.
    pub fn set_cache_limit(&self, bytes: usize) {
        self.shared.limit.store(bytes, Ordering::Relaxed);
    }

    pub fn cached_frames(&self) -> Vec<i64> {
        self.shared.cache.lock().frames.keys().copied().collect()
    }

    /// Decoded frames, and what is held in system RAM (for the timeline's cache bars).
    pub fn cache_map(&self) -> CacheMap {
        let mut m = CacheMap::default();
        {
            let c = self.shared.cache.lock();
            m.decoded.reserve(c.frames.len());
            for (&i, f) in &c.frames {
                m.decoded.push(i);
                if !f.on_gpu() {
                    m.ram.push(i);
                }
            }
        }
        m.ram.extend(self.shared.packet_frames.lock().iter().copied());
        m
    }

    /// The decoder making the pictures: "FFmpeg" or a GPU decoder's name.
    pub fn decoder_name(&self) -> String {
        self.shared.decoder.lock().clone()
    }
}

impl Drop for Source {
    fn drop(&mut self) {
        self.shared.cancel.store(true, Ordering::Relaxed);
        self.tx = None; // closes the channel so the worker exits
        if let Some(w) = self.worker.take() {
            let _ = w.join();
        }
        if let Some(i) = self.indexer.take() {
            let _ = i.join();
        }
    }
}

unsafe fn open_decoder(st: *mut ff::AVStream) -> anyhow::Result<*mut ff::AVCodecContext> {
    unsafe { open_decoder_threads(st, (ff::FF_THREAD_FRAME | ff::FF_THREAD_SLICE) as i32) }
}

/// FFmpeg decoder with the given threading (`FF_THREAD_SLICE` alone: no frame delay,
/// one picture out per packet of an intra codec).
unsafe fn open_decoder_threads(st: *mut ff::AVStream, thread_type: i32) -> anyhow::Result<*mut ff::AVCodecContext> {
    unsafe {
        let par = (*st).codecpar;
        let codec = ff::avcodec_find_decoder((*par).codec_id);
        if codec.is_null() {
            anyhow::bail!("no decoder for this codec");
        }
        let mut ctx = ff::avcodec_alloc_context3(codec);
        ffutil::check(ff::avcodec_parameters_to_context(ctx, par), "decoder params")?;
        (*ctx).pkt_timebase = (*st).time_base;
        (*ctx).thread_count = std::env::var("BRP_DEC_THREADS").ok().and_then(|s| s.parse().ok()).unwrap_or(0);
        (*ctx).thread_type = thread_type;
        let r = ff::avcodec_open2(ctx, codec, std::ptr::null_mut());
        if r < 0 {
            ff::avcodec_free_context(&mut ctx);
            ffutil::check(r, "open decoder")?;
        }
        Ok(ctx)
    }
}

struct SeekState {
    need: i64,
    attempt: usize,
}

struct Worker {
    input: Input,
    dec: *mut ff::AVCodecContext,
    stream: i32,
    pkt: Packet,
    frame: *mut ff::AVFrame,
    shared: Arc<Shared>,
    rx: Receiver<Msg>,
    info: MediaInfo,
    conv: Converter,
    req: Request,
    index: Option<Arc<FrameIndex>>,
    /// Frame number of the last decoded frame since the last seek.
    last: Option<i64>,
    last_pts: Option<i64>,
    /// Timestamp treated as frame 0 while there is no index yet (first decoded frame).
    est_origin: Option<i64>,
    draining: bool,
    eof: bool,
    seek: Option<SeekState>,
    unreachable: HashSet<i64>,
    /// Number of frames, learned at end of stream when there is no index yet.
    end_frame: Option<i64>,
    /// Frame numbers already used for interlace detection.
    analyzed: HashSet<i64>,
    desc: StreamDesc,
    gpu: GpuState,
    /// FFmpeg decoder (slice threads only) for packets the GPU decoder refuses.
    fallback: *mut ff::AVCodecContext,
    /// Packets in a row the GPU decoder refused.
    refused: u32,
    /// Compressed packets of frames the GPU decoder made, by frame number (with their
    /// timestamps): frames that left GPU memory are decoded again from these, without
    /// seeking or reading the file.
    packets: HashMap<i64, (i64, Vec<u8>)>,
    packet_bytes: usize,
    /// Size of the last picture the GPU decoder made.
    gpu_frame_bytes: usize,
}

enum GpuState {
    /// No GPU decoders registered yet (asked again at the next seek).
    Untried,
    On(Box<dyn GpuDecoder>),
    /// FFmpeg decodes this stream.
    Off,
}

/// A decoded picture before it goes into the cache.
enum Raw {
    Av(*mut ff::AVFrame),
    /// The picture, its timestamp and its packet.
    Gpu(GpuPicture, i64, Vec<u8>),
}

unsafe impl Send for Worker {}

const AHEAD: i64 = 12;
const BEHIND_PLAYING: i64 = 24;
const MAX_SEEK_ATTEMPTS: usize = 6;
/// GPU-decoded frames may always keep this much around the playhead.
const VRAM_WINDOW_MIN: usize = 128 << 20;

impl Worker {
    fn run(&mut self) {
        loop {
            // Take the latest request; block only when there is nothing to decode.
            let need = self.next_needed();
            let msg = if need.is_some() {
                match self.rx.try_recv() {
                    Ok(m) => Some(m),
                    Err(crossbeam_channel::TryRecvError::Empty) => None,
                    Err(_) => return,
                }
            } else {
                match self.rx.recv_timeout(Duration::from_millis(100)) {
                    Ok(m) => Some(m),
                    Err(crossbeam_channel::RecvTimeoutError::Timeout) => None,
                    Err(_) => return,
                }
            };
            if self.shared.cancel.load(Ordering::Relaxed) {
                return;
            }
            if let Some(m) = msg {
                self.handle(m);
                while let Ok(m) = self.rx.try_recv() {
                    self.handle(m);
                }
                continue;
            }
            let Some(need) = need else { continue };
            if let Err(e) = self.produce(need) {
                log::warn!("decode error at frame {need}: {e}");
                self.unreachable.insert(need);
            }
        }
    }

    fn handle(&mut self, m: Msg) {
        match m {
            Msg::Request(r) => self.req = r,
            Msg::IndexReady => {
                self.index = self.shared.index.read().clone();
                // Re-number cached frames from their exact timestamps; keep decoder state.
                if let Some(ix) = self.index.clone() {
                    let mut c = self.shared.cache.lock();
                    let old: Vec<_> = c.frames.drain().map(|(_, f)| f).collect();
                    for f in old {
                        let i = ix.frame_of_pts(f.pts);
                        f.set_index(i);
                        c.frames.insert(i, f);
                    }
                    c.bytes = c.frames.values().map(|f| f.byte_size()).sum();
                    drop(c);
                    self.packets = std::mem::take(&mut self.packets).into_values().map(|(pts, p)| (ix.frame_of_pts(pts), (pts, p))).collect();
                    *self.shared.packet_frames.lock() = self.packets.keys().copied().collect();
                    if let Some(l) = self.last {
                        let _ = l;
                        self.last = self.last_pts.map(|p| ix.frame_of_pts(p));
                    }
                }
                self.unreachable.clear();
                self.end_frame = None;
            }
        }
    }

    fn frame_count(&self) -> i64 {
        if let Some(ix) = &self.index {
            return ix.len() as i64;
        }
        self.end_frame.unwrap_or(i64::MAX)
    }

    fn pts_of(&self, frame: i64) -> i64 {
        match &self.index {
            Some(ix) => ix.pts_of(frame).unwrap_or(i64::MAX),
            None => {
                let tb = self.info.time_base;
                self.est_origin.unwrap_or(self.info.start_pts) + (frame as f64 / self.info.frame_rate * tb.1 as f64 / tb.0.max(1) as f64).round() as i64
            }
        }
    }

    fn frame_of_pts(&self, pts: i64) -> i64 {
        match &self.index {
            Some(ix) => ix.frame_of_pts(pts),
            None => {
                let tb = self.info.time_base;
                ((pts - self.est_origin.unwrap_or(self.info.start_pts)) as f64 * tb.0 as f64 / tb.1.max(1) as f64 * self.info.frame_rate).round() as i64
            }
        }
    }

    /// Most bytes of decoded frames to keep, holding `held` now.
    fn budget(&self, held: usize) -> usize {
        let b = self.ram_budget();
        // Pictures decoded on the GPU are kept in GPU memory: only a window around the
        // playhead (what playback and background effects read); the packets of the
        // others stay in system memory.
        match &self.gpu {
            GpuState::On(d) => {
                let window = (self.req.ahead.max(AHEAD) + BEHIND_PLAYING + 16) as usize * self.gpu_frame_bytes.max(1);
                b.min(d.vram_budget(held)).min(window.max(VRAM_WINDOW_MIN))
            }
            _ => b,
        }
    }

    fn ram_budget(&self) -> usize {
        match self.shared.limit.load(Ordering::Relaxed) {
            0 => crate::mem::cache_budget(),
            b => b,
        }
    }

    /// `need` decoded again from its kept packet (no seek). False if there is none or
    /// the decoder refused it.
    fn from_packet(&mut self, need: i64) -> bool {
        let GpuState::On(dec) = &mut self.gpu else { return false };
        let Some((pts, data)) = self.packets.get(&need) else { return false };
        let pts = *pts;
        match dec.decode(data) {
            Ok(pic) => {
                self.note_interlaced(pic.interlaced);
                let vf = Arc::new(VideoFrame::from_gpu(pic, need, pts));
                self.insert(need, vf);
                true
            }
            Err(e) => {
                log::debug!("frame {need} from its kept packet: {e:#}");
                false
            }
        }
    }

    fn keep_packet(&mut self, idx: i64, pts: i64, data: Vec<u8>) {
        self.packet_bytes += data.len();
        if let Some((_, old)) = self.packets.insert(idx, (pts, data)) {
            self.packet_bytes -= old.len();
        }
        self.shared.packet_frames.lock().insert(idx);
        // In system memory, under the same budget decoded frames have there (these
        // are several times smaller): drop those farthest from the playhead.
        let budget = self.ram_budget();
        if self.packet_bytes > budget {
            let t = self.req.frame;
            let mut keys: Vec<i64> = self.packets.keys().copied().collect();
            keys.sort_unstable_by_key(|&f| std::cmp::Reverse((f - t).abs()));
            for f in keys {
                if self.packet_bytes <= budget * 9 / 10 {
                    break;
                }
                if let Some((_, p)) = self.packets.remove(&f) {
                    self.packet_bytes -= p.len();
                    self.shared.packet_frames.lock().remove(&f);
                }
            }
        }
    }

    /// Use the GPU decoder if one is registered and takes this stream. Only at a
    /// seek: FFmpeg's frame threads may hold pictures in between.
    fn try_gpu(&mut self) {
        if !matches!(self.gpu, GpuState::Untried) {
            return;
        }
        let Some(f) = crate::gpudec::factory() else { return };
        self.gpu = match f.open(&self.desc) {
            Some(d) => {
                log::info!("decoding with {}", d.name());
                *self.shared.decoder.lock() = d.name().to_string();
                GpuState::On(d)
            }
            None => GpuState::Off,
        };
    }

    /// One packet the GPU decoder refused, through a slice-threaded FFmpeg decoder.
    unsafe fn decode_fallback(&mut self) -> anyhow::Result<*mut ff::AVFrame> {
        unsafe {
            if self.fallback.is_null() {
                self.fallback = open_decoder_threads(self.input.stream(self.stream), ff::FF_THREAD_SLICE as i32)?;
            }
            ffutil::check(ff::avcodec_send_packet(self.fallback, self.pkt.0), "send packet")?;
            let out = ff::av_frame_alloc();
            let r = ff::avcodec_receive_frame(self.fallback, out);
            if r < 0 {
                let mut o = out;
                ff::av_frame_free(&mut o);
                ffutil::check(r, "receive frame")?;
            }
            Ok(out)
        }
    }

    fn is_cached(&self, f: i64) -> bool {
        self.shared.cache.lock().frames.contains_key(&f)
    }

    /// Next frame to decode, in priority order around the playhead.
    fn next_needed(&self) -> Option<i64> {
        let n = self.frame_count();
        let t = self.req.frame.clamp(0, (n - 1).max(0));
        let mut order: Vec<i64> = Vec::with_capacity(40);
        order.push(t);
        if self.req.playing && self.req.dir >= 0 {
            order.extend(t + 1..=t + self.req.ahead.max(AHEAD));
        } else if self.req.playing {
            order.extend((t - BEHIND_PLAYING..t).rev());
        } else {
            order.extend(t + 1..=t + 3);
            order.extend((t - 8..t).rev());
        }
        let c = self.shared.cache.lock();
        let wanted = |f: &i64| *f >= 0 && *f < n && !c.frames.contains_key(f) && !self.unreachable.contains(f);
        if let Some(f) = order.into_iter().find(wanted) {
            return Some(f);
        }
        if self.req.playing || !self.req.fill {
            return None;
        }
        // Paused: the next frame ahead, decoded in the background, while the cache has
        // room, or while it is nearer the playhead than the farthest frame kept (the
        // eviction then drops that one: the cache settles on the frames around the
        // playhead, and decoding stops).
        let next = (t + 4..n).find(wanted)?;
        let budget = self.budget(c.bytes);
        if c.bytes < budget * 8 / 10 {
            return Some(next);
        }
        let farthest = c.frames.keys().map(|f| (f - t).abs()).max().unwrap_or(0);
        (next - t < farthest - 1).then_some(next)
    }

    /// Decode until `need` is in the cache (or found to be unreachable).
    fn produce(&mut self, need: i64) -> anyhow::Result<()> {
        // GPU decoders registered after this file opened (the player opens the file
        // while the GPU starts): restart cleanly at `need` with them.
        let late_gpu = matches!(self.gpu, GpuState::Untried) && self.last.is_some() && crate::gpudec::factory().is_some();
        if !late_gpu && self.from_packet(need) {
            return Ok(());
        }
        if late_gpu || self.should_seek(need) {
            self.seek_to(need, 0)?;
        }
        // Decode a bounded batch, then return so new requests are seen quickly.
        for _ in 0..8 {
            if self.is_cached(need) {
                return Ok(());
            }
            let Some(raw) = self.decode_next()? else {
                // End of stream before reaching `need`.
                if self.index.is_none() {
                    self.end_frame = Some(self.last.map_or(1, |l| l + 1));
                }
                if !self.is_cached(need) {
                    self.unreachable.insert(need);
                }
                return Ok(());
            };
            let got = self.store(raw)?;
            if let Some(s) = self.seek.take() {
                if got > s.need && s.attempt + 1 < MAX_SEEK_ATTEMPTS {
                    log::debug!("seek overshoot: wanted {}, got {got}, retrying earlier", s.need);
                    self.seek_to(s.need, s.attempt + 1)?;
                    continue;
                }
            }
            if got == need {
                // Made (the cache keeps the frame stored last, see `evict`).
                return Ok(());
            }
            if got > need && !self.is_cached(need) {
                // Passed it without producing it.
                log::debug!("frame {need} unreachable: passed (got {got})");
                self.unreachable.insert(need);
                return Ok(());
            }
        }
        Ok(())
    }

    fn should_seek(&self, need: i64) -> bool {
        let Some(last) = self.last else { return self.seek.is_none() };
        if self.eof || need <= last {
            return true;
        }
        let gap = need - last;
        match &self.index {
            Some(ix) => gap > 48 && ix.keyframe_between(self.pts_of(last), self.pts_of(need)),
            None => gap as f64 > self.info.frame_rate * 2.0,
        }
    }

    fn seek_to(&mut self, need: i64, attempt: usize) -> anyhow::Result<()> {
        self.try_gpu();
        if need > 0 && self.index.is_none() && self.est_origin.is_none() {
            self.find_origin()?;
        }
        let need_pts = self.pts_of(need);
        let ts = match &self.index {
            Some(ix) => ix
                .keyframes_before(need_pts)
                .nth(attempt)
                .or_else(|| ix.key_pts.first().copied())
                .unwrap_or(need_pts),
            None => {
                let tb = self.info.time_base;
                need_pts - (attempt as f64 * 2.0 * tb.1 as f64 / tb.0.max(1) as f64) as i64
            }
        };
        log::trace!("seek for frame {need} (attempt {attempt}) to ts {ts}");
        unsafe {
            let r = ff::av_seek_frame(self.input.ctx, self.stream, ts, ff::AVSEEK_FLAG_BACKWARD);
            if r < 0 {
                // Fall back to the start of the file.
                ffutil::check(
                    ff::av_seek_frame(self.input.ctx, self.stream, i64::MIN, ff::AVSEEK_FLAG_BACKWARD),
                    "seek",
                )?;
            }
            ff::avcodec_flush_buffers(self.dec);
        }
        self.last = None;
        self.draining = false;
        self.eof = false;
        self.seek = Some(SeekState { need, attempt });
        Ok(())
    }

    /// Without an index, frames are numbered from the file's first picture
    /// (`est_origin`). A seek made before any picture was decoded would otherwise take
    /// the first picture after the seek as frame 0: decode the file's first picture
    /// once (it goes into the cache as frame 0), then seek.
    fn find_origin(&mut self) -> anyhow::Result<()> {
        unsafe {
            ffutil::check(
                ff::av_seek_frame(self.input.ctx, self.stream, i64::MIN, ff::AVSEEK_FLAG_BACKWARD),
                "seek to start",
            )?;
            ff::avcodec_flush_buffers(self.dec);
        }
        self.last = None;
        self.draining = false;
        self.eof = false;
        // A few pictures at most: one without a timestamp sets no origin.
        for _ in 0..4 {
            let Some(raw) = self.decode_next()? else { break };
            self.store(raw)?;
            if self.est_origin.is_some() {
                break;
            }
        }
        log::debug!("frame 0 at pts {:?} (no index yet)", self.est_origin);
        Ok(())
    }

    /// Next decoded frame (owned), or None at end of stream.
    fn decode_next(&mut self) -> anyhow::Result<Option<Raw>> {
        unsafe {
            // GPU decoders take intra-only streams: one packet, one picture, no delay.
            while let GpuState::On(dec) = &mut self.gpu {
                if ff::av_read_frame(self.input.ctx, self.pkt.0) < 0 {
                    self.eof = true;
                    return Ok(None);
                }
                let p = &*self.pkt.0;
                if p.stream_index != self.stream || p.data.is_null() {
                    ff::av_packet_unref(self.pkt.0);
                    continue;
                }
                let pts = if p.pts != ffutil::AV_NOPTS_VALUE { p.pts } else { p.dts };
                let data = std::slice::from_raw_parts(p.data, p.size as usize);
                let out = match dec.decode(data) {
                    Ok(pic) => {
                        self.refused = 0;
                        Ok(Raw::Gpu(pic, pts, data.to_vec()))
                    }
                    Err(e) => {
                        let name = dec.name().to_string();
                        log::debug!("{name}: {e:#}; this packet goes to FFmpeg");
                        self.refused += 1;
                        if self.refused >= 8 {
                            // Not a stray packet: this stream is not for the GPU decoder
                            // (an intra codec: FFmpeg can take over at any packet).
                            log::info!("{name} refuses this stream ({e:#}); FFmpeg decodes it");
                            self.gpu = GpuState::Off;
                            *self.shared.decoder.lock() = "FFmpeg".into();
                        }
                        self.decode_fallback().map(Raw::Av)
                    }
                };
                ff::av_packet_unref(self.pkt.0);
                return out.map(Some);
            }
            loop {
                let r = ff::avcodec_receive_frame(self.dec, self.frame);
                if r == 0 {
                    let out = ff::av_frame_alloc();
                    ff::av_frame_move_ref(out, self.frame);
                    return Ok(Some(Raw::Av(out)));
                }
                if r == ffutil::AVERROR_EOF {
                    self.eof = true;
                    return Ok(None);
                }
                if r != ffutil::AVERROR_EAGAIN {
                    ffutil::check(r, "receive frame")?;
                }
                if self.draining {
                    self.eof = true;
                    return Ok(None);
                }
                let r = ff::av_read_frame(self.input.ctx, self.pkt.0);
                if r < 0 {
                    ff::avcodec_send_packet(self.dec, std::ptr::null());
                    self.draining = true;
                    continue;
                }
                if (*self.pkt.0).stream_index == self.stream {
                    let s = ff::avcodec_send_packet(self.dec, self.pkt.0);
                    if s < 0 && s != ffutil::AVERROR_EAGAIN {
                        log::debug!("send packet: {}", ffutil::err_str(s));
                    }
                }
                ff::av_packet_unref(self.pkt.0);
            }
        }
    }

    /// Put a decoded frame in the cache. Returns its frame number.
    fn store(&mut self, raw: Raw) -> anyhow::Result<i64> {
        unsafe {
            let raw = match raw {
                Raw::Av(f) => f,
                Raw::Gpu(pic, pts, packet) => {
                    let idx = self.number(pts);
                    self.note_interlaced(pic.interlaced);
                    let vf = Arc::new(VideoFrame::from_gpu(pic, idx, pts));
                    self.gpu_frame_bytes = vf.byte_size();
                    self.insert(idx, vf);
                    self.keep_packet(idx, pts, packet);
                    return Ok(idx);
                }
            };
            let mut pts = (*raw).best_effort_timestamp;
            if pts == ffutil::AV_NOPTS_VALUE {
                pts = (*raw).pts;
            }
            let idx = self.number(pts);
            self.note_interlaced((*raw).flags & ff::AV_FRAME_FLAG_INTERLACED != 0);

            let (ptr, layout) = match classify((*raw).format) {
                Some(l) => (raw, l),
                None => {
                    let conv = self.conv.to_rgba16(raw);
                    let mut r = raw;
                    ff::av_frame_free(&mut r);
                    (conv?, crate::frame::FrameLayout::Rgba16)
                }
            };
            let vf = Arc::new(VideoFrame::new(ptr, idx, pts, layout));
            self.insert(idx, vf);
            Ok(idx)
        }
    }

    /// Frame number of a decoded picture's timestamp (and remember it as the last).
    fn number(&mut self, pts: i64) -> i64 {
        if self.index.is_none() && self.est_origin.is_none() && pts != ffutil::AV_NOPTS_VALUE {
            self.est_origin = Some(pts);
        }
        let idx = if pts == ffutil::AV_NOPTS_VALUE { self.last.map_or(0, |l| l + 1) } else { self.frame_of_pts(pts) };
        self.last = Some(idx);
        self.last_pts = Some(pts);
        log::trace!("decoded frame {idx} pts {pts}");
        idx
    }

    fn note_interlaced(&self, il: bool) {
        if self.shared.interlaced.load(Ordering::Relaxed) == 0 {
            self.shared.interlaced.store(if il { 2 } else { 1 }, Ordering::Relaxed);
        }
    }

    /// Put a frame in the cache (evicting as needed); sample interlace detection.
    fn insert(&mut self, idx: i64, vf: Arc<VideoFrame>) {
        {
            let mut c = self.shared.cache.lock();
            c.bytes += vf.byte_size();
            if let Some(old) = c.frames.insert(idx, vf) {
                c.bytes -= old.byte_size();
            }
            self.evict(&mut c, idx);
            // Content-based interlace detection on a sample of frame triples.
            let center = idx - 1;
            let triple = if center >= 1 && center % 5 == 0 && self.analyzed.len() < 24 && !self.analyzed.contains(&center) {
                match (c.frames.get(&(center - 1)), c.frames.get(&center), c.frames.get(&idx)) {
                    (Some(a), Some(b), Some(n)) => Some((a.clone(), b.clone(), n.clone())),
                    _ => None,
                }
            } else {
                None
            };
            drop(c);
            if let Some((a, b, n)) = triple {
                self.analyzed.insert(center);
                let kind = crate::detect::classify(&a, &b, &n);
                self.shared.votes.lock().add(kind);
            }
        }
    }

    /// Drop the frames farthest from the playhead until under budget.
    /// `keep` (the frame just stored) stays: dropping it would make whoever asked for it
    /// wait forever (it was made, so it is not decoded again).
    fn evict(&self, c: &mut Cache, keep: i64) {
        let budget = self.budget(c.bytes);
        if c.bytes <= budget {
            return;
        }
        let t = self.req.frame;
        let mut keys: Vec<i64> = c.frames.keys().copied().collect();
        // Frames behind the playhead matter less when playing forward, and vice versa.
        let cost = |f: i64| {
            let d = f - t;
            let behind = if self.req.dir >= 0 { d < 0 } else { d > 0 };
            d.abs() * if behind && self.req.playing { 4 } else { 1 }
        };
        keys.sort_unstable_by_key(|&f| std::cmp::Reverse(cost(f)));
        for f in keys {
            if c.bytes <= budget * 9 / 10 {
                break;
            }
            if f == keep {
                continue;
            }
            if let Some(old) = c.frames.remove(&f) {
                c.bytes -= old.byte_size();
            }
        }
    }
}

impl Drop for Worker {
    fn drop(&mut self) {
        unsafe {
            ff::av_frame_free(&mut self.frame);
            ff::avcodec_free_context(&mut self.dec);
            if !self.fallback.is_null() {
                ff::avcodec_free_context(&mut self.fallback);
            }
        }
    }
}

// ---- Image sequences: one file per frame (DESIGN.md §7c) ------------------------------

/// Decode threads for image sequences (EXR files are also decompressed on all cores).
const SEQ_THREADS: usize = 3;

/// Makes frame N of a sequence.
type DecodeFn = Arc<dyn Fn(i64) -> anyhow::Result<VideoFrame> + Send + Sync>;

struct SeqState {
    req: parking_lot::Mutex<Request>,
    /// Frames being decoded now, and frames that failed.
    busy: parking_lot::Mutex<HashSet<i64>>,
    failed: parking_lot::Mutex<HashSet<i64>>,
    wake: parking_lot::Condvar,
    stop: AtomicBool,
}

/// MediaInfo for a sequence of `n` files at `rate`, the rest from `base` (the first
/// file's own).
fn sequence_info(base: MediaInfo, path: &str, files: &[std::path::PathBuf], container: &str, rate: (i32, i32)) -> MediaInfo {
    let n = files.len() as i64;
    let fps = rate.0 as f64 / rate.1.max(1) as f64;
    MediaInfo {
        path: path.to_string(),
        file_name: crate::sequence::display_name(files),
        container: container.into(),
        frame_rate: fps,
        frame_rate_q: rate,
        // One tick per frame.
        time_base: (rate.1, rate.0),
        start_pts: 0,
        start_secs: 0.0,
        duration_secs: n as f64 / fps,
        frame_count_hint: n,
        field_order: crate::FieldOrder::Progressive,
        audio: None,
        timecode: None,
        video_stream: 0,
        ..base
    }
}

/// Frame rate of image sequences whose files do not say.
const SEQUENCE_RATE: (i32, i32) = (24, 1);

impl Source {
    /// An EXR sequence (the numbered files `path` belongs to, or a folder's), showing
    /// layer `layer` (by name; None: the default, see `ExrSequence::default_layer`).
    pub fn open_exr(path: &str, layer: Option<&str>) -> anyhow::Result<Source> {
        let seq = Arc::new(crate::exr_seq::scan(path)?);
        let li = layer.and_then(|n| seq.layer_named(n)).unwrap_or_else(|| seq.default_layer());
        let base = MediaInfo {
            path: String::new(),
            file_name: String::new(),
            container: String::new(),
            codec: format!("exr {}", seq.compression.to_ascii_lowercase()),
            pix_fmt: if seq.float32 { "float".into() } else { "half".into() },
            width: seq.width,
            height: seq.height,
            dar: seq.width as f64 / seq.height.max(1) as f64 * seq.pixel_aspect.max(0.01) as f64,
            frame_rate: 0.0,
            frame_rate_q: (0, 1),
            time_base: (1, 1),
            start_pts: 0,
            start_secs: 0.0,
            duration_secs: 0.0,
            frame_count_hint: 0,
            color: crate::ColorInfo {
                matrix: crate::Matrix::Rgb,
                full_range: true,
                transfer: crate::Transfer::Linear,
                primaries: if seq.bt2020 { crate::Primaries::Bt2020 } else { crate::Primaries::Bt709 },
            },
            field_order: crate::FieldOrder::Progressive,
            has_alpha: seq.layers[li].has_alpha(),
            bit_depth: if seq.float32 { 32 } else { 16 },
            audio: None,
            timecode: None,
            video_stream: 0,
            rotation: 0,
            layers: seq.layers.iter().map(|l| (l.name.clone(), l.label())).collect(),
            layer: li,
        };
        let container = if seq.files.len() > 1 { "OpenEXR sequence" } else { "OpenEXR" };
        let info = sequence_info(base, path, &seq.files, container, seq.fps.unwrap_or(SEQUENCE_RATE));
        log::info!("EXR: {} file(s), {}x{}, layer {:?} of {}", seq.files.len(), seq.width, seq.height, seq.layers[li].name, seq.layers.len());
        let alpha = seq.layers[li].has_alpha();
        let s = seq.clone();
        let decode: DecodeFn = Arc::new(move |f| {
            let px = s.decode(f as usize, li)?;
            let bits: &[u16] = unsafe { std::slice::from_raw_parts(px.as_ptr() as *const u16, px.len()) };
            VideoFrame::from_rgba_f16(s.width as usize, s.height as usize, bits, alpha, f, f)
        });
        Self::open_frames(info, seq.files.len() as i64, "OpenEXR", decode)
    }

    /// The numbered still images (PNG, TIFF, DPX, JPEG…) `path` belongs to, one file per
    /// frame, decoded by FFmpeg.
    fn open_image_sequence(path: &str, files: Vec<std::path::PathBuf>) -> anyhow::Result<Source> {
        let first = files[0].to_string_lossy().to_string();
        let input = Input::open(&first)?;
        let vs = unsafe { ff::av_find_best_stream(input.ctx, ff::AVMediaType::AVMEDIA_TYPE_VIDEO, -1, -1, std::ptr::null_mut(), 0) };
        anyhow::ensure!(vs >= 0, "no picture in {first}");
        let base = unsafe { info::read_info(&first, &input, vs) };
        drop(input);
        let info = sequence_info(base, path, &files, "image sequence", SEQUENCE_RATE);
        log::info!("image sequence: {} files, {}x{} {} {}", files.len(), info.width, info.height, info.codec, info.pix_fmt);
        let n = files.len() as i64;
        let files = Arc::new(files);
        let decode: DecodeFn = Arc::new(move |f| decode_image(&files[f as usize].to_string_lossy(), f));
        Self::open_frames(info, n, "FFmpeg (image sequence)", decode)
    }

    /// A source of `n` frames made one at a time by `decode` (image sequences), on
    /// several threads around the playhead, with the same cache and requests as video.
    fn open_frames(info: MediaInfo, n: i64, decoder: &str, decode: DecodeFn) -> anyhow::Result<Source> {
        let shared = Arc::new(Shared {
            cache: Mutex::new(Cache { frames: HashMap::new(), bytes: 0 }),
            index: RwLock::new(Some(Arc::new(FrameIndex { pts: (0..n).collect(), key_pts: (0..n).collect() }))),
            interlaced: AtomicU8::new(1),
            votes: Mutex::new(Default::default()),
            cancel: AtomicBool::new(false),
            limit: 0.into(),
            decoder: Mutex::new(decoder.into()),
            packet_frames: Mutex::new(HashSet::new()),
        });
        let state = Arc::new(SeqState {
            req: parking_lot::Mutex::new(Request { frame: 0, playing: false, dir: 1, ahead: 0, fill: false }),
            busy: Default::default(),
            failed: Default::default(),
            wake: parking_lot::Condvar::new(),
            stop: AtomicBool::new(false),
        });
        let (tx, rx) = crossbeam_channel::unbounded::<Msg>();
        let mut decoders = Vec::new();
        for k in 0..SEQ_THREADS {
            let (decode, shared, state) = (decode.clone(), shared.clone(), state.clone());
            decoders.push(std::thread::Builder::new().name(format!("brp-seq-{k}")).spawn(move || seq_decode_loop(n, &*decode, &shared, &state))?);
        }
        // Takes requests, wakes the decoders; stops them when the source goes.
        let worker = {
            let state = state.clone();
            std::thread::Builder::new().name("brp-seq".into()).spawn(move || {
                while let Ok(m) = rx.recv() {
                    if let Msg::Request(r) = m {
                        *state.req.lock() = r;
                        state.wake.notify_all();
                    }
                }
                state.stop.store(true, Ordering::Relaxed);
                state.wake.notify_all();
                for d in decoders {
                    let _ = d.join();
                }
            })?
        };
        Ok(Source { info, shared, tx: Some(tx), worker: Some(worker), indexer: None })
    }
}

/// One still image decoded by FFmpeg (frame `idx` of a sequence), in a layout the
/// renderer takes (converted when needed, as video frames are).
fn decode_image(path: &str, idx: i64) -> anyhow::Result<VideoFrame> {
    let input = Input::open(path)?;
    unsafe {
        let vs = ff::av_find_best_stream(input.ctx, ff::AVMediaType::AVMEDIA_TYPE_VIDEO, -1, -1, std::ptr::null_mut(), 0);
        anyhow::ensure!(vs >= 0, "no picture");
        let mut dec = open_decoder_threads(input.stream(vs), ff::FF_THREAD_SLICE as i32)?;
        let pkt = Packet::new();
        let mut frame = ff::av_frame_alloc();
        let mut got = false;
        let mut err = None;
        loop {
            let r = ff::av_read_frame(input.ctx, pkt.0);
            if r < 0 {
                // End of file: flush (decoders with a delay).
                ff::avcodec_send_packet(dec, std::ptr::null());
                got = ff::avcodec_receive_frame(dec, frame) >= 0;
                break;
            }
            if (*pkt.0).stream_index == vs {
                let s = ff::avcodec_send_packet(dec, pkt.0);
                ff::av_packet_unref(pkt.0);
                if s < 0 && s != ffutil::AVERROR_EAGAIN {
                    err = Some(ffutil::err_str(s));
                    break;
                }
                if ff::avcodec_receive_frame(dec, frame) >= 0 {
                    got = true;
                    break;
                }
            } else {
                ff::av_packet_unref(pkt.0);
            }
        }
        ff::avcodec_free_context(&mut dec);
        if !got {
            ff::av_frame_free(&mut frame);
            anyhow::bail!("no picture decoded{}", err.map(|e| format!(": {e}")).unwrap_or_default());
        }
        let (ptr, layout) = match classify((*frame).format) {
            Some(l) => (frame, l),
            None => {
                let conv = Converter::new().to_rgba16(frame);
                ff::av_frame_free(&mut frame);
                (conv?, crate::frame::FrameLayout::Rgba16)
            }
        };
        Ok(VideoFrame::new(ptr, idx, idx, layout))
    }
}

/// One sequence decode thread: the most needed frame nobody is decoding yet, again
/// and again.
fn seq_decode_loop(n: i64, decode: &(dyn Fn(i64) -> anyhow::Result<VideoFrame> + Send + Sync), shared: &Shared, state: &SeqState) {
    loop {
        if state.stop.load(Ordering::Relaxed) || shared.cancel.load(Ordering::Relaxed) {
            return;
        }
        let pick = {
            let mut busy = state.busy.lock();
            let req = *state.req.lock();
            let f = seq_next_needed(&req, n, shared, &busy, &state.failed.lock());
            if let Some(f) = f {
                busy.insert(f);
            } else {
                state.wake.wait_for(&mut busy, Duration::from_millis(100));
            }
            f
        };
        let Some(f) = pick else { continue };
        let t = std::time::Instant::now();
        match decode(f) {
            Ok(vf) => {
                log::trace!("sequence frame {f} in {:?}", t.elapsed());
                let req = *state.req.lock();
                let mut c = shared.cache.lock();
                c.bytes += vf.byte_size();
                if let Some(old) = c.frames.insert(f, Arc::new(vf)) {
                    c.bytes -= old.byte_size();
                }
                seq_evict(&mut c, &req, shared, f);
            }
            Err(e) => {
                log::warn!("sequence frame {f}: {e:#}");
                state.failed.lock().insert(f);
            }
        }
        state.busy.lock().remove(&f);
        state.wake.notify_all();
    }
}

fn seq_budget(shared: &Shared) -> usize {
    match shared.limit.load(Ordering::Relaxed) {
        0 => crate::mem::cache_budget(),
        b => b,
    }
}

/// Next frame to decode, in the same order of priority as video (`Worker::next_needed`).
fn seq_next_needed(req: &Request, n: i64, shared: &Shared, busy: &HashSet<i64>, failed: &HashSet<i64>) -> Option<i64> {
    let t = req.frame.clamp(0, (n - 1).max(0));
    let mut order: Vec<i64> = vec![t];
    if req.playing && req.dir >= 0 {
        order.extend(t + 1..=t + req.ahead.max(AHEAD));
    } else if req.playing {
        order.extend((t - BEHIND_PLAYING..t).rev());
    } else {
        order.extend(t + 1..=t + 3);
        order.extend((t - 8..t).rev());
    }
    let c = shared.cache.lock();
    let wanted = |f: &i64| *f >= 0 && *f < n && !c.frames.contains_key(f) && !busy.contains(f) && !failed.contains(f);
    if let Some(f) = order.into_iter().find(wanted) {
        return Some(f);
    }
    if req.playing || !req.fill {
        return None;
    }
    let next = (t + 4..n).find(wanted)?;
    let budget = seq_budget(shared);
    if c.bytes < budget * 8 / 10 {
        return Some(next);
    }
    let farthest = c.frames.keys().map(|f| (f - t).abs()).max().unwrap_or(0);
    (next - t < farthest - 1).then_some(next)
}

/// Drop the frames farthest from the playhead until under budget (`keep` stays).
fn seq_evict(c: &mut Cache, req: &Request, shared: &Shared, keep: i64) {
    let budget = seq_budget(shared);
    if c.bytes <= budget {
        return;
    }
    let t = req.frame;
    let mut keys: Vec<i64> = c.frames.keys().copied().collect();
    let cost = |f: i64| {
        let d = f - t;
        let behind = if req.dir >= 0 { d < 0 } else { d > 0 };
        d.abs() * if behind && req.playing { 4 } else { 1 }
    };
    keys.sort_unstable_by_key(|&f| std::cmp::Reverse(cost(f)));
    for f in keys {
        if c.bytes <= budget * 9 / 10 {
            break;
        }
        if f == keep {
            continue;
        }
        if let Some(old) = c.frames.remove(&f) {
            c.bytes -= old.byte_size();
        }
    }
}
