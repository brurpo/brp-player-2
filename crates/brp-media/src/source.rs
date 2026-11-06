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
    budget: usize,
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
            budget: 2048 * 1024 * 1024,
        });
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
                req: Request { frame: 0, playing: false, dir: 1 },
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

    pub fn cached_frames(&self) -> Vec<i64> {
        self.shared.cache.lock().frames.keys().copied().collect()
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
        (*ctx).thread_type = (ff::FF_THREAD_FRAME | ff::FF_THREAD_SLICE) as i32;
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
}

unsafe impl Send for Worker {}

const AHEAD: i64 = 12;
const BEHIND_PLAYING: i64 = 24;
const MAX_SEEK_ATTEMPTS: usize = 6;

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
            order.extend(t + 1..=t + AHEAD);
        } else if self.req.playing {
            order.extend((t - BEHIND_PLAYING..t).rev());
        } else {
            order.extend(t + 1..=t + 3);
            order.extend((t - 8..t).rev());
        }
        let c = self.shared.cache.lock();
        order
            .into_iter()
            .find(|&f| f >= 0 && f < n && !c.frames.contains_key(&f) && !self.unreachable.contains(&f))
    }

    /// Decode until `need` is in the cache (or found to be unreachable).
    fn produce(&mut self, need: i64) -> anyhow::Result<()> {
        if self.should_seek(need) {
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
            if got > need && !self.is_cached(need) {
                // Passed it without producing it.
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

    /// Next decoded frame (owned), or None at end of stream.
    fn decode_next(&mut self) -> anyhow::Result<Option<*mut ff::AVFrame>> {
        unsafe {
            loop {
                let r = ff::avcodec_receive_frame(self.dec, self.frame);
                if r == 0 {
                    let out = ff::av_frame_alloc();
                    ff::av_frame_move_ref(out, self.frame);
                    return Ok(Some(out));
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
    fn store(&mut self, raw: *mut ff::AVFrame) -> anyhow::Result<i64> {
        unsafe {
            let mut pts = (*raw).best_effort_timestamp;
            if pts == ffutil::AV_NOPTS_VALUE {
                pts = (*raw).pts;
            }
            if self.index.is_none() && self.est_origin.is_none() && pts != ffutil::AV_NOPTS_VALUE {
                self.est_origin = Some(pts);
            }
            let idx = if pts == ffutil::AV_NOPTS_VALUE {
                self.last.map_or(0, |l| l + 1)
            } else {
                self.frame_of_pts(pts)
            };
            self.last = Some(idx);
            self.last_pts = Some(pts);

            if self.shared.interlaced.load(Ordering::Relaxed) == 0 {
                let il = (*raw).flags & ff::AV_FRAME_FLAG_INTERLACED != 0;
                self.shared.interlaced.store(if il { 2 } else { 1 }, Ordering::Relaxed);
            }

            let (ptr, layout) = match classify((*raw).format) {
                Some(l) => (raw, l),
                None => {
                    let conv = self.conv.to_rgba16(raw);
                    let mut r = raw;
                    ff::av_frame_free(&mut r);
                    (conv?, crate::frame::FrameLayout::Rgba16)
                }
            };
            log::trace!("stored frame {idx} pts {pts}");
            let vf = Arc::new(VideoFrame::new(ptr, idx, pts, layout));
            let mut c = self.shared.cache.lock();
            c.bytes += vf.byte_size();
            if let Some(old) = c.frames.insert(idx, vf) {
                c.bytes -= old.byte_size();
            }
            self.evict(&mut c);
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
            Ok(idx)
        }
    }

    /// Drop the frames farthest from the playhead until under budget.
    fn evict(&self, c: &mut Cache) {
        if c.bytes <= self.shared.budget {
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
            if c.bytes <= self.shared.budget * 9 / 10 {
                break;
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
        }
    }
}
