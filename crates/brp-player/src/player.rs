//! Playback state: clock, current frame, In/Out range, loop, speed, and view (zoom/pan).

use brp_media::{Audio, Request, Source, VideoFrame};
use brp_render::DeintMethod;
use crossbeam_channel::Receiver;
use std::sync::Arc;
use std::time::Instant;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DeintMode {
    /// Deinterlace interlaced sources with the best method.
    Auto,
    Off,
    /// NNEDI3 + motion-adaptive.
    Best,
    Nnedi,
    Bwdif,
}

pub struct Player {
    pub source: Option<Source>,
    pub source_id: u64,
    /// Frame the user is looking at (target). The frame on screen may lag while decoding.
    pub current: i64,
    pub playing: bool,
    pub dir: i8,
    pub speed: f64,
    pub loop_on: bool,
    pub in_point: i64,
    pub out_point: i64,
    clock_base: f64,
    clock_start: Instant,
    /// When play was last pressed (logs how long the first pictures take).
    play_pressed: Option<Instant>,
    /// Frame drawn last (kept while the next one decodes).
    pub shown: Option<Arc<VideoFrame>>,
    pub error: Option<String>,
    pub opened_at: Option<Instant>,
    pub first_frame_logged: bool,
    pub view: View,
    audio: Option<Audio>,
    audio_rx: Option<Receiver<anyhow::Result<Option<Audio>>>>,
    audio_running: bool,
    drift_log: Instant,
    pub volume: f32,
    pub muted: bool,
    /// Output device chosen by the user (None = system default).
    pub audio_device: Option<String>,
    pub deint_mode: DeintMode,
    /// Show each field as its own picture (50i -> 50p) when deinterlacing.
    pub field_rate: bool,
    /// Field of `current` on screen (0 = first in time).
    pub field: u8,
    pub fx: crate::effects::Effects,
    /// History pictures the active effects need (set by the window each frame).
    pub fx_warmup: u32,
    /// The user is dragging the playhead (or an In/Out handle): show raw frames, ask
    /// for no processing; the frame stopped on is processed on release.
    pub scrubbing: bool,
    /// Picture size after effects, when they change it (upscalers).
    pub fx_output: Option<(u32, u32)>,
    /// AI processing (interpolation, motion blur, upscaling).
    pub ai: crate::ai::Ai,
    /// In-between picture of `current` on screen with AI interpolation (0 = the frame).
    pub sub: u8,
    /// Paced playback (AI effects) is keeping real time: audio plays.
    realtime: bool,
}

#[derive(Clone, Copy, Debug)]
pub struct View {
    /// log2 zoom on top of "fit to window".
    pub zoom: f32,
    /// Pan in physical pixels.
    pub pan: [f32; 2],
    /// Show at 1 video pixel = 1 screen pixel.
    pub one_to_one: bool,
}

impl Default for View {
    fn default() -> Self {
        View { zoom: 0.0, pan: [0.0, 0.0], one_to_one: false }
    }
}

impl Player {
    pub fn new() -> Self {
        Player {
            source: None,
            source_id: 0,
            current: 0,
            playing: false,
            dir: 1,
            speed: 1.0,
            loop_on: true,
            in_point: 0,
            out_point: 0,
            clock_base: 0.0,
            clock_start: Instant::now(),
            play_pressed: None,
            shown: None,
            error: None,
            opened_at: None,
            first_frame_logged: false,
            view: View::default(),
            audio: None,
            audio_rx: None,
            audio_running: false,
            drift_log: Instant::now(),
            volume: 1.0,
            muted: false,
            audio_device: None,
            deint_mode: DeintMode::Auto,
            field_rate: true,
            field: 0,
            fx: Default::default(),
            fx_warmup: 0,
            scrubbing: false,
            fx_output: None,
            ai: crate::ai::Ai::new(),
            sub: 0,
            realtime: false,
        }
    }

    pub fn open(&mut self, path: &str) {
        let t = Instant::now();
        self.source = None;
        self.shown = None;
        self.audio = None;
        self.audio_running = false;
        match Source::open(path) {
            Ok(src) => {
                log::info!(
                    "opened {} in {:?}: {} {}x{} {}",
                    src.info.file_name,
                    t.elapsed(),
                    src.info.codec,
                    src.info.width,
                    src.info.height,
                    src.info.pix_fmt
                );
                self.source = Some(src);
                self.source_id += 1;
                self.error = None;
                self.current = 0;
                self.in_point = 0;
                self.out_point = i64::MAX;
                self.speed = 1.0;
                self.dir = 1;
                self.view = View::default();
                self.opened_at = Some(t);
                self.first_frame_logged = false;
                // EXR files and image sequences carry no sound.
                if !self.source.as_ref().is_some_and(|s| s.info.container.starts_with("OpenEXR") || s.info.container == "image sequence") {
                    self.load_audio(path);
                }
                // Effects depend on the source (transfer, size): re-send the chain.
                self.fx.dirty = true;
                self.ai.clear();
                self.ai.set_source(self.source_id);
                self.sub = 0;
                self.play();
            }
            Err(e) => {
                log::error!("open {path}: {e}");
                self.error = Some(format!("{e}"));
                self.playing = false;
            }
        }
    }

    /// EXR: show layer `name` instead. The sequence opens again with it (so every
    /// cache, the window's and the effects', starts over for the new pictures); the
    /// position, In/Out, view and playback stay.
    pub fn set_layer(&mut self, name: &str) {
        let Some(path) = self.source.as_ref().map(|s| s.info.path.clone()) else { return };
        match Source::open_exr(&path, Some(name)) {
            Ok(src) => {
                log::info!("EXR layer {name:?}");
                self.source = Some(src);
                self.source_id += 1;
                self.shown = None;
                self.error = None;
                self.fx.dirty = true;
                self.ai.clear();
                self.ai.set_source(self.source_id);
            }
            Err(e) => {
                log::error!("EXR layer {name}: {e:#}");
                self.error = Some(format!("{e}"));
            }
        }
    }

    pub fn frame_count(&self) -> i64 {
        self.source.as_ref().map_or(0, |s| s.frame_count())
    }

    pub fn last_frame(&self) -> i64 {
        (self.frame_count() - 1).max(0)
    }

    /// Out point clamped to the real length (the length can grow once the index is ready).
    pub fn out_frame(&self) -> i64 {
        self.out_point.min(self.last_frame())
    }

    /// Slow motion factor: the clip plays this many times slower (1 = off).
    pub fn slow(&self) -> i64 {
        if self.ai.paces() { self.ai.slowdown() as i64 } else { 1 }
    }

    /// Pictures per source frame on the output timeline: fields (deinterlacing at
    /// field rate before the AI stack) times slow motion's in-betweens.
    pub fn per_frame(&self) -> i64 {
        if self.ai.paces() { self.ai.subs() as i64 } else { 1 }
    }

    /// Frames of the output (slow motion and fields add frames; none after the last one).
    pub fn out_count(&self) -> i64 {
        self.last_frame() * self.per_frame() + 1
    }

    /// Output frame on screen.
    pub fn out_index(&self) -> i64 {
        self.current * self.per_frame() + self.sub as i64
    }

    /// Output frame number of a source frame.
    pub fn out_of(&self, frame: i64) -> i64 {
        frame * self.per_frame()
    }

    /// Timecode of an output frame (slow motion: the time in the longer clip).
    pub fn out_timecode(&self, out: i64) -> String {
        let k = self.per_frame();
        if k == 1 {
            return self.timecode(out);
        }
        let fps = self.source.as_ref().map_or(25.0, |s| s.info.frame_rate).max(1.0);
        let secs = out as f64 * self.slow() as f64 / (k as f64 * fps);
        let (h, m, s) = ((secs / 3600.0) as i64, (secs / 60.0) as i64 % 60, secs % 60.0);
        format!("{h:02}:{m:02}:{s:06.3}")
    }

    /// Go to an output frame.
    pub fn seek_out(&mut self, out: i64) {
        let k = self.per_frame();
        let out = out.clamp(0, self.out_count() - 1);
        self.seek(out / k);
        self.sub = (out % k) as u8;
        self.reset_clock();
    }

    /// How the AI stack gets the source deinterlaced (before anything else): the
    /// method in use, the detected field order, and field rate.
    pub fn ai_source_deint(&self) -> Option<crate::ai::SourceDeint> {
        let method = self.deint_method()?;
        let tff = self.source.as_ref().and_then(|s| s.field_detection().1).map(|(_, tff)| tff);
        Some(crate::ai::SourceDeint { method, tff, fields: self.field_rate })
    }

    pub fn has_range(&self) -> bool {
        self.in_point > 0 || self.out_frame() < self.last_frame()
    }

    pub fn native_fps(&self) -> f64 {
        self.source.as_ref().map_or(0.0, |s| s.info.frame_rate)
    }

    fn reset_clock(&mut self) {
        // Start of the picture on screen: a field, an in-between, or the frame.
        let (part, parts) = if self.field_rate_active() {
            (self.field as f64, 2.0)
        } else {
            (self.sub as f64, self.subs() as f64)
        };
        if let Some(s) = &self.source {
            let (t0, t1) = (s.frame_time(self.current), s.frame_end_time(self.current));
            self.clock_base = t0 + (t1 - t0) * part / parts;
        }
        self.clock_start = Instant::now();
        // Every jump in time passes through here: restart (or stop) audio to match.
        self.sync_audio(true);
    }

    /// Open the audio stream on a background thread (device start-up can take a while).
    fn load_audio(&mut self, path: &str) {
        let (tx, rx) = crossbeam_channel::bounded(1);
        let (path, device) = (path.to_string(), self.audio_device.clone());
        std::thread::spawn(move || {
            let _ = tx.send(Audio::open(&path, device.as_deref()));
        });
        self.audio_rx = Some(rx);
    }

    /// Re-open audio on another output device, keeping the playback position.
    pub fn set_audio_device(&mut self, device: Option<String>) {
        self.audio_device = device;
        self.audio = None;
        self.audio_running = false;
        if let Some(path) = self.source.as_ref().map(|s| s.info.path.clone()) {
            self.load_audio(&path);
        }
    }

    pub fn has_audio(&self) -> bool {
        self.audio.is_some()
    }

    pub fn set_volume(&mut self, v: f32) {
        self.volume = v.clamp(0.0, 1.0);
        if let Some(a) = &self.audio {
            a.set_volume(self.volume);
        }
    }

    pub fn set_muted(&mut self, m: bool) {
        self.muted = m;
        if let Some(a) = &self.audio {
            a.set_muted(m);
        }
    }

    /// Audio plays only for normal forward playback at native speed; with AI effects,
    /// only while their processing keeps real time (playback waits for each picture).
    fn audio_wanted(&self) -> bool {
        self.playing && self.dir > 0 && (self.speed - 1.0).abs() < 0.01 && self.slow() == 1 && (!self.ai.paces() || self.realtime)
    }

    /// Start, restart or stop audio to match playback. `restart` = time jumped.
    fn sync_audio(&mut self, restart: bool) {
        let wanted = self.audio_wanted();
        let (Some(a), Some(src)) = (&self.audio, &self.source) else { return };
        match (wanted, self.audio_running) {
            (true, false) => a.start(src.frame_time(self.current) + src.info.start_secs),
            (true, true) if restart => a.start(src.frame_time(self.current) + src.info.start_secs),
            (false, true) => a.stop(),
            _ => {}
        }
        self.audio_running = wanted;
    }

    /// Pick up audio once it has opened; keep the video clock locked to it.
    fn poll_audio(&mut self) {
        if let Some(rx) = &self.audio_rx {
            if let Ok(res) = rx.try_recv() {
                self.audio_rx = None;
                match res {
                    Ok(Some(a)) => {
                        a.set_volume(self.volume);
                        a.set_muted(self.muted);
                        self.audio = Some(a);
                        self.audio_running = false;
                        self.sync_audio(false);
                    }
                    Ok(None) => {}
                    Err(e) => log::warn!("audio unavailable: {e}"),
                }
            }
        }
        if !self.audio_running {
            return;
        }
        let (Some(a), Some(src)) = (&self.audio, &self.source) else { return };
        if let Some(pos) = a.position() {
            let heard = pos - src.info.start_secs;
            let t = self.clock_base + self.clock_start.elapsed().as_secs_f64() * self.speed;
            let drift = heard - t;
            if self.drift_log.elapsed().as_secs_f64() > 1.0 {
                log::debug!("a/v drift {:+.1} ms at {:.2}s", drift * 1000.0, heard);
                self.drift_log = Instant::now();
            }
            // Slew gently; jump only if far off (e.g. right after a start).
            self.clock_base += if drift.abs() > 0.25 { drift } else { drift * 0.05 };
        }
    }

    pub fn play(&mut self) {
        if self.source.is_none() {
            return;
        }
        // Pressing play at the end without loop restarts from In.
        if self.dir > 0 && self.current >= self.out_frame() {
            self.current = self.in_point;
        }
        if self.dir < 0 && self.current <= self.in_point {
            self.current = self.out_frame();
        }
        self.realtime = false;
        self.playing = true;
        self.play_pressed = Some(Instant::now());
        self.reset_clock();
    }

    pub fn pause(&mut self) {
        self.playing = false;
        self.sync_audio(false);
    }

    pub fn toggle(&mut self) {
        if self.playing {
            self.pause()
        } else {
            self.dir = 1;
            self.play()
        }
    }

    /// J / L shuttle: play in a direction.
    pub fn shuttle(&mut self, dir: i8) {
        self.dir = dir;
        self.play();
    }

    pub fn seek(&mut self, frame: i64) {
        self.realtime = false;
        self.current = frame.clamp(0, self.last_frame());
        self.field = 0;
        self.sub = 0;
        self.reset_clock();
    }

    /// Pictures per source frame from AI interpolation (1 = off). Field-rate
    /// deinterlacing takes precedence.
    pub fn subs(&self) -> u8 {
        self.ai.subs()
    }

    /// Step by frames; by single fields when showing fields (field-rate deinterlacing),
    /// by in-between pictures with AI interpolation.
    pub fn step(&mut self, delta: i64) {
        self.pause();
        let subs = self.subs() as i64;
        if self.field_rate_active() && delta.abs() == 1 {
            let pos = (self.current * 2 + self.field as i64 + delta).clamp(0, self.last_frame() * 2 + 1);
            self.current = pos.div_euclid(2);
            self.field = pos.rem_euclid(2) as u8;
            self.reset_clock();
        } else if subs > 1 && delta.abs() == 1 {
            let pos = (self.current * subs + self.sub as i64 + delta).clamp(0, self.last_frame() * subs);
            self.current = pos.div_euclid(subs);
            self.sub = pos.rem_euclid(subs) as u8;
            self.reset_clock();
        } else {
            self.seek(self.current + delta);
        }
    }

    /// Method in use for the current file, if any.
    pub fn deint_method(&self) -> Option<DeintMethod> {
        match self.deint_mode {
            DeintMode::Off => None,
            DeintMode::Best => Some(DeintMethod::NnediAdaptive),
            DeintMode::Nnedi => Some(DeintMethod::Nnedi),
            DeintMode::Bwdif => Some(DeintMethod::Bwdif),
            DeintMode::Auto => self.source_interlaced().then_some(DeintMethod::NnediAdaptive),
        }
    }

    /// Auto mode's decision: picture-content analysis when it is confident, otherwise
    /// the decoder's per-frame flag.
    pub fn source_interlaced(&self) -> bool {
        let Some(s) = &self.source else { return false };
        match s.field_detection().1 {
            Some((interlaced, _)) => interlaced,
            None => s.interlaced().unwrap_or(false),
        }
    }

    /// Field order for deinterlacing `f`: detected from content when known, else the
    /// frame flag (progressive-flagged frames default to top field first).
    pub fn field_tff(&self, f: &VideoFrame) -> bool {
        match self.source.as_ref().and_then(|s| s.field_detection().1) {
            Some((_, tff)) => tff,
            None => f.tff || !f.interlaced,
        }
    }

    pub fn field_rate_active(&self) -> bool {
        // AI output replaces the deinterlaced picture: no fields while AI effects are on.
        self.field_rate && self.deint_method().is_some() && !self.ai.active()
    }

    /// Deterministic position of the picture on screen: frame, or 2*frame+field when
    /// showing fields. Consecutive values = consecutive pictures.
    pub fn seq(&self) -> i64 {
        let f = self.shown.as_ref().map_or(self.current, |f| f.index());
        if self.field_rate_active() {
            f * 2 + self.field as i64
        } else {
            let subs = self.subs() as i64;
            f * subs + self.sub as i64
        }
    }

    /// The `n` pictures before the one on screen (oldest first) that are decoded:
    /// (frame, prev, next, seq, second field). Used to rebuild effect history.
    #[allow(clippy::type_complexity)]
    pub fn history(&self, n: u32) -> Vec<(Arc<VideoFrame>, Option<Arc<VideoFrame>>, Option<Arc<VideoFrame>>, i64, bool)> {
        let Some(src) = &self.source else { return Vec::new() };
        let fields = self.field_rate_active();
        let seq = self.seq();
        let mut out = Vec::new();
        for s in (seq - n as i64..seq).filter(|&s| s >= 0) {
            let subs = self.subs() as i64;
            let (frame, second) = if fields { (s.div_euclid(2), s.rem_euclid(2) == 1) } else { (s.div_euclid(subs), false) };
            if let Some(f) = src.get(frame) {
                out.push((f, src.get(frame - 1), src.get(frame + 1), s, second));
            }
        }
        out
    }

    /// Transfer of the source, for shaders' linearize().
    pub fn transfer(&self) -> brp_render::Transfer {
        match self.source.as_ref().map(|s| s.info.color.transfer) {
            Some(brp_media::Transfer::Pq) => brp_render::Transfer::Pq,
            Some(brp_media::Transfer::Hlg) => brp_render::Transfer::Hlg,
            Some(brp_media::Transfer::Linear) => brp_render::Transfer::Linear,
            _ => brp_render::Transfer::Sdr,
        }
    }

    /// Decoded neighbours of the frame on screen (for deinterlacing).
    pub fn neighbours(&self) -> (Option<Arc<VideoFrame>>, Option<Arc<VideoFrame>>) {
        match (&self.source, &self.shown) {
            (Some(src), Some(f)) => (src.get(f.index() - 1), src.get(f.index() + 1)),
            _ => (None, None),
        }
    }

    pub fn set_in(&mut self) {
        self.in_point = self.current;
        if self.out_point <= self.in_point {
            self.out_point = i64::MAX;
        }
    }

    pub fn set_out(&mut self) {
        self.out_point = self.current;
        if self.in_point >= self.out_point {
            self.in_point = 0;
        }
    }

    pub fn clear_range(&mut self) {
        self.in_point = 0;
        self.out_point = i64::MAX;
    }

    pub fn set_fps(&mut self, fps: f64) {
        let native = self.native_fps();
        if fps > 0.0 && native > 0.0 {
            self.speed = fps / native;
            self.reset_clock();
        }
    }

    /// Playback with AI effects: every picture (frame or in-between) in order, each
    /// shown once its processed version is ready — real time when processing keeps
    /// up, slower when it does not, never skipping.
    fn tick_paced(&mut self) {
        // Audio joins once the pictures for the next moments are ready (processing is
        // ahead of real time), and stops when playback falls behind (below).
        if !self.realtime && self.dir > 0 {
            let mut p = (self.current, self.sub);
            let buffered = (0..REALTIME_BUFFER).all(|_| {
                p = self.ai.advance(p, 1);
                p.0 > self.out_frame() || self.ai.ready(p)
            });
            if buffered {
                self.realtime = true;
                log::debug!("playback keeping real time at frame {}: audio on", self.current);
                // Restart the clock on this picture so audio starts in step with it.
                self.reset_clock();
            }
        }
        let Some(src) = &self.source else { return };
        let subs = self.subs() as f64;
        let start_of = |(n, sub): (i64, u8)| {
            let (t0, t1) = (src.frame_time(n), src.frame_end_time(n));
            t0 + (t1 - t0) * sub as f64 / subs
        };
        let dir = self.dir;
        // Slow motion: each picture lasts a whole frame, so source time runs k times slower.
        let t = self.clock_base + self.clock_start.elapsed().as_secs_f64() * self.speed * dir as f64 / self.slow() as f64;
        let cur = (self.current, self.sub);
        let mut next = self.ai.advance(cur, dir);
        // Nothing to interpolate towards after the last frame.
        if next.0 >= self.last_frame() && next.1 > 0 {
            next = if dir > 0 { (self.last_frame() + 1, 0) } else { (self.last_frame(), 0) };
        }
        // When `next` is due: forward when its time starts; backward when the time of
        // the picture on screen is left behind.
        let due_at = if dir > 0 { start_of(next) } else { start_of(cur) };
        let due = if dir > 0 { t >= due_at } else { t < due_at };
        if !due {
            return;
        }
        let (lo, hi) = (self.in_point, self.out_frame());
        if next.0 > hi || next.0 < lo {
            if self.loop_on {
                self.seek(if dir > 0 { lo } else { hi });
            } else {
                self.playing = false;
            }
            return;
        }
        if !self.ai.ready(next) {
            // Hold the picture on screen: playback waits for the next one.
            return;
        }
        // Late by more than a picture (processing slower than real time): carry on
        // from this picture's time instead of trying to catch up.
        let after = self.ai.advance(next, dir);
        let late = if dir > 0 { t >= start_of(after) } else { t < start_of(next) };
        if late {
            if self.realtime {
                self.realtime = false;
                self.sync_audio(false);
                log::debug!("playback behind real time at frame {}: audio off", next.0);
            }
            self.clock_base = due_at;
            self.clock_start = Instant::now();
        }
        if let Some(t0) = self.play_pressed {
            log::debug!("paced: picture {next:?} shown {:.0} ms after play", t0.elapsed().as_secs_f64() * 1e3);
            if next.0 > self.current + 30 {
                self.play_pressed = None;
            }
        }
        self.current = next.0;
        self.sub = next.1;
        self.field = 0;
    }

    /// Advance the clock and pick the frame to show. Returns true while it needs
    /// to be called again soon (playing, or waiting for a frame to decode).
    pub fn tick(&mut self) -> bool {
        self.poll_audio();
        // BRP_TEST_SEEK_AFTER=secs:frame: jump there (as Go to In) that long after opening.
        if let (Some(t), Some((at, f))) = (self.opened_at, std::env::var("BRP_TEST_SEEK_AFTER").ok().and_then(|v| v.split_once(':').and_then(|(a, b)| Some((a.parse::<f64>().ok()?, b.parse::<i64>().ok()?))))) {
            static DONE: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
            if t.elapsed().as_secs_f64() >= at && !DONE.swap(true, std::sync::atomic::Ordering::Relaxed) {
                log::info!("test: seek to {f} from {}", self.current);
                self.seek(f);
            }
        }
        // BRP_TEST_PLAY_AFTER=secs: press play that long after opening (timing the start).
        if let (Some(t), Some(at)) = (self.opened_at, std::env::var("BRP_TEST_PLAY_AFTER").ok().and_then(|v| v.parse::<f64>().ok())) {
            static PRESSED: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
            if !self.playing && t.elapsed().as_secs_f64() >= at && !PRESSED.swap(true, std::sync::atomic::Ordering::Relaxed) {
                log::info!("test: play at frame {}", self.current);
                self.play();
            }
        }
        let Some(src) = &self.source else { return false };
        if self.playing && self.ai.paces() {
            self.tick_paced();
        } else if self.playing {
            let t = self.clock_base + self.clock_start.elapsed().as_secs_f64() * self.speed * self.dir as f64;
            let mut target = src.frame_at_time(t.max(0.0));
            let (lo, hi) = (self.in_point, self.out_frame());
            // Compare times, not frame numbers: frame lookup clamps to the last frame,
            // so without an Out point the frame number never passes the end.
            let past_end = target > hi || t >= src.frame_end_time(hi);
            let before_start = target < lo || t < src.frame_time(lo);
            if self.dir > 0 && past_end {
                if self.loop_on {
                    log::debug!("loop: end of range reached, back to frame {lo}");
                    target = lo;
                    self.seek(lo);
                } else {
                    target = hi;
                    self.playing = false;
                }
            } else if self.dir < 0 && before_start {
                if self.loop_on {
                    target = hi;
                    self.seek(hi);
                } else {
                    target = lo;
                    self.playing = false;
                }
            }
            self.current = target;
            let subs = self.subs();
            let src = self.source.as_ref().unwrap();
            let (t0, t1) = (src.frame_time(target), src.frame_end_time(target));
            let frac = if t1 > t0 { ((t - t0) / (t1 - t0)).clamp(0.0, 0.999) } else { 0.0 };
            if self.field_rate_active() {
                self.field = (frac >= 0.5) as u8;
                self.sub = 0;
            } else {
                self.field = 0;
                self.sub = (frac * subs as f64) as u8;
                // Nothing to interpolate towards after the last frame.
                if target >= self.last_frame() {
                    self.sub = 0;
                }
            }
        }
        let deint = self.deint_method().is_some();
        let src = self.source.as_ref().unwrap();
        src.request(Request { frame: self.current, playing: self.playing, dir: self.dir, ahead: if self.ai.active() && !self.scrubbing { 60 } else { 0 }, fill: !self.scrubbing });

        let exact = src.get(self.current);
        let mut waiting = exact.is_none();
        // Deinterlacing needs both neighbours for the exact result; keep redrawing until
        // they arrive (the picture is replaced when they do).
        if deint && !self.playing {
            let n = self.current;
            waiting |= (n > 0 && src.get(n - 1).is_none()) || (n < self.last_frame() && src.get(n + 1).is_none());
        }
        // Stateful effects after a jump: keep redrawing until their history is decoded.
        if self.fx_warmup > 0 && !self.playing && !self.scrubbing {
            let want = (self.fx_warmup as i64).min(self.seq()) as usize;
            waiting |= self.history(self.fx_warmup).len() < want;
        }
        let pick = exact.or_else(|| {
            // While waiting: during playback hold the current picture (drop late frames);
            // while scrubbing show the closest decoded frame.
            if self.playing && self.shown.is_some() { None } else { src.nearest(self.current) }
        });
        if let Some(f) = pick {
            self.shown = Some(f);
        }
        if !self.first_frame_logged && self.shown.is_some() {
            if let Some(t) = self.opened_at {
                log::info!("first frame on screen {:?} after open", t.elapsed());
            }
            self.first_frame_logged = true;
        }
        // AI results arrive asynchronously: keep redrawing while one is being made.
        if self.ai.active() && self.ai.busy() && !self.scrubbing {
            waiting = true;
        }
        self.playing || waiting
    }

    /// The AI-processed picture on screen, if AI is on and it is ready.
    pub fn ai_frame(&mut self) -> Option<Arc<brp_render::compute::Picture>> {
        if !self.ai.active() || self.scrubbing {
            return None;
        }
        let n = self.shown.as_ref()?.index();
        let sub = if n == self.current { self.sub } else { 0 };
        let src = self.source.as_ref()?;
        let playback = self.playing.then(|| crate::ai::Playback {
            interval: 1.0 / (src.info.frame_rate.max(1.0) * self.speed.max(0.01)),
            dir: self.dir,
        });
        self.ai.get(src, (n, sub), playback)
    }

    /// Timecode for a frame: container timecode if present, else HH:MM:SS.mmm.
    pub fn timecode(&self, frame: i64) -> String {
        let Some(src) = &self.source else { return "--:--:--:--".into() };
        let fps = src.info.frame_rate.round().max(1.0) as i64;
        if let Some(tc) = &src.info.timecode {
            let parts: Vec<i64> = tc.split([':', ';', '.']).filter_map(|p| p.parse().ok()).collect();
            if parts.len() == 4 {
                let start = ((parts[0] * 60 + parts[1]) * 60 + parts[2]) * fps + parts[3];
                let t = start + frame;
                let sep = if tc.contains(';') { ';' } else { ':' };
                return format!("{:02}:{:02}:{:02}{sep}{:02}", t / (fps * 3600), (t / (fps * 60)) % 60, (t / fps) % 60, t % fps);
            }
        }
        let secs = src.frame_time(frame);
        let ms = (secs * 1000.0).round() as i64;
        format!("{:02}:{:02}:{:02}.{:03}", ms / 3_600_000, (ms / 60_000) % 60, (ms / 1000) % 60, ms % 1000)
    }

    /// Where the video goes, in physical pixels, for a viewport of `vw` x `vh`.
    /// `area`: where the picture goes (x0, y0, x1, y1, physical pixels): the window in
    /// fullscreen, the space between the bars in a window.
    pub fn video_rect(&self, area: [f32; 4]) -> Option<[f32; 4]> {
        let (vw, vh) = (area[2] - area[0], area[3] - area[1]);
        let src = self.source.as_ref()?;
        let (w, h) = (src.info.width as f32, src.info.height as f32);
        if w <= 0.0 || h <= 0.0 || vw <= 0.0 || vh <= 0.0 {
            return None;
        }
        let dar = src.info.dar as f32;
        // Display size in square pixels, after rotation.
        let (disp_w, h) = if src.info.rotation % 180 == 90 { (h, h * dar) } else { (h * dar, h) };
        // 1:1 means one picture pixel per screen pixel, after effects that resize.
        let fx_scale = match self.fx_output {
            Some((_, oh)) if src.info.height > 0 => {
                let src_h = if src.info.rotation % 180 == 90 { src.info.width } else { src.info.height };
                oh as f32 / src_h as f32
            }
            _ => 1.0,
        };
        let base = if self.view.one_to_one { fx_scale } else { (vw / disp_w).min(vh / h) };
        let scale = base * 2f32.powf(self.view.zoom);
        let (rw, rh) = (disp_w * scale, h * scale);
        let cx = area[0] + vw * 0.5 + self.view.pan[0];
        let cy = area[1] + vh * 0.5 + self.view.pan[1];
        Some([cx - rw * 0.5, cy - rh * 0.5, cx + rw * 0.5, cy + rh * 0.5])
    }

    /// Zoom by `steps` keeping the point under the cursor fixed.
    pub fn zoom_at(&mut self, steps: f32, cursor: [f32; 2], area: [f32; 4]) {
        let (vw, vh) = (area[2] - area[0], area[3] - area[1]);
        let cursor = [cursor[0] - area[0], cursor[1] - area[1]];
        let old = self.view.zoom;
        let new = (old + steps).clamp(-4.0, 6.0);
        let k = 2f32.powf(new - old);
        let c = [vw * 0.5 + self.view.pan[0], vh * 0.5 + self.view.pan[1]];
        // Keep cursor fixed: new_center = cursor - (cursor - center) * k
        self.view.pan[0] = cursor[0] - (cursor[0] - c[0]) * k - vw * 0.5;
        self.view.pan[1] = cursor[1] - (cursor[1] - c[1]) * k - vh * 0.5;
        self.view.zoom = new;
    }

    pub fn reset_view(&mut self) {
        self.view = View::default();
    }

    pub fn toggle_one_to_one(&mut self) {
        let on = !self.view.one_to_one;
        self.view = View { one_to_one: on, ..View::default() };
    }

    /// True when video pixels map to screen pixels at least 1:1 (use nearest sampling).
    pub fn pixel_exact(&self, area: [f32; 4]) -> bool {
        match (self.video_rect(area), &self.source) {
            (Some(_), Some(_)) => self.view.one_to_one && self.view.zoom >= 0.0,
            _ => false,
        }
    }
}

/// Pictures that must be ready ahead before audio plays with AI effects (~0.4 s).
const REALTIME_BUFFER: usize = 12;
