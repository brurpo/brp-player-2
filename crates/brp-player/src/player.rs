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
                self.load_audio(path);
                self.play();
            }
            Err(e) => {
                log::error!("open {path}: {e}");
                self.error = Some(format!("{e}"));
                self.playing = false;
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

    pub fn has_range(&self) -> bool {
        self.in_point > 0 || self.out_frame() < self.last_frame()
    }

    pub fn native_fps(&self) -> f64 {
        self.source.as_ref().map_or(0.0, |s| s.info.frame_rate)
    }

    fn reset_clock(&mut self) {
        let second = self.field == 1 && self.field_rate_active();
        if let Some(s) = &self.source {
            self.clock_base = if second {
                (s.frame_time(self.current) + s.frame_end_time(self.current)) * 0.5
            } else {
                s.frame_time(self.current)
            };
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

    /// Audio plays only for normal forward playback at native speed.
    fn audio_wanted(&self) -> bool {
        self.playing && self.dir > 0 && (self.speed - 1.0).abs() < 0.01
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
        self.playing = true;
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
        self.current = frame.clamp(0, self.last_frame());
        self.field = 0;
        self.reset_clock();
    }

    /// Step by frames; by single fields when showing fields (field-rate deinterlacing).
    pub fn step(&mut self, delta: i64) {
        self.pause();
        if self.field_rate_active() && delta.abs() == 1 {
            let pos = (self.current * 2 + self.field as i64 + delta).clamp(0, self.last_frame() * 2 + 1);
            self.current = pos.div_euclid(2);
            self.field = pos.rem_euclid(2) as u8;
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
        self.field_rate && self.deint_method().is_some()
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

    /// Advance the clock and pick the frame to show. Returns true while it needs
    /// to be called again soon (playing, or waiting for a frame to decode).
    pub fn tick(&mut self) -> bool {
        self.poll_audio();
        let Some(src) = &self.source else { return false };
        if self.playing {
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
            if self.field_rate_active() {
                let src = self.source.as_ref().unwrap();
                let mid = (src.frame_time(target) + src.frame_end_time(target)) * 0.5;
                self.field = (t >= mid) as u8;
            } else {
                self.field = 0;
            }
        }
        let deint = self.deint_method().is_some();
        let src = self.source.as_ref().unwrap();
        src.request(Request { frame: self.current, playing: self.playing, dir: self.dir });

        let exact = src.get(self.current);
        let mut waiting = exact.is_none();
        // Deinterlacing needs both neighbours for the exact result; keep redrawing until
        // they arrive (the picture is replaced when they do).
        if deint && !self.playing {
            let n = self.current;
            waiting |= (n > 0 && src.get(n - 1).is_none()) || (n < self.last_frame() && src.get(n + 1).is_none());
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
        self.playing || waiting
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
    pub fn video_rect(&self, vw: f32, vh: f32) -> Option<[f32; 4]> {
        let src = self.source.as_ref()?;
        let (w, h) = (src.info.width as f32, src.info.height as f32);
        if w <= 0.0 || h <= 0.0 || vw <= 0.0 || vh <= 0.0 {
            return None;
        }
        let dar = src.info.dar as f32;
        // Display size in square pixels, after rotation.
        let (disp_w, h) = if src.info.rotation % 180 == 90 { (h, h * dar) } else { (h * dar, h) };
        let base = if self.view.one_to_one { 1.0 } else { (vw / disp_w).min(vh / h) };
        let scale = base * 2f32.powf(self.view.zoom);
        let (rw, rh) = (disp_w * scale, h * scale);
        let cx = vw * 0.5 + self.view.pan[0];
        let cy = vh * 0.5 + self.view.pan[1];
        Some([cx - rw * 0.5, cy - rh * 0.5, cx + rw * 0.5, cy + rh * 0.5])
    }

    /// Zoom by `steps` keeping the point under the cursor fixed.
    pub fn zoom_at(&mut self, steps: f32, cursor: [f32; 2], vw: f32, vh: f32) {
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
    pub fn pixel_exact(&self, vw: f32, vh: f32) -> bool {
        match (self.video_rect(vw, vh), &self.source) {
            (Some(_), Some(_)) => self.view.one_to_one && self.view.zoom >= 0.0,
            _ => false,
        }
    }
}
