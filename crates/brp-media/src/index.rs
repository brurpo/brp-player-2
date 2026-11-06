//! Frame index: maps presentation frame numbers to exact timestamps.
//!
//! Frame numbers are positions in this table, never `time * fps`, so variable
//! frame rate and odd time bases stay exact.

use crate::ffutil::{self, Input, Packet};
use ffmpeg_sys_next as ff;
use std::sync::atomic::{AtomicBool, Ordering};

const AV_PKT_FLAG_DISCARD: i32 = 4;

#[derive(Debug, Default)]
pub struct FrameIndex {
    /// Presentation timestamps of every frame, sorted ascending. Frame N has `pts[N]`.
    pub pts: Vec<i64>,
    /// Timestamps of keyframes (decode entry points), sorted ascending.
    pub key_pts: Vec<i64>,
}

impl FrameIndex {
    pub fn len(&self) -> usize {
        self.pts.len()
    }

    pub fn is_empty(&self) -> bool {
        self.pts.is_empty()
    }

    pub fn pts_of(&self, frame: i64) -> Option<i64> {
        self.pts.get(usize::try_from(frame).ok()?).copied()
    }

    /// Frame number for a decoded timestamp. Exact match, or the closest frame.
    pub fn frame_of_pts(&self, pts: i64) -> i64 {
        match self.pts.binary_search(&pts) {
            Ok(i) => i as i64,
            Err(0) => 0,
            Err(i) if i >= self.pts.len() => self.pts.len() as i64 - 1,
            Err(i) => {
                let (a, b) = (self.pts[i - 1], self.pts[i]);
                if pts - a <= b - pts { i as i64 - 1 } else { i as i64 }
            }
        }
    }

    /// Keyframes at or before `pts`, latest first.
    pub fn keyframes_before(&self, pts: i64) -> impl Iterator<Item = i64> + '_ {
        let n = self.key_pts.partition_point(|&k| k <= pts);
        self.key_pts[..n].iter().rev().copied()
    }

    /// True if a keyframe lies strictly between the two timestamps.
    pub fn keyframe_between(&self, after: i64, upto: i64) -> bool {
        let i = self.key_pts.partition_point(|&k| k <= after);
        self.key_pts.get(i).is_some_and(|&k| k <= upto)
    }
}

/// Scan every packet of the video stream (no decoding) and build the index.
pub fn build(path: &str, stream: i32, cancel: &AtomicBool) -> anyhow::Result<FrameIndex> {
    let input = Input::open(path)?;
    input.discard_others(stream);
    let pkt = Packet::new();
    let mut pts = Vec::with_capacity(4096);
    let mut keys = Vec::with_capacity(256);
    loop {
        if cancel.load(Ordering::Relaxed) {
            anyhow::bail!("index cancelled");
        }
        let r = unsafe { ff::av_read_frame(input.ctx, pkt.0) };
        if r == ffutil::AVERROR_EOF {
            break;
        }
        if r < 0 {
            log::warn!("index: read error {}", ffutil::err_str(r));
            break;
        }
        unsafe {
            let p = &*pkt.0;
            if p.stream_index == stream && p.flags & AV_PKT_FLAG_DISCARD == 0 {
                let t = if p.pts != ffutil::AV_NOPTS_VALUE { p.pts } else { p.dts };
                if t != ffutil::AV_NOPTS_VALUE {
                    pts.push(t);
                    if p.flags & ff::AV_PKT_FLAG_KEY != 0 {
                        keys.push(t);
                    }
                }
            }
            ff::av_packet_unref(pkt.0);
        }
    }
    pts.sort_unstable();
    pts.dedup();
    keys.sort_unstable();
    keys.dedup();
    if keys.is_empty() && !pts.is_empty() {
        keys.push(pts[0]);
    }
    // Open GOP at the start of the stream: pictures shown before the first keyframe
    // reference a GOP that does not exist and cannot be decoded. They are not frames.
    if let Some(&k0) = keys.first() {
        let lead = pts.partition_point(|&p| p < k0);
        if lead > 0 {
            log::debug!("index: dropping {lead} undecodable leading pictures");
            pts.drain(..lead);
        }
    }
    Ok(FrameIndex { pts, key_pts: keys })
}
