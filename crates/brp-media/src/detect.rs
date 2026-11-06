//! Interlace detection from picture content (port of FFmpeg `idet`'s per-frame metric),
//! because container and codec flags are often wrong.

use crate::frame::{FrameLayout, VideoFrame};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FieldKind {
    Tff,
    Bff,
    Progressive,
    Undetermined,
}

/// FFmpeg `idet` defaults.
const INTERLACE_THRESHOLD: f64 = 1.04;
const PROGRESSIVE_THRESHOLD: f64 = 1.5;

fn sample(f: &VideoFrame, data: &[u8], stride: usize, x: usize, y: usize) -> i32 {
    if f.bytes_per_sample() == 2 {
        let i = y * stride + x * 2;
        u16::from_le_bytes([data[i], data[i + 1]]) as i32
    } else {
        data[y * stride + x] as i32
    }
}

/// Classify `cur` using its neighbours (luma only).
pub fn classify(prev: &VideoFrame, cur: &VideoFrame, next: &VideoFrame) -> FieldKind {
    if !matches!(cur.layout, FrameLayout::Yuv { .. }) || prev.width() != cur.width() || next.height() != cur.height() {
        return FieldKind::Undetermined;
    }
    let (w, h) = (cur.width() as usize, cur.height() as usize);
    let (pd, ps) = prev.plane(0);
    let (cd, cs) = cur.plane(0);
    let (nd, ns) = next.plane(0);
    let mut alpha = [0i64; 2];
    let mut delta = 0i64;
    // Every second pixel column is plenty for a decision and halves the cost.
    for y in 2..h.saturating_sub(2) {
        let (mut a_prev, mut a_next, mut d) = (0i64, 0i64, 0i64);
        for x in (0..w).step_by(2) {
            let up = sample(cur, cd, cs, x, y - 1);
            let down = sample(cur, cd, cs, x, y + 1);
            a_prev += (up + down - 2 * sample(prev, pd, ps, x, y)).abs() as i64;
            a_next += (up + down - 2 * sample(next, nd, ns, x, y)).abs() as i64;
            d += (up + down - 2 * sample(cur, cd, cs, x, y)).abs() as i64;
        }
        alpha[y & 1] += a_prev;
        alpha[(y ^ 1) & 1] += a_next;
        delta += d;
    }
    let (a0, a1) = (alpha[0] as f64, alpha[1] as f64);
    if a0 > INTERLACE_THRESHOLD * a1 {
        FieldKind::Tff
    } else if a1 > INTERLACE_THRESHOLD * a0 {
        FieldKind::Bff
    } else if a1 > PROGRESSIVE_THRESHOLD * delta as f64 {
        FieldKind::Progressive
    } else {
        FieldKind::Undetermined
    }
}

/// Votes over many frames.
#[derive(Clone, Copy, Debug, Default)]
pub struct Votes {
    pub tff: u32,
    pub bff: u32,
    pub progressive: u32,
    pub undetermined: u32,
}

impl Votes {
    pub fn add(&mut self, k: FieldKind) {
        match k {
            FieldKind::Tff => self.tff += 1,
            FieldKind::Bff => self.bff += 1,
            FieldKind::Progressive => self.progressive += 1,
            FieldKind::Undetermined => self.undetermined += 1,
        }
    }

    pub fn total(&self) -> u32 {
        self.tff + self.bff + self.progressive + self.undetermined
    }

    /// (interlaced, top field first), once there is enough evidence.
    /// Static scenes vote "undetermined", so only decided frames count.
    pub fn decision(&self) -> Option<(bool, bool)> {
        let decided = self.tff + self.bff + self.progressive;
        if decided < 4 {
            return None;
        }
        let interlaced = self.tff + self.bff;
        Some((interlaced > self.progressive, self.tff >= self.bff))
    }
}
