//! Video stabilisation from the motion between frames.
//!
//! For each pair of neighbouring frames, the camera's movement is fitted to their motion
//! (a small grid of it, read back): a similarity (shift, rotation, zoom) by iteratively
//! reweighted least squares, so what moves on its own (people, cars) counts less than
//! the scene. A frame's path is those fits composed over a window each way (stopping
//! at scene cuts and the clip's ends); the steady camera is their Gaussian-weighted
//! mean, and the frame is warped onto it (Catmull-Rom), zoomed in a little so the
//! edges the correction uncovers stay mostly outside the picture.

use crate::gpu::Ops;
use crate::vk::{PushTensor, Tensor};
use anyhow::Result;
use brp_render::compute::{Kernel, Push, Recorder, PRELUDE};
use std::sync::Arc;

/// A similarity in picture pixels about the picture's centre: (x, y) goes to
/// (a x - b y + tx, b x + a y + ty).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Sim {
    pub a: f64,
    pub b: f64,
    pub tx: f64,
    pub ty: f64,
}

impl Sim {
    pub const IDENTITY: Sim = Sim { a: 1.0, b: 0.0, tx: 0.0, ty: 0.0 };

    pub fn apply(&self, x: f64, y: f64) -> (f64, f64) {
        (self.a * x - self.b * y + self.tx, self.b * x + self.a * y + self.ty)
    }

    /// `self` after `first`.
    pub fn after(&self, first: &Sim) -> Sim {
        let (tx, ty) = self.apply(first.tx, first.ty);
        Sim { a: self.a * first.a - self.b * first.b, b: self.b * first.a + self.a * first.b, tx, ty }
    }

    pub fn inverse(&self) -> Sim {
        let d = self.a * self.a + self.b * self.b;
        let (a, b) = (self.a / d, -self.b / d);
        Sim { a, b, tx: -(a * self.tx - b * self.ty), ty: -(b * self.tx + a * self.ty) }
    }
}

/// The camera's movement from a motion grid: `flow` [2][gh][gw] (in units of
/// `unit` picture pixels: grids of a motion found at a smaller size), each cell centred
/// on its part of a w x h picture. None when too little of the picture follows one
/// movement (fewer than a fifth of the cells near it).
pub fn fit(flow: &crate::Image, w: usize, h: usize, unit: f64) -> Option<Sim> {
    let (gw, gh) = (flow.w, flow.h);
    let n = gw * gh;
    let mut pts = Vec::with_capacity(n);
    for y in 0..gh {
        for x in 0..gw {
            let i = y * gw + x;
            let (dx, dy) = (flow.data[i] as f64 * unit, flow.data[n + i] as f64 * unit);
            if !dx.is_finite() || !dy.is_finite() {
                continue;
            }
            let px = (x as f64 + 0.5) * w as f64 / gw as f64 - w as f64 / 2.0;
            let py = (y as f64 + 0.5) * h as f64 / gh as f64 - h as f64 / 2.0;
            pts.push((px, py, px + dx, py + dy));
        }
    }
    if pts.len() < 16 {
        return None;
    }
    // Start: the similarity two cells give (q = z p + t, z complex) that most cells
    // agree with (within 1.5 px), over pairs drawn by a fixed sequence (repeatable).
    let err = |s: &Sim, &(x, y, qx, qy): &(f64, f64, f64, f64)| {
        let (ex, ey) = s.apply(x, y);
        ((ex - qx).powi(2) + (ey - qy).powi(2)).sqrt()
    };
    let mut seed = 0x9e37_79b9_7f4a_7c15u64;
    let mut next = |m: usize| {
        seed ^= seed << 13;
        seed ^= seed >> 7;
        seed ^= seed << 17;
        (seed % m as u64) as usize
    };
    let mut best = (0usize, Sim::IDENTITY);
    for _ in 0..300 {
        let (p, q) = (pts[next(pts.len())], pts[next(pts.len())]);
        let (dx, dy) = (p.0 - q.0, p.1 - q.1);
        let d = dx * dx + dy * dy;
        if d < 100.0 {
            continue;
        }
        let (ux, uy) = (p.2 - q.2, p.3 - q.3);
        let (a, b) = ((ux * dx + uy * dy) / d, (uy * dx - ux * dy) / d);
        let s = Sim { a, b, tx: p.2 - (a * p.0 - b * p.1), ty: p.3 - (b * p.0 + a * p.1) };
        let count = pts.iter().filter(|t| err(&s, t) < 1.5).count();
        if count > best.0 {
            best = (count, s);
        }
    }
    let mut s = best.1;
    let mut wts: Vec<f64> = pts.iter().map(|t| if err(&s, t) < 3.0 { 1.0 } else { 0.0 }).collect();
    let mut inliers = 0.0;
    for _ in 0..6 {
        // Normal equations for (a, b, tx, ty).
        let mut m = [[0f64; 4]; 4];
        let mut r = [0f64; 4];
        for (&(x, y, qx, qy), &wt) in pts.iter().zip(&wts) {
            for (row, q) in [([x, -y, 1.0, 0.0], qx), ([y, x, 0.0, 1.0], qy)] {
                for i in 0..4 {
                    for j in 0..4 {
                        m[i][j] += wt * row[i] * row[j];
                    }
                    r[i] += wt * row[i] * q;
                }
            }
        }
        let sol = solve4(m, r)?;
        s = Sim { a: sol[0], b: sol[1], tx: sol[2], ty: sol[3] };
        // Tukey weights against a robust spread of the cells that follow the fit (at
        // least a pixel, at most 4).
        let res: Vec<f64> = pts.iter().map(|t| err(&s, t)).collect();
        let mut near: Vec<f64> = res.iter().zip(&wts).filter(|(_, w)| **w > 0.0).map(|(e, _)| *e).collect();
        near.sort_by(f64::total_cmp);
        let c = near.get(near.len() / 2).map_or(1.0, |m| (m * 1.4826 * 3.0).clamp(1.0, 4.0));
        inliers = 0.0;
        for (wt, &e) in wts.iter_mut().zip(&res) {
            let u = e / c;
            *wt = if u < 1.0 { (1.0 - u * u).powi(2) } else { 0.0 };
            if u < 0.5 {
                inliers += 1.0;
            }
        }
    }
    (inliers / pts.len() as f64 >= 0.2).then_some(s)
}

fn solve4(mut m: [[f64; 4]; 4], mut r: [f64; 4]) -> Option<[f64; 4]> {
    for c in 0..4 {
        let p = (c..4).max_by(|&i, &j| m[i][c].abs().total_cmp(&m[j][c].abs()))?;
        if m[p][c].abs() < 1e-12 {
            return None;
        }
        m.swap(c, p);
        r.swap(c, p);
        for i in 0..4 {
            if i != c {
                let f = m[i][c] / m[c][c];
                for j in 0..4 {
                    m[i][j] -= f * m[c][j];
                }
                r[i] -= f * r[c];
            }
        }
    }
    Some([r[0] / m[0][0], r[1] / m[1][1], r[2] / m[2][2], r[3] / m[3][3]])
}

/// The steady camera for a frame: `steps[i]` is the movement from frame `i - radius`
/// to the next (index `radius` is from the frame itself to the next one; None at a cut
/// or outside the clip), so the path covers `radius` frames each way. Returns the
/// frame's correction: where each pixel of the steadied picture comes from in it.
pub fn steady(steps: &[Option<Sim>], radius: usize) -> Sim {
    // Path: from this frame to frame t + j, composed outward until a gap.
    let mut path: Vec<(i64, Sim)> = vec![(0, Sim::IDENTITY)];
    let mut cur = Sim::IDENTITY;
    for j in 0..radius {
        let Some(s) = steps.get(radius + j).copied().flatten() else { break };
        cur = s.after(&cur);
        path.push((j as i64 + 1, cur));
    }
    let mut cur = Sim::IDENTITY;
    for j in 0..radius {
        // Back: the inverse of the step from t - j - 1 to t - j.
        let Some(s) = steps.get(radius - 1 - j).copied().flatten() else { break };
        cur = s.inverse().after(&cur);
        path.push((-(j as i64) - 1, cur));
    }
    let sigma = (radius as f64 / 2.0).max(1.0);
    let (mut acc, mut total) = ([0f64; 4], 0.0);
    for (j, s) in &path {
        let wt = (-(*j as f64).powi(2) / (2.0 * sigma * sigma)).exp();
        for (a, v) in acc.iter_mut().zip([s.a, s.b, s.tx, s.ty]) {
            *a += wt * v;
        }
        total += wt;
    }
    let mean = Sim { a: acc[0] / total, b: acc[1] / total, tx: acc[2] / total, ty: acc[3] / total };
    // A point at p in this frame is at mean(p) for the steady camera: the steadied
    // picture at q shows this frame at mean^-1(q).
    mean.inverse()
}

/// The picture resampled: output pixel q (about the centre) from `src` at
/// s(q / zoom), Catmull-Rom, edges replicated.
const WARP: &str = r#"
layout(local_size_x = 16, local_size_y = 8, local_size_z = 1) in;
layout(push_constant) uniform P { F32 a; F32 o; int C, H, W; float sa, sb, tx, ty, zoom; } p;
float cr(float t, float a, float b, float c, float d) {
    return b + 0.5 * t * (c - a + t * (2.0 * a - 5.0 * b + 4.0 * c - d + t * (3.0 * (b - c) + d - a)));
}
float at(int c, int x, int y) { return p.a.v[(c * p.H + clamp(y, 0, p.H - 1)) * p.W + clamp(x, 0, p.W - 1)]; }
void main() {
    int x = int(gl_GlobalInvocationID.x), y = int(gl_GlobalInvocationID.y), c = int(gl_GlobalInvocationID.z);
    if (x >= p.W || y >= p.H || c >= p.C) return;
    float qx = (float(x) + 0.5 - float(p.W) * 0.5) / p.zoom, qy = (float(y) + 0.5 - float(p.H) * 0.5) / p.zoom;
    float sx = p.sa * qx - p.sb * qy + p.tx + float(p.W) * 0.5 - 0.5;
    float sy = p.sb * qx + p.sa * qy + p.ty + float(p.H) * 0.5 - 0.5;
    int x0 = int(floor(sx)), y0 = int(floor(sy));
    float fx = sx - float(x0), fy = sy - float(y0);
    float r[4];
    for (int j = 0; j < 4; j++)
        r[j] = cr(fx, at(c, x0 - 1, y0 - 1 + j), at(c, x0, y0 - 1 + j), at(c, x0 + 1, y0 - 1 + j), at(c, x0 + 2, y0 - 1 + j));
    float v = cr(fy, r[0], r[1], r[2], r[3]);
    // Alpha and colour stay in range (Catmull-Rom overshoots at edges).
    p.o.v[(c * p.H + y) * p.W + x] = c == 3 ? clamp(v, 0.0, 1.0) : v;
}
"#;

pub struct Stabiliser {
    ops: Arc<Ops>,
    warp: Arc<Kernel>,
}

/// Columns of the motion grid read back per pair.
pub const GRID_W: usize = 96;

impl Stabiliser {
    pub fn new(ops: &Arc<Ops>) -> Result<Self> {
        let warp = ops.compute.kernel("stab_warp", &format!("{PRELUDE}{WARP}"), 0)?;
        Ok(Stabiliser { ops: ops.clone(), warp })
    }

    /// The motion grid to read back for `fit`: [2][gh][GRID_W] (box averages).
    pub fn grid(&self, rec: &mut Recorder, flow: &Tensor) -> Result<Tensor> {
        let gw = GRID_W.min(flow.w);
        let gh = (flow.h * gw / flow.w).max(1);
        self.ops.resize(rec, flow, gw, gh)
    }

    /// `img` as seen by the steady camera (`correction` from `steady`), zoomed in by
    /// `zoom` (>= 1).
    pub fn warp(&self, rec: &mut Recorder, img: &Tensor, correction: Sim, zoom: f32) -> Result<Tensor> {
        let out = self.ops.tensor(img.c, img.h, img.w)?;
        let push = Push::new()
            .t(img)
            .t(&out)
            .i(img.c as i32)
            .i(img.h as i32)
            .i(img.w as i32)
            .f(correction.a as f32)
            .f(correction.b as f32)
            .f(correction.tx as f32)
            .f(correction.ty as f32)
            .f(zoom.max(1.0));
        rec.dispatch(&self.warp, Recorder::groups(&self.warp, [img.w as u32, img.h as u32, img.c as u32]), &push)?;
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fit_finds_a_similarity_despite_a_moving_object() {
        let truth = Sim { a: 1.01 * 0.02f64.cos(), b: 1.01 * 0.02f64.sin(), tx: 3.0, ty: -2.0 };
        let (gw, gh, w, h) = (64, 36, 1920, 1080);
        let mut img = crate::Image::new(gw, gh, 2);
        let n = gw * gh;
        for y in 0..gh {
            for x in 0..gw {
                let px = (x as f64 + 0.5) * w as f64 / gw as f64 - w as f64 / 2.0;
                let py = (y as f64 + 0.5) * h as f64 / gh as f64 - h as f64 / 2.0;
                let (qx, qy) = truth.apply(px, py);
                // A third of the picture moves on its own.
                let own = if x < gw / 3 { 25.0 } else { 0.0 };
                img.data[y * gw + x] = (qx - px + own) as f32;
                img.data[n + y * gw + x] = (qy - py) as f32;
            }
        }
        let s = fit(&img, w, h, 1.0).unwrap();
        assert!((s.a - truth.a).abs() < 1e-3 && (s.b - truth.b).abs() < 1e-3, "{s:?}");
        assert!((s.tx - truth.tx).abs() < 0.1 && (s.ty - truth.ty).abs() < 0.1, "{s:?}");
    }

    #[test]
    fn steady_removes_shake_and_keeps_a_pan() {
        // A steady pan of 4 px per frame with +-3 px shake: the corrected positions
        // follow the pan within a pixel.
        let shake = |k: i64| if k % 2 == 0 { 3.0 } else { -3.0 };
        let pos = |k: i64| 4.0 * k as f64 + shake(k);
        let radius = 8;
        let t = 20;
        let steps: Vec<Option<Sim>> = (0..2 * radius).map(|i| {
            let k = t - radius as i64 + i as i64;
            Some(Sim { a: 1.0, b: 0.0, tx: pos(k + 1) - pos(k), ty: 0.0 })
        }).collect();
        let c = steady(&steps, radius);
        // Frame t's content point at the origin ends at c^-1(0) in the steady picture.
        let (x, _) = c.inverse().apply(0.0, 0.0);
        assert!((x - -shake(t)).abs() < 1.0, "moved {x}, shake {}", shake(t));
    }
}
