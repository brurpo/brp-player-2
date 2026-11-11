//! Temporal window for per-frame AI models (upscaler flicker).
//!
//! Single-image models (Real-ESRGAN) invent detail from each frame on its own, so
//! grain and compression noise, different in every frame, become detail that
//! shimmers. Here each frame is first merged with its neighbours (t-2 .. t+2, moved
//! onto it along the motion) where they agree with it; the model then sees a cleaner
//! picture that changes little from one frame to the next (consecutive frames share
//! most of their window). Every frame gets the same treatment whatever was shown
//! before: no state, any frame can be made on its own.

use crate::Image;

/// Motion between two pictures: pixel (x, y) of the target picture is at
/// (x + dx, y + dy) in the other one. In this field's own pixels.
pub struct Flow {
    pub w: usize,
    pub h: usize,
    pub dx: Vec<f32>,
    pub dy: Vec<f32>,
}

impl Flow {
    /// Displacement at (x, y) in this field's pixels, bilinear, clamped to the edges.
    pub fn at(&self, x: f32, y: f32) -> (f32, f32) {
        let x = x.clamp(0.0, (self.w - 1) as f32);
        let y = y.clamp(0.0, (self.h - 1) as f32);
        let (x0, y0) = (x as usize, y as usize);
        let (x1, y1) = ((x0 + 1).min(self.w - 1), (y0 + 1).min(self.h - 1));
        let (ax, ay) = (x - x0 as f32, y - y0 as f32);
        let s = |p: &[f32]| {
            let top = p[y0 * self.w + x0] * (1.0 - ax) + p[y0 * self.w + x1] * ax;
            let bot = p[y1 * self.w + x0] * (1.0 - ax) + p[y1 * self.w + x1] * ax;
            top * (1.0 - ay) + bot * ay
        };
        (s(&self.dx), s(&self.dy))
    }
}


/// Bilinear sample of a plane at (x, y), or None outside it.
fn sample(p: &[f32], w: usize, h: usize, x: f32, y: f32) -> Option<f32> {
    if x < -0.5 || y < -0.5 || x > w as f32 - 0.5 || y > h as f32 - 0.5 {
        return None;
    }
    let x = x.clamp(0.0, (w - 1) as f32);
    let y = y.clamp(0.0, (h - 1) as f32);
    let (x0, y0) = (x as usize, y as usize);
    let (x1, y1) = ((x0 + 1).min(w - 1), (y0 + 1).min(h - 1));
    let (ax, ay) = (x - x0 as f32, y - y0 as f32);
    let top = p[y0 * w + x0] * (1.0 - ax) + p[y0 * w + x1] * ax;
    let bot = p[y1 * w + x0] * (1.0 - ax) + p[y1 * w + x1] * ax;
    Some(top * (1.0 - ay) + bot * ay)
}

/// Catmull-Rom (bicubic) sample at (x, y), or None outside the plane. Sharper than
/// bilinear: repeated fetches of the carried-over picture do not blur it frame
/// after frame.
fn sample_sharp(p: &[f32], w: usize, h: usize, x: f32, y: f32) -> Option<f32> {
    if x < -0.5 || y < -0.5 || x > w as f32 - 0.5 || y > h as f32 - 0.5 {
        return None;
    }
    let (fx, fy) = (x.floor(), y.floor());
    let (tx, ty) = (x - fx, y - fy);
    let k = |t: f32| {
        let (t2, t3) = (t * t, t * t * t);
        [(-t3 + 2.0 * t2 - t) * 0.5, (3.0 * t3 - 5.0 * t2 + 2.0) * 0.5, (-3.0 * t3 + 4.0 * t2 + t) * 0.5, (t3 - t2) * 0.5]
    };
    let (kx, ky) = (k(tx), k(ty));
    let (x0, y0) = (fx as isize - 1, fy as isize - 1);
    let mut v = 0.0;
    for (j, wy) in ky.iter().enumerate() {
        let yy = (y0 + j as isize).clamp(0, h as isize - 1) as usize;
        let row = &p[yy * w..(yy + 1) * w];
        let mut r = 0.0;
        for (i, wx) in kx.iter().enumerate() {
            r += wx * row[(x0 + i as isize).clamp(0, w as isize - 1) as usize];
        }
        v += wy * r;
    }
    Some(v)
}

/// Rows of `out` (planes of `w`) in parallel: `f(y, row)` for each plane `c`.
fn par_rows(out: &mut Image, f: impl Fn(usize, usize, &mut [f32]) + Sync) {
    let (w, h) = (out.w, out.h);
    let threads = std::thread::available_parallelism().map_or(4, |n| n.get()).min(16);
    let rows_per = h.div_ceil(threads).max(1);
    std::thread::scope(|sc| {
        for (c, plane) in out.data.chunks_mut(w * h).enumerate() {
            for (k, chunk) in plane.chunks_mut(w * rows_per).enumerate() {
                let f = &f;
                sc.spawn(move || {
                    for (r, row) in chunk.chunks_mut(w).enumerate() {
                        f(c, k * rows_per + r, row);
                    }
                });
            }
        }
    });
}

fn luma(img: &Image) -> Vec<f32> {
    let n = img.w * img.h;
    (0..n).map(|i| 0.299 * img.data[i] + 0.587 * img.data[n + i] + 0.114 * img.data[2 * n + i]).collect()
}

/// Sum over a (2r+1)² window (edges clamped), separable.
fn box_sum(p: &[f32], w: usize, h: usize, r: usize) -> Vec<f32> {
    let mut tmp = vec![0.0; w * h];
    for y in 0..h {
        let row = &p[y * w..(y + 1) * w];
        for x in 0..w {
            tmp[y * w + x] = row[x.saturating_sub(r)..=(x + r).min(w - 1)].iter().sum();
        }
    }
    let mut out = vec![0.0; w * h];
    for y in 0..h {
        for x in 0..w {
            out[y * w + x] = (y.saturating_sub(r)..=(y + r).min(h - 1)).map(|yy| tmp[yy * w + x]).sum();
        }
    }
    out
}

/// Half size (2x2 average).
fn half(p: &[f32], w: usize, h: usize) -> (Vec<f32>, usize, usize) {
    let (hw, hh) = (w / 2, h / 2);
    let mut out = vec![0.0; hw * hh];
    for y in 0..hh {
        for x in 0..hw {
            let i = 2 * y * w + 2 * x;
            out[y * hw + x] = (p[i] + p[i + 1] + p[i + w] + p[i + w + 1]) * 0.25;
        }
    }
    (out, hw, hh)
}

/// Motion between two pictures (pixel (x, y) of `cur` is at (x + dx, y + dy) in
/// `prev`), at their size: pyramidal Lucas-Kanade on luma, coarse to fine (handles
/// some 30 px per frame), 7x7 windows so grain averages out, and the motion of flat
/// areas filled in from their surroundings.
pub fn motion(prev: &Image, cur: &Image) -> Flow {
    let (w, h) = (cur.w, cur.h);
    let t0 = std::time::Instant::now();
    let mut pyr = vec![(luma(prev), luma(cur), w, h)];
    while pyr.len() < 5 {
        let (lp, lc, lw, lh) = pyr.last().unwrap();
        if *lw < 64 || *lh < 64 {
            break;
        }
        let (p2, w2, h2) = half(lp, *lw, *lh);
        let (c2, _, _) = half(lc, *lw, *lh);
        pyr.push((p2, c2, w2, h2));
    }
    let (_, _, cw, ch) = pyr.last().unwrap();
    let (mut dx, mut dy, mut fw, mut fh) = (vec![0.0f32; cw * ch], vec![0.0f32; cw * ch], *cw, *ch);
    // Large pictures: the motion is estimated at half size (the field is
    // resolution-free; warping scales it), about 4x less work at full HD.
    let top = if w > MOTION_MAX_W && pyr.len() > 2 && std::env::var_os("BRP_MOTION_FULL").is_none() { 1 } else { 0 };
    // Undetermined motion is filled in one level above the finest (or at the finest
    // for small pictures), which then gets a single refining pass.
    let fill_level = top + (pyr.len() > top + 1) as usize;
    for (level, (lp, lc, lw, lh)) in pyr.iter().enumerate().rev().filter(|(l, _)| *l >= top) {
        if (fw, fh) != (*lw, *lh) {
            // Up a level: twice the size, twice the motion.
            let (mut ux, mut uy) = (vec![0.0; lw * lh], vec![0.0; lw * lh]);
            for y in 0..*lh {
                for x in 0..*lw {
                    let (sx, sy) = ((x / 2).min(fw - 1), (y / 2).min(fh - 1));
                    ux[y * lw + x] = dx[sy * fw + sx] * 2.0;
                    uy[y * lw + x] = dy[sy * fw + sx] * 2.0;
                }
            }
            (dx, dy, fw, fh) = (ux, uy, *lw, *lh);
        }
        let finest = level == top;
        let iterations = if finest && fill_level > top { 1 } else { 3 };
        let mut conf = Vec::new();
        for _ in 0..iterations {
            conf = lk_step(&mut dx, &mut dy, lp, lc, fw, fh, 3);
        }
        if level == fill_level {
            fill(&mut dx, &mut dy, &conf, fw, fh, FILL_RADIUS >> level);
            if finest {
                conf = lk_step(&mut dx, &mut dy, lp, lc, fw, fh, 3);
            }
        }
        if finest {
            // Real motion is smooth; per-pixel estimates are noisy (grain), and the
            // noise would interpolate each frame's neighbours slightly differently.
            smooth_flow(&mut dx, &mut dy, &conf, fw, fh, SMOOTH_RADIUS >> top);
        }
    }
    log::trace!("motion: {fw}x{fh} in {:?}", t0.elapsed());
    Flow { w: fw, h: fh, dx, dy }
}

/// Motion averaged over a (2r+1)² window, weighted by how well each pixel's motion
/// is determined (its structure).
fn smooth_flow(dx: &mut [f32], dy: &mut [f32], structure: &[f32], w: usize, h: usize, r: usize) {
    let weight: Vec<f32> = structure.iter().map(|&l| l / (l + STRUCTURE) + 1e-3).collect();
    let wx: Vec<f32> = weight.iter().zip(dx.iter()).map(|(k, d)| k * d).collect();
    let wy: Vec<f32> = weight.iter().zip(dy.iter()).map(|(k, d)| k * d).collect();
    let (sw, sx, sy) = (box_blur(&weight, w, h, r), box_blur(&wx, w, h, r), box_blur(&wy, w, h, r));
    for i in 0..w * h {
        dx[i] = sx[i] / sw[i];
        dy[i] = sy[i] / sw[i];
    }
}

/// Where the picture has no structure the motion is not determined (any motion
/// matches a flat area): fill it from the well-determined motion around.
fn fill(dx: &mut [f32], dy: &mut [f32], structure: &[f32], w: usize, h: usize, radius: usize) {
    let weight: Vec<f32> = structure.iter().map(|&l| l / (l + STRUCTURE)).collect();
    let (wx, wy): (Vec<f32>, Vec<f32>) = (0..w * h).map(|i| (weight[i] * dx[i], weight[i] * dy[i])).unzip();
    let sums: Vec<Vec<f32>> = std::thread::scope(|sc| {
        let hs: Vec<_> = [&weight, &wx, &wy].into_iter().map(|p| sc.spawn(move || box_sum(p, w, h, radius))).collect();
        hs.into_iter().map(|j| j.join().unwrap()).collect()
    });
    for i in 0..w * h {
        if sums[0][i] > 1e-3 {
            let k = weight[i];
            dx[i] = k * dx[i] + (1.0 - k) * sums[1][i] / sums[0][i];
            dy[i] = k * dy[i] + (1.0 - k) * sums[2][i] / sums[0][i];
        }
    }
}

/// Structure (smallest eigenvalue of the 7x7 gradient tensor) at which motion counts
/// as half determined.
const STRUCTURE: f32 = 0.02;
/// Radius over which undetermined motion is filled from its surroundings.
const FILL_RADIUS: usize = 12;
/// Radius over which the final motion is averaged (weighted by structure).
const SMOOTH_RADIUS: usize = 8;
/// Wider pictures get their motion at half size.
const MOTION_MAX_W: usize = 1280;

/// One Lucas-Kanade update of the motion (dx, dy) from `lc` back to `lp`. Returns
/// the structure per pixel (smallest eigenvalue of the windowed gradient tensor).
fn lk_step(dx: &mut [f32], dy: &mut [f32], lp: &[f32], lc: &[f32], w: usize, h: usize, r: usize) -> Vec<f32> {
    let n = w * h;
    // Previous picture moved by the current estimate.
    let mut warped = vec![0.0f32; n];
    {
        let (dx, dy) = (&*dx, &*dy);
        par_chunks(&mut warped, w, |y, row| {
            for (x, v) in row.iter_mut().enumerate() {
                let i = y * w + x;
                *v = sample(lp, w, h, x as f32 + dx[i], y as f32 + dy[i]).unwrap_or(f32::NAN);
            }
        });
    }
    // Gradient tensor and error terms per pixel: gx², gxgy, gy², gx·e, gy·e.
    let mut terms = vec![[0.0f32; 5]; n];
    par_chunks(&mut terms, w, |y, row| {
        if y == 0 || y + 1 >= h {
            return;
        }
        for x in 1..w - 1 {
            let i = y * w + x;
            let (gx, gy) = ((warped[i + 1] - warped[i - 1]) * 0.5, (warped[i + w] - warped[i - w]) * 0.5);
            let e = lc[i] - warped[i];
            if gx.is_finite() && gy.is_finite() && e.is_finite() {
                row[x] = [gx * gx, gx * gy, gy * gy, gx * e, gy * e];
            }
        }
    });
    // 7x7 window sums (separable).
    let mut tmp = vec![[0.0f32; 5]; n];
    par_chunks(&mut tmp, w, |y, row| {
        let src = &terms[y * w..(y + 1) * w];
        for (x, o) in row.iter_mut().enumerate() {
            for t in &src[x.saturating_sub(r)..=(x + r).min(w - 1)] {
                for k in 0..5 {
                    o[k] += t[k];
                }
            }
        }
    });
    let mut sums = vec![[0.0f32; 5]; n];
    par_chunks(&mut sums, w, |y, row| {
        for yy in y.saturating_sub(r)..=(y + r).min(h - 1) {
            for (o, t) in row.iter_mut().zip(&tmp[yy * w..(yy + 1) * w]) {
                for k in 0..5 {
                    o[k] += t[k];
                }
            }
        }
    });
    // Solve per pixel.
    let mut structure = vec![0.0f32; n];
    let threads = std::thread::available_parallelism().map_or(4, |n| n.get()).min(16);
    let rows = h.div_ceil(threads).max(1) * w;
    std::thread::scope(|sc| {
        for (k, ((cx, cy), cs)) in dx.chunks_mut(rows).zip(dy.chunks_mut(rows)).zip(structure.chunks_mut(rows)).enumerate() {
            let sums = &sums;
            sc.spawn(move || {
                for j in 0..cs.len() {
                    let [a, b, c, ex, ey] = sums[k * rows + j];
                    let s = ((a + c) * 0.5 - (((a - c) * 0.5).powi(2) + b * b).sqrt()).max(0.0);
                    cs[j] = s;
                    let det = a * c - b * b;
                    // Too little structure to say anything (flat area, or an edge in one direction).
                    if det < 1e-8 || s < 1e-4 {
                        continue;
                    }
                    cx[j] += ((c * ex - b * ey) / det).clamp(-1.0, 1.0);
                    cy[j] += ((a * ey - b * ex) / det).clamp(-1.0, 1.0);
                }
            });
        }
    });
    structure
}

/// `f(y, row)` for each row of `data` (rows of `w`), in parallel.
fn par_chunks<T: Send>(data: &mut [T], w: usize, f: impl Fn(usize, &mut [T]) + Sync) {
    let h = data.len() / w.max(1);
    let threads = std::thread::available_parallelism().map_or(4, |n| n.get()).min(16);
    let rows_per = h.div_ceil(threads).max(1);
    std::thread::scope(|sc| {
        for (k, chunk) in data.chunks_mut(w * rows_per).enumerate() {
            let f = &f;
            sc.spawn(move || {
                for (r, row) in chunk.chunks_mut(w).enumerate() {
                    f(k * rows_per + r, row);
                }
            });
        }
    });
}

/// Box blur (mean over a (2r+1)² window, edges clamped), separable and in parallel.
fn box_blur(p: &[f32], w: usize, h: usize, r: usize) -> Vec<f32> {
    let mut tmp = vec![0.0f32; w * h];
    par_chunks(&mut tmp, w, |y, row| {
        let src = &p[y * w..(y + 1) * w];
        // Running sum along the row.
        let mut s: f32 = (0..=r).map(|x| src[x.min(w - 1)]).sum::<f32>() + r as f32 * src[0];
        for (x, o) in row.iter_mut().enumerate() {
            *o = s;
            let add = src[(x + r + 1).min(w - 1)];
            let sub = src[x.saturating_sub(r)];
            s += add - sub;
        }
    });
    let mut out = vec![0.0f32; w * h];
    let k = 1.0 / ((2 * r + 1) * (2 * r + 1)) as f32;
    par_chunks(&mut out, w, |y, row| {
        for yy in y as isize - r as isize..=y as isize + r as isize {
            let yy = yy.clamp(0, h as isize - 1) as usize;
            for (o, t) in row.iter_mut().zip(&tmp[yy * w..(yy + 1) * w]) {
                *o += t;
            }
        }
        row.iter_mut().for_each(|o| *o *= k);
    });
    out
}

/// Agreement below which (on average over the picture) a neighbour is taken to be
/// another shot (scene cut).
pub const CUT: f32 = 0.3;
/// Weight of the pictures 2 away (more likely misaligned) relative to those 1 away.
pub const FAR_WEIGHT: f32 = 0.6;
/// Source mismatch (mean RGB difference, after averaging) above which a moved
/// neighbour is not used.
const TRUST: f32 = 0.05;
/// Mismatch treated as noise (fully used below).
const NOISE: f32 = 0.006;

/// Motion `a` then `b`: from the target of `a`, through `a`'s other picture, to
/// `b`'s other picture (e.g. t -> t-1 then t-1 -> t-2 gives t -> t-2).
pub fn chain(a: &Flow, b: &Flow) -> Flow {
    let (w, h) = (a.w, a.h);
    let (mut dx, mut dy) = (vec![0.0; w * h], vec![0.0; w * h]);
    let kb = b.w as f32 / w as f32;
    for y in 0..h {
        for x in 0..w {
            let i = y * w + x;
            let (ax, ay) = (a.dx[i], a.dy[i]);
            let (bx, by) = b.at((x as f32 + ax + 0.5) * kb - 0.5, (y as f32 + ay + 0.5) * kb - 0.5);
            dx[i] = ax + bx / kb;
            dy[i] = ay + by / kb;
        }
    }
    Flow { w, h, dx, dy }
}

/// `img` moved onto the target picture of `flow` (sharp interpolation). Pixels that
/// come from outside `img` are NaN.
pub fn warp(img: &Image, flow: &Flow) -> Image {
    let (w, h) = (img.w, img.h);
    let k = flow.w as f32 / w as f32;
    let mut out = Image::new(w, h, img.channels);
    par_rows(&mut out, |c, y, row| {
        for (x, o) in row.iter_mut().enumerate() {
            let (dx, dy) = flow.at((x as f32 + 0.5) * k - 0.5, (y as f32 + 0.5) * k - 0.5);
            *o = sample_sharp(img.plane(c), w, h, x as f32 + dx / k, y as f32 + dy / k).unwrap_or(f32::NAN);
        }
    });
    out
}

/// How much `warped` (a neighbour moved onto `cur`) agrees with `cur`, per pixel,
/// 0..1. The signed difference is averaged over 5x5 first: grain is zero-mean and
/// cancels, real mismatches (occlusion, wrong motion, another shot) do not.
pub fn agreement(cur: &Image, warped: &Image) -> Vec<f32> {
    let (w, h) = (cur.w, cur.h);
    let n = w * h;
    let mut sum = vec![0.0f32; n];
    let mut outside = vec![0.0f32; n];
    for c in 0..3 {
        let d: Vec<f32> = cur.plane(c).iter().zip(warped.plane(c)).map(|(a, b)| if b.is_nan() { 0.0 } else { a - b }).collect();
        for (s, v) in sum.iter_mut().zip(box_blur(&d, w, h, 2)) {
            *s += v.abs() / 3.0;
        }
    }
    for (o, v) in outside.iter_mut().zip(warped.plane(0)) {
        *o = v.is_nan() as u8 as f32;
    }
    let outside = box_blur(&outside, w, h, 2);
    sum.iter().zip(&outside).map(|(e, o)| if *o > 0.0 { 0.0 } else { (1.0 - (e - NOISE).max(0.0) / TRUST).clamp(0.0, 1.0) }).collect()
}

pub fn mean(v: &[f32]) -> f32 {
    v.iter().sum::<f32>() / v.len().max(1) as f32
}


/// Neighbour values are kept within mean ± CLAMP_SD standard deviations of the
/// current picture's 3x3 around the pixel (as TAA's variance clipping). Grain sets
/// the spread, so grain-sized differences pass; a misplaced thin detail does not.
const CLAMP_SD: f32 = 1.25;
/// Per-pixel difference to `cur`, in multiples of the typical one, up to which a
/// neighbour counts fully, and from which it does not count.
const PIXEL_KEEP: f32 = 2.5;
const PIXEL_REJECT: f32 = 5.0;

/// `cur` merged with moved neighbours: each `(picture, agreement, weight)` counts
/// `weight * agreement` per pixel (`cur` counts 1), after its value is clamped to
/// what `cur` has around the pixel (against ghosts of details the agreement, an
/// average over 5x5, cannot see). `strength` 0..1 goes from `cur` to the full merge.
/// Colour only; any alpha is `cur`'s.
pub fn fuse(cur: &Image, neighbours: &[(&Image, &[f32], f32)], strength: f32) -> Image {
    let (w, h) = (cur.w, cur.h);
    let clamp = std::env::var_os("BRP_NO_CLAMP").is_none();
    let mut out = cur.clone();
    let n = w * h;
    // Per neighbour: the typical per-pixel difference to `cur` (median over a sample).
    // Where aligned it is grain; a pixel far beyond it is something else (a moving
    // thin detail the 5x5 agreement cannot see) and is not used.
    let typical: Vec<f32> = neighbours
        .iter()
        .map(|(img, _, _)| {
            let mut d: Vec<f32> = (0..n).step_by(7).map(|i| (0..3).map(|c| (cur.plane(c)[i] - img.plane(c)[i]).abs()).sum::<f32>() / 3.0).filter(|v| v.is_finite()).collect();
            if d.is_empty() {
                return 1.0;
            }
            let m = d.len() / 2;
            d.select_nth_unstable_by(m, f32::total_cmp).1.max(0.002)
        })
        .collect();
    let mut colour = Image { w, h, channels: 3, data: out.data[..3 * w * h].to_vec() };
    par_rows(&mut colour, |c, y, row| {
        let plane = cur.plane(c);
        let (r0, r1) = (y.saturating_sub(1), (y + 1).min(h - 1));
        for (x, o) in row.iter_mut().enumerate() {
            let i = y * w + x;
            let u = *o;
            let (lo, hi) = if clamp {
                let (mut s, mut s2, mut n) = (0.0, 0.0, 0.0);
                for yy in r0..=r1 {
                    for v in &plane[yy * w + x.saturating_sub(1)..=yy * w + (x + 1).min(w - 1)] {
                        s += v;
                        s2 += v * v;
                        n += 1.0;
                    }
                }
                let mean = s / n;
                let sd = (s2 / n - mean * mean).max(0.0).sqrt();
                (mean - CLAMP_SD * sd, mean + CLAMP_SD * sd)
            } else {
                (f32::NEG_INFINITY, f32::INFINITY)
            };
            let (mut s, mut wsum) = (u, 1.0);
            for (j, (img, agree, weight)) in neighbours.iter().enumerate() {
                let v = img.plane(c)[i];
                let d = (0..3).map(|cc| (cur.plane(cc)[i] - img.plane(cc)[i]).abs()).sum::<f32>() / 3.0;
                let m = typical[j];
                let pixel = ((PIXEL_REJECT * m - d) / ((PIXEL_REJECT - PIXEL_KEEP) * m)).clamp(0.0, 1.0);
                let k = agree[i] * weight * pixel;
                if k > 0.0 && v.is_finite() {
                    s += k * v.clamp(lo, hi);
                    wsum += k;
                }
            }
            *o = u + strength * (s / wsum - u);
        }
    });
    out.data[..3 * w * h].copy_from_slice(&colour.data);
    out
}

/// Motion blur of `cur` from its motion to the previous picture (`back`) and to the
/// next one (`fwd`; either may be missing at the ends of the clip): a camera
/// shutter open for `shutter` of the frame time (0.5 = 180°), centred on the frame.
///
/// A gather, as in McGuire et al. "A reconstruction filter for plausible motion
/// blur": each pixel samples along the largest motion around it (16 px tiles), so
/// the blur of a moving object also spreads over the still background beside it;
/// samples are weighted by whose motion covers the pixel, faster counting as nearer.
pub fn motion_blur(cur: &Image, back: Option<&Flow>, fwd: Option<&Flow>, shutter: f32, samples: usize) -> Image {
    let (w, h) = (cur.w, cur.h);
    let n = w * h;
    // Blur vector per pixel (pixels over the exposure): half the motion from the
    // previous to the next picture, times the shutter.
    let at = |f: &Flow, x: usize, y: usize| {
        let k = f.w as f32 / w as f32;
        let (dx, dy) = f.at((x as f32 + 0.5) * k - 0.5, (y as f32 + 0.5) * k - 0.5);
        (dx / k, dy / k)
    };
    let mut vel = vec![[0.0f32; 2]; n];
    par_chunks(&mut vel, w, |y, row| {
        for (x, v) in row.iter_mut().enumerate() {
            let m = match (back, fwd) {
                (Some(b), Some(f)) => {
                    let ((bx, by), (fx, fy)) = (at(b, x, y), at(f, x, y));
                    ((fx - bx) * 0.5, (fy - by) * 0.5)
                }
                (None, Some(f)) => at(f, x, y),
                (Some(b), None) => {
                    let (bx, by) = at(b, x, y);
                    (-bx, -by)
                }
                (None, None) => (0.0, 0.0),
            };
            *v = [m.0 * shutter, m.1 * shutter];
        }
    });
    // Largest motion per 16x16 tile, then over each tile's neighbours.
    const TILE: usize = 16;
    let (tw, th) = (w.div_ceil(TILE), h.div_ceil(TILE));
    let len2 = |v: [f32; 2]| v[0] * v[0] + v[1] * v[1];
    let mut tiles = vec![[0.0f32; 2]; tw * th];
    for ty in 0..th {
        for tx in 0..tw {
            let mut best = [0.0f32; 2];
            for y in ty * TILE..((ty + 1) * TILE).min(h) {
                for v in &vel[y * w + tx * TILE..y * w + ((tx + 1) * TILE).min(w)] {
                    if len2(*v) > len2(best) {
                        best = *v;
                    }
                }
            }
            tiles[ty * tw + tx] = best;
        }
    }
    let mut neigh = vec![[0.0f32; 2]; tw * th];
    for ty in 0..th {
        for tx in 0..tw {
            let mut best = [0.0f32; 2];
            for y in ty.saturating_sub(1)..=(ty + 1).min(th - 1) {
                for x in tx.saturating_sub(1)..=(tx + 1).min(tw - 1) {
                    if len2(tiles[y * tw + x]) > len2(best) {
                        best = tiles[y * tw + x];
                    }
                }
            }
            neigh[ty * tw + tx] = best;
        }
    }

    let cone = |d: f32, v: f32| (1.0 - d / v.max(1e-3)).clamp(0.0, 1.0);
    let smooth = |a: f32, b: f32, x: f32| {
        let t = ((x - a) / (b - a)).clamp(0.0, 1.0);
        t * t * (3.0 - 2.0 * t)
    };
    let cylinder = |d: f32, v: f32| 1.0 - smooth(0.95 * v, 1.05 * v + 1e-3, d);
    // Soft "in front of": faster motion counts as nearer.
    let front = |a: f32, b: f32| (1.0 + (a - b) * 0.5).clamp(0.0, 1.0);
    let samples = samples.max(2);
    let mut out = cur.clone();
    let colour = 3.min(cur.channels);
    let mut rgb = Image { w, h, channels: colour, data: out.data[..colour * n].to_vec() };
    let (r, rest) = rgb.data.split_at_mut(n);
    let (g, b) = rest.split_at_mut(n);
    let threads = std::thread::available_parallelism().map_or(4, |n| n.get()).min(16);
    let rows_per = h.div_ceil(threads).max(1);
    std::thread::scope(|sc| {
        for (k, ((cr, cg), cb)) in r.chunks_mut(w * rows_per).zip(g.chunks_mut(w * rows_per)).zip(b.chunks_mut(w * rows_per)).enumerate() {
            let (vel, neigh) = (&vel, &neigh);
            sc.spawn(move || {
                let mut rows = [cr, cg, cb];
                for r in 0..rows[0].len() / w {
                    let y = k * rows_per + r;
                    for x in 0..w {
                        let vn = neigh[(y / TILE) * tw + x / TILE];
                        if len2(vn) < 0.25 {
                            continue;
                        }
                        let i = y * w + x;
                        let vx = vel[i];
                        let lx = len2(vx).sqrt();
                        // Fixed per-pixel jitter hides the sampling pattern (no
                        // randomness: the same frame always gives the same picture).
                        let jitter = ((((x as u32).wrapping_mul(73856093) ^ (y as u32).wrapping_mul(19349663)) % 1024) as f32 / 1024.0) - 0.5;
                        let mut wsum = 1.0 / lx.max(1.0);
                        let mut sum = [0.0f32; 3];
                        for c in 0..3 {
                            sum[c] = cur.plane(c)[i] * wsum;
                        }
                        for s in 0..samples {
                            if s == (samples - 1) / 2 {
                                continue;
                            }
                            let t = -0.5 + (s as f32 + jitter * 0.95 + 0.5) / samples as f32;
                            let (ox, oy) = (vn[0] * t, vn[1] * t);
                            let d = (ox * ox + oy * oy).sqrt();
                            let (px, py) = (x as f32 + ox, y as f32 + oy);
                            let (qx, qy) = (px.round().clamp(0.0, (w - 1) as f32) as usize, py.round().clamp(0.0, (h - 1) as f32) as usize);
                            let ly = len2(vel[qy * w + qx]).sqrt();
                            let wgt = front(ly, lx) * cone(d, ly * 0.5) + front(lx, ly) * cone(d, lx * 0.5) + cylinder(d, ly * 0.5) * cylinder(d, lx * 0.5) * 2.0;
                            if wgt <= 0.0 {
                                continue;
                            }
                            for (c, acc) in sum.iter_mut().enumerate() {
                                *acc += wgt * sample(cur.plane(c), w, h, px.clamp(0.0, (w - 1) as f32), py.clamp(0.0, (h - 1) as f32)).unwrap_or(0.0);
                            }
                            wsum += wgt;
                        }
                        for (c, row) in rows.iter_mut().enumerate() {
                            row[r * w + x] = sum[c] / wsum;
                        }
                    }
                }
            });
        }
    });
    out.data[..colour * n].copy_from_slice(&rgb.data);
    out
}
