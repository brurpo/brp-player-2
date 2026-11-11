//! Impulse removal in the denoiser (salt and pepper, fireflies): on PNG frames of one
//! shot, the middle one is denoised with the impulse filter off and on, from the
//! frames as they are and from copies with impulses added.
//!   impulsecheck out_prefix a.png b.png ...
//! BRP_SP=d: salt and pepper on that share of pixels (all channels 0 or 1).
//! BRP_FF=n: n fireflies per frame, 1 or 2 pixels, linear light 5..200 put in display
//!   space as the player does for linear sources (gamma 2.2, no ceiling).
//! BRP_IMP=k: the impulse setting (default 0.5). BRP_RADIUS: window (default 3).
//! BRP_CROP=x,y,size: crops (x3) of input / off / on / clean off / clean on.
//! BRP_RUNS: timing runs (default 5). BRP_RAW=1: impulse filter only (no merge, no DCT).
//! BRP_DUMP=1 / BRP_AT=x,y: details of what is left. BRP_GPU picks the GPU by name.

use brp_ai::denoise::{DenoiseSettings, Denoiser};
use brp_ai::gpu::Ops;
use brp_ai::gpu_temporal::Temporal;
use brp_ai::vk::Tensor;
use brp_ai::Image;
use brp_render::compute::{Compute, Recorder};
use brp_render::{GpuContext, GpuInstance};
use std::sync::Arc;

fn main() -> anyhow::Result<()> {
    env_logger::init();
    let a: Vec<String> = std::env::args().skip(1).collect();
    anyhow::ensure!(a.len() >= 4, "impulsecheck out_prefix a.png b.png c.png ...");
    let envf = |k: &str| std::env::var(k).ok().and_then(|v| v.parse::<f32>().ok());
    let prefix = &a[0];
    let clean: Vec<Image> = a[1..].iter().map(|p| load(p)).collect::<anyhow::Result<_>>()?;
    let (w, h) = (clean[0].w, clean[0].h);
    let n = w * h;
    let mid = clean.len() / 2;
    let radius = (envf("BRP_RADIUS").unwrap_or(3.0) as usize).min(mid).min(clean.len() - 1 - mid);
    let k = envf("BRP_IMP").unwrap_or(0.5);

    // Impulses, fresh on every frame; the middle frame's are remembered.
    let mut seed = 0x1234_5678u64;
    let mut rnd = move || {
        // splitmix64 (a plain LCG put fireflies at the same places on different frames).
        seed = seed.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = seed;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        ((z ^ (z >> 31)) >> 40) as f32 / (1u64 << 24) as f32
    };
    let sp = envf("BRP_SP").unwrap_or(0.0);
    let ff = envf("BRP_FF").unwrap_or(0.0) as usize;
    // Real small detail that must stay: BRP_PAN=px per frame (diagonal, sub-pixel:
    // every frame is the middle one moved, bilinear), BRP_STARS=n stars moving with it
    // (Gaussian points, sigma 0.35 px, energy kept, at sub-pixel places, 0.3..1 above the picture;
    // a fifth are HDR, linear 5..50). They are part of the clean frames.
    let mut clean = clean;
    let pan = envf("BRP_PAN").unwrap_or(0.0);
    let nstars = envf("BRP_STARS").unwrap_or(0.0) as usize;
    let stars: Vec<(f32, f32, f32)> = (0..nstars)
        .map(|_| {
            let (x, y) = (8.0 + rnd() * (w - 16) as f32, 8.0 + rnd() * (h - 16) as f32);
            let amp = if rnd() < 0.2 { (5.0 * 10f32.powf(rnd())).powf(1.0 / 2.2) } else { 0.3 + 0.7 * rnd() };
            (x, y, amp)
        })
        .collect();
    if pan != 0.0 || nstars > 0 {
        let base = clean[mid].clone();
        let at = |c: usize, x: f32, y: f32| {
            let (x, y) = (x.clamp(0.0, (w - 1) as f32), y.clamp(0.0, (h - 1) as f32));
            let (x0, y0) = (x.floor() as usize, y.floor() as usize);
            let (x1, y1) = ((x0 + 1).min(w - 1), (y0 + 1).min(h - 1));
            let (fx, fy) = (x - x0 as f32, y - y0 as f32);
            let v = |xx: usize, yy: usize| base.data[c * n + yy * w + xx];
            (v(x0, y0) * (1.0 - fx) + v(x1, y0) * fx) * (1.0 - fy) + (v(x0, y1) * (1.0 - fx) + v(x1, y1) * fx) * fy
        };
        for (fi, f) in clean.iter_mut().enumerate() {
            let s = (fi as f32 - mid as f32) * pan;
            // The scene moves by +s: this frame at x shows the base at x - s.
            for c in 0..3 {
                for y in 0..h {
                    for x in 0..w {
                        f.data[c * n + y * w + x] = at(c, x as f32 - s, y as f32 - 0.6 * s);
                    }
                }
            }
            for &(sx, sy, amp) in &stars {
                // Energy kept wherever it falls (as a sensor or an antialiased render):
                // the weights sum to 1, so a star on a pixel corner is 4 x amp / 4.
                let (cx, cy) = (sx + s, sy + 0.6 * s);
                let (ys, xs) = ((cy as usize).saturating_sub(2)..((cy as usize) + 3).min(h), (cx as usize).saturating_sub(2)..((cx as usize) + 3).min(w));
                let g = |x: usize, y: usize| (-((x as f32 - cx).powi(2) + (y as f32 - cy).powi(2)) / (2.0 * 0.35 * 0.35)).exp();
                let total: f32 = ys.clone().flat_map(|y| xs.clone().map(move |x| (x, y))).map(|(x, y)| g(x, y)).sum();
                for y in ys.clone() {
                    for x in xs.clone() {
                        for c in 0..3 {
                            f.data[c * n + y * w + x] += amp * g(x, y) / total;
                        }
                    }
                }
            }
        }
    }
    // BRP_GRAIN=sigma: fresh grain on every frame, part of the clean frames (grainy
    // footage must not lose its grain's peaks to the impulse test).
    if let Some(sigma) = envf("BRP_GRAIN") {
        for f in clean.iter_mut() {
            for v in f.data.iter_mut() {
                *v += sigma * 3.46 * (rnd() - 0.5);
            }
        }
    }
    let mut hit = vec![false; n];
    let noisy: Vec<Image> = clean
        .iter()
        .enumerate()
        .map(|(fi, f)| {
            let mut f = Image { w: f.w, h: f.h, channels: 3, data: f.data[..3 * n].to_vec() };
            let mark = |j: usize, hit: &mut Vec<bool>| {
                if fi == mid {
                    hit[j] = true;
                }
            };
            if sp > 0.0 {
                for j in 0..n {
                    if rnd() < sp {
                        let v = if rnd() < 0.5 { 0.0 } else { 1.0 };
                        for c in 0..3 {
                            f.data[c * n + j] = v;
                        }
                        mark(j, &mut hit);
                    }
                }
            }
            for _ in 0..ff {
                let (x, y) = ((rnd() * (w - 2) as f32) as usize, (rnd() * (h - 2) as f32) as usize);
                let l = 5.0 * 40f32.powf(rnd());
                let tint = [0.6 + 0.4 * rnd(), 0.6 + 0.4 * rnd(), 0.6 + 0.4 * rnd()];
                let px: &[usize] = match (rnd() * 3.0) as usize {
                    0 => &[0],
                    1 => &[0, 1],
                    _ => &[0, w],
                };
                for &o in px {
                    let j = y * w + x + o;
                    for c in 0..3 {
                        f.data[c * n + j] = (l * tint[c]).powf(1.0 / 2.2);
                    }
                    mark(j, &mut hit);
                }
            }
            f
        })
        .collect();
    println!("{}x{}, {} frames, middle {mid}, radius {radius}, impulse {k}; {} pixels hit in the middle frame", w, h, clean.len(), hit.iter().filter(|&&b| b).count());

    let ctx = Arc::new(GpuContext::headless(GpuInstance::new()?, std::env::var("BRP_GPU").ok().as_deref())?);
    let compute: Arc<Compute> = Compute::new(ctx);
    let ops = Ops::new(&compute)?;
    let tm = Temporal::new(&ops)?;
    let dn = Denoiser::new(&ops)?;
    let runs = envf("BRP_RUNS").unwrap_or(5.0) as usize;

    // Denoise the middle frame of `frames` (motion found once, outside the timing).
    let run = |frames: &[Image], imp: f32| -> anyhow::Result<(Image, f64)> {
        let t: Vec<Tensor> = frames.iter().map(|f| Tensor::from_image(&compute, f)).collect::<anyhow::Result<_>>()?;
        let mut rec = Recorder::new(&compute)?;
        let back: Vec<Tensor> = (1..=radius).map(|d| tm.motion(&mut rec, &t[mid - d], &t[mid - d + 1])).collect::<anyhow::Result<_>>()?;
        let fwd: Vec<Tensor> = (1..=radius).map(|d| tm.motion(&mut rec, &t[mid + d], &t[mid + d - 1])).collect::<anyhow::Result<_>>()?;
        rec.submit_wait()?;
        let sb: Vec<(&Tensor, &Tensor)> = (1..=radius).map(|d| (&t[mid - d], &back[d - 1])).collect();
        let sf: Vec<(&Tensor, &Tensor)> = (1..=radius).map(|d| (&t[mid + d], &fwd[d - 1])).collect();
        let mut s = DenoiseSettings { impulse: imp, ..Default::default() };
        if std::env::var("BRP_RAW").is_ok() {
            (s.temporal, s.spatial) = (0.0, 0.0);
        }
        let mut best = f64::MAX;
        let mut out = None;
        for _ in 0..runs.max(1) {
            let mut rec = Recorder::new(&compute)?;
            let o = dn.record(&mut rec, &tm, &t[mid], [&sb, &sf], &s)?;
            best = best.min(rec.submit_wait()?.as_secs_f64() * 1e3);
            out = Some(o);
        }
        Ok((out.unwrap().to_image(&compute)?, best))
    };
    let (off, t_off) = run(&noisy, 0.0)?;
    let (on, t_on) = run(&noisy, k)?;
    let (c_off, _) = run(&clean, 0.0)?;
    let (c_on, _) = run(&clean, k)?;
    println!("time (best of {runs}, denoise only): off {t_off:.2} ms, on {t_on:.2} ms, added {:.2} ms", t_on - t_off);

    // Against the clean frames denoised without the filter.
    let dev = |x: &Image, j: usize| (0..3).map(|c| (x.data[c * n + j] - c_off.data[c * n + j]).abs()).fold(0f32, f32::max);
    let near = |j: usize| {
        let (x, y) = ((j % w) as isize, (j / w) as isize);
        (-3..=3).any(|dy| (-3..=3).any(|dx| {
            let (xx, yy) = (x + dx, y + dy);
            xx >= 0 && yy >= 0 && (xx as usize) < w && (yy as usize) < h && hit[yy as usize * w + xx as usize]
        }))
    };
    for (name, x) in [("off", &off), ("on", &on)] {
        let (mut left, mut cnt, mut ring, mut rc, mut far, mut fc) = (0, 0, 0f64, 0, 0f64, 0);
        for j in 0..n {
            let d = dev(x, j);
            if hit[j] {
                cnt += 1;
                if d > 0.1 {
                    left += 1;
                }
            } else if near(j) {
                ring += d as f64;
                rc += 1;
            } else {
                far += d as f64;
                fc += 1;
            }
        }
        println!(
            "impulses {name}: left (>0.1 from clean) {left} of {cnt} ({:.1}%); mean error within 3 px of them {:.4}, elsewhere {:.5}",
            100.0 * left as f64 / cnt.max(1) as f64,
            ring / rc.max(1) as f64,
            far / fc.max(1) as f64
        );
    }
    // What is left, by kind: [firefly, salt, pepper] (left, of), and the clean
    // picture's level under the salt and pepper left.
    let mut kinds = [(0, 0); 3];
    let mut lv = [0f64; 3];
    for j in (0..n).filter(|&j| hit[j]) {
        let v = noisy[mid].data[j];
        let kd = if v > 1.0 { 0 } else if v > 0.5 { 1 } else { 2 };
        kinds[kd].1 += 1;
        if dev(&on, j) > 0.1 {
            if std::env::var("BRP_DUMP").is_ok() && kinds[kd].0 < 4 {
                let (x, y) = (j % w, j / w);
                println!("left {kd} at {x},{y}: noisy 3x3 (green) / on / clean_off:");
                for yy in y.saturating_sub(1)..=(y + 1).min(h - 1) {
                    let row = |im: &Image| (x.saturating_sub(1)..=(x + 1).min(w - 1)).map(|xx| format!("{:.2}", im.data[n + yy * w + xx])).collect::<Vec<_>>().join(" ");
                    println!("   {}   |   {}   |   {}", row(&noisy[mid]), row(&on), row(&c_off));
                }
            }
            kinds[kd].0 += 1;
            lv[kd] += clean[mid].data[j] as f64;
        }
    }
    if let Ok(v) = std::env::var("BRP_AT") {
        let p: Vec<usize> = v.split(',').filter_map(|x| x.parse().ok()).collect();
        let j = p[1] * w + p[0];
        println!("at {v}: noisy {:?}, on {:?}", (0..3).map(|c| noisy[mid].data[c * n + j]).collect::<Vec<_>>(), (0..3).map(|c| on.data[c * n + j]).collect::<Vec<_>>());
        for fi in 0..noisy.len() {
            let m = (p[1] - 1..=p[1] + 1).flat_map(|y| (p[0] - 1..=p[0] + 1).map(move |x| y * w + x)).map(|q| noisy[fi].data[q]).fold(0f32, f32::max);
            println!("  frame {fi}: 3x3 max red {m:.2}");
        }
    }
    if std::env::var("BRP_DUMP").is_ok() {
        let mut band = [0; 8];
        for j in (0..n).filter(|&j| hit[j] && dev(&on, j) > 0.1) {
            band[(j / w) * 8 / h] += 1;
        }
        println!("left per eighth of the height: {band:?}");
    }
    println!(
        "left by kind: fireflies {}/{}, salt {}/{} (clean level there {:.2}), pepper {}/{} (clean level {:.2})",
        kinds[0].0, kinds[0].1, kinds[1].0, kinds[1].1, lv[1] / kinds[1].0.max(1) as f64, kinds[2].0, kinds[2].1, lv[2] / kinds[2].0.max(1) as f64
    );
    let (mut s, mut mx, mut c2, mut c8) = (0f64, 0f32, 0, 0);
    let mut diff = Image { w, h, channels: 3, data: vec![0.0; 3 * n] };
    for j in 0..n {
        let d = dev(&c_on, j);
        s += d as f64;
        mx = mx.max(d);
        if d > 2.0 / 255.0 {
            c2 += 1;
        }
        if d > 8.0 / 255.0 {
            c8 += 1;
        }
        for c in 0..3 {
            diff.data[c * n + j] = d * 8.0;
        }
    }
    println!("clean frames, filter on vs off: mean {:.6}, max {mx:.4}, pixels over 2/255 {c2} ({:.4}%), over 8/255 {c8}", s / n as f64, 100.0 * c2 as f64 / n as f64);
    save(&format!("{prefix}_cleandiff.png"), &diff, None)?;
    if std::env::var("BRP_DUMP").is_ok() {
        let mut big: Vec<usize> = (0..n).filter(|&j| dev(&c_on, j) > 8.0 / 255.0).collect();
        big.sort_by(|&p, &q| dev(&c_on, q).total_cmp(&dev(&c_on, p)));
        let mut shown: Vec<usize> = Vec::new();
        for &j in &big {
            if shown.len() >= 4 || shown.iter().any(|&s| (s % w).abs_diff(j % w) < 4 && (s / w).abs_diff(j / w) < 4) {
                continue;
            }
            shown.push(j);
            let (x, y) = (j % w, j / w);
            let star = stars.iter().any(|&(sx, sy, _)| (sx - x as f32).abs() < 3.0 && (sy - y as f32).abs() < 3.0);
            println!("clean changed at {x},{y} (star near: {star}): clean in / off / on (green):");
            for yy in y.saturating_sub(1)..=(y + 1).min(h - 1) {
                let row = |im: &Image| (x.saturating_sub(1)..=(x + 1).min(w - 1)).map(|xx| format!("{:.2}", im.data[n + yy * w + xx])).collect::<Vec<_>>().join(" ");
                println!("   {}   |   {}   |   {}", row(&clean[mid]), row(&c_off), row(&c_on));
            }
        }
    }
    if !stars.is_empty() {
        // Each star's peak (brightest pixel of its 3x3, green), filter on vs off,
        // against its amplitude.
        let (mut lost, mut worst, mut sum) = (0, 0f32, 0f32);
        for &(sx, sy, amp) in &stars {
            let peak = |im: &Image| {
                let mut m = f32::MIN;
                for y in sy as usize - 1..=sy as usize + 1 {
                    for x in sx as usize - 1..=sx as usize + 1 {
                        m = m.max(im.data[n + y * w + x]);
                    }
                }
                m
            };
            let dim = (peak(&c_off) - peak(&c_on)) / amp;
            sum += dim.max(0.0);
            worst = worst.max(dim);
            if dim > 0.1 {
                if std::env::var("BRP_DUMP").is_ok() && lost < 6 {
                    println!("star at {sx:.2},{sy:.2} amp {amp:.2} dimmed {:.0}%: clean in / off / on (green):", 100.0 * dim);
                    for y in sy as usize - 1..=sy as usize + 1 {
                        let row = |im: &Image| (sx as usize - 1..=sx as usize + 1).map(|x| format!("{:.2}", im.data[n + y * w + x])).collect::<Vec<_>>().join(" ");
                        println!("   {}   |   {}   |   {}", row(&clean[mid]), row(&c_off), row(&c_on));
                    }
                }
                lost += 1;
            }
        }
        println!("stars ({}): dimmed over 10% by the filter {lost}, worst {:.1}%, mean {:.2}%", stars.len(), 100.0 * worst, 100.0 * sum / stars.len() as f32);
    }
    if let Some(v) = std::env::var("BRP_CROP").ok() {
        let p: Vec<usize> = v.split(',').filter_map(|x| x.parse().ok()).collect();
        let crop = (p[0], p[1], p[2]);
        for (name, x) in [("in", &noisy[mid]), ("off", &off), ("on", &on), ("clean_off", &c_off), ("clean_on", &c_on)] {
            save(&format!("{prefix}_{name}.png"), x, Some(crop))?;
        }
    }
    Ok(())
}

fn load(path: &str) -> anyhow::Result<Image> {
    let dec = png::Decoder::new(std::io::BufReader::new(std::fs::File::open(path)?));
    let mut r = dec.read_info()?;
    let mut buf = vec![0; r.output_buffer_size().unwrap()];
    let info = r.next_frame(&mut buf)?;
    let ch = info.color_type.samples();
    anyhow::ensure!(info.bit_depth == png::BitDepth::Eight && ch >= 3, "need 8-bit RGB(A) PNG");
    let (w, h) = (info.width as usize, info.height as usize);
    let mut img = Image { w, h, channels: 3, data: vec![0.0; 3 * w * h] };
    for i in 0..w * h {
        for c in 0..3 {
            img.data[c * w * h + i] = buf[i * ch + c] as f32 / 255.0;
        }
    }
    Ok(img)
}

/// 8-bit PNG (clamped), whole or a square crop shown 3x.
fn save(path: &str, img: &Image, crop: Option<(usize, usize, usize)>) -> anyhow::Result<()> {
    let n = img.w * img.h;
    let (x0, y0, sw, sh, z) = match crop {
        Some((x, y, s)) => (x, y, s.min(img.w - x), s.min(img.h - y), 3),
        None => (0, 0, img.w, img.h, 1),
    };
    let mut enc = png::Encoder::new(std::io::BufWriter::new(std::fs::File::create(path)?), (sw * z) as u32, (sh * z) as u32);
    enc.set_color(png::ColorType::Rgb);
    enc.set_depth(png::BitDepth::Eight);
    let mut bytes = Vec::with_capacity(sw * sh * z * z * 3);
    for y in 0..sh * z {
        for x in 0..sw * z {
            let j = (y0 + y / z) * img.w + x0 + x / z;
            for c in 0..3 {
                bytes.push((img.data[c * n + j].clamp(0.0, 1.0) * 255.0).round() as u8);
            }
        }
    }
    enc.write_header()?.write_image_data(&bytes)?;
    Ok(())
}
