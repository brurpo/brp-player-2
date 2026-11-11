//! Runs user shaders offline on real frames and reports what they do.
//!
//!   cargo run -p brp-render --release --example fxcheck -- <video> <frame> <out.png> <shader.glsl[:param=v,param=v]>...
//!       [--frames N]   run N consecutive frames ending at <frame> (temporal effects)
//!       [--target WxH] output size (OUTPUT in WIDTH/HEIGHT/WHEN)
//!
//! Prints NaN/Inf counts, channel means and the difference to the unprocessed frame,
//! and writes the last result (sRGB 8-bit) to <out.png>.

use brp_media::{Request, Source, VideoFrame};
use brp_render::offline_fx::{OfflineEffects, Rgba};
use brp_render::usershader;
use brp_render::{EffectSpec, GpuContext, GpuInstance, Transfer};
use std::sync::Arc;
use std::time::{Duration, Instant};

const PASSTHROUGH: &str = "//!HOOK MAIN\n//!BIND HOOKED\nvec4 hook() { return HOOKED_tex(HOOKED_pos); }\n";

fn get(src: &Source, n: i64) -> Arc<VideoFrame> {
    src.request(Request { frame: n, playing: false, dir: 1, ahead: 0, fill: false });
    let t = Instant::now();
    loop {
        if let Some(f) = src.get(n) {
            return f;
        }
        assert!(t.elapsed() < Duration::from_secs(30), "frame {n} not decoded");
        std::thread::sleep(Duration::from_millis(1));
    }
}

fn spec(id: u64, arg: &str) -> anyhow::Result<EffectSpec> {
    let (path, params) = match arg.find(".glsl:") {
        Some(i) => (&arg[..i + 5], Some(&arg[i + 6..])),
        None => (arg, None),
    };
    let (name, text) = if path == "passthrough" {
        ("passthrough".to_string(), PASSTHROUGH.to_string())
    } else {
        (std::path::Path::new(path).file_name().unwrap().to_string_lossy().to_string(), std::fs::read_to_string(path)?)
    };
    let file = usershader::parse::parse(&name, &text)?;
    let mut values: Vec<f64> = file.params.iter().map(|p| p.default).collect();
    for kv in params.into_iter().flat_map(|p| p.split(',')) {
        let (k, v) = kv.split_once('=').ok_or_else(|| anyhow::anyhow!("bad param {kv}"))?;
        let i = file.params.iter().position(|p| p.name == k).ok_or_else(|| anyhow::anyhow!("{name}: no param {k}"))?;
        values[i] = v.parse()?;
    }
    Ok(EffectSpec { id, file: Arc::new(file), params: values, enabled: true })
}

fn stats(label: &str, img: &Rgba, base: Option<&Rgba>) {
    let n = img.data.len() as f64;
    let (mut bad, mut mean, mut lo, mut hi) = (0usize, [0f64; 4], [f32::MAX; 4], [f32::MIN; 4]);
    let black = img.data.iter().filter(|p| p[0].max(p[1]).max(p[2]) < 0.004).count();
    print!("[black {:.2}%] ", black as f64 * 100.0 / n);
    for px in &img.data {
        if px.iter().any(|v| !v.is_finite()) {
            bad += 1;
            continue;
        }
        for c in 0..4 {
            mean[c] += px[c] as f64 / n;
            lo[c] = lo[c].min(px[c]);
            hi[c] = hi[c].max(px[c]);
        }
    }
    print!(
        "{label}: {}x{}, non-finite {bad}, mean rgb {:.4} {:.4} {:.4}, min {:.3} {:.3} {:.3}, max {:.3} {:.3} {:.3}",
        img.w, img.h, mean[0], mean[1], mean[2], lo[0], lo[1], lo[2], hi[0], hi[1], hi[2]
    );
    if let Some(b) = base.filter(|b| b.w == img.w && b.h == img.h) {
        let mut d = [0f64; 3];
        let mut max = 0f32;
        for (p, q) in img.data.iter().zip(&b.data) {
            for c in 0..3 {
                let x = (p[c] - q[c]).abs();
                if x.is_finite() {
                    d[c] += x as f64 / n;
                    max = max.max(x);
                }
            }
        }
        print!(", mean |diff| {:.4} {:.4} {:.4}, max {:.3}", d[0], d[1], d[2], max);
    }
    println!();
}

fn save_png(path: &str, img: &Rgba) -> anyhow::Result<()> {
    let f = std::io::BufWriter::new(std::fs::File::create(path)?);
    let mut enc = png::Encoder::new(f, img.w, img.h);
    enc.set_color(png::ColorType::Rgb);
    enc.set_depth(png::BitDepth::Eight);
    let mut w = enc.write_header()?;
    let bytes: Vec<u8> = img.data.iter().flat_map(|p| [p[0], p[1], p[2]]).map(|v| if v.is_finite() { (v.clamp(0.0, 1.0) * 255.0).round() as u8 } else { 255 }).collect();
    w.write_image_data(&bytes)?;
    Ok(())
}

fn main() -> anyhow::Result<()> {
    env_logger::init();
    let args: Vec<String> = std::env::args().skip(1).collect();
    let flag = |name: &str| args.iter().position(|a| a == name).map(|i| args[i + 1].clone());
    let frames: i64 = flag("--frames").map(|v| v.parse().unwrap()).unwrap_or(1);
    // Use every Nth frame (simulates faster motion for temporal effects).
    let step: i64 = flag("--step").map(|v| v.parse().unwrap()).unwrap_or(1);
    let pos: Vec<&String> = {
        let mut v = Vec::new();
        let mut skip = false;
        for a in &args {
            if skip {
                skip = false;
            } else if a.starts_with("--") {
                skip = true;
            } else {
                v.push(a);
            }
        }
        v
    };
    let [file, frame, out, shaders @ ..] = pos.as_slice() else {
        anyhow::bail!("usage: fxcheck <video> <frame> <out.png> <shader[:p=v,...]>... [--frames N] [--target WxH]");
    };
    let n: i64 = frame.parse()?;
    let src = Source::open(file)?;
    let t = Instant::now();
    while !src.index_ready() && t.elapsed() < Duration::from_secs(60) {
        std::thread::sleep(Duration::from_millis(5));
    }
    let first = get(&src, n);
    let target = flag("--target")
        .and_then(|t| t.split_once('x').map(|(a, b)| (a.parse().unwrap(), b.parse().unwrap())))
        .unwrap_or((first.width(), first.height()));
    let color = src.info.color;
    let transfer = match color.transfer {
        brp_media::Transfer::Pq => Transfer::Pq,
        brp_media::Transfer::Hlg => Transfer::Hlg,
        brp_media::Transfer::Linear => Transfer::Linear,
        _ => Transfer::Sdr,
    };
    let ctx = GpuContext::headless(GpuInstance::new()?, std::env::var("BRP_GPU").ok().as_deref())?;
    let mut fx = OfflineEffects::new(std::sync::Arc::new(ctx))?;
    println!("{} frame {n} {}x{} on {}", src.info.file_name, first.width(), first.height(), fx.device_name());

    // Baseline: the frame as MAIN, unprocessed.
    let errs = fx.set_effects(&[spec(1, "passthrough")?], transfer);
    assert!(errs.is_empty(), "{errs:?}");
    let (base, _) = fx.run(&first, color, n, target)?;
    stats("source", &base, None);

    let specs: Vec<EffectSpec> = shaders.iter().enumerate().map(|(i, s)| spec(10 + i as u64, s)).collect::<anyhow::Result<_>>()?;
    let t = Instant::now();
    let errs = fx.set_effects(&specs, transfer);
    println!("compiled in {:?}", t.elapsed());
    for e in &errs {
        println!("ERROR: {e}");
    }
    fx.reset_history();
    let mut last = None;
    for k in (0..frames).rev() {
        let f = get(&src, n - k * step);
        let (img, gpu) = fx.run(&f, color, n - k, target)?;
        stats(&format!("frame {} ({:.1} ms)", n - k, gpu.as_secs_f64() * 1e3), &img, Some(&base));
        last = Some(img);
    }
    let last = last.unwrap();
    save_png(out, &last)?;
    save_png(&out.replace(".png", "_src.png"), &base)?;
    // libplacebo reference (tools/plref.sh): planar G, B, R 16-bit.
    let reference = flag("--ref").map(|path| -> anyhow::Result<Rgba> {
        let raw = std::fs::read(&path)?;
        let (w, h) = flag("--ref-size")
            .and_then(|t| t.split_once('x').map(|(a, b)| (a.parse().unwrap(), b.parse().unwrap())))
            .unwrap_or((last.w, last.h));
        let n = (w * h) as usize;
        anyhow::ensure!(raw.len() == n * 6, "reference is {} bytes, expected {}", raw.len(), n * 6);
        let s = |plane: usize, i: usize| u16::from_le_bytes([raw[(plane * n + i) * 2], raw[(plane * n + i) * 2 + 1]]) as f32 / 65535.0;
        Ok(Rgba { w, h, data: (0..n).map(|i| [s(2, i), s(0, i), s(1, i), 1.0]).collect() })
    });
    let reference = reference.transpose()?;
    if let Some(r) = &reference {
        stats("libplacebo", r, Some(&base));
        // Compare with both clamped to the displayable range.
        let clamped = |img: &Rgba| Rgba { w: img.w, h: img.h, data: img.data.iter().map(|p| p.map(|v| v.clamp(0.0, 1.0))).collect() };
        stats("ours vs libplacebo", &clamped(&last), Some(r));
    }
    if let Some(c) = flag("--crop") {
        let v: Vec<u32> = c.split(',').map(|x| x.parse().unwrap()).collect();
        let zoom = flag("--zoom").map(|z| z.parse().unwrap()).unwrap_or(2);
        let mut panels = vec![&base, &last];
        panels.extend(reference.as_ref());
        save_crop(&out.replace(".png", "_crop.png"), &base, &panels, (v[0], v[1], v[2], v[3]), zoom)?;
    }
    println!("wrote {out}");
    Ok(())
}

/// Source | result side by side for a crop (x, y, w, h), each pixel scaled `zoom` times.
fn save_crop(path: &str, base: &Rgba, imgs: &[&Rgba], crop: (u32, u32, u32, u32), zoom: u32) -> anyhow::Result<()> {
    let (cx, cy, cw, ch) = crop;
    let (ow, oh) = ((cw * zoom + 4) * imgs.len() as u32, ch * zoom);
    let mut bytes = vec![0u8; (ow * oh * 3) as usize];
    for (k, im) in imgs.iter().enumerate() {
        // Sample the result at the same relative position (it may be scaled).
        let (sx, sy) = (im.w as f32 / base.w as f32, im.h as f32 / base.h as f32);
        for y in 0..oh {
            for x in 0..cw * zoom {
                let px = (((cx as f32 + (x as f32 + 0.5) / zoom as f32) * sx) as u32).min(im.w - 1);
                let py = (((cy as f32 + (y as f32 + 0.5) / zoom as f32) * sy) as u32).min(im.h - 1);
                let p = im.data[(py * im.w + px) as usize];
                let o = ((y * ow + x + k as u32 * (cw * zoom + 4)) * 3) as usize;
                for c in 0..3 {
                    bytes[o + c] = if p[c].is_finite() { (p[c].clamp(0.0, 1.0) * 255.0).round() as u8 } else { 255 };
                }
            }
        }
    }
    let f = std::io::BufWriter::new(std::fs::File::create(path)?);
    let mut enc = png::Encoder::new(f, ow, oh);
    enc.set_color(png::ColorType::Rgb);
    enc.set_depth(png::BitDepth::Eight);
    enc.write_header()?.write_image_data(&bytes)?;
    Ok(())
}
