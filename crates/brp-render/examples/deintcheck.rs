//! Golden test: our GPU deinterlacers vs FFmpeg's `nnedi` and `bwdif` on the same frame.
//!
//!   cargo run -p brp-render --release --example deintcheck -- <file> <frame> [ffmpeg.exe]
//!
//! Needs a GPL FFmpeg (for the `nnedi` filter) and `nnedi3_weights.bin`.

use brp_media::{Request, Source, VideoFrame};
use brp_render::offline::{Offline, Plane};
use brp_render::{DeintMethod, DeintParams, GpuContext, GpuInstance};
use std::sync::Arc;
use std::time::{Duration, Instant};

fn get(src: &Source, n: i64) -> Arc<VideoFrame> {
    src.request(Request { frame: n, playing: false, dir: 1 });
    let t = Instant::now();
    loop {
        if let Some(f) = src.get(n) {
            return f;
        }
        assert!(t.elapsed() < Duration::from_secs(20), "frame {n} not decoded");
        std::thread::sleep(Duration::from_millis(1));
    }
}

/// Planar 8-bit frame from FFmpeg (rawvideo, yuv420p).
fn ffmpeg_frame(ffmpeg: &str, file: &str, filter: &str, pick: usize, w: usize, h: usize, cwd: &std::path::Path) -> Vec<u8> {
    let file = std::path::absolute(file).unwrap();
    let out = std::process::Command::new(ffmpeg)
        .current_dir(cwd)
        .args(["-v", "error", "-i", file.to_str().unwrap(), "-vf", filter, "-fps_mode", "passthrough", "-f", "rawvideo", "-pix_fmt", "yuv420p", "-"])
        .output()
        .expect("run ffmpeg");
    assert!(out.status.success(), "ffmpeg failed: {}", String::from_utf8_lossy(&out.stderr));
    let size = w * h * 3 / 2;
    assert!(out.stdout.len() >= size * (pick + 1), "ffmpeg gave {} bytes", out.stdout.len());
    out.stdout[size * pick..size * (pick + 1)].to_vec()
}

fn compare(name: &str, ours: &[Plane], reference: &[u8], trunc: bool) {
    let mut off = 0;
    for (p, plane) in ours.iter().take(3).enumerate() {
        let n = (plane.w * plane.h) as usize;
        let r = &reference[off..off + n];
        off += n;
        let (mut max, mut sum, mut exact, mut se) = (0i32, 0u64, 0usize, 0f64);
        for (a, &b) in plane.data.iter().zip(r) {
            // FFmpeg converts float results to integers by truncation (av_clip_uint8).
            let a = if trunc { (a * 255.0 + 1e-3).floor() } else { (a * 255.0).round() }.clamp(0.0, 255.0) as i32;
            let d = (a - b as i32).abs();
            max = max.max(d);
            sum += d as u64;
            exact += (d == 0) as usize;
            se += (d * d) as f64;
        }
        let mse = se / n as f64;
        let psnr = if mse == 0.0 { f64::INFINITY } else { 10.0 * (255.0f64 * 255.0 / mse).log10() };
        println!(
            "  {name} plane {p} ({}x{}): exact {:.2}%, mean |d| {:.4}, max |d| {max}, PSNR {psnr:.2} dB",
            plane.w,
            plane.h,
            100.0 * exact as f64 / n as f64,
            sum as f64 / n as f64
        );
    }
}

fn main() -> anyhow::Result<()> {
    env_logger::init();
    brp_media::init();
    let mut args = std::env::args().skip(1);
    let file = args.next().expect("file");
    let n: i64 = args.next().and_then(|s| s.parse().ok()).unwrap_or(10);
    let ffmpeg = args.next().unwrap_or_else(|| "ffmpeg".into());
    let weights = std::env::var("BRP_NNEDI_WEIGHTS")
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|_| std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("assets/nnedi3_weights.bin"));

    let src = Source::open(&file)?;
    while !src.index_ready() {
        std::thread::sleep(Duration::from_millis(1));
    }
    let (prev, cur, next) = (get(&src, n - 1), get(&src, n), get(&src, n + 1));
    let (w, h) = (cur.width() as usize, cur.height() as usize);
    println!("{} frame {n}: {}x{} {:?}", src.info.file_name, w, h, cur.layout);
    let tb = src.info.time_base;
    println!("  pts {} = {:.3}s absolute (prev {}, next {})", cur.pts, cur.pts as f64 * tb.0 as f64 / tb.1 as f64, prev.pts, next.pts);

    if std::env::var_os("RAW_ALIGN").is_some() {
        // Which FFmpeg frame number holds the same picture as our frame n?
        let (y, stride) = cur.plane(0);
        for k in n - 3..=n + 3 {
            let r = ffmpeg_frame(&ffmpeg, &file, &format!("select=eq(n\\,{k})"), 0, w, h, std::path::Path::new("."));
            let same = (0..h).all(|row| y[row * stride..row * stride + w] == r[row * w..row * w + w]);
            println!("  ffmpeg n={k}: {}", if same { "IDENTICAL" } else { "different" });
        }
        return Ok(());
    }
    let ctx = GpuContext::headless(GpuInstance::new()?, std::env::var("BRP_GPU").ok().as_deref())?;
    let mut off = Offline::new(ctx, Some(weights.clone()))?;
    println!("GPU: {}", off.device_name());

    let top = DeintParams { method: DeintMethod::Nnedi, tff: true, second_field: false, light: false };
    // Warm-up run (pipeline creation, weights upload), then timed run.
    off.deinterlace(&prev, &cur, &next, top)?;
    let (nn, t_nn) = off.deinterlace(&prev, &cur, &next, top)?;
    let wdir = weights.parent().unwrap();
    let ref_nn = ffmpeg_frame(&ffmpeg, &file, &format!("select=eq(n\\,{n}),nnedi=weights=nnedi3_weights.bin:field=t:deint=all"), 0, w, h, wdir);
    println!("NNEDI3 (GPU {t_nn:?} incl. readback):");
    compare("nnedi", &nn, &ref_nn, true);

    let bw = DeintParams { method: DeintMethod::Bwdif, tff: true, second_field: false, light: false };
    let (b, t_b) = off.deinterlace(&prev, &cur, &next, bw)?;
    println!("bwdif (GPU {t_b:?}):");
    // bwdif delays one frame; find the output whose kept (top) lines equal frame n.
    let filter = format!("select=between(n\\,{}\\,{}),bwdif=mode=send_frame:parity=tff:deint=all", n - 1, n + 1);
    let (y, stride) = cur.plane(0);
    let pick = (0..3)
        .find(|&k| {
            let r = ffmpeg_frame(&ffmpeg, &file, &filter, k, w, h, wdir);
            (0..h).step_by(2).all(|row| y[row * stride..row * stride + w] == r[row * w..row * w + w])
        })
        .unwrap_or_else(|| {
            for k in 0..3 {
                let r = ffmpeg_frame(&ffmpeg, &file, &filter, k, w, h, wdir);
                for (name, f) in [("prev", &prev), ("cur", &cur), ("next", &next)] {
                    let (fy, fs) = f.plane(0);
                    let even = (0..h).step_by(2).filter(|&row| fy[row * fs..row * fs + w] == r[row * w..row * w + w]).count();
                    let odd = (1..h).step_by(2).filter(|&row| fy[row * fs..row * fs + w] == r[row * w..row * w + w]).count();
                    println!("  out#{k} vs {name}: even rows equal {even}/{}, odd rows equal {odd}/{}", h / 2, h / 2);
                }
            }
            panic!("no bwdif output holds frame n")
        });
    if pick != 1 {
        println!("  (FFmpeg output #{pick} holds frame {n})");
    }
    let ref_b = ffmpeg_frame(&ffmpeg, &file, &filter, pick, w, h, wdir);
    compare("bwdif", &b, &ref_b, true);

    let ad = DeintParams { method: DeintMethod::NnediAdaptive, tff: true, second_field: false, light: false };
    let (_, t_a) = off.deinterlace(&prev, &cur, &next, ad)?;
    println!("NNEDI3 + motion-adaptive: GPU {t_a:?} (no FFmpeg equivalent)");
    Ok(())
}
