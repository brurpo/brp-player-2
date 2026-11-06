//! Verifies frame-exact seeking: every frame reached by a random seek must be
//! byte-identical to the same frame reached by sequential decoding.
//!
//!   cargo run -p brp-media --release --example seekcheck -- <file> [random_seeks]

use brp_media::{Request, Source, VideoFrame};
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use std::sync::Arc;
use std::time::{Duration, Instant};

fn hash(f: &VideoFrame) -> u64 {
    let mut h = DefaultHasher::new();
    for p in 0..f.plane_count() {
        let (data, stride) = f.plane(p);
        let (w, rows) = f.plane_size(p);
        let row_bytes = w as usize * f.bytes_per_sample();
        for y in 0..rows as usize {
            data[y * stride..y * stride + row_bytes].hash(&mut h);
        }
    }
    h.finish()
}

fn wait(src: &Source, frame: i64, playing: bool) -> Option<Arc<VideoFrame>> {
    src.request(Request { frame, playing, dir: 1 });
    let t = Instant::now();
    loop {
        if let Some(f) = src.get(frame) {
            return Some(f);
        }
        if t.elapsed() > Duration::from_secs(20) {
            return None;
        }
        std::thread::sleep(Duration::from_millis(1));
    }
}

fn main() -> anyhow::Result<()> {
    env_logger::init();
    brp_media::init();
    let path = std::env::args().nth(1).expect("file");
    let seeks: usize = std::env::args().nth(2).and_then(|s| s.parse().ok()).unwrap_or(40);

    let t0 = Instant::now();
    let src = Source::open(&path)?;
    let opened = t0.elapsed();
    let first = wait(&src, 0, false).expect("first frame");
    let first_t = t0.elapsed();
    while !src.index_ready() {
        std::thread::sleep(Duration::from_millis(1));
    }
    let index_t = t0.elapsed();
    let i = &src.info;
    println!(
        "{}: {} {}x{} {} {:.3}fps frames={} layout={:?} hdr={} alpha={} tc={:?} audio={:?}",
        i.file_name, i.codec, i.width, i.height, i.pix_fmt, i.frame_rate, src.frame_count(),
        first.layout, i.color.is_hdr(), i.has_alpha, i.timecode, i.audio
    );
    println!("open {opened:?}, first frame {first_t:?}, index {index_t:?}, interlaced={:?}", src.interlaced());

    // Pass 1: sequential decode, reference hashes.
    let n = src.frame_count();
    let mut reference = Vec::with_capacity(n as usize);
    let t = Instant::now();
    for f in 0..n {
        let fr = wait(&src, f, true).unwrap_or_else(|| panic!("sequential frame {f} missing"));
        reference.push(hash(&fr));
    }
    println!("sequential: {n} frames in {:?} ({:.1} fps)", t.elapsed(), n as f64 / t.elapsed().as_secs_f64());

    // Pass 2: random seeks (fresh source so nothing is cached), plus a backward walk.
    drop(src);
    let src = Source::open(&path)?;
    while !src.index_ready() {
        std::thread::sleep(Duration::from_millis(1));
    }
    let mut rng: u64 = 0x9E3779B97F4A7C15;
    let mut targets: Vec<i64> = (0..seeks)
        .map(|_| {
            rng ^= rng << 13;
            rng ^= rng >> 7;
            rng ^= rng << 17;
            (rng % n as u64) as i64
        })
        .collect();
    targets.extend([n - 1, 0, n / 2, n - 2]);
    let mut bad = 0;
    let mut worst = Duration::ZERO;
    for &f in &targets {
        let t = Instant::now();
        let fr = wait(&src, f, false).unwrap_or_else(|| panic!("seek frame {f} missing"));
        worst = worst.max(t.elapsed());
        if hash(&fr) != reference[f as usize] {
            bad += 1;
            println!("  MISMATCH at frame {f} (got index {}, pts {})", fr.index(), fr.pts);
        }
    }
    println!("random seeks: {} targets, {bad} mismatches, worst latency {worst:?}", targets.len());

    // Pass 3: step backward from the end (the reverse-play case).
    let t = Instant::now();
    let mut worst = Duration::ZERO;
    for f in (n.saturating_sub(60).max(0)..n).rev() {
        let s = Instant::now();
        let fr = wait(&src, f, false).expect("back step");
        worst = worst.max(s.elapsed());
        if hash(&fr) != reference[f as usize] {
            bad += 1;
            println!("  MISMATCH stepping back at {f}");
        }
    }
    println!("step back 60: {:?} total, worst step {worst:?}", t.elapsed());
    if bad > 0 {
        anyhow::bail!("{bad} mismatches");
    }
    println!("OK");
    Ok(())
}
