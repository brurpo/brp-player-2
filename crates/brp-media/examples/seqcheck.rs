//! Image sequences opened from one of their files (DESIGN.md §7c): compares every
//! frame with FFmpeg's own sequence reader (a %0Nd pattern) and times decoding.
//!
//!   cargo run -p brp-media --release --example seqcheck -- <one file of the sequence> <pattern with %0Nd>
use brp_media::{Request, Source};
use std::sync::Arc;
use std::time::{Duration, Instant};

fn wait(src: &Source, i: i64) -> Arc<brp_media::VideoFrame> {
    let t = Instant::now();
    loop {
        if let Some(f) = src.get(i) {
            return f;
        }
        assert!(t.elapsed() < Duration::from_secs(30), "frame {i} not decoded");
        std::thread::sleep(Duration::from_millis(1));
    }
}

fn main() -> anyhow::Result<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let ours = Source::open(&args[0])?;
    let n = ours.frame_count();
    println!("{}: {} frames, {}x{} {} {} {:?}, {} fps, {} ({})", ours.info.file_name, n, ours.info.width, ours.info.height, ours.info.codec, ours.info.pix_fmt, ours.info.color.transfer, ours.info.frame_rate, ours.info.container, ours.decoder_name());
    // Decoding rate: all frames, playing from 0.
    let t = Instant::now();
    ours.request(Request { frame: 0, playing: true, dir: 1, ahead: n, fill: false });
    for i in 0..n {
        wait(&ours, i);
    }
    let s = t.elapsed().as_secs_f64();
    println!("decoded {n} frames in {s:.2} s = {:.1} fps", n as f64 / s);
    // Same pictures as FFmpeg's sequence reader, frame by frame.
    let theirs = Source::open(&args[1])?;
    let t = Instant::now();
    while !theirs.index_ready() && t.elapsed() < Duration::from_secs(60) {
        std::thread::sleep(Duration::from_millis(5));
    }
    anyhow::ensure!(theirs.frame_count() == n, "FFmpeg reads {} frames, we read {n}", theirs.frame_count());
    let mut same = 0;
    for i in 0..n {
        theirs.request(Request { frame: i, playing: false, dir: 1, ahead: 0, fill: false });
        let (a, b) = (wait(&ours, i), wait(&theirs, i));
        let equal = a.layout == b.layout
            && (0..a.plane_count()).all(|p| {
                let ((da, sa), (db, sb)) = (a.plane(p), b.plane(p));
                let (w, h) = a.plane_size(p);
                let row = w as usize * a.bytes_per_sample();
                (0..h as usize).all(|y| da[y * sa..y * sa + row] == db[y * sb..y * sb + row])
            });
        if equal {
            same += 1;
        } else {
            println!("frame {i}: DIFFERENT");
        }
    }
    println!("{same}/{n} frames identical to FFmpeg's sequence reader");
    Ok(())
}
