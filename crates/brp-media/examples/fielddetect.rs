//! Prints content-based interlace detection for a file.
//!   cargo run -p brp-media --release --example fielddetect -- <file>...
use brp_media::{Request, Source};
use std::time::{Duration, Instant};

fn main() -> anyhow::Result<()> {
    brp_media::init();
    for path in std::env::args().skip(1) {
        let src = Source::open(&path)?;
        while !src.index_ready() {
            std::thread::sleep(Duration::from_millis(1));
        }
        let n = src.frame_count().min(150);
        for f in 0..n {
            src.request(Request { frame: f, playing: true, dir: 1 });
            let t = Instant::now();
            while src.get(f).is_none() && t.elapsed() < Duration::from_secs(10) {
                std::thread::sleep(Duration::from_millis(1));
            }
        }
        std::thread::sleep(Duration::from_millis(50));
        let (v, d) = src.field_detection();
        println!(
            "{:<45} flags: {:<12} content: tff {:>2} bff {:>2} prog {:>2} undet {:>2} -> {}",
            src.info.file_name,
            match src.interlaced() { Some(true) => "interlaced", Some(false) => "progressive", None => "?" },
            v.tff, v.bff, v.progressive, v.undetermined,
            match d { Some((true, tff)) => format!("INTERLACED {}", if tff { "TFF" } else { "BFF" }), Some((false, _)) => "progressive".into(), None => "not enough evidence".into() }
        );
    }
    Ok(())
}
