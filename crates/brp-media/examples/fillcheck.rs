//! Paused cache filling: pause at a frame with `fill` on and watch the cache grow
//! ahead of the playhead, stop at the budget, and keep the frames nearest the
//! playhead after a jump. Then playing: the cache runs ahead of the playhead, and
//! stays inside a looped range.
//!
//!   cargo run -p brp-media --release --example fillcheck -- <file> [budget MB]

use brp_media::{Request, Source};
use std::time::{Duration, Instant};

fn report(src: &Source, at: i64, t: Instant) {
    let mut c = src.cached_frames();
    c.sort_unstable();
    let (lo, hi) = (c.first().copied().unwrap_or(0), c.last().copied().unwrap_or(0));
    println!("  {:>5.1} s: {} frames cached, {lo}..{hi} (playhead {at})", t.elapsed().as_secs_f64(), c.len());
}

fn main() -> anyhow::Result<()> {
    let a: Vec<String> = std::env::args().skip(1).collect();
    let mb: usize = a.get(1).and_then(|v| v.parse().ok()).unwrap_or(512);
    brp_media::mem::set_cache_budget(mb << 20);
    let src = Source::open(&a[0])?;
    for at in [100i64, 600] {
        println!("paused at {at}, budget {mb} MB:");
        let t = Instant::now();
        for _ in 0..16 {
            src.request(Request { frame: at, playing: false, dir: 1, ahead: 0, fill: true, range: (0, i64::MAX), looping: false });
            std::thread::sleep(Duration::from_millis(500));
            report(&src, at, t);
        }
    }
    // Playing: the fill reads ahead of the playhead up to the budget, then moves along
    // with it; looping in a range, it stays inside the range.
    let fps = src.info.frame_rate.max(1.0);
    for (name, range, looping) in [("playing from 0", (0, i64::MAX), false), ("looping in 200..=260", (200, 260), true)] {
        println!("{name}, budget {mb} MB:");
        let t = Instant::now();
        let mut said = 0;
        while t.elapsed().as_secs_f64() < 3.0 {
            let played = (t.elapsed().as_secs_f64() * fps) as i64;
            let at = if looping { range.0 + played % (range.1 - range.0 + 1) } else { played };
            src.request(Request { frame: at, playing: true, dir: 1, ahead: 0, fill: true, range, looping });
            std::thread::sleep(Duration::from_millis(10));
            if (t.elapsed().as_secs_f64() * 2.0) as i32 > said {
                said += 1;
                report(&src, at, t);
            }
        }
    }
    Ok(())
}
