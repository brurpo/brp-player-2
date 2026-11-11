//! Reads an EXR sequence's layers and times decoding (DESIGN.md §7c).
//!   cargo run -p brp-media --release --example exrcheck -- <file.exr|folder> [layer]
use std::time::Instant;

fn main() -> anyhow::Result<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let t = Instant::now();
    let seq = brp_media::exr_seq::scan(&args[0])?;
    println!("{} files from #{:?}, {}x{}, {}, float32 {}, fps {:?}, 2020 {}, scanned in {:?}", seq.files.len(), seq.first_number, seq.width, seq.height, seq.compression, seq.float32, seq.fps, seq.bt2020, t.elapsed());
    for (i, l) in seq.layers.iter().enumerate() {
        println!("  {i}: {}{}", l.label(), if i == seq.default_layer() { "  (default)" } else { "" });
    }
    if args.iter().any(|a| a == "--source") {
        return source_rate(&args[0]);
    }
    let layer = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(seq.default_layer());
    for i in 0..seq.files.len().min(3) {
        let t = Instant::now();
        let px = seq.decode(i, layer)?;
        let n = px.len() / 4;
        let mut mean = [0f64; 4];
        let mut max = [0f32; 4];
        for p in px.chunks_exact(4) {
            for c in 0..4 {
                let v = f32::from(p[c]);
                mean[c] += v as f64 / n as f64;
                max[c] = max[c].max(v);
            }
        }
        println!("file {i} layer {layer}: {:?}; mean {:.4} {:.4} {:.4} a {:.3}; max {:.2} {:.2} {:.2}", t.elapsed(), mean[0], mean[1], mean[2], mean[3], max[0], max[1], max[2]);
    }
    Ok(())
}

/// Frames per second through `Source` (its decode threads), playing from frame 0.
#[allow(dead_code)]
fn source_rate(path: &str) -> anyhow::Result<()> {
    let src = brp_media::Source::open(path)?;
    let n = src.frame_count();
    let t = Instant::now();
    src.request(brp_media::Request { frame: 0, playing: true, dir: 1, ahead: n, fill: false });
    while (0..n).any(|i| src.get(i).is_none()) {
        std::thread::sleep(std::time::Duration::from_millis(1));
    }
    let s = t.elapsed().as_secs_f64();
    println!("Source: {n} frames in {s:.2} s = {:.1} fps", n as f64 / s);
    Ok(())
}
