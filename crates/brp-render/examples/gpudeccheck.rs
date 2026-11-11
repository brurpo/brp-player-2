//! GPU decoders vs FFmpeg: every frame compared plane by plane, and decode speed.
//!
//!   cargo run -p brp-render --release --example gpudeccheck -- <file>... [--frames=N] [--bench=N] [--source] [--scrub=N]
//!
//! For each file: decodes the first N packets with the GPU decoder and the same
//! frames with FFmpeg (`Source` with the GPU off), and reports exact / max difference
//! / PSNR per plane. `--bench=N` times N GPU decodes of each packet (median per
//! frame). `--source` also checks the frames a GPU-decoding `Source` delivers;
//! `--scrub=N` checks N random seeks on it.
//! `BRP_GPU=<name>` picks the GPU.

use brp_media::gpudec::{self, GpuPicture};
use brp_media::{Request, Source, VideoFrame};
use brp_render::{GpuContext, GpuInstance};
use std::sync::Arc;
use std::time::{Duration, Instant};

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

#[derive(Default)]
struct Stat {
    max: u32,
    sq: f64,
    n: u64,
    diff: u64,
}

/// Compare plane `p` of `r` (FFmpeg) with GPU bytes laid out at (offset, stride).
fn compare(r: &VideoFrame, p: usize, gpu: &[u8], (off, stride): (u64, u64), st: &mut Stat) {
    let (w, h) = r.plane_size(p);
    let bps = r.bytes_per_sample();
    let (data, rs) = r.plane(p);
    let wide = !r.layout.is_rgb() && bps == 2;
    let row = w as usize * bps;
    for y in 0..h as usize {
        let a = &data[y * rs..y * rs + row];
        let b = &gpu[off as usize + y * stride as usize..][..row];
        if a == b {
            st.n += if wide { w as u64 } else { row as u64 };
            continue;
        }
        if wide {
            for (x, y2) in a.chunks_exact(2).zip(b.chunks_exact(2)) {
                let d = (u16::from_le_bytes([x[0], x[1]]) as i32 - u16::from_le_bytes([y2[0], y2[1]]) as i32).unsigned_abs();
                st.max = st.max.max(d);
                st.sq += (d * d) as f64;
                st.diff += (d != 0) as u64;
                st.n += 1;
            }
        } else {
            for (x, y2) in a.iter().zip(b) {
                let d = (*x as i32 - *y2 as i32).unsigned_abs();
                st.max = st.max.max(d);
                st.sq += (d * d) as f64;
                st.diff += (d != 0) as u64;
                st.n += 1;
            }
        }
    }
}

fn report(name: &str, stats: &[Stat], bits: u32) -> bool {
    let peak = ((1u64 << bits) - 1) as f64;
    let mut exact = true;
    let parts: Vec<String> = stats
        .iter()
        .enumerate()
        .map(|(i, s)| {
            if s.max == 0 {
                format!("p{i} exact")
            } else {
                exact = false;
                let mse = s.sq / s.n.max(1) as f64;
                format!("p{i} max {} ({} differ) {:.2} dB", s.max, s.diff, 10.0 * (peak * peak / mse).log10())
            }
        })
        .collect();
    println!("  {name}: {}", parts.join(", "));
    exact
}

fn main() -> anyhow::Result<()> {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("warn")).init();
    brp_media::init();
    let args: Vec<String> = std::env::args().skip(1).collect();
    let opt = |k: &str| args.iter().find_map(|a| a.strip_prefix(k)).map(|v| v.parse::<usize>().unwrap());
    let frames = opt("--frames=").unwrap_or(24);
    let bench = opt("--bench=").unwrap_or(0);
    let scrub = opt("--scrub=").unwrap_or(0);
    let check_source = args.iter().any(|a| a == "--source");
    let files: Vec<&String> = args.iter().filter(|a| !a.starts_with("--")).collect();

    let ctx = Arc::new(GpuContext::headless(GpuInstance::new()?, std::env::var("BRP_GPU").ok().as_deref())?);
    println!("GPU: {}", ctx.device_name);
    let factory = brp_render::gpudec::factory(&ctx);
    gpudec::set_factory(Some(factory.clone()));
    let mut all_exact = true;

    for file in files {
        println!("{file}");
        let (desc, packets) = gpudec::read_packets(file, frames)?;
        let Some(mut dec) = factory.open(&desc) else {
            println!("  no GPU decoder for this stream");
            continue;
        };
        let reference = Source::open_with(file, false)?;
        let mut stats: Vec<Stat> = Vec::new();
        let mut bits = 8;
        let mut refused = 0;
        let mut first: Option<GpuPicture> = None;
        let t_all = Instant::now();
        for (i, pkt) in packets.iter().enumerate() {
            let pic = match dec.decode(pkt) {
                Ok(p) => p,
                Err(e) => {
                    if refused == 0 {
                        println!("  frame {i}: refused: {e:#}");
                    }
                    refused += 1;
                    continue;
                }
            };
            let r = get(&reference, i as i64);
            assert_eq!((r.width(), r.height()), (pic.width, pic.height), "size");
            assert_eq!(r.layout, pic.layout, "layout");
            if let brp_media::FrameLayout::Yuv { bits: b, .. } = r.layout {
                bits = b as u32;
            }
            let bytes = pic.planes.download()?;
            let offs = pic.planes.plane_offsets();
            if i == 0 && std::env::var_os("BRP_DIFF_HIST").is_some() {
                hist(&r, &bytes, offs[0]);
            }
            stats.resize_with(r.plane_count(), Default::default);
            for (p, st) in stats.iter_mut().enumerate() {
                compare(&r, p, &bytes, offs[p], st);
            }
            if first.is_none() {
                first = Some(pic);
            }
        }
        let decoded = packets.len() - refused;
        println!("  {} frames on the GPU, {refused} refused ({:?} with checks)", decoded, t_all.elapsed());
        if decoded > 0 {
            all_exact &= report("vs FFmpeg", &stats, bits);
        }
        if let Some(p) = &first {
            println!("  {}x{} {:?} interlaced {} tff {}", p.width, p.height, p.layout, p.interlaced, p.tff);
        }

        if check_source && decoded > 0 {
            let src = Source::open(file)?;
            let mut st: Vec<Stat> = Vec::new();
            for i in 0..packets.len() as i64 {
                let a = get(&reference, i);
                let b = get(&src, i);
                st.resize_with(a.plane_count(), Default::default);
                for (p, s) in st.iter_mut().enumerate() {
                    let (data, stride) = b.plane(p);
                    compare(&a, p, data, (0, stride as u64), s);
                }
            }
            println!("  Source decoder: {}", src.decoder_name());
            all_exact &= report("Source (GPU) vs FFmpeg", &st, bits);
            // Back to frames that have left GPU memory since: decoded again from the
            // packets the source keeps (no seek).
            let mut st: Vec<Stat> = Vec::new();
            let mut seed = 0x9E3779B97F4A7C15u64;
            let back = (packets.len() as u64 / 2).max(1);
            let t = Instant::now();
            for _ in 0..100 {
                seed ^= seed << 13;
                seed ^= seed >> 7;
                seed ^= seed << 17;
                let i = (seed % back) as i64;
                let a = get(&reference, i);
                let b = get(&src, i);
                st.resize_with(a.plane_count(), Default::default);
                for (p, s) in st.iter_mut().enumerate() {
                    let (data, stride) = b.plane(p);
                    compare(&a, p, data, (0, stride as u64), s);
                }
            }
            all_exact &= report(&format!("100 frames from kept packets ({:?})", t.elapsed()), &st, bits);
        }

        if scrub > 0 && decoded > 0 {
            // Random jumps (scrubbing): every frame must be the exact one.
            let src = Source::open(file)?;
            let n = packets.len() as u64;
            let mut seed = 0x2545F4914F6CDD1Du64;
            let mut st: Vec<Stat> = Vec::new();
            let t = Instant::now();
            for k in 0..scrub {
                seed ^= seed << 13;
                seed ^= seed >> 7;
                seed ^= seed << 17;
                let i = (seed % n) as i64;
                let a = get(&reference, i);
                let b = get(&src, i);
                st.resize_with(a.plane_count(), Default::default);
                let before = st.first().map_or(0, |s| s.diff);
                for (p, s) in st.iter_mut().enumerate() {
                    let (data, stride) = b.plane(p);
                    compare(&a, p, data, (0, stride as u64), s);
                }
                if st[0].diff != before {
                    println!("  seek {k}: frame {i} differs (index ready: {})", src.index_ready());
                }
            }
            all_exact &= report(&format!("{scrub} random seeks ({:?})", t.elapsed()), &st, bits);
        }

        if bench > 0 && decoded > 0 {
            let mut per = Vec::new();
            for pkt in &packets {
                let t = Instant::now();
                for _ in 0..bench {
                    if dec.decode(pkt).is_err() {
                        break;
                    }
                }
                per.push(t.elapsed().as_secs_f64() * 1e3 / bench as f64);
            }
            per.sort_by(f64::total_cmp);
            let mb = packets.iter().map(|p| p.len()).sum::<usize>() as f64 / packets.len() as f64 / 1e6;
            println!("  GPU decode: median {:.2} ms/frame, min {:.2} ({:.2} MB/packet)", per[per.len() / 2], per[0], mb);
        }
    }
    println!("{}", if all_exact { "ALL EXACT" } else { "DIFFERENCES" });
    Ok(())
}

/// Where in the 8x8 blocks plane 0 differs (count, and sum of FFmpeg - GPU).
fn hist(r: &VideoFrame, gpu: &[u8], (off, stride): (u64, u64)) {
    let (w, h) = r.plane_size(0);
    let (data, rs) = r.plane(0);
    let mut cnt = [[0u32; 8]; 8];
    let mut sum = [[0i64; 8]; 8];
    let mut by_val = std::collections::BTreeMap::<i32, (u32, u32)>::new();
    for y in 0..h as usize {
        for x in 0..w as usize {
            let a = u16::from_le_bytes([data[y * rs + 2 * x], data[y * rs + 2 * x + 1]]) as i32;
            let o = off as usize + y * stride as usize + 2 * x;
            let b = u16::from_le_bytes([gpu[o], gpu[o + 1]]) as i32;
            let e = by_val.entry(a / 64).or_default();
            e.1 += 1;
            if a != b {
                cnt[y % 8][x % 8] += 1;
                sum[y % 8][x % 8] += (a - b) as i64;
                e.0 += 1;
            }
        }
    }
    for y in 0..8 {
        println!("    {:?}   {:?}", cnt[y], sum[y]);
    }
    println!("    differ by value/64: {:?}", by_val);
}
