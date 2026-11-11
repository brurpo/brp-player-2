//! Test bench for the GPU encoders: raw planar frames in, a muxed file out, with
//! the GPU time per frame.
//!
//! cargo run --release -p brp-ai --example gpuenc -- --raw=in.yuv --size=1920x1080
//!     --pix=yuv422p10le --codec=prores:hq --out=out.mov [--frames=N] [--rate=25]
//!     [--tff|--bff] [--alpha] [--repeat=K]
//!
//! Codecs: prores:<proxy|lt|standard|hq|4444|4444xq>, dnx:<cid>, dxv[:alpha].

use anyhow::{bail, Context as _, Result};
use brp_ai::gpuenc::{self, Codec};
use brp_media::encode::{self, Precoded, RateControl, VideoConfig, Writer};
use brp_render::compute::{Compute, Recorder};
use brp_render::{GpuContext, GpuInstance};
use std::io::Read as _;
use std::sync::Arc;
use std::time::Instant;

fn arg<'a>(args: &'a [String], k: &str) -> Option<&'a str> {
    args.iter().find_map(|a| a.strip_prefix(&format!("--{k}="))).or_else(|| args.iter().any(|a| a == &format!("--{k}")).then_some(""))
}

fn main() -> Result<()> {
    env_logger::init();
    let args: Vec<String> = std::env::args().collect();
    let raw = arg(&args, "raw").context("--raw")?;
    let (w, h) = arg(&args, "size").and_then(|s| s.split_once('x')).map(|(a, b)| (a.parse::<u32>().unwrap(), b.parse::<u32>().unwrap())).context("--size=WxH")?;
    let pix = arg(&args, "pix").unwrap_or("yuv422p10le");
    let out_path = arg(&args, "out").context("--out")?;
    let rate: i32 = arg(&args, "rate").and_then(|r| r.parse().ok()).unwrap_or(25);
    let repeat: usize = arg(&args, "repeat").and_then(|r| r.parse().ok()).unwrap_or(1);
    let max_frames: usize = arg(&args, "frames").and_then(|r| r.parse().ok()).unwrap_or(usize::MAX);
    let interlaced = if arg(&args, "tff").is_some() { Some(true) } else if arg(&args, "bff").is_some() { Some(false) } else { None };
    let alpha = arg(&args, "alpha").is_some();
    let codec_s = arg(&args, "codec").context("--codec")?;
    let layout = encode::pack_layout(pix, w, h).context("pixel format")?;

    let (codec, pc) = match codec_s.split(':').collect::<Vec<_>>().as_slice() {
        ["prores", prof] => {
            let profile = gpuenc::prores::Settings::profile_of(prof).context("profile")?;
            (
                Codec::ProRes(gpuenc::prores::Settings { profile, alpha, interlaced, color: (1, 1, 1), vendor: *b"apl0" }),
                Precoded { codec: "prores".into(), tag: Some(gpuenc::prores::tag(profile)), profile: profile as i32, bits_per_coded_sample: if alpha { 32 } else { 0 }, bits_per_raw_sample: 10, bit_rate: 0, intra_only: true },
            )
        }
        ["dnx", cid] => {
            let cid: u32 = cid.parse()?;
            let c = gpuenc::dnxhd::cid(cid).context("cid")?;
            let profile = match cid {
                1270 => 5,
                1271 => 4,
                1272 => 3,
                1273 => 2,
                1274 => 1,
                _ => 0,
            };
            let fs = gpuenc::dnxhd::frame_size(c, w, h) as i64;
            (
                Codec::Dnx(gpuenc::dnxhd::Settings { profile: cid.to_string(), interlaced }),
                Precoded { codec: "dnxhd".into(), tag: Some(if profile > 0 { *b"AVdh" } else { *b"AVdn" }), profile, bits_per_coded_sample: 0, bits_per_raw_sample: if cid >= 1272 || c.bit_depth == 8 { 8 } else { 10 }, bit_rate: fs * 8 * rate as i64, intra_only: true },
            )
        }
        ["dxv"] | ["dxv", "alpha"] => (
            Codec::Dxv(gpuenc::dxv::Settings { alpha }),
            Precoded { codec: "dxv".into(), tag: Some(*b"DXD3"), profile: -99, bits_per_coded_sample: 0, bits_per_raw_sample: 0, bit_rate: 0, intra_only: true },
        ),
        other => bail!("codec {other:?}"),
    };

    let ctx = Arc::new(GpuContext::headless(GpuInstance::new()?, std::env::var("BRP_GPU").ok().as_deref())?);
    println!("GPU: {}", ctx.device_name);
    let compute = Compute::new(ctx);
    let t0 = Instant::now();
    let mut enc = gpuenc::create(&compute, &codec, &layout)?;
    println!("encoder ready in {:.0} ms", t0.elapsed().as_secs_f64() * 1e3);

    let v = VideoConfig {
        encoder: String::new(),
        pix_fmt: pix.into(),
        width: w,
        height: h,
        rate: (rate, 1),
        sar: (1, 1),
        rate_control: RateControl::Default,
        options: Vec::new(),
        gop: None,
        color: Some(brp_media::ColorInfo { matrix: brp_media::Matrix::Bt709, full_range: false, transfer: brp_media::Transfer::Sdr, primaries: brp_media::Primaries::Bt709 }),
        rotation: 0,
        filter: None,
        field_order: interlaced,
        precoded: Some(pc),
    };
    let mut writer = Writer::create(out_path, None, &v, None, &[])?;

    // Raw frames as ffmpeg writes them (planes back to back, no padding).
    let info = encode::pix_fmt_info(pix).context("pix")?;
    let mut f = std::fs::File::open(raw)?;
    let mut frame = 0usize;
    let mut gpu_ms = Vec::new();
    let mut bytes_total = 0u64;
    loop {
        if frame >= max_frames {
            break;
        }
        let mut data = vec![0u8; 0];
        let mut ok = true;
        let mut planes_raw = Vec::new();
        for p in &layout.planes {
            let row = (p.w * p.step as u32 * p.bytes as u32) as usize;
            let mut buf = vec![0u8; row * p.h as usize];
            if f.read_exact(&mut buf).is_err() {
                ok = false;
                break;
            }
            planes_raw.push((row, buf));
        }
        let _ = info.bits;
        if !ok {
            break;
        }
        data.resize(layout.total as usize, 0);
        for (p, (row, buf)) in layout.planes.iter().zip(&planes_raw) {
            for y in 0..p.h as usize {
                data[p.offset as usize + y * p.stride as usize..][..*row].copy_from_slice(&buf[y * row..][..*row]);
            }
        }
        let planes = compute.upload(&data, "planes")?;
        let mut pkt = Vec::new();
        for _ in 0..repeat {
            let mut rec = Recorder::new(&compute)?;
            enc.record(&mut rec, &planes, &layout)?;
            let t = rec.submit_wait()?;
            gpu_ms.push(t.as_secs_f64() * 1e3);
            pkt = enc.packet()?;
        }
        bytes_total += pkt.len() as u64;
        writer.write(&pkt)?;
        frame += 1;
    }
    writer.finish()?;
    gpu_ms.sort_by(|a, b| a.total_cmp(b));
    let med = gpu_ms.get(gpu_ms.len() / 2).copied().unwrap_or(0.0);
    println!("{frame} frames, {:.1} KB/frame, GPU {:.2} ms/frame (median of {}), min {:.2}", bytes_total as f64 / frame.max(1) as f64 / 1024.0, med, gpu_ms.len(), gpu_ms.first().copied().unwrap_or(0.0));
    Ok(())
}
