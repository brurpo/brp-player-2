//! Exports a synthetic clip through the external x264/x265 encoder (GPL ffmpeg.exe,
//! see `external.rs`), with audio from a source file, then reads it back.
//!
//!   cargo run -p brp-media --release --example extenc -- <out dir> [audio source]
//!
//! Needs `BRP_FFMPEG_GPL` (or `<exe dir>/ffmpeg-gpl/ffmpeg.exe`).

use brp_media::encode::{self, AudioConfig, AudioSource, RateControl, VideoConfig, Writer};
use brp_media::Source;
use std::time::Instant;

fn main() -> anyhow::Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let out = &args[1];
    let audio_src = args.get(2);
    println!("external program: {:?}", brp_media::external::program());
    for (enc, pix, file, interlaced) in [
        ("libx264", "yuv420p", "x264.mp4", None),
        ("libx265", "yuv420p10le", "x265.mov", None),
        ("libx264", "yuv420p", "x264_tff.mkv", Some(true)),
    ] {
        let info = encode::encoder(enc).ok_or_else(|| anyhow::anyhow!("{enc} not available"))?;
        println!("{enc}: {} pix {}", info.long_name, info.pix_fmts.len());
        let (w, h) = (1920u32, 1080u32);
        let v = VideoConfig {
            encoder: enc.into(),
            pix_fmt: pix.into(),
            width: w,
            height: h,
            rate: (25, 1),
            sar: (1, 1),
            rate_control: RateControl::Quality(20.0),
            options: vec![("preset".into(), "fast".into())],
            gop: Some(25),
            color: Some(brp_media::info::ColorInfo { matrix: brp_media::info::Matrix::Bt709, full_range: false, transfer: brp_media::info::Transfer::Sdr, primaries: brp_media::info::Primaries::Bt709 }),
            rotation: 0,
            filter: None,
            field_order: interlaced,
            precoded: None,
        };
        let path = format!("{out}/{file}");
        let a = audio_src.map(|p| (AudioConfig { encoder: "aac".into(), kbps: 192, sample_rate: 0 }, AudioSource { path: p.clone(), start: 0.0, end: 4.0, tempo: 1.0, keep_pitch: true, length: 4.0 }));
        let mut wr = Writer::create(&path, None, &v, a.as_ref().map(|(c, s)| (c, s)), &[("comment".into(), "extenc test".into())])?;
        let lay = wr.layout().clone();
        let mut buf = vec![0u8; lay.total as usize];
        let t = Instant::now();
        let n = 100;
        for i in 0..n {
            // A moving gradient in every plane.
            for (pi, p) in lay.planes.iter().enumerate() {
                for y in 0..p.h as usize {
                    for x in 0..(p.w as usize * p.step as usize) {
                        let v = ((x + y + i * 7 + pi * 40) % 256) as u16;
                        let o = p.offset as usize + y * p.stride as usize + x * p.bytes as usize;
                        if p.bytes == 2 {
                            buf[o..o + 2].copy_from_slice(&(v << 2).to_le_bytes());
                        } else {
                            buf[o] = v as u8;
                        }
                    }
                }
            }
            wr.write(&buf)?;
        }
        wr.finish()?;
        let dt = t.elapsed().as_secs_f64();
        let src = Source::open(&path)?;
        let i = &src.info;
        println!(
            "  {path}: {} {} {}x{} {:.2} fps, {} frames, audio {:?}, field {:?}, {:.1} MB, {:.0} fps export",
            i.container, i.codec, i.width, i.height, i.frame_rate, src.frame_count(), i.audio, i.field_order,
            std::fs::metadata(&path)?.len() as f64 / 1e6, n as f64 / dt
        );
    }
    Ok(())
}
