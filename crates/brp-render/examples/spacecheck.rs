//! Checks the effects' working colour spaces (DESIGN.md §7a) on real frames.
//!
//!   cargo run -p brp-render --release --example spacecheck [-- --exr <sequence> --4k]
//!
//! For an SDR, a PQ, an HLG (the PQ frame re-tagged) and a linear EXR frame: runs
//! effects that change nothing in each working space, alone and stacked, and compares
//! with the frame through no conversion, after the display transform (the maths of
//! `video.frag`): the largest and mean difference in 8- and 10-bit levels. Then a gain
//! in linear light on the HDR frames: highlights above SDR white must scale, not clip.
//! `--4k` also times the conversions on a 4K frame.

use brp_media::{ColorInfo, Primaries, Request, Source, Transfer as Tf, VideoFrame};
use brp_render::offline_fx::{OfflineEffects, Rgba};
use brp_render::usershader;
use brp_render::{EffectSpec, GpuContext, GpuInstance, Transfer};
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

fn open(path: &str) -> anyhow::Result<Source> {
    let src = Source::open(path)?;
    let t = Instant::now();
    while !src.index_ready() && t.elapsed() < Duration::from_secs(60) {
        std::thread::sleep(Duration::from_millis(5));
    }
    Ok(src)
}

fn effect(id: u64, name: &str, params: &[(&str, f64)]) -> anyhow::Result<EffectSpec> {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/spaces").join(format!("{name}.glsl"));
    let file = usershader::parse::parse(name, &std::fs::read_to_string(&path)?)?;
    let mut values: Vec<f64> = file.params.iter().map(|p| p.default).collect();
    for (k, v) in params {
        let i = file.params.iter().position(|p| p.name == *k).unwrap();
        values[i] = *v;
    }
    Ok(EffectSpec { id, file: Arc::new(file), params: values, enabled: true })
}

fn transfer(c: &ColorInfo) -> Transfer {
    match c.transfer {
        Tf::Pq => Transfer::Pq,
        Tf::Hlg => Transfer::Hlg,
        Tf::Linear => Transfer::Linear,
        Tf::Sdr => Transfer::Sdr,
    }
}

// ---- the display transform of video.frag, on the CPU ----

fn pq_to_nits(e: f32) -> f32 {
    let (m1, m2, c1, c2, c3) = (0.1593017578125f32, 78.84375f32, 0.8359375f32, 18.8515625f32, 18.6875f32);
    let p = e.clamp(0.0, 1.0).powf(1.0 / m2);
    ((p - c1).max(0.0) / (c2 - c3 * p)).powf(1.0 / m1) * 10000.0
}

fn hlg_to_nits(e: [f32; 3]) -> [f32; 3] {
    let (a, b, c) = (0.17883277f32, 0.28466892f32, 0.55991073f32);
    let s = e.map(|e| {
        let e = e.clamp(0.0, 1.0);
        if e < 0.5 { e * e / 3.0 } else { (((e - c) / a).exp() + b) / 12.0 }
    });
    let ys = 0.2627 * s[0] + 0.6780 * s[1] + 0.0593 * s[2];
    let k = ys.max(1e-6).powf(0.2) * 1000.0;
    s.map(|v| v * k)
}

/// Light relative to SDR white, Rec.709 (HDR sources), as video.frag computes it
/// before the tone map.
fn hdr_light(p: &[f32; 4], c: &ColorInfo) -> [f32; 3] {
    let nits = if c.transfer == Tf::Pq { [pq_to_nits(p[0]), pq_to_nits(p[1]), pq_to_nits(p[2])] } else { hlg_to_nits([p[0], p[1], p[2]]) };
    let l = nits.map(|n| n / 203.0);
    if c.primaries == Primaries::Bt2020 {
        [
            1.6605 * l[0] - 0.5876 * l[1] - 0.0728 * l[2],
            -0.1246 * l[0] + 1.1329 * l[1] - 0.0083 * l[2],
            -0.0182 * l[0] - 0.1006 * l[1] + 1.1187 * l[2],
        ]
    } else {
        l
    }
}

fn display(p: &[f32; 4], c: &ColorInfo) -> [f32; 3] {
    match c.transfer {
        Tf::Sdr => [p[0], p[1], p[2]].map(|v| v.clamp(0.0, 1.0)),
        Tf::Linear => [p[0], p[1], p[2]].map(|v| v.clamp(0.0, 1.0).powf(1.0 / 2.2)),
        Tf::Pq | Tf::Hlg => {
            let l = hdr_light(p, c).map(|v| v.max(0.0));
            let peak = 1000.0f32 / 203.0;
            let m = l[0].max(l[1]).max(l[2]);
            if m <= 0.0 {
                return [0.0; 3];
            }
            let t = m * (1.0 + m / (peak * peak)) / (1.0 + m);
            l.map(|v| (v * t / m).clamp(0.0, 1.0).powf(1.0 / 2.4))
        }
    }
}

/// Difference after the display transform, in 8-bit levels: largest, mean, share of
/// channel values off by more than one level, the 99.99th percentile, and the largest
/// difference in displayed light relative to the pixel's brightest channel.
fn compare(a: &Rgba, b: &Rgba, c: &ColorInfo) -> (f64, f64, f64, f64, f64) {
    let mut d: Vec<f32> = Vec::with_capacity(a.data.len() * 3);
    let mut rel = 0f64;
    for (p, q) in a.data.iter().zip(&b.data) {
        let (x, y) = (display(p, c), display(q, c));
        let (lx, ly) = (x.map(|v| v.powf(2.4)), y.map(|v| v.powf(2.4)));
        let m = lx[0].max(lx[1]).max(lx[2]).max(1e-3);
        for k in 0..3 {
            d.push((x[k] - y[k]).abs() * 255.0);
            rel = rel.max(((lx[k] - ly[k]).abs() / m) as f64);
        }
    }
    let n = d.len() as f64;
    let mean = d.iter().map(|&v| v as f64).sum::<f64>() / n;
    let over = d.iter().filter(|&&v| v > 1.0).count() as f64 / n;
    d.sort_by(f32::total_cmp);
    let p9999 = d[((n * 0.9999) as usize).min(d.len() - 1)] as f64;
    (*d.last().unwrap() as f64, mean, over, p9999, rel)
}

/// Light relative to SDR white in the source primaries (HDR sources).
fn src_light(p: &[f32; 4], c: &ColorInfo) -> [f32; 3] {
    let nits = if c.transfer == Tf::Pq { [pq_to_nits(p[0]), pq_to_nits(p[1]), pq_to_nits(p[2])] } else { hlg_to_nits([p[0], p[1], p[2]]) };
    nits.map(|n| n / 203.0)
}

fn main() -> anyhow::Result<()> {
    env_logger::init();
    let args: Vec<String> = std::env::args().skip(1).collect();
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../footage");
    let root = root.to_string_lossy();
    let ctx = Arc::new(GpuContext::headless(GpuInstance::new()?, std::env::var("BRP_GPU").ok().as_deref())?);
    println!("GPU: {}", ctx.device_name);

    let sdr = open(&format!("{root}/test.mp4"))?;
    let pq = open(&format!("{root}/20250725_164717.mp4"))?;
    // The multi-layer EXRs in footage/exr do not open yet (FFmpeg reads single-layer
    // files only): pass a sequence made by our export, e.g. the PQ clip as linear EXR
    // (`brp-player footage/20250725_164717.mp4 --export=x/pq.exr --container=exr --range=0-11`).
    let flag = |k: &str| args.iter().position(|a| a == k).and_then(|i| args.get(i + 1)).cloned();
    let exr = flag("--exr").map(|p| open(&p)).transpose()?;
    let hlg_color = ColorInfo { transfer: Tf::Hlg, ..pq.info.color };
    let cases: Vec<(&str, Arc<VideoFrame>, ColorInfo)> = vec![
        ("SDR test.mp4", get(&sdr, 10), sdr.info.color),
        ("PQ 20250725_164717.mp4", get(&pq, 10), pq.info.color),
        ("HLG (PQ frame re-tagged)", get(&pq, 10), hlg_color),
    ];
    let mut cases = cases;
    if let Some(e) = &exr {
        cases.push(("EXR (linear)", get(e, 0), e.info.color));
    }
    let stacks: Vec<(&str, Vec<&str>)> = vec![
        ("linear", vec!["pass_linear"]),
        ("display", vec!["pass_display"]),
        ("pq", vec!["pass_pq"]),
        ("mpv LINEAR", vec!["pass_mpv_linear"]),
        ("mpv SIGMOID", vec!["pass_sigmoid"]),
        ("source+linear+display+pq+linear", vec!["pass_source", "pass_linear", "pass_display", "pass_pq", "pass_mpv_linear", "pass_linear"]),
    ];
    let (mut worst, mut worst_p) = (0f64, 0f64);
    for (label, frame, color) in &cases {
        // One runner per source: its frame cache knows frames by number only.
        let mut fx = OfflineEffects::new(ctx.clone())?;
        let t = transfer(color);
        let target = (frame.width(), frame.height());
        let errs = fx.set_effects(&[effect(1, "pass_source", &[])?], t);
        anyhow::ensure!(errs.is_empty(), "{errs:?}");
        let (base, _) = fx.run(frame, *color, 0, target)?;
        println!("\n{label}: {}x{} {:?} {:?}", frame.width(), frame.height(), color.transfer, color.primaries);
        for (name, list) in &stacks {
            let specs: Vec<EffectSpec> = list.iter().enumerate().map(|(i, n)| effect(10 + i as u64, n, &[])).collect::<anyhow::Result<_>>()?;
            let errs = fx.set_effects(&specs, t);
            anyhow::ensure!(errs.is_empty(), "{errs:?}");
            let (out, _) = fx.run(frame, *color, 0, target)?;
            let (max, mean, over, p, rel) = compare(&out, &base, color);
            worst = worst.max(mean);
            worst_p = worst_p.max(p);
            println!("  {name:<34} mean {mean:.3}, 99.99% {p:.2}, max {max:.2} (8-bit levels); {:.3}% of values > 1 level; light {rel:.1e} of the pixel", over * 100.0);
        }
        // HDR through a linear effect: a quarter of the light.
        if color.transfer == Tf::Pq || color.transfer == Tf::Hlg {
            let errs = fx.set_effects(&[effect(20, "gain_linear", &[("GAIN", 0.25)])?], t);
            anyhow::ensure!(errs.is_empty(), "{errs:?}");
            let (out, _) = fx.run(frame, *color, 0, target)?;
            let (mut peak_in, mut peak_out, mut err, mut n) = (0f32, 0f32, 0f32, 0usize);
            for (p, q) in base.data.iter().zip(&out.data) {
                let (a, b) = (src_light(p, color), src_light(q, color));
                let (ma, mb) = (a[0].max(a[1]).max(a[2]), b[0].max(b[1]).max(b[2]));
                peak_in = peak_in.max(ma);
                peak_out = peak_out.max(mb);
                if ma > 1.0 {
                    // Each channel's error relative to the pixel's brightness.
                    for k in 0..3 {
                        err = err.max((b[k] - a[k] * 0.25).abs() / (ma * 0.25));
                    }
                    n += 1;
                }
            }
            if std::env::var_os("SPACECHECK_DEBUG").is_some() {
                let mut v: Vec<f32> = base.data.iter().map(|p| src_light(p, color).into_iter().fold(0f32, f32::max)).collect();
                v.sort_by(f32::total_cmp);
                println!("  debug: brightest channel percentiles 1% {:.3} 50% {:.3} 99% {:.3} (SDR white = 1); pixel 0 {:?}", v[v.len() / 100], v[v.len() / 2], v[v.len() * 99 / 100], base.data[0]);
            }
            println!(
                "  linear gain 0.25: brightest channel {:.0} -> {:.0} nits; {n} pixel(s) above SDR white, largest relative error {:.2e}",
                peak_in * 203.0,
                peak_out * 203.0,
                err
            );
        }
    }
    // Pass: the mean stays well under a level. (The largest differences are float16
    // rounding seen through the gamma's steep start: channels near zero of saturated
    // colours, a few levels in 8-bit, ~1e-3 of the pixel's light.)
    let ok = worst < 0.5;
    println!("\nworst pass-through: mean {worst:.3}, 99.99th percentile {worst_p:.2} (8-bit levels): {}", if ok { "OK" } else { "TOO LARGE" });
    if !ok {
        std::process::exit(1);
    }

    if args.iter().any(|a| a == "--4k") {
        // Cost of one conversion: 16 pass-through effects alternating source and
        // linear (16 conversions) against 16 in source (none), same passes otherwise.
        for (label, path) in [("1080p", format!("{root}/test.mp4")), ("4K", format!("{root}/gpudec/prores_hq_4k.mov"))] {
            let mut fx = OfflineEffects::new(ctx.clone())?;
            let src = open(&path)?;
            let f = get(&src, 0);
            let color = src.info.color;
            let target = (f.width(), f.height());
            let mut times = Vec::new();
            for names in [["pass_source", "pass_source"], ["pass_linear", "pass_source"]] {
                let specs: Vec<EffectSpec> = (0..16).map(|i| effect(30 + i as u64, names[i % 2], &[])).collect::<anyhow::Result<_>>()?;
                fx.set_effects(&specs, transfer(&color));
                let mut best = Duration::MAX;
                for _ in 0..7 {
                    best = best.min(fx.run(&f, color, 0, target)?.1);
                }
                times.push(best.as_secs_f64() * 1e3);
            }
            println!("{label} {}x{}: 16 passes {:.2} ms; with 16 conversions {:.2} ms: {:.3} ms per conversion", f.width(), f.height(), times[0], times[1], (times[1] - times[0]) / 16.0);
        }
    }
    Ok(())
}
