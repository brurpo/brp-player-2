//! The learned denoiser in our runner against PyTorch: frames written by
//! `ref_dump.py` (tools/denoise_train: real frames and the PyTorch model's outputs,
//! frame by frame with state) are run the same way here and compared.
//!   learnedcheck REF_DIR [RUNS]
//! REF_DIR holds shape.txt ("frames height width"), in.f32 and out.f32 (planar f32,
//! linear light = code value ^ 2.4). Prints, per frame, the PSNR against PyTorch's
//! output on display code values, and how far the input was from it (what the
//! model changed). RUNS > 0 (and BRP_SIZE=WxH, default 1920x1080): timing on a
//! synthetic picture of that size, state carried.
//! BRP_DN_F32=1: the picture networks in f32. BRP_GPU picks the GPU by name.

use brp_ai::gpu::Ops;
use brp_ai::learned::Learned;
use brp_ai::vk::Tensor;
use brp_ai::Image;
use brp_render::compute::{Compute, Recorder};
use brp_render::{GpuContext, GpuInstance};
use std::sync::Arc;

fn read_f32(path: &std::path::Path) -> anyhow::Result<Vec<f32>> {
    Ok(std::fs::read(path)?.chunks_exact(4).map(|b| f32::from_le_bytes(b.try_into().unwrap())).collect())
}

/// PSNR on display code values (linear ^ 1/2.4), peak 1.
fn psnr(a: &[f32], b: &[f32]) -> f64 {
    let v = |x: f32| x.max(0.0).powf(1.0 / 2.4) as f64;
    let mse = a.iter().zip(b).map(|(x, y)| (v(*x) - v(*y)).powi(2)).sum::<f64>() / a.len() as f64;
    -10.0 * mse.max(1e-20).log10()
}

fn main() -> anyhow::Result<()> {
    env_logger::init();
    let a: Vec<String> = std::env::args().skip(1).collect();
    anyhow::ensure!(!a.is_empty(), "learnedcheck REF_DIR [RUNS]");
    let dir = std::path::Path::new(&a[0]);
    let shape: Vec<usize> = std::fs::read_to_string(dir.join("shape.txt"))?.split_whitespace().map(|v| v.parse()).collect::<Result<_, _>>()?;
    let (t, h, w) = (shape[0], shape[1], shape[2]);
    let n = 3 * h * w;
    let (input, want) = (read_f32(&dir.join("in.f32"))?, read_f32(&dir.join("out.f32"))?);
    anyhow::ensure!(input.len() == t * n && want.len() == t * n, "sizes do not match shape.txt");

    let ctx = Arc::new(GpuContext::headless(GpuInstance::new()?, std::env::var("BRP_GPU").ok().as_deref())?);
    let compute: Arc<Compute> = Compute::new(ctx);
    let ops = Ops::new(&compute)?;
    let dn = Learned::new(&ops, &brp_ai::models_dir())?;
    println!("model: {}; {t} frames of {w}x{h}", dn.source);

    let frames: Vec<Tensor> = (0..t).map(|i| Tensor::from_image(&compute, &Image { w, h, channels: 3, data: input[i * n..(i + 1) * n].to_vec() })).collect::<anyhow::Result<_>>()?;
    let mut state: Option<Tensor> = None;
    let mut worst = f64::MAX;
    for i in 0..t {
        let mut rec = Recorder::new(&compute)?;
        // As ref_dump.py: the first frame has no picture before and no state.
        let before = if i > 0 { Some(&frames[i - 1]) } else { None };
        let after = frames.get(i + 1);
        let (out, st) = dn.record(&mut rec, &frames[i], before, after, state.as_ref(), true, 1.0, true)?;
        rec.submit_wait()?;
        state = Some(st);
        let got = out.to_image(&compute)?;
        let (p, moved) = (psnr(&got.data, &want[i * n..(i + 1) * n]), psnr(&input[i * n..(i + 1) * n], &want[i * n..(i + 1) * n]));
        worst = worst.min(p);
        println!("frame {i:2}: {p:6.2} dB against PyTorch (input: {moved:5.2} dB)");
    }
    println!("worst frame: {worst:.2} dB");

    let runs: usize = a.get(1).and_then(|v| v.parse().ok()).unwrap_or(0);
    if runs > 0 {
        let size = std::env::var("BRP_SIZE").unwrap_or_else(|_| "1920x1080".into());
        let (sw, sh) = size.split_once('x').and_then(|(a, b)| Some((a.parse::<usize>().ok()?, b.parse::<usize>().ok()?))).unwrap_or((1920, 1080));
        // The reference frames tiled over the size (real texture and noise).
        let big: Vec<Tensor> = (0..3.min(t))
            .map(|i| {
                let mut img = Image::new(sw, sh, 3);
                for c in 0..3 {
                    for y in 0..sh {
                        for x in 0..sw {
                            img.data[(c * sh + y) * sw + x] = input[i * n + (c * h + y % h) * w + x % w];
                        }
                    }
                }
                Tensor::from_image(&compute, &img)
            })
            .collect::<anyhow::Result<_>>()?;
        let mut state: Option<Tensor> = None;
        let mut times = Vec::new();
        for _ in 0..runs {
            let mut rec = Recorder::new(&compute)?;
            let (out, st) = dn.record(&mut rec, &big[1 % big.len()], Some(&big[0]), big.get(2), state.as_ref(), true, 1.0, true)?;
            times.push(rec.submit_wait()?.as_secs_f64() * 1e3);
            if let Some(old) = state.replace(st) {
                ops.pool.put(old.buf);
            }
            ops.pool.put(out.buf);
        }
        times.sort_by(|a, b| a.partial_cmp(b).unwrap());
        println!("{sw}x{sh}: {:.1} ms per frame (median of {runs}; best {:.1})", times[times.len() / 2], times[0]);
    }
    Ok(())
}
