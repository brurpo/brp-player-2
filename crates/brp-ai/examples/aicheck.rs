//! Runs the AI models and the temporal processing on PNG frames, on the GPU.
//!   aicheck upscale <model> in.png out.png        (model: general4x general2x fast2x fast3x fast4x anime4x
//!                                                  generalv3 natural2x swin4x dat4x datreal4x)
//!   aicheck rife <model> a.png b.png t out.png    (model: v4.26 v4.25-lite)
//!   aicheck matting out_prefix a.png b.png ...    (frames of one shot, in order)
//!   aicheck flow a.png                            (motion of a shifted copy; BRP_SHIFT=px)
//!   aicheck lkcheck                               (motion on synthetic waves of 3-32 px; BRP_SHIFT=px)
//!   aicheck temporal <model> out_prefix a.png ... (per frame vs the temporal window: flicker, fidelity.
//!       Synthetic: BRP_GRAIN=sigma (BRP_GRAIN_SIZE=s coarse, BRP_GRAIN_SHOT=1 brighter = stronger),
//!       BRP_PAN=px per frame, BRP_LINE=1 (ghosting). Window: BRP_RADIUS, BRP_DECAY, BRP_STRENGTH,
//!       BRP_SMOOTH. Denoiser: BRP_DENOISE=spatial (BRP_DN_MODE, BRP_DN_TAU, BRP_DN_CHROMA).
//!       Merge after upscaling: BRP_POST=r. Motion: BRP_EXACT_FLOW=1 (the true pan),
//!       BRP_FLOW_ERR=1 (its error, overall and where the picture has detail; the clean
//!       previous picture warped by it against the clean current one; maps _ferr, _wres),
//!       BRP_FLOW_ONLY=1 (just that, on one frame), BRP_HINT=1 (each motion also tries the
//!       next pair's), BRP_EXACT_OFF=px (the true pan plus a constant error).)
//!   aicheck gtemp a.png                           (GPU temporal ops against the CPU reference; BRP_DBG=1:
//!                                                  the near neighbour's agreement)
//!   aicheck vkop "<layer line>" c h w             (one network layer on a test tensor)
//!   aicheck searaft|waft out_dir a.png b.png ...  (AI flow of each consecutive pair, raw f32 for tools/flowref)
//!   aicheck warm                                  (build every AI shader, as at start-up; shader cache stats)
//! BRP_GPU picks the GPU by name.

use brp_ai::gpu::{Matting, Ops, Rife, Upscaler};
use brp_ai::gpu_temporal::Temporal;
use brp_ai::rife::RifeModel;
use brp_ai::upscale::UpscaleModel;
use brp_ai::vk::Tensor;
use brp_ai::Image;
use brp_render::compute::{Compute, Recorder};
use brp_render::{GpuContext, GpuInstance};
use std::sync::Arc;
use std::time::Instant;

fn gpu() -> anyhow::Result<(Arc<Compute>, Arc<Ops>)> {
    let ctx = Arc::new(GpuContext::headless(GpuInstance::new()?, std::env::var("BRP_GPU").ok().as_deref())?);
    let compute = Compute::new(ctx);
    let ops = Ops::new(&compute)?;
    Ok((compute, ops))
}

fn upscale_model(s: &str) -> UpscaleModel {
    match s {
        "general4x" => UpscaleModel::General4x,
        "general2x" => UpscaleModel::General2x,
        "fast3x" => UpscaleModel::Fast3x,
        "fast4x" => UpscaleModel::Fast4x,
        "anime4x" => UpscaleModel::Anime4x,
        "generalv3" => UpscaleModel::GeneralV3,
        "natural2x" => UpscaleModel::Natural2x,
        "swin4x" => UpscaleModel::Swin4x,
        "dat4x" => UpscaleModel::Dat4x,
        "datreal4x" => UpscaleModel::DatReal4x,
        _ => UpscaleModel::Fast2x,
    }
}

fn main() -> anyhow::Result<()> {
    env_logger::init();
    let a: Vec<String> = std::env::args().skip(1).collect();
    let models = brp_ai::models_dir();
    match a.first().map(String::as_str) {
        Some("upscale") => {
            let m = upscale_model(&a[1]);
            let (compute, ops) = gpu()?;
            let t = Instant::now();
            let dn: f32 = std::env::var("BRP_DN").ok().and_then(|v| v.parse().ok()).unwrap_or(0.5);
            let up = Upscaler::with_denoise(&ops, &models, m, dn)?;
            println!("loaded {m:?} in {:?}", t.elapsed());
            let img = load(&a[2])?;
            let ti = Tensor::from_image(&compute, &img)?;
            let mut out = None;
            let runs: usize = std::env::var("BRP_RUNS").ok().and_then(|v| v.parse().ok()).unwrap_or(3);
            for i in 0..runs {
                let mut rec = Recorder::new(&compute)?;
                let o = up.record(&mut rec, &ti)?;
                let t = rec.submit_wait()?;
                println!("run {i}: {}x{} -> {}x{} in {t:?}", img.w, img.h, o.w, o.h);
                out = Some(o);
            }
            let r = compute.ctx.allocator.lock().unwrap().generate_report();
            println!("GPU memory held: {} MB allocated, {} MB in blocks", r.total_allocated_bytes >> 20, r.total_capacity_bytes >> 20);
            save(&a[3], &out.unwrap().to_image(&compute)?)?;
        }
        Some("rife") => {
            let m = if a[1] == "v4.25-lite" { RifeModel::V4_25Lite } else { RifeModel::V4_26 };
            let (compute, ops) = gpu()?;
            let t = Instant::now();
            let rife = Rife::new(&ops, &models, m)?;
            println!("loaded {m:?} in {:?}", t.elapsed());
            let (x, y) = (load(&a[2])?, load(&a[3])?);
            let (tx, ty) = (Tensor::from_image(&compute, &x)?, Tensor::from_image(&compute, &y)?);
            let tt: f32 = a[4].parse()?;
            let mut out = None;
            for i in 0..3 {
                let mut rec = Recorder::new(&compute)?;
                let o = rife.record(&mut rec, &tx, &ty, tt)?;
                let t = rec.submit_wait()?;
                println!("run {i}: {}x{} t={tt} in {t:?}", x.w, x.h);
                out = Some(o);
            }
            save(&a[5], &out.unwrap().to_image(&compute)?)?;
        }
        Some("interp") => {
            // aicheck interp rife|builtin|searaft|waft a.png c.png b.png out.png: the
            // picture halfway between a and c, scored against the real middle one b
            // (PSNR over RGB), and saved.
            let (compute, ops) = gpu()?;
            let tm = Temporal::new(&ops)?;
            let (x, y, truth) = (load(&a[2])?, load(&a[3])?, load(&a[4])?);
            let (tx, ty) = (Tensor::from_image(&compute, &x)?, Tensor::from_image(&compute, &y)?);
            // "+rife": RIFE's picture where the motion proves wrong.
            let (method, hybrid) = (a[1].trim_end_matches("+rife"), a[1].ends_with("+rife"));
            let rife = if method == "rife" || hybrid { Some(Rife::new(&ops, &models, RifeModel::V4_26)?) } else { None };
            let sr = if method == "searaft" { Some(brp_ai::flow::SeaRaft::new(&ops, &models)?) } else { None };
            let wf = if method == "waft" { Some(brp_ai::flow::Waft::new(&ops, &models)?) } else { None };
            let mut out = None;
            let mut aligned: Option<Tensor> = None;
            for _ in 0..2 {
                let mut keep = Vec::new();
                let mut rec = Recorder::new(&compute)?;
                let o = if let (Some(r), false) = (&rife, hybrid) {
                    r.record(&mut rec, &tx, &ty, 0.5)?
                } else {
                    let (fab, fba) = if let Some(s) = &sr {
                        (s.flow(&mut rec, &tx, &ty)?, s.flow(&mut rec, &ty, &tx)?)
                    } else if let Some(w) = &wf {
                        (w.flow(&mut rec, &tx, &ty)?, w.flow(&mut rec, &ty, &tx)?)
                    } else {
                        (tm.motion(&mut rec, &ty, &tx)?, tm.motion(&mut rec, &tx, &ty)?)
                    };
                    let shot = ops.filled(&mut rec, 1, 1, 1, 1.0)?;
                    // How well the motion lines c up onto a (c moved along a's motion).
                    let al = ops.tensor(3, tx.h, tx.w)?;
                    tm.warp_into(&mut rec, &ty, &fab, &al)?;
                    aligned = Some(al);
                    let fbk = match &rife {
                        Some(r) => Some(r.record(&mut rec, &tx, &ty, 0.5)?),
                        None => None,
                    };
                    let o = tm.interpolate(&mut rec, &tx, &ty, &fab, &fba, 0.5, &shot, fbk.as_ref())?;
                    keep.extend([fab, fba, shot]);
                    keep.extend(fbk);
                    o
                };
                let t = rec.submit_wait()?;
                drop(keep);
                println!("{}: {}x{} in {t:?}", a[1], x.w, x.h);
                out = Some(o);
            }
            let img = out.unwrap().to_image(&compute)?;
            let n = img.w * img.h;
            let mut se = 0.0f64;
            for c in 0..3 {
                for i in 0..n {
                    let d = (img.data[c * n + i] - truth.data[c * n + i]) as f64;
                    se += d * d;
                }
            }
            println!("{}: PSNR {:.2} dB against the real middle picture", a[1], 10.0 * (1.0 / (se / (3 * n) as f64)).log10());
            if let Some(al) = aligned {
                let al = al.to_image(&compute)?;
                let (mut se, mut cnt) = (0.0f64, 0usize);
                for c in 0..3 {
                    for i in 0..n {
                        let v = al.data[c * n + i];
                        if v.is_finite() {
                            let d = (v - x.data[c * n + i]) as f64;
                            se += d * d;
                            cnt += 1;
                        }
                    }
                }
                println!("{}: c moved onto a: PSNR {:.2} dB", a[1], 10.0 * (1.0 / (se / cnt as f64)).log10());
            }
            save(&a[5], &img)?;
        }
        Some("matting") => {
            let frames: Vec<Image> = a[2..].iter().map(|p| load(p)).collect::<anyhow::Result<_>>()?;
            let (w, h) = (frames[0].w, frames[0].h);
            let (compute, ops) = gpu()?;
            let tm = Temporal::new(&ops)?;
            let t = Instant::now();
            let mut m = Matting::new(&ops, &models, w, h)?;
            println!("loaded matting for {w}x{h} in {:?}", t.elapsed());
            for (i, f) in frames.iter().enumerate() {
                let src = Tensor::from_image(&compute, f)?;
                let mut rec = Recorder::new(&compute)?;
                let r = m.record(&mut rec, &src)?;
                let comp = tm.matte(&mut rec, &r, &src, 1)?;
                let matte = tm.matte(&mut rec, &r, &src, 3)?;
                let t = rec.submit_wait()?;
                let matte = matte.to_image(&compute)?;
                let mean = matte.plane(0).iter().sum::<f32>() / (w * h) as f32;
                println!("frame {i}: {t:?}, mean alpha {mean:.3}");
                save(&format!("{}_{i}_comp.png", a[1]), &comp.to_image(&compute)?)?;
                save(&format!("{}_{i}_alpha.png", a[1]), &matte)?;
            }
        }
        Some("flow") => {
            // Expect (-sh, -sh*5/12).
            let a0 = load(&a[1])?;
            let sh: usize = std::env::var("BRP_SHIFT").ok().and_then(|v| v.parse().ok()).unwrap_or(12);
            let mut b0 = a0.clone();
            let n = a0.w * a0.h;
            for c in 0..3 {
                for y in 0..a0.h {
                    for x in 0..a0.w {
                        let (sx, sy) = (x.saturating_sub(sh), y.saturating_sub(sh * 5 / 12));
                        b0.data[c * n + y * a0.w + x] = a0.data[c * n + sy * a0.w + sx];
                    }
                }
            }
            let (compute, ops) = gpu()?;
            let tm = Temporal::new(&ops)?;
            let (ta, tb) = (Tensor::from_image(&compute, &a0)?, Tensor::from_image(&compute, &b0)?);
            let mut rec = Recorder::new(&compute)?;
            let f = tm.motion(&mut rec, &ta, &tb)?;
            let t = rec.submit_wait()?;
            let f = f.to_image(&compute)?;
            let k = f.w as f32 / a0.w as f32;
            let median = |p: &[f32]| {
                let mut v: Vec<f32> = p.iter().map(|v| v / k).collect();
                v.sort_by(f32::total_cmp);
                v[v.len() / 2]
            };
            println!("{}x{} in {t:?}, median dx {:.2} dy {:.2}", f.w, f.h, median(f.plane(0)), median(f.plane(1)));
        }
        Some("custom") => {
            // aicheck custom <model.param> <model.bin> in.png out.png: any ncnn-format
            // network (input "in0", output "out0", values 0..1) on the whole picture,
            // f16 activations; BRP_SUMS16=1 also sums in f16.
            let (compute, _ops) = gpu()?;
            let text = std::fs::read_to_string(&a[1])?;
            let mut net = brp_ai::vk::Net::load(&compute, &text, &std::fs::read(&a[2])?)?;
            net.set_half(true);
            net.set_fp16_sums(std::env::var_os("BRP_SUMS16").is_some());
            let img = load(&a[3])?;
            let ti = Tensor::from_image(&compute, &img)?;
            let mut out = None;
            for i in 0..3 {
                let mut rec = Recorder::new(&compute)?;
                let names = net.blob_names();
                let (i_name, o_name) = if names.iter().any(|n| n == "in0") {
                    ("in0", "out0")
                } else if names.iter().any(|n| n == "img") {
                    ("img", "out")
                } else {
                    ("data", "output")
                };
                // BRP_BLOBS=a,b: also print those blobs' largest magnitudes.
                let extra = std::env::var("BRP_BLOBS").unwrap_or_default();
                let mut want = vec![o_name];
                want.extend(extra.split(',').filter(|s| !s.is_empty()));
                let mut os = net.record(&mut rec, &[(i_name, &ti)], &want)?;
                let o = os.remove(0);
                let t = rec.submit_wait()?;
                if i == 2 {
                    for (n, b) in want[1..].iter().zip(&os) {
                        let img = b.to_image(&compute)?;
                        let m = img.data.iter().fold(0f32, |m, v| if v.is_finite() { m.max(v.abs()) } else { f32::INFINITY });
                        println!("{n}: {}x{}x{} max |v| {m}{}", b.c, b.h, b.w, if b.half { " (f16)" } else { "" });
                    }
                }
                println!("run {i}: {}x{} -> {}x{} in {t:?}", img.w, img.h, o.w, o.h);
                out = Some(o);
            }
            let r = compute.ctx.allocator.lock().unwrap().generate_report();
            println!("GPU memory held: {} MB allocated", r.total_allocated_bytes >> 20);
            let out = out.unwrap().to_image(&compute)?;
            // BRP_REF=out.f32 (from onnxref): PSNR against ONNX Runtime's raw output.
            if let Ok(p) = std::env::var("BRP_REF") {
                let raw = std::fs::read(&p)?;
                let r: Vec<f32> = raw[12..].chunks_exact(4).map(|b| f32::from_le_bytes(b.try_into().unwrap())).collect();
                anyhow::ensure!(r.len() == out.data.len(), "reference has {} values, output {}", r.len(), out.data.len());
                let mse = r.iter().zip(&out.data).map(|(a, b)| ((a - b) as f64).powi(2)).sum::<f64>() / r.len() as f64;
                let worst = r.iter().zip(&out.data).map(|(a, b)| (a - b).abs()).fold(0f32, f32::max);
                println!("against the reference: {:.1} dB, largest difference {worst:.4}", 10.0 * (1.0 / mse.max(1e-20)).log10());
            }
            save(&a[4], &out)?;
        }
        Some("temporal") => temporal_check(&a, &models)?,
        Some("gtemp") => gtemp_check(&a[1])?,
        Some("lkcheck") => lk_check()?,
        Some("vkop") => vk_op(&a[1..])?,
        Some("stab") => {
            // aicheck stab radius zoom out_dir a.png b.png ...: stabilised frames (built-in
            // motion) as out_dir/sNNNN.png, and the jitter before and after: the mean
            // change from one frame-to-frame camera movement to the next (shift px,
            // rotation in 1/1000 rad), refitted on the output.
            use brp_ai::stabilise::{fit, steady, Sim, Stabiliser};
            let (compute, ops) = gpu()?;
            let tm = Temporal::new(&ops)?;
            let st = Stabiliser::new(&ops)?;
            let (radius, zoom): (usize, f32) = (a[1].parse()?, a[2].parse()?);
            std::fs::create_dir_all(&a[3])?;
            let frames: Vec<Image> = a[4..].iter().map(|p| load(p)).collect::<anyhow::Result<_>>()?;
            let steps = |imgs: &[Tensor]| -> anyhow::Result<Vec<Option<Sim>>> {
                let mut out = Vec::new();
                for i in 1..imgs.len() {
                    let mut rec = Recorder::new(&compute)?;
                    // Pixel of frame i-1 into frame i (as the player's motion(from, to)).
                    let f = tm.motion(&mut rec, &imgs[i], &imgs[i - 1])?;
                    let g = st.grid(&mut rec, &f)?;
                    rec.submit_wait()?;
                    let g = g.to_image(&compute)?;
                    out.push(fit(&g, imgs[i].w, imgs[i].h, imgs[i].w as f64 / f.w as f64));
                }
                Ok(out)
            };
            let jitter = |s: &[Option<Sim>]| {
                let v: Vec<(f64, f64, f64)> = s.windows(2).filter_map(|w| Some((w[0]?, w[1]?))).map(|(p, q)| ((q.tx - p.tx).abs(), (q.ty - p.ty).abs(), (q.b - p.b).abs() * 1000.0)).collect();
                let n = v.len().max(1) as f64;
                (v.iter().map(|x| x.0).sum::<f64>() / n, v.iter().map(|x| x.1).sum::<f64>() / n, v.iter().map(|x| x.2).sum::<f64>() / n)
            };
            let ins: Vec<Tensor> = frames.iter().map(|f| Tensor::from_image(&compute, f)).collect::<anyhow::Result<_>>()?;
            let s = steps(&ins)?;
            let mut outs = Vec::new();
            for t in 0..ins.len() {
                let win: Vec<Option<Sim>> = (0..2 * radius).map(|i| {
                    let k = t as i64 - radius as i64 + i as i64;
                    if k < 0 { None } else { s.get(k as usize).copied().flatten() }
                }).collect();
                let c = steady(&win, radius);
                let mut rec = Recorder::new(&compute)?;
                let o = st.warp(&mut rec, &ins[t], c, zoom)?;
                rec.submit_wait()?;
                save(&format!("{}/s{t:04}.png", a[3]), &o.to_image(&compute)?)?;
                outs.push(o);
            }
            let s2 = steps(&outs)?;
            let (j1, j2) = (jitter(&s), jitter(&s2));
            println!("jitter before: x {:.2} px, y {:.2} px, rotation {:.2} mrad", j1.0, j1.1, j1.2);
            println!("jitter after:  x {:.2} px, y {:.2} px, rotation {:.2} mrad", j2.0, j2.1, j2.2);
            println!("fits found: {} of {} pairs", s.iter().flatten().count(), s.len());
        }
        Some("depth") => {
            // aicheck depth vda|da out_dir a.png b.png ...: depth of each frame in order
            // (Video Depth Anything streams through them), raw f32 as depth_ref.py writes
            // (out_dir/dNNNN.f32), and timings.
            let (compute, ops) = gpu()?;
            let model = if a[1] == "da" { brp_ai::depth::DepthModel::Frame } else { brp_ai::depth::DepthModel::Video };
            let d = brp_ai::depth::Depth::new(&ops, &models, model)?;
            std::fs::create_dir_all(&a[2])?;
            let mut times = Vec::new();
            for (i, p) in a[3..].iter().enumerate() {
                let img = load(p)?;
                let t = Tensor::from_image(&compute, &img)?;
                let mut rec = Recorder::new(&compute)?;
                let o = d.record(&mut rec, &t)?;
                times.push(rec.submit_wait()?);
                let o = o.to_image(&compute)?;
                let mut raw = Vec::new();
                for v in [1u32, o.h as u32, o.w as u32] {
                    raw.extend(v.to_le_bytes());
                }
                o.data.iter().for_each(|v| raw.extend(v.to_le_bytes()));
                std::fs::write(format!("{}/d{i:04}.f32", a[2]), raw)?;
            }
            times.sort();
            println!("{:?}: {} frames, median {:?} per frame (first {:?})", model, a.len() - 3, times[times.len() / 2], times.last());
            let r = compute.ctx.allocator.lock().unwrap().generate_report();
            println!("GPU memory held: {} MB allocated", r.total_allocated_bytes >> 20);
        }
        Some(m @ ("searaft" | "waft")) => {
            // aicheck searaft|waft out_dir a.png b.png ...: the model's flow from each frame
            // to the next, written as flowdump writes ours (for tools/flowref/score.py).
            let (compute, ops) = gpu()?;
            enum M { S(brp_ai::flow::SeaRaft), W(brp_ai::flow::Waft) }
            let model = if m == "waft" { M::W(brp_ai::flow::Waft::new(&ops, &models)?) } else { M::S(brp_ai::flow::SeaRaft::new(&ops, &models)?) };
            std::fs::create_dir_all(&a[1])?;
            let frames: Vec<Image> = a[2..].iter().map(|p| load(p)).collect::<anyhow::Result<_>>()?;
            for i in 1..frames.len() {
                let (ta, tb) = (Tensor::from_image(&compute, &frames[i - 1])?, Tensor::from_image(&compute, &frames[i])?);
                let mut f = None;
                for _ in 0..2 {
                    let mut rec = Recorder::new(&compute)?;
                    let flow = match &model {
                        M::S(s) => s.flow(&mut rec, &ta, &tb)?,
                        M::W(w) => w.flow(&mut rec, &ta, &tb)?,
                    };
                    let t = rec.submit_wait()?;
                    println!("pair {}: {}x{} in {t:?}", i - 1, flow.w, flow.h);
                    f = Some(flow.to_image(&compute)?);
                }
                let f = f.unwrap();
                let mut raw = Vec::new();
                for d in [2u32, f.h as u32, f.w as u32] {
                    raw.extend(d.to_le_bytes());
                }
                f.data.iter().for_each(|v| raw.extend(v.to_le_bytes()));
                std::fs::write(format!("{}/{:03}.f32", a[1], i - 1), raw)?;
            }
            let r = compute.ctx.allocator.lock().unwrap().generate_report();
            println!("GPU memory held: {} MB allocated", r.total_allocated_bytes >> 20);
        }
        Some("flowdump") => {
            // aicheck flowdump out_dir a.png b.png ...: our motion from each frame to the next
            // as raw f32 (c, h, w as u32, then dx, dy planes; pixels at its own size,
            // half the picture's above 1280 wide), for tools/flowref/score.py.
            let (compute, ops) = gpu()?;
            let tm = Temporal::new(&ops)?;
            std::fs::create_dir_all(&a[1])?;
            let frames: Vec<Image> = a[2..].iter().map(|p| load(p)).collect::<anyhow::Result<_>>()?;
            for i in 1..frames.len() {
                let (ta, tb) = (Tensor::from_image(&compute, &frames[i - 1])?, Tensor::from_image(&compute, &frames[i])?);
                let mut rec = Recorder::new(&compute)?;
                let flow = tm.motion(&mut rec, &tb, &ta)?;
                let t = rec.submit_wait()?;
                let f = flow.to_image(&compute)?;
                let mut raw = Vec::new();
                for d in [2u32, f.h as u32, f.w as u32] {
                    raw.extend(d.to_le_bytes());
                }
                f.data.iter().for_each(|v| raw.extend(v.to_le_bytes()));
                std::fs::write(format!("{}/{:03}.f32", a[1], i - 1), raw)?;
                println!("pair {}: {}x{} in {t:?}", i - 1, f.w, f.h);
            }
        }
        Some("shots") => {
            // aicheck shots a.png b.png ...: each consecutive pair's same-shot measure (mean
            // agreement after motion; under brp_ai::gpu_temporal::CUT = a scene cut).
            let (compute, ops) = gpu()?;
            let tm = Temporal::new(&ops)?;
            let frames: Vec<Image> = a[1..].iter().map(|p| load(p)).collect::<anyhow::Result<_>>()?;
            for i in 1..frames.len() {
                let (ta, tb) = (Tensor::from_image(&compute, &frames[i - 1])?, Tensor::from_image(&compute, &frames[i])?);
                let mut rec = Recorder::new(&compute)?;
                let flow = tm.motion(&mut rec, &tb, &ta)?;
                let shot = tm.same_shot(&mut rec, &ta, &tb, &flow)?;
                rec.submit_wait()?;
                let v = shot.to_image(&compute)?.data[0];
                // What the consumers do with it: the in-between picture at t = 0.3 (stood in
                // for by `b`) becomes `a` across a cut; motion blur from the motion to `b`
                // alone leaves `a` as it is across a cut.
                let mut rec = Recorder::new(&compute)?;
                let pick = tm.shot_pick(&mut rec, &tb, &ta, &tb, 0.3, &shot)?;
                let back = tm.motion(&mut rec, &tb, &ta)?;
                let blur = tm.motion_blur(&mut rec, &ta, None, Some(&back), [None, Some(&shot)], 1.0, 16)?;
                rec.submit_wait()?;
                let diff = |x: &Tensor| -> anyhow::Result<f32> {
                    let (p, q) = (x.to_image(&compute)?, frames[i - 1].clone());
                    Ok(p.data.iter().zip(&q.data).map(|(u, v)| (u - v).abs()).sum::<f32>() / p.data.len() as f32)
                };
                println!("{} -> {}: {v:.3}{}  (pick vs a {:.4}, blur vs a {:.4})", a[i], a[i + 1], if v < brp_ai::gpu_temporal::CUT { "  CUT" } else { "" }, diff(&pick)?, diff(&blur)?);
            }
        }
        Some("warm") => {
            // aicheck warm: build every AI shader (as the player does at start-up).
            let (compute, _ops) = gpu()?;
            let props = compute.ctx.properties();
            if !brp_render::shader_cache::needs_warm_up(&props) {
                println!("already built for this build and driver: nothing to do");
                return Ok(());
            }
            let t = Instant::now();
            brp_ai::warm::warm_up(&compute, &models, &|d, n| println!("  {d}/{n} at {:.1} s", t.elapsed().as_secs_f64()))?;
            compute.ctx.save_pipeline_cache();
            brp_render::shader_cache::warm_up_done(&props);
            use std::sync::atomic::Ordering::Relaxed;
            println!("done in {:.1} s: {} compiled, {} from the cache", t.elapsed().as_secs_f64(), brp_render::shader_cache::COMPILED.load(Relaxed), brp_render::shader_cache::HITS.load(Relaxed));
        }
        _ => anyhow::bail!("usage: see the top of crates/brp-ai/examples/aicheck.rs"),
    }
    if std::env::var_os("BRP_CACHE_STATS").is_some() {
        use std::sync::atomic::Ordering::Relaxed;
        println!("shaders: {} compiled, {} from the cache", brp_render::shader_cache::COMPILED.load(Relaxed), brp_render::shader_cache::HITS.load(Relaxed));
    }
    Ok(())
}

/// Upscales a sequence per frame and through the temporal window; prints flicker and
/// fidelity.
///   BRP_GRAIN=sigma: frames are the first picture with fresh grain each;
///   BRP_PAN=1: ... also moving right 1 px per frame (flicker measured exactly);
///   BRP_LINE=1: ... with a thin red line moving 4 px per frame (ghosting measured).
fn temporal_check(a: &[String], models: &std::path::Path) -> anyhow::Result<()> {
    let (compute, ops) = gpu()?;
    let tm = Temporal::new(&ops)?;
    let up = Upscaler::new(&ops, models, upscale_model(&a[1]))?;
    let mut frames: Vec<Image> = a[3..].iter().map(|p| load(p)).collect::<anyhow::Result<_>>()?;
    let pan = std::env::var("BRP_PAN").is_ok();
    // BRP_PAN=s: pixels per frame (default 1).
    let speed: usize = std::env::var("BRP_PAN").ok().and_then(|v| v.parse().ok()).unwrap_or(1);
    let line = std::env::var("BRP_LINE").is_ok();
    // BRP_RADIUS: neighbours each way (default 2); BRP_DECAY: weight per step out.
    let radius: usize = std::env::var("BRP_RADIUS").ok().and_then(|v| v.parse().ok()).unwrap_or(2);
    let decay: f32 = std::env::var("BRP_DECAY").ok().and_then(|v| v.parse().ok()).unwrap_or(brp_ai::gpu_temporal::DECAY);
    // BRP_SMOOTH=k: smooth the grain left after the merge (range k x the grain level).
    let smooth: f32 = std::env::var("BRP_SMOOTH").ok().and_then(|v| v.parse().ok()).unwrap_or(0.0);
    const LINE_X0: usize = 120;
    const LINE_SPEED: usize = 4;
    let mut clean: Vec<Image> = Vec::new();
    if let Some(sigma) = std::env::var("BRP_GRAIN").ok().and_then(|v| v.parse::<f32>().ok()) {
        let base = frames[0].clone();
        let mut seed = 12345u64;
        let mut rnd = || {
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            ((seed >> 33) as f32 / (1u64 << 31) as f32) - 0.5
        };
        let (w, h, n) = (base.w, base.h, base.w * base.h);
        frames = (0..2 * (radius + std::env::var("BRP_POST").ok().and_then(|v| v.parse::<usize>().ok()).unwrap_or(0)) + 5)
            .map(|t| {
                let mut f = base.clone();
                if pan {
                    for c in 0..3 {
                        for y in 0..h {
                            for x in 0..w {
                                f.data[c * n + y * w + x] = base.data[c * n + y * w + x.saturating_sub(t * speed)];
                            }
                        }
                    }
                }
                if line {
                    let lx = LINE_X0 + LINE_SPEED * t;
                    for y in 0..h {
                        for (c, v) in [1.0f32, 0.15, 0.15].into_iter().enumerate() {
                            f.data[c * n + y * w + lx] = v;
                            f.data[c * n + y * w + lx + 1] = v;
                        }
                    }
                }
                clean.push(f.clone());
                // BRP_GRAIN_SIZE=s: grain made at 1/s size and enlarged (coarse, like
                // film); BRP_GRAIN_SHOT=1: stronger in bright areas (like sensors).
                let size: usize = std::env::var("BRP_GRAIN_SIZE").ok().and_then(|v| v.parse().ok()).unwrap_or(1).max(1);
                let shot = std::env::var("BRP_GRAIN_SHOT").is_ok();
                let (gw, gh) = (w.div_ceil(size) + 1, h.div_ceil(size) + 1);
                let g: Vec<f32> = (0..3 * gw * gh).map(|_| rnd()).collect();
                // Bilinear enlargement keeps less than the full deviation: rescale.
                let gain = if size > 1 { 1.5 } else { 1.0 };
                for c in 0..3 {
                    for y in 0..h {
                        for x in 0..w {
                            let (fx, fy) = (x as f32 / size as f32, y as f32 / size as f32);
                            let (x0, y0) = (fx as usize, fy as usize);
                            let (ax, ay) = (fx - x0 as f32, fy - y0 as f32);
                            let at = |xx: usize, yy: usize| g[c * gw * gh + yy * gw + xx];
                            let v = (at(x0, y0) * (1.0 - ax) + at(x0 + 1, y0) * ax) * (1.0 - ay) + (at(x0, y0 + 1) * (1.0 - ax) + at(x0 + 1, y0 + 1) * ax) * ay;
                            let p = &mut f.data[c * n + y * w + x];
                            let k = if shot { 2.0 * p.max(0.0).sqrt() * 0.7 } else { 1.0 };
                            *p = (*p + sigma * 3.46 * gain * k * v).clamp(0.0, 1.0);
                        }
                    }
                }
                f
            })
            .collect();
    }
    // BRP_GRAIN_REAL=sigma: fresh grain on each of the given (real) frames, which are
    // kept as the clean reference.
    if let Some(sigma) = std::env::var("BRP_GRAIN_REAL").ok().and_then(|v| v.parse::<f32>().ok()) {
        let mut seed = 4242u64;
        let mut rnd = || {
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            ((seed >> 33) as f32 / (1u64 << 31) as f32) - 0.5
        };
        clean = frames.clone();
        for f in frames.iter_mut() {
            for v in f.data.iter_mut() {
                *v = (*v + sigma * 3.46 * rnd()).clamp(0.0, 1.0);
            }
        }
    }
    let strength: f32 = std::env::var("BRP_STRENGTH").ok().and_then(|v| v.parse().ok()).unwrap_or(1.0);
    // BRP_DENOISE=k: our denoiser (spatial strength k) in place of the merge;
    // BRP_DN_MODE (0 hard threshold, 1 Wiener), BRP_DN_TAU, BRP_DN_CHROMA.
    let envf = |k: &str| std::env::var(k).ok().and_then(|v| v.parse::<f32>().ok());
    let dn = match envf("BRP_DENOISE") {
        Some(k) => {
            let mut s = brp_ai::denoise::DenoiseSettings { spatial: k, decay, temporal: strength, ..Default::default() };
            if let Some(m) = envf("BRP_DN_MODE") {
                s.mode = m as i32;
            }
            if let Some(t) = envf("BRP_DN_TAU") {
                s.tau = t;
            }
            if let Some(c) = envf("BRP_DN_CHROMA") {
                s.chroma = c;
            }
            Some((brp_ai::denoise::Denoiser::new(&ops)?, s))
        }
        None => None,
    };
    let (w0, h0) = (frames[0].w, frames[0].h);
    let t: Vec<Tensor> = frames.iter().map(|f| Tensor::from_image(&compute, f)).collect::<anyhow::Result<_>>()?;
    let mean_abs = |x: &Image, y: &Image| x.data.iter().zip(&y.data).map(|(a, b)| (a - b).abs()).sum::<f32>() / x.data.len() as f32;
    // Exact flicker for the synthetic pan: output vs previous output moved by the
    // known motion (whole output pixels), away from the borders.
    let exact = |cur: &Image, prev: &Image| {
        let k = cur.w / w0;
        let sh = if pan { k * speed } else { 0 };
        let (w, h, n) = (cur.w, cur.h, cur.w * cur.h);
        let (mut s, mut cnt) = (0f64, 0f64);
        for c in 0..3 {
            for y in 16..h - 16 {
                for x in 16 + sh..w - 16 {
                    s += (cur.data[c * n + y * w + x] - prev.data[c * n + y * w + x - sh]).abs() as f64;
                    cnt += 1.0;
                }
            }
        }
        s / cnt
    };
    // The grain left (output less clean) of picture i against that of i - 1 moved along
    // the motion of the clean pictures: mean |difference| relative to independent ones.
    let coherence = |rp: &Image, out: &Image, i: usize| -> anyhow::Result<f64> {
        let (cp, cc) = (Tensor::from_image(&compute, &clean[i - 1])?, Tensor::from_image(&compute, &clean[i])?);
        let mut r2 = Recorder::new(&compute)?;
        let f = tm.motion(&mut r2, &cp, &cc)?;
        let rpt = Tensor::from_image(&compute, rp)?;
        let wd = ops.tensor(3, h0, w0)?;
        tm.warp_into(&mut r2, &rpt, &f, &wd)?;
        r2.submit_wait()?;
        let wr = wd.to_image(&compute)?;
        let n = w0 * h0;
        let (mut d, mut a2, mut b2, mut cnt) = (0f64, 0f64, 0f64, 0f64);
        for c in 0..3 {
            for y in 8..h0 - 8 {
                for x in 8..w0 - 8 {
                    let k = c * n + y * w0 + x;
                    let (a, b) = ((out.data[k] - clean[i].data[k]) as f64, wr.data[k] as f64);
                    if !b.is_finite() {
                        continue;
                    }
                    d += (a - b).abs();
                    a2 += a * a;
                    b2 += b * b;
                    cnt += 1.0;
                }
            }
        }
        // For independent Gaussian leftovers E|a - b| = sqrt(2/pi) sqrt(var a + var b).
        Ok(d / cnt / ((2.0 / std::f64::consts::PI) * (a2 + b2) / cnt).sqrt())
    };
    let mut prev: Option<(Image, Image)> = None;
    let mut prev_in: Option<(Image, Image)> = None;
    let mut prev_res: Option<Image> = None;
    let mut wins: Vec<Option<Tensor>> = vec![None; frames.len()];
    for i in radius..frames.len().saturating_sub(radius) {
        // (Run before this frame's batch is recorded: it submits its own work.)
        let taa_out = match std::env::var("BRP_TAA").ok().and_then(|v| v.parse::<usize>().ok()) {
            Some(k) if i >= 1 => Some(taa(&compute, &ops, &tm, &t, &frames, i, k.min(i))?),
            _ => None,
        };
        let mut rec = Recorder::new(&compute)?;
        // Steps outward: picture i - d (or i + d) from the one a step nearer.
        // BRP_HINT=1: each motion also tries the plain motion of the next pair along
        // (the same way round) as a candidate.
        let hinted = std::env::var("BRP_HINT").is_ok();
        let mo = |rec: &mut Recorder, a: usize, b: usize, ha: usize, hb: usize| -> anyhow::Result<Tensor> {
            if !hinted {
                return tm.motion(rec, &t[a], &t[b]);
            }
            let h = tm.motion(rec, &t[ha], &t[hb])?;
            let m = tm.motion_hinted(rec, &t[a], &t[b], Some(&h))?;
            ops.recycle(rec, h);
            Ok(m)
        };
        let back: Vec<Tensor> = (1..=radius).map(|d| mo(&mut rec, i - d, i - d + 1, i - d + 1, i - d + 2)).collect::<anyhow::Result<_>>()?;
        let fwd: Vec<Tensor> = (1..=radius).map(|d| mo(&mut rec, i + d, i + d - 1, i + d - 1, i + d - 2)).collect::<anyhow::Result<_>>()?;
        if pan && std::env::var("BRP_FLOW_ERR").is_ok() {
            // Error of the estimated motion against the true -1 px (x), 0 (y).
            let f = back[0].clone();
            rec.submit_wait()?;
            rec = Recorder::new(&compute)?;
            let img = f.to_image(&compute)?;
            let n = img.w * img.h;
            let (mut s, mut s2, mut big) = (0f64, 0f64, 0usize);
            for k in 0..n {
                let e = (img.data[k] + speed as f32) as f64;
                let ey = img.data[n + k] as f64;
                s += e;
                s2 += e * e + ey * ey;
                if (e * e + ey * ey).sqrt() > 0.25 {
                    big += 1;
                }
            }
            println!("  motion error ({}x{}): mean dx {:+.4}, rms {:.4} px, over 0.25 px: {:.1}%", img.w, img.h, s / n as f64, (s2 / n as f64).sqrt(), 100.0 * big as f64 / n as f64);
            // Error map: red = dx error, green = dy error (0.5 = none, 1 px per 0.25), blue = size.
            let mut map = Image::new(img.w, img.h, 3);
            for k in 0..n {
                let (e, ey) = (img.data[k] + speed as f32, img.data[n + k]);
                map.data[k] = 0.5 + 0.25 * e;
                map.data[n + k] = 0.5 + 0.25 * ey;
                map.data[2 * n + k] = (e * e + ey * ey).sqrt();
            }
            save(&format!("{}_{i}_ferr.png", a[2]), &map)?;
            // What the merge sees: the clean previous picture moved by the estimate,
            // against the clean current one (mean |diff|, share over 4/255).
            if !clean.is_empty() {
                let cp = Tensor::from_image(&compute, &clean[i - 1])?;
                let wd = ops.tensor(3, h0, w0)?;
                tm.warp_into(&mut rec, &cp, &f, &wd)?;
                rec.submit_wait()?;
                rec = Recorder::new(&compute)?;
                let wi = wd.to_image(&compute)?;
                let cc = &clean[i];
                let m = w0 * h0;
                let (mut s, mut cnt, mut big) = (0f64, 0usize, 0usize);
                for y in 8..h0 - 8 {
                    for x in 8 + speed * i..w0 - 8 {
                        let k = y * w0 + x;
                        let d = (0..3).map(|c| (wi.data[c * m + k] - cc.data[c * m + k]).abs()).sum::<f32>() / 3.0;
                        if !d.is_finite() {
                            continue;
                        }
                        s += d as f64;
                        cnt += 1;
                        if d > 4.0 / 255.0 {
                            big += 1;
                        }
                    }
                }
                println!("  warp residual (clean): mean {:.5}, over 4/255: {:.2}%", s / cnt.max(1) as f64, 100.0 * big as f64 / cnt.max(1) as f64);
                // Where the clean picture has detail (as below): what the eye follows.
                let lum = |x: usize, y: usize| { let j = y * w0 + x; 0.3 * cc.data[j] + 0.59 * cc.data[m + j] + 0.11 * cc.data[2 * m + j] };
                let (mut s, mut cnt, mut big) = (0f64, 0usize, 0usize);
                for y in 8..h0 - 8 {
                    for x in 8 + speed * i..w0 - 8 {
                        let g = ((lum(x + 1, y) - lum(x - 1, y)).powi(2) + (lum(x, y + 1) - lum(x, y - 1)).powi(2)).sqrt() * 0.5;
                        let k = y * w0 + x;
                        let d = (0..3).map(|c| (wi.data[c * m + k] - cc.data[c * m + k]).abs()).sum::<f32>() / 3.0;
                        if g < 0.03 || !d.is_finite() {
                            continue;
                        }
                        s += d as f64;
                        cnt += 1;
                        if d > 8.0 / 255.0 {
                            big += 1;
                        }
                    }
                }
                println!("  warp residual where detail: mean {:.5}, over 8/255: {:.2}%", s / cnt.max(1) as f64, 100.0 * big as f64 / cnt.max(1) as f64);
                let mut rm = wi.clone();
                for k in 0..3 * m {
                    let d = (wi.data[k] - cc.data[k]).abs();
                    rm.data[k] = if d.is_finite() { d * 20.0 } else { 1.0 };
                }
                save(&format!("{}_{i}_wres.png", a[2]), &rm)?;
            }
            // Only where the clean picture has detail (luma gradient over 0.03).
            if !clean.is_empty() && img.w == w0 {
                let cl = &clean[i];
                let lum = |x: usize, y: usize| { let j = y * w0 + x; 0.3 * cl.data[j] + 0.59 * cl.data[n + j] + 0.11 * cl.data[2 * n + j] };
                let (mut s2, mut cnt, mut big) = (0f64, 0usize, 0usize);
                for y in 1..h0 - 1 {
                    for x in 1..w0 - 1 {
                        let g = ((lum(x + 1, y) - lum(x - 1, y)).powi(2) + (lum(x, y + 1) - lum(x, y - 1)).powi(2)).sqrt() * 0.5;
                        if g < 0.03 {
                            continue;
                        }
                        let k = y * w0 + x;
                        let (e, ey) = ((img.data[k] + speed as f32) as f64, img.data[n + k] as f64);
                        s2 += e * e + ey * ey;
                        cnt += 1;
                        if (e * e + ey * ey).sqrt() > 0.25 {
                            big += 1;
                        }
                    }
                }
                println!("  motion error where detail ({:.1}% of pixels): rms {:.4} px, over 0.25 px: {:.1}%", 100.0 * cnt as f64 / n as f64, (s2 / cnt.max(1) as f64).sqrt(), 100.0 * big as f64 / cnt.max(1) as f64);
            }
            // BRP_FLOW_ONLY=1: just these measurements, on one frame.
            if std::env::var("BRP_FLOW_ONLY").is_ok() {
                return Ok(());
            }
        }
        // BRP_EXACT_FLOW=1 (with BRP_PAN): the known motion in place of the estimate.
        let exact_flow = |rec: &mut Recorder, dx: f32| -> anyhow::Result<Tensor> {
            let f = ops.filled(rec, 2, h0, w0, dx)?;
            let z = ops.filled(rec, 1, h0, w0, 0.0)?;
            ops.copy(rec, &z, (0, 0, 0), (1, h0, w0), &f, (1, 0, 0), false, false)?;
            Ok(f)
        };
        let (back, fwd) = if pan && std::env::var("BRP_EXACT_FLOW").is_ok() {
            ((0..radius).map(|_| exact_flow(&mut rec, -(speed as f32) + envf("BRP_EXACT_OFF").unwrap_or(0.0))).collect::<anyhow::Result<Vec<_>>>()?, (0..radius).map(|_| exact_flow(&mut rec, speed as f32 - envf("BRP_EXACT_OFF").unwrap_or(0.0))).collect::<anyhow::Result<Vec<_>>>()?)
        } else {
            (back, fwd)
        };
        let sb: Vec<(&Tensor, &Tensor)> = (1..=radius).map(|d| (&t[i - d], &back[d - 1])).collect();
        let sf: Vec<(&Tensor, &Tensor)> = (1..=radius).map(|d| (&t[i + d], &fwd[d - 1])).collect();
        let fused = match &dn { Some((d, s)) => d.record(&mut rec, &tm, &t[i], [&sb, &sf], s)?, None => tm.window(&mut rec, &t[i], [&sb, &sf], strength, decay, smooth)? };
        // BRP_TAA=k: in place of the merge, the user's earlier TAA denoiser (history of
        // the k pictures before, moved along the motion, clamped to the blended 3x3
        // min/max in YCoCg, mixed at a per-pixel rate 0.05-0.5 that rises with clamping).
        let fused = match taa_out { Some(x) => { ops.recycle(&mut rec, fused); x }, None => fused };
        let raw = up.record(&mut rec, &t[i])?;
        let win = up.record(&mut rec, &fused)?;
        let win_t = win.clone();
        let (raw_s, win_s) = (ops.resize(&mut rec, &raw, w0, h0)?, ops.resize(&mut rec, &win, w0, h0)?);
        let took = rec.submit_wait()?;
        let (raw, win, fused) = (raw.to_image(&compute)?, win.to_image(&compute)?, fused.to_image(&compute)?);
        println!("frame {i}: window + 2 upscales {took:?}");
        if line {
            // Ghosting: error against the clean frame beside the line's path (not on
            // the line itself), and on the line (is it kept?).
            let cur = &frames[i];
            let lx = LINE_X0 + LINE_SPEED * i;
            let (w, h, n) = (cur.w, cur.h, cur.w * cur.h);
            let err = |img: &Image, on: bool| {
                let (mut s, mut cnt) = (0f64, 0f64);
                for c in 0..3 {
                    for y in 8..h - 8 {
                        for x in lx.saturating_sub(12)..(lx + 14).min(w) {
                            if (x == lx || x == lx + 1) != on {
                                continue;
                            }
                            s += (img.data[c * n + y * w + x] - clean[i].data[c * n + y * w + x]).abs() as f64;
                            cnt += 1.0;
                        }
                    }
                }
                s / cnt
            };
            println!("  line: beside it (ghosts) input {:.4} merged {:.4}; on it input {:.4} merged {:.4}", err(cur, false), err(&fused, false), err(cur, true), err(&fused, true));
        }
        // BRP_STEADY=x0,y0,x1,y1 (real footage, no reference): in that rectangle, the
        // change from the previous picture moved along the motion (lower: steadier),
        // and the fine noise (mean |pixel - mean of its 4 neighbours|), input and output.
        if let (Some(rect), Some((pi, pf))) = (std::env::var("BRP_STEADY").ok(), &prev_in) {
            let r: Vec<usize> = rect.split(',').filter_map(|v| v.parse().ok()).collect();
            let (x0, y0, x1, y1) = (r[0].max(1), r[1].max(1), r[2].min(w0 - 1), r[3].min(h0 - 1));
            let mut r2 = Recorder::new(&compute)?;
            let f = tm.motion(&mut r2, &t[i - 1], &t[i])?;
            let (ta, tb) = (Tensor::from_image(&compute, pi)?, Tensor::from_image(&compute, pf)?);
            let (wa, wb) = (ops.tensor(3, h0, w0)?, ops.tensor(3, h0, w0)?);
            tm.warp_into(&mut r2, &ta, &f, &wa)?;
            tm.warp_into(&mut r2, &tb, &f, &wb)?;
            r2.submit_wait()?;
            let (wa, wb) = (wa.to_image(&compute)?, wb.to_image(&compute)?);
            let n = w0 * h0;
            let change = |cur: &Image, prev: &Image| {
                let (mut s, mut cnt) = (0f64, 0f64);
                for c in 0..3 {
                    for y in y0..y1 {
                        for x in x0..x1 {
                            let k = c * n + y * w0 + x;
                            if prev.data[k].is_finite() {
                                s += (cur.data[k] - prev.data[k]).abs() as f64;
                                cnt += 1.0;
                            }
                        }
                    }
                }
                s / cnt
            };
            let fine = |img: &Image| {
                let (mut s, mut cnt) = (0f64, 0f64);
                for c in 0..3 {
                    for y in y0..y1 {
                        for x in x0..x1 {
                            let k = c * n + y * w0 + x;
                            let d = img.data[k] - 0.25 * (img.data[k - 1] + img.data[k + 1] + img.data[k - w0] + img.data[k + w0]);
                            s += d.abs() as f64;
                            cnt += 1.0;
                        }
                    }
                }
                s / cnt
            };
            println!("  steady: change input {:.5} output {:.5}; fine noise input {:.5} output {:.5}", change(&frames[i], &wa), change(&fused, &wb), fine(&frames[i]), fine(&fused));
            // How much the merge used: noise left as a share of one picture's (1 = only
            // this picture), as pictures' worth (1 / that); map saved as _fac.
            let mut r3 = Recorder::new(&compute)?;
            let fac = ops.tensor(1, h0, w0)?;
            let m = tm.window_with_noise(&mut r3, &t[i], [&sb, &sf], strength, decay, &fac)?;
            r3.submit_wait()?;
            let _ = m;
            let fi = fac.to_image(&compute)?;
            let (mut s, mut cnt) = (0f64, 0f64);
            for y in y0..y1 {
                for x in x0..x1 {
                    s += fi.data[y * w0 + x] as f64;
                    cnt += 1.0;
                }
            }
            println!("  merged: pictures' worth {:.2}", cnt / s);
            let mut fm = Image::new(w0, h0, 3);
            for c in 0..3 {
                fm.data[c * n..(c + 1) * n].copy_from_slice(&fi.data[..n]);
            }
            save(&format!("{}_{i}_fac.png", a[2]), &fm)?;
        }
        if let Some((pr, pw)) = &prev {
            if !clean.is_empty() {
                println!("  flicker (exact): per frame {:.4}, window {:.4}", exact(&raw, pr), exact(&win, pw));
                if let Some((pi, pf)) = &prev_in {
                    println!("  input flicker (exact): source {:.4}, merged {:.4}", exact(&frames[i], pi), exact(&fused, pf));
                }
                println!("  grain left (vs clean, before upscaling): input {:.4}, merged {:.4}", mean_abs(&frames[i], &clean[i]), mean_abs(&fused, &clean[i]));
                if let Some(rp) = &prev_res {
                    println!("  grain left moving with the picture (0 = with it, 1 = fresh each frame): {:.3}", coherence(rp, &fused, i)?);
                }
            }
            let src = &frames[i];
            println!("  fidelity (vs source): per frame {:.4}, window {:.4}", mean_abs(&raw_s.to_image(&compute)?, src), mean_abs(&win_s.to_image(&compute)?, src));
        }
        save(&format!("{}_{i}_raw.png", a[2]), &raw)?;
        save(&format!("{}_{i}_win.png", a[2]), &win)?;
        save(&format!("{}_{i}_in.png", a[2]), &fused)?;
        prev = Some((raw, win));
        if !clean.is_empty() {
            let mut r = fused.clone();
            r.data.iter_mut().zip(&clean[i].data).for_each(|(a, b)| *a -= b);
            prev_res = Some(r);
        }
        prev_in = Some((frames[i].clone(), fused));
        wins[i] = Some(win_t);
    }
    // BRP_POST=r: then merge the upscaled pictures too, r each way, at the output
    // size (the input motion, scaled up by the warp).
    let post: usize = std::env::var("BRP_POST").ok().and_then(|v| v.parse().ok()).unwrap_or(0);
    if post > 0 {
        let mut prev: Option<(Image, Image)> = None;
        let grad = |img: &Image| {
            let (w, h, n) = (img.w, img.h, img.w * img.h);
            let mut s = 0f64;
            for c in 0..3 {
                for y in 0..h - 1 {
                    for x in 0..w - 1 {
                        let v = img.data[c * n + y * w + x];
                        s += ((img.data[c * n + y * w + x + 1] - v).abs() + (img.data[c * n + (y + 1) * w + x] - v).abs()) as f64;
                    }
                }
            }
            s / (3 * n) as f64
        };
        for i in radius + post..frames.len().saturating_sub(radius + post) {
            let mut rec = Recorder::new(&compute)?;
            let back: Vec<Tensor> = (1..=post).map(|d| tm.motion(&mut rec, &t[i - d], &t[i - d + 1])).collect::<anyhow::Result<_>>()?;
            let fwd: Vec<Tensor> = (1..=post).map(|d| tm.motion(&mut rec, &t[i + d], &t[i + d - 1])).collect::<anyhow::Result<_>>()?;
            let sb: Vec<(&Tensor, &Tensor)> = (1..=post).map(|d| (wins[i - d].as_ref().unwrap(), &back[d - 1])).collect();
            let sf: Vec<(&Tensor, &Tensor)> = (1..=post).map(|d| (wins[i + d].as_ref().unwrap(), &fwd[d - 1])).collect();
            let cur = wins[i].as_ref().unwrap();
            let out = tm.window(&mut rec, cur, [&sb, &sf], 1.0, decay, 0.0)?;
            let small = ops.resize(&mut rec, &out, w0, h0)?;
            let took = rec.submit_wait()?;
            let (out, cur) = (out.to_image(&compute)?, cur.to_image(&compute)?);
            println!("post frame {i}: {took:?}; detail (mean gradient) before {:.4} after {:.4}", grad(&cur), grad(&out));
            if let Some((pc, po)) = &prev {
                if !clean.is_empty() {
                    println!("  post flicker (exact): window only {:.4}, + after upscaling {:.4}", exact(&cur, pc), exact(&out, po));
                }
                println!("  post fidelity (vs source): {:.4}", mean_abs(&small.to_image(&compute)?, &frames[i]));
            }
            save(&format!("{}_{i}_post.png", a[2]), &out)?;
            prev = Some((cur, out));
        }
    }
    Ok(())
}

fn load(path: &str) -> anyhow::Result<Image> {
    let dec = png::Decoder::new(std::io::BufReader::new(std::fs::File::open(path)?));
    let mut r = dec.read_info()?;
    let mut buf = vec![0; r.output_buffer_size().unwrap()];
    let info = r.next_frame(&mut buf)?;
    let ch = info.color_type.samples();
    anyhow::ensure!(info.bit_depth == png::BitDepth::Eight && ch >= 3, "need 8-bit RGB(A) PNG");
    let (w, h) = (info.width as usize, info.height as usize);
    let mut img = Image::new(w, h, 3);
    for i in 0..w * h {
        for c in 0..3 {
            img.data[c * w * h + i] = buf[i * ch + c] as f32 / 255.0;
        }
    }
    Ok(img)
}

fn save(path: &str, img: &Image) -> anyhow::Result<()> {
    let mut enc = png::Encoder::new(std::io::BufWriter::new(std::fs::File::create(path)?), img.w as u32, img.h as u32);
    enc.set_color(png::ColorType::Rgb);
    enc.set_depth(png::BitDepth::Eight);
    let n = img.w * img.h;
    let bytes: Vec<u8> = (0..n).flat_map(|i| (0..3).map(move |c| (c, i))).map(|(c, i)| (img.data[c.min(img.channels - 1) * n + i].clamp(0.0, 1.0) * 255.0).round() as u8).collect();
    enc.write_header()?.write_image_data(&bytes)?;
    // The exact values beside it (comparing with references).
    let raw: Vec<u8> = img.data.iter().flat_map(|v| v.to_le_bytes()).collect();
    std::fs::write(format!("{path}.f32"), raw)?;
    Ok(())
}

fn vk_op(a: &[String]) -> anyhow::Result<()> {
    use brp_ai::vk::{Net as VkNet, Tensor};
    use brp_render::compute::{Compute, Recorder};
    use brp_render::{GpuContext, GpuInstance};
    let (c, h, w): (usize, usize, usize) = (a[1].parse()?, a[2].parse()?, a[3].parse()?);
    let param = format!("7767517\n3 3\nInput a 0 1 a\nInput b 0 1 b\n{}\n", a[0]);
    let ctx = std::sync::Arc::new(GpuContext::headless(GpuInstance::new()?, None)?);
    let compute = Compute::new(ctx);
    let net = VkNet::load(&compute, &param, &[])?;
    let mut img = Image::new(w, h, c);
    img.data.iter_mut().enumerate().for_each(|(i, v)| *v = (i % 7) as f32);
    let ta = Tensor::from_image(&compute, &img)?;
    let tb = Tensor::from_image(&compute, &img)?;
    let mut rec = Recorder::new(&compute)?;
    let out = net.record(&mut rec, &[("a", &ta), ("b", &tb)], &["out"])?;
    let t = rec.submit_wait()?;
    let o = out[0].to_image(&compute)?;
    let bad = o.data.iter().zip(&img.data).filter(|(x, y)| (**x - 2.0 * **y).abs() > 1e-4).count();
    println!("{}x{}x{} ({} MB): {t:?}, {bad} wrong", c, h, w, c * h * w * 4 >> 20);
    Ok(())
}

/// GPU temporal ops against the CPU reference on a sequence made from one picture:
/// panning 3 px right per frame with fresh grain.
fn gtemp_check(png: &str) -> anyhow::Result<()> {
    use brp_ai::gpu::Ops;
    use brp_ai::gpu_temporal::Temporal;
    use brp_ai::temporal::{self, Flow};
    use brp_ai::vk::Tensor;
    use brp_render::compute::{Compute, Recorder};
    use brp_render::{GpuContext, GpuInstance};
    let base = load(png)?;
    let (w, h, n) = (base.w, base.h, base.w * base.h);
    let mut seed = 777u64;
    let mut rnd = || {
        seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        ((seed >> 33) as f32 / (1u64 << 31) as f32) - 0.5
    };
    let shifted = |s: usize| -> Image {
        let mut f = base.clone();
        for c in 0..3 {
            for y in 0..h {
                for x in 0..w {
                    f.data[c * n + y * w + x] = base.data[c * n + y * w + x.saturating_sub(s)];
                }
            }
        }
        f
    };
    let frames: Vec<Image> = (0..5)
        .map(|t| {
            let mut f = shifted(3 * t);
            f.data.iter_mut().for_each(|v| *v += 0.03 * rnd());
            f
        })
        .collect();
    let ctx = std::sync::Arc::new(GpuContext::headless(GpuInstance::new()?, std::env::var("BRP_GPU").ok().as_deref())?);
    let compute = Compute::new(ctx);
    let ops = Ops::new(&compute)?;
    let tm = Temporal::new(&ops)?;
    let t: Vec<Tensor> = frames.iter().map(|f| Tensor::from_image(&compute, f)).collect::<anyhow::Result<_>>()?;
    let flow_of = |t: &Tensor| -> anyhow::Result<Flow> {
        let im = t.to_image(&compute)?;
        let m = im.w * im.h;
        Ok(Flow { w: im.w, h: im.h, dx: im.data[..m].to_vec(), dy: im.data[m..2 * m].to_vec() })
    };
    let fdiff = |a: &Flow, b: &Flow| -> (f32, f32) {
        let d: Vec<f32> = a.dx.iter().zip(&b.dx).chain(a.dy.iter().zip(&b.dy)).map(|(p, q)| (p - q).abs()).collect();
        (d.iter().sum::<f32>() / d.len() as f32, d.iter().cloned().fold(0.0, f32::max))
    };
    let idiff = |a: &Image, b: &Image| -> (f32, f32) {
        let d: Vec<f32> = a.data[..3 * n].iter().zip(&b.data[..3 * n]).map(|(p, q)| (p - q).abs()).filter(|v| v.is_finite()).collect();
        (d.iter().sum::<f32>() / d.len() as f32, d.iter().cloned().fold(0.0, f32::max))
    };
    // Motion 2 -> 1: true dx = -3.
    let tc = Instant::now();
    let cm = temporal::motion(&frames[1], &frames[2]);
    let cpu_t = tc.elapsed();
    let mut times = Vec::new();
    let mut gm = None;
    for _ in 0..3 {
        let mut rec = Recorder::new(&compute)?;
        let m = tm.motion(&mut rec, &t[1], &t[2])?;
        times.push(rec.submit_wait()?);
        gm = Some(m);
    }
    let gm = flow_of(&gm.unwrap())?;
    let mean_dx = |f: &Flow| f.dx.iter().sum::<f32>() / f.dx.len() as f32;
    println!("motion {}x{}: CPU {cpu_t:?} mean dx {:.3}; GPU {times:?} mean dx {:.3}; diff {:?}", gm.w, gm.h, mean_dx(&cm), mean_dx(&gm), fdiff(&cm, &gm));

    // Window around frame 2.
    let tc = Instant::now();
    let cflows = [temporal::motion(&frames[1], &frames[2]), temporal::motion(&frames[0], &frames[1]), temporal::motion(&frames[3], &frames[2]), temporal::motion(&frames[4], &frames[3])];
    let mut parts = Vec::new();
    for (near, far, m1, m2) in [(1, 0, &cflows[0], &cflows[1]), (3, 4, &cflows[2], &cflows[3])] {
        let w1 = temporal::warp(&frames[near], m1);
        let a1 = temporal::agreement(&frames[2], &w1);
        let w2 = temporal::warp(&frames[far], &temporal::chain(m1, m2));
        let a2 = temporal::agreement(&frames[2], &w2);
        println!("  CPU agreement near {near} {:.3}, far {far} {:.3}", temporal::mean(&a1), temporal::mean(&a2));
        parts.push((w2, a2, temporal::FAR_WEIGHT));
        parts.push((w1, a1, 1.0));
    }
    let refs: Vec<(&Image, &[f32], f32)> = parts.iter().map(|(i, a, k)| (i, a.as_slice(), *k)).collect();
    let cw = temporal::fuse(&frames[2], &refs, 1.0);
    let cpu_t = tc.elapsed();
    let mut times = Vec::new();
    let mut gw = None;
    for _ in 0..3 {
        let mut rec = Recorder::new(&compute)?;
        let f = [tm.motion(&mut rec, &t[1], &t[2])?, tm.motion(&mut rec, &t[0], &t[1])?, tm.motion(&mut rec, &t[3], &t[2])?, tm.motion(&mut rec, &t[4], &t[3])?];
        let o = tm.window(&mut rec, &t[2], [&[(&t[1], &f[0]), (&t[0], &f[1])], &[(&t[3], &f[2]), (&t[4], &f[3])]], 1.0, temporal::FAR_WEIGHT, 0.0)?;
        if std::env::var_os("BRP_DBG").is_some() {
            let st = tm.neighbour_stats(&mut rec, &t[2], &t[1], &f[0])?;
            let zero = ops.filled(&mut rec, 2, t[2].h, t[2].w, 0.0)?;
            let st0 = tm.neighbour_stats(&mut rec, &t[2], &t[2], &zero)?;
            rec.submit_wait()?;
            println!("  near neighbour: agreement {:?}; itself, no motion: {:?}", st.to_image(&compute)?.data, st0.to_image(&compute)?.data);
            rec = Recorder::new(&compute)?;
        }
        times.push(rec.submit_wait()?);
        gw = Some(o);
    }
    let gw = gw.unwrap().to_image(&compute)?;
    let clean = shifted(6);
    println!("window: CPU {cpu_t:?} (with motion), GPU {times:?} (with motion); CPU-GPU diff {:?}", idiff(&cw, &gw));
    println!("  grain left (mean abs vs clean): input {:.4}, CPU {:.4}, GPU {:.4}", idiff(&frames[2], &clean).0, idiff(&cw, &clean).0, idiff(&gw, &clean).0);

    // Motion blur of frame 2.
    let tc = Instant::now();
    let back = temporal::motion(&frames[1], &frames[2]);
    let fwd = temporal::motion(&frames[3], &frames[2]);
    let cb = temporal::motion_blur(&frames[2], Some(&back), Some(&fwd), 1.0, 16);
    let cpu_t = tc.elapsed();
    let mut rec = Recorder::new(&compute)?;
    let b = tm.motion(&mut rec, &t[1], &t[2])?;
    let f = tm.motion(&mut rec, &t[3], &t[2])?;
    let gb = tm.motion_blur(&mut rec, &t[2], Some(&b), Some(&f), [None, None], 1.0, 16)?;
    let gpu_t = rec.submit_wait()?;
    let gb = gb.to_image(&compute)?;
    println!("blur: CPU {cpu_t:?}, GPU {gpu_t:?} (with motion); diff {:?}; change vs input CPU {:.4} GPU {:.4}", idiff(&cb, &gb), idiff(&cb, &frames[2]).0, idiff(&gb, &frames[2]).0);
    if let Ok(dir) = std::env::var("BRP_VK_OUT") {
        save(&format!("{dir}/gt_window_cpu.png"), &cw)?;
        save(&format!("{dir}/gt_window_gpu.png"), &gw)?;
        save(&format!("{dir}/gt_blur_gpu.png"), &gb)?;
    }
    Ok(())
}

/// Motion on synthetic waves shifted by a known amount: error per wave period.
fn lk_check() -> anyhow::Result<()> {
    let (compute, ops) = gpu()?;
    let tm = Temporal::new(&ops)?;
    let (w, h) = (256usize, 256usize);
    let shift: f32 = std::env::var("BRP_SHIFT").ok().and_then(|v| v.parse().ok()).unwrap_or(1.0);
    for period in [32.0f32, 16.0, 8.0, 5.0, 3.0] {
        let make = |dx: f32| {
            let mut img = Image::new(w, h, 3);
            for y in 0..h {
                for x in 0..w {
                    let (fx, fy) = (x as f32 - dx, y as f32);
                    let v = 0.5 + 0.2 * (fx * std::f32::consts::TAU / period).sin() * (fy * std::f32::consts::TAU / (period * 1.3)).cos();
                    for c in 0..3 {
                        img.data[c * w * h + y * w + x] = v;
                    }
                }
            }
            img
        };
        let prev = Tensor::from_image(&compute, &make(0.0))?;
        let cur = Tensor::from_image(&compute, &make(shift))?;
        let mut rec = Recorder::new(&compute)?;
        let f = tm.motion(&mut rec, &prev, &cur)?;
        rec.submit_wait()?;
        let img = f.to_image(&compute)?;
        let n = img.w * img.h;
        let (mut s, mut s2, mut cnt) = (0f64, 0f64, 0f64);
        for y in 32..img.h - 32 {
            for x in 32..img.w - 32 {
                let k = y * img.w + x;
                let (e, ey) = ((img.data[k] + shift) as f64, img.data[n + k] as f64);
                s += e;
                s2 += e * e + ey * ey;
                cnt += 1.0;
            }
        }
        println!("period {period:>4} px, shift {shift}: mean dx error {:+.4}, rms {:.4} px", s / cnt, (s2 / cnt).sqrt());
    }
    Ok(())
}

/// The user's earlier TAA denoiser (fragment_shader.frag), for comparison: history
/// from picture i - k, carried to picture i one step at a time along the motion.
fn taa(compute: &brp_render::compute::Compute, ops: &std::sync::Arc<brp_ai::gpu::Ops>, tm: &Temporal, t: &[Tensor], frames: &[Image], i: usize, k: usize) -> anyhow::Result<Tensor> {
    let (w, h) = (frames[i].w, frames[i].h);
    let n = w * h;
    let enc = |r: f32, g: f32, b: f32| { let co = r - b; let tt = b + co / 2.0; let cg = g - tt; [tt + cg / 2.0, co, cg] };
    let dec = |v: [f32; 3]| { let tt = v[0] - v[2] / 2.0; let g = v[2] + tt; let b = tt - v[1] / 2.0; [b + v[1], g, b] };
    let mut hist = frames[i - k].clone();
    let mut rate = vec![0.5f32; n];
    for j in i - k + 1..=i {
        let mut rec = Recorder::new(compute)?;
        let f = tm.motion(&mut rec, &t[j - 1], &t[j])?;
        let ht = Tensor::from_image(compute, &hist)?;
        let wd = ops.tensor(3, h, w)?;
        tm.warp_into(&mut rec, &ht, &f, &wd)?;
        // The rate travels with the history.
        let mut rimg = Image::new(w, h, 3);
        rimg.data[..n].copy_from_slice(&rate);
        let rt = Tensor::from_image(compute, &rimg)?;
        let wr = ops.tensor(3, h, w)?;
        tm.warp_into(&mut rec, &rt, &f, &wr)?;
        rec.submit_wait()?;
        let (wh, wrate) = (wd.to_image(compute)?, wr.to_image(compute)?);
        let x = &frames[j];
        let px = |xx: usize, yy: usize| { let q = yy * w + xx; enc(x.data[q], x.data[n + q], x.data[2 * n + q]) };
        let mut next = x.clone();
        for y in 0..h {
            for xx in 0..w {
                let q = y * w + xx;
                let cur = [x.data[q], x.data[n + q], x.data[2 * n + q]];
                let hv = [wh.data[q], wh.data[n + q], wh.data[2 * n + q]];
                if !hv[0].is_finite() {
                    rate[q] = 0.5;
                    continue;
                }
                let lm = wrate.data[q].clamp(0.05, 0.5);
                // Mix in squared space at the carried rate, then clamp in YCoCg.
                let m: Vec<f32> = (0..3).map(|c| (hv[c].max(0.0).powi(2) * (1.0 - lm) + cur[c].max(0.0).powi(2) * lm).sqrt()).collect();
                let a = enc(m[0], m[1], m[2]);
                let (xl, xr, yu, yd) = (xx.saturating_sub(1), (xx + 1).min(w - 1), y.saturating_sub(1), (y + 1).min(h - 1));
                let cross = [px(xx, y), px(xr, y), px(xl, y), px(xx, yd), px(xx, yu)];
                let diag = [px(xr, yd), px(xl, yd), px(xr, yu), px(xl, yu)];
                let mut out = [0f32; 3];
                let mut clamp_amt = 0f32;
                for c in 0..3 {
                    let (mut lo, mut hi) = (f32::MAX, f32::MIN);
                    cross.iter().for_each(|v| { lo = lo.min(v[c]); hi = hi.max(v[c]); });
                    let (mut dlo, mut dhi) = (lo, hi);
                    diag.iter().for_each(|v| { dlo = dlo.min(v[c]); dhi = dhi.max(v[c]); });
                    let (lo, hi) = ((lo + dlo) * 0.5, (hi + dhi) * 0.5);
                    out[c] = a[c].clamp(lo, hi);
                    clamp_amt += (out[c] - a[c]).powi(2);
                }
                let r = 1.0 / (1.0 / lm + 1.0) + clamp_amt * 4.0;
                rate[q] = r.clamp(0.05, 0.5);
                let o = dec(out);
                for c in 0..3 {
                    next.data[c * n + q] = o[c];
                }
            }
        }
        hist = next;
        if std::env::var("BRP_TAA_DBG").is_ok() { let mx = hist.data.iter().cloned().fold(f32::MIN, f32::max); let mean = hist.data.iter().sum::<f32>() / hist.data.len() as f32; let xm = frames[j].data.iter().sum::<f32>() / hist.data.len() as f32; println!("    taa step {j}: max {mx:.3} mean {mean:.4} (input mean {xm:.4})"); }
    }
    Tensor::from_image(compute, &hist)
}
