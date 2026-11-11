//! Shaders built at start-up ("Compiling shaders…" with a progress bar), so adding an
//! effect does not wait for them:
//! - the effect shader files (.glsl in the shader folders), every run: unchanged ones
//!   are found in the disk cache at once, new or edited ones are compiled;
//! - the AI kernels, when this build of the app has not built them yet on this GPU
//!   and driver (see `brp_render::shader_cache`).
//!
//! The progress shows only when there is real work (after a moment), not for a run
//! that finds everything in the cache.

use brp_render::{GpuContext, Transfer};
use std::sync::{Arc, Mutex, Once};
use std::time::{Duration, Instant};

/// (done, total, started) while the shaders are being built.
static PROGRESS: Mutex<Option<(usize, usize, Instant)>> = Mutex::new(None);

/// Colour transfers an effect is compiled for (its code depends on the source's).
const TRANSFERS: [Transfer; 4] = [Transfer::Sdr, Transfer::Pq, Transfer::Hlg, Transfer::Linear];

fn set(done: usize, total: usize) {
    let mut p = PROGRESS.lock().unwrap();
    let started = p.map_or_else(Instant::now, |(_, _, s)| s);
    *p = Some((done, total, started));
}

/// Start building the shaders in the background, once per run.
pub fn start(ctx: &Arc<GpuContext>) {
    static STARTED: Once = Once::new();
    STARTED.call_once(|| {
        if brp_render::shader_cache::dir().is_none() {
            return;
        }
        let props = ctx.properties();
        let ai = brp_render::shader_cache::needs_warm_up(&props);
        let ctx = ctx.clone();
        let spawned = std::thread::Builder::new().name("brp-shaders".into()).spawn(move || {
            // After the first pictures are up: start-up stays as fast as before.
            std::thread::sleep(Duration::from_millis(1000));
            let t = Instant::now();
            let files = crate::effects::library();
            let fx_total = files.len() * TRANSFERS.len();
            set(0, fx_total);
            // (file, transfer) pairs on several threads (each has its own compiler),
            // leaving two cores to playback.
            let parsed: Vec<_> = files
                .iter()
                .filter_map(|path| {
                    let text = std::fs::read_to_string(path).ok()?;
                    let name = path.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default();
                    brp_render::usershader::parse::parse(&name, &text).map_err(|e| log::debug!("shader {name}: {e:#}")).ok()
                })
                .collect();
            let jobs: Vec<_> = parsed.iter().flat_map(|f| TRANSFERS.map(|t| (f, t))).collect();
            let next = std::sync::atomic::AtomicUsize::new(0);
            let done = std::sync::atomic::AtomicUsize::new(fx_total - jobs.len());
            let threads = std::thread::available_parallelism().map_or(2, |n| n.get().saturating_sub(2).max(1));
            std::thread::scope(|s| {
                for _ in 0..threads {
                    s.spawn(|| loop {
                        let i = next.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                        let Some((file, tr)) = jobs.get(i) else { break };
                        if let Err(e) = brp_render::effects::precompile(file, *tr) {
                            log::debug!("shader {}: {e:#}", file.name);
                        }
                        set(done.fetch_add(1, std::sync::atomic::Ordering::Relaxed) + 1, fx_total);
                    });
                }
            });
            if ai {
                let compute = brp_render::compute::Compute::new(ctx.clone());
                let r = brp_ai::warm::warm_up(&compute, &brp_ai::models_dir(), &|d, n| set(fx_total + d, fx_total + n));
                drop(compute);
                ctx.save_pipeline_cache();
                match r {
                    Ok(()) => {
                        brp_render::shader_cache::warm_up_done(&props);
                        brp_render::shader_cache::prune();
                    }
                    Err(e) => log::warn!("building shaders: {e:#}"),
                }
            }
            use std::sync::atomic::Ordering::Relaxed;
            log::info!(
                "shaders ready in {:.1} s ({} effect files{}): {} compiled, {} from the cache",
                t.elapsed().as_secs_f64(),
                files.len(),
                if ai { ", AI kernels" } else { "" },
                brp_render::shader_cache::COMPILED.load(Relaxed),
                brp_render::shader_cache::HITS.load(Relaxed)
            );
            *PROGRESS.lock().unwrap() = None;
        });
        if spawned.is_err() {
            *PROGRESS.lock().unwrap() = None;
        }
    });
}

/// (done, total) while shaders are being built, once that has taken a moment (a run
/// that finds everything in the cache shows nothing).
pub fn progress() -> Option<(usize, usize)> {
    let p = *PROGRESS.lock().unwrap();
    p.filter(|(_, _, s)| s.elapsed() > Duration::from_millis(300)).map(|(d, n, _)| (d, n))
}
