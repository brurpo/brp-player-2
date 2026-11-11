//! Compiles every effect shader (.glsl) in a folder for each colour transfer, as the
//! player does at start-up (on all cores but two), and prints the time and shader
//! cache counts.
//!   fxprecompile <folder>
use brp_render::Transfer;
use std::sync::atomic::{AtomicUsize, Ordering::Relaxed};

fn main() -> anyhow::Result<()> {
    let dir = std::env::args().nth(1).unwrap_or_else(|| "shaders".into());
    let t = std::time::Instant::now();
    let mut files = Vec::new();
    for e in std::fs::read_dir(&dir)?.flatten() {
        let p = e.path();
        if p.extension().is_some_and(|x| x.eq_ignore_ascii_case("glsl")) {
            let name = p.file_name().unwrap().to_string_lossy().into_owned();
            match brp_render::usershader::parse::parse(&name, &std::fs::read_to_string(&p)?) {
                Ok(f) => files.push(f),
                Err(e) => println!("{name}: {e:#}"),
            }
        }
    }
    let jobs: Vec<_> = files.iter().flat_map(|f| [Transfer::Sdr, Transfer::Pq, Transfer::Hlg, Transfer::Linear].map(|t| (f, t))).collect();
    let next = AtomicUsize::new(0);
    let threads = std::thread::available_parallelism().map_or(2, |n| n.get().saturating_sub(2).max(1));
    std::thread::scope(|s| {
        for _ in 0..threads {
            s.spawn(|| loop {
                let Some((f, tr)) = jobs.get(next.fetch_add(1, Relaxed)) else { break };
                if let Err(e) = brp_render::effects::precompile(f, *tr) {
                    println!("{} {tr:?}: {}", f.name, format!("{e:#}").lines().next().unwrap_or(""));
                }
            });
        }
    });
    println!(
        "{} files on {threads} threads in {:.2} s: {} compiled, {} from the cache",
        files.len(),
        t.elapsed().as_secs_f64(),
        brp_render::shader_cache::COMPILED.load(Relaxed),
        brp_render::shader_cache::HITS.load(Relaxed)
    );
    Ok(())
}
