//! Parses every user shader in a folder and compiles every pass.
//!   cargo run -p brp-render --release --example shadercheck -- old/shaders [--show NAME#PASS]

use brp_render::usershader::{self, Transfer};

fn main() -> anyhow::Result<()> {
    let dir = std::env::args().nth(1).unwrap_or_else(|| "old/shaders".into());
    let show = std::env::args().skip_while(|a| a != "--show").nth(1);
    let mut files: Vec<_> = std::fs::read_dir(&dir)?.flatten().map(|e| e.path()).filter(|p| p.extension().is_some_and(|x| x == "glsl")).collect();
    files.sort();
    let (mut ok, mut bad, mut passes_total) = (0, 0, 0);
    for path in files {
        let name = path.file_name().unwrap().to_string_lossy().to_string();
        let text = std::fs::read_to_string(&path)?;
        let file = match usershader::parse::parse(&name, &text) {
            Ok(f) => f,
            Err(e) => {
                println!("PARSE FAIL {name}: {e:#}");
                bad += 1;
                continue;
            }
        };
        let params: Vec<f64> = file.params.iter().map(|p| p.default).collect();
        let t = std::time::Instant::now();
        let mut errors = Vec::new();
        let mut modes = std::collections::BTreeSet::new();
        for (i, pass) in file.passes.iter().enumerate() {
            passes_total += 1;
            let binds = usershader::bind_kinds(&file, pass);
            match usershader::build_pass(&file, i, &pass.hooks[0], &binds, &params, Transfer::Sdr) {
                Ok((g, _, mode)) => {
                    modes.insert(format!("{mode:?}"));
                    if show.as_deref() == Some(&format!("{name}#{i}")) {
                        println!("{}", g.glsl);
                    }
                }
                Err(e) => errors.push(format!("pass {i}: {}", e.to_string().lines().take(4).collect::<Vec<_>>().join(" | "))),
            }
        }
        let status = if errors.is_empty() { ok += 1; "ok" } else { bad += 1; "FAIL" };
        println!(
            "{status:4} {name:<36} {:>3} passes, {:>2} params, {} storage tex, params as {:?}, {:?}",
            file.passes.len(),
            file.params.len(),
            file.textures.iter().filter(|t| t.storage).count(),
            modes,
            t.elapsed()
        );
        for e in errors.iter().take(3) {
            println!("       {e}");
        }
    }
    println!("{ok} files ok, {bad} failed, {passes_total} passes");
    Ok(())
}
