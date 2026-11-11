// Copies the FFmpeg shared libraries next to the executable so it runs without PATH changes.
use std::path::PathBuf;

fn main() {
    println!("cargo:rerun-if-env-changed=FFMPEG_DIR");
    copy_nnedi_weights();
    copy_shaders();
    copy_ai();
    let Ok(ff) = std::env::var("FFMPEG_DIR") else { return };
    let out = PathBuf::from(std::env::var("OUT_DIR").unwrap());
    // OUT_DIR = target/<profile>/build/<pkg>/out
    let Some(profile_dir) = out.ancestors().nth(3) else { return };
    let bin = PathBuf::from(ff).join(if cfg!(windows) { "bin" } else { "lib" });
    let Ok(entries) = std::fs::read_dir(&bin) else { return };
    for e in entries.flatten() {
        let p = e.path();
        let is_lib = p.extension().is_some_and(|x| x == "dll" || x == "so" || x == "dylib");
        if is_lib {
            let dst = profile_dir.join(p.file_name().unwrap());
            let same = std::fs::metadata(&dst).ok().map(|m| m.len()) == std::fs::metadata(&p).ok().map(|m| m.len());
            if !same {
                let _ = std::fs::copy(&p, &dst);
            }
        }
    }
}

/// NNEDI3 weights live next to the executable.
fn copy_nnedi_weights() {
    let src = PathBuf::from(std::env::var("CARGO_MANIFEST_DIR").unwrap()).join("../brp-render/assets/nnedi3_weights.bin");
    println!("cargo:rerun-if-changed={}", src.display());
    let out = PathBuf::from(std::env::var("OUT_DIR").unwrap());
    if let Some(profile_dir) = out.ancestors().nth(3) {
        let dst = profile_dir.join("nnedi3_weights.bin");
        if std::fs::metadata(&dst).map(|m| m.len()).ok() != std::fs::metadata(&src).map(|m| m.len()).ok() {
            let _ = std::fs::copy(&src, &dst);
        }
    }
}

/// Bundled effects (repo `shaders/`) go to `<exe dir>/shaders`.
fn copy_shaders() {
    let src = PathBuf::from(std::env::var("CARGO_MANIFEST_DIR").unwrap()).join("../../shaders");
    println!("cargo:rerun-if-changed={}", src.display());
    let out = PathBuf::from(std::env::var("OUT_DIR").unwrap());
    let Some(profile_dir) = out.ancestors().nth(3) else { return };
    let dst = profile_dir.join("shaders");
    let _ = std::fs::create_dir_all(&dst);
    let mut shipped = std::collections::HashSet::new();
    if let Ok(entries) = std::fs::read_dir(&src) {
        for e in entries.flatten() {
            let p = e.path();
            if p.extension().is_some_and(|x| x == "glsl") {
                let name = p.file_name().unwrap().to_owned();
                let _ = std::fs::copy(&p, dst.join(&name));
                shipped.insert(name);
            }
        }
    }
    // The exe's folder mirrors the repo: drop effects that were removed there.
    if let Ok(entries) = std::fs::read_dir(&dst) {
        for e in entries.flatten() {
            let p = e.path();
            if p.extension().is_some_and(|x| x == "glsl") && !shipped.contains(p.file_name().unwrap()) {
                let _ = std::fs::remove_file(&p);
            }
        }
    }
}

/// AI models from `third_party/models` (see tools/fetch_ai.sh), if present.
fn copy_ai() {
    let root = PathBuf::from(std::env::var("CARGO_MANIFEST_DIR").unwrap()).join("../../third_party");
    println!("cargo:rerun-if-changed={}", root.join("models").display());
    let out = PathBuf::from(std::env::var("OUT_DIR").unwrap());
    let Some(profile_dir) = out.ancestors().nth(3) else { return };
    copy_tree(&root.join("models"), &profile_dir.join("models"));
}

fn copy_if_changed(src: &std::path::Path, dst: &std::path::Path) {
    let len = |p: &std::path::Path| std::fs::metadata(p).ok().map(|m| m.len());
    if len(src) != len(dst) {
        let _ = std::fs::copy(src, dst);
    }
}

fn copy_tree(src: &std::path::Path, dst: &std::path::Path) {
    let Ok(entries) = std::fs::read_dir(src) else { return };
    let _ = std::fs::create_dir_all(dst);
    for e in entries.flatten() {
        let p = e.path();
        if p.is_dir() {
            copy_tree(&p, &dst.join(e.file_name()));
        } else {
            copy_if_changed(&p, &dst.join(e.file_name()));
        }
    }
}
