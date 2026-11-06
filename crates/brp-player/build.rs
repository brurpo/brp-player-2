// Copies the FFmpeg shared libraries next to the executable so it runs without PATH changes.
use std::path::PathBuf;

fn main() {
    println!("cargo:rerun-if-env-changed=FFMPEG_DIR");
    copy_nnedi_weights();
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
