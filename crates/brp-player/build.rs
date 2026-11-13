// Copies the FFmpeg shared libraries next to the executable so it runs without PATH changes.
use std::path::PathBuf;

fn main() {
    println!("cargo:rerun-if-env-changed=FFMPEG_DIR");
    copy_nnedi_weights();
    copy_shaders();
    copy_ai();
    copy_ocio();
    embed_icon();
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

/// OpenColorIO (DESIGN.md §7d): its library next to the executable, and the Maya and
/// Blender configs (`third_party/ocio-configs`, see tools/fetch_ocio_configs.ps1) in
/// `<exe dir>/ocio`.
fn copy_ocio() {
    println!("cargo:rerun-if-env-changed=OCIO_DIR");
    let out = PathBuf::from(std::env::var("OUT_DIR").unwrap());
    let Some(profile_dir) = out.ancestors().nth(3) else { return };
    if let Ok(dir) = std::env::var("OCIO_DIR") {
        let bin = PathBuf::from(dir).join("bin");
        if let Ok(entries) = std::fs::read_dir(&bin) {
            for e in entries.flatten() {
                let p = e.path();
                if p.extension().is_some_and(|x| x == "dll") {
                    copy_if_changed(&p, &profile_dir.join(p.file_name().unwrap()));
                }
            }
        }
    }
    let configs = PathBuf::from(std::env::var("CARGO_MANIFEST_DIR").unwrap()).join("../../third_party/ocio-configs");
    println!("cargo:rerun-if-changed={}", configs.display());
    copy_tree(&configs, &profile_dir.join("ocio"));
}

/// The executable's own icon (Explorer, shortcuts, a taskbar button pinned or
/// grouped): the app icon (`assets/icons/app_icon.png`, its faint tile keyed out as
/// for the window icon) at the sizes Windows asks for, as an .ico in a resource
/// linked into the executable. Without it Windows shows its generic program icon.
/// Needs a resource compiler: llvm-rc (next to LIBCLANG_PATH's libclang, or in
/// LLVM's default folder) or the Windows SDK's rc.exe; without one, a warning.
fn embed_icon() {
    if std::env::var("CARGO_CFG_TARGET_OS").as_deref() != Ok("windows") {
        return;
    }
    let src = PathBuf::from(std::env::var("CARGO_MANIFEST_DIR").unwrap()).join("assets/icons/app_icon.png");
    println!("cargo:rerun-if-changed={}", src.display());
    println!("cargo:rerun-if-env-changed=LIBCLANG_PATH");
    let out = PathBuf::from(std::env::var("OUT_DIR").unwrap());
    let ico = match make_ico(&src) {
        Ok(b) => b,
        Err(e) => {
            println!("cargo:warning=executable icon: {e}");
            return;
        }
    };
    std::fs::write(out.join("app.ico"), ico).unwrap();
    // Icon id 1: the lowest id is the one Windows shows for the file.
    std::fs::write(out.join("app.rc"), "1 ICON \"app.ico\"\n").unwrap();
    let mut rcs: Vec<PathBuf> = Vec::new();
    if let Ok(d) = std::env::var("LIBCLANG_PATH") {
        rcs.push(PathBuf::from(d).join("llvm-rc.exe"));
    }
    rcs.push(PathBuf::from(r"C:\Program Files\LLVM\bin\llvm-rc.exe"));
    // The Windows SDK's newest x64 rc.exe.
    let kits = PathBuf::from(r"C:\Program Files (x86)\Windows Kits\10\bin");
    let mut sdk: Vec<PathBuf> = std::fs::read_dir(&kits).into_iter().flatten().flatten().map(|e| e.path().join("x64").join("rc.exe")).filter(|p| p.is_file()).collect();
    sdk.sort();
    rcs.extend(sdk.into_iter().rev());
    let Some(rc) = rcs.into_iter().find(|p| p.is_file()) else {
        println!("cargo:warning=executable icon: no resource compiler (llvm-rc or the Windows SDK's rc.exe): the executable keeps Windows' generic icon");
        return;
    };
    let res = out.join("app.res");
    let ok = std::process::Command::new(&rc).current_dir(&out).args(["/nologo", "/fo"]).arg(&res).arg("app.rc").status().is_ok_and(|s| s.success());
    if !ok {
        println!("cargo:warning=executable icon: {} failed", rc.display());
        return;
    }
    // MSVC's linker takes a compiled resource file like an object file.
    println!("cargo:rustc-link-arg-bins={}", res.display());
}

/// An .ico (PNG-compressed entries) of the app icon at 16-256 px.
fn make_ico(src: &std::path::Path) -> Result<Vec<u8>, String> {
    let bytes = std::fs::read(src).map_err(|e| format!("{}: {e}", src.display()))?;
    let mut dec = png::Decoder::new(std::io::Cursor::new(bytes));
    dec.set_transformations(png::Transformations::EXPAND | png::Transformations::STRIP_16);
    let mut reader = dec.read_info().map_err(|e| e.to_string())?;
    let mut buf = vec![0; reader.output_buffer_size().unwrap_or(0)];
    let info = reader.next_frame(&mut buf).map_err(|e| e.to_string())?;
    let (w, h) = (info.width as usize, info.height as usize);
    let mut rgba: Vec<u8> = match info.color_type {
        png::ColorType::Rgba => buf[..w * h * 4].to_vec(),
        png::ColorType::Rgb => buf[..w * h * 3].chunks(3).flat_map(|p| [p[0], p[1], p[2], 255]).collect(),
        png::ColorType::GrayscaleAlpha => buf[..w * h * 2].chunks(2).flat_map(|p| [p[0], p[0], p[0], p[1]]).collect(),
        png::ColorType::Grayscale => buf[..w * h].iter().flat_map(|&g| [g, g, g, 255]).collect(),
        c => return Err(format!("unexpected PNG colour type {c:?}")),
    };
    // As icons.rs `key_out_tile`: the faint tile behind the icon made transparent.
    for p in rgba.chunks_mut(4) {
        let a = p[3] as u32;
        p[3] = if a <= 12 { 0 } else { ((a - 12) * 255 / (255 - 12)) as u8 };
    }
    let mut entries = Vec::new();
    for size in [16usize, 20, 24, 32, 40, 48, 64, 128, 256] {
        let small = downscale(w, h, &rgba, size);
        let mut png_bytes = Vec::new();
        {
            let mut enc = png::Encoder::new(&mut png_bytes, size as u32, size as u32);
            enc.set_color(png::ColorType::Rgba);
            enc.set_depth(png::BitDepth::Eight);
            let mut wr = enc.write_header().map_err(|e| e.to_string())?;
            wr.write_image_data(&small).map_err(|e| e.to_string())?;
        }
        entries.push((size, png_bytes));
    }
    // ICONDIR, then an ICONDIRENTRY per image, then the images.
    let mut ico = Vec::new();
    ico.extend_from_slice(&[0, 0, 1, 0]);
    ico.extend_from_slice(&(entries.len() as u16).to_le_bytes());
    let mut offset = 6 + 16 * entries.len();
    for (size, data) in &entries {
        let d = if *size >= 256 { 0 } else { *size as u8 };
        ico.extend_from_slice(&[d, d, 0, 0]);
        ico.extend_from_slice(&1u16.to_le_bytes());
        ico.extend_from_slice(&32u16.to_le_bytes());
        ico.extend_from_slice(&(data.len() as u32).to_le_bytes());
        ico.extend_from_slice(&(offset as u32).to_le_bytes());
        offset += data.len();
    }
    for (_, data) in &entries {
        ico.extend_from_slice(data);
    }
    Ok(ico)
}

/// Area-average downscale of a square RGBA picture to `size` x `size` (as icons.rs,
/// premultiplied so transparent pixels don't darken edges).
fn downscale(w: usize, h: usize, rgba: &[u8], size: usize) -> Vec<u8> {
    let mut out = vec![0u8; size * size * 4];
    for y in 0..size {
        for x in 0..size {
            let (x0, x1) = (x * w / size, ((x + 1) * w / size).max(x * w / size + 1));
            let (y0, y1) = (y * h / size, ((y + 1) * h / size).max(y * h / size + 1));
            let mut acc = [0u64; 4];
            for yy in y0..y1 {
                for xx in x0..x1 {
                    let p = &rgba[(yy * w + xx) * 4..][..4];
                    let a = p[3] as u64;
                    acc[0] += p[0] as u64 * a;
                    acc[1] += p[1] as u64 * a;
                    acc[2] += p[2] as u64 * a;
                    acc[3] += a;
                }
            }
            let n = ((x1 - x0) * (y1 - y0)) as u64;
            let o = &mut out[(y * size + x) * 4..][..4];
            if acc[3] > 0 {
                o[0] = (acc[0] / acc[3]) as u8;
                o[1] = (acc[1] / acc[3]) as u8;
                o[2] = (acc[2] / acc[3]) as u8;
            }
            o[3] = (acc[3] / n) as u8;
        }
    }
    out
}
