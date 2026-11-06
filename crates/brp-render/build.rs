// Compiles the renderer's GLSL shaders to SPIR-V with glslc from the Vulkan SDK.
use std::path::PathBuf;
use std::process::Command;

fn main() {
    let out = PathBuf::from(std::env::var("OUT_DIR").unwrap());
    let sdk = std::env::var("VULKAN_SDK").expect("VULKAN_SDK must point to the Vulkan SDK");
    let bin = if cfg!(windows) { "Bin/glslc.exe" } else { "bin/glslc" };
    let glslc = PathBuf::from(&sdk).join(bin);
    println!("cargo:rerun-if-env-changed=VULKAN_SDK");
    // (source, output name, extra defines)
    let shaders: &[(&str, &str, &[&str])] = &[
        ("video.vert", "video.vert", &[]),
        ("video.frag", "video.frag", &[]),
        ("nnedi3.comp", "nnedi3.comp", &[]),
        ("nnedi3.comp", "nnedi3_fast.comp", &["-DFAST"]),
        ("bwdif.comp", "bwdif.comp", &[]),
    ];
    for (file, name, defines) in shaders {
        let src = PathBuf::from("shaders").join(file);
        println!("cargo:rerun-if-changed={}", src.display());
        let dst = out.join(format!("{name}.spv"));
        let status = Command::new(&glslc)
            .args(*defines)
            .args(["-O", "--target-env=vulkan1.3", "-o"])
            .arg(&dst)
            .arg(&src)
            .status()
            .unwrap_or_else(|e| panic!("running {}: {e}", glslc.display()));
        assert!(status.success(), "glslc failed for {name}");
    }
}
