//! Background removal with RobustVideoMatting (MobileNetV3, GPL-3.0 weights from
//! PeterL1n/RobustVideoMatting), converted from the official ONNX export with pnnx.
//!
//! The network is recurrent: four hidden states carry information from frame to
//! frame, so results improve over the first frames after a jump. It runs at a low
//! internal resolution (long side about 512) and refines edges at full resolution.
//!
//! The converted graph has the sizes of its trace baked in (1080x1920 input, 0.25
//! downsampling); `graph_for` rewrites them for each input size.
//! The network runs on our own Vulkan runner: see [`crate::gpu`].

use anyhow::{bail, Result};
use std::path::{Path, PathBuf};

/// Internal (low-resolution) size: long side near 512, multiple of 16 so every
/// encoder level divides exactly.
#[doc(hidden)]
pub fn low_res(w: usize, h: usize) -> (usize, usize) {
    let s = (512.0 / w.max(h) as f64).min(1.0);
    let r = |v: usize| (((v as f64 * s) / 16.0).round() as usize).max(1) * 16;
    (r(w), r(h))
}

#[doc(hidden)]
pub fn files(models: &Path) -> (PathBuf, PathBuf) {
    let dir = models.join("rvm");
    (dir.join("rvm_mobilenetv3.param"), dir.join("rvm_mobilenetv3.bin"))
}

/// The traced graph rewritten for a `w`x`h` input.
#[doc(hidden)]
pub fn graph_for(text: &str, w: usize, h: usize) -> Result<String> {
    let (lw, lh) = low_res(w, h);
    let checks = [("1=0.25 2=0.25", 1), ("-23310=2,135,240", 1), ("3=1080 4=1920", 2)];
    for (pat, n) in checks {
        if text.matches(pat).count() != n {
            bail!("unexpected matting graph (\"{pat}\")");
        }
    }
    let text = text
        .replace("1=0.25 2=0.25", &format!("3={lh} 4={lw}"))
        .replace("-23310=2,135,240", &format!("-23310=2,{},{}", lh / 2, lw / 2))
        .replace("3=1080 4=1920", &format!("3={h} 4={w}"));
    Ok(replace_channel_means(&text))
}

/// The two "mean over the RGB channels" Reduction layers give wrong values on NVIDIA
/// GPUs (ncnn's Vulkan Reduction, full-resolution input). Same maths with simpler
/// layers: slice the three channels, add them, scale by 1/3.
fn replace_channel_means(text: &str) -> String {
    let mut out = Vec::new();
    let (mut layers, mut blobs) = (0, 0);
    for line in text.lines() {
        let f: Vec<&str> = line.split_whitespace().collect();
        if f.len() == 11 && f[0] == "Reduction" && f[2..4] == ["1", "1"] && f[6..] == ["0=3", "1=0", "-23303=1,0", "4=1", "5=1"][..] {
            let (name, src, dst) = (f[1], f[4], f[5]);
            out.push(format!("Slice {name}_s 1 3 {src} {name}_r {name}_g {name}_b -23300=3,1,1,1 1=0"));
            out.push(format!("BinaryOp {name}_rg 2 1 {name}_r {name}_g {name}_rg 0=0"));
            out.push(format!("BinaryOp {name}_rgb 2 1 {name}_rg {name}_b {name}_sum 0=0"));
            out.push(format!("BinaryOp {name} 1 1 {name}_sum {dst} 0=2 1=1 2=0.33333334"));
            layers += 3;
            blobs += 5;
            continue;
        }
        out.push(line.to_string());
    }
    if let Some(counts) = out.get_mut(1) {
        let n: Vec<usize> = counts.split_whitespace().filter_map(|v| v.parse().ok()).collect();
        if n.len() == 2 {
            *counts = format!("{} {}", n[0] + layers, n[1] + blobs);
        }
    }
    out.join("\n") + "\n"
}

