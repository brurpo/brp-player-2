//! AI models on the GPU: our own Vulkan runner for ncnn-format networks (any vendor,
//! tensor cores where the GPU has them), on the renderer's device.
//!
//! - [`vk`]: the network runner (tensors, layers, `.param`/`.bin` loading).
//! - [`gpu`]: the models as picture operations (upscale, interpolation, background
//!   removal) on GPU tensors.
//! - [`gpu_temporal`]: motion, the temporal window and motion blur, on the GPU.
//! - [`upscale`], [`rife`], [`matting`]: which model files, sizes and padding.
//! - [`temporal`]: the CPU version of `gpu_temporal`, kept as the reference its
//!   results are checked against.
//!
//! Pictures are planar RGB(A) f32 in 0..1, in the source's transfer (models are
//! trained on gamma-encoded pictures).

pub mod denoise;
pub mod depth;
pub mod flow;
pub mod gpu;
pub mod gpu_temporal;
pub mod gpuenc;
pub mod learned;
pub mod matting;
pub mod pack;
pub mod birefnet;
pub mod restore;
pub mod rife;
pub mod stabilise;
pub mod temporal;
pub mod upscale;
pub mod vk;
pub mod warm;

use anyhow::{bail, Result};

/// Planar f32 image: `channels` planes of `w * h`.
#[derive(Clone)]
pub struct Image {
    pub w: usize,
    pub h: usize,
    pub channels: usize,
    pub data: Vec<f32>,
}

impl Image {
    pub fn new(w: usize, h: usize, channels: usize) -> Self {
        Image { w, h, channels, data: vec![0.0; w * h * channels] }
    }

    pub fn plane(&self, c: usize) -> &[f32] {
        &self.data[c * self.w * self.h..(c + 1) * self.w * self.h]
    }

    pub fn plane_mut(&mut self, c: usize) -> &mut [f32] {
        let n = self.w * self.h;
        &mut self.data[c * n..(c + 1) * n]
    }
}

/// Where the model files live: `<exe dir>/models`, or `BRP_MODELS`.
pub fn models_dir() -> std::path::PathBuf {
    if let Some(d) = std::env::var_os("BRP_MODELS") {
        return d.into();
    }
    // Next to the exe, or its parent (for examples/).
    let exe_dir = std::env::current_exe().ok().and_then(|p| p.parent().map(|d| d.to_path_buf()));
    exe_dir
        .iter()
        .flat_map(|d| [Some(d.join("models")), d.parent().map(|p| p.join("models"))])
        .flatten()
        .find(|d| d.is_dir())
        .unwrap_or_else(|| "models".into())
}

/// The model files are there (else, where they were looked for).
pub fn check_models() -> Result<()> {
    let dir = models_dir();
    if !dir.join("realesrgan").is_dir() {
        bail!("no models in {}", dir.display());
    }
    Ok(())
}
