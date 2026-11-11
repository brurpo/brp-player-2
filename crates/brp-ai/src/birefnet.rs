//! Background removal with BiRefNet (ZhengPeng7/BiRefNet, MIT), the lite matting
//! weights (Swin-T backbone): soft alpha with fine edges (hair, glass) on any subject,
//! one picture at a time (no history: a jump needs nothing replayed).
//!
//! The graph (`pth2ncnn birefnet`, at 1024 x 1024 as the model was trained) takes the
//! picture squeezed to a square, as the official inference resizes it; the alpha is
//! stretched back. Measured against a plain-PyTorch version of the official network
//! (tools/airef/birefnet_ref.py): alpha within 0.008, 0.00006 on average (f16
//! activations). About 52 ms per picture on an RTX 5070 Ti.

use crate::gpu::Ops;
use crate::vk::{Net, Tensor};
use anyhow::{Context, Result};
use brp_render::compute::Recorder;
use std::path::{Path, PathBuf};
use std::sync::Arc;

/// The network's input size.
pub const SIZE: usize = 1024;

fn base(models: &Path) -> PathBuf {
    models.join("birefnet").join("birefnet_lite_matting")
}

pub struct BiRefNet {
    ops: Arc<Ops>,
    net: Net,
}

impl BiRefNet {
    pub fn available(models: &Path) -> bool {
        base(models).with_extension("param").exists()
    }

    pub fn new(ops: &Arc<Ops>, models: &Path) -> Result<Self> {
        let base = base(models);
        let param = std::fs::read_to_string(base.with_extension("param")).with_context(|| format!("{}.param", base.display()))?;
        let mut net = Net::load(&ops.compute, &param, &std::fs::read(base.with_extension("bin"))?)?;
        net.set_half(true);
        net.share_pool(&ops.pool);
        net.set_flush_every(60);
        Ok(BiRefNet { ops: ops.clone(), net })
    }

    /// The alpha of `img` (1 channel, its size; an alpha `img` already has is not
    /// used here).
    pub fn alpha(&self, rec: &mut Recorder, img: &Tensor) -> Result<Tensor> {
        let ops = &self.ops;
        let x = ops.tensor(3, SIZE, SIZE)?;
        ops.resize_into(rec, img, 0, 3, &x, 0)?;
        let a = self.net.record(rec, &[("img", &x)], &["out"])?.remove(0);
        ops.recycle(rec, x);
        let out = ops.tensor(1, img.h, img.w)?;
        ops.resize_into(rec, &a, 0, 1, &out, 0)?;
        ops.recycle(rec, a);
        Ok(out)
    }

    /// `img`'s colour with `alpha` (4 channels).
    pub fn with_alpha(&self, rec: &mut Recorder, img: &Tensor, alpha: &Tensor) -> Result<Tensor> {
        let ops = &self.ops;
        let out = ops.tensor(4, img.h, img.w)?;
        ops.copy(rec, img, (0, 0, 0), (3, img.h, img.w), &out, (0, 0, 0), false, false)?;
        ops.copy(rec, alpha, (0, 0, 0), (1, img.h, img.w), &out, (3, 0, 0), false, false)?;
        Ok(out)
    }
}
