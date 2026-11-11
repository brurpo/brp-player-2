//! Picture restoration networks that give back a picture of the same size:
//! - Deblur: NAFNet trained on REDS (megvii-research/NAFNet, MIT): video frames with
//!   motion blur and compression; removes motion and focus blur. Its channel attention
//!   averages over a local box (TLC), as the official test setting.
//! - Clean-up: SCUNet "real PSNR" (cszn/SCUNet, Apache-2.0): noise, blocking and
//!   compression artefacts (+0.4..0.6 dB on 0.8-1.5 Mbit/s H.264, a little softening).
//! Graphs from `pth2ncnn nafnet` / `pth2ncnn scunet`, measured against plain-PyTorch
//! versions of the official networks (tools/airef): 71 and 76 dB.

use crate::gpu::Ops;
use crate::vk::{Net, PushTensor, Tensor};
use anyhow::{Context, Result};
use brp_render::compute::{Kernel, Push, Recorder, PRELUDE};
use std::path::Path;
use std::sync::Arc;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RestoreModel {
    Deblur,
    CleanUp,
}

impl RestoreModel {
    fn file(self) -> (&'static str, &'static str) {
        match self {
            RestoreModel::Deblur => ("nafnet", "nafnet_reds_w64"),
            RestoreModel::CleanUp => ("scunet", "scunet_color_real_psnr"),
        }
    }

    /// Input sizes the network takes (multiples), and whether padding repeats the
    /// edge (else zeros), as each official forward pads.
    fn pad(self) -> (usize, bool) {
        match self {
            RestoreModel::Deblur => (16, false),
            RestoreModel::CleanUp => (64, true),
        }
    }
}

/// Where the network ran away: a pixel it moved by more than DIVERGE (or made
/// non-finite). NAFNet REDS does on dense, already sharp detail at full size (crowds,
/// foliage in aerial shots): its activations grow to 1e5-1e6 and the output to
/// +-1000, the same in the official network (tools/airef/nafnet_ref.py). On normal
/// frames 0.002% of pixels move by more than 0.35.
const DIVERGE: f32 = 0.35;
/// The window (2 R + 1)^2 around a pixel whose share of runaway pixels decides how
/// much of the original it shows; FULL of them shows it alone.
const FALLBACK_R: i32 = 8;
const FALLBACK_FULL: f32 = 0.02;

const BAD: &str = r#"
layout(local_size_x = 16, local_size_y = 8, local_size_z = 1) in;
layout(push_constant) uniform P { F32 a; F32 b; F32 o; int W, H; float t; } p;
void main() {
    int x = int(gl_GlobalInvocationID.x), y = int(gl_GlobalInvocationID.y);
    if (x >= p.W || y >= p.H) return;
    int i = y * p.W + x, n = p.W * p.H;
    bool bad = false;
    for (int c = 0; c < 3; c++) {
        float v = p.b.v[c * n + i];
        bad = bad || isnan(v) || isinf(v) || abs(v - p.a.v[c * n + i]) > p.t;
    }
    p.o.v[i] = bad ? 1.0 : 0.0;
}
"#;

const ROWSUM: &str = r#"
layout(local_size_x = 16, local_size_y = 8, local_size_z = 1) in;
layout(push_constant) uniform P { F32 a; F32 o; int W, H, R; } p;
void main() {
    int x = int(gl_GlobalInvocationID.x), y = int(gl_GlobalInvocationID.y);
    if (x >= p.W || y >= p.H) return;
    float s = 0.0;
    for (int d = -p.R; d <= p.R; d++) s += p.a.v[y * p.W + clamp(x + d, 0, p.W - 1)];
    p.o.v[y * p.W + x] = s;
}
"#;

const FALLBACK: &str = r#"
layout(local_size_x = 16, local_size_y = 8, local_size_z = 1) in;
layout(push_constant) uniform P { F32 rows; F32 a; F32 b; int W, H, R; float full; } p;
void main() {
    int x = int(gl_GlobalInvocationID.x), y = int(gl_GlobalInvocationID.y);
    if (x >= p.W || y >= p.H) return;
    float s = 0.0;
    for (int d = -p.R; d <= p.R; d++) s += p.rows.v[clamp(y + d, 0, p.H - 1) * p.W + x];
    float side = float(2 * p.R + 1);
    float w = clamp(s / (side * side) / p.full, 0.0, 1.0);
    if (w <= 0.0) return;
    int i = y * p.W + x, n = p.W * p.H;
    for (int c = 0; c < 3; c++) {
        float v = p.b.v[c * n + i];
        if (isnan(v) || isinf(v)) v = p.a.v[c * n + i];
        p.b.v[c * n + i] = mix(v, p.a.v[c * n + i], w);
    }
}
"#;

pub struct Restorer {
    ops: Arc<Ops>,
    net: Net,
    pub model: RestoreModel,
    bad: Arc<Kernel>,
    rowsum: Arc<Kernel>,
    fallback: Arc<Kernel>,
}

impl Restorer {
    pub fn available(models: &Path, model: RestoreModel) -> bool {
        let (dir, stem) = model.file();
        models.join(dir).join(format!("{stem}.param")).exists()
    }

    pub fn new(ops: &Arc<Ops>, models: &Path, model: RestoreModel) -> Result<Self> {
        let (dir, stem) = model.file();
        let base = models.join(dir).join(stem);
        let param = std::fs::read_to_string(base.with_extension("param")).with_context(|| format!("{}.param", base.display()))?;
        let mut net = Net::load(&ops.compute, &param, &std::fs::read(base.with_extension("bin"))?)?;
        net.set_half(true);
        net.share_pool(&ops.pool);
        // About 0.3 s of GPU work per 1080p picture on a fast card: in pieces, so none
        // nears Windows' 2 s limit on slower ones.
        net.set_flush_every(40);
        let k = |name: &str, body: &str| ops.compute.kernel(&format!("restore_{name}"), &format!("{PRELUDE}{body}"), 0);
        Ok(Restorer { ops: ops.clone(), net, model, bad: k("bad", BAD)?, rowsum: k("rowsum", ROWSUM)?, fallback: k("fallback", FALLBACK)? })
    }

    /// Where the network ran away (see DIVERGE), `out` (3 channels, `img`'s size)
    /// shows `img` instead, faded in over the window around it.
    fn fall_back(&self, rec: &mut Recorder, img: &Tensor, out: &Tensor) -> Result<()> {
        let ops = &self.ops;
        let (w, h) = (img.w, img.h);
        let g = |k: &Kernel| Recorder::groups(k, [w as u32, h as u32, 1]);
        let bad = ops.tensor(1, h, w)?;
        rec.dispatch(&self.bad, g(&self.bad), &Push::new().t(img).t(out).t(&bad).i(w as i32).i(h as i32).f(DIVERGE))?;
        let rows = ops.tensor(1, h, w)?;
        rec.dispatch(&self.rowsum, g(&self.rowsum), &Push::new().t(&bad).t(&rows).i(w as i32).i(h as i32).i(FALLBACK_R))?;
        rec.dispatch(&self.fallback, g(&self.fallback), &Push::new().t(&rows).t(img).t(out).i(w as i32).i(h as i32).i(FALLBACK_R).f(FALLBACK_FULL))?;
        ops.recycle(rec, bad);
        ops.recycle(rec, rows);
        Ok(())
    }

    /// `img` (3+ channels; alpha kept as it is) restored, mixed with the original by
    /// `strength` (0..1).
    pub fn record(&self, rec: &mut Recorder, img: &Tensor, strength: f32) -> Result<Tensor> {
        let ops = &self.ops;
        let (m, replicate) = self.model.pad();
        let (pw, ph) = (img.w.div_ceil(m) * m, img.h.div_ceil(m) * m);
        let x = ops.tensor(3, ph, pw)?;
        ops.copy(rec, img, (0, 0, 0), (3, ph, pw), &x, (0, 0, 0), replicate, false)?;
        let y = self.net.record(rec, &[("img", &x)], &["out"])?.remove(0);
        ops.recycle(rec, x);
        let out = ops.tensor(img.c, img.h, img.w)?;
        ops.copy(rec, &y, (0, 0, 0), (3, img.h, img.w), &out, (0, 0, 0), false, false)?;
        ops.recycle(rec, y);
        self.fall_back(rec, img, &out)?;
        if img.c > 3 {
            ops.copy(rec, img, (3, 0, 0), (img.c - 3, img.h, img.w), &out, (3, 0, 0), false, false)?;
        }
        let s = strength.clamp(0.0, 1.0);
        if s >= 0.999 {
            return Ok(out);
        }
        let mixed = ops.mix(rec, img, &out, s)?;
        ops.recycle(rec, out);
        Ok(mixed)
    }
}
