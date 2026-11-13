//! Building every AI shader ahead of use (at start-up), so adding an effect does not
//! wait for compiles. The shared objects (operations, motion, denoiser) build theirs
//! when made; each network builds its kernel variants while it records, so each is
//! recorded once on a small blank picture and the recording dropped unsubmitted
//! (`Recorder::dry`: nothing of it reaches the GPU, even the parts a network would
//! submit on the way).
//! What gets compiled lands in the disk caches (`brp_render::shader_cache`), where the
//! effects' own objects find it later.

use crate::gpu::{Matting, Ops, Rife, Upscaler};
use crate::gpu_temporal::Temporal;
use crate::rife::RifeModel;
use crate::upscale::UpscaleModel;
use anyhow::Result;
use brp_render::compute::{Compute, Recorder};
use std::path::Path;
use std::sync::Arc;

/// Build all shaders the AI effects use. `progress(done, total)` after each step;
/// models missing on disk are skipped.
pub fn warm_up(compute: &Arc<Compute>, models: &Path, progress: &dyn Fn(usize, usize)) -> Result<()> {
    let total = 1 + UpscaleModel::ALL.len() + RifeModel::ALL.len() + 3;
    let mut done = 0;
    let mut step = |r: Result<()>, what: &str| {
        if let Err(e) = r {
            log::info!("shader warm-up: {what} skipped: {e:#}");
        }
        done += 1;
        progress(done, total);
    };
    let ops = Ops::new(compute)?;
    let shared = (|| -> Result<()> {
        Temporal::new(&ops)?;
        crate::denoise::Denoiser::new(&ops)?;
        Ok(())
    })();
    step(shared, "motion and denoiser");
    // A blank 64 px picture: one tile with its border, as the player records it.
    let blank = |rec: &mut Recorder, w: usize, h: usize| ops.filled(rec, 3, h, w, 0.5);
    for m in UpscaleModel::ALL {
        let r = (|| -> Result<()> {
            let up = Upscaler::new(&ops, models, m)?;
            let mut rec = Recorder::dry(compute)?;
            let img = blank(&mut rec, 64, 64)?;
            up.record(&mut rec, &img)?;
            Ok(())
        })();
        step(r, &format!("{m:?}"));
    }
    for m in RifeModel::ALL {
        let r = (|| -> Result<()> {
            let rife = Rife::new(&ops, models, m)?;
            let mut rec = Recorder::dry(compute)?;
            let (a, b) = (blank(&mut rec, 64, 64)?, blank(&mut rec, 64, 64)?);
            rife.record(&mut rec, &a, &b, 0.5)?;
            Ok(())
        })();
        step(r, &format!("{m:?}"));
    }
    let r = (|| -> Result<()> {
        if !Matting::available(models) {
            anyhow::bail!("no matting model");
        }
        let mut net = Matting::new(&ops, models, 128, 128)?;
        let mut rec = Recorder::dry(compute)?;
        let img = blank(&mut rec, 128, 128)?;
        net.record(&mut rec, &img)?;
        Ok(())
    })();
    step(r, "matting");
    let r = (|| -> Result<()> {
        if !crate::flow::SeaRaft::available(models) {
            anyhow::bail!("no SEA-RAFT model");
        }
        let sr = crate::flow::SeaRaft::new(&ops, models)?;
        let mut rec = Recorder::dry(compute)?;
        let (a, b) = (blank(&mut rec, 128, 128)?, blank(&mut rec, 128, 128)?);
        sr.flow(&mut rec, &a, &b)?;
        Ok(())
    })();
    step(r, "SEA-RAFT");
    let r = (|| -> Result<()> {
        if !crate::flow::Waft::available(models) {
            anyhow::bail!("no WAFT model");
        }
        let wf = crate::flow::Waft::new(&ops, models)?;
        let mut rec = Recorder::dry(compute)?;
        let (a, b) = (blank(&mut rec, 128, 128)?, blank(&mut rec, 128, 128)?);
        wf.flow(&mut rec, &a, &b)?;
        Ok(())
    })();
    step(r, "WAFT");
    Ok(())
}
