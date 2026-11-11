//! Checks the compute foundation: upload, a kernel through buffer pointers, readback.

use brp_render::compute::{Compute, Push, Recorder, PRELUDE};
use brp_render::{GpuContext, GpuInstance};
use std::sync::Arc;

fn main() -> anyhow::Result<()> {
    env_logger::init();
    let ctx = Arc::new(GpuContext::headless(GpuInstance::new()?, std::env::var("BRP_GPU").ok().as_deref())?);
    println!("GPU {} (fp16 {})", ctx.device_name, ctx.fp16);
    let compute = Compute::new(ctx);
    let n = 1 << 20;
    let data: Vec<f32> = (0..n).map(|i| i as f32).collect();
    let src = compute.upload(bytemuck_f32(&data), "src")?;
    let dst = compute.buffer(n as u64 * 4, "dst")?;
    let src_text = format!(
        "{PRELUDE}
layout(local_size_x = 256) in;
layout(push_constant) uniform P {{ F32 src; F32 dst; uint n; float k; }} p;
void main() {{
    uint i = gl_GlobalInvocationID.x;
    if (i < p.n) p.dst.v[i] = p.src.v[i] * p.k + 1.0;
}}"
    );
    let k = compute.kernel("check_axpb", &src_text, 0)?;
    let mut rec = Recorder::new(&compute)?;
    rec.dispatch(&k, Recorder::groups(&k, [n, 1, 1]), &Push::new().ptr(&src).ptr(&dst).u(n).f(2.0))?;
    let t = rec.submit_wait()?;
    let out = compute.download(&dst, 0, n as u64 * 4)?;
    let out: Vec<f32> = out.chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect();
    let bad = out.iter().enumerate().filter(|(i, v)| **v != *i as f32 * 2.0 + 1.0).count();
    println!("{n} values in {t:?}, {bad} wrong");
    anyhow::ensure!(bad == 0, "compute check failed");
    Ok(())
}

fn bytemuck_f32(v: &[f32]) -> &[u8] {
    unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, v.len() * 4) }
}
