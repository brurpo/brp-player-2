//! Whether the GPU's cooperative matrices (tensor cores) are real hardware. Vulkan
//! cannot say: a driver may offer them emulated in plain shader code, slower than
//! the kernels written without them (an AMD Radeon iGPU ran SwinIR 8x slower with
//! them, long enough per submission for Windows to reset the GPU). So their speed is
//! measured once per GPU and driver against plain fp32 multiply-adds, and they are
//! used only when clearly faster. The verdict is kept in the shader cache folder
//! (`coopmat.txt`), so later runs do not measure again.

use crate::compute::{Compute, Push, Recorder, PRELUDE};
use crate::GpuContext;
use anyhow::Result;
use std::sync::Arc;

/// Tensor cores count as real when their multiply-adds run at least this fraction of
/// the plain fp32 rate. Measured: RTX 5070 Ti 2x (1.45x while another program used
/// the GPU), AMD Radeon iGPU (emulated) 0.01x: far from both.
const MIN_RATIO: f64 = 0.5;

/// Each measurement grows its work until one run takes this long (timing noise),
/// within `MAX_RUN` (far under Windows' 2 s limit, even on the slowest GPU).
const MIN_RUN: f64 = 0.005;
const MAX_RUN: f64 = 0.25;

/// `ctx` with `coopmat` (and `coopmat_acc16`) off when its tensor cores are far
/// slower than plain arithmetic (emulated).
pub fn check(ctx: GpuContext) -> GpuContext {
    if !ctx.coopmat {
        return ctx;
    }
    let props = unsafe { ctx.instance.get_physical_device_properties(ctx.physical) };
    let key = crate::shader_cache::device_id(&props);
    let file = crate::shader_cache::dir().map(|d| d.join("coopmat.txt"));
    let cached = file.as_ref().and_then(|f| std::fs::read_to_string(f).ok()).and_then(|s| {
        let mut l = s.lines();
        (l.next() == Some(key.as_str())).then(|| l.next() == Some("fast"))
    });
    let fast = match cached {
        Some(f) => f,
        None => {
            let ctx = Arc::new(ctx);
            let measured = measure(&ctx);
            let ctx = Arc::try_unwrap(ctx).ok().expect("tensor-core check: the context is still shared");
            let fast = match measured {
                Ok((coop, plain)) => {
                    let fast = coop >= MIN_RATIO * plain;
                    log::info!(
                        "tensor cores: {:.1} GMAC/s against {:.1} plain ({:.2}x): {}",
                        coop / 1e9,
                        plain / 1e9,
                        coop / plain,
                        if fast { "used" } else { "far slower than plain arithmetic (emulated), not used" }
                    );
                    fast
                }
                // Not measurable (a compile error): not trusted.
                Err(e) => {
                    log::warn!("tensor cores not used: their check failed: {e:#}");
                    false
                }
            };
            if let Some(f) = &file {
                let _ = std::fs::write(f, format!("{key}\n{}\n", if fast { "fast" } else { "slow" }));
            }
            return with(ctx, fast);
        }
    };
    if !fast {
        log::info!("tensor cores not used (measured as emulated on this GPU and driver)");
    }
    with(ctx, fast)
}

fn with(mut ctx: GpuContext, fast: bool) -> GpuContext {
    if !fast {
        ctx.coopmat = false;
        ctx.coopmat_acc16 = false;
    }
    ctx
}

/// (tensor-core, plain) multiply-adds per second.
fn measure(ctx: &Arc<GpuContext>) -> Result<(f64, f64)> {
    let compute = Compute::new(ctx.clone());
    let sg = ctx.subgroup_size.max(1);
    let out = compute.buffer(4 << 10, "coopcheck")?;
    // 4 subgroups per workgroup, each 4 independent 16x16x16 products per step.
    let coop = compute.kernel(
        "coopcheck_coop",
        &format!(
            "{PRELUDE}
#extension GL_KHR_cooperative_matrix : require
#extension GL_KHR_memory_scope_semantics : require
#extension GL_KHR_shader_subgroup_basic : require
#extension GL_EXT_shader_explicit_arithmetic_types_float16 : require
layout(local_size_x = {lx}) in;
layout(push_constant) uniform P {{ F32 o; int iters; }} p;
shared float s_c[4 * 4 * 256];
void main() {{
    coopmat<float16_t, gl_ScopeSubgroup, 16, 16, gl_MatrixUseA> a = coopmat<float16_t, gl_ScopeSubgroup, 16, 16, gl_MatrixUseA>(float16_t(0.001));
    coopmat<float16_t, gl_ScopeSubgroup, 16, 16, gl_MatrixUseB> b = coopmat<float16_t, gl_ScopeSubgroup, 16, 16, gl_MatrixUseB>(float16_t(0.001));
    coopmat<float, gl_ScopeSubgroup, 16, 16, gl_MatrixUseAccumulator> c[4];
    for (int j = 0; j < 4; j++) c[j] = coopmat<float, gl_ScopeSubgroup, 16, 16, gl_MatrixUseAccumulator>(0.0);
    for (int i = 0; i < p.iters; i++)
        for (int j = 0; j < 4; j++) c[j] = coopMatMulAdd(a, b, c[j]);
    int sgid = int(gl_SubgroupID);
    for (int j = 0; j < 4; j++) coopMatStore(c[j], s_c, (sgid * 4 + j) * 256, 16, gl_CooperativeMatrixLayoutRowMajor);
    barrier();
    if (gl_LocalInvocationID.x == 0) p.o.v[gl_WorkGroupID.x % 1024] = s_c[0] + s_c[256] + s_c[512] + s_c[768];
}}",
            lx = sg * 4
        ),
        0,
    )?;
    // The same number of threads, each 16 independent fp32 multiply-adds per step.
    let plain = compute.kernel(
        "coopcheck_plain",
        &format!(
            "{PRELUDE}
layout(local_size_x = {lx}) in;
layout(push_constant) uniform P {{ F32 o; int iters; }} p;
void main() {{
    vec4 a = vec4(0.999), b = vec4(0.001);
    vec4 c0 = vec4(float(gl_LocalInvocationID.x)), c1 = c0 + 1.0, c2 = c0 + 2.0, c3 = c0 + 3.0;
    for (int i = 0; i < p.iters; i++) {{
        c0 = fma(c0, a, b); c1 = fma(c1, a, b); c2 = fma(c2, a, b); c3 = fma(c3, a, b);
    }}
    vec4 s = c0 + c1 + c2 + c3;
    if (s.x + s.y + s.z + s.w == 12345.0) p.o.v[gl_WorkGroupID.x % 1024] = s.x;
}}",
            lx = sg * 4
        ),
        0,
    )?;
    let groups = 256u32;
    let threads = groups as f64 * sg as f64 * 4.0;
    // Multiply-adds per step: per subgroup 4 products of 16^3; per thread 16.
    let coop_macs = groups as f64 * 4.0 * 4.0 * 4096.0;
    let plain_macs = threads * 16.0;
    let run = |k: &Arc<crate::compute::Kernel>, iters: i32| -> Result<f64> {
        let mut rec = Recorder::new(&compute)?;
        rec.dispatch(k, [groups, 1, 1], &Push::new().ptr(&out).i(iters))?;
        Ok(rec.submit_wait()?.as_secs_f64())
    };
    // Rate in MACs per second: work doubled until a run is long enough to time.
    let rate = |k: &Arc<crate::compute::Kernel>, per_step: f64| -> Result<f64> {
        let mut iters = 16;
        loop {
            let t = run(k, iters)?;
            if t >= MIN_RUN || t * 2.0 > MAX_RUN || iters >= 1 << 24 {
                // Best of two at this size (the first may have raised the clocks).
                let t = t.min(run(k, iters)?);
                return Ok(per_step * iters as f64 / t.max(1e-9));
            }
            iters *= 2;
        }
    };
    let p1 = rate(&plain, plain_macs)?;
    let c1 = rate(&coop, coop_macs)?;
    let p2 = rate(&plain, plain_macs)?;
    Ok((c1, p1.max(p2)))
}
