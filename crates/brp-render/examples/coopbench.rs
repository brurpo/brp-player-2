//! Raw cooperative-matrix throughput: subgroups doing only multiply-adds in
//! registers (fp16 x fp16 -> fp32 and -> fp16 accumulators).

use brp_render::compute::{Compute, Push, Recorder, PRELUDE};
use brp_render::{GpuContext, GpuInstance};
use std::sync::Arc;

fn main() -> anyhow::Result<()> {
    env_logger::init();
    let ctx = Arc::new(GpuContext::headless(GpuInstance::new()?, std::env::var("BRP_GPU").ok().as_deref())?);
    anyhow::ensure!(ctx.coopmat, "no cooperative matrices");
    let sg = ctx.subgroup_size;
    let compute = Compute::new(ctx);
    let out = compute.buffer(1 << 20, "out")?;
    for (name, acc_t) in [("fp32 accumulate", "float"), ("fp16 accumulate", "float16_t")] {
        let src = format!(
            "{PRELUDE}
#extension GL_KHR_cooperative_matrix : require
#extension GL_KHR_memory_scope_semantics : require
#extension GL_KHR_shader_subgroup_basic : require
#extension GL_EXT_shader_explicit_arithmetic_types_float16 : require
layout(local_size_x = {lx}) in;
layout(push_constant) uniform P {{ F32 o; int iters; }} p;
shared {acc_t} s_c[16 * 256];
void main() {{
    coopmat<float16_t, gl_ScopeSubgroup, 16, 16, gl_MatrixUseA> a = coopmat<float16_t, gl_ScopeSubgroup, 16, 16, gl_MatrixUseA>(float16_t(0.001));
    coopmat<float16_t, gl_ScopeSubgroup, 16, 16, gl_MatrixUseB> b = coopmat<float16_t, gl_ScopeSubgroup, 16, 16, gl_MatrixUseB>(float16_t(0.001));
    coopmat<{acc_t}, gl_ScopeSubgroup, 16, 16, gl_MatrixUseAccumulator> c[4];
    for (int j = 0; j < 4; j++) c[j] = coopmat<{acc_t}, gl_ScopeSubgroup, 16, 16, gl_MatrixUseAccumulator>({acc_t}(0.0));
    for (int i = 0; i < p.iters; i++)
        for (int j = 0; j < 4; j++) c[j] = coopMatMulAdd(a, b, c[j]);
    int sgid = int(gl_SubgroupID);
    for (int j = 0; j < 4; j++) coopMatStore(c[j], s_c, (sgid * 4 + j) * 256, 16, gl_CooperativeMatrixLayoutRowMajor);
    if (gl_LocalInvocationID.x == 0) p.o.v[gl_WorkGroupID.x % 1024] = float(s_c[0]) + float(s_c[256]) + float(s_c[512]) + float(s_c[768]);
}}",
            lx = sg * 4
        );
        let k = compute.kernel(&format!("coopbench_{acc_t}"), &src, 0)?;
        let (groups, iters) = (4096u32, 1000i32);
        let mut best = f64::MAX;
        for _ in 0..3 {
            let mut rec = Recorder::new(&compute)?;
            rec.dispatch(&k, [groups, 1, 1], &Push::new().ptr(&out).i(iters))?;
            best = best.min(rec.submit_wait()?.as_secs_f64());
        }
        let macs = groups as f64 * 4.0 * iters as f64 * 4.0 * 4096.0;
        println!("{name}: {:.1} TMAC/s ({:.1} TFLOPS) in {:.1} ms", macs / best / 1e12, 2.0 * macs / best / 1e12, best * 1e3);
    }
    Ok(())
}
