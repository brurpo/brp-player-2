//! Neural networks on the renderer's Vulkan device: our own runner for ncnn model
//! files (`.param` + `.bin`). Every layer is a compute kernel; a whole network is
//! recorded into one submission, so pictures never leave the GPU.
//!
//! Tensors are planar f32, `[c][h][w]` (a vector of n values is `[n][1][1]`).
//! Buffers are reused from a pool across layers and runs.

use crate::Image;
use anyhow::{bail, Context, Result};
use brp_render::compute::{Buffer, Compute, Kernel, Push, Recorder, PRELUDE};
use std::collections::HashMap;
use std::sync::{Arc, Mutex};

/// A GPU tensor `[c][h][w]` of f32, or of f16 inside networks that store their
/// activations at half precision (see `Net::set_half`; never outside a network).
#[derive(Clone)]
pub struct Tensor {
    pub buf: Arc<Buffer>,
    /// Byte offset of the tensor in `buf` (a view into a larger buffer).
    pub off: u64,
    pub c: usize,
    pub h: usize,
    pub w: usize,
    /// Elements are f16.
    pub half: bool,
}

impl Tensor {
    /// Channels `c0..c0 + c` of this tensor (no copy).
    pub fn view(&self, c0: usize, c: usize) -> Tensor {
        Tensor { buf: self.buf.clone(), off: self.off + (c0 * self.h * self.w) as u64 * self.esize(), c, h: self.h, w: self.w, half: self.half }
    }

    /// Bytes per element.
    pub fn esize(&self) -> u64 {
        if self.half { 2 } else { 4 }
    }

    pub fn len(&self) -> usize {
        self.c * self.h * self.w
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Upload a planar image (its channels become the tensor's).
    pub fn from_image(compute: &Compute, img: &Image) -> Result<Tensor> {
        let bytes = unsafe { std::slice::from_raw_parts(img.data.as_ptr() as *const u8, img.data.len() * 4) };
        Ok(Tensor { buf: Arc::new(compute.upload(bytes, "tensor")?), off: 0, c: img.channels, h: img.h, w: img.w, half: false })
    }

    /// Read back as a planar image.
    pub fn to_image(&self, compute: &Compute) -> Result<Image> {
        let raw = compute.download(&self.buf, self.off, self.len() as u64 * self.esize())?;
        let data = if self.half {
            raw.chunks_exact(2).map(|b| f16_to_f32(u16::from_le_bytes(b.try_into().unwrap()))).collect()
        } else {
            raw.chunks_exact(4).map(|b| f32::from_le_bytes(b.try_into().unwrap())).collect()
        };
        Ok(Image { w: self.w, h: self.h, channels: self.c, data })
    }
}

/// Buffers by size, reused between layers and runs.
#[derive(Default)]
pub struct Pool {
    /// Free buffers by size (best fit: the smallest that is big enough, up to twice).
    free: Mutex<std::collections::BTreeMap<u64, Vec<Arc<Buffer>>>>,
    /// Bytes this pool has allocated (it keeps what it gets back).
    held: std::sync::atomic::AtomicU64,
}

impl Pool {
    pub fn get(&self, compute: &Compute, bytes: u64) -> Result<Arc<Buffer>> {
        let bytes = bytes.max(4).div_ceil(256) * 256;
        {
            let mut free = self.free.lock().unwrap();
            let fit = free.range_mut(bytes..=bytes * 2).find(|(_, v)| !v.is_empty()).map(|(k, _)| *k);
            if let Some(k) = fit {
                return Ok(free.get_mut(&k).unwrap().pop().unwrap());
            }
        }
        let total = self.held.fetch_add(bytes, std::sync::atomic::Ordering::Relaxed) + bytes;
        log::trace!("nn: new buffer {} MB (this pool: {} MB)", bytes >> 20, total >> 20);
        Ok(Arc::new(compute.buffer(bytes, "blob")?))
    }

    pub fn tensor(&self, compute: &Compute, c: usize, h: usize, w: usize) -> Result<Tensor> {
        self.tensor_of(compute, c, h, w, false)
    }

    /// A tensor of f32, or f16 if `half`.
    pub fn tensor_of(&self, compute: &Compute, c: usize, h: usize, w: usize, half: bool) -> Result<Tensor> {
        let es = if half { 2 } else { 4 };
        Ok(Tensor { buf: self.get(compute, (c * h * w) as u64 * es)?, off: 0, c, h, w, half })
    }

    /// Give a buffer back (when nothing else holds it).
    pub fn put(&self, b: Arc<Buffer>) {
        if Arc::strong_count(&b) == 1 {
            self.free.lock().unwrap().entry(b.size).or_default().push(b);
        }
    }

    pub fn clear(&self) {
        self.free.lock().unwrap().clear();
    }

    /// GPU memory this pool has allocated.
    pub fn held(&self) -> u64 {
        self.held.load(std::sync::atomic::Ordering::Relaxed)
    }
}

// ---- kernels ---------------------------------------------------------------------

const ACT: &str = r#"
// erf, Abramowitz & Stegun 7.1.26 (error under 1.5e-7).
float erf_(float x) {
    float t = 1.0 / (1.0 + 0.3275911 * abs(x));
    float y = 1.0 - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t * exp(-x * x);
    return x < 0.0 ? -y : y;
}
// 9: GELU (exact), fused into convolutions.
float act(float v, int a, float a0, float a1) {
    if (a == 9) return 0.5 * v * (1.0 + erf_(v * 0.70710678));
    if (a == 1) return max(v, 0.0);
    if (a == 2) return v > 0.0 ? v : v * a0;
    if (a == 3) return clamp(v, a0, a1);
    if (a == 4) return 1.0 / (1.0 + exp(-v));
    if (a == 6) return v * clamp(v * a0 + a1, 0.0, 1.0);
    return v;
}
"#;

/// Dense convolution: each invocation makes 8 output channels of one pixel (the
/// weights are the same across the workgroup, so they are broadcast).
const CONV: &str = r#"
layout(local_size_x = 16, local_size_y = 8, local_size_z = 1) in;
layout(push_constant) uniform P {
    F32 inp; F32 wt; F32 bias; F32 outp;
    int C, H, W, OC, OH, OW, kw, kh, sw, sh, dw, dh, pl, pt, act_type, has_bias;
    float a0, a1;
} p;
#define OCB 8
void main() {
    int x = int(gl_GlobalInvocationID.x), y = int(gl_GlobalInvocationID.y);
    int oc0 = int(gl_WorkGroupID.z) * OCB;
    if (x >= p.OW || y >= p.OH) return;
    float acc[OCB];
    for (int o = 0; o < OCB; o++) acc[o] = (p.has_bias != 0 && oc0 + o < p.OC) ? p.bias.v[oc0 + o] : 0.0;
    int KK = p.kw * p.kh;
    int ix0 = x * p.sw - p.pl, iy0 = y * p.sh - p.pt;
    int nout = min(OCB, p.OC - oc0);
    for (int ic = 0; ic < p.C; ic++) {
        int ib = ic * p.H * p.W;
        for (int ky = 0; ky < p.kh; ky++) {
            int iy = iy0 + ky * p.dh;
            if (iy < 0 || iy >= p.H) continue;
            for (int kx = 0; kx < p.kw; kx++) {
                int ix = ix0 + kx * p.dw;
                if (ix < 0 || ix >= p.W) continue;
                float v = p.inp.v[ib + iy * p.W + ix];
                int wi = (oc0 * p.C + ic) * KK + ky * p.kw + kx;
                for (int o = 0; o < OCB; o++)
                    if (o < nout) acc[o] += v * p.wt.v[wi + o * p.C * KK];
            }
        }
    }
    for (int o = 0; o < nout; o++)
        p.outp.v[((oc0 + o) * p.OH + y) * p.OW + x] = act(acc[o], p.act_type, p.a0, p.a1);
}
"#;

/// Convolution with kernels up to 3x3, stride 1, no dilation (most layers), tiled:
/// a workgroup makes 64 output channels x an 8x8 block of pixels; input channels
/// are staged 8 at a time in shared memory (the input block with its border, and
/// the weights); each invocation accumulates 4 channels x 4 pixels.
const CONV_TILED: &str = r#"
// OCG groups of 4 output channels x PXG groups of 4 pixels (a BW x BH block).
layout(local_size_x = PXG, local_size_y = OCG, local_size_z = 1) in;
layout(push_constant) uniform P {
    F32 inp; F32 wt; F32 bias; F32 outp;
    int C, H, W, OC, OH, OW, kw, kh, sw, sh, dw, dh, pl, pt, act_type, has_bias;
    float a0, a1;
} p;
#define BH (PXG * 4 / BW)
#define NT (PXG * OCG)
shared vec4 s_w[8][9][OCG];
shared float s_i[8][BH + 2][BW + 2];
void main() {
    int tx = int(gl_LocalInvocationID.x), ty = int(gl_LocalInvocationID.y);
    int lid = ty * PXG + tx;
    int bx = int(gl_WorkGroupID.x) * BW, by = int(gl_WorkGroupID.y) * BH, oc0 = int(gl_WorkGroupID.z) * OCG * 4;
    int KK = p.kw * p.kh, TW = BW + p.kw - 1, TH = BH + p.kh - 1, TN = TH * TW;
    vec4 acc[4] = vec4[4](vec4(0.0), vec4(0.0), vec4(0.0), vec4(0.0));
    for (int ic0 = 0; ic0 < p.C; ic0 += 8) {
        for (int e = lid; e < 8 * TN; e += NT) {
            int icl = e / TN, r = e % TN, yy = r / TW, xx = r % TW;
            int ic = ic0 + icl, iy = by + yy - p.pt, ix = bx + xx - p.pl;
            s_i[icl][yy][xx] = (ic < p.C && iy >= 0 && iy < p.H && ix >= 0 && ix < p.W) ? p.inp.v[(ic * p.H + iy) * p.W + ix] : 0.0;
        }
        for (int e = lid; e < 8 * KK * OCG; e += NT) {
            int og = e % OCG, r = e / OCG, k = r % KK, icl = r / KK;
            int ic = ic0 + icl;
            vec4 v = vec4(0.0);
            for (int o = 0; o < 4; o++) {
                int oc = oc0 + og * 4 + o;
                if (oc < p.OC && ic < p.C) v[o] = p.wt.v[(oc * p.C + ic) * KK + k];
            }
            s_w[icl][k][og] = v;
        }
        barrier();
        int nic = min(8, p.C - ic0);
        for (int icl = 0; icl < nic; icl++)
            for (int ky = 0; ky < p.kh; ky++)
                for (int kx = 0; kx < p.kw; kx++) {
                    vec4 wv = s_w[icl][ky * p.kw + kx][ty];
                    for (int j = 0; j < 4; j++) {
                        int pi = tx * 4 + j;
                        acc[j] += s_i[icl][pi / BW + ky][pi % BW + kx] * wv;
                    }
                }
        barrier();
    }
    for (int j = 0; j < 4; j++) {
        int pi = tx * 4 + j;
        int y = by + pi / BW, x = bx + pi % BW;
        if (y >= p.OH || x >= p.OW) continue;
        for (int o = 0; o < 4; o++) {
            int oc = oc0 + ty * 4 + o;
            if (oc >= p.OC) continue;
            float v = acc[j][o] + (p.has_bias != 0 ? p.bias.v[oc] : 0.0);
            p.outp.v[(oc * p.OH + y) * p.OW + x] = act(v, p.act_type, p.a0, p.a1);
        }
    }
}
"#;

/// 3x3 stride-1 convolution on tensor cores (cooperative matrices, fp16 x fp16 ->
/// fp32): per tap, weights (16 output x 16 input channels) times input (16 input
/// channels x 16 pixels of a row).
///
/// The input block is staged in shared memory pixel-major (16 input channels of a
/// pixel side by side), so every tap is the same block shifted by whole pixels: one
/// copy serves all 9 taps, and each 16-pixel run starts aligned. Weights are read by
/// the tensor cores straight from memory (fp16 [oc (padded to 64)][tap][ic (padded to
/// 16)]). The next 16 input channels are fetched into registers while the current
/// ones are multiplied (two shared buffers, one barrier per step).
///
/// A workgroup is NSG subgroups, each RS rows of 16 pixels x MT tiles of 16 output
/// channels. The output step can also apply a PReLU and add another tensor
/// (`rc0 * conv + rc1 * res`): layers fused into the convolution at load time, which
/// saves a pass over the activations each.
const CONV_COOP: &str = r#"
#extension GL_KHR_cooperative_matrix : require
#extension GL_KHR_memory_scope_semantics : require
#extension GL_KHR_shader_subgroup_basic : require
#extension GL_EXT_shader_explicit_arithmetic_types_float16 : require
#extension GL_EXT_control_flow_attributes : require
// 16-byte elements, so the matrix loads can be wide.
layout(buffer_reference, std430, buffer_reference_align = 16) buffer W16 { uvec4 v[]; };
layout(local_size_x = SG * NSG, local_size_y = 1, local_size_z = 1) in;
// Element types are compiled in (IN16, OUT16, RES16: the input, output, `res` are
// f16), with pointers of the right type.
layout(buffer_reference, scalar, buffer_reference_align = 2) buffer H16 { float16_t v[]; };
#if IN16
// f16 input is read as 32-bit words (the tensor starts 4-byte aligned): 16-bit
// loads were twice as slow here.
#define IN_T U32
#else
#define IN_T F32
#endif
#if OUT16
#define OUT_T H16
#define ST_OUT(i, x) p.outp.v[i] = float16_t(clamp(x, -65504.0, 65504.0))
#else
#define OUT_T F32
#define ST_OUT(i, x) p.outp.v[i] = (x)
#endif
#if RES16
#define RES_T H16
#else
#define RES_T F32
#endif
#define LD_RES(i) float(p.res.v[i])
layout(push_constant) uniform P {
    IN_T inp; W16 wt; F32 bias; OUT_T outp; RES_T res; F32 slope;
    int C, H, W, OC, OH, OW, pl, pt, act_type, flags;
    float a0, a1, rc0, rc1; int ICP;
} p;
// flags: 1 bias, 2 add `res`, 4 PReLU, 8 PReLU slope per channel.
#define NT (SG * NSG)
#define PR (RS * NSG)
#ifndef KS
#define KS 3
#endif
// Kernel size KS x KS (3, or 1 for the Linear layers of transformers).
#define IW (16 + KS - 1)
#define NPIX ((PR + KS - 1) * IW)
// Halves per staged pixel (16 channels + 8 padding: matrix loads hit different banks).
#define SP 24
#define BUF (NPIX * SP)
#define ITEMS (NPIX * 8)
#define PER ((ITEMS + NT - 1) / NT)
shared uvec4 s_in[2 * BUF / 8];
shared float s_c[NSG * 256];
void main() {
    int lid = int(gl_LocalInvocationID.x);
    int sg = int(gl_SubgroupID), lane = int(gl_SubgroupInvocationID);
    int bx = int(gl_WorkGroupID.x) * 16, by = int(gl_WorkGroupID.y) * PR, oc0 = int(gl_WorkGroupID.z) * MT * 16;
    int plane = p.H * p.W;
#if ACC16
#define ACC_T float16_t
#else
#define ACC_T float
#endif
    coopmat<ACC_T, gl_ScopeSubgroup, 16, 16, gl_MatrixUseAccumulator> acc[MT * RS];
    [[unroll]] for (int j = 0; j < MT * RS; j++) acc[j] = coopmat<ACC_T, gl_ScopeSubgroup, 16, 16, gl_MatrixUseAccumulator>(ACC_T(0.0));
    // Each item: a channel pair of one staged pixel. Consecutive threads take
    // consecutive pixels of a channel (coalesced reads).
    // Values are kept as loaded until staged after the products, so the loads
    // overlap them.
#if IN16
    // The words holding the two values, and which half is the first one's.
    // (Which half is recomputed when staging rather than kept: registers are short.
    // The channel is even, so it is the pixel's parity within its plane.)
    uint ra[PER], rb[PER];
#define LOADS(j, s) if (ic < p.C) ra[j] = p.inp.v[(s) >> 1]; if (ic + 1 < p.C) rb[j] = p.inp.v[((s) + plane) >> 1];
#define ZERO(j) ra[j] = 0u; rb[j] = 0u;
#define PAR(q) (uint((by + (q) / IW - p.pt) * p.W + bx + (q) % IW - p.pl) & 1u)
#define PACK(j) (((ra[j] >> (16u * PAR(q))) & 0xFFFFu) | (((rb[j] >> (16u * ((PAR(q) + uint(plane)) & 1u))) & 0xFFFFu) << 16))
#else
    vec2 reg[PER];
#define LOADS(j, s) if (ic < p.C) reg[j].x = p.inp.v[s]; if (ic + 1 < p.C) reg[j].y = p.inp.v[(s) + plane];
#define ZERO(j) reg[j] = vec2(0.0);
#define PACK(j) packHalf2x16(reg[j])
#endif
    int chunks = p.ICP / 16;
    #define FETCH(c) [[unroll]] for (int j = 0; j < PER; j++) { \
        int e = lid + j * NT, cp = e / NPIX, q = e - cp * NPIX, yy = q / IW, xx = q - yy * IW; \
        int iy = by + yy - p.pt, ix = bx + xx - p.pl, ic = (c) * 16 + 2 * cp; \
        ZERO(j) \
        if (e < ITEMS && iy >= 0 && iy < p.H && ix >= 0 && ix < p.W) { \
            int s = (ic * p.H + iy) * p.W + ix; \
            LOADS(j, s) } }
    #define STAGE(b) [[unroll]] for (int j = 0; j < PER; j++) { \
        int e = lid + j * NT, cp = e / NPIX, q = e - cp * NPIX; \
        if (e < ITEMS) { int o = (b) * BUF + q * SP + 2 * cp; \
            s_in[o >> 3][(o >> 1) & 3] = PACK(j); } }
    FETCH(0)
    STAGE(0)
    barrier();
    for (int c = 0; c < chunks; c++) {
        int b = c & 1;
        bool more = c + 1 < chunks;
        if (more) { FETCH(c + 1) }
        // Fully unrolled: the compiler then overlaps the taps' loads and products
        // (it uses all 255 registers; partial unrolling cut that to 170-220 but was
        // 15-40% slower).
        [[unroll]] for (int k = 0; k < KS * KS; k++) {
            const int ky = k / KS, kx = k % KS;
            coopmat<float16_t, gl_ScopeSubgroup, 16, 16, gl_MatrixUseB> mb[RS];
            [[unroll]] for (int rs = 0; rs < RS; rs++)
                coopMatLoad(mb[rs], s_in, (b * BUF + ((sg * RS + rs + ky) * IW + kx) * SP) / 8, SP / 8, gl_CooperativeMatrixLayoutColumnMajor);
            [[unroll]] for (int m = 0; m < MT; m++) {
                coopmat<float16_t, gl_ScopeSubgroup, 16, 16, gl_MatrixUseA> ma;
                coopMatLoad(ma, p.wt.v, (((oc0 + m * 16) * (KS * KS) + k) * p.ICP + c * 16) / 8, KS * KS * p.ICP / 8, gl_CooperativeMatrixLayoutRowMajor);
                [[unroll]] for (int rs = 0; rs < RS; rs++) acc[m * RS + rs] = coopMatMulAdd(ma, mb[rs], acc[m * RS + rs]);
            }
        }
        if (more) { STAGE(b ^ 1) }
        barrier();
    }
    [[unroll]] for (int j = 0; j < MT * RS; j++) {
        int m = j / RS, r = sg * RS + j % RS;
        coopMatStore(coopmat<float, gl_ScopeSubgroup, 16, 16, gl_MatrixUseAccumulator>(acc[j]), s_c, sg * 256, 16, gl_CooperativeMatrixLayoutRowMajor);
        subgroupMemoryBarrierShared();
        subgroupBarrier();
        int y = by + r;
        for (int e = lane; e < 256; e += SG) {
            int oc = oc0 + m * 16 + e / 16, x = bx + e % 16;
            if (oc < p.OC && y < p.OH && x < p.OW) {
                float v = s_c[sg * 256 + e] + ((p.flags & 1) != 0 ? p.bias.v[oc] : 0.0);
                v = act(v, p.act_type, p.a0, p.a1);
                if ((p.flags & 4) != 0) {
                    float s = p.slope.v[(p.flags & 8) != 0 ? oc : 0];
                    v = v > 0.0 ? v : v * s;
                }
                int o = (oc * p.OH + y) * p.OW + x;
                if ((p.flags & 2) != 0) v = p.rc0 * v + p.rc1 * LD_RES(o);
                ST_OUT(o, v);
            }
        }
        subgroupBarrier();
    }
}
"#;

/// 3x3 stride-1 convolution by Winograd F(2x2, 3x3) on tensor cores: each 2x2 block
/// of outputs (a "tile") comes from its 4x4 input block, transformed (V = Bt d B),
/// multiplied position by position with the transformed weights (U = G g Gt, done at
/// load: [16 positions][oc (padded to 64)][ic (padded to 16)] fp16) and transformed
/// back (Y = At M A): 16 products per tile and channel pair instead of 36.
///
/// Per 16 input channels: the input block is staged (as in CONV_COOP), transformed
/// into shared memory as 16 matrices (16 channels x tiles), then each subgroup
/// multiplies its XS of the 16 positions: weights (16 output x 16 input channels)
/// times transformed input, for MT channel tiles x TG groups of 16 tiles. At the end
/// the positions are gathered in shared memory and transformed back, 2x2 outputs per
/// invocation, with the same output step as CONV_COOP. A workgroup makes MT x 16
/// channels x (16 x 2 TH) pixels (8 x TH tiles).
const CONV_WINO: &str = r#"
#extension GL_KHR_cooperative_matrix : require
#extension GL_KHR_memory_scope_semantics : require
#extension GL_KHR_shader_subgroup_basic : require
#extension GL_EXT_shader_explicit_arithmetic_types_float16 : require
#extension GL_EXT_control_flow_attributes : require
layout(buffer_reference, std430, buffer_reference_align = 16) buffer W16 { uvec4 v[]; };
layout(local_size_x = SG * NSG, local_size_y = 1, local_size_z = 1) in;
layout(buffer_reference, scalar, buffer_reference_align = 2) buffer H16 { float16_t v[]; };
#if IN16
#define IN_T U32
#else
#define IN_T F32
#endif
#if OUT16
#define OUT_T H16
#define ST_OUT(i, x) p.outp.v[i] = float16_t(clamp(x, -65504.0, 65504.0))
#else
#define OUT_T F32
#define ST_OUT(i, x) p.outp.v[i] = (x)
#endif
#if RES16
#define RES_T H16
#else
#define RES_T F32
#endif
#define LD_RES(i) float(p.res.v[i])
layout(push_constant) uniform P {
    IN_T inp; W16 wt; F32 bias; OUT_T outp; RES_T res; F32 slope;
    int C, H, W, OC, OH, OW, pl, pt, act_type, flags;
    float a0, a1, rc0, rc1; int ICP, OCP;
} p;
#define NT (SG * NSG)
#define XS (16 / NSG)
#define TW 8
#define NTI (TW * TH)
#define TG (NTI / 16)
#define PR (2 * TH)
#define IW (2 * TW + 2)
#define NPIX ((PR + 2) * IW)
#define SP 24
#define ITEMS (NPIX * 8)
#define PER ((ITEMS + NT - 1) / NT)
// Staged input (pixel-major, SP halves per pixel), transformed input ([position]
// [tile][channel], 24 halves per tile), and positions gathered for the output.
shared uvec4 s_in[NPIX * SP / 8];
shared uvec4 s_v[16 * NTI * SP / 8];
shared float s_m[16 * 256];
uint rd_in(int o) { return s_in[o >> 3][(o >> 1) & 3]; }
void main() {
    int lid = int(gl_LocalInvocationID.x);
    int sg = int(gl_SubgroupID);
    int bx = int(gl_WorkGroupID.x) * 2 * TW, by = int(gl_WorkGroupID.y) * PR, oc0 = int(gl_WorkGroupID.z) * MT * 16;
    int plane = p.H * p.W;
    coopmat<float, gl_ScopeSubgroup, 16, 16, gl_MatrixUseAccumulator> acc[XS * MT * TG];
    [[unroll]] for (int j = 0; j < XS * MT * TG; j++) acc[j] = coopmat<float, gl_ScopeSubgroup, 16, 16, gl_MatrixUseAccumulator>(0.0);
#if IN16
    uint ra[PER], rb[PER];
#define LOADS(j, s) if (ic < p.C) ra[j] = p.inp.v[(s) >> 1]; if (ic + 1 < p.C) rb[j] = p.inp.v[((s) + plane) >> 1];
#define ZERO(j) ra[j] = 0u; rb[j] = 0u;
#define PAR(q) (uint((by + (q) / IW - p.pt) * p.W + bx + (q) % IW - p.pl) & 1u)
#define PACK(j) (((ra[j] >> (16u * PAR(q))) & 0xFFFFu) | (((rb[j] >> (16u * ((PAR(q) + uint(plane)) & 1u))) & 0xFFFFu) << 16))
#else
    vec2 reg[PER];
#define LOADS(j, s) if (ic < p.C) reg[j].x = p.inp.v[s]; if (ic + 1 < p.C) reg[j].y = p.inp.v[(s) + plane];
#define ZERO(j) reg[j] = vec2(0.0);
#define PACK(j) packHalf2x16(reg[j])
#endif
    #define FETCH(c) [[unroll]] for (int j = 0; j < PER; j++) { \
        int e = lid + j * NT, cp = e / NPIX, q = e - cp * NPIX, yy = q / IW, xx = q - yy * IW; \
        int iy = by + yy - p.pt, ix = bx + xx - p.pl, ic = (c) * 16 + 2 * cp; \
        ZERO(j) \
        if (e < ITEMS && iy >= 0 && iy < p.H && ix >= 0 && ix < p.W) { \
            int s = (ic * p.H + iy) * p.W + ix; \
            LOADS(j, s) } }
    #define STAGE() [[unroll]] for (int j = 0; j < PER; j++) { \
        int e = lid + j * NT, cp = e / NPIX, q = e - cp * NPIX; \
        if (e < ITEMS) { int o = q * SP + 2 * cp; \
            s_in[o >> 3][(o >> 1) & 3] = PACK(j); } }
    int chunks = p.ICP / 16;
    FETCH(0)
    STAGE()
    for (int c = 0; c < chunks; c++) {
        // This step's weights first: their latency overlaps the transform (4% faster).
        coopmat<float16_t, gl_ScopeSubgroup, 16, 16, gl_MatrixUseA> wa[XS * MT];
        [[unroll]] for (int x = 0; x < XS; x++) [[unroll]] for (int m = 0; m < MT; m++)
            coopMatLoad(wa[x * MT + m], p.wt.v, (((sg * XS + x) * p.OCP + oc0 + m * 16) * p.ICP + c * 16) / 8, p.ICP / 8, gl_CooperativeMatrixLayoutRowMajor);
        barrier();
        // Input transform: one tile x channel pair per item.
        for (int it = lid; it < NTI * 8; it += NT) {
            int t = it >> 3, cp = it & 7, ty = t / TW, tx = t - ty * TW;
            vec2 d[4][4];
            [[unroll]] for (int i = 0; i < 4; i++) [[unroll]] for (int j = 0; j < 4; j++)
                d[i][j] = unpackHalf2x16(rd_in(((2 * ty + i) * IW + 2 * tx + j) * SP + 2 * cp));
            vec2 r[4][4];
            [[unroll]] for (int j = 0; j < 4; j++) {
                r[0][j] = d[0][j] - d[2][j]; r[1][j] = d[1][j] + d[2][j];
                r[2][j] = d[2][j] - d[1][j]; r[3][j] = d[1][j] - d[3][j];
            }
            [[unroll]] for (int i = 0; i < 4; i++) {
                vec2 v[4] = vec2[4](r[i][0] - r[i][2], r[i][1] + r[i][2], r[i][2] - r[i][1], r[i][1] - r[i][3]);
                [[unroll]] for (int j = 0; j < 4; j++) {
                    int o = ((i * 4 + j) * NTI + t) * SP + 2 * cp;
                    s_v[o >> 3][(o >> 1) & 3] = packHalf2x16(v[j]);
                }
            }
        }
        bool more = c + 1 < chunks;
        if (more) { FETCH(c + 1) }
        barrier();
        [[unroll]] for (int x = 0; x < XS; x++) {
            int xi = sg * XS + x;
            coopmat<float16_t, gl_ScopeSubgroup, 16, 16, gl_MatrixUseB> mb[TG];
            [[unroll]] for (int g = 0; g < TG; g++)
                coopMatLoad(mb[g], s_v, ((xi * NTI + g * 16) * SP) / 8, SP / 8, gl_CooperativeMatrixLayoutColumnMajor);
            [[unroll]] for (int m = 0; m < MT; m++) {
                [[unroll]] for (int g = 0; g < TG; g++) acc[(x * MT + m) * TG + g] = coopMatMulAdd(wa[x * MT + m], mb[g], acc[(x * MT + m) * TG + g]);
            }
        }
        // The staged block is not read any more (the transform is done).
        if (more) { STAGE() }
    }
    // Output: per channel tile and tile group, the 16 positions gathered, then 2x2
    // outputs per (channel, tile).
    [[unroll]] for (int m = 0; m < MT; m++) [[unroll]] for (int g = 0; g < TG; g++) {
        barrier();
        [[unroll]] for (int x = 0; x < XS; x++)
            coopMatStore(acc[(x * MT + m) * TG + g], s_m, (sg * XS + x) * 256, 16, gl_CooperativeMatrixLayoutRowMajor);
        barrier();
        for (int it = lid; it < 256; it += NT) {
            int ocl = it >> 4, tl = it & 15;
            int oc = oc0 + m * 16 + ocl, t = g * 16 + tl, ty = t / TW, tx = t - ty * TW;
            float mm[16];
            [[unroll]] for (int k = 0; k < 16; k++) mm[k] = s_m[k * 256 + ocl * 16 + tl];
            float u0[4], u1[4];
            [[unroll]] for (int j = 0; j < 4; j++) {
                u0[j] = mm[j] + mm[4 + j] + mm[8 + j];
                u1[j] = mm[4 + j] - mm[8 + j] - mm[12 + j];
            }
            float y[4] = float[4](u0[0] + u0[1] + u0[2], u0[1] - u0[2] - u0[3], u1[0] + u1[1] + u1[2], u1[1] - u1[2] - u1[3]);
            if (oc >= p.OC) continue;
            float bias = (p.flags & 1) != 0 ? p.bias.v[oc] : 0.0;
            float slope = (p.flags & 4) != 0 ? p.slope.v[(p.flags & 8) != 0 ? oc : 0] : 1.0;
            [[unroll]] for (int k = 0; k < 4; k++) {
                int oy = by + 2 * ty + (k >> 1), ox = bx + 2 * tx + (k & 1);
                if (oy >= p.OH || ox >= p.OW) continue;
                float v = act(y[k] + bias, p.act_type, p.a0, p.a1);
                if ((p.flags & 4) != 0) v = v > 0.0 ? v : v * slope;
                int o = (oc * p.OH + oy) * p.OW + ox;
                if ((p.flags & 2) != 0) v = p.rc0 * v + p.rc1 * LD_RES(o);
                ST_OUT(o, v);
            }
        }
    }
}
"#;

/// LayerNorm across the channels of each pixel (transformers keep tokens as the
/// pixels of a planar [c][h][w] map): y = (x - mean) / sqrt(var + eps) * g + b.
const CHANNEL_NORM: &str = r#"
// 32 consecutive pixels per workgroup (coalesced loads), their channels split 8 ways
// and summed through shared memory: 8x the threads of one per pixel (few-pixel maps,
// such as transformer tokens, otherwise leave the GPU mostly idle).
layout(local_size_x = 32, local_size_y = 8) in;
layout(push_constant) uniform P { F32 a; F32 o; F32 gb; int C, n; float eps; int ah, oh; } p;
shared float s_sum[8][32];
void main() {
    int lx = int(gl_LocalInvocationID.x), g = int(gl_LocalInvocationID.y);
    int i = int(gl_WorkGroupID.x + gl_WorkGroupID.y * gl_NumWorkGroups.x) * 32 + lx;
    bool ok = i < p.n;
    // Mean, then the variance of the deviations (not E[x^2] - mean^2: with values of
    // 4e5 that cancels in f32 when they are close together).
    float s = 0.0;
    if (ok) for (int c = g; c < p.C; c += 8) s += ld(p.a, c * p.n + i, p.ah != 0);
    s_sum[g][lx] = s;
    barrier();
    float mean = 0.0;
    for (int k = 0; k < 8; k++) mean += s_sum[k][lx];
    mean /= float(p.C);
    barrier();
    float s2 = 0.0;
    if (ok) for (int c = g; c < p.C; c += 8) { float v = ld(p.a, c * p.n + i, p.ah != 0) - mean; s2 += v * v; }
    s_sum[g][lx] = s2;
    barrier();
    float var = 0.0;
    for (int k = 0; k < 8; k++) var += s_sum[k][lx];
    float inv = inversesqrt(var / float(p.C) + p.eps);
    if (ok) for (int c = g; c < p.C; c += 8) st(p.o, c * p.n + i, p.oh != 0, (ld(p.a, c * p.n + i, p.ah != 0) - mean) * inv * p.gb.v[c] + p.gb.v[p.C + c]);
}
"#;

/// Swin window attention, on the output of the qkv projection (planar [3C][h][w]:
/// q, k, v, each heads x head_dim channels). One workgroup per (window, head), one
/// invocation per token of the window. The cyclic shift (torch.roll by -shift before,
/// +shift after) is index arithmetic: token (sy, sx) of the shifted map is pixel
/// ((sy + shift) % H, (sx + shift) % W), read and written there. Scores get the
/// relative position bias and, for shifted windows, the mask that keeps regions
/// apart (-100 between tokens of different regions, as SwinIR's calculate_mask).
const WINDOW_ATTN: &str = r#"
layout(local_size_x = WS * WS) in;
layout(push_constant) uniform P { F32 qkv; F32 o; F32 table; int C, H, W, heads, shift, ih, oh; } p;
#define N (WS * WS)
#define HD_MAX 64
shared float s_k[N * HD_MAX];
shared float s_v[N * HD_MAX];
int region(int s, int size) { return s < size - WS ? 0 : (s < size - p.shift ? 1 : 2); }
void main() {
    int t = int(gl_LocalInvocationID.x);
    int nwx = p.W / WS, win = int(gl_WorkGroupID.x), head = int(gl_WorkGroupID.y);
    int wy = win / nwx, wx = win - wy * nwx;
    int ty = t / WS, tx = t - ty * WS;
    int sy = wy * WS + ty, sx = wx * WS + tx;
    int px = ((sy + p.shift) % p.H) * p.W + (sx + p.shift) % p.W;
    int hd = p.C / p.heads, plane = p.H * p.W;
    int qc = head * hd, kc = p.C + head * hd, vc = 2 * p.C + head * hd;
    float scale = inversesqrt(float(hd));
    float q[HD_MAX];
    for (int d = 0; d < hd; d++) {
        q[d] = ld(p.qkv, (qc + d) * plane + px, p.ih != 0) * scale;
        s_k[t * HD_MAX + d] = ld(p.qkv, (kc + d) * plane + px, p.ih != 0);
        s_v[t * HD_MAX + d] = ld(p.qkv, (vc + d) * plane + px, p.ih != 0);
    }
    barrier();
    int my_label = p.shift > 0 ? region(sy, p.H) * 3 + region(sx, p.W) : 0;
    float sc[N];
    float mx = -1e30;
    for (int j = 0; j < N; j++) {
        int jy = j / WS, jx = j - jy * WS;
        float s = 0.0;
        for (int d = 0; d < hd; d++) s += q[d] * s_k[j * HD_MAX + d];
        int idx = (ty - jy + WS - 1) * (2 * WS - 1) + (tx - jx + WS - 1);
        s += p.table.v[idx * p.heads + head];
        if (p.shift > 0) {
            int l = region(wy * WS + jy, p.H) * 3 + region(wx * WS + jx, p.W);
            if (l != my_label) s -= 100.0;
        }
        sc[j] = s;
        mx = max(mx, s);
    }
    float sum = 0.0;
    for (int j = 0; j < N; j++) { sc[j] = exp(sc[j] - mx); sum += sc[j]; }
    float inv = 1.0 / sum;
    for (int d = 0; d < hd; d++) {
        float acc = 0.0;
        for (int j = 0; j < N; j++) acc += sc[j] * s_v[j * HD_MAX + d];
        st(p.o, (head * hd + d) * plane + px, p.oh != 0, acc * inv);
    }
}
"#;

/// CONV1X1_COOP's shape: (subgroups, 16-pixel tiles per subgroup, input channels per
/// step). Small workgroups won on SwinIR-M (RTX 5070 Ti, all 1x1 layers, 256x256
/// input, three runs): (2,1,32) 150 ms, (4,1,32) 160-185, (8,1,32) 175-198,
/// (8,2,32) 260, (4,4,32) 480; the 3x3 kernel run as 1x1: 270.
const GEMM_SHAPE: (usize, usize, usize) = (2, 1, 32);

/// 1x1 convolution of an f16 tensor on tensor cores, as the matrix product it is:
/// out[oc][px] = W[oc][ic] x in[ic][px], the planar input being the right-hand
/// matrix as it lies in memory. A workgroup makes 64 output channels x TN pixels
/// (flat: no 2D tile, no border). Each step reads KC input channels x TN pixels in
/// 16-byte pieces and stages them pixel-major (a pixel's channels side by side: the
/// tensor cores read the right-hand matrix fastest that way); the next step is
/// fetched into registers during the products. The output step writes 8 pixels of
/// a channel per 16-byte store. Same push constants as CONV_COOP. Needs the plane a
/// multiple of 8 pixels and the tensors 16-byte aligned.
const CONV1X1_COOP: &str = r#"
#extension GL_KHR_cooperative_matrix : require
#extension GL_KHR_memory_scope_semantics : require
#extension GL_KHR_shader_subgroup_basic : require
#extension GL_EXT_shader_explicit_arithmetic_types_float16 : require
#extension GL_EXT_control_flow_attributes : require
layout(buffer_reference, std430, buffer_reference_align = 16) buffer W16 { uvec4 v[]; };
layout(buffer_reference, scalar, buffer_reference_align = 2) buffer H16 { float16_t v[]; };
#ifndef NSG
#define NSG 8
#endif
#define MT 4
#ifndef RS
#define RS 2
#endif
#define TN (NSG * RS * 16)
#ifndef KC
#define KC 32
#endif
// Halves per staged pixel (8 of padding: the 8 rows of a matrix load hit
// different banks).
#define SPK (KC + 8)
#define NT (SG * NSG)
#define PER (KC / 2 * TN / 8 / NT)
layout(local_size_x = NT, local_size_y = 1, local_size_z = 1) in;
// UNALIGNED: planes not a multiple of 8 pixels (or tensors not 16-byte aligned):
// the 8-pixel loads and stores are made of single values, with bounds checks.
#ifndef UNALIGNED
#define UNALIGNED 0
#endif
#if OUT16
#define OUT_T H16
#define ST_OUT(i, x) p.outp.v[i] = float16_t(clamp(x, -65504.0, 65504.0))
#else
#define OUT_T F32
#define ST_OUT(i, x) p.outp.v[i] = (x)
#endif
#if RES16
#define RES_T H16
#else
#define RES_T F32
#endif
#define LD_RES(i) float(p.res.v[i])
layout(push_constant) uniform P {
    W16 inp; W16 wt; F32 bias; OUT_T outp; RES_T res; F32 slope;
    int C, H, W, OC, OH, OW, pl, pt, act_type, flags;
    float a0, a1, rc0, rc1; int ICP;
} p;
shared uvec4 s_b[2 * TN * SPK / 8];
#if UNALIGNED
uvec4 LOAD8(int ic, int px, int plane) {
    if (ic >= p.C) return uvec4(0u);
    H16 h = H16(uint64_t(p.inp));
    int b = ic * plane + px;
    float v[8];
    [[unroll]] for (int q = 0; q < 8; q++) v[q] = px + q < plane ? float(h.v[b + q]) : 0.0;
    return uvec4(packHalf2x16(vec2(v[0], v[1])), packHalf2x16(vec2(v[2], v[3])), packHalf2x16(vec2(v[4], v[5])), packHalf2x16(vec2(v[6], v[7])));
}
#else
#define LOAD8(ic, px, plane) ((ic) < p.C && (px) < (plane) ? p.inp.v[((ic) * (plane) + (px)) >> 3] : uvec4(0u))
#endif
// The output step reuses s_b (as floats) once the products are done.
#if ACC16
#define ACC_T float16_t
#else
#define ACC_T float
#endif
void main() {
    int lid = int(gl_LocalInvocationID.x);
    int sg = int(gl_SubgroupID), lane = int(gl_SubgroupInvocationID);
    int plane = p.H * p.W;
    int n0 = int(gl_WorkGroupID.x) * TN, oc0 = int(gl_WorkGroupID.z) * MT * 16;
    coopmat<ACC_T, gl_ScopeSubgroup, 16, 16, gl_MatrixUseAccumulator> acc[MT * RS];
    [[unroll]] for (int j = 0; j < MT * RS; j++) acc[j] = coopmat<ACC_T, gl_ScopeSubgroup, 16, 16, gl_MatrixUseAccumulator>(ACC_T(0.0));
    uvec4 ra[PER], rb[PER];
    // Item e: 8 pixels of a channel pair (two 16-byte reads); consecutive threads,
    // consecutive pixels. Staged pixel-major (a pixel's channels side by side, as
    // the tensor cores read the right-hand matrix fastest): 32-bit stores of pairs.
    #define FETCH(s) [[unroll]] for (int j = 0; j < PER; j++) { \
        int e = lid + j * NT, pr = e / (TN / 8), px = n0 + (e - pr * (TN / 8)) * 8, ic = (s) * KC + 2 * pr; \
        ra[j] = LOAD8(ic, px, plane); \
        rb[j] = LOAD8(ic + 1, px, plane); }
    #define STAGE(b) [[unroll]] for (int j = 0; j < PER; j++) { \
        int e = lid + j * NT, pr = e / (TN / 8), col = e - pr * (TN / 8); \
        [[unroll]] for (int q = 0; q < 8; q++) { \
            uint sh = 16u * uint(q & 1); \
            uint w = ((ra[j][q >> 1] >> sh) & 0xFFFFu) | (((rb[j][q >> 1] >> sh) & 0xFFFFu) << 16); \
            int o = (b) * TN * SPK + (col * 8 + q) * SPK + 2 * pr; \
            s_b[o >> 3][(o >> 1) & 3] = w; } }
    int steps = (p.ICP + KC - 1) / KC;
    FETCH(0)
    STAGE(0)
    barrier();
    for (int s = 0; s < steps; s++) {
        int b = s & 1;
        bool more = s + 1 < steps;
        if (more) { FETCH(s + 1) }
        [[unroll]] for (int h = 0; h < KC / 16; h++) {
            int k0 = s * KC + h * 16;
            if (k0 < p.ICP) {
                coopmat<float16_t, gl_ScopeSubgroup, 16, 16, gl_MatrixUseB> mb[RS];
                [[unroll]] for (int rs = 0; rs < RS; rs++)
                    coopMatLoad(mb[rs], s_b, (b * TN * SPK + (sg * RS * 16 + rs * 16) * SPK + h * 16) / 8, SPK / 8, gl_CooperativeMatrixLayoutColumnMajor);
                [[unroll]] for (int m = 0; m < MT; m++) {
                    coopmat<float16_t, gl_ScopeSubgroup, 16, 16, gl_MatrixUseA> ma;
                    coopMatLoad(ma, p.wt.v, ((oc0 + m * 16) * p.ICP + k0) / 8, p.ICP / 8, gl_CooperativeMatrixLayoutRowMajor);
                    [[unroll]] for (int rs = 0; rs < RS; rs++) acc[m * RS + rs] = coopMatMulAdd(ma, mb[rs], acc[m * RS + rs]);
                }
            }
        }
        if (more) { STAGE(b ^ 1) }
        barrier();
    }
    [[unroll]] for (int j = 0; j < MT * RS; j++) {
        int m = j / RS, rs = j % RS;
        coopMatStore(coopmat<float, gl_ScopeSubgroup, 16, 16, gl_MatrixUseAccumulator>(acc[j]), s_b, sg * 64, 4, gl_CooperativeMatrixLayoutRowMajor);
        subgroupMemoryBarrierShared();
        subgroupBarrier();
        // A lane per 8 consecutive pixels of one channel: one 16-byte store (f16).
        for (int t = lane; t < 32; t += SG) {
            int r = t >> 1, c8 = (t & 1) * 8;
            int oc = oc0 + m * 16 + r, px = n0 + sg * RS * 16 + rs * 16 + c8;
            if (oc < p.OC && px < plane) {
                float bias = (p.flags & 1) != 0 ? p.bias.v[oc] : 0.0;
                float sl = (p.flags & 4) != 0 ? p.slope.v[(p.flags & 8) != 0 ? oc : 0] : 1.0;
                int o = oc * plane + px;
                float v[8];
                [[unroll]] for (int q = 0; q < 8; q++) {
                    int e = r * 16 + c8 + q;
                    float x = act(uintBitsToFloat(s_b[sg * 64 + (e >> 2)][e & 3]) + bias, p.act_type, p.a0, p.a1);
                    x = x > 0.0 ? x : x * sl;
                    if ((p.flags & 2) != 0 && (UNALIGNED == 0 || px + q < plane)) x = p.rc0 * x + p.rc1 * LD_RES(o + q);
                    v[q] = OUT16 != 0 ? clamp(x, -65504.0, 65504.0) : x;
                }
#if UNALIGNED
                [[unroll]] for (int q = 0; q < 8; q++) if (px + q < plane) ST_OUT(o + q, v[q]);
#elif OUT16
                W16(uint64_t(p.outp)).v[o >> 3] = uvec4(packHalf2x16(vec2(v[0], v[1])), packHalf2x16(vec2(v[2], v[3])), packHalf2x16(vec2(v[4], v[5])), packHalf2x16(vec2(v[6], v[7])));
#else
                [[unroll]] for (int q = 0; q < 8; q++) p.outp.v[o + q] = v[q];
#endif
            }
        }
        subgroupBarrier();
    }
}
"#;

/// WINDOW_ATTN on tensor cores: per (window, head), S = Q Kt and O = P V as fp16
/// matrix products (f32 sums), softmax in f32. N = WS*WS tokens (a multiple of 16,
/// at most 64), HDP = the head's channels padded to 16 (at most 64). One subgroup
/// per 16 query tokens.
const WINDOW_ATTN_COOP: &str = r#"
#extension GL_KHR_cooperative_matrix : require
#extension GL_KHR_memory_scope_semantics : require
#extension GL_KHR_shader_subgroup_basic : require
#extension GL_KHR_shader_subgroup_arithmetic : require
#extension GL_EXT_shader_explicit_arithmetic_types_float16 : require
#extension GL_EXT_control_flow_attributes : require
#define N (WS * WS)
#define NSG (N / 16)
layout(local_size_x = SG * NSG) in;
layout(push_constant) uniform P { F32 qkv; F32 o; F32 table; int C, H, W, heads, shift, ih, oh; } p;
shared float16_t s_q[N * HDP];
shared float16_t s_k[N * HDP];
shared float16_t s_v[N * HDP];
// Per subgroup: its 16 rows of S, then P, in place (fp16: 20 KB in all, so more
// workgroups fit per multiprocessor). O goes over the subgroup's own Q rows.
shared float16_t s_p[NSG * 16 * N];
int region(int s, int size) { return s < size - WS ? 0 : (s < size - p.shift ? 1 : 2); }
int pixel(int wy, int wx, int t) {
    int sy = wy * WS + t / WS, sx = wx * WS + t % WS;
    return ((sy + p.shift) % p.H) * p.W + (sx + p.shift) % p.W;
}
void main() {
    int lid = int(gl_LocalInvocationID.x);
    int sg = int(gl_SubgroupID), lane = int(gl_SubgroupInvocationID);
    int nwx = p.W / WS, win = int(gl_WorkGroupID.x), head = int(gl_WorkGroupID.y);
    int wy = win / nwx, wx = win - wy * nwx;
    int hd = p.C / p.heads, plane = p.H * p.W;
    float scale = inversesqrt(float(hd));
    // Consecutive threads: consecutive tokens of one channel.
    for (int e = lid; e < N * HDP; e += SG * NSG) {
        int d = e / N, t = e - d * N, px = pixel(wy, wx, t);
        float q = 0.0, k = 0.0, v = 0.0;
        if (d < hd) {
            int c = head * hd + d;
            q = ld(p.qkv, c * plane + px, p.ih != 0) * scale;
            k = ld(p.qkv, (p.C + c) * plane + px, p.ih != 0);
            v = ld(p.qkv, (2 * p.C + c) * plane + px, p.ih != 0);
        }
        s_q[t * HDP + d] = float16_t(q);
        s_k[t * HDP + d] = float16_t(k);
        s_v[t * HDP + d] = float16_t(v);
    }
    barrier();
    int row0 = sg * 16, sb = sg * 16 * N;
    [[unroll]] for (int nj = 0; nj < N / 16; nj++) {
        coopmat<float, gl_ScopeSubgroup, 16, 16, gl_MatrixUseAccumulator> acc = coopmat<float, gl_ScopeSubgroup, 16, 16, gl_MatrixUseAccumulator>(0.0);
        [[unroll]] for (int kk = 0; kk < HDP / 16; kk++) {
            coopmat<float16_t, gl_ScopeSubgroup, 16, 16, gl_MatrixUseA> a;
            coopmat<float16_t, gl_ScopeSubgroup, 16, 16, gl_MatrixUseB> b;
            coopMatLoad(a, s_q, row0 * HDP + kk * 16, HDP, gl_CooperativeMatrixLayoutRowMajor);
            // Kt: K's rows read as columns.
            coopMatLoad(b, s_k, nj * 16 * HDP + kk * 16, HDP, gl_CooperativeMatrixLayoutColumnMajor);
            acc = coopMatMulAdd(a, b, acc);
        }
        coopMatStore(coopmat<float16_t, gl_ScopeSubgroup, 16, 16, gl_MatrixUseAccumulator>(acc), s_p, sb + nj * 16, N, gl_CooperativeMatrixLayoutRowMajor);
    }
    subgroupMemoryBarrierShared();
    subgroupBarrier();
    // Softmax of each row, with the position bias and (shifted windows) the mask.
    for (int r = 0; r < 16; r++) {
        int i = row0 + r, iy = i / WS, ix = i - iy * WS;
        int my_label = p.shift > 0 ? region(wy * WS + iy, p.H) * 3 + region(wx * WS + ix, p.W) : 0;
        float sv[(N + SG - 1) / SG];
        float mx = -1e30;
        [[unroll]] for (int u = 0; u < (N + SG - 1) / SG; u++) {
            int j = lane + u * SG;
            float s = -1e30;
            if (j < N) {
                int jy = j / WS, jx = j - jy * WS;
                s = float(s_p[sb + r * N + j]) + p.table.v[((iy - jy + WS - 1) * (2 * WS - 1) + (ix - jx + WS - 1)) * p.heads + head];
                if (p.shift > 0 && region(wy * WS + jy, p.H) * 3 + region(wx * WS + jx, p.W) != my_label) s -= 100.0;
            }
            sv[u] = s;
            mx = max(mx, s);
        }
        mx = subgroupMax(mx);
        float sum = 0.0;
        [[unroll]] for (int u = 0; u < (N + SG - 1) / SG; u++) {
            sv[u] = lane + u * SG < N ? exp(sv[u] - mx) : 0.0;
            sum += sv[u];
        }
        float inv = 1.0 / subgroupAdd(sum);
        [[unroll]] for (int u = 0; u < (N + SG - 1) / SG; u++) {
            int j = lane + u * SG;
            if (j < N) s_p[sb + r * N + j] = float16_t(sv[u] * inv);
        }
    }
    subgroupMemoryBarrierShared();
    subgroupBarrier();
    [[unroll]] for (int dj = 0; dj < HDP / 16; dj++) {
        coopmat<float, gl_ScopeSubgroup, 16, 16, gl_MatrixUseAccumulator> acc = coopmat<float, gl_ScopeSubgroup, 16, 16, gl_MatrixUseAccumulator>(0.0);
        [[unroll]] for (int kk = 0; kk < N / 16; kk++) {
            coopmat<float16_t, gl_ScopeSubgroup, 16, 16, gl_MatrixUseA> a;
            coopmat<float16_t, gl_ScopeSubgroup, 16, 16, gl_MatrixUseB> b;
            coopMatLoad(a, s_p, sb + kk * 16, N, gl_CooperativeMatrixLayoutRowMajor);
            coopMatLoad(b, s_v, kk * 16 * HDP + dj * 16, HDP, gl_CooperativeMatrixLayoutRowMajor);
            acc = coopMatMulAdd(a, b, acc);
        }
        // Over this subgroup's own Q rows (no longer needed; others read only theirs).
        coopMatStore(coopmat<float16_t, gl_ScopeSubgroup, 16, 16, gl_MatrixUseAccumulator>(acc), s_q, row0 * HDP + dj * 16, HDP, gl_CooperativeMatrixLayoutRowMajor);
    }
    subgroupMemoryBarrierShared();
    subgroupBarrier();
    for (int e = lane; e < 16 * hd; e += SG) {
        int d = e / 16, r = e - d * 16;
        st(p.o, (head * hd + d) * plane + pixel(wy, wx, row0 + r), p.oh != 0, float(s_q[(row0 + r) * HDP + d]));
    }
}
"#;

/// DAT's spatial window attention (Chen et al. 2023, "Dual Aggregation Transformer"),
/// on the qkv projection [3C][h][w]. The channels are split into two halves
/// ("branches"), each with its own rectangular window (wh x ww: 8x32, and 32x8) and
/// its own relative position bias table [(2wh-1)(2ww-1)][heads]; this runs one
/// branch (channels cb..cb + C/2). Shifted blocks roll by (sy, sx) as WINDOW_ATTN
/// does (index arithmetic) and mask across the roll's seams (-100). One workgroup per
/// (window, head), one invocation per query token; keys and values pass through
/// shared memory KB tokens at a time, with a running softmax (no N x N scores kept).
/// The scalar kernel: the reference for DAT_ATTN_COOP, and the fallback without
/// tensor cores.
const DAT_ATTN: &str = r#"
layout(local_size_x = N) in;
layout(push_constant) uniform P { F32 qkv; F32 o; F32 table; int C, H, W, heads, hd, cb, wh, ww, sy, sx, ih, oh; } p;
#define KB 64
#define HD_MAX 32
shared float s_k[KB * HD_MAX];
shared float s_v[KB * HD_MAX];
int region(int s, int size, int win, int sh) { return s < size - win ? 0 : (s < size - sh ? 1 : 2); }
int pixel(int wy, int wx, int t) {
    int y = wy * p.wh + t / p.ww, x = wx * p.ww + t % p.ww;
    return ((y + p.sy) % p.H) * p.W + (x + p.sx) % p.W;
}
void main() {
    int t = int(gl_LocalInvocationID.x);
    int nwx = p.W / p.ww, win = int(gl_WorkGroupID.x), head = int(gl_WorkGroupID.y);
    int wy = win / nwx, wx = win - wy * nwx;
    int ty = t / p.ww, tx = t - ty * p.ww;
    int plane = p.H * p.W, px = pixel(wy, wx, t);
    int qc = p.cb + head * p.hd;
    // Only the last row and column of windows straddle the roll's seams (elsewhere
    // every token is in region 0): the mask is skipped in the others.
    bool shifted = (p.sy > 0 || p.sx > 0) && (wy == p.H / p.wh - 1 || wx == nwx - 1);
    int my_label = shifted ? region(wy * p.wh + ty, p.H, p.wh, p.sy) * 3 + region(wx * p.ww + tx, p.W, p.ww, p.sx) : 0;
    float scale = inversesqrt(float(p.hd));
    float q[HD_MAX], acc[HD_MAX];
    for (int d = 0; d < HD_MAX; d++) {
        q[d] = d < p.hd ? ld(p.qkv, (qc + d) * plane + px, p.ih != 0) * scale : 0.0;
        acc[d] = 0.0;
    }
    float m = -1e30, l = 0.0;
    for (int j0 = 0; j0 < N; j0 += KB) {
        barrier();
        for (int e = t; e < KB * p.hd; e += N) {
            int d = e / KB, j = e - d * KB, jp = pixel(wy, wx, j0 + j);
            s_k[j * HD_MAX + d] = ld(p.qkv, (p.C + qc + d) * plane + jp, p.ih != 0);
            s_v[j * HD_MAX + d] = ld(p.qkv, (2 * p.C + qc + d) * plane + jp, p.ih != 0);
        }
        barrier();
        for (int j = 0; j < KB; j++) {
            int jt = j0 + j, jy = jt / p.ww, jx = jt - jy * p.ww;
            float s = 0.0;
            for (int d = 0; d < p.hd; d++) s += q[d] * s_k[j * HD_MAX + d];
            s += p.table.v[((ty - jy + p.wh - 1) * (2 * p.ww - 1) + (tx - jx + p.ww - 1)) * p.heads + head];
            if (shifted && region(wy * p.wh + jy, p.H, p.wh, p.sy) * 3 + region(wx * p.ww + jx, p.W, p.ww, p.sx) != my_label) s -= 100.0;
            float mn = max(m, s);
            float a = exp(m - mn), e = exp(s - mn);
            l = l * a + e;
            for (int d = 0; d < p.hd; d++) acc[d] = acc[d] * a + e * s_v[j * HD_MAX + d];
            m = mn;
        }
    }
    float inv = 1.0 / l;
    for (int d = 0; d < p.hd; d++) st(p.o, (qc + d) * plane + px, p.oh != 0, acc[d] * inv);
}
"#;

/// DAT_ATTN on tensor cores. Windows of N = 256 tokens are too large to keep all
/// scores (WINDOW_ATTN_COOP keeps a subgroup's 16 x N), so each subgroup takes 16
/// query tokens and streams the keys in blocks of KB: a first pass finds each row's
/// softmax maximum and sum, a second makes P = exp(S - max) / sum per block and adds
/// P V into its accumulators (S computed twice, but no rescaling of the output). K
/// and V of the window sit in shared memory as fp16 (32 KB), the head's channels
/// padded to HDP (<= KB). Per subgroup only 1 KB (Q, then each block's S and P in
/// place, then O, all fp16), so up to 16 subgroups fit (one per 16 queries: twice
/// the speed of 4 with f32 scratch; a one-pass variant with running rescaling of O,
/// which needs O in f32, fitted only 4 and was slower). Lane l < 32 takes row l & 15
/// and half l >> 4 of each block's columns (rotated: distinct banks), its partner
/// l ^ 16 the other half.
const DAT_ATTN_COOP: &str = r#"
#extension GL_KHR_cooperative_matrix : require
#extension GL_KHR_memory_scope_semantics : require
#extension GL_KHR_shader_subgroup_basic : require
#extension GL_KHR_shader_subgroup_shuffle : require
#extension GL_EXT_shader_explicit_arithmetic_types_float16 : require
#extension GL_EXT_control_flow_attributes : require
#define KB 32
layout(local_size_x = SG * NSG) in;
layout(push_constant) uniform P { F32 qkv; F32 o; F32 table; int C, H, W, heads, hd, cb, wh, ww, sy, sx, ih, oh; } p;
shared float16_t s_k[N * HDP];
shared float16_t s_v[N * HDP];
// Per subgroup: Q, then each block's S and P, then O, all fp16 (1 KB).
shared float16_t s_p[NSG * 16 * KB];
int region(int s, int size, int win, int sh) { return s < size - win ? 0 : (s < size - sh ? 1 : 2); }
int pixel(int wy, int wx, int t) {
    int y = wy * p.wh + t / p.ww, x = wx * p.ww + t % p.ww;
    return ((y + p.sy) % p.H) * p.W + (x + p.sx) % p.W;
}
void main() {
    int lid = int(gl_LocalInvocationID.x), sg = int(gl_SubgroupID), lane = int(gl_SubgroupInvocationID);
    int nwx = p.W / p.ww, win = int(gl_WorkGroupID.x), head = int(gl_WorkGroupID.y);
    int wy = win / nwx, wx = win - wy * nwx;
    int plane = p.H * p.W, qc = p.cb + head * p.hd;
    // Only the last row and column of windows straddle the roll's seams (elsewhere
    // every token is in region 0): the mask is skipped in the others.
    bool shifted = (p.sy > 0 || p.sx > 0) && (wy == p.H / p.wh - 1 || wx == nwx - 1);
    float scale = inversesqrt(float(p.hd));
    // K and V of the window; consecutive threads, consecutive tokens of a channel.
    for (int e = lid; e < N * HDP; e += SG * NSG) {
        int d = e / N, t = e - d * N;
        float k = 0.0, v = 0.0;
        if (d < p.hd) {
            int px = pixel(wy, wx, t);
            k = ld(p.qkv, (p.C + qc + d) * plane + px, p.ih != 0);
            v = ld(p.qkv, (2 * p.C + qc + d) * plane + px, p.ih != 0);
        }
        s_k[t * HDP + d] = float16_t(k);
        s_v[t * HDP + d] = float16_t(v);
    }
    barrier();
    int sb = sg * 16 * KB;
    int r = lane & 15, hf = (lane >> 4) & 1;
    bool live = lane < 32;
    for (int rb = sg; rb < N / 16; rb += NSG) {
        // Q rows (scaled), row-major in s_p, into registers.
        for (int e = lane; e < 16 * HDP; e += SG) {
            int d = e / 16, rr = e - d * 16;
            float q = d < p.hd ? ld(p.qkv, (qc + d) * plane + pixel(wy, wx, rb * 16 + rr), p.ih != 0) * scale : 0.0;
            s_p[sb + rr * HDP + d] = float16_t(q);
        }
        subgroupMemoryBarrierShared();
        subgroupBarrier();
        coopmat<float16_t, gl_ScopeSubgroup, 16, 16, gl_MatrixUseA> qa[HDP / 16];
        [[unroll]] for (int kk = 0; kk < HDP / 16; kk++) coopMatLoad(qa[kk], s_p, sb + kk * 16, HDP, gl_CooperativeMatrixLayoutRowMajor);
        subgroupBarrier();
        int i = rb * 16 + r, iy = i / p.ww, ix = i - iy * p.ww;
        int my_label = shifted ? region(wy * p.wh + iy, p.H, p.wh, p.sy) * 3 + region(wx * p.ww + ix, p.W, p.ww, p.sx) : 0;
        float m = -1e30, l = 0.0;
        coopmat<float, gl_ScopeSubgroup, 16, 16, gl_MatrixUseAccumulator> acc[HDP / 16];
        for (int pass = 0; pass < 2; pass++) {
            [[unroll]] for (int dj = 0; dj < HDP / 16; dj++) acc[dj] = coopmat<float, gl_ScopeSubgroup, 16, 16, gl_MatrixUseAccumulator>(0.0);
            float inv = 1.0 / l;
            for (int kb = 0; kb < N / KB; kb++) {
                // S = Q Kt for this block of keys, column-major in s_p (fp16; P replaces it in
                // place, each lane rewriting only the entries it read).
                [[unroll]] for (int nj = 0; nj < KB / 16; nj++) {
                    coopmat<float, gl_ScopeSubgroup, 16, 16, gl_MatrixUseAccumulator> s = coopmat<float, gl_ScopeSubgroup, 16, 16, gl_MatrixUseAccumulator>(0.0);
                    [[unroll]] for (int kk = 0; kk < HDP / 16; kk++) {
                        coopmat<float16_t, gl_ScopeSubgroup, 16, 16, gl_MatrixUseB> b;
                        coopMatLoad(b, s_k, (kb * KB + nj * 16) * HDP + kk * 16, HDP, gl_CooperativeMatrixLayoutColumnMajor);
                        s = coopMatMulAdd(qa[kk], b, s);
                    }
                    coopMatStore(coopmat<float16_t, gl_ScopeSubgroup, 16, 16, gl_MatrixUseAccumulator>(s), s_p, sb + nj * 256, 16, gl_CooperativeMatrixLayoutColumnMajor);
                }
                subgroupMemoryBarrierShared();
                subgroupBarrier();
                // This lane's 16 scores, with the position bias and the shift mask. The
                // halves start at different columns (no shared-memory bank conflicts).
                float sv[16];
                float bm = -1e30;
                [[unroll]] for (int u = 0; u < 16; u++) {
                    int c = hf * 16 + ((u + hf) & 15), j = kb * KB + c;
                    int jy = j / p.ww, jx = j - jy * p.ww;
                    float s = -1e30;
                    if (live) {
                        s = float(s_p[sb + c * 16 + r]) + p.table.v[((iy - jy + p.wh - 1) * (2 * p.ww - 1) + (ix - jx + p.ww - 1)) * p.heads + head];
                        if (shifted && region(wy * p.wh + jy, p.H, p.wh, p.sy) * 3 + region(wx * p.ww + jx, p.W, p.ww, p.sx) != my_label) s -= 100.0;
                    }
                    sv[u] = s;
                    bm = max(bm, s);
                }
                if (pass == 0) {
                    bm = max(bm, subgroupShuffleXor(bm, 16));
                    float mn = max(m, bm), sum = 0.0;
                    [[unroll]] for (int u = 0; u < 16; u++) sum += exp(sv[u] - mn);
                    sum += subgroupShuffleXor(sum, 16);
                    l = l * exp(m - mn) + sum;
                    m = mn;
                } else {
                    if (live) {
                        [[unroll]] for (int u = 0; u < 16; u++) {
                            int c = hf * 16 + ((u + hf) & 15);
                            s_p[sb + c * 16 + r] = float16_t(exp(sv[u] - m) * inv);
                        }
                    }
                    subgroupMemoryBarrierShared();
                    subgroupBarrier();
                    [[unroll]] for (int kk = 0; kk < KB / 16; kk++) {
                        coopmat<float16_t, gl_ScopeSubgroup, 16, 16, gl_MatrixUseA> a;
                        coopMatLoad(a, s_p, sb + kk * 256, 16, gl_CooperativeMatrixLayoutColumnMajor);
                        [[unroll]] for (int dj = 0; dj < HDP / 16; dj++) {
                            coopmat<float16_t, gl_ScopeSubgroup, 16, 16, gl_MatrixUseB> b;
                            coopMatLoad(b, s_v, (kb * KB + kk * 16) * HDP + dj * 16, HDP, gl_CooperativeMatrixLayoutRowMajor);
                            acc[dj] = coopMatMulAdd(a, b, acc[dj]);
                        }
                    }
                }
                subgroupBarrier();
            }
        }
        // O: row-major in s_p (fp16), then to the output pixels.
        [[unroll]] for (int dj = 0; dj < HDP / 16; dj++) coopMatStore(coopmat<float16_t, gl_ScopeSubgroup, 16, 16, gl_MatrixUseAccumulator>(acc[dj]), s_p, sb + dj * 16, HDP, gl_CooperativeMatrixLayoutRowMajor);
        subgroupMemoryBarrierShared();
        subgroupBarrier();
        for (int e = lane; e < 16 * p.hd; e += SG) {
            int d = e / 16, rr = e - d * 16;
            st(p.o, (qc + d) * plane + pixel(wy, wx, rb * 16 + rr), p.oh != 0, float(s_p[sb + rr * HDP + d]));
        }
        subgroupBarrier();
    }
}
"#;

/// Attention between all tokens (ViT blocks: DINOv2, the refinement transformer of
/// WAFT), on the qkv projection [3C][N] (any h x w as N tokens; q, k, v each heads x hd
/// channels, head-major). Tensor cores: a workgroup takes 16 query tokens per
/// subgroup of one head; keys and values pass through shared memory KB at a time
/// (fp16, the head's channels padded to HDP <= KB). Two passes over the keys: each
/// row's softmax maximum and sum, then P = exp(S - max) / sum and P V into the
/// accumulators (no rescaling of the output). Keys past N are masked; queries past N
/// are computed but not stored. Lane l < 32: row l & 15, half l >> 4 of each block's
/// columns (rotated: distinct banks).
const GLOBAL_ATTN_COOP: &str = r#"
#extension GL_KHR_cooperative_matrix : require
#extension GL_KHR_memory_scope_semantics : require
#extension GL_KHR_shader_subgroup_basic : require
#extension GL_KHR_shader_subgroup_shuffle : require
#extension GL_KHR_shader_subgroup_vote : require
#extension GL_EXT_shader_explicit_arithmetic_types_float16 : require
#extension GL_EXT_control_flow_attributes : require
#define KB 64
#define HALF_KB (KB / 2)
// A row's reference maximum is raised (and its sums rescaled) only when a block's
// scores pass it by more than this: exp(8) = 2981 still fits f16 probabilities.
#define TAU 8.0
layout(local_size_x = SG * NSG) in;
layout(buffer_reference, std430, buffer_reference_align = 16) buffer KV4 { uvec4 v[]; };
// kv: keys then values, [head][token][HDP] f16, tokens padded to NP (a multiple of
// KB) with zeros (KV_T). Each subgroup reads its tiles straight from it (the cache
// serves the workgroup's other subgroups): no staging, no workgroup barriers.
layout(push_constant) uniform P { F32 qkv; F32 o; KV4 kv; int C, N, heads, hd, ih, oh, NP; } p;
// Per subgroup: the block's scores (f32, column-major 16 x KB), also the staging of
// the output when it is rescaled or written; the probabilities (f16, the same layout;
// Q first). 6 KB: 8 subgroups in 48 KB.
shared float s_s[NSG * 16 * KB];
shared float16_t s_p[NSG * 16 * KB];
void main() {
    int lid = int(gl_LocalInvocationID.x), sg = int(gl_SubgroupID), lane = int(gl_SubgroupInvocationID);
    int head = int(gl_WorkGroupID.y);
    int q0 = (int(gl_WorkGroupID.x) * NSG + sg) * 16;
    int qc = head * p.hd;
    float scale = inversesqrt(float(p.hd));
    int sb = sg * 16 * KB;
    for (int e = lane; e < 16 * HDP; e += SG) {
        int d = e / 16, r = e - d * 16, t = q0 + r;
        float q = d < p.hd && t < p.N ? ld(p.qkv, (qc + d) * p.N + t, p.ih != 0) * scale : 0.0;
        s_p[sb + r * HDP + d] = float16_t(q);
    }
    subgroupMemoryBarrierShared();
    subgroupBarrier();
    coopmat<float16_t, gl_ScopeSubgroup, 16, 16, gl_MatrixUseA> qa[HDP / 16];
    [[unroll]] for (int kk = 0; kk < HDP / 16; kk++) coopMatLoad(qa[kk], s_p, sb + kk * 16, HDP, gl_CooperativeMatrixLayoutRowMajor);
    subgroupBarrier();
    int r = lane & 15, hf = (lane >> 4) & 1;
    bool live = lane < 32;
    float m = -1e30, l = 0.0;
    coopmat<float, gl_ScopeSubgroup, 16, 16, gl_MatrixUseAccumulator> acc[HDP / 16];
    [[unroll]] for (int dj = 0; dj < HDP / 16; dj++) acc[dj] = coopmat<float, gl_ScopeSubgroup, 16, 16, gl_MatrixUseAccumulator>(0.0);
    int blocks = (p.N + KB - 1) / KB;
    for (int kb = 0; kb < blocks; kb++) {
        int base = (head * p.NP + kb * KB) * (HDP / 8), vbase = base + p.heads * p.NP * (HDP / 8);
        [[unroll]] for (int nj = 0; nj < KB / 16; nj++) {
            coopmat<float, gl_ScopeSubgroup, 16, 16, gl_MatrixUseAccumulator> s = coopmat<float, gl_ScopeSubgroup, 16, 16, gl_MatrixUseAccumulator>(0.0);
            [[unroll]] for (int kk = 0; kk < HDP / 16; kk++) {
                coopmat<float16_t, gl_ScopeSubgroup, 16, 16, gl_MatrixUseB> b;
                coopMatLoad(b, p.kv.v, base + (nj * 16 * HDP + kk * 16) / 8, HDP / 8, gl_CooperativeMatrixLayoutColumnMajor);
                s = coopMatMulAdd(qa[kk], b, s);
            }
            coopMatStore(s, s_s, sb + nj * 256, 16, gl_CooperativeMatrixLayoutColumnMajor);
        }
        subgroupMemoryBarrierShared();
        subgroupBarrier();
        // Lane (r, hf): half the block's columns of row r (rotated: no bank conflicts).
        float sv[HALF_KB];
        float bm = -1e30;
        [[unroll]] for (int u = 0; u < HALF_KB; u++) {
            int c = hf * HALF_KB + ((u + hf) & (HALF_KB - 1));
            float s = -1e30;
            if (live && kb * KB + c < p.N) s = s_s[sb + c * 16 + r];
            sv[u] = s;
            bm = max(bm, s);
        }
        bm = max(bm, subgroupShuffleXor(bm, 16));
        bool raise = live && bm > m + TAU;
        if (subgroupAny(raise)) {
            // Every row moves to its block maximum where that is higher; the output so
            // far is scaled through shared memory (rare after the first blocks).
            float mn = max(m, bm), a = exp(m - mn);
            if (kb > 0) {
                subgroupBarrier();
                [[unroll]] for (int dj = 0; dj < HDP / 16; dj++) coopMatStore(acc[dj], s_s, sb + dj * 256, 16, gl_CooperativeMatrixLayoutColumnMajor);
                subgroupMemoryBarrierShared();
                subgroupBarrier();
                if (live) {
                    for (int c = hf * (HDP / 2); c < (hf + 1) * (HDP / 2); c++) s_s[sb + c * 16 + r] *= a;
                }
                subgroupMemoryBarrierShared();
                subgroupBarrier();
                [[unroll]] for (int dj = 0; dj < HDP / 16; dj++) coopMatLoad(acc[dj], s_s, sb + dj * 256, 16, gl_CooperativeMatrixLayoutColumnMajor);
            }
            l *= a;
            m = mn;
        }
        if (live) {
            float sum = 0.0;
            [[unroll]] for (int u = 0; u < HALF_KB; u++) {
                int c = hf * HALF_KB + ((u + hf) & (HALF_KB - 1));
                float e = exp(sv[u] - m);
                sum += e;
                s_p[sb + c * 16 + r] = float16_t(e);
            }
            l += sum;
        }
        subgroupMemoryBarrierShared();
        subgroupBarrier();
        [[unroll]] for (int kk = 0; kk < KB / 16; kk++) {
            coopmat<float16_t, gl_ScopeSubgroup, 16, 16, gl_MatrixUseA> a;
            coopMatLoad(a, s_p, sb + kk * 256, 16, gl_CooperativeMatrixLayoutColumnMajor);
            [[unroll]] for (int dj = 0; dj < HDP / 16; dj++) {
                coopmat<float16_t, gl_ScopeSubgroup, 16, 16, gl_MatrixUseB> b;
                coopMatLoad(b, p.kv.v, vbase + (kk * 16 * HDP + dj * 16) / 8, HDP / 8, gl_CooperativeMatrixLayoutRowMajor);
                acc[dj] = coopMatMulAdd(a, b, acc[dj]);
            }
        }
        subgroupBarrier();
    }
    l += subgroupShuffleXor(l, 16);
    // Each lane writes rows lane & 15 below (SG is a multiple of 16): that row's sum.
    l = subgroupShuffle(l, lane & 15);
    [[unroll]] for (int dj = 0; dj < HDP / 16; dj++) coopMatStore(acc[dj], s_s, sb + dj * 256, 16, gl_CooperativeMatrixLayoutColumnMajor);
    subgroupMemoryBarrierShared();
    subgroupBarrier();
    for (int e = lane; e < 16 * p.hd; e += SG) {
        int d = e / 16, rr = e - d * 16, t = q0 + rr;
        if (t < p.N) st(p.o, (qc + d) * p.N + t, p.oh != 0, s_s[sb + d * 16 + rr] / l);
    }
}
"#;

/// Keys and values of a qkv projection ([3C][N]) as [head][token][HDP] f16, zero past
/// hd, keys then values: GLOBAL_ATTN_COOP loads a block of keys as one contiguous run.
const KV_T: &str = r#"
layout(local_size_x = 64, local_size_y = 4) in;
layout(push_constant) uniform P { F32 qkv; F32 kv; int C, N, heads, hd, ih, NP; } p;
void main() {
    int d = int(gl_GlobalInvocationID.x), j = int(gl_GlobalInvocationID.y), z = int(gl_GlobalInvocationID.z);
    if (j >= p.NP || d >= HDP) return;
    int which = z / p.heads, h = z - which * p.heads;
    float v = d < p.hd && j < p.N ? ld(p.qkv, ((1 + which) * p.C + h * p.hd + d) * p.N + j, p.ih != 0) : 0.0;
    F16S(uint64_t(p.kv)).v[(z * p.NP + j) * HDP + d] = float16_t(v);
}
"#;

/// GLOBAL_ATTN_COOP without tensor cores: one invocation per query token, a running
/// softmax over all keys, which pass through shared memory 64 at a time.
const GLOBAL_ATTN: &str = r#"
layout(local_size_x = 64) in;
layout(push_constant) uniform P { F32 qkv; F32 o; int C, N, heads, hd, ih, oh; } p;
#define KB 64
#define HD_MAX 64
shared float s_k[KB * HD_MAX];
shared float s_v[KB * HD_MAX];
void main() {
    int t = int(gl_LocalInvocationID.x), i = int(gl_GlobalInvocationID.x), head = int(gl_WorkGroupID.y);
    int qc = head * p.hd;
    float scale = inversesqrt(float(p.hd));
    float q[HD_MAX], acc[HD_MAX];
    for (int d = 0; d < HD_MAX; d++) {
        q[d] = d < p.hd && i < p.N ? ld(p.qkv, (qc + d) * p.N + i, p.ih != 0) * scale : 0.0;
        acc[d] = 0.0;
    }
    float m = -1e30, l = 0.0;
    for (int j0 = 0; j0 < p.N; j0 += KB) {
        barrier();
        for (int e = t; e < KB * p.hd; e += 64) {
            int d = e / KB, j = e - d * KB, jj = j0 + j;
            s_k[j * HD_MAX + d] = jj < p.N ? ld(p.qkv, (p.C + qc + d) * p.N + jj, p.ih != 0) : 0.0;
            s_v[j * HD_MAX + d] = jj < p.N ? ld(p.qkv, (2 * p.C + qc + d) * p.N + jj, p.ih != 0) : 0.0;
        }
        barrier();
        for (int j = 0; j < KB && j0 + j < p.N; j++) {
            float s = 0.0;
            for (int d = 0; d < p.hd; d++) s += q[d] * s_k[j * HD_MAX + d];
            float mn = max(m, s), a = exp(m - mn), e = exp(s - mn);
            l = l * a + e;
            for (int d = 0; d < p.hd; d++) acc[d] = acc[d] * a + e * s_v[j * HD_MAX + d];
            m = mn;
        }
    }
    if (i >= p.N) return;
    for (int d = 0; d < p.hd; d++) st(p.o, (qc + d) * p.N + i, p.oh != 0, acc[d] / l);
}
"#;

/// Pixel unshuffle by r (inverse of PIXEL_SHUFFLE): out[c r r + i r + j][y][x] =
/// in[c][y r + i][x r + j] (torch's order). A patch embedding (r x r convolution,
/// stride r) is this, then a 1x1 convolution.
const PIXEL_UNSHUFFLE: &str = r#"
layout(local_size_x = 16, local_size_y = 8, local_size_z = 1) in;
layout(push_constant) uniform P { F32 a; F32 o; int C, H, W, r, ah, oh; } p;
void main() {
    int x = int(gl_GlobalInvocationID.x), y = int(gl_GlobalInvocationID.y), oc = int(gl_GlobalInvocationID.z);
    int OH = p.H / p.r, OW = p.W / p.r;
    if (x >= OW || y >= OH || oc >= p.C * p.r * p.r) return;
    int c = oc / (p.r * p.r), k = oc - c * p.r * p.r, i = k / p.r, j = k - i * p.r;
    st(p.o, (oc * OH + y) * OW + x, p.oh != 0, ld(p.a, (c * p.H + y * p.r + i) * p.W + x * p.r + j, p.ah != 0));
}
"#;

/// The top-left OH x OW of the input, zeros past its edge (Swin's padding to whole
/// windows, and the crop back).
const CANVAS: &str = r#"
layout(local_size_x = 16, local_size_y = 8, local_size_z = 1) in;
layout(push_constant) uniform P { F32 a; F32 o; int C, H, W, OH, OW, ah, oh; } p;
void main() {
    int x = int(gl_GlobalInvocationID.x), y = int(gl_GlobalInvocationID.y), c = int(gl_GlobalInvocationID.z);
    if (x >= p.OW || y >= p.OH || c >= p.C) return;
    float v = (x < p.W && y < p.H) ? ld(p.a, (c * p.H + y) * p.W + x, p.ah != 0) : 0.0;
    st(p.o, (c * p.OH + y) * p.OW + x, p.oh != 0, v);
}
"#;

/// The input cut in TH x TW tiles, each tile a channel: b c (hg h) (wg w) ->
/// b (c hg wg) h w (BiRefNet's image input to every decoder level).
const TILES: &str = r#"
layout(local_size_x = 16, local_size_y = 8, local_size_z = 1) in;
layout(push_constant) uniform P { F32 a; F32 o; int C, H, W, TH, TW, ah, oh; } p;
void main() {
    int x = int(gl_GlobalInvocationID.x), y = int(gl_GlobalInvocationID.y), oc = int(gl_GlobalInvocationID.z);
    int hg = p.H / p.TH, wg = p.W / p.TW;
    if (x >= p.TW || y >= p.TH || oc >= p.C * hg * wg) return;
    int c = oc / (hg * wg), g = oc - c * hg * wg, gy = g / wg, gx = g - gy * wg;
    st(p.o, (oc * p.TH + y) * p.TW + x, p.oh != 0, ld(p.a, (c * p.H + gy * p.TH + y) * p.W + gx * p.TW + x, p.ah != 0));
}
"#;

/// Modulated deformable convolution (torchvision's deform_conv2d, one offset group,
/// stride 1), its sampling half: [C x K^2][H][W], channel c * K^2 + tap = input
/// channel c bilinearly sampled at the tap moved by its offset (zero outside, as
/// torchvision), times 2 sigmoid(modulator). `om`: the offsets (dy, dx per tap,
/// 2 K^2 channels) then the modulator logits (K^2). A 1x1 convolution with the
/// weights as [out][C x K^2] finishes it.
const DEFORM_IM2COL: &str = r#"
layout(local_size_x = 16, local_size_y = 8, local_size_z = 1) in;
layout(push_constant) uniform P { F32 a; F32 om; F32 o; int C, H, W, K, pad, ah, mh, oh; } p;
void main() {
    int x = int(gl_GlobalInvocationID.x), y = int(gl_GlobalInvocationID.y), tap = int(gl_GlobalInvocationID.z);
    int KK = p.K * p.K;
    if (x >= p.W || y >= p.H || tap >= KK) return;
    int n = p.H * p.W, i = y * p.W + x, ky = tap / p.K, kx = tap - ky * p.K;
    float dy = ld(p.om, (2 * tap) * n + i, p.mh != 0), dx = ld(p.om, (2 * tap + 1) * n + i, p.mh != 0);
    float m = 2.0 / (1.0 + exp(-ld(p.om, (2 * KK + tap) * n + i, p.mh != 0)));
    float h = float(y - p.pad + ky) + dy, w = float(x - p.pad + kx) + dx;
    bool outside = h <= -1.0 || h >= float(p.H) || w <= -1.0 || w >= float(p.W);
    int h0 = int(floor(h)), w0 = int(floor(w)), h1 = h0 + 1, w1 = w0 + 1;
    float lh = h - float(h0), lw = w - float(w0), hh = 1.0 - lh, hw = 1.0 - lw;
    float k1 = (h0 >= 0 && w0 >= 0) ? hh * hw * m : 0.0;
    float k2 = (h0 >= 0 && w1 <= p.W - 1) ? hh * lw * m : 0.0;
    float k3 = (h1 <= p.H - 1 && w0 >= 0) ? lh * hw * m : 0.0;
    float k4 = (h1 <= p.H - 1 && w1 <= p.W - 1) ? lh * lw * m : 0.0;
    int i1 = clamp(h0, 0, p.H - 1) * p.W + clamp(w0, 0, p.W - 1), i2 = clamp(h0, 0, p.H - 1) * p.W + clamp(w1, 0, p.W - 1);
    int i3 = clamp(h1, 0, p.H - 1) * p.W + clamp(w0, 0, p.W - 1), i4 = clamp(h1, 0, p.H - 1) * p.W + clamp(w1, 0, p.W - 1);
    for (int c = 0; c < p.C; c++) {
        float v = 0.0;
        if (!outside) {
            int b = c * n;
            v = k1 * ld(p.a, b + i1, p.ah != 0) + k2 * ld(p.a, b + i2, p.ah != 0) + k3 * ld(p.a, b + i3, p.ah != 0) + k4 * ld(p.a, b + i4, p.ah != 0);
        }
        st(p.o, (c * KK + tap) * n + i, p.oh != 0, v);
    }
}
"#;

/// Global average of each channel, step 1: sums of chunks of 4096 pixels
/// (part[c][chunk]); step 2 (GLOBAL_AVG) adds them in order.
const GLOBAL_SUM: &str = r#"
layout(local_size_x = 256) in;
layout(push_constant) uniform P { F32 a; F32 part; int plane, chunks, ih; } p;
shared float s[256];
void main() {
    int t = int(gl_LocalInvocationID.x), ck = int(gl_WorkGroupID.x), c = int(gl_WorkGroupID.y);
    float acc = 0.0;
    int p0 = ck * 4096, p1 = min(p0 + 4096, p.plane);
    for (int i = p0 + t; i < p1; i += 256) acc += ld(p.a, c * p.plane + i, p.ih != 0);
    s[t] = acc;
    barrier();
    for (int k = 128; k > 0; k >>= 1) {
        if (t < k) s[t] += s[t + k];
        barrier();
    }
    if (t == 0) p.part.v[c * p.chunks + ck] = s[0];
}
"#;

/// DAT's channel attention (as XCiT: attention between channels, over all pixels),
/// step 1: per head, sums over a chunk of pixels of q_i k_j (hd x hd), q_i^2 and
/// k_j^2, into part[chunk][head][hd * hd + 2 hd]. Chunks are summed in a fixed order
/// afterwards (no atomics: the same result every run).
const CA_GRAM: &str = r#"
layout(local_size_x = 256) in;
layout(push_constant) uniform P { F32 qkv; F32 part; int C, plane, hd, chunk, ih; } p;
#define S 64
#define SP (S + 1)
#define HD_MAX 32
shared float s_q[HD_MAX * SP];
shared float s_k[HD_MAX * SP];
void main() {
    int t = int(gl_LocalInvocationID.x), ck = int(gl_WorkGroupID.x), head = int(gl_WorkGroupID.y);
    int items = p.hd * p.hd + 2 * p.hd;
    // Up to 5 items per invocation: hd <= 32 makes at most 1088.
    float acc[5] = float[5](0.0, 0.0, 0.0, 0.0, 0.0);
    int qc = head * p.hd, p0 = ck * p.chunk, p1 = min(p0 + p.chunk, p.plane);
    for (int s0 = p0; s0 < p1; s0 += S) {
        barrier();
        for (int e = t; e < p.hd * S; e += 256) {
            int d = e / S, j = e - d * S, px = s0 + j;
            bool in_ = px < p1;
            s_q[d * SP + j] = in_ ? ld(p.qkv, (qc + d) * p.plane + px, p.ih != 0) : 0.0;
            s_k[d * SP + j] = in_ ? ld(p.qkv, (p.C + qc + d) * p.plane + px, p.ih != 0) : 0.0;
        }
        barrier();
        for (int u = 0; u < 5; u++) {
            int it = t + u * 256;
            if (it >= items) break;
            int i, j;
            bool qq = false, kk = false;
            if (it < p.hd * p.hd) { i = it / p.hd; j = it - i * p.hd; }
            else if (it < p.hd * p.hd + p.hd) { i = j = it - p.hd * p.hd; qq = true; }
            else { i = j = it - p.hd * p.hd - p.hd; kk = true; }
            float s = 0.0;
            for (int x = 0; x < S; x++) {
                float a = kk ? s_k[i * SP + x] : s_q[i * SP + x];
                float b = qq ? s_q[j * SP + x] : s_k[j * SP + x];
                s += a * b;
            }
            acc[u] += s;
        }
    }
    int heads = int(gl_NumWorkGroups.y);
    for (int u = 0; u < 5; u++) {
        int it = t + u * 256;
        if (it < items) p.part.v[(ck * heads + head) * items + it] = acc[u];
    }
}
"#;

/// CA_GRAM on tensor cores: [Q; K] (the head's q and k channels, padded to 32 each)
/// times its transpose over the chunk's pixels, as 16 x 16 tiles: the four of Q Kt,
/// and the diagonal tiles of Q Qt and K Kt (whose diagonals are the squared norms).
/// Two tiles per subgroup; pixels staged 64 at a time as fp16, sums in f32.
const CA_GRAM_COOP: &str = r#"
#extension GL_KHR_cooperative_matrix : require
#extension GL_KHR_memory_scope_semantics : require
#extension GL_KHR_shader_subgroup_basic : require
#extension GL_EXT_shader_explicit_arithmetic_types_float16 : require
#extension GL_EXT_control_flow_attributes : require
#define CP 64
#define ST (CP + 8)
layout(local_size_x = SG * 4) in;
layout(push_constant) uniform P { F32 qkv; F32 part; int C, plane, hd, chunk, ih; } p;
shared float16_t s_x[64 * ST];
shared float s_o[8 * 256];
void main() {
    int lid = int(gl_LocalInvocationID.x), sg = int(gl_SubgroupID);
    int ck = int(gl_WorkGroupID.x), head = int(gl_WorkGroupID.y);
    int qc = head * p.hd, p0 = ck * p.chunk, p1 = min(p0 + p.chunk, p.plane);
    // Tiles (row block, column block) of the 64 x 64 product; rows 0-31 q, 32-63 k.
    const ivec2 T[8] = ivec2[8](ivec2(0, 2), ivec2(0, 3), ivec2(1, 2), ivec2(1, 3), ivec2(0, 0), ivec2(1, 1), ivec2(2, 2), ivec2(3, 3));
    coopmat<float, gl_ScopeSubgroup, 16, 16, gl_MatrixUseAccumulator> acc[2];
    acc[0] = coopmat<float, gl_ScopeSubgroup, 16, 16, gl_MatrixUseAccumulator>(0.0);
    acc[1] = coopmat<float, gl_ScopeSubgroup, 16, 16, gl_MatrixUseAccumulator>(0.0);
    for (int s0 = p0; s0 < p1; s0 += CP) {
        barrier();
        for (int e = lid; e < 64 * CP; e += SG * 4) {
            int row = e / CP, j = e - row * CP, px = s0 + j, d = row & 31;
            float v = 0.0;
            if (d < p.hd && px < p1) v = ld(p.qkv, ((row < 32 ? 0 : p.C) + qc + d) * p.plane + px, p.ih != 0);
            s_x[row * ST + j] = float16_t(v);
        }
        barrier();
        [[unroll]] for (int u = 0; u < 2; u++) {
            ivec2 tl = T[sg * 2 + u];
            [[unroll]] for (int kk = 0; kk < CP / 16; kk++) {
                coopmat<float16_t, gl_ScopeSubgroup, 16, 16, gl_MatrixUseA> a;
                coopmat<float16_t, gl_ScopeSubgroup, 16, 16, gl_MatrixUseB> b;
                coopMatLoad(a, s_x, tl.x * 16 * ST + kk * 16, ST, gl_CooperativeMatrixLayoutRowMajor);
                coopMatLoad(b, s_x, tl.y * 16 * ST + kk * 16, ST, gl_CooperativeMatrixLayoutColumnMajor);
                acc[u] = coopMatMulAdd(a, b, acc[u]);
            }
        }
    }
    [[unroll]] for (int u = 0; u < 2; u++) coopMatStore(acc[u], s_o, (sg * 2 + u) * 256, 16, gl_CooperativeMatrixLayoutRowMajor);
    barrier();
    int items = p.hd * p.hd + 2 * p.hd, heads = int(gl_NumWorkGroups.y);
    for (int it = lid; it < items; it += SG * 4) {
        float v;
        if (it < p.hd * p.hd) {
            int i = it / p.hd, j = it - i * p.hd;
            v = s_o[((i >> 4) * 2 + (j >> 4)) * 256 + (i & 15) * 16 + (j & 15)];
        } else {
            // q norms, then k norms: diagonals of tiles 4, 5 and 6, 7.
            int k = it - p.hd * p.hd, kq = k < p.hd ? 0 : 1, i = k - kq * p.hd;
            v = s_o[(4 + kq * 2 + (i >> 4)) * 256 + (i & 15) * 17];
        }
        p.part.v[(ck * heads + head) * items + it] = v;
    }
}
"#;

/// Depthwise 3x3 convolution (stride 1, padding 1): four outputs along a row per
/// invocation, from 3 rows of 6 inputs. A second input, if given, multiplies the result
/// (DAT's gated feed-forward).
const DW3: &str = r#"
layout(local_size_x = 16, local_size_y = 8, local_size_z = 1) in;
layout(push_constant) uniform P { F32 inp; F32 wt; F32 bias; F32 outp; F32 mul; int C, H, W, act_type, has_bias; float a0, a1; int ih, oh, has_mul, mh; } p;
void main() {
    int x0 = int(gl_GlobalInvocationID.x) * 4, y = int(gl_GlobalInvocationID.y), c = int(gl_GlobalInvocationID.z);
    if (x0 >= p.W || y >= p.H || c >= p.C) return;
    float w[9];
    for (int k = 0; k < 9; k++) w[k] = p.wt.v[c * 9 + k];
    float b = p.has_bias != 0 ? p.bias.v[c] : 0.0;
    float o[4] = float[4](b, b, b, b);
    int base = c * p.H * p.W;
    bool ih = p.ih != 0;
    for (int ky = 0; ky < 3; ky++) {
        int iy = y + ky - 1;
        if (iy < 0 || iy >= p.H) continue;
        float r[6];
        for (int i = 0; i < 6; i++) {
            int ix = x0 - 1 + i;
            r[i] = ix >= 0 && ix < p.W ? ld(p.inp, base + iy * p.W + ix, ih) : 0.0;
        }
        for (int q = 0; q < 4; q++) o[q] += r[q] * w[ky * 3] + r[q + 1] * w[ky * 3 + 1] + r[q + 2] * w[ky * 3 + 2];
    }
    for (int q = 0; q < 4; q++)
        if (x0 + q < p.W) {
            int i = base + y * p.W + x0 + q;
            float v = act(o[q], p.act_type, p.a0, p.a1);
            if (p.has_mul != 0) v *= ld(p.mul, i, p.mh != 0);
            st(p.outp, i, p.oh != 0, v);
        }
}
"#;

/// Step 2: per head, the chunks summed, then attn[i][j] = softmax over j of
/// temperature * q_i.k_j / (|q_i| |k_j|) (F.normalize: norms at least 1e-12).
const CA_SOFTMAX: &str = r#"
layout(local_size_x = 256) in;
layout(push_constant) uniform P { F32 part; F32 temp; F32 attn; int hd, chunks, heads; } p;
// The chunks' sums, all items at once (consecutive threads, consecutive items).
shared float s_g[32 * 32 + 64];
void main() {
    int t = int(gl_LocalInvocationID.x), head = int(gl_WorkGroupID.x);
    int items = p.hd * p.hd + 2 * p.hd;
    for (int it = t; it < items; it += 256) {
        float s = 0.0;
        for (int c = 0; c < p.chunks; c++) s += p.part.v[(c * p.heads + head) * items + it];
        s_g[it] = s;
    }
    barrier();
    // Row i: its 30 scores, then their softmax.
    int i = t;
    if (i >= p.hd) return;
    float qn = max(sqrt(s_g[p.hd * p.hd + i]), 1e-12);
    float v[32];
    float mx = -1e30;
    for (int j = 0; j < p.hd; j++) {
        v[j] = s_g[i * p.hd + j] / (qn * max(sqrt(s_g[p.hd * p.hd + p.hd + j]), 1e-12)) * p.temp.v[head];
        mx = max(mx, v[j]);
    }
    float sum = 0.0;
    for (int j = 0; j < p.hd; j++) { v[j] = exp(v[j] - mx); sum += v[j]; }
    for (int j = 0; j < p.hd; j++) p.attn.v[(head * p.hd + i) * p.hd + j] = v[j] / sum;
}
"#;

/// CA_APPLY on tensor cores: the head's attention (fp16, padded to 32 x 32) times its
/// v channels for 64 pixels; a subgroup per 16 pixels.
const CA_APPLY_COOP: &str = r#"
#extension GL_KHR_cooperative_matrix : require
#extension GL_KHR_memory_scope_semantics : require
#extension GL_KHR_shader_subgroup_basic : require
#extension GL_EXT_shader_explicit_arithmetic_types_float16 : require
#extension GL_EXT_control_flow_attributes : require
#define TP 64
#define VS (TP + 8)
layout(local_size_x = SG * 4) in;
layout(push_constant) uniform P { F32 qkv; F32 attn; F32 o; int C, plane, hd, ih, oh; } p;
shared float16_t s_a[32 * 32];
shared float16_t s_v[32 * VS];
shared float s_o[32 * TP];
void main() {
    int lid = int(gl_LocalInvocationID.x), sg = int(gl_SubgroupID);
    int p0 = int(gl_WorkGroupID.x) * TP, head = int(gl_WorkGroupID.y);
    for (int e = lid; e < 32 * 32; e += SG * 4) {
        int i = e / 32, j = e - i * 32;
        s_a[e] = float16_t(i < p.hd && j < p.hd ? p.attn.v[(head * p.hd + i) * p.hd + j] : 0.0);
    }
    for (int e = lid; e < 32 * TP; e += SG * 4) {
        int d = e / TP, x = e - d * TP, px = p0 + x;
        float v = d < p.hd && px < p.plane ? ld(p.qkv, (2 * p.C + head * p.hd + d) * p.plane + px, p.ih != 0) : 0.0;
        s_v[d * VS + x] = float16_t(v);
    }
    barrier();
    [[unroll]] for (int mt = 0; mt < 2; mt++) {
        coopmat<float, gl_ScopeSubgroup, 16, 16, gl_MatrixUseAccumulator> acc = coopmat<float, gl_ScopeSubgroup, 16, 16, gl_MatrixUseAccumulator>(0.0);
        [[unroll]] for (int kk = 0; kk < 2; kk++) {
            coopmat<float16_t, gl_ScopeSubgroup, 16, 16, gl_MatrixUseA> a;
            coopmat<float16_t, gl_ScopeSubgroup, 16, 16, gl_MatrixUseB> b;
            coopMatLoad(a, s_a, mt * 16 * 32 + kk * 16, 32, gl_CooperativeMatrixLayoutRowMajor);
            coopMatLoad(b, s_v, kk * 16 * VS + sg * 16, VS, gl_CooperativeMatrixLayoutRowMajor);
            acc = coopMatMulAdd(a, b, acc);
        }
        coopMatStore(acc, s_o, mt * 16 * TP + sg * 16, TP, gl_CooperativeMatrixLayoutRowMajor);
    }
    barrier();
    for (int e = lid; e < p.hd * TP; e += SG * 4) {
        int i = e / TP, x = e - i * TP, px = p0 + x;
        if (px < p.plane) st(p.o, (head * p.hd + i) * p.plane + px, p.oh != 0, s_o[i * TP + x]);
    }
}
"#;

/// Step 3: out[head hd + i][px] = sum over j of attn[head][i][j] v[head hd + j][px].
const CA_APPLY: &str = r#"
layout(local_size_x = 256) in;
layout(push_constant) uniform P { F32 qkv; F32 attn; F32 o; int C, plane, hd, ih, oh; } p;
#define HD_MAX 32
shared float s_a[HD_MAX * HD_MAX];
void main() {
    int t = int(gl_LocalInvocationID.x), head = int(gl_WorkGroupID.y);
    int px = int(gl_WorkGroupID.x) * 256 + t;
    for (int e = t; e < p.hd * p.hd; e += 256) s_a[e] = p.attn.v[head * p.hd * p.hd + e];
    barrier();
    if (px >= p.plane) return;
    float v[HD_MAX];
    for (int j = 0; j < HD_MAX; j++) v[j] = j < p.hd ? ld(p.qkv, (2 * p.C + head * p.hd + j) * p.plane + px, p.ih != 0) : 0.0;
    for (int i = 0; i < p.hd; i++) {
        float s = 0.0;
        for (int j = 0; j < p.hd; j++) s += s_a[i * p.hd + j] * v[j];
        st(p.o, (head * p.hd + i) * p.plane + px, p.oh != 0, s);
    }
}
"#;

/// DAT's adaptive interaction between its attention and convolution branches: mode 0
/// (spatial blocks) att * sigmoid(cmap[c]) + conv * sigmoid(smap[px]); mode 1
/// (channel blocks) att * sigmoid(smap[px]) + conv * sigmoid(cmap[c]).
const AIM: &str = r#"
layout(local_size_x = 256) in;
layout(push_constant) uniform P { F32 att; F32 conv; F32 cmap; F32 smap; F32 o; int n, plane, mode, ah, ch, sh, oh; } p;
void main() {
    int i = int(gl_GlobalInvocationID.x + gl_GlobalInvocationID.y * gl_NumWorkGroups.x * gl_WorkGroupSize.x);
    if (i >= p.n) return;
    int c = i / p.plane, px = i - c * p.plane;
    float gc = 1.0 / (1.0 + exp(-p.cmap.v[c]));
    float gs = 1.0 / (1.0 + exp(-ld(p.smap, px, p.sh != 0)));
    float a = ld(p.att, i, p.ah != 0), v = ld(p.conv, i, p.ch != 0);
    st(p.o, i, p.oh != 0, p.mode == 0 ? a * gc + v * gs : a * gs + v * gc);
}
"#;

/// Global average, step 2: each channel's chunk sums (GLOBAL_SUM), in order.
const GLOBAL_AVG: &str = r#"
layout(local_size_x = 64) in;
layout(push_constant) uniform P { F32 part; F32 o; int C, plane, chunks; } p;
void main() {
    int c = int(gl_GlobalInvocationID.x);
    if (c >= p.C) return;
    float acc = 0.0;
    for (int k = 0; k < p.chunks; k++) acc += p.part.v[c * p.chunks + k];
    p.o.v[c] = acc / float(p.plane);
}
"#;

/// Transposed convolution 4x4, stride 2 (RIFE's upsampling), same result as DECONV:
/// an output pixel's parity picks its 4 taps (2 rows x 2 columns of the kernel), so
/// no tap is tried and skipped. Each invocation makes 8 output channels; weights are
/// rearranged at load time to [ic][tap][oc (padded to 8)], two 16-byte reads per tap.
const DECONV_S2: &str = r#"
layout(buffer_reference, std430, buffer_reference_align = 16) buffer V4 { vec4 v[]; };
layout(local_size_x = 16, local_size_y = 8, local_size_z = 1) in;
layout(push_constant) uniform P {
    F32 inp; V4 wt; F32 bias; F32 outp;
    int C, H, W, OC, OH, OW, pl, pt, OCP, act_type, has_bias;
    float a0, a1;
} p;
void main() {
    int x = int(gl_GlobalInvocationID.x), y = int(gl_GlobalInvocationID.y), oc0 = int(gl_GlobalInvocationID.z) * 8;
    if (x >= p.OW || y >= p.OH) return;
    int bx = x + p.pl, by = y + p.pt;
    int kx0 = bx & 1, ky0 = by & 1;
    // Tap kx0 reads column sx0, tap kx0 + 2 column sx0 - 1 (rows alike).
    int sx0 = (bx - kx0) >> 1, sy0 = (by - ky0) >> 1;
    bool cx[2] = bool[2](sx0 < p.W, sx0 >= 1 && sx0 - 1 < p.W);
    bool cy[2] = bool[2](sy0 < p.H, sy0 >= 1 && sy0 - 1 < p.H);
    vec4 lo = vec4(0.0), hi = vec4(0.0);
    if (p.has_bias != 0) {
        for (int o = 0; o < 4; o++) {
            if (oc0 + o < p.OC) lo[o] = p.bias.v[oc0 + o];
            if (oc0 + 4 + o < p.OC) hi[o] = p.bias.v[oc0 + 4 + o];
        }
    }
    int plane = p.H * p.W, q = p.OCP / 4;
    for (int ic = 0; ic < p.C; ic++) {
        int ib = ic * plane;
        for (int j = 0; j < 2; j++) {
            if (!cy[j]) continue;
            int row = ib + (sy0 - j) * p.W, ky = ky0 + 2 * j;
            for (int i = 0; i < 2; i++) {
                if (!cx[i]) continue;
                float v = p.inp.v[row + sx0 - i];
                int w = ((ic * 16 + ky * 4 + kx0 + 2 * i) * q) + oc0 / 4;
                lo += v * p.wt.v[w];
                hi += v * p.wt.v[w + 1];
            }
        }
    }
    for (int o = 0; o < 8; o++) {
        int oc = oc0 + o;
        if (oc < p.OC) p.outp.v[(oc * p.OH + y) * p.OW + x] = act(o < 4 ? lo[o] : hi[o - 4], p.act_type, p.a0, p.a1);
    }
}
"#;

/// Grouped (and depthwise) convolution: one output value per invocation.
const CONV_GROUP: &str = r#"
layout(local_size_x = 16, local_size_y = 8, local_size_z = 1) in;
layout(push_constant) uniform P {
    F32 inp; F32 wt; F32 bias; F32 outp;
    int C, H, W, OC, OH, OW, kw, kh, sw, sh, dw, dh, pl, pt, act_type, has_bias;
    float a0, a1; int G, ih, oh;
} p;
void main() {
    int x = int(gl_GlobalInvocationID.x), y = int(gl_GlobalInvocationID.y), oc = int(gl_GlobalInvocationID.z);
    if (x >= p.OW || y >= p.OH || oc >= p.OC) return;
    int icg = p.C / p.G, ocg = p.OC / p.G, g = oc / ocg;
    bool ih = p.ih != 0;
    int KK = p.kw * p.kh;
    float acc = p.has_bias != 0 ? p.bias.v[oc] : 0.0;
    int ix0 = x * p.sw - p.pl, iy0 = y * p.sh - p.pt;
    for (int i = 0; i < icg; i++) {
        int ic = g * icg + i;
        int ib = ic * p.H * p.W;
        for (int ky = 0; ky < p.kh; ky++) {
            int iy = iy0 + ky * p.dh;
            if (iy < 0 || iy >= p.H) continue;
            for (int kx = 0; kx < p.kw; kx++) {
                int ix = ix0 + kx * p.dw;
                if (ix < 0 || ix >= p.W) continue;
                acc += ld(p.inp, ib + iy * p.W + ix, ih) * p.wt.v[(oc * icg + i) * KK + ky * p.kw + kx];
            }
        }
    }
    st(p.outp, (oc * p.OH + y) * p.OW + x, p.oh != 0, act(acc, p.act_type, p.a0, p.a1));
}
"#;

/// Transposed convolution: input pixel i adds weight k at output i*stride + k*dilation
/// (ncnn's form, kernel not flipped), as a gather; then the padding is cut.
const DECONV: &str = r#"
layout(local_size_x = 16, local_size_y = 8, local_size_z = 1) in;
layout(push_constant) uniform P {
    F32 inp; F32 wt; F32 bias; F32 outp;
    int C, H, W, OC, OH, OW, kw, kh, sw, sh, dw, dh, pl, pt, act_type, has_bias;
    float a0, a1;
} p;
void main() {
    int x = int(gl_GlobalInvocationID.x), y = int(gl_GlobalInvocationID.y), oc = int(gl_GlobalInvocationID.z);
    if (x >= p.OW || y >= p.OH || oc >= p.OC) return;
    int KK = p.kw * p.kh;

    int bx = x + p.pl, by = y + p.pt;
    float acc = p.has_bias != 0 ? p.bias.v[oc] : 0.0;
    for (int ic = 0; ic < p.C; ic++) {
        int ib = ic * p.H * p.W;
        int wb = (oc * p.C + ic) * KK;
        for (int ky = 0; ky < p.kh; ky++) {
            int sys = by - ky * p.dh;
            if (sys < 0 || sys % p.sh != 0) continue;
            int sy = sys / p.sh;
            if (sy >= p.H) continue;
            for (int kx = 0; kx < p.kw; kx++) {
                int sxs = bx - kx * p.dw;
                if (sxs < 0 || sxs % p.sw != 0) continue;
                int sx = sxs / p.sw;
                if (sx >= p.W) continue;
                acc += p.inp.v[ib + sy * p.W + sx] * p.wt.v[wb + ky * p.kw + kx];
            }
        }
    }
    p.outp.v[(oc * p.OH + y) * p.OW + x] = act(acc, p.act_type, p.a0, p.a1);
}
"#;

const INNER: &str = r#"
layout(local_size_x = 64) in;
layout(push_constant) uniform P { F32 inp; F32 wt; F32 bias; F32 outp; int N, OC, act_type, has_bias; float a0, a1; } p;
void main() {
    int o = int(gl_GlobalInvocationID.x);
    if (o >= p.OC) return;
    float acc = p.has_bias != 0 ? p.bias.v[o] : 0.0;
    for (int i = 0; i < p.N; i++) acc += p.inp.v[i] * p.wt.v[o * p.N + i];
    p.outp.v[o] = act(acc, p.act_type, p.a0, p.a1);
}
"#;

/// Element-wise a (op) b with broadcasting of size-1 dimensions; `b` may be a scalar.
/// Loads and stores of tensors that may be f32 or f16. Kernels on hot paths pass
/// compile-time constants (variants per element type: a branch per load would
/// serialize the loads). Stores to f16 saturate at its largest value.
const HALF: &str = r#"
#extension GL_EXT_shader_explicit_arithmetic_types_float16 : require
layout(buffer_reference, scalar, buffer_reference_align = 2) buffer F16S { float16_t v[]; };
#ifndef IN16
#define IN16 0
#endif
#ifndef OUT16
#define OUT16 0
#endif
#ifndef RES16
#define RES16 0
#endif
#ifndef A16
#define A16 0
#endif
#ifndef B16
#define B16 0
#endif
#ifndef O16
#define O16 0
#endif
float ld(F32 b, int i, bool h) { return h ? float(F16S(uint64_t(b)).v[i]) : b.v[i]; }
void st(F32 b, int i, bool h, float v) {
    if (h) F16S(uint64_t(b)).v[i] = float16_t(clamp(v, -65504.0, 65504.0)); else b.v[i] = v;
}
"#;

/// f16 <-> f32 (`to_half`: f32 -> f16).
const CAST: &str = r#"
layout(local_size_x = 256) in;
layout(push_constant) uniform P { F32 a; F32 o; int n, to_half; } p;
void main() {
    int i = int(gl_GlobalInvocationID.x + gl_GlobalInvocationID.y * gl_NumWorkGroups.x * gl_WorkGroupSize.x);
    if (i >= p.n) return;
    st(p.o, i, p.to_half != 0, ld(p.a, i, p.to_half == 0));
}
"#;

/// Element-wise ops with broadcasting. A16, B16, O16 (compiled in) make `a`, `b`,
/// the output f16.
const BINARY: &str = r#"
layout(local_size_x = 16, local_size_y = 8, local_size_z = 1) in;
layout(push_constant) uniform P {
    F32 a; F32 b; F32 o;
    int C, H, W, ac, ah, aw, bc, bh, bw, op, scalar;
    float s, ka, kb;
} p;
void main() {
    int x = int(gl_GlobalInvocationID.x), y = int(gl_GlobalInvocationID.y), c = int(gl_GlobalInvocationID.z);
    if (x >= p.W || y >= p.H || c >= p.C) return;
    float va = ld(p.a, ((p.ac == 1 ? 0 : c) * p.ah + (p.ah == 1 ? 0 : y)) * p.aw + (p.aw == 1 ? 0 : x), A16 != 0);
    float vb = p.scalar != 0 ? p.s : ld(p.b, ((p.bc == 1 ? 0 : c) * p.bh + (p.bh == 1 ? 0 : y)) * p.bw + (p.bw == 1 ? 0 : x), B16 != 0);
    float r;
    switch (p.op) {
        case 0: r = va + vb; break;
        case 1: r = va - vb; break;
        case 2: r = va * vb; break;
        case 3: r = va / vb; break;
        case 4: r = max(va, vb); break;
        case 5: r = min(va, vb); break;
        case 6: r = pow(va, vb); break;
        case 7: r = vb - va; break;
        case 8: r = vb / va; break;
        case 9: r = pow(vb, va); break;
        default: r = p.ka * va + p.kb * vb; break;
    }
    st(p.o, (c * p.H + y) * p.W + x, O16 != 0, r);
}
"#;

/// Activations: 1 (leaky) relu, 2 prelu, 3 sigmoid, 4 tanh, 5 hard sigmoid,
/// 6 hard swish, 7 clip, 8 swish (x sigmoid(x)), 9 GELU (exact, with erf).
const UNARY: &str = r#"
layout(local_size_x = 256) in;
layout(push_constant) uniform P { F32 a; F32 o; F32 slope; int n, hw, op, nslope; float a0, a1; } p;
void main() {
    int i = int(gl_GlobalInvocationID.x + gl_GlobalInvocationID.y * gl_NumWorkGroups.x * gl_WorkGroupSize.x);
    if (i >= p.n) return;
    float v = p.a.v[i], r = v;
    switch (p.op) {
        case 1: r = v > 0.0 ? v : v * p.a0; break;
        case 2: r = v > 0.0 ? v : v * p.slope.v[p.nslope > 1 ? i / p.hw : 0]; break;
        case 3: r = 1.0 / (1.0 + exp(-v)); break;
        case 4: r = tanh(v); break;
        case 5: r = clamp(v * p.a0 + p.a1, 0.0, 1.0); break;
        case 6: r = v * clamp(v * p.a0 + p.a1, 0.0, 1.0); break;
        case 7: r = clamp(v, p.a0, p.a1); break;
        case 8: r = v / (1.0 + exp(-v)); break;
        case 9: r = 0.5 * v * (1.0 + erf_(v * 0.70710678)); break;
    }
    p.o.v[i] = r;
}
"#;

/// Resize: 1 nearest, 2 bilinear (align_corner 0/1).
const INTERP: &str = r#"
layout(local_size_x = 16, local_size_y = 8, local_size_z = 1) in;
layout(push_constant) uniform P { F32 a; F32 o; int C, H, W, OH, OW, mode, align; } p;
void main() {
    int x = int(gl_GlobalInvocationID.x), y = int(gl_GlobalInvocationID.y), c = int(gl_GlobalInvocationID.z);
    if (x >= p.OW || y >= p.OH || c >= p.C) return;
    int base = c * p.H * p.W;
    float r;
    if (p.mode == 1) {
        int sx = min(int(float(x) * float(p.W) / float(p.OW)), p.W - 1);
        int sy = min(int(float(y) * float(p.H) / float(p.OH)), p.H - 1);
        r = p.a.v[base + sy * p.W + sx];
    } else if (p.mode == 3) {
        // Bicubic as ncnn: A = -0.75, taps clamped to the edges.
        float fx = (float(x) + 0.5) * float(p.W) / float(p.OW) - 0.5;
        float fy = (float(y) + 0.5) * float(p.H) / float(p.OH) - 0.5;
        int sx = int(floor(fx)), sy = int(floor(fy));
        float tx = fx - float(sx), ty = fy - float(sy);
        const float A = -0.75;
        vec4 cx, cy;
        float u0 = tx + 1.0, u1 = tx, u2 = 1.0 - tx;
        cx.x = A * u0 * u0 * u0 - 5.0 * A * u0 * u0 + 8.0 * A * u0 - 4.0 * A;
        cx.y = (A + 2.0) * u1 * u1 * u1 - (A + 3.0) * u1 * u1 + 1.0;
        cx.z = (A + 2.0) * u2 * u2 * u2 - (A + 3.0) * u2 * u2 + 1.0;
        cx.w = 1.0 - cx.x - cx.y - cx.z;
        u0 = ty + 1.0; u1 = ty; u2 = 1.0 - ty;
        cy.x = A * u0 * u0 * u0 - 5.0 * A * u0 * u0 + 8.0 * A * u0 - 4.0 * A;
        cy.y = (A + 2.0) * u1 * u1 * u1 - (A + 3.0) * u1 * u1 + 1.0;
        cy.z = (A + 2.0) * u2 * u2 * u2 - (A + 3.0) * u2 * u2 + 1.0;
        cy.w = 1.0 - cy.x - cy.y - cy.z;
        r = 0.0;
        for (int j = 0; j < 4; j++) {
            int yy = clamp(sy - 1 + j, 0, p.H - 1);
            float row = 0.0;
            for (int i = 0; i < 4; i++) row += cx[i] * p.a.v[base + yy * p.W + clamp(sx - 1 + i, 0, p.W - 1)];
            r += cy[j] * row;
        }
    } else {
        float fx, fy;
        if (p.align != 0) {
            fx = p.OW > 1 ? float(x) * float(p.W - 1) / float(p.OW - 1) : 0.0;
            fy = p.OH > 1 ? float(y) * float(p.H - 1) / float(p.OH - 1) : 0.0;
        } else {
            fx = max((float(x) + 0.5) * float(p.W) / float(p.OW) - 0.5, 0.0);
            fy = max((float(y) + 0.5) * float(p.H) / float(p.OH) - 0.5, 0.0);
        }
        int x0 = min(int(fx), p.W - 1), y0 = min(int(fy), p.H - 1);
        int x1 = min(x0 + 1, p.W - 1), y1 = min(y0 + 1, p.H - 1);
        float ax = fx - float(x0), ay = fy - float(y0);
        float t = mix(p.a.v[base + y0 * p.W + x0], p.a.v[base + y0 * p.W + x1], ax);
        float b = mix(p.a.v[base + y1 * p.W + x0], p.a.v[base + y1 * p.W + x1], ax);
        r = mix(t, b, ay);
    }
    p.o.v[(c * p.OH + y) * p.OW + x] = r;
}
"#;

/// Every second pixel in each direction (f32 or f16, both the same).
/// The patches of a convolution, unfolded: channel (ic * KH + ky) * KW + kx holds
/// input channel ic at tap (ky, kx) for every output pixel (zero outside the input
/// and in the padding channels up to CP), f16.
const IM2COL: &str = r#"
layout(local_size_x = 64, local_size_y = 4) in;
layout(push_constant) uniform P { F32 x; F32 o; int C, H, W, OH, OW, KW, KH, SW, SH, PL, PT, CP, ih; } p;
void main() {
    int ox = int(gl_GlobalInvocationID.x), oy = int(gl_GlobalInvocationID.y), c = int(gl_GlobalInvocationID.z);
    if (ox >= p.OW || oy >= p.OH || c >= p.CP) return;
    int taps = p.KW * p.KH, i = c / taps, t = c - i * taps, ky = t / p.KW, kx = t - ky * p.KW;
    int y = oy * p.SH - p.PT + ky, x = ox * p.SW - p.PL + kx;
    float v = i < p.C && y >= 0 && y < p.H && x >= 0 && x < p.W ? ld(p.x, (i * p.H + y) * p.W + x, p.ih != 0) : 0.0;
    F16S(uint64_t(p.o)).v[(c * p.OH + oy) * p.OW + ox] = float16_t(v);
}
"#;

const SUBSAMPLE: &str = r#"
layout(local_size_x = 16, local_size_y = 8, local_size_z = 1) in;
layout(push_constant) uniform P { F32 a; F32 o; int C, H, W, OH, OW, half_; } p;
void main() {
    int x = int(gl_GlobalInvocationID.x), y = int(gl_GlobalInvocationID.y), c = int(gl_GlobalInvocationID.z);
    if (x >= p.OW || y >= p.OH || c >= p.C) return;
    bool h = p.half_ != 0;
    st(p.o, (c * p.OH + y) * p.OW + x, h, ld(p.a, (c * p.H + 2 * y) * p.W + 2 * x, h));
}
"#;

/// Copy a region: out[oc0 + c][oy0 + y][ox0 + x] = in[ic0 + c][iy0 + y][ix0 + x].
const COPY: &str = r#"
layout(local_size_x = 16, local_size_y = 8, local_size_z = 1) in;
layout(push_constant) uniform P { F32 a; F32 o; int C, H, W, iH, iW, ic0, iy0, ix0, oH, oW, oc0, oy0, ox0; } p;
void main() {
    int x = int(gl_GlobalInvocationID.x), y = int(gl_GlobalInvocationID.y), c = int(gl_GlobalInvocationID.z);
    if (x >= p.W || y >= p.H || c >= p.C) return;
    p.o.v[((p.oc0 + c) * p.oH + p.oy0 + y) * p.oW + p.ox0 + x] = p.a.v[((p.ic0 + c) * p.iH + p.iy0 + y) * p.iW + p.ix0 + x];
}
"#;

const PIXEL_SHUFFLE: &str = r#"
layout(local_size_x = 16, local_size_y = 8, local_size_z = 1) in;
layout(push_constant) uniform P { F32 a; F32 o; int OC, H, W, r; } p;
void main() {
    int x = int(gl_GlobalInvocationID.x), y = int(gl_GlobalInvocationID.y), c = int(gl_GlobalInvocationID.z);
    int OH = p.H * p.r, OW = p.W * p.r;
    if (x >= OW || y >= OH || c >= p.OC) return;
    int ic = c * p.r * p.r + (y % p.r) * p.r + (x % p.r);
    p.o.v[(c * OH + y) * OW + x] = p.a.v[(ic * p.H + y / p.r) * p.W + x / p.r];
}
"#;

/// Average pooling: global (k = 0) or k x k with stride s (no padding).
const POOL: &str = r#"
layout(local_size_x = 16, local_size_y = 8, local_size_z = 1) in;
layout(push_constant) uniform P { F32 a; F32 o; int C, H, W, OH, OW, k, s; } p;
void main() {
    int x = int(gl_GlobalInvocationID.x), y = int(gl_GlobalInvocationID.y), c = int(gl_GlobalInvocationID.z);
    if (x >= p.OW || y >= p.OH || c >= p.C) return;
    int base = c * p.H * p.W;
    float acc = 0.0;
    if (p.k == 0) {
        for (int i = 0; i < p.H * p.W; i++) acc += p.a.v[base + i];
        p.o.v[c] = acc / float(p.H * p.W);
        return;
    }
    for (int j = 0; j < p.k; j++)
        for (int i = 0; i < p.k; i++)
            acc += p.a.v[base + min(y * p.s + j, p.H - 1) * p.W + min(x * p.s + i, p.W - 1)];
    p.o.v[(c * p.OH + y) * p.OW + x] = acc / float(p.k * p.k);
}
"#;

/// Mean over channels (keeping one channel).
/// NAFNet's local average (TLC, test-time local converter): the mean over a k1 x k2
/// box, the box kept inside the picture (clamped), as the official AvgPool2d with
/// replicate padding. Pass 1: column sums along y (running window, f32).
const BOX_V: &str = r#"
layout(local_size_x = 64) in;
layout(push_constant) uniform P { F32 a; F32 t; int C, H, W, k1; int ah; } p;
void main() {
    int x = int(gl_GlobalInvocationID.x), c = int(gl_GlobalInvocationID.z);
    if (x >= p.W || c >= p.C) return;
    int base = c * p.H * p.W + x, pt = (p.k1 - 1) / 2;
    float s = 0.0;
    for (int y = 0; y < p.k1; y++) s += ld(p.a, base + y * p.W, p.ah != 0);
    int s0 = 0;
    for (int y = 0; y < p.H; y++) {
        int want = clamp(y - pt, 0, p.H - p.k1);
        while (s0 < want) { s += ld(p.a, base + (s0 + p.k1) * p.W, p.ah != 0) - ld(p.a, base + s0 * p.W, p.ah != 0); s0++; }
        p.t.v[base + y * p.W] = s;
    }
}
"#;

/// Pass 2: per row, a prefix sum in shared memory, then each pixel's k2 window.
const BOX_H: &str = r#"
layout(local_size_x = 256) in;
layout(push_constant) uniform P { F32 t; F32 o; int C, H, W, k1, k2; int oh; } p;
shared float s_p[8193];
shared float s_tot[256];
void main() {
    int y = int(gl_WorkGroupID.x), c = int(gl_WorkGroupID.y), l = int(gl_LocalInvocationID.x);
    int row = (c * p.H + y) * p.W, L = (p.W + 255) / 256, b0 = min(l * L, p.W), b1 = min(b0 + L, p.W);
    float s = 0.0;
    for (int i = b0; i < b1; i++) s += p.t.v[row + i];
    s_tot[l] = s;
    barrier();
    for (int off = 1; off < 256; off <<= 1) {
        float v = l >= off ? s_tot[l - off] : 0.0;
        barrier();
        s_tot[l] += v;
        barrier();
    }
    float run = l > 0 ? s_tot[l - 1] : 0.0;
    if (l == 0) s_p[0] = 0.0;
    for (int i = b0; i < b1; i++) { run += p.t.v[row + i]; s_p[i + 1] = run; }
    barrier();
    int pl = (p.k2 - 1) / 2;
    float inv = 1.0 / float(p.k1 * p.k2);
    for (int x = l; x < p.W; x += 256) {
        int s0 = clamp(x - pl, 0, p.W - p.k2);
        st(p.o, row + x, p.oh != 0, (s_p[s0 + p.k2] - s_p[s0]) * inv);
    }
}
"#;

/// GroupNorm (Video Depth Anything's temporal modules): one workgroup per group of
/// C / G channels (contiguous in a planar tensor); mean, then the variance of the
/// deviations, then gamma, beta per channel.
const GROUP_NORM: &str = r#"
layout(local_size_x = 256) in;
layout(push_constant) uniform P { F32 a; F32 o; F32 gb; int C, n, G; float eps; int ah, oh; } p;
shared float s_r[256];
float total(float v) {
    int l = int(gl_LocalInvocationID.x);
    s_r[l] = v;
    barrier();
    for (int s = 128; s > 0; s >>= 1) {
        if (l < s) s_r[l] += s_r[l + s];
        barrier();
    }
    float r = s_r[0];
    barrier();
    return r;
}
void main() {
    int g = int(gl_WorkGroupID.x), l = int(gl_LocalInvocationID.x);
    int cg = p.C / p.G, cnt = cg * p.n, base = g * cnt;
    float s = 0.0;
    for (int i = l; i < cnt; i += 256) s += ld(p.a, base + i, p.ah != 0);
    float mean = total(s) / float(cnt);
    float s2 = 0.0;
    for (int i = l; i < cnt; i += 256) { float v = ld(p.a, base + i, p.ah != 0) - mean; s2 += v * v; }
    float inv = inversesqrt(total(s2) / float(cnt) + p.eps);
    for (int i = l; i < cnt; i += 256) {
        int c = g * cg + i / p.n;
        st(p.o, base + i, p.oh != 0, (ld(p.a, base + i, p.ah != 0) - mean) * inv * p.gb.v[c] + p.gb.v[p.C + c]);
    }
}
"#;

const MEAN_C: &str = r#"
layout(local_size_x = 256) in;
layout(push_constant) uniform P { F32 a; F32 o; int C, HW; } p;
void main() {
    int i = int(gl_GlobalInvocationID.x + gl_GlobalInvocationID.y * gl_NumWorkGroups.x * gl_WorkGroupSize.x);
    if (i >= p.HW) return;
    float acc = 0.0;
    for (int c = 0; c < p.C; c++) acc += p.a.v[c * p.HW + i];
    p.o.v[i] = acc / float(p.C);
}
"#;

/// RIFE's warp: sample the image at (x + fx, y + fy), bilinear (flow channels 0, 1).
const WARP: &str = r#"
layout(local_size_x = 16, local_size_y = 8, local_size_z = 1) in;
layout(push_constant) uniform P { F32 img; F32 flow; F32 o; int C, H, W, FW, FHW; } p;
void main() {
    int x = int(gl_GlobalInvocationID.x), y = int(gl_GlobalInvocationID.y), c = int(gl_GlobalInvocationID.z);
    if (x >= p.W || y >= p.H || c >= p.C) return;
    int hw = p.H * p.W, i = y * p.W + x;
    // As the reference: corners clamped to the image, weights from the clamped corner
    // (outside the picture this extrapolates, and the networks are trained with it).
    int fi = y * p.FW + x;
    float sx = float(x) + p.flow.v[fi];
    float sy = float(y) + p.flow.v[p.FHW + fi];
    int fx0 = int(floor(sx)), fy0 = int(floor(sy));
    int x0 = clamp(fx0, 0, p.W - 1), y0 = clamp(fy0, 0, p.H - 1);
    int x1 = clamp(fx0 + 1, 0, p.W - 1), y1 = clamp(fy0 + 1, 0, p.H - 1);
    float ax = sx - float(x0), ay = sy - float(y0);
    int b = c * hw;
    float t = mix(p.img.v[b + y0 * p.W + x0], p.img.v[b + y0 * p.W + x1], ax);
    float u = mix(p.img.v[b + y1 * p.W + x0], p.img.v[b + y1 * p.W + x1], ax);
    p.o.v[b + i] = mix(t, u, ay);
}
"#;

struct Kernels {
    conv: Arc<Kernel>,
    conv_group: Arc<Kernel>,
    /// Tiled convolution: 64 channels x 8x8 pixels, and 32 channels x 16x8.
    conv_tiled: Arc<Kernel>,
    conv_tiled32: Arc<Kernel>,
    deconv: Arc<Kernel>,
    deconv_s2: Arc<Kernel>,
    cast: Arc<Kernel>,
    channel_norm: Arc<Kernel>,
    inner: Arc<Kernel>,
    binary: Arc<Kernel>,
    unary: Arc<Kernel>,
    interp: Arc<Kernel>,
    copy: Arc<Kernel>,
    pixel_shuffle: Arc<Kernel>,
    pool: Arc<Kernel>,
    mean_c: Arc<Kernel>,
    box_v: Arc<Kernel>,
    group_norm: Arc<Kernel>,
    box_h: Arc<Kernel>,
    warp: Arc<Kernel>,
    /// DAT: channel attention (three steps), the interaction mix, global average.
    ca_gram: Arc<Kernel>,
    ca_softmax: Arc<Kernel>,
    ca_apply: Arc<Kernel>,
    aim: Arc<Kernel>,
    global_avg: Arc<Kernel>,
    global_sum: Arc<Kernel>,
    dw3: Arc<Kernel>,
    subsample: Arc<Kernel>,
    im2col: Arc<Kernel>,
    pixel_unshuffle: Arc<Kernel>,
    canvas: Arc<Kernel>,
    tiles: Arc<Kernel>,
    deform_im2col: Arc<Kernel>,
    global_attn: Arc<Kernel>,
    /// Tensor-core 3x3 convolution, when the GPU has cooperative matrices: for more
    /// than 32 output channels, and for up to 32.
    conv_coop: Option<(CoopConv, CoopConv)>,
    /// The same by Winograd F(2x2, 3x3) (CONV_WINO).
    conv_wino: Option<(CoopConv, CoopConv)>,
}

/// A tensor-core convolution kernel and its workgroup's output block: `oc`
/// channels x 16 x `rows` pixels.
struct CoopConv {
    /// All f32.
    kernel: Arc<Kernel>,
    oc: usize,
    rows: usize,
    name: String,
    shape: (usize, usize, usize),
}

impl CoopConv {
    /// `mt` tiles of 16 channels, `rs` rows of 16 pixels per subgroup, `nsg`
    /// subgroups. `BRP_COOP64` / `BRP_COOP32` ("mt,rs,nsg") override, for tuning.
    fn new(c: &Compute, name: &str, (mt, rs, nsg): (usize, usize, usize)) -> Result<CoopConv> {
        let (mt, rs, nsg) = std::env::var(format!("BRP_{}", name.to_uppercase()))
            .ok()
            .and_then(|v| {
                let n: Vec<usize> = v.split(',').filter_map(|x| x.trim().parse().ok()).collect();
                (n.len() == 3).then(|| (n[0], n[1], n[2]))
            })
            .unwrap_or((mt, rs, nsg));
        let kernel = Self::build(c, name, (mt, rs, nsg), 0)?;
        // Winograd ("wino..."): the second number is tile rows (2 pixels each).
        let rows = if name.starts_with("wino") { 2 * rs } else { rs * nsg };
        log::debug!("nn: tensor-core convolution {name}: {} channels x 16x{rows} pixels per workgroup", mt * 16);
        Ok(CoopConv { kernel, oc: mt * 16, rows, name: name.to_string(), shape: (mt, rs, nsg) })
    }

    /// The kernel for element types `dt` (bit 0 input f16, 1 output, 2 `res`, 3 sums).
    /// Winograd always sums in f32 (ACC16 applies to the direct kernel).
    fn build(c: &Compute, name: &str, (mt, rs, nsg): (usize, usize, usize), dt: u8) -> Result<Arc<Kernel>> {
        let sg = c.ctx.subgroup_size;
        let defs = format!(
            "#define SG {sg}\n#define MT {mt}\n#define RS {rs}\n#define TH {rs}\n#define NSG {nsg}\n#define IN16 {}\n#define OUT16 {}\n#define RES16 {}\n#define ACC16 {}\n",
            dt & 1,
            dt >> 1 & 1,
            dt >> 2 & 1,
            dt >> 3 & 1
        );
        // "...k1": the same kernel for 1x1 convolutions.
        let defs = if name.ends_with("k1") { format!("{defs}#define KS 1\n") } else { defs };
        let src = if name.starts_with("wino") { CONV_WINO } else { CONV_COOP };
        c.kernel(&format!("nn_{name}_{mt}_{rs}_{nsg}_{dt}"), &format!("{PRELUDE}{defs}{HALF}{ACT}{src}"), 0)
    }
}

impl Kernels {
    fn new(c: &Compute) -> Result<Kernels> {
        let k = |name: &str, body: &str| c.kernel(&format!("nn_{name}"), &format!("{PRELUDE}{HALF}{ACT}{body}"), 0);
        Ok(Kernels {
            conv: k("conv", CONV)?,
            conv_group: k("conv_group", CONV_GROUP)?,
            conv_tiled: k("conv_tiled", &format!("#define OCG 16\n#define PXG 16\n#define BW 8\n{CONV_TILED}"))?,
            conv_tiled32: k("conv_tiled32", &format!("#define OCG 8\n#define PXG 32\n#define BW 16\n{CONV_TILED}"))?,
            deconv: k("deconv", DECONV)?,
            deconv_s2: k("deconv_s2", DECONV_S2)?,
            cast: k("cast", CAST)?,
            channel_norm: k("channel_norm", CHANNEL_NORM)?,
            inner: k("inner", INNER)?,
            binary: k("binary", BINARY)?,
            unary: k("unary", UNARY)?,
            interp: k("interp", INTERP)?,
            copy: k("copy", COPY)?,
            pixel_shuffle: k("pixel_shuffle", PIXEL_SHUFFLE)?,
            pool: k("pool", POOL)?,
            mean_c: k("mean_c", MEAN_C)?,
            box_v: k("box_v", BOX_V)?,
            group_norm: k("group_norm", GROUP_NORM)?,
            box_h: k("box_h", BOX_H)?,
            warp: k("warp", WARP)?,
            ca_gram: k("ca_gram", CA_GRAM)?,
            ca_softmax: k("ca_softmax", CA_SOFTMAX)?,
            ca_apply: k("ca_apply", CA_APPLY)?,
            aim: k("aim", AIM)?,
            global_avg: k("global_avg", GLOBAL_AVG)?,
            global_sum: k("global_sum", GLOBAL_SUM)?,
            dw3: k("dw3", DW3)?,
            subsample: k("subsample", SUBSAMPLE)?,
            im2col: k("im2col", IM2COL)?,
            pixel_unshuffle: k("pixel_unshuffle", PIXEL_UNSHUFFLE)?,
            canvas: k("canvas", CANVAS)?,
            tiles: k("tiles", TILES)?,
            deform_im2col: k("deform_im2col", DEFORM_IM2COL)?,
            global_attn: k("global_attn", GLOBAL_ATTN)?,
            conv_coop: if c.ctx.coopmat && std::env::var_os("BRP_NO_COOPMAT").is_none() {
                // 3 pixel rows per subgroup: measured best on the RTX 5070 Ti (2 or 4
                // rows, 2 or 8 subgroups, fewer channel tiles: 10-25% slower).
                let both = CoopConv::new(c, "coop64", (4, 3, 4)).and_then(|a| Ok((a, CoopConv::new(c, "coop32", (2, 3, 4))?)));
                match both {
                    Ok(pair) => Some(pair),
                    Err(e) => {
                        log::warn!("tensor-core convolution unavailable: {e:#}");
                        None
                    }
                }
            } else {
                None
            },
            conv_wino: if c.ctx.coopmat && std::env::var_os("BRP_NO_COOPMAT").is_none() && wino_enabled() {
                // (channel tiles, tile rows, subgroups): 64 x 16x8 and 32 x 16x16 pixels.
                let both = CoopConv::new(c, "wino64", (4, 4, 8)).and_then(|a| Ok((a, CoopConv::new(c, "wino32", (2, 8, 8))?)));
                match both {
                    Ok(pair) => Some(pair),
                    Err(e) => {
                        log::warn!("Winograd convolution unavailable: {e:#}");
                        None
                    }
                }
            } else {
                None
            },
        })
    }
}

// ---- model files -------------------------------------------------------------------

/// Layer parameters: `id=value` and arrays `-233xx=n,v1,v2...`.
#[derive(Default, Debug, Clone)]
struct Params(HashMap<i32, String>);

impl Params {
    fn parse(tokens: &[&str]) -> Params {
        let mut m = HashMap::new();
        for t in tokens {
            if let Some((k, v)) = t.split_once('=') {
                if let Ok(k) = k.parse::<i32>() {
                    // Arrays: -23300 - id.
                    let id = if k <= -23300 { -k - 23300 } else { k };
                    m.insert(id, v.to_string());
                }
            }
        }
        Params(m)
    }
    fn i(&self, id: i32, d: i32) -> i32 {
        self.0.get(&id).and_then(|v| v.parse::<f64>().ok()).map_or(d, |v| v as i32)
    }
    fn f(&self, id: i32, d: f32) -> f32 {
        self.0.get(&id).and_then(|v| v.parse().ok()).unwrap_or(d)
    }
    /// Array values (without the leading count).
    fn arr(&self, id: i32) -> Vec<f64> {
        self.0.get(&id).map_or(Vec::new(), |v| v.split(',').skip(1).filter_map(|x| x.parse().ok()).collect())
    }
}

#[derive(Debug, Clone)]
struct Layer {
    ty: String,
    name: String,
    inputs: Vec<usize>,
    outputs: Vec<usize>,
    p: Params,
    /// Byte offsets of weight and bias (or other data) in the weight buffer.
    weight: u64,
    bias: Option<u64>,
    /// fp16 weights for the tensor-core convolution: offset in `weights16`, input
    /// channels padded to 16.
    w16: Option<(u64, i32)>,
    /// Larger kernels over few input channels (a 7x7 stem): run as a 1x1 tensor-core
    /// convolution over the unfolded patches (IM2COL); `w16` then holds 1x1 weights.
    im2col: bool,
    /// Layers merged into this (tensor-core) convolution's output step.
    fuse: Fuse,
    /// 4x4 stride-2 transposed convolution: rearranged weights (offset in the weight
    /// buffer) and output channels padded to 8.
    wdeconv: Option<(u64, usize)>,
    /// Winograd weights (CONV_WINO): offset in `weights16`, [16][oc padded to 64][ic
    /// padded to 16].
    wino: Option<u64>,
}

/// Winograd convolutions (CONV_WINO) on request (`BRP_NN_WINO`): on the RTX 5070 Ti
/// they are 10-35% slower than the direct kernel (see DESIGN.md).
fn wino_enabled() -> bool {
    std::env::var_os("BRP_NN_WINO").is_some()
}

/// See `fuse_layers`.
#[derive(Clone, Copy, Debug, Default)]
struct Fuse {
    /// PReLU: slope offset in the weight buffer, and whether there is one per channel.
    prelu: Option<(u64, bool)>,
    /// Output = rc0 * convolution + rc1 * the layer's second input.
    res: Option<(f32, f32)>,
}

/// Merges into each tensor-core convolution the layer that alone reads its output
/// when that is a PReLU or a two-input Eltwise sum: the convolution then writes that
/// layer's output itself, one pass over the activations less each.
fn fuse_layers(layers: &mut Vec<Layer>, nblobs: usize) {
    let mut gone = vec![false; layers.len()];
    let mut producer: Vec<Option<usize>> = vec![None; nblobs];
    let mut readers = vec![0usize; nblobs];
    for (i, l) in layers.iter().enumerate() {
        for &o in &l.outputs {
            producer[o] = Some(i);
        }
        for &b in &l.inputs {
            readers[b] += 1;
        }
    }
    let mut fused = 0;
    for i in 0..layers.len() {
        if layers[i].ty != "Convolution" || layers[i].w16.is_none() {
            continue;
        }
        loop {
            if layers[i].outputs.len() != 1 {
                break;
            }
            let out = layers[i].outputs[0];
            if readers[out] != 1 {
                break;
            }
            let Some(j) = (i + 1..layers.len()).find(|&j| !gone[j] && layers[j].inputs.contains(&out)) else { break };
            let (r, f) = (&layers[j], layers[i].fuse);
            // An f32-output flag (param 30) must agree.
            if r.p.i(30, 0) != layers[i].p.i(30, 0) {
                break;
            }
            // A stride-2 3x3 runs at stride 1 before subsampling: nothing fuses after it
            // (a 1x1 subsamples its input first, so it can).
            if layers[i].p.i(3, 1) != 1 && layers[i].p.i(1, 0) != 1 {
                break;
            }
            if r.outputs.len() != 1 {
                break;
            }
            let o = r.outputs[0];
            if r.ty == "PReLU" && f.prelu.is_none() && f.res.is_none() {
                let prelu = (r.weight, r.p.i(0, 1) > 1);
                layers[i].fuse.prelu = Some(prelu);
            } else if r.ty == "GELU" && f.prelu.is_none() && f.res.is_none() && layers[i].p.i(9, 0) == 0 {
                // As the convolution's own activation (applied before PReLU and sums).
                layers[i].p.0.insert(9, "9".into());
            } else if (r.ty == "Eltwise" && r.p.i(0, 1) == 1 || r.ty == "BinaryOp" && r.p.i(0, 0) == 0 && r.p.i(1, 0) == 0 && r.p.i(20, 0) == 0) && r.inputs.len() == 2 && f.res.is_none() {
                // An Eltwise sum (with its coefficients), or a BinaryOp add of two
                // tensors (not a broadcast constant: MemoryData; sizes are checked
                // when it runs). BinaryOp param 20=1 (ours): a broadcast, never fused.
                let coef = if r.ty == "Eltwise" { r.p.arr(1) } else { Vec::new() };
                let (ka, kb) = (coef.first().copied().unwrap_or(1.0) as f32, coef.get(1).copied().unwrap_or(1.0) as f32);
                let o2 = if r.inputs[0] == out { r.inputs[1] } else { r.inputs[0] };
                if producer[o2].is_some_and(|p| layers[p].ty == "MemoryData") {
                    break;
                }
                let (other, rc0, rc1) = if r.inputs[0] == out { (r.inputs[1], ka, kb) } else { (r.inputs[0], kb, ka) };
                // The other input must exist when the convolution runs.
                if other == out || producer[other].is_some_and(|p| p >= i) {
                    break;
                }
                layers[i].fuse.res = Some((rc0, rc1));
                layers[i].inputs.push(other);
            } else {
                break;
            }
            layers[i].outputs = vec![o];
            producer[o] = Some(i);
            gone[j] = true;
            fused += 1;
        }
    }
    let mut idx = 0;
    layers.retain(|_| {
        idx += 1;
        !gone[idx - 1]
    });
    if fused > 0 {
        log::debug!("nn: {fused} layers fused into convolutions");
    }
}

/// Reads ncnn's model data: flagged weight arrays (fp32 / fp16) and raw arrays.
struct Bin<'a> {
    data: &'a [u8],
    pos: usize,
}

impl Bin<'_> {
    fn raw(&mut self, n: usize) -> Result<Vec<f32>> {
        let b = self.data.get(self.pos..self.pos + n * 4).context("model data too short")?;
        self.pos += n * 4;
        Ok(b.chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect())
    }

    /// An array with a 4-byte type tag (0: fp32, 0x01306B47: fp16).
    fn flagged(&mut self, n: usize) -> Result<Vec<f32>> {
        let tag = u32::from_le_bytes(self.data.get(self.pos..self.pos + 4).context("model data too short")?.try_into()?);
        self.pos += 4;
        match tag {
            0 => self.raw(n),
            0x01306B47 => {
                let b = self.data.get(self.pos..self.pos + n * 2).context("model data too short")?;
                self.pos += (n * 2).div_ceil(4) * 4;
                Ok(b.chunks_exact(2).map(|c| f16_to_f32(u16::from_le_bytes([c[0], c[1]]))).collect())
            }
            t => bail!("unsupported weight encoding 0x{t:08x}"),
        }
    }
}

fn f16_to_f32(h: u16) -> f32 {
    let s = if h & 0x8000 != 0 { -1.0 } else { 1.0 };
    let e = ((h >> 10) & 0x1f) as i32;
    let m = (h & 0x3ff) as f32;
    match e {
        0 => s * m * 2f32.powi(-24),
        31 => {
            if m == 0.0 {
                s * f32::INFINITY
            } else {
                f32::NAN
            }
        }
        _ => s * (1.0 + m / 1024.0) * 2f32.powi(e - 15),
    }
}

/// Layers recorded by all nets, and the most layers per submission (`Net::set_flush_limit`).
static LAYERS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
static FLUSH_LIMIT: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);

/// A network loaded on the GPU.
pub struct Net {
    compute: Arc<Compute>,
    k: Kernels,
    layers: Vec<Layer>,
    blobs: HashMap<String, usize>,
    /// Layers reading each blob (to free it after the last one).
    readers: Vec<usize>,
    weights: Buffer,
    weights16: Buffer,
    pub pool: Arc<Pool>,
    /// Concat planning (see `plan_concats`): blob -> (group, channel offset), and
    /// group sizes in channels.
    place: HashMap<usize, (usize, usize)>,
    group_channels: Vec<usize>,
    /// Per run: each group's buffer (with its h, w), and the slot the layer being
    /// recorded writes its output into.
    run: Mutex<RunState>,
    /// Activations between layers stored as f16 (see `set_half`).
    half: bool,
    /// Tensor-core products summed in f16 (see `set_fp16_sums`).
    acc16: bool,
    /// Kernels compiled for f16 element types: (kernel, types) -> variant.
    variants: Mutex<HashMap<(usize, u8), Arc<Kernel>>>,
    /// Submit after every this many layers (0: never): see `set_flush_every`.
    flush_every: std::sync::atomic::AtomicUsize,
}

#[derive(Default)]
struct RunState {
    groups: Vec<Option<(Arc<Buffer>, usize, usize, bool)>>,
    pending: Option<(usize, usize)>,
}

/// Dense blocks join a growing list of maps again and again (cat(x, x1),
/// cat(x, x1, x2), ...), each a prefix of the next. Plan one buffer per such chain:
/// every member gets a fixed channel slot and is written there by the layer that
/// makes it, so the joins are views, not copies.
fn plan_concats(layers: &[Layer], nblobs: usize) -> (HashMap<usize, (usize, usize)>, Vec<usize>) {
    // Split outputs are their input.
    let mut root: Vec<usize> = (0..nblobs).collect();
    let mut producer: Vec<Option<usize>> = vec![None; nblobs];
    let mut chans: Vec<Option<usize>> = vec![None; nblobs];
    for (li, l) in layers.iter().enumerate() {
        let p = &l.p;
        let inc = |i: usize| l.inputs.get(i).and_then(|b| chans[*b]);
        let c = match l.ty.as_str() {
            "Convolution" | "ConvolutionDepthWise" | "Deconvolution" | "InnerProduct" => Some(p.i(0, 0) as usize),
            "Split" | "ReLU" | "PReLU" | "Sigmoid" | "TanH" | "HardSigmoid" | "HardSwish" | "Swish" | "GELU" | "Clip" | "Interp" | "ChannelNorm" => inc(0),
            "WindowAttention" => Some(p.i(0, 0) as usize),
            "Eltwise" | "BinaryOp" => l.inputs.iter().map(|b| chans[*b]).collect::<Option<Vec<_>>>().map(|v| v.into_iter().max().unwrap_or(0)),
            "Concat" if p.i(0, 0) == 0 => l.inputs.iter().map(|b| chans[*b]).sum(),
            _ => None,
        };
        for &o in &l.outputs {
            chans[o] = c;
            if l.ty == "Split" {
                root[o] = root[l.inputs[0]];
            } else {
                producer[o] = Some(li);
            }
        }
    }
    let mut groups: Vec<Vec<usize>> = Vec::new();
    let mut owner: HashMap<usize, usize> = HashMap::new();
    for l in layers.iter().filter(|l| l.ty == "Concat" && l.p.i(0, 0) == 0) {
        let srcs: Vec<usize> = l.inputs.iter().map(|b| root[*b]).collect();
        let placeable = srcs.iter().all(|s| {
            chans[*s].is_some() && producer[*s].is_some_and(|li| layers[li].outputs.len() == 1 && !matches!(layers[li].ty.as_str(), "Input" | "MemoryData" | "Split" | "Concat"))
        });
        let mut uniq = srcs.clone();
        uniq.sort_unstable();
        uniq.dedup();
        if !placeable || uniq.len() != srcs.len() {
            continue;
        }
        let group = srcs.first().and_then(|s| owner.get(s)).copied();
        match group {
            Some(g) => {
                let gs = &groups[g];
                let n = gs.len().min(srcs.len());
                if gs[..n] != srcs[..n] || srcs[n..].iter().any(|s| owner.contains_key(s)) {
                    continue;
                }
                for &s in &srcs[n..] {
                    owner.insert(s, g);
                }
                groups[g].extend_from_slice(&srcs[n..]);
            }
            None if srcs.iter().all(|s| !owner.contains_key(s)) => {
                for &s in &srcs {
                    owner.insert(s, groups.len());
                }
                groups.push(srcs);
            }
            None => {}
        }
    }
    let mut place = HashMap::new();
    let mut sizes = Vec::new();
    for (g, srcs) in groups.iter().enumerate() {
        let mut off = 0;
        for &s in srcs {
            place.insert(s, (g, off));
            off += chans[s].unwrap_or(0);
        }
        sizes.push(off);
    }
    (place, sizes)
}

impl Net {
    pub fn load(compute: &Arc<Compute>, param: &str, bin: &[u8]) -> Result<Net> {
        let mut lines = param.lines().filter(|l| !l.trim().is_empty());
        if lines.next().map(str::trim) != Some("7767517") {
            bail!("not an ncnn param file");
        }
        lines.next();
        let mut blobs: HashMap<String, usize> = HashMap::new();
        let mut blob = |name: &str| -> usize {
            let n = blobs.len();
            *blobs.entry(name.to_string()).or_insert(n)
        };
        let mut layers = Vec::new();
        let mut bin = Bin { data: bin, pos: 0 };
        let mut wdata: Vec<f32> = Vec::new();
        let mut put = |v: Vec<f32>| -> u64 {
            while wdata.len() % 4 != 0 {
                wdata.push(0.0);
            }
            let off = wdata.len() as u64 * 4;
            wdata.extend(v);
            off
        };
        for line in lines {
            let t: Vec<&str> = line.split_whitespace().collect();
            if t.len() < 4 {
                continue;
            }
            let (ni, no) = (t[2].parse::<usize>()?, t[3].parse::<usize>()?);
            let inputs: Vec<usize> = t[4..4 + ni].iter().map(|n| blob(n)).collect();
            let outputs: Vec<usize> = t[4 + ni..4 + ni + no].iter().map(|n| blob(n)).collect();
            let p = Params::parse(&t[4 + ni + no..]);
            let ty = t[0].to_string();
            let (mut weight, mut bias) = (0, None);
            match ty.as_str() {
                "Convolution" | "ConvolutionDepthWise" | "Deconvolution" | "InnerProduct" => {
                    let size = p.i(if ty == "InnerProduct" { 2 } else { 6 }, 0) as usize;
                    let nout = p.i(0, 0) as usize;
                    weight = put(bin.flagged(size)?);
                    if p.i(if ty == "InnerProduct" { 1 } else { 5 }, 0) != 0 {
                        bias = Some(put(bin.raw(nout)?));
                    }
                }
                "PReLU" => weight = put(bin.raw(p.i(0, 1).max(1) as usize)?),
                "MemoryData" => {
                    let n = (p.i(0, 1).max(1) * p.i(1, 1).max(1) * p.i(2, 1).max(1)) as usize;
                    weight = put(bin.raw(n)?);
                }
                // Ours (transformer upscalers, written by pth2ncnn): LayerNorm across the
                // channels of each pixel (gamma, beta), and window attention (its
                // relative position bias table, [(2 window - 1)^2][heads]).
                "ChannelNorm" => weight = put(bin.raw(2 * p.i(0, 0) as usize)?),
                // ncnn's GroupNorm: 0 = groups, 1 = channels, 2 = eps (gamma, beta).
                "GroupNorm" => weight = put(bin.raw(2 * p.i(1, 0) as usize)?),
                "WindowAttention" => {
                    let w = p.i(2, 8) as usize;
                    weight = put(bin.raw((2 * w - 1) * (2 * w - 1) * p.i(1, 1) as usize)?);
                }
                // DAT (pth2ncnn dat): the two branches' position bias tables, and the
                // channel attention's temperature per head.
                "DATAttention" => {
                    let (s0, s1) = (p.i(2, 8) as usize, p.i(3, 32) as usize);
                    weight = put(bin.raw(2 * (2 * s0 - 1) * (2 * s1 - 1) * p.i(1, 1) as usize)?);
                }
                "ChannelAttention" => weight = put(bin.raw(p.i(1, 1) as usize)?),
                _ => {}
            }
            layers.push(Layer { ty, name: t[1].to_string(), inputs, outputs, p, weight, bias, w16: None, im2col: false, fuse: Fuse::default(), wdeconv: None, wino: None });
        }
        if bin.pos != bin.data.len() {
            log::warn!("nn: {} bytes of model data unused", bin.data.len() - bin.pos);
        }
        // 4x4 stride-2 transposed convolutions (DECONV_S2): weights as [ic][tap][oc
        // padded to 8], from ncnn's [oc][ic][tap].
        for l in &mut layers {
            let p = &l.p;
            let (kw, kh) = (p.i(1, 0), p.i(11, p.i(1, 0)));
            let (dw, sw) = (p.i(2, 1), p.i(3, 1));
            if l.ty != "Deconvolution" || (kw, kh) != (4, 4) || (sw, p.i(13, sw), dw, p.i(12, dw)) != (2, 2, 1, 1) {
                continue;
            }
            let oc = p.i(0, 0) as usize;
            let ic = p.i(6, 0) as usize / (oc * 16);
            let ocp = oc.div_ceil(8) * 8;
            let src = wdata[(l.weight / 4) as usize..][..oc * ic * 16].to_vec();
            while wdata.len() % 4 != 0 {
                wdata.push(0.0);
            }
            let off = wdata.len() as u64 * 4;
            for i in 0..ic {
                for t in 0..16 {
                    for o in 0..ocp {
                        wdata.push(if o < oc { src[(o * ic + i) * 16 + t] } else { 0.0 });
                    }
                }
            }
            l.wdeconv = Some((off, ocp));
        }
        let bytes = unsafe { std::slice::from_raw_parts(wdata.as_ptr() as *const u8, wdata.len() * 4) };
        let weights = compute.upload(bytes, "weights")?;
        // Tensor-core convolutions: weights as fp16 [oc (padded to 64)][tap][ic (padded
        // to 16)], for dense convolutions up to 3x3 with stride 1.
        let mut w16: Vec<u16> = Vec::new();
        if compute.ctx.coopmat {
            for l in &mut layers {
                let p = &l.p;
                let (kw, kh) = (p.i(1, 0), p.i(11, p.i(1, 0)));
                let (sw, dw) = (p.i(3, 1), p.i(2, 1));
                let (oc, kk) = (p.i(0, 0) as usize, (kw * kh) as usize);
                if l.ty == "Convolution" && kk > 9 && (dw, p.i(12, dw)) == (1, 1) && p.i(4, 0) >= 0 {
                    // Unfolded patches: [oc (padded to 64)][ic * kk (padded to 16)], the
                    // source order (ic, tap) kept.
                    let ick = p.i(6, 0) as usize / oc;
                    if ick <= 4096 {
                        let (icp, ocp) = (ick.div_ceil(16) * 16, oc.div_ceil(64) * 64);
                        while w16.len() % 4 != 0 {
                            w16.push(0);
                        }
                        let off = w16.len() as u64 * 2;
                        let src = &wdata[(l.weight / 4) as usize..];
                        for o in 0..ocp {
                            for i in 0..icp {
                                w16.push(f32_to_f16(if o < oc && i < ick { src[o * ick + i] } else { 0.0 }));
                            }
                        }
                        l.w16 = Some((off, icp as i32));
                        l.im2col = true;
                        continue;
                    }
                }
                // Stride 2 too: run at stride 1 and subsampled (see `layer`).
                if l.ty != "Convolution" || !matches!((kw, kh), (3, 3) | (1, 1)) || !matches!((sw, p.i(13, sw)), (1, 1) | (2, 2)) || (dw, p.i(12, dw)) != (1, 1) || p.i(4, 0) < 0 {
                    continue;
                }
                let (oc, kk) = (p.i(0, 0) as usize, (kw * kh) as usize);
                let ic = p.i(6, 0) as usize / (oc * kk);
                let (icp, ocp) = (ic.div_ceil(16) * 16, oc.div_ceil(64) * 64);
                while w16.len() % 4 != 0 {
                    w16.push(0);
                }
                let off = w16.len() as u64 * 2;
                let src = &wdata[(l.weight / 4) as usize..];
                for o in 0..ocp {
                    for k in 0..kk {
                        for i in 0..icp {
                            let v = if o < oc && i < ic { src[(o * ic + i) * kk + k] } else { 0.0 };
                            w16.push(f32_to_f16(v));
                        }
                    }
                }
                l.w16 = Some((off, icp as i32));
                if !wino_enabled() || kk != 9 {
                    continue;
                }
                // Winograd: U = G g Gt per (oc, ic), position-major.
                const G: [[f32; 3]; 4] = [[1.0, 0.0, 0.0], [0.5, 0.5, 0.5], [0.5, -0.5, 0.5], [0.0, 0.0, 1.0]];
                let mut u = vec![0.0f32; 16 * ocp * icp];
                for o in 0..oc {
                    for i in 0..ic {
                        let g = &src[(o * ic + i) * 9..][..9];
                        for a in 0..4 {
                            for b in 0..4 {
                                let mut s = 0.0;
                                for y in 0..3 {
                                    for x in 0..3 {
                                        s += G[a][y] * g[y * 3 + x] * G[b][x];
                                    }
                                }
                                u[((a * 4 + b) * ocp + o) * icp + i] = s;
                            }
                        }
                    }
                }
                let off = w16.len() as u64 * 2;
                w16.extend(u.iter().map(|v| f32_to_f16(*v)));
                l.wino = Some(off);
            }
        }
        w16.resize(w16.len().max(2), 0);
        let bytes16 = unsafe { std::slice::from_raw_parts(w16.as_ptr() as *const u8, w16.len() * 2) };
        let weights16 = compute.upload(bytes16, "weights fp16")?;
        let k = Kernels::new(compute)?;
        if k.conv_coop.is_some() && std::env::var_os("BRP_NN_NAIVE").is_none() && std::env::var_os("BRP_NN_NO_FUSE").is_none() {
            fuse_layers(&mut layers, blobs.len());
        }
        let mut readers = vec![0usize; blobs.len()];
        for l in &layers {
            for &i in &l.inputs {
                readers[i] += 1;
            }
        }
        let (place, group_channels) = plan_concats(&layers, blobs.len());
        Ok(Net { compute: compute.clone(), k, layers, blobs, readers, weights, weights16, pool: Default::default(), place, group_channels, run: Mutex::new(RunState::default()), half: false, acc16: false, variants: Mutex::default(), flush_every: Default::default() })
    }

    /// Store the activations between layers as f16 where the layers support it
    /// (tensor-core convolutions, element-wise ops, concats): half the memory
    /// traffic. For networks whose values fit f16 well (upscalers); inputs and
    /// outputs stay f32. `BRP_NN_FP32` keeps f32.
    pub fn set_half(&mut self, on: bool) {
        self.half = on && self.k.conv_coop.is_some() && std::env::var_os("BRP_NN_FP32").is_none();
    }

    /// Take buffers from (and give them back to) `pool`, shared with other nets run
    /// in the same recordings: their working memory is not needed at once.
    pub fn share_pool(&mut self, pool: &Arc<Pool>) {
        self.pool = pool.clone();
    }

    /// Submit the work recorded so far after every `n` layers (0: never), so one
    /// large run never stays on the GPU long in one piece (Windows resets a GPU that
    /// does not respond for about 2 seconds). See `Recorder::flush`.
    pub fn set_flush_every(&self, n: usize) {
        self.flush_every.store(n, std::sync::atomic::Ordering::Relaxed);
    }

    /// Layers recorded by every net so far (to measure GPU time per layer).
    pub fn layers_recorded() -> u64 {
        LAYERS.load(std::sync::atomic::Ordering::Relaxed)
    }

    /// Every net also submits after at most this many layers (0: no limit). The
    /// viewer keeps each submission short so the window's own frames get the GPU in
    /// between (a long submission holds them back: the window stutters).
    pub fn set_flush_limit(n: usize) {
        FLUSH_LIMIT.store(n, std::sync::atomic::Ordering::Relaxed);
    }

    pub fn layer_count(&self) -> usize {
        self.layers.len()
    }

    /// Sum the tensor-core products in f16: twice the tensor cores' f32-sum rate on
    /// GeForce cards, and half the accumulator registers (10-13% faster on the big
    /// upscalers, 6% slower on the small ones; output within 56-64 dB). `BRP_NN_FP32`
    /// keeps f32.
    pub fn set_fp16_sums(&mut self, on: bool) {
        self.acc16 = on && self.k.conv_coop.is_some() && self.compute.ctx.coopmat_acc16 && std::env::var_os("BRP_NN_FP32").is_none() && std::env::var_os("BRP_NN_FP32_SUMS").is_none();
    }

    pub fn blob_names(&self) -> Vec<String> {
        let mut v: Vec<(usize, String)> = self.blobs.iter().map(|(k, v)| (*v, k.clone())).collect();
        v.sort();
        v.into_iter().map(|(_, k)| k).collect()
    }

    /// Record the network into `rec` (nothing runs until it is submitted). Returns
    /// the requested blobs; they stay valid while held.
    pub fn record(&self, rec: &mut Recorder, inputs: &[(&str, &Tensor)], outputs: &[&str]) -> Result<Vec<Tensor>> {
        let mut blob: Vec<Option<Tensor>> = vec![None; self.blobs.len()];
        *self.run.lock().unwrap() = RunState { groups: vec![None; self.group_channels.len()], pending: None };
        for (name, t) in inputs {
            let i = *self.blobs.get(*name).with_context(|| format!("no input {name}"))?;
            blob[i] = Some((*t).clone());
        }
        let wanted: Vec<usize> = outputs.iter().map(|n| self.blobs.get(*n).copied().with_context(|| format!("no blob {n}"))).collect::<Result<_>>()?;
        let mut left = self.readers.clone();
        // BRP_FLUSH_LIMIT=n: as if the viewer had set that limit (tests).
        let limit = match FLUSH_LIMIT.load(std::sync::atomic::Ordering::Relaxed) {
            0 => std::env::var("BRP_FLUSH_LIMIT").ok().and_then(|v| v.parse().ok()).unwrap_or(0),
            l => l,
        };
        let flush_every = match (self.flush_every.load(std::sync::atomic::Ordering::Relaxed), limit) {
            (0, l) | (l, 0) => l,
            (n, l) => n.min(l),
        };
        LAYERS.fetch_add(self.layers.len() as u64, std::sync::atomic::Ordering::Relaxed);
        // Debugging (BRP_NN_SYNC): wait for the GPU before each layer, so a fault names
        // the layer (a device lost after it points at work outside the nets).
        let sync = std::env::var_os("BRP_NN_SYNC").is_some();
        for (li, l) in self.layers.iter().enumerate() {
            if sync {
                rec.sync().with_context(|| format!("before layer {li} {} ({})", l.name, l.ty))?;
            } else if flush_every > 0 && li > 0 && li % flush_every == 0 {
                rec.flush()?;
            }
            if l.outputs.iter().all(|o| blob[*o].is_some()) {
                continue; // an input given directly
            }
            if l.ty == "Input" {
                bail!("input {} not given", l.name);
            }
            let outs = {
                let mut ins: Vec<Tensor> = l.inputs.iter().map(|i| blob[*i].clone().with_context(|| format!("{}: input not computed", l.name))).collect::<Result<_>>()?;
                // f16 inputs of layers that only read f32: converted copies, back to
                // the pool once this layer is recorded.
                let mut converted = Vec::new();
                let takes = self.takes_half(l);
                for (i, t) in ins.iter_mut().enumerate() {
                    // The convolution reads f16 input as 32-bit words: it must start
                    // 4-byte aligned.
                    let misaligned = i == 0 && l.ty == "Convolution" && t.off % 4 != 0;
                    if t.half && (!takes || misaligned) {
                        *t = self.cast(rec, t, false)?;
                        converted.push(t.buf.clone());
                    }
                }
                {
                    let mut run = self.run.lock().unwrap();
                    run.pending = if l.outputs.len() == 1 && !matches!(l.ty.as_str(), "Split" | "Concat") && std::env::var_os("BRP_NN_NO_PLAN").is_none() { self.place.get(&l.outputs[0]).copied() } else { None };
                }
                rec.label = if l.ty.starts_with("Conv") { format!("{} k{} ->{}{}", l.ty, l.p.i(1, 0), l.p.i(0, 0), if l.w16.is_some() && self.k.conv_coop.is_some() { " (tensor)" } else { "" }) } else if l.ty == "Deconvolution" { format!("{} k{} {}->{} at {}x{}", l.ty, l.p.i(1, 0), ins[0].c, l.p.i(0, 0), ins[0].w, ins[0].h) } else { l.ty.clone() };
                let outs = self.layer(rec, l, &ins).with_context(|| format!("layer {} ({})", l.name, l.ty));
                drop(ins);
                for b in converted {
                    self.pool.put(b);
                }
                outs?
            };
            for (o, t) in l.outputs.iter().zip(outs) {
                blob[*o] = Some(t);
            }
            // Inputs nobody reads any more: their buffer goes back to the pool (later
            // layers of this recording may reuse it: dispatches are ordered by
            // barriers) unless another blob shares it (Split), whose release will.
            // Buffers are never destroyed here: the pool keeps them, and the
            // caller holds inputs and outputs until the recording has run.
            for &i in &l.inputs {
                left[i] -= 1;
                if left[i] == 0 && !wanted.contains(&i) && !inputs.iter().any(|(n, _)| self.blobs[*n] == i) {
                    if let Some(t) = blob[i].take() {
                        // Shared: the other blob still holds the buffer (and pools
                        // it when it is freed); dropping this reference is enough.
                        if !blob.iter().flatten().any(|o| Arc::ptr_eq(&o.buf, &t.buf)) {
                            // A concat group read for the last time: no more producers
                            // write into it, so it goes back now (not at the end: the
                            // big models have dozens of them per run).
                            let mut run = self.run.lock().unwrap();
                            if let Some(g) = run.groups.iter_mut().find(|g| g.as_ref().is_some_and(|(b, _, _, _)| Arc::ptr_eq(b, &t.buf))) {
                                *g = None;
                            }
                            drop(run);
                            self.pool.put(t.buf);
                        }
                    }
                }
            }
            if wanted.iter().all(|w| blob[*w].is_some()) {
                break;
            }
        }
        // Group buffers: back to the pool (the pool keeps them alive for the recorded
        // work); outputs that are views hold theirs.
        for (b, _, _, _) in std::mem::take(&mut self.run.lock().unwrap().groups).into_iter().flatten() {
            self.pool.put(b);
        }
        // Outputs are f32.
        let mut outs = Vec::new();
        for w in &wanted {
            let t = blob[*w].clone().context("output not computed")?;
            outs.push(if t.half {
                let o = self.cast(rec, &t, false)?;
                // The f16 tensor goes back to the pool, which keeps its buffer alive:
                // the recorded work has not run yet (dropping it here freed memory
                // the last layer then wrote into: device lost in the player).
                blob[*w] = None;
                self.pool.put(t.buf);
                o
            } else {
                t
            });
        }
        Ok(outs)
    }

    /// A layer's output: in its planned slot of a concat group when it has one
    /// (the first output allocated by the layer being recorded), else from the pool.
    fn new_tensor(&self, c: usize, h: usize, w: usize) -> Result<Tensor> {
        self.new_tensor_of(c, h, w, false)
    }

    /// The same, f16 if `half`. A group's element type is its first member's; a
    /// member of the other type gets a tensor of its own (the concat then copies).
    fn new_tensor_of(&self, c: usize, h: usize, w: usize, half: bool) -> Result<Tensor> {
        let es = if half { 2 } else { 4 };
        let mut run = self.run.lock().unwrap();
        if let Some((g, off)) = run.pending.take() {
            if run.groups[g].is_none() {
                let total = self.group_channels[g];
                run.groups[g] = Some((self.pool.get(&self.compute, (total * h * w) as u64 * es)?, h, w, half));
            }
            let (buf, gh, gw, gh16) = run.groups[g].clone().unwrap();
            if (gh, gw, gh16) == (h, w, half) && off + c <= self.group_channels[g] {
                return Ok(Tensor { buf, off: (off * h * w) as u64 * es, c, h, w, half });
            }
        }
        drop(run);
        self.pool.tensor_of(&self.compute, c, h, w, half)
    }

    /// `t` as f32 or f16 (a converted copy, or `t` itself).
    fn cast(&self, rec: &mut Recorder, t: &Tensor, half: bool) -> Result<Tensor> {
        if t.half == half {
            return Ok(t.clone());
        }
        let out = self.pool.tensor_of(&self.compute, t.c, t.h, t.w, half)?;
        let label = std::mem::replace(&mut rec.label, format!("cast {} ch to {}", t.c, if half { "f16" } else { "f32" }));
        rec.dispatch(&self.k.cast, flat(&self.k.cast, t.len()), &Push::new().t(t).t(&out).i(t.len() as i32).i(half as i32))?;
        rec.label = label;
        Ok(out)
    }

    /// The global average of each channel of `x` ([C][1][1]): sums of chunks (many
    /// workgroups), then per channel.
    fn global_mean(&self, rec: &mut Recorder, x: &Tensor) -> Result<Tensor> {
        let k = &self.k;
        let plane = x.h * x.w;
        let chunks = plane.div_ceil(4096);
        let part = self.pool.get(&self.compute, (x.c * chunks * 4) as u64)?;
        let push = Push::new().t(x).ptr_at(&part, 0).i(plane as i32).i(chunks as i32).i(x.half as i32);
        rec.dispatch(&k.global_sum, [chunks as u32, x.c as u32, 1], &push)?;
        let out = self.new_tensor(x.c, 1, 1)?;
        let push = Push::new().ptr_at(&part, 0).t(&out).i(x.c as i32).i(plane as i32).i(chunks as i32);
        rec.dispatch(&k.global_avg, flat(&k.global_avg, x.c), &push)?;
        self.pool.put(part);
        Ok(out)
    }

    /// The layer reads and writes f16 tensors itself (others get f32 copies).
    fn takes_half(&self, l: &Layer) -> bool {
        match l.ty.as_str() {
            "Split" | "Concat" | "BinaryOp" | "Eltwise" | "ChannelNorm" | "WindowAttention" => true,
            "ChannelView" | "DATAttention" | "ChannelAttention" | "AIM" | "GlobalAttention" | "PixelUnshuffle" => true,
            "Canvas" | "Tiles" | "DeformIm2col" => true,
            "Convolution" => self.coop_for(l).is_some(),
            // Grouped (CONV_GROUP) and global average pooling (GLOBAL_AVG).
            "ConvolutionDepthWise" => l.p.i(7, 1) > 1,
            "Pooling" => l.p.i(4, 0) != 0,
            "BoxMean" | "GroupNorm" => true,
            _ => false,
        }
    }

    /// A kernel compiled for f16 element types `dt` (built on first use).
    fn variant(&self, id: usize, dt: u8, build: impl FnOnce() -> Result<Arc<Kernel>>) -> Result<Arc<Kernel>> {
        let mut v = self.variants.lock().unwrap();
        if let Some(k) = v.get(&(id, dt)) {
            return Ok(k.clone());
        }
        let k = build()?;
        v.insert((id, dt), k.clone());
        Ok(k)
    }

    /// The tensor-core kernel for a convolution, if it runs on one.
    fn coop_for(&self, l: &Layer) -> Option<((u64, i32), &(CoopConv, CoopConv))> {
        match (l.ty.as_str(), l.w16, &self.k.conv_coop) {
            ("Convolution", Some(w), Some(kc)) if std::env::var_os("BRP_NN_NAIVE").is_none() => Some((w, kc)),
            _ => None,
        }
    }

    fn layer(&self, rec: &mut Recorder, l: &Layer, ins: &[Tensor]) -> Result<Vec<Tensor>> {
        let p = &l.p;
        let k = &self.k;
        let g3 = |kern: &Kernel, w: usize, h: usize, c: usize| Recorder::groups(kern, [w as u32, h as u32, c as u32]);
        Ok(match l.ty.as_str() {
            "Split" => vec![ins[0].clone(); l.outputs.len()],
            "Tile" if p.arr(2).iter().all(|r| *r == 1.0) => vec![ins[0].clone()],
            "MemoryData" => {
                let (w, h, c) = (p.i(0, 1).max(1) as usize, p.i(1, 1).max(1) as usize, p.i(2, 1).max(1) as usize);
                let t = self.new_tensor(c, h, w)?;
                let src = &self.weights;
                rec.copy(src, l.weight, &t.buf, t.off, (c * h * w) as u64 * 4);
                vec![t]
            }
            "Convolution" if l.im2col && self.coop_for(l).is_some() => {
                let x = &ins[0];
                let (oc, kw) = (p.i(0, 0) as usize, p.i(1, 0));
                let (kh, sw) = (p.i(11, kw), p.i(3, 1));
                let (sh, pl) = (p.i(13, sw), p.i(4, 0));
                let (pt, pr) = (p.i(14, pl), p.i(15, pl));
                let pb = p.i(16, pt);
                let ow = ((x.w as i32 + pl + pr - kw) / sw + 1).max(1) as usize;
                let oh = ((x.h as i32 + pt + pb - kh) / sh + 1).max(1) as usize;
                let icp = l.w16.unwrap().1 as usize;
                let mut l1 = l.clone();
                for (key, v) in [(1, 1), (11, 1), (2, 1), (12, 1), (3, 1), (13, 1), (4, 0), (14, 0), (15, 0), (16, 0), (6, (oc * icp) as i32)] {
                    l1.p.0.insert(key, v.to_string());
                }
                l1.im2col = false;
                // In strips of output rows, so the patches take at most 256 MB.
                let hs = ((256usize << 20) / (icp * ow * 2)).clamp(1, oh);
                let whole = if hs < oh { Some(self.new_tensor_of(oc, oh, ow, self.half && p.i(30, 0) == 0)?) } else { None };
                let mut result = None;
                for y0 in (0..oh).step_by(hs) {
                    let h = hs.min(oh - y0);
                    let col = self.pool.tensor_of(&self.compute, icp, h, ow, true)?;
                    let push = Push::new().t(x).t(&col).i(x.c as i32).i(x.h as i32).i(x.w as i32).i(h as i32).i(ow as i32).i(kw).i(kh).i(sw).i(sh).i(pl).i(pt - (y0 as i32) * sh).i(icp as i32).i(x.half as i32);
                    rec.label = format!("Convolution k{kw} ->{oc}: unfold");
                    rec.dispatch(&k.im2col, g3(&k.im2col, ow, h, icp), &push)?;
                    rec.label = format!("Convolution k{kw} ->{oc} (tensor, unfolded)");
                    let strip = self.layer(rec, &l1, &[col.clone()])?.remove(0);
                    self.pool.put(col.buf);
                    match &whole {
                        None => result = Some(strip),
                        Some(w) => {
                            let es = if strip.half { 2 } else { 4 };
                            anyhow::ensure!(strip.half == w.half, "unfolded convolution: strip and output types differ");
                            for o in 0..oc {
                                rec.copy(&strip.buf, strip.off + (o * h * ow) as u64 * es, &w.buf, w.off + ((o * oh + y0) * ow) as u64 * es, (h * ow) as u64 * es);
                            }
                            self.pool.put(strip.buf);
                        }
                    }
                }
                vec![whole.or(result).context("no output rows")?]
            }
            "Convolution" if l.p.i(3, 1) == 2 && l.p.i(13, 2) == 2 && self.coop_for(l).is_some() => {
                // Stride 2 on the tensor-core path: a 1x1 subsamples its input, then runs;
                // a 3x3 runs at stride 1, then is subsampled (4x the products, still far
                // faster than the scalar kernel).
                let mut l1 = l.clone();
                l1.p.0.insert(3, "1".into());
                l1.p.0.insert(13, "1".into());
                if p.i(1, 0) == 1 {
                    let x2 = self.subsample(rec, &ins[0], false)?;
                    let mut ins2 = ins.to_vec();
                    ins2[0] = x2.clone();
                    let outs = self.layer(rec, &l1, &ins2)?;
                    drop(ins2);
                    self.pool.put(x2.buf);
                    outs
                } else {
                    let full = self.layer(rec, &l1, ins)?.remove(0);
                    let out = self.subsample(rec, &full, true)?;
                    self.pool.put(full.buf);
                    vec![out]
                }
            }
            "Convolution" | "ConvolutionDepthWise" | "Deconvolution" => {
                let x = &ins[0];
                let oc = p.i(0, 0) as usize;
                let kw = p.i(1, 0);
                let kh = p.i(11, kw);
                let dw = p.i(2, 1);
                let dh = p.i(12, dw);
                let sw = p.i(3, 1);
                let sh = p.i(13, sw);
                let pl = p.i(4, 0);
                let pt = p.i(14, pl);
                let pr = p.i(15, pl);
                let pb = p.i(16, pt);
                let (ekw, ekh) = (dw * (kw - 1) + 1, dh * (kh - 1) + 1);
                let (ow, oh) = if l.ty == "Deconvolution" {
                    let ow = (x.w as i32 - 1) * sw + ekw + p.i(18, 0) - pl - pr;
                    let oh = (x.h as i32 - 1) * sh + ekh + p.i(19, 0) - pt - pb;
                    (ow, oh)
                } else {
                    if pl < 0 {
                        bail!("automatic padding not supported");
                    }
                    ((x.w as i32 + pl + pr - ekw) / sw + 1, (x.h as i32 + pt + pb - ekh) / sh + 1)
                };
                let (ow, oh) = (ow.max(1) as usize, oh.max(1) as usize);
                let coop = self.coop_for(l);
                let groups = if l.ty == "ConvolutionDepthWise" { p.i(7, 1) } else { 1 };
                // Param 30=1 (our own): keep this output f32 (values beyond f16's range).
                // Grouped convolutions (CONV_GROUP) read and write f16 themselves.
                let out = self.new_tensor_of(oc, oh, ow, self.half && (coop.is_some() || groups > 1) && p.i(30, 0) == 0)?;
                let act = p.i(9, 0);
                let ap = p.arr(10);
                let (a0, a1) = (ap.first().copied().unwrap_or(0.0) as f32, ap.get(1).copied().unwrap_or(0.0) as f32);
                let push = Push::new()
                    .t(&x)
                    .ptr_at(&self.weights, l.weight)
                    .ptr_at(&self.weights, l.bias.unwrap_or(0))
                    .t(&out)
                    .i(x.c as i32)
                    .i(x.h as i32)
                    .i(x.w as i32)
                    .i(oc as i32)
                    .i(oh as i32)
                    .i(ow as i32)
                    .i(kw)
                    .i(kh)
                    .i(sw)
                    .i(sh)
                    .i(dw)
                    .i(dh)
                    .i(pl)
                    .i(pt)
                    .i(act)
                    .i(l.bias.is_some() as i32)
                    .f(a0)
                    .f(a1);
                if let (Some((off, ocp)), None) = (l.wdeconv, std::env::var_os("BRP_NN_NAIVE")) {
                    let push = Push::new()
                        .t(&x)
                        .ptr_at(&self.weights, off)
                        .ptr_at(&self.weights, l.bias.unwrap_or(0))
                        .t(&out)
                        .i(x.c as i32)
                        .i(x.h as i32)
                        .i(x.w as i32)
                        .i(oc as i32)
                        .i(oh as i32)
                        .i(ow as i32)
                        .i(pl)
                        .i(pt)
                        .i(ocp as i32)
                        .i(act)
                        .i(l.bias.is_some() as i32)
                        .f(a0)
                        .f(a1);
                    rec.dispatch(&k.deconv_s2, g3(&k.deconv_s2, ow, oh, ocp / 8), &push)?;
                } else if l.ty == "Deconvolution" {
                    rec.dispatch(&k.deconv, g3(&k.deconv, ow, oh, oc), &push)?;
                } else if let Some(((off, icp), kc)) = coop {
                    // Fused layers (see `fuse_layers`): the residual is the second input.
                    let res = match l.fuse.res {
                        Some(_) => {
                            let r = ins.get(1).context("fused sum: second input missing")?;
                            if (r.c, r.h, r.w) != (oc, oh, ow) {
                                bail!("fused sum: {}x{}x{} added to {oc}x{oh}x{ow}", r.c, r.h, r.w);
                            }
                            Some(r)
                        }
                        None => None,
                    };
                    let (rc0, rc1) = l.fuse.res.unwrap_or((1.0, 0.0));
                    let flags = l.bias.is_some() as i32
                        | (res.is_some() as i32) << 1
                        | (l.fuse.prelu.is_some() as i32) << 2
                        | (l.fuse.prelu.is_some_and(|p| p.1) as i32) << 3
;
                    let dt = x.half as u8 | (out.half as u8) << 1 | (res.is_some_and(|r| r.half) as u8) << 2 | (self.acc16 as u8) << 3;
                    // Winograd when enabled (BRP_NN_WINO; slower than the direct kernel here).
                    let wino = match (l.wino, &k.conv_wino) {
                        (Some(woff), Some(kw2)) => Some((woff, if oc <= 32 { &kw2.1 } else { &kw2.0 })),
                        _ => None,
                    };
                    let push = Push::new()
                        .t(&x)
                        .ptr_at(&self.weights16, wino.map_or(off, |w| w.0))
                        .ptr_at(&self.weights, l.bias.unwrap_or(0))
                        .t(&out)
                        .t(res.unwrap_or(&out))
                        .ptr_at(&self.weights, l.fuse.prelu.map_or(0, |p| p.0))
                        .i(x.c as i32)
                        .i(x.h as i32)
                        .i(x.w as i32)
                        .i(oc as i32)
                        .i(oh as i32)
                        .i(ow as i32)
                        .i(pl)
                        .i(pt)
                        .i(act)
                        .i(flags)
                        .f(a0)
                        .f(a1)
                        .f(rc0)
                        .f(rc1)
                        .i(icp)
                        .i(oc.div_ceil(64) as i32 * 64);
                    let v = match wino {
                        Some((_, w)) => w,
                        None if oc <= 32 => &kc.1,
                        None => &kc.0,
                    };
                    let id = (oc <= 32) as usize + if wino.is_some() { 3 } else { 0 };
                    let plane = x.h * x.w;
                    if kw == 1 && x.half && oc > 32 && std::env::var_os("BRP_NN_NO_GEMM").is_none() {
                        // Planes of 8n pixels in 16-byte aligned tensors load 8 pixels at once;
                        // others (token counts like 769) in single values.
                        let unaligned = plane % 8 != 0 || x.off % 16 != 0 || out.off % 16 != 0;
                        rec.label.push_str(if unaligned { " 1x1 gemm f16 unaligned" } else { " 1x1 gemm f16" });
                        let sg = self.compute.ctx.subgroup_size;
                        // (subgroups, 16-pixel tiles per subgroup, channels per step);
                        // BRP_GEMM="nsg,rs,kc" overrides, for tuning.
                        let (nsg, rs, kc) = std::env::var("BRP_GEMM")
                            .ok()
                            .and_then(|v| {
                                let n: Vec<usize> = v.split(',').filter_map(|x| x.trim().parse().ok()).collect();
                                (n.len() == 3).then(|| (n[0], n[1], n[2]))
                            })
                            .unwrap_or(GEMM_SHAPE);
                        // Each thread stages 32 * rs / sg loads per step: at least one on
                        // 64-wide subgroups (AMD).
                        let rs = rs.max(sg as usize / 32);
                        let kern = self.variant(40 + unaligned as usize * 1000, dt, || {
                            let defs = format!("#define SG {sg}\n#define NSG {nsg}\n#define RS {rs}\n#define KC {kc}\n#define IN16 1\n#define OUT16 {}\n#define RES16 {}\n#define ACC16 {}\n#define UNALIGNED {}\n", dt >> 1 & 1, dt >> 2 & 1, dt >> 3 & 1, unaligned as u8);
                            self.compute.kernel(&format!("nn_conv1x1_{nsg}_{rs}_{kc}_{dt}_{}", unaligned as u8), &format!("{PRELUDE}{defs}{HALF}{ACT}{CONV1X1_COOP}"), 0)
                        })?;
                        rec.dispatch(&kern, [plane.div_ceil(nsg * rs * 16) as u32, 1, oc.div_ceil(64) as u32], &push)?;
                        return Ok(vec![out]);
                    }
                    let kern = if kw == 1 {
                        // 1x1 (built on first use): one tap, no border around the staged block.
                        rec.label.push_str(" 1x1");
                        self.variant(id + 5, dt, || CoopConv::build(&self.compute, &format!("{}k1", v.name), v.shape, dt))?
                    } else if dt == 0 {
                        v.kernel.clone()
                    } else {
                        self.variant(id, dt, || CoopConv::build(&self.compute, &v.name, v.shape, dt))?
                    };
                    if wino.is_some() {
                        rec.label.push_str(" wino");
                    }
                    if dt != 0 {
                        rec.label.push_str(" f16");
                    }
                    rec.dispatch(&kern, [ow.div_ceil(16) as u32, oh.div_ceil(v.rows) as u32, oc.div_ceil(v.oc) as u32], &push)?;
                } else if groups == 1 && kw <= 3 && kh <= 3 && (sw, sh, dw, dh) == (1, 1, 1, 1) && std::env::var_os("BRP_NN_NAIVE").is_none() {
                    if oc <= 32 {
                        rec.dispatch(&k.conv_tiled32, [ow.div_ceil(16) as u32, oh.div_ceil(8) as u32, oc.div_ceil(32) as u32], &push)?;
                    } else {
                        rec.dispatch(&k.conv_tiled, [ow.div_ceil(8) as u32, oh.div_ceil(8) as u32, oc.div_ceil(64) as u32], &push)?;
                    }
                } else if groups == 1 {
                    let gr = Recorder::groups(&k.conv, [ow as u32, oh as u32, 1]);
                    rec.dispatch(&k.conv, [gr[0], gr[1], oc.div_ceil(8) as u32], &push)?;
                } else if groups as usize == oc && x.c == oc && (kw, kh, sw, sh, dw, dh, pl, pt) == (3, 3, 1, 1, 1, 1, 1, 1) && (ow, oh) == (x.w, x.h) {
                    let mul = ins.get(1);
                    if mul.is_some_and(|m| (m.c, m.h, m.w) != (oc, oh, ow)) {
                        bail!("depthwise gate: sizes do not match");
                    }
                    let push = Push::new()
                        .t(x)
                        .ptr_at(&self.weights, l.weight)
                        .ptr_at(&self.weights, l.bias.unwrap_or(0))
                        .t(&out)
                        .t(mul.unwrap_or(&out))
                        .i(oc as i32)
                        .i(oh as i32)
                        .i(ow as i32)
                        .i(act)
                        .i(l.bias.is_some() as i32)
                        .f(a0)
                        .f(a1)
                        .i(x.half as i32)
                        .i(out.half as i32);
                    let push = push.i(mul.is_some() as i32).i(mul.is_some_and(|m| m.half) as i32);
                    rec.dispatch(&k.dw3, g3(&k.dw3, ow.div_ceil(4), oh, oc), &push)?;
                } else {
                    if ins.len() > 1 && l.fuse.res.is_none() {
                        bail!("depthwise gate: only 3x3 depthwise convolutions with padding 1");
                    }
                    rec.dispatch(&k.conv_group, g3(&k.conv_group, ow, oh, oc), &push.i(groups).i(x.half as i32).i(out.half as i32))?;
                }
                vec![out]
            }
            "InnerProduct" => {
                let x = &ins[0];
                let oc = p.i(0, 0) as usize;
                let out = self.new_tensor(oc, 1, 1)?;
                let ap = p.arr(10);
                let push = Push::new()
                    .t(&x)
                    .ptr_at(&self.weights, l.weight)
                    .ptr_at(&self.weights, l.bias.unwrap_or(0))
                    .t(&out)
                    .i(x.len() as i32)
                    .i(oc as i32)
                    .i(p.i(9, 0))
                    .i(l.bias.is_some() as i32)
                    .f(ap.first().copied().unwrap_or(0.0) as f32)
                    .f(ap.get(1).copied().unwrap_or(0.0) as f32);
                rec.dispatch(&k.inner, Recorder::groups(&k.inner, [oc as u32, 1, 1]), &push)?;
                vec![out]
            }
            "BinaryOp" | "Eltwise" => {
                let (a, op, scalar, s, ka, kb) = if l.ty == "Eltwise" {
                    let coef = p.arr(1);
                    let op = match p.i(0, 1) {
                        0 => 2,
                        2 => 4,
                        _ if coef.is_empty() => 0,
                        _ => 100,
                    };
                    (&ins[0], op, false, 0.0, coef.first().copied().unwrap_or(1.0) as f32, coef.get(1).copied().unwrap_or(1.0) as f32)
                } else {
                    (&ins[0], p.i(0, 0), p.i(1, 0) != 0, p.f(2, 0.0), 1.0, 1.0)
                };
                if l.ty == "Eltwise" && ins.len() > 2 {
                    bail!("Eltwise with {} inputs", ins.len());
                }
                let b = if scalar { a } else { &ins[1] };
                let (c, h, w) = (a.c.max(b.c), a.h.max(b.h), a.w.max(b.w));
                let out = self.new_tensor_of(c, h, w, (a.half || (!scalar && b.half)) && p.i(30, 0) == 0)?;
                let fl = a.half as u8 | ((!scalar && b.half) as u8) << 1 | (out.half as u8) << 2;
                let kern = if fl == 0 {
                    k.binary.clone()
                } else {
                    self.variant(2, fl, || {
                        let defs = format!("#define A16 {}\n#define B16 {}\n#define O16 {}\n", fl & 1, fl >> 1 & 1, fl >> 2 & 1);
                        self.compute.kernel(&format!("nn_binary_{fl}"), &format!("{PRELUDE}{defs}{HALF}{ACT}{BINARY}"), 0)
                    })?
                };
                let push = Push::new()
                    .t(&a)
                    .t(&b)
                    .t(&out)
                    .i(c as i32)
                    .i(h as i32)
                    .i(w as i32)
                    .i(a.c as i32)
                    .i(a.h as i32)
                    .i(a.w as i32)
                    .i(b.c as i32)
                    .i(b.h as i32)
                    .i(b.w as i32)
                    .i(op)
                    .i(scalar as i32)
                    .f(s)
                    .f(ka)
                    .f(kb)
;
                rec.dispatch(&kern, g3(&kern, w, h, c), &push)?;
                vec![out]
            }
            "ReLU" | "PReLU" | "Sigmoid" | "TanH" | "HardSigmoid" | "HardSwish" | "Swish" | "GELU" | "Clip" => {
                let x = &ins[0];
                let out = self.new_tensor(x.c, x.h, x.w)?;
                let (op, a0, a1) = match l.ty.as_str() {
                    "ReLU" => (1, p.f(0, 0.0), 0.0),
                    "PReLU" => (2, 0.0, 0.0),
                    "Sigmoid" => (3, 0.0, 0.0),
                    "TanH" => (4, 0.0, 0.0),
                    "HardSigmoid" => (5, p.f(0, 0.2), p.f(1, 0.5)),
                    "HardSwish" => (6, p.f(0, 0.2), p.f(1, 0.5)),
                    "Swish" => (8, 0.0, 0.0),
                    "GELU" => (9, 0.0, 0.0),
                    _ => (7, p.f(0, f32::MIN), p.f(1, f32::MAX)),
                };
                let push = Push::new()
                    .t(&x)
                    .t(&out)
                    .ptr_at(&self.weights, l.weight)
                    .i(x.len() as i32)
                    .i((x.h * x.w) as i32)
                    .i(op)
                    .i(p.i(0, 1))
                    .f(a0)
                    .f(a1);
                rec.dispatch(&k.unary, flat(&k.unary, x.len()), &push)?;
                vec![out]
            }
            "ChannelNorm" => {
                let x = &ins[0];
                let out = self.new_tensor_of(x.c, x.h, x.w, self.half)?;
                let n = x.h * x.w;
                let push = Push::new().t(x).t(&out).ptr_at(&self.weights, l.weight).i(x.c as i32).i(n as i32).f(p.f(1, 1e-5)).i(x.half as i32).i(out.half as i32);
                let groups = n.div_ceil(32);
                let gx = groups.min(65535);
                rec.dispatch(&k.channel_norm, [gx as u32, groups.div_ceil(gx) as u32, 1], &push)?;
                vec![out]
            }
            "WindowAttention" => {
                // Input: the qkv projection, [3C][h][w].
                let x = &ins[0];
                let (c, heads, ws, shift) = (p.i(0, 0) as usize, p.i(1, 1) as usize, p.i(2, 8) as usize, p.i(3, 0));
                if x.c != 3 * c || x.h % ws != 0 || x.w % ws != 0 || c / heads > 64 {
                    bail!("window attention: {}x{}x{} input for {c} channels, window {ws}", x.c, x.h, x.w);
                }
                // Tensor cores when windows are 16..64 tokens (a multiple of 16).
                let hdp = (c / heads).div_ceil(16) * 16;
                let coop = self.compute.ctx.coopmat && (ws * ws) % 16 == 0 && ws * ws <= 64 && std::env::var_os("BRP_NN_NAIVE").is_none();
                let kern = if coop {
                    let sg = self.compute.ctx.subgroup_size;
                    rec.label.push_str(" (tensor)");
                    self.variant(100 + ws * 100 + hdp, 0, || {
                        self.compute.kernel(&format!("nn_window_attn_coop_{ws}_{hdp}"), &format!("{PRELUDE}#define WS {ws}\n#define HDP {hdp}\n#define SG {sg}\n{HALF}{ACT}{WINDOW_ATTN_COOP}"), 0)
                    })?
                } else {
                    self.variant(20 + ws, 0, || {
                        self.compute.kernel(&format!("nn_window_attn_{ws}"), &format!("{PRELUDE}#define WS {ws}\n{HALF}{ACT}{WINDOW_ATTN}"), 0)
                    })?
                };
                let out = self.new_tensor_of(c, x.h, x.w, self.half)?;
                let push = Push::new().t(x).t(&out).ptr_at(&self.weights, l.weight).i(c as i32).i(x.h as i32).i(x.w as i32).i(heads as i32).i(shift).i(x.half as i32).i(out.half as i32);
                rec.dispatch(&kern, [((x.h / ws) * (x.w / ws)) as u32, heads as u32, 1], &push)?;
                vec![out]
            }
            "GlobalAttention" => {
                // Input: the qkv projection [3C][h][w], every token attending to all.
                let x = &ins[0];
                let (c, heads) = (p.i(0, 0) as usize, p.i(1, 1) as usize);
                let hd = c / heads;
                let n = x.h * x.w;
                if x.c != 3 * c || hd > 64 {
                    bail!("global attention: {} channels for {c}, {heads} heads", x.c);
                }
                let out = self.new_tensor_of(c, x.h, x.w, self.half)?;
                let push = Push::new().t(x).t(&out).i(c as i32).i(n as i32).i(heads as i32).i(hd as i32).i(x.half as i32).i(out.half as i32);
                let sg = self.compute.ctx.subgroup_size as usize;
                let hdp = hd.div_ceil(16) * 16;
                let shm = unsafe { self.compute.ctx.instance.get_physical_device_properties(self.compute.ctx.physical) }.limits.max_compute_shared_memory_size as usize;
                // Shared memory: keys and values, then per subgroup f32 scores, f16
                // probabilities and row sums (48 KB: 4 subgroups).
                let nsg = std::env::var("BRP_ATTN_NSG").ok().and_then(|v| v.parse().ok()).unwrap_or_else(|| [8usize, 4, 2, 1].into_iter().find(|&g| g * 16 * 64 * 6 <= shm).unwrap_or(1));
                if self.compute.ctx.coopmat && sg >= 32 && std::env::var_os("BRP_NN_NAIVE").is_none() {
                    rec.label.push_str(" (tensor)");
                    let kern = self.variant(5000 + hdp + nsg * 1000, 0, || {
                        self.compute.kernel(&format!("nn_global_attn_coop_{hdp}_{nsg}"), &format!("{PRELUDE}#define HDP {hdp}
#define SG {sg}
#define NSG {nsg}
{HALF}{ACT}{GLOBAL_ATTN_COOP}"), 0)
                    })?;
                    let kvt = self.variant(90_000 + hdp, 0, || self.compute.kernel(&format!("nn_kv_t_{hdp}"), &format!("{PRELUDE}#define HDP {hdp}\n{HALF}{KV_T}"), 0))?;
                    let np = n.div_ceil(64) * 64;
                    let kv = self.pool.tensor_of(&self.compute, 2 * heads * hdp, 1, np, true)?;
                    rec.label = "GlobalAttention: keys and values".into();
                    let tpush = Push::new().t(x).t(&kv).i(c as i32).i(n as i32).i(heads as i32).i(hd as i32).i(x.half as i32).i(np as i32);
                    rec.dispatch(&kvt, Recorder::groups(&kvt, [hdp as u32, np as u32, 2 * heads as u32]), &tpush)?;
                    rec.label = "GlobalAttention (tensor)".into();
                    let push = Push::new().t(x).t(&out).t(&kv).i(c as i32).i(n as i32).i(heads as i32).i(hd as i32).i(x.half as i32).i(out.half as i32).i(np as i32);
                    rec.dispatch(&kern, [n.div_ceil(16 * nsg) as u32, heads as u32, 1], &push)?;
                    // Back to the pool (which keeps it alive until the work has run).
                    self.pool.put(kv.buf);
                } else {
                    rec.dispatch(&k.global_attn, [n.div_ceil(64) as u32, heads as u32, 1], &push)?;
                }
                vec![out]
            }
            // Ours (pth2ncnn birefnet): 0 = height, 1 = width of the output (the input's
            // top-left, zeros past it).
            "Canvas" => {
                let x = &ins[0];
                let (oh, ow) = (p.i(0, 0) as usize, p.i(1, 0) as usize);
                let out = self.new_tensor_of(x.c, oh, ow, self.half)?;
                let push = Push::new().t(x).t(&out).i(x.c as i32).i(x.h as i32).i(x.w as i32).i(oh as i32).i(ow as i32).i(x.half as i32).i(out.half as i32);
                rec.dispatch(&k.canvas, g3(&k.canvas, ow, oh, x.c), &push)?;
                vec![out]
            }
            // 0 = tile height, 1 = tile width.
            "Tiles" => {
                let x = &ins[0];
                let (th, tw) = (p.i(0, 1) as usize, p.i(1, 1) as usize);
                if th == 0 || tw == 0 || x.h % th != 0 || x.w % tw != 0 {
                    bail!("tiles of {th}x{tw} from {}x{}", x.w, x.h);
                }
                let oc = x.c * (x.h / th) * (x.w / tw);
                let out = self.new_tensor_of(oc, th, tw, self.half)?;
                let push = Push::new().t(x).t(&out).i(x.c as i32).i(x.h as i32).i(x.w as i32).i(th as i32).i(tw as i32).i(x.half as i32).i(out.half as i32);
                rec.dispatch(&k.tiles, g3(&k.tiles, tw, th, oc), &push)?;
                vec![out]
            }
            // Inputs: the map, its offsets and modulator logits; 0 = kernel size, 1 = padding.
            "DeformIm2col" => {
                let (x, om) = (&ins[0], &ins[1]);
                let (kk, pad) = (p.i(0, 1) as usize, p.i(1, 0));
                if om.c != 3 * kk * kk || (om.h, om.w) != (x.h, x.w) {
                    bail!("deformable sampling: {} offset channels for a {kk}x{kk} kernel", om.c);
                }
                let out = self.new_tensor_of(x.c * kk * kk, x.h, x.w, self.half)?;
                let push = Push::new().t(x).t(om).t(&out).i(x.c as i32).i(x.h as i32).i(x.w as i32).i(kk as i32).i(pad).i(x.half as i32).i(om.half as i32).i(out.half as i32);
                rec.dispatch(&k.deform_im2col, g3(&k.deform_im2col, x.w, x.h, kk * kk), &push)?;
                vec![out]
            }
            "PixelUnshuffle" => {
                let x = &ins[0];
                let r = p.i(0, 1) as usize;
                if x.h % r != 0 || x.w % r != 0 {
                    bail!("pixel unshuffle by {r} of {}x{}", x.w, x.h);
                }
                let oc = x.c * r * r;
                let out = self.new_tensor_of(oc, x.h / r, x.w / r, self.half)?;
                let push = Push::new().t(x).t(&out).i(x.c as i32).i(x.h as i32).i(x.w as i32).i(r as i32).i(x.half as i32).i(out.half as i32);
                rec.dispatch(&k.pixel_unshuffle, g3(&k.pixel_unshuffle, x.w / r, x.h / r, oc), &push)?;
                vec![out]
            }
            // Channels start..start + count of the input, without a copy.
            "ChannelView" => {
                let (c0, n) = (p.i(0, 0) as usize, p.i(1, 0) as usize);
                if c0 + n > ins[0].c {
                    bail!("channel view {c0}+{n} of {} channels", ins[0].c);
                }
                vec![ins[0].view(c0, n)]
            }
            "DATAttention" => {
                // Input: the qkv projection, [3C][h][w]; two branches of C/2 channels.
                let x = &ins[0];
                let (c, heads, s0, s1, shifted) = (p.i(0, 0) as usize, p.i(1, 1) as usize, p.i(2, 8) as usize, p.i(3, 32) as usize, p.i(4, 0) != 0);
                let hd = c / 2 / heads;
                if x.c != 3 * c || x.h % s0.max(s1) != 0 || x.w % s0.max(s1) != 0 || hd > 32 || s0 * s1 > 1024 {
                    bail!("DAT attention: {}x{}x{} input for {c} channels, windows {s0}x{s1}", x.c, x.h, x.w);
                }
                let n = s0 * s1;
                // Tensor cores (subgroups of at least 32: the softmax's lane layout).
                let sg = self.compute.ctx.subgroup_size;
                let coop = self.compute.ctx.coopmat && sg >= 32 && n % 32 == 0 && std::env::var_os("BRP_NN_NAIVE").is_none();
                let hdp = hd.div_ceil(16) * 16;
                // Subgroups per workgroup: as many as shared memory and the workgroup size
                // allow, up to one per 16 queries (RTX 5070 Ti, 256x256: 4 = 30 ms, 8 = 31,
                // 16 = 19). BRP_DAT_NSG overrides.
                let props = unsafe { self.compute.ctx.instance.get_physical_device_properties(self.compute.ctx.physical) };
                let (shm, inv) = (props.limits.max_compute_shared_memory_size as usize, props.limits.max_compute_work_group_invocations as usize);
                let fits = |g: usize| n % (16 * g) == 0 && 2 * n * hdp * 2 + g * 16 * 32 * 2 <= shm && sg as usize * g <= inv;
                let nsg: usize = std::env::var("BRP_DAT_NSG").ok().and_then(|v| v.parse().ok()).unwrap_or_else(|| [16, 8, 4, 2, 1].into_iter().find(|&g| fits(g)).unwrap_or(1));
                let coop = coop && fits(nsg);
                let kern = if coop {
                    rec.label.push_str(" (tensor)");
                    self.variant(3000 + n * 64 + hdp + nsg * 1_000_000, 0, || {
                        self.compute.kernel(&format!("nn_dat_attn_coop_{n}_{hdp}_{nsg}"), &format!("{PRELUDE}#define N {n}\n#define HDP {hdp}\n#define SG {sg}\n#define NSG {nsg}\n{HALF}{ACT}{DAT_ATTN_COOP}"), 0)
                    })?
                } else {
                    self.variant(200 + n, 0, || self.compute.kernel(&format!("nn_dat_attn_{n}"), &format!("{PRELUDE}#define N {n}\n{HALF}{ACT}{DAT_ATTN}"), 0))?
                };
                let out = self.new_tensor_of(c, x.h, x.w, self.half)?;
                let table = ((2 * s0 - 1) * (2 * s1 - 1) * heads) as u64 * 4;
                for b in 0..2 {
                    let (wh, ww) = if b == 0 { (s0, s1) } else { (s1, s0) };
                    let (sy, sx) = if shifted { (wh / 2, ww / 2) } else { (0, 0) };
                    let push = Push::new()
                        .t(x)
                        .t(&out)
                        .ptr_at(&self.weights, l.weight + b as u64 * table)
                        .i(c as i32)
                        .i(x.h as i32)
                        .i(x.w as i32)
                        .i(heads as i32)
                        .i(hd as i32)
                        .i((b * c / 2) as i32)
                        .i(wh as i32)
                        .i(ww as i32)
                        .i(sy as i32)
                        .i(sx as i32)
                        .i(x.half as i32)
                        .i(out.half as i32);
                    rec.dispatch(&kern, [((x.h / wh) * (x.w / ww)) as u32, heads as u32, 1], &push)?;
                }
                vec![out]
            }
            "ChannelAttention" => {
                let x = &ins[0];
                let (c, heads) = (p.i(0, 0) as usize, p.i(1, 1) as usize);
                let hd = c / heads;
                if x.c != 3 * c || hd > 32 {
                    bail!("channel attention: {} input channels for {c}, {heads} heads", x.c);
                }
                let plane = x.h * x.w;
                // Chunks of 1024 pixels: enough workgroups to fill the GPU (4096: 25% slower).
                let chunk = std::env::var("BRP_CA_CHUNK").ok().and_then(|v| v.parse().ok()).unwrap_or(1024usize);
                let chunks = plane.div_ceil(chunk);
                let items = hd * hd + 2 * hd;
                let part = self.pool.get(&self.compute, (chunks * heads * items * 4) as u64)?;
                let attn = self.pool.get(&self.compute, (heads * hd * hd * 4) as u64)?;
                let push = Push::new().t(x).ptr_at(&part, 0).i(c as i32).i(plane as i32).i(hd as i32).i(chunk as i32).i(x.half as i32);
                let gram = if self.compute.ctx.coopmat && std::env::var_os("BRP_NN_NAIVE").is_none() {
                    let sg = self.compute.ctx.subgroup_size;
                    self.variant(4000, 0, || self.compute.kernel("nn_ca_gram_coop", &format!("{PRELUDE}#define SG {sg}\n{HALF}{ACT}{CA_GRAM_COOP}"), 0))?
                } else {
                    k.ca_gram.clone()
                };
                rec.label = "ChannelAttention sums".into();
                rec.dispatch(&gram, [chunks as u32, heads as u32, 1], &push)?;
                rec.label = "ChannelAttention softmax".into();
                let push = Push::new().ptr_at(&part, 0).ptr_at(&self.weights, l.weight).ptr_at(&attn, 0).i(hd as i32).i(chunks as i32).i(heads as i32);
                rec.dispatch(&k.ca_softmax, [heads as u32, 1, 1], &push)?;
                let out = self.new_tensor_of(c, x.h, x.w, self.half)?;
                rec.label = "ChannelAttention apply".into();
                let push = Push::new().t(x).ptr_at(&attn, 0).t(&out).i(c as i32).i(plane as i32).i(hd as i32).i(x.half as i32).i(out.half as i32);
                if self.compute.ctx.coopmat && std::env::var_os("BRP_NN_NAIVE").is_none() {
                    let sg = self.compute.ctx.subgroup_size;
                    let kern = self.variant(4001, 0, || self.compute.kernel("nn_ca_apply_coop", &format!("{PRELUDE}#define SG {sg}\n{HALF}{ACT}{CA_APPLY_COOP}"), 0))?;
                    rec.dispatch(&kern, [plane.div_ceil(64) as u32, heads as u32, 1], &push)?;
                } else {
                    rec.dispatch(&k.ca_apply, [plane.div_ceil(256) as u32, heads as u32, 1], &push)?;
                }
                // Back to the pool: later dispatches of this recording run after these.
                self.pool.put(part);
                self.pool.put(attn);
                vec![out]
            }
            "AIM" => {
                let (att, conv, cmap, smap) = (&ins[0], &ins[1], &ins[2], &ins[3]);
                if (conv.c, conv.h, conv.w) != (att.c, att.h, att.w) || cmap.len() != att.c || cmap.half || (smap.c, smap.h, smap.w) != (1, att.h, att.w) {
                    bail!("interaction: sizes do not match");
                }
                let out = self.new_tensor_of(att.c, att.h, att.w, self.half)?;
                let push = Push::new()
                    .t(att)
                    .t(conv)
                    .t(cmap)
                    .t(smap)
                    .t(&out)
                    .i(att.len() as i32)
                    .i((att.h * att.w) as i32)
                    .i(p.i(0, 0))
                    .i(att.half as i32)
                    .i(conv.half as i32)
                    .i(smap.half as i32)
                    .i(out.half as i32);
                rec.dispatch(&k.aim, flat(&k.aim, att.len()), &push)?;
                vec![out]
            }
            "Interp" => {
                let x = &ins[0];
                let (mut oh, mut ow) = (p.i(3, 0) as usize, p.i(4, 0) as usize);
                // A second input: the size to resize to (ncnn's reference blob).
                if let Some(r) = ins.get(1) {
                    (oh, ow) = (r.h, r.w);
                }
                if oh == 0 || ow == 0 {
                    oh = (x.h as f32 * p.f(1, 1.0)) as usize;
                    ow = (x.w as f32 * p.f(2, 1.0)) as usize;
                }
                let out = self.new_tensor(x.c, oh, ow)?;
                let mode = p.i(0, 1);
                if !(1..=3).contains(&mode) {
                    bail!("resize type {mode}");
                }
                let push = Push::new().t(&x).t(&out).i(x.c as i32).i(x.h as i32).i(x.w as i32).i(oh as i32).i(ow as i32).i(mode).i(p.i(6, 0));
                rec.dispatch(&k.interp, g3(&k.interp, ow, oh, x.c), &push)?;
                vec![out]
            }
            "Concat" => {
                if p.i(0, 0) != 0 {
                    bail!("concat on axis {}", p.i(0, 0));
                }
                let (h, w) = (ins[0].h, ins[0].w);
                let c: usize = ins.iter().map(|t| t.c).sum();
                // Planned: the inputs already sit one after another in one buffer.
                let mut next = ins[0].off;
                let half = ins[0].half;
                let contiguous = ins.iter().all(|t| {
                    let ok = Arc::ptr_eq(&t.buf, &ins[0].buf) && t.off == next && (t.h, t.w, t.half) == (h, w, half);
                    next += t.len() as u64 * t.esize();
                    ok
                });
                if contiguous {
                    return Ok(vec![Tensor { buf: ins[0].buf.clone(), off: ins[0].off, c, h, w, half }]);
                }
                // Copied (as f32).
                let out = self.new_tensor(c, h, w)?;
                let mut c0 = 0;
                for t in ins {
                    let t32 = self.cast(rec, t, false)?;
                    self.copy(rec, &t32, (0, 0, 0), (t.c, h, w), &out, (c0, 0, 0))?;
                    if t.half {
                        self.pool.put(t32.buf);
                    }
                    c0 += t.c;
                }
                vec![out]
            }
            "Slice" => {
                let axis = p.i(1, 0);
                let sizes = p.arr(0);
                let x = &ins[0];
                let total = [x.c, x.h, x.w][axis as usize];
                let n = l.outputs.len();
                let mut outs = Vec::new();
                let mut start = 0usize;
                for i in 0..n {
                    let len = match sizes.get(i) {
                        Some(&s) if s >= 0.0 => s as usize,
                        _ => (total - start) / (n - i),
                    };
                    let (mut c, mut h, mut w) = (x.c, x.h, x.w);
                    let mut off = (0, 0, 0);
                    match axis {
                        0 => (c, off.0) = (len, start),
                        1 => (h, off.1) = (len, start),
                        _ => (w, off.2) = (len, start),
                    }
                    let out = self.new_tensor(c, h, w)?;
                    self.copy(rec, x, off, (c, h, w), &out, (0, 0, 0))?;
                    outs.push(out);
                    start += len;
                }
                outs
            }
            "Crop" => {
                let x = &ins[0];
                let (starts, ends, axes) = (p.arr(9), p.arr(10), p.arr(11));
                if starts.is_empty() {
                    bail!("old-style Crop parameters");
                }
                let mut lo = [0usize; 3];
                let mut hi = [x.c, x.h, x.w];
                for (j, &a) in axes.iter().enumerate() {
                    let a = a as usize;
                    let dim = hi[a];
                    let s = starts[j].max(0.0) as usize;
                    let e = ends.get(j).copied().unwrap_or(dim as f64);
                    let e = if e >= 2147483647.0 || e < 0.0 { dim } else { (e as usize).min(dim) };
                    lo[a] = s.min(dim);
                    hi[a] = e;
                }
                let size = (hi[0] - lo[0], hi[1] - lo[1], hi[2] - lo[2]);
                let out = self.new_tensor(size.0, size.1, size.2)?;
                self.copy(rec, x, (lo[0], lo[1], lo[2]), size, &out, (0, 0, 0))?;
                vec![out]
            }
            "PixelShuffle" => {
                let x = &ins[0];
                let r = p.i(0, 1) as usize;
                if p.i(1, 0) != 0 {
                    bail!("pixel shuffle mode {}", p.i(1, 0));
                }
                let oc = x.c / (r * r);
                let out = self.new_tensor(oc, x.h * r, x.w * r)?;
                let push = Push::new().t(&x).t(&out).i(oc as i32).i(x.h as i32).i(x.w as i32).i(r as i32);
                rec.dispatch(&k.pixel_shuffle, g3(&k.pixel_shuffle, x.w * r, x.h * r, oc), &push)?;
                vec![out]
            }
            "Pooling" => {
                let x = &ins[0];
                if p.i(0, 0) != 1 {
                    bail!("max pooling");
                }
                if p.i(4, 0) != 0 {
                    return Ok(vec![self.global_mean(rec, x)?]);
                }
                let (kk, s, out) = if p.i(4, 0) != 0 {
                    (0, 1, self.new_tensor(x.c, 1, 1)?)
                } else {
                    let kk = p.i(1, 1) as usize;
                    let s = p.i(2, 1) as usize;
                    if p.i(3, 0) != 0 || p.i(14, 0) != 0 {
                        bail!("padded pooling");
                    }
                    (kk, s, self.new_tensor(x.c, (x.h - kk) / s + 1, (x.w - kk) / s + 1)?)
                };
                let push = Push::new().t(&x).t(&out).i(x.c as i32).i(x.h as i32).i(x.w as i32).i(out.h as i32).i(out.w as i32).i(kk as i32).i(s as i32);
                rec.dispatch(&k.pool, g3(&k.pool, out.w, out.h, x.c), &push)?;
                vec![out]
            }
            "GroupNorm" => {
                let x = &ins[0];
                let (g, c) = (p.i(0, 1) as usize, p.i(1, 0) as usize);
                if c != x.c || g == 0 || c % g != 0 {
                    bail!("group norm of {} channels as {c} in {g} groups", x.c);
                }
                let out = self.new_tensor_of(x.c, x.h, x.w, self.half)?;
                let push = Push::new().t(x).t(&out).ptr_at(&self.weights, l.weight).i(c as i32).i((x.h * x.w) as i32).i(g as i32).f(p.f(2, 1e-5)).i(x.half as i32).i(out.half as i32);
                rec.dispatch(&k.group_norm, [g as u32, 1, 1], &push)?;
                vec![out]
            }
            // NAFNet's local average (TLC): 0 = box height, 1 = width, in this map's
            // pixels; a box covering the whole map is the global average ([C][1][1]).
            "BoxMean" => {
                let x = &ins[0];
                let (kh, kw) = (p.i(0, 1) as usize, p.i(1, p.i(0, 1)) as usize);
                if kh >= x.h && kw >= x.w {
                    return Ok(vec![self.global_mean(rec, x)?]);
                }
                let (k1, k2) = (kh.min(x.h), kw.min(x.w));
                if x.w > 8192 {
                    bail!("box mean of a map {} wide", x.w);
                }
                let t = self.pool.tensor_of(&self.compute, x.c, x.h, x.w, false)?;
                let push = Push::new().t(x).t(&t).i(x.c as i32).i(x.h as i32).i(x.w as i32).i(k1 as i32).i(x.half as i32);
                rec.dispatch(&k.box_v, g3(&k.box_v, x.w, 1, x.c), &push)?;
                let out = self.new_tensor_of(x.c, x.h, x.w, self.half)?;
                let push = Push::new().t(&t).t(&out).i(x.c as i32).i(x.h as i32).i(x.w as i32).i(k1 as i32).i(k2 as i32).i(out.half as i32);
                rec.dispatch(&k.box_h, [x.h as u32, x.c as u32, 1], &push)?;
                self.pool.put(t.buf);
                vec![out]
            }
            "Reduction" => {
                let x = &ins[0];
                if p.i(0, 0) != 3 || p.arr(3) != vec![0.0] {
                    bail!("reduction other than the channel mean");
                }
                let out = self.new_tensor(1, x.h, x.w)?;
                let push = Push::new().t(&x).t(&out).i(x.c as i32).i((x.h * x.w) as i32);
                rec.dispatch(&k.mean_c, flat(&k.mean_c, x.h * x.w), &push)?;
                vec![out]
            }
            "rife.Warp" => {
                let (img, flow) = (&ins[0], &ins[1]);
                let out = self.new_tensor(img.c, img.h, img.w)?;
                if flow.h < img.h || flow.w < img.w {
                    bail!("flow smaller than the picture");
                }
                let push = Push::new().t(&img).t(&flow).t(&out).i(img.c as i32).i(img.h as i32).i(img.w as i32).i(flow.w as i32).i((flow.h * flow.w) as i32);
                rec.dispatch(&k.warp, g3(&k.warp, img.w, img.h, img.c), &push)?;
                vec![out]
            }
            t => bail!("layer type {t} not supported"),
        })
    }

    /// Every second pixel of `x` in each direction (stride-2 convolutions). `planned`:
    /// the layer's output (its concat slot, if any), else a scratch tensor.
    fn subsample(&self, rec: &mut Recorder, x: &Tensor, planned: bool) -> Result<Tensor> {
        let (oh, ow) = (x.h.div_ceil(2), x.w.div_ceil(2));
        let out = if planned { self.new_tensor_of(x.c, oh, ow, x.half)? } else { self.pool.tensor_of(&self.compute, x.c, oh, ow, x.half)? };
        let k = &self.k.subsample;
        let push = Push::new().t(x).t(&out).i(x.c as i32).i(x.h as i32).i(x.w as i32).i(oh as i32).i(ow as i32).i(x.half as i32);
        rec.dispatch(k, Recorder::groups(k, [ow as u32, oh as u32, x.c as u32]), &push)?;
        Ok(out)
    }

    /// Copy a (c, h, w) region from `src` at `from` into `dst` at `to`.
    fn copy(&self, rec: &mut Recorder, src: &Tensor, from: (usize, usize, usize), size: (usize, usize, usize), dst: &Tensor, to: (usize, usize, usize)) -> Result<()> {
        let k = &self.k.copy;
        let push = Push::new()
            .t(&src)
            .t(&dst)
            .i(size.0 as i32)
            .i(size.1 as i32)
            .i(size.2 as i32)
            .i(src.h as i32)
            .i(src.w as i32)
            .i(from.0 as i32)
            .i(from.1 as i32)
            .i(from.2 as i32)
            .i(dst.h as i32)
            .i(dst.w as i32)
            .i(to.0 as i32)
            .i(to.1 as i32)
            .i(to.2 as i32);
        rec.dispatch(k, Recorder::groups(k, [size.2 as u32, size.1 as u32, size.0 as u32]), &push)
    }
}

/// Workgroups for `n` items of a flat kernel (index over x and y: one dimension is
/// limited to 65535 workgroups).
fn flat(k: &Kernel, n: usize) -> [u32; 3] {
    let groups = n.div_ceil(k.local[0] as usize).max(1);
    let x = groups.min(65535);
    [x as u32, groups.div_ceil(x) as u32, 1]
}

/// f32 to IEEE half (round to nearest even; overflow to infinity).
fn f32_to_f16(v: f32) -> u16 {
    let b = v.to_bits();
    let sign = ((b >> 16) & 0x8000) as u16;
    let exp = ((b >> 23) & 0xff) as i32;
    let man = b & 0x7f_ffff;
    if exp == 255 {
        return sign | 0x7c00 | if man != 0 { 0x200 } else { 0 };
    }
    let e = exp - 127 + 15;
    if e >= 31 {
        return sign | 0x7c00;
    }
    if e <= 0 {
        if e < -10 {
            return sign;
        }
        let m = (man | 0x80_0000) >> (1 - e);
        let round = (m >> 12) & 1;
        return sign | ((m >> 13) + round) as u16;
    }
    let m = man >> 13;
    let rest = man & 0x1fff;
    let mut h = sign as u32 | ((e as u32) << 10) | m;
    if rest > 0x1000 || (rest == 0x1000 && (m & 1) == 1) {
        h += 1;
    }
    h as u16
}

/// Passing tensors to kernels: the pointer to their first value.
pub trait PushTensor {
    fn t(self, t: &Tensor) -> Push;
}

impl PushTensor for Push {
    fn t(self, t: &Tensor) -> Push {
        self.ptr_at(&t.buf, t.off)
    }
}
