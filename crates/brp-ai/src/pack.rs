//! Export output on the GPU: the finished picture (a planar RGB(A) tensor) is scaled
//! to the output size, placed on the output canvas (fit, fill, stretch), converted
//! when the export changes the colour encoding (HDR to SDR, to linear for EXR), and
//! written straight into the encoder's pixel format (planar or packed, 8 to 16 bits,
//! float or half float, any chroma subsampling; see `brp_media::encode::PackLayout`).
//! Only those bytes are copied back for the encoder.

use crate::gpu::Ops;
use crate::vk::{PushTensor, Tensor};
use anyhow::{ensure, Context as _, Result};
use brp_media::encode::PackLayout;
use brp_render::compute::{Buffer, Kernel, Push, Recorder, PRELUDE};
use std::sync::Arc;

/// One axis of a resize (Catmull-Rom, widened when shrinking so it averages over the
/// pixels it drops). `axis` 0 resizes x, 1 resizes y. Output always has 4 channels.
const RESIZE: &str = r#"
layout(local_size_x = 16, local_size_y = 8, local_size_z = 1) in;
layout(push_constant) uniform P { F32 a; F32 o; int C, W, H, OW, OH, axis, flags; } p;
float cubic(float x) {
    x = abs(x);
    return x < 1.0 ? (1.5 * x - 2.5) * x * x + 1.0 : (x < 2.0 ? ((-0.5 * x + 2.5) * x - 4.0) * x + 2.0 : 0.0);
}
vec4 load(int x, int y) {
    int n = p.W * p.H, i = y * p.W + x;
    vec4 v = vec4(p.a.v[i], p.a.v[n + i], p.a.v[2 * n + i], p.C > 3 ? p.a.v[3 * n + i] : 1.0);
    // The picture the player shows is half float.
    if ((p.flags & 1) != 0) v = vec4(unpackHalf2x16(packHalf2x16(v.xy)), unpackHalf2x16(packHalf2x16(v.zw)));
    if ((p.flags & 2) != 0) v.rgb *= v.a;
    return v;
}
void main() {
    int x = int(gl_GlobalInvocationID.x), y = int(gl_GlobalInvocationID.y);
    if (x >= p.OW || y >= p.OH) return;
    int sn = p.axis == 0 ? p.W : p.H, dn = p.axis == 0 ? p.OW : p.OH, pos = p.axis == 0 ? x : y;
    vec4 acc = vec4(0.0);
    if (sn == dn) {
        acc = load(x, y);
    } else {
        float scale = max(float(sn) / float(dn), 1.0);
        float c = (float(pos) + 0.5) * float(sn) / float(dn) - 0.5;
        int lo = int(floor(c - 2.0 * scale)) + 1, hi = int(floor(c + 2.0 * scale));
        float ws = 0.0;
        for (int i = lo; i <= hi; i++) {
            float w = cubic((float(i) - c) / scale);
            int ii = clamp(i, 0, sn - 1);
            acc += w * (p.axis == 0 ? load(ii, y) : load(x, ii));
            ws += w;
        }
        acc /= ws;
    }
    int on = p.OW * p.OH, oi = y * p.OW + x;
    p.o.v[oi] = acc.r; p.o.v[on + oi] = acc.g; p.o.v[2 * on + oi] = acc.b; p.o.v[3 * on + oi] = acc.a;
}
"#;

/// The picture placed on the output canvas, with the colour conversion; RGBA out.
/// flags: 1 round to half float, 2 input premultiplied, 4 keep alpha (else
/// composite over black), 8 BT.2020 primaries to BT.709 (HDR conversions).
/// conv: 0 none, 1 PQ to SDR, 2 HLG to SDR, 3 linear to gamma 2.2 (as the viewer
/// shows linear files), 4 SDR to linear, 5 PQ to linear, 6 HLG to linear (1.0 = SDR
/// white, 203 nits).
const CANVAS: &str = r#"
layout(local_size_x = 16, local_size_y = 8, local_size_z = 1) in;
layout(push_constant) uniform P { F32 a; F32 o; int C, CW, CH, W, H, ox, oy, flags, conv; float peak; } p;
const float SDR_WHITE_NITS = 203.0;
vec3 pq_to_nits(vec3 e) {
    const float m1 = 0.1593017578125, m2 = 78.84375;
    const float c1 = 0.8359375, c2 = 18.8515625, c3 = 18.6875;
    vec3 q = pow(clamp(e, 0.0, 1.0), vec3(1.0 / m2));
    return pow(max(q - c1, 0.0) / (c2 - c3 * q), vec3(1.0 / m1)) * 10000.0;
}
vec3 hlg_to_nits(vec3 e) {
    const float a = 0.17883277, b = 0.28466892, c = 0.55991073;
    vec3 lo = e * e / 3.0;
    vec3 hi = (exp((e - c) / a) + b) / 12.0;
    vec3 scene = mix(lo, hi, step(0.5, e));
    float ys = dot(scene, vec3(0.2627, 0.6780, 0.0593));
    return scene * pow(max(ys, 1e-6), 0.2) * 1000.0;
}
vec3 bt2020_to_709(vec3 c) {
    return mat3(1.6605, -0.1246, -0.0182, -0.5876, 1.1329, -0.1006, -0.0728, -0.0083, 1.1187) * c;
}
vec3 tonemap(vec3 c, float peak) {
    float m = max(max(c.r, c.g), c.b);
    if (m <= 0.0) return vec3(0.0);
    float t = m * (1.0 + m / (peak * peak)) / (1.0 + m);
    return c * (t / m);
}
void main() {
    int x = int(gl_GlobalInvocationID.x), y = int(gl_GlobalInvocationID.y);
    if (x >= p.W || y >= p.H) return;
    int cx = x - p.ox, cy = y - p.oy;
    bool keep_alpha = (p.flags & 4) != 0;
    vec4 v;
    if (cx < 0 || cy < 0 || cx >= p.CW || cy >= p.CH) {
        v = vec4(0.0, 0.0, 0.0, keep_alpha ? 0.0 : 1.0);
    } else {
        int n = p.CW * p.CH, i = cy * p.CW + cx;
        v = vec4(p.a.v[i], p.a.v[n + i], p.a.v[2 * n + i], p.C > 3 ? p.a.v[3 * n + i] : 1.0);
        if ((p.flags & 1) != 0) v = vec4(unpackHalf2x16(packHalf2x16(v.xy)), unpackHalf2x16(packHalf2x16(v.zw)));
        if ((p.flags & 2) != 0) v.rgb = v.a > 1e-6 ? v.rgb / v.a : vec3(0.0);
        vec3 rgb = v.rgb;
        if (p.conv == 1 || p.conv == 2 || p.conv == 5 || p.conv == 6) {
            vec3 nits = (p.conv == 1 || p.conv == 5) ? pq_to_nits(rgb) : hlg_to_nits(clamp(rgb, 0.0, 1.0));
            vec3 lin = nits / SDR_WHITE_NITS;
            if ((p.flags & 8) != 0) lin = bt2020_to_709(lin);
            rgb = p.conv <= 2 ? pow(clamp(tonemap(max(lin, 0.0), p.peak), 0.0, 1.0), vec3(1.0 / 2.4)) : lin;
        } else if (p.conv == 3) {
            rgb = pow(clamp(rgb, 0.0, 1.0), vec3(1.0 / 2.2));
        } else if (p.conv == 4) {
            rgb = pow(max(rgb, 0.0), vec3(2.4));
        }
        v.rgb = rgb;
    }
    if (!keep_alpha) {
        v.rgb *= clamp(v.a, 0.0, 1.0);
        v.a = 1.0;
    }
    int n = p.W * p.H, i = y * p.W + x;
    p.o.v[i] = v.r; p.o.v[n + i] = v.g; p.o.v[2 * n + i] = v.b; p.o.v[3 * n + i] = v.a;
}
"#;

/// One plane of the canvas written in the encoder's format: a thread per 32-bit word.
const PACK: &str = r#"
layout(local_size_x = 64, local_size_y = 1, local_size_z = 1) in;
layout(push_constant) uniform P { F32 a; U32 o; int W, H, PW, PH, SW, SB, STEP, C01, C23, SHIFT, FLT, CHROMA, CWS, CHS, YUV, FULL, BITS; float kr, kb; } p;
vec4 px(int x, int y) {
    int n = p.W * p.H, i = y * p.W + x;
    return vec4(p.a.v[i], p.a.v[n + i], p.a.v[2 * n + i], p.a.v[3 * n + i]);
}
vec4 fetch(int sx, int sy) {
    if (p.CHROMA == 0) return px(min(sx, p.W - 1), min(sy, p.H - 1));
    int bw = 1 << p.CWS, bh = 1 << p.CHS;
    vec4 s = vec4(0.0);
    float cnt = 0.0;
    for (int j = 0; j < bh; j++)
        for (int i = 0; i < bw; i++) {
            // Interlaced 4:2:0 (CHROMA 2): a chroma row averages lines of its own field.
            int X = sx * bw + i, Y = p.CHROMA == 2 ? (sy / 2) * 4 + (sy % 2) + 2 * j : sy * bh + j;
            if (X < p.W && Y < p.H) { s += px(X, Y); cnt += 1.0; }
        }
    return s / max(cnt, 1.0);
}
float value(int comp, vec4 c) {
    if (p.YUV != 0) {
        float kg = 1.0 - p.kr - p.kb;
        float Y = p.kr * c.r + kg * c.g + p.kb * c.b;
        if (comp == 0) return Y;
        if (comp == 1) return (c.b - Y) / (2.0 * (1.0 - p.kb));
        if (comp == 2) return (c.r - Y) / (2.0 * (1.0 - p.kr));
        return c.a;
    }
    return c[comp];
}
uint quant(int comp, float v) {
    if (p.FLT != 0) {
        if (p.SB == 4) return floatBitsToUint(v);
        return packHalf2x16(vec2(v, 0.0)) & 0xffffu;
    }
    float mx = float((1 << p.BITS) - 1);
    float unit = float(1 << max(p.BITS - 8, 0));
    float code;
    if (comp == 3 || p.YUV == 0) code = v * mx;
    else if (p.FULL != 0) code = comp == 0 ? v * mx : v * mx + float(1 << (p.BITS - 1));
    else code = comp == 0 ? (16.0 + 219.0 * v) * unit : (128.0 + 224.0 * v) * unit;
    return uint(clamp(floor(code + 0.5), 0.0, mx)) << uint(p.SHIFT);
}
void main() {
    int wx = int(gl_GlobalInvocationID.x), y = int(gl_GlobalInvocationID.y);
    if (wx >= p.SW || y >= p.PH) return;
    int per = 4 / p.SB;
    uint word = 0u;
    uint mask = p.SB == 4 ? 0xffffffffu : ((1u << uint(8 * p.SB)) - 1u);
    for (int k = 0; k < per; k++) {
        int s = wx * per + k;
        int pi = s / p.STEP, slot = s % p.STEP;
        if (pi >= p.PW) break;
        int code = ((slot < 2 ? p.C01 : p.C23) >> (8 * (slot & 1))) & 255;
        int comp = code - 1;
        if (comp < 0) continue;
        uint q = quant(comp, value(comp, fetch(pi, y)));
        word |= (q & mask) << uint(8 * p.SB * k);
    }
    p.o.v[y * p.SW + wx] = word;
}
"#;

/// Two pictures woven into one interlaced frame: even lines from `a` when
/// `a_top`, else from `b`.
const WEAVE: &str = r#"
layout(local_size_x = 16, local_size_y = 8, local_size_z = 1) in;
layout(push_constant) uniform P { F32 a; F32 b; F32 o; int W, H, a_top; } p;
void main() {
    int x = int(gl_GlobalInvocationID.x), y = int(gl_GlobalInvocationID.y);
    if (x >= p.W || y >= p.H) return;
    bool from_a = ((y & 1) == 0) == (p.a_top != 0);
    int n = p.W * p.H, i = y * p.W + x;
    for (int c = 0; c < 4; c++) p.o.v[c * n + i] = from_a ? p.a.v[c * n + i] : p.b.v[c * n + i];
}
"#;

/// Colour conversion on the way out.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum ColorConv {
    None,
    /// HDR (PQ or HLG) to SDR BT.709 with the viewer's tone mapping. `peak`: HDR
    /// peak relative to SDR white.
    HdrToSdr { pq: bool, bt2020: bool, peak: f32 },
    /// A linear-light source to gamma 2.2 (what the viewer shows).
    LinearToGamma,
    /// To linear light (EXR): from SDR (gamma 2.4), PQ or HLG (1.0 = SDR white).
    ToLinear { from: brp_media::Transfer, bt2020: bool },
}

/// Where the picture goes on the output canvas and how it is written.
#[derive(Clone, Debug)]
pub struct OutputSpec {
    /// The scaled picture's place: x, y (may be negative: cropped), width, height.
    pub content: (i32, i32, u32, u32),
    pub conv: ColorConv,
    /// Keep the alpha channel (else the picture is composited over black).
    pub alpha: bool,
    /// Y'CbCr coefficients (kr, kb) and range for YUV formats.
    pub kr_kb: (f32, f32),
    pub full_range: bool,
    /// Round through half float first, as the player's pictures are (export then
    /// matches the viewer exactly).
    pub as_shown: bool,
    /// The encoder's format and size (the canvas).
    pub layout: PackLayout,
    /// Interlaced output: the first picture is the first field in time; true when
    /// that is the top field (TFF).
    pub tff: bool,
    /// Encode on the GPU (our encoders): the packed planes go to the encoder and
    /// only its packet comes back.
    pub encode: Option<crate::gpuenc::Codec>,
}

pub struct Packer {
    ops: Arc<Ops>,
    resize: Arc<Kernel>,
    canvas: Arc<Kernel>,
    pack: Arc<Kernel>,
    weave: Arc<Kernel>,
    dev: Option<Buffer>,
    host: Option<Buffer>,
    bytes: u64,
    encoder: Option<Box<dyn crate::gpuenc::Encoder>>,
}

fn g2(k: &Kernel, w: usize, h: usize) -> [u32; 3] {
    Recorder::groups(k, [w as u32, h as u32, 1])
}

impl Packer {
    pub fn new(ops: &Arc<Ops>) -> Result<Packer> {
        let c = &ops.compute;
        let k = |name: &str, body: &str| c.kernel(&format!("pack_{name}"), &format!("{PRELUDE}{body}"), 0);
        Ok(Packer { ops: ops.clone(), resize: k("resize", RESIZE)?, canvas: k("canvas", CANVAS)?, pack: k("planes", PACK)?, weave: k("weave", WEAVE)?, dev: None, host: None, bytes: 0, encoder: None })
    }

    /// Record `srcs` (planar RGB(A) f32: one picture, or two fields' pictures in time
    /// order for interlaced output) through scaling, placement, conversion,
    /// weaving and packing into a buffer the CPU reads with [`bytes`] once the
    /// recording ran.
    pub fn record(&mut self, rec: &mut Recorder, srcs: &[&Tensor], spec: &OutputSpec) -> Result<()> {
        ensure!(!srcs.is_empty() && srcs.len() <= 2, "export: one picture or two fields");
        let l = &spec.layout;
        let (w, h) = (l.w as usize, l.h as usize);
        let interlaced = srcs.len() == 2;
        let canvas = if interlaced {
            let a = self.canvas_of(rec, srcs[0], spec)?;
            let b = self.canvas_of(rec, srcs[1], spec)?;
            let out = self.ops.tensor(4, h, w)?;
            rec.dispatch(&self.weave, g2(&self.weave, w, h), &Push::new().t(&a).t(&b).t(&out).i(w as i32).i(h as i32).i(spec.tff as i32))?;
            self.ops.recycle(rec, a);
            self.ops.recycle(rec, b);
            out
        } else {
            self.canvas_of(rec, srcs[0], spec)?
        };
        self.pack_canvas(rec, &canvas, spec, interlaced)?;
        self.ops.recycle(rec, canvas);
        Ok(())
    }

    /// One picture scaled, placed and converted onto the output canvas (RGBA).
    fn canvas_of(&mut self, rec: &mut Recorder, src: &Tensor, spec: &OutputSpec) -> Result<Tensor> {
        ensure!(!src.half && src.c >= 3, "export: unexpected picture tensor");
        let l = &spec.layout;
        let (w, h) = (l.w as usize, l.h as usize);
        let (cx, cy, cw, ch) = spec.content;
        let (cw, ch) = (cw.max(1) as usize, ch.max(1) as usize);
        let resize = (cw, ch) != (src.w, src.h);
        let premul = resize && src.c > 3;
        let round = spec.as_shown as i32;
        // Scaled picture (4 channels), or the source itself.
        let scaled = if resize {
            let tmp = self.ops.tensor(4, src.h, cw)?;
            let push = Push::new().t(src).t(&tmp).i(src.c as i32).i(src.w as i32).i(src.h as i32).i(cw as i32).i(src.h as i32).i(0).i(round | (premul as i32) << 1);
            rec.dispatch(&self.resize, g2(&self.resize, cw, src.h), &push)?;
            let out = self.ops.tensor(4, ch, cw)?;
            let push = Push::new().t(&tmp).t(&out).i(4).i(cw as i32).i(src.h as i32).i(cw as i32).i(ch as i32).i(1).i(0);
            rec.dispatch(&self.resize, g2(&self.resize, cw, ch), &push)?;
            self.ops.recycle(rec, tmp);
            Some(out)
        } else {
            None
        };
        let input = scaled.as_ref().unwrap_or(src);
        let (conv, bt2020, peak) = match spec.conv {
            ColorConv::None => (0, false, 1.0),
            ColorConv::HdrToSdr { pq, bt2020, peak } => (if pq { 1 } else { 2 }, bt2020, peak),
            ColorConv::LinearToGamma => (3, false, 1.0),
            ColorConv::ToLinear { from, bt2020 } => match from {
                brp_media::Transfer::Sdr => (4, false, 1.0),
                brp_media::Transfer::Pq => (5, bt2020, 1.0),
                brp_media::Transfer::Hlg => (6, bt2020, 1.0),
                brp_media::Transfer::Linear => (0, false, 1.0),
            },
        };
        let flags = (if resize { 0 } else { round }) | (premul as i32) << 1 | (spec.alpha as i32) << 2 | (bt2020 as i32) << 3;
        let canvas = self.ops.tensor(4, h, w)?;
        let push = Push::new().t(input).t(&canvas).i(input.c as i32).i(cw as i32).i(ch as i32).i(w as i32).i(h as i32).i(cx).i(cy).i(flags).i(conv).f(peak);
        rec.dispatch(&self.canvas, g2(&self.canvas, w, h), &push)?;
        if let Some(s) = scaled {
            self.ops.recycle(rec, s);
        }
        Ok(canvas)
    }

    /// The canvas written in the encoder's planes, copied for the CPU.
    fn pack_canvas(&mut self, rec: &mut Recorder, canvas: &Tensor, spec: &OutputSpec, interlaced: bool) -> Result<()> {
        let l = &spec.layout;
        let (w, h) = (l.w as usize, l.h as usize);
        if self.bytes < l.total || self.dev.is_none() {
            let bytes = l.total.next_multiple_of(1 << 16);
            self.dev = Some(self.ops.compute.buffer(bytes, "export picture")?);
            self.host = Some(self.ops.compute.readback(bytes, "export readback")?);
            self.bytes = bytes;
        }
        let dev = self.dev.as_ref().unwrap();
        for pl in &l.planes {
            let code = |i: usize| (pl.comps[i] + 1) as i32 & 255;
            let words = (pl.stride / 4) as usize;
            let push = Push::new()
                .t(canvas)
                .ptr_at(dev, pl.offset)
                .i(w as i32)
                .i(h as i32)
                .i(pl.w as i32)
                .i(pl.h as i32)
                .i(words as i32)
                .i(pl.bytes as i32)
                .i(pl.step as i32)
                .i(code(0) | code(1) << 8)
                .i(code(2) | code(3) << 8)
                .i(pl.shift as i32)
                .i(l.float as i32)
                .i(if pl.chroma && interlaced && l.ch > 0 { 2 } else { pl.chroma as i32 })
                .i(l.cw as i32)
                .i(l.ch as i32)
                .i(l.yuv as i32)
                .i(spec.full_range as i32)
                .i(l.bits as i32)
                .f(spec.kr_kb.0)
                .f(spec.kr_kb.1);
            rec.dispatch(&self.pack, g2(&self.pack, words, pl.h as usize), &push)?;
        }
        match &spec.encode {
            Some(codec) => {
                // Our GPU encoder takes the planes from here; only its packet comes back.
                if self.encoder.as_ref().is_none_or(|e| e.codec() != codec) {
                    self.encoder = None;
                    self.encoder = Some(crate::gpuenc::create(&self.ops.compute, codec, l)?);
                }
                self.encoder.as_mut().unwrap().record(rec, dev, l)?;
            }
            None => {
                self.encoder = None;
                rec.copy(dev, 0, self.host.as_ref().unwrap(), 0, l.total);
            }
        }
        Ok(())
    }

    /// The packed picture (valid once the recording that made it has run).
    pub fn bytes(&mut self, len: u64) -> Result<&[u8]> {
        let host = self.host.as_mut().context("nothing packed yet")?;
        Ok(&host.mapped().context("readback not mapped")?[..len as usize])
    }

    /// What the last recording made for the encoder thread: the packed picture
    /// (`len` bytes), or the finished packet when the GPU encoded it.
    pub fn output(&mut self, len: u64) -> Result<Vec<u8>> {
        match &mut self.encoder {
            Some(e) => e.packet(),
            None => Ok(self.bytes(len)?.to_vec()),
        }
    }
}
