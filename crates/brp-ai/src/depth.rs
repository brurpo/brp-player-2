//! Depth from a picture: relative inverse depth (near = large), for depth effects.
//!
//! Two models, the same network but for the temporal part (both Apache-2.0, ViT-S):
//! - Video Depth Anything Small (Chen et al. 2025): steady from frame to frame. Its
//!   head attends, at four places, over earlier frames' hidden states: streamed as its
//!   official video_depth_stream.py, a window of 32 (the first frame after a reset, the
//!   oldest kept one, the latest 29, and the frame itself).
//! - Depth Anything V2 Small: one frame alone (no cache, cheaper; flickers more).
//!
//! The picture is resized (cubic) so its short side is 518 (multiples of 14) and
//! normalized (ImageNet); DINOv2 (12 blocks over 14 px patches), the DPT head, and the
//! depth resized back to the picture (bilinear, align_corners) and kept >= 0.
//! Graphs from `pth2ncnn depth`. Measured against the official networks (written out in
//! tools/airef/depth_ref.py), and for steadiness in tools/airef/depth_stability.py.

use crate::gpu::Ops;
use crate::vk::{Net, PushTensor, Tensor};
use anyhow::{Context, Result};
use brp_render::compute::{Kernel, Push, Recorder, PRELUDE};
use std::path::Path;
use std::sync::{Arc, Mutex};

const DIM: usize = 384;
/// The temporal window (the model's position encodings) and cache slots: the stream's
/// list holds the reset frame and up to 42 later ones before it drops the oldest (all
/// are read again as older entries go), plus a scratch slot.
const WINDOW: usize = 32;
const SLOTS: usize = 44;

/// Resize (OpenCV's INTER_CUBIC: A = -0.75, half-pixel centres, edges replicated) and
/// ImageNet normalization, to [3][oh][ow].
const PREP: &str = r#"
layout(local_size_x = 16, local_size_y = 8, local_size_z = 1) in;
layout(push_constant) uniform P { F32 a; F32 o; int H, W, OH, OW; } p;
float cw(float x) {
    const float A = -0.75;
    x = abs(x);
    return x <= 1.0 ? ((A + 2.0) * x - (A + 3.0)) * x * x + 1.0 : x < 2.0 ? ((A * x - 5.0 * A) * x + 8.0 * A) * x - 4.0 * A : 0.0;
}
void main() {
    int x = int(gl_GlobalInvocationID.x), y = int(gl_GlobalInvocationID.y), c = int(gl_GlobalInvocationID.z);
    if (x >= p.OW || y >= p.OH || c >= 3) return;
    float sx = (float(x) + 0.5) * float(p.W) / float(p.OW) - 0.5, sy = (float(y) + 0.5) * float(p.H) / float(p.OH) - 0.5;
    int x0 = int(floor(sx)), y0 = int(floor(sy));
    float acc = 0.0;
    for (int j = -1; j <= 2; j++) {
        float wy = cw(sy - float(y0 + j));
        int yy = clamp(y0 + j, 0, p.H - 1);
        for (int i = -1; i <= 2; i++) {
            int xx = clamp(x0 + i, 0, p.W - 1);
            acc += wy * cw(sx - float(x0 + i)) * p.a.v[(c * p.H + yy) * p.W + xx];
        }
    }
    const float mean[3] = float[3](0.485, 0.456, 0.406), sd[3] = float[3](0.229, 0.224, 0.225);
    p.o.v[(c * p.OH + y) * p.OW + x] = (acc - mean[c]) / sd[c];
}
"#;

/// The depth (one channel, the network's input size) to the picture's size: bilinear,
/// align_corners=True, kept >= 0.
const OUT_RESIZE: &str = r#"
layout(local_size_x = 16, local_size_y = 8, local_size_z = 1) in;
layout(push_constant) uniform P { F32 a; F32 o; int H, W, OH, OW; } p;
void main() {
    int x = int(gl_GlobalInvocationID.x), y = int(gl_GlobalInvocationID.y);
    if (x >= p.OW || y >= p.OH) return;
    float sx = p.OW > 1 ? float(x) * float(p.W - 1) / float(p.OW - 1) : 0.0;
    float sy = p.OH > 1 ? float(y) * float(p.H - 1) / float(p.OH - 1) : 0.0;
    int x0 = min(int(sx), p.W - 1), y0 = min(int(sy), p.H - 1);
    int x1 = min(x0 + 1, p.W - 1), y1 = min(y0 + 1, p.H - 1);
    float fx = sx - float(x0), fy = sy - float(y0);
    float a = mix(p.a.v[y0 * p.W + x0], p.a.v[y0 * p.W + x1], fx);
    float b = mix(p.a.v[y1 * p.W + x0], p.a.v[y1 * p.W + x1], fx);
    p.o.v[y * p.OW + x] = max(mix(a, b, fy), 0.0);
}
"#;

const F16: &str = r#"
#extension GL_EXT_shader_explicit_arithmetic_types_float16 : require
layout(buffer_reference, scalar, buffer_reference_align = 2) buffer F16S { float16_t v[]; };
float ldx(F32 b, int i, bool h) { return h ? float(F16S(uint64_t(b)).v[i]) : b.v[i]; }
"#;

/// A frame's projected keys or values (f32 or f16) into its cache slot (f16).
const STORE: &str = r#"
layout(local_size_x = 256) in;
layout(push_constant) uniform P { F32 a; F32 o; int n, slot, ah; } p;
void main() {
    int i = int(gl_GlobalInvocationID.x + gl_GlobalInvocationID.y * gl_NumWorkGroups.x * gl_WorkGroupSize.x);
    if (i >= p.n) return;
    F16S(uint64_t(p.o)).v[p.slot * p.n + i] = float16_t(ldx(p.a, i, p.ah != 0));
}
"#;

/// Attention over time at each map position, 8 heads: the frame's query (its own
/// projection plus its window place's) against the keys and values of the window's
/// frames (cache slots, f16; plus each one's window place), softmax, weighted sum.
/// Window place i holds slot `order` byte i; the frame itself is the last place.
const TATTN: &str = r#"
layout(local_size_x = 64) in;
layout(push_constant) uniform P { F32 q; F32 o; F32 k; F32 v; F32 pe; int C, n, T, qh; uint order[8]; } p;
void main() {
    int i = int(gl_GlobalInvocationID.x), h = int(gl_GlobalInvocationID.y);
    if (i >= p.n) return;
    int hd = p.C / 8, c0 = h * hd;
    float q[48], acc[48];
    // pe: [3][32][C] (q, k, v) for this attention block.
    for (int d = 0; d < hd; d++) {
        q[d] = ldx(p.q, (c0 + d) * p.n + i, p.qh != 0) + p.pe.v[(p.T - 1) * p.C + c0 + d];
        acc[d] = 0.0;
    }
    float scale = inversesqrt(float(hd)), m = -1e30, l = 0.0;
    for (int t = 0; t < p.T; t++) {
        int slot = int((p.order[t >> 2] >> ((t & 3) * 8)) & 255u);
        int base = slot * p.C * p.n;
        float s = 0.0;
        for (int d = 0; d < hd; d++)
            s += q[d] * (float(F16S(uint64_t(p.k)).v[base + (c0 + d) * p.n + i]) + p.pe.v[(32 + t) * p.C + c0 + d]);
        s *= scale;
        float mn = max(m, s), r = exp(m - mn), e = exp(s - mn);
        l = l * r + e;
        for (int d = 0; d < hd; d++)
            acc[d] = acc[d] * r + e * (float(F16S(uint64_t(p.v)).v[base + (c0 + d) * p.n + i]) + p.pe.v[(64 + t) * p.C + c0 + d]);
        m = mn;
    }
    for (int d = 0; d < hd; d++) p.o.v[(c0 + d) * p.n + i] = acc[d] / l;
}
"#;

/// Which model.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DepthModel {
    /// Video Depth Anything Small: steady (temporal head with a frame cache).
    Video,
    /// Depth Anything V2 Small: each frame alone.
    Frame,
}

impl DepthModel {
    fn stem(self) -> &'static str {
        match self {
            DepthModel::Video => "vda_vits",
            DepthModel::Frame => "da2_vits",
        }
    }
}

/// One temporal module: its three graphs, the position encodings through its
/// projections ([2 blocks][3][32][C]) and the cache (per block, keys and values: SLOTS x
/// [C][h][w] f16, made at the first frame of a size).
struct Module {
    nets: [Net; 3],
    pe: Tensor,
    c: usize,
    cache: Option<[Tensor; 4]>,
}

/// The stream's window list (slot ids, as the official frame_cache_list) and the frame
/// count since the reset.
struct Stream {
    size: (usize, usize),
    list: Vec<usize>,
    id: usize,
}

pub struct Depth {
    ops: Arc<Ops>,
    pub model: DepthModel,
    dpatch: Net,
    dblocks: Net,
    head: [Net; 4],
    mm: Vec<Mutex<Module>>,
    prep: Arc<Kernel>,
    out_resize: Arc<Kernel>,
    seq_build: Arc<Kernel>,
    seq_strip: Arc<Kernel>,
    store: Arc<Kernel>,
    tattn: Arc<Kernel>,
    cls: Vec<f32>,
    dpos: Vec<f32>,
    pos: Mutex<std::collections::HashMap<(usize, usize), Arc<(Tensor, Tensor)>>>,
    stream: Mutex<Option<Stream>>,
}

impl Depth {
    pub fn available(models: &Path, model: DepthModel) -> bool {
        models.join("depth").join(format!("{}_hd.param", model.stem())).exists()
    }

    pub fn new(ops: &Arc<Ops>, models: &Path, model: DepthModel) -> Result<Self> {
        let c = &ops.compute;
        let dir = models.join("depth");
        let load = |part: &str| -> Result<Net> {
            let stem = dir.join(format!("{}_{part}", model.stem()));
            let param = std::fs::read_to_string(stem.with_extension("param")).with_context(|| format!("{}.param", stem.display()))?;
            let mut net = Net::load(c, &param, &std::fs::read(stem.with_extension("bin"))?)?;
            net.set_half(true);
            net.share_pool(&ops.pool);
            Ok(net)
        };
        let k = |name: &str, body: &str| c.kernel(&format!("depth_{name}"), &format!("{PRELUDE}{body}"), 0);
        let raw: Vec<f32> = std::fs::read(dir.join(format!("{}_pos.bin", model.stem())))?.chunks_exact(4).map(|b| f32::from_le_bytes(b.try_into().unwrap())).collect();
        anyhow::ensure!(raw.len() == DIM + 1370 * DIM, "{}_pos.bin: {} values", model.stem(), raw.len());
        let mut cls = raw[..DIM].to_vec();
        cls.iter_mut().zip(&raw[DIM..2 * DIM]).for_each(|(a, b)| *a += b);
        let mut mm = Vec::new();
        if model == DepthModel::Video {
            for j in 0..4 {
                let pe: Vec<f32> = std::fs::read(dir.join(format!("{}_mm{j}_pe.bin", model.stem())))?.chunks_exact(4).map(|b| f32::from_le_bytes(b.try_into().unwrap())).collect();
                let ch = pe.len() / (2 * 3 * WINDOW);
                anyhow::ensure!(ch % 8 == 0 && ch / 8 <= 48, "temporal module {j}: {ch} channels");
                let pe = Tensor::from_image(c, &crate::Image { w: ch, h: 2 * 3 * WINDOW, channels: 1, data: pe })?;
                mm.push(Mutex::new(Module { nets: [load(&format!("mm{j}_a"))?, load(&format!("mm{j}_b"))?, load(&format!("mm{j}_c"))?], pe, c: ch, cache: None }));
            }
        }
        Ok(Depth {
            ops: ops.clone(),
            model,
            dpatch: load("dpatch")?,
            dblocks: load("dblocks")?,
            head: [load("ha")?, load("hb")?, load("hc")?, load("hd")?],
            mm,
            prep: k("prep", PREP)?,
            out_resize: k("out_resize", OUT_RESIZE)?,
            seq_build: k("seq_build", crate::flow::SEQ_BUILD)?,
            seq_strip: k("seq_strip", crate::flow::SEQ_STRIP)?,
            store: k("store", &format!("{F16}{STORE}"))?,
            tattn: k("tattn", &format!("{F16}{TATTN}"))?,
            cls,
            dpos: crate::flow::planar(&raw[2 * DIM..], 1369, DIM),
            pos: Default::default(),
            stream: Mutex::new(None),
        })
    }

    /// The network's input size for a w x h picture (the official Resize: the short
    /// side to 518, aspect kept, multiples of 14 rounded, at least 518; Video Depth
    /// Anything shrinks it for pictures wider than 16:9).
    pub fn input_size(&self, w: usize, h: usize) -> (usize, usize) {
        let mut size = 518usize;
        let ratio = w.max(h) as f64 / w.min(h) as f64;
        if self.model == DepthModel::Video && ratio > 1.78 {
            size = (((size as f64 * 1.777 / ratio) as usize) as f64 / 14.0).round() as usize * 14;
        }
        let sc = (size as f64 / h as f64).max(size as f64 / w as f64);
        let m = |v: f64| {
            let r = (v / 14.0).round() * 14.0;
            if r < size as f64 { ((v / 14.0).ceil() * 14.0) as usize } else { r as usize }
        };
        (m(w as f64 * sc), m(h as f64 * sc))
    }

    fn pos(&self, ph: usize, pw: usize) -> Result<Arc<(Tensor, Tensor)>> {
        if let Some(p) = self.pos.lock().unwrap().get(&(ph, pw)) {
            return Ok(p.clone());
        }
        let c = &self.ops.compute;
        let dp = crate::flow::bicubic(&self.dpos, DIM, 37, 37, ((ph as f64 + 0.1) / 37.0, (pw as f64 + 0.1) / 37.0), ph, pw);
        let t = |data: Vec<f32>, h: usize, w: usize| Tensor::from_image(c, &crate::Image { w, h, channels: DIM, data });
        let p = Arc::new((t(self.cls.clone(), 1, 1)?, t(dp, ph, pw)?));
        self.pos.lock().unwrap().insert((ph, pw), p.clone());
        Ok(p)
    }

    /// Forget the frames seen (after a seek or a cut): the next frame starts a new
    /// window.
    pub fn reset(&self) {
        *self.stream.lock().unwrap() = None;
    }

    /// Depth of `img` (picture, 3+ channels 0..1): [1][h][w], relative inverse depth.
    /// Video Depth Anything continues its window from the previous call (consecutive
    /// frames; `reset` between unrelated ones).
    pub fn record(&self, rec: &mut Recorder, img: &Tensor) -> Result<Tensor> {
        let ops = &self.ops;
        let (iw, ih) = self.input_size(img.w, img.h);
        let x = ops.tensor(3, ih, iw)?;
        rec.label = "depth: resize".into();
        let push = Push::new().t(img).t(&x).i(img.h as i32).i(img.w as i32).i(ih as i32).i(iw as i32);
        rec.dispatch(&self.prep, Recorder::groups(&self.prep, [iw as u32, ih as u32, 3]), &push)?;

        // DINOv2: patch tokens, the class token and position embeddings, the blocks.
        let (ph, pw) = (ih / 14, iw / 14);
        let pos = self.pos(ph, pw)?;
        let tok = self.dpatch.record(rec, &[("img", &x)], &["tokens"])?.remove(0);
        let n = ph * pw;
        let seq = ops.tensor(DIM, 1, n + 1)?;
        rec.label = "depth: token sequence".into();
        rec.dispatch(&self.seq_build, Recorder::groups(&self.seq_build, [(n + 1) as u32, DIM as u32, 1]), &Push::new().t(&tok).t(&pos.0).t(&pos.1).t(&seq).i(DIM as i32).i(n as i32))?;
        ops.recycle(rec, tok);
        let layers = self.dblocks.record(rec, &[("x", &seq)], &["l2", "l5", "l8", "l11"])?;
        ops.recycle(rec, seq);
        let mut maps = Vec::new();
        for l in layers {
            let m = ops.tensor(DIM, ph, pw)?;
            rec.label = "depth: strip class token".into();
            rec.dispatch(&self.seq_strip, Recorder::groups(&self.seq_strip, [n as u32, DIM as u32, 1]), &Push::new().t(&l).t(&m).i(DIM as i32).i(n as i32))?;
            ops.recycle(rec, l);
            maps.push(m);
        }
        let mut r = self.head[0].record(rec, &[("l2", &maps[0]), ("l5", &maps[1]), ("l8", &maps[2]), ("l11", &maps[3])], &["r0", "r1", "r2", "r3"])?;
        for m in maps {
            ops.recycle(rec, m);
        }
        rec.flush()?;

        let d = if self.model == DepthModel::Video {
            let mut st = self.stream.lock().unwrap();
            let fresh = st.as_ref().is_none_or(|s| s.size != (iw, ih));
            if fresh {
                // The first frame alone fills the window (the official stream), then runs
                // again against it, so the output does not jump at the second frame.
                *st = Some(Stream { size: (iw, ih), list: Vec::new(), id: 0 });
                for m in &self.mm {
                    m.lock().unwrap().cache = None;
                }
                let s = st.as_mut().unwrap();
                let first = self.head_temporal(rec, &r, &[], 0)?;
                ops.recycle(rec, first);
                s.list = vec![0; WINDOW];
                let order: Vec<usize> = s.list[0..2].iter().chain(&s.list[s.list.len() - (WINDOW - 3)..]).copied().collect();
                self.head_temporal(rec, &r, &order, SLOTS - 1)?
            } else {
                let s = st.as_mut().unwrap();
                s.id += 1;
                let order: Vec<usize> = s.list[0..2].iter().chain(&s.list[s.list.len() - (WINDOW - 3)..]).copied().collect();
                let new = (0..SLOTS - 1).find(|i| !s.list.contains(i)).context("depth: no free cache slot")?;
                let d = self.head_temporal(rec, &r, &order, new)?;
                s.list.push(new);
                if s.id + WINDOW > 42 {
                    s.list.remove(1);
                }
                d
            }
        } else {
            let d = self.head_temporal(rec, &r, &[], 0)?;
            d
        };
        for t in r.drain(..) {
            ops.recycle(rec, t);
        }
        let out = ops.tensor(1, img.h, img.w)?;
        rec.label = "depth: to the picture's size".into();
        let push = Push::new().t(&d).t(&out).i(ih as i32).i(iw as i32).i(img.h as i32).i(img.w as i32);
        rec.dispatch(&self.out_resize, Recorder::groups(&self.out_resize, [img.w as u32, img.h as u32, 1]), &push)?;
        ops.recycle(rec, d);
        ops.recycle(rec, x);
        Ok(out)
    }

    /// The DPT head from r0..r3, with the temporal modules when this is Video Depth
    /// Anything: the frame's keys and values go to cache slot `slot`, and it attends
    /// over the slots in `order` then itself (`order` empty: itself alone, window place
    /// 0).
    fn head_temporal(&self, rec: &mut Recorder, r: &[Tensor], order: &[usize], slot: usize) -> Result<Tensor> {
        let ops = &self.ops;
        let video = self.model == DepthModel::Video;
        let tm = |j: usize, x: &Tensor, rec: &mut Recorder| -> Result<Option<Tensor>> {
            if !video {
                return Ok(None);
            }
            Ok(Some(self.temporal(rec, j, x, order, slot)?))
        };
        let r2 = tm(0, &r[2], rec)?;
        let r3 = tm(1, &r[3], rec)?;
        let mut o = self.head[1].record(rec, &[("r2", r2.as_ref().unwrap_or(&r[2])), ("r3", r3.as_ref().unwrap_or(&r[3]))], &["l3rn", "p4"])?;
        for t in [r2, r3].into_iter().flatten() {
            ops.recycle(rec, t);
        }
        let (l3rn, p4) = (o.remove(0), o.remove(0));
        let p4t = tm(2, &p4, rec)?;
        let mut o = self.head[2].record(rec, &[("p4", p4t.as_ref().unwrap_or(&p4)), ("l3rn", &l3rn), ("r1", &r[1])], &["l2rn", "p3"])?;
        for t in [Some(p4), p4t, Some(l3rn)].into_iter().flatten() {
            ops.recycle(rec, t);
        }
        let (l2rn, p3) = (o.remove(0), o.remove(0));
        let p3t = tm(3, &p3, rec)?;
        // The input's size for the output: any tensor of it (r0 is 4x the patch grid).
        let size = ops.tensor(1, r[0].h * 14 / 4, r[0].w * 14 / 4)?;
        let d = self.head[3].record(rec, &[("p3", p3t.as_ref().unwrap_or(&p3)), ("l2rn", &l2rn), ("r0", &r[0]), ("img", &size)], &["depth"])?.remove(0);
        for t in [Some(p3), p3t, Some(l2rn), Some(size)].into_iter().flatten() {
            ops.recycle(rec, t);
        }
        rec.flush()?;
        Ok(d)
    }

    /// Temporal module `j` on map `x` ([C][h][w]).
    fn temporal(&self, rec: &mut Recorder, j: usize, x: &Tensor, order: &[usize], slot: usize) -> Result<Tensor> {
        let ops = &self.ops;
        let mut m = self.mm[j].lock().unwrap();
        let (c, n) = (m.c, x.h * x.w);
        if m.cache.as_ref().is_none_or(|t| t[0].h != x.h || t[0].w != x.w) {
            let mk = || -> Result<Tensor> { Ok(Tensor { buf: Arc::new(ops.compute.buffer((SLOTS * c * n * 2) as u64, "depth cache")?), off: 0, c: SLOTS * c, h: x.h, w: x.w, half: true }) };
            m.cache = Some([mk()?, mk()?, mk()?, mk()?]);
        }
        let cache = m.cache.clone().unwrap();
        let t_len = order.len() + 1;
        let mut packed = [0u32; 8];
        for (i, s) in order.iter().chain(std::iter::once(&slot)).enumerate() {
            packed[i / 4] |= (*s as u32) << ((i % 4) * 8);
        }
        let attend = |rec: &mut Recorder, b: usize, q: &Tensor, k: &Tensor, v: &Tensor| -> Result<Tensor> {
            for (src, dst) in [(k, &cache[2 * b]), (v, &cache[2 * b + 1])] {
                rec.label = "depth: cache".into();
                let push = Push::new().t(src).t(dst).i((c * n) as i32).i(slot as i32).i(src.half as i32);
                let groups = (c * n).div_ceil(256);
                rec.dispatch(&self.store, [groups.min(65535) as u32, groups.div_ceil(65535) as u32, 1], &push)?;
            }
            let o = ops.tensor(c, x.h, x.w)?;
            rec.label = "depth: attention over time".into();
            let pe = m.pe.view(0, 1);
            let pe = Tensor { off: pe.off + (b * 3 * WINDOW * c * 4) as u64, ..pe };
            let mut push = Push::new().t(q).t(&o).t(&cache[2 * b]).t(&cache[2 * b + 1]).t(&pe).i(c as i32).i(n as i32).i(t_len as i32).i(q.half as i32);
            for w in packed {
                push = push.i(w as i32);
            }
            rec.dispatch(&self.tattn, [n.div_ceil(64) as u32, 8, 1], &push)?;
            Ok(o)
        };
        let mut a = m.nets[0].record(rec, &[("x", x)], &["y", "q", "k", "v"])?;
        let (y, q, k, v) = (a.remove(0), a.remove(0), a.remove(0), a.remove(0));
        let o = attend(rec, 0, &q, &k, &v)?;
        for t in [q, k, v] {
            ops.recycle(rec, t);
        }
        let mut b = m.nets[1].record(rec, &[("y", &y), ("o", &o)], &["y1", "q", "k", "v"])?;
        ops.recycle(rec, y);
        ops.recycle(rec, o);
        let (y1, q, k, v) = (b.remove(0), b.remove(0), b.remove(0), b.remove(0));
        let o = attend(rec, 1, &q, &k, &v)?;
        for t in [q, k, v] {
            ops.recycle(rec, t);
        }
        let out = m.nets[2].record(rec, &[("x", x), ("y1", &y1), ("o", &o)], &["out"])?.remove(0);
        ops.recycle(rec, y1);
        ops.recycle(rec, o);
        Ok(out)
    }
}

// ---- depth effects -------------------------------------------------------------------

/// The depth's largest value (float bits compare as uints: depth is >= 0), into r[0].
const RANGE_MAX: &str = r#"
layout(local_size_x = 256) in;
layout(push_constant) uniform P { F32 d; U32 r; int n; } p;
void main() {
    int i = int(gl_GlobalInvocationID.x + gl_GlobalInvocationID.y * gl_NumWorkGroups.x * gl_WorkGroupSize.x);
    if (i < p.n) atomicMax(p.r.v[0], floatBitsToUint(max(p.d.v[i], 0.0)));
}
"#;

/// Histogram of depth / max, 256 bins at r[4..].
const RANGE_HIST: &str = r#"
layout(local_size_x = 256) in;
layout(push_constant) uniform P { F32 d; U32 r; int n; } p;
void main() {
    int i = int(gl_GlobalInvocationID.x + gl_GlobalInvocationID.y * gl_NumWorkGroups.x * gl_WorkGroupSize.x);
    if (i >= p.n) return;
    float m = max(uintBitsToFloat(p.r.v[0]), 1e-12);
    atomicAdd(p.r.v[4 + clamp(int(p.d.v[i] / m * 256.0), 0, 255)], 1u);
}
"#;

/// The 2nd and 98th percentiles from the histogram, as floats at r[1], r[2].
const RANGE_PCT: &str = r#"
layout(local_size_x = 1) in;
layout(push_constant) uniform P { U32 r; int n; } p;
void main() {
    float m = uintBitsToFloat(p.r.v[0]);
    uint lo = uint(float(p.n) * 0.02), hi = uint(float(p.n) * 0.98), acc = 0u;
    float a = 0.0, b = m;
    bool fa = false;
    for (int k = 0; k < 256; k++) {
        acc += p.r.v[4 + k];
        if (!fa && acc > lo) { a = float(k) / 256.0 * m; fa = true; }
        if (acc >= hi) { b = float(k + 1) / 256.0 * m; break; }
    }
    p.r.v[1] = floatBitsToUint(a);
    p.r.v[2] = floatBitsToUint(max(b, a + 1e-6));
}
"#;

/// The picture with a depth effect. Depth scaled by the range to 0 (far) .. 1 (near).
/// mode 0: the depth as grey; 1: lens blur (the blur radius grows with the distance
/// from the focus depth; a sample counts where its own blur reaches this pixel, and a
/// farther one only where this pixel is blurred as much: far things do not bleed over
/// sharp near ones); 2: fog (towards light grey with distance); 3: depth key (alpha:
/// near kept, far transparent, around the focus depth).
const DEPTH_FX: &str = r#"
layout(local_size_x = 16, local_size_y = 8, local_size_z = 1) in;
layout(push_constant) uniform P { F32 img; F32 d; F32 r; F32 o; int C, OC, H, W, mode; float focus, amount, radius; } p;
float dn(int x, int y) {
    float lo = p.r.v[1], hi = p.r.v[2];
    return clamp((p.d.v[clamp(y, 0, p.H - 1) * p.W + clamp(x, 0, p.W - 1)] - lo) / (hi - lo), 0.0, 1.0);
}
float px(int c, int x, int y) { return p.img.v[(c * p.H + clamp(y, 0, p.H - 1)) * p.W + clamp(x, 0, p.W - 1)]; }
void main() {
    int x = int(gl_GlobalInvocationID.x), y = int(gl_GlobalInvocationID.y);
    if (x >= p.W || y >= p.H) return;
    int n = p.H * p.W, i = y * p.W + x;
    float d = dn(x, y);
    vec4 v = vec4(px(0, x, y), px(1, x, y), px(2, x, y), p.C > 3 ? px(3, x, y) : 1.0);
    if (p.mode == 0) {
        v = vec4(d, d, d, 1.0);
    } else if (p.mode == 1) {
        float coc = p.radius * abs(d - p.focus);
        vec4 acc = v / max(coc * coc, 1.0);
        float wsum = 1.0 / max(coc * coc, 1.0);
        const int N = 64;
        for (int k = 1; k < N; k++) {
            float rr = p.radius * sqrt(float(k) / float(N));
            float an = float(k) * 2.39996323;
            int sx = x + int(round(cos(an) * rr)), sy = y + int(round(sin(an) * rr));
            float ds = dn(sx, sy), cs = p.radius * abs(ds - p.focus);
            // Its blur reaches here; behind this pixel it counts only as far as this
            // pixel's own blur.
            float reach = ds >= d - 0.02 ? cs : min(cs, coc);
            if (rr <= reach) {
                float w = 1.0 / max(cs * cs, 1.0);
                acc += w * vec4(px(0, sx, sy), px(1, sx, sy), px(2, sx, sy), p.C > 3 ? px(3, sx, sy) : 1.0);
                wsum += w;
            }
        }
        v = acc / wsum;
    } else if (p.mode == 2) {
        float f = clamp(p.amount * (1.0 - d) * 1.5, 0.0, 1.0);
        v.rgb = mix(v.rgb, vec3(0.82, 0.84, 0.86), f);
    } else {
        float soft = 0.01 + 0.25 * p.amount;
        v.a *= smoothstep(p.focus - soft, p.focus + soft, d);
    }
    for (int c = 0; c < p.OC; c++) p.o.v[c * n + i] = v[c];
}
"#;

/// Depth effects on a picture from its depth.
pub struct DepthFx {
    ops: Arc<Ops>,
    max: Arc<Kernel>,
    hist: Arc<Kernel>,
    pct: Arc<Kernel>,
    fx: Arc<Kernel>,
}

/// What a depth effect makes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DepthOutput {
    Map = 0,
    LensBlur = 1,
    Fog = 2,
    Key = 3,
}

impl DepthFx {
    pub fn new(ops: &Arc<Ops>) -> Result<Self> {
        let k = |name: &str, body: &str| ops.compute.kernel(&format!("depthfx_{name}"), &format!("{PRELUDE}{body}"), 0);
        Ok(DepthFx { ops: ops.clone(), max: k("max", RANGE_MAX)?, hist: k("hist", RANGE_HIST)?, pct: k("pct", RANGE_PCT)?, fx: k("fx", DEPTH_FX)? })
    }

    /// `img` (3+ channels) with the effect, from `depth` ([1][h][w], its size). `focus`
    /// 0 (far) .. 1 (near); `amount` 0..1 (blur size, fog density, key softness).
    pub fn apply(&self, rec: &mut Recorder, img: &Tensor, depth: &Tensor, out: DepthOutput, focus: f32, amount: f32) -> Result<Tensor> {
        let ops = &self.ops;
        let n = depth.h * depth.w;
        let r = ops.filled(rec, 1, 1, 4 + 256, 0.0)?;
        let groups = n.div_ceil(256);
        let g2 = [groups.min(65535) as u32, groups.div_ceil(65535) as u32, 1];
        rec.label = "depth fx: range".into();
        rec.dispatch(&self.max, g2, &Push::new().t(depth).t(&r).i(n as i32))?;
        rec.dispatch(&self.hist, g2, &Push::new().t(depth).t(&r).i(n as i32))?;
        rec.dispatch(&self.pct, [1, 1, 1], &Push::new().t(&r).i(n as i32))?;
        let oc = match out {
            DepthOutput::Key => 4,
            DepthOutput::Map => 3,
            _ => img.c,
        };
        let o = ops.tensor(oc, img.h, img.w)?;
        // Blur radius in pixels: up to 24 at 1080 lines.
        let radius = amount.clamp(0.0, 1.0) * 24.0 * img.h as f32 / 1080.0;
        rec.label = "depth fx".into();
        let push = Push::new().t(img).t(depth).t(&r).t(&o).i(img.c as i32).i(oc as i32).i(img.h as i32).i(img.w as i32).i(out as i32).f(focus).f(amount).f(radius);
        rec.dispatch(&self.fx, Recorder::groups(&self.fx, [img.w as u32, img.h as u32, 1]), &push)?;
        ops.recycle(rec, r);
        Ok(o)
    }
}
