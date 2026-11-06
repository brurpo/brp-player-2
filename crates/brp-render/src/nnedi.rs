//! NNEDI3 weights: parsed and normalised exactly like FFmpeg's `vf_nnedi.c`
//! (`read_weights` + `subtract_mean_*`), then packed into one buffer for the GPU.

use anyhow::{bail, Result};

pub const WEIGHTS_SIZE: usize = 13_574_928;
const XDIM: [usize; 7] = [8, 16, 32, 48, 8, 16, 32];
const YDIM: [usize; 7] = [6, 6, 6, 6, 4, 4, 4];
const NNS: [usize; 5] = [16, 32, 64, 128, 256];

/// Network choice (FFmpeg option names in comments).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct NnediConfig {
    /// Window: 0 s8x6, 1 s16x6, 2 s32x6, 3 s48x6, 4 s8x4, 5 s16x4, 6 s32x4.
    pub nsize: usize,
    /// Neurons: 0 n16, 1 n32, 2 n64, 3 n128, 4 n256.
    pub nns: usize,
    /// 0 = weights trained for absolute error, 1 = squared error.
    pub etype: usize,
    /// Run both quality networks and average them ("slow").
    pub qual2: bool,
}

impl Default for NnediConfig {
    /// FFmpeg's defaults: s32x4, n32, abs, fast.
    fn default() -> Self {
        NnediConfig { nsize: 6, nns: 1, etype: 0, qual2: false }
    }
}

/// Packed weights for one configuration.
/// Layout (floats): prescreener "new" [k0 4x64][b0 4][k1 4x4][b1 4], then the predictor
/// [softmax q1 nns*n][elliott q1 nns*n][softmax bias q1 nns][elliott bias q1 nns]
/// and the same four arrays for q2.
pub struct NnediWeights {
    pub data: Vec<f32>,
    pub xdim: u32,
    pub ydim: u32,
    pub nns: u32,
    pub model_offset: u32,
}

struct Prescreener {
    k0: Vec<f32>, // 4 x stride
    b0: [f32; 4],
    k1: Vec<f32>,
    b1: [f32; 4],
}

struct Model {
    nsize: usize,
    nns: usize,
    sm_q1: Vec<f32>,
    el_q1: Vec<f32>,
    smb_q1: Vec<f32>,
    elb_q1: Vec<f32>,
    sm_q2: Vec<f32>,
    el_q2: Vec<f32>,
    smb_q2: Vec<f32>,
    elb_q2: Vec<f32>,
}

struct Reader<'a> {
    f: &'a [f32],
    pos: usize,
}

impl Reader<'_> {
    fn take(&mut self, n: usize) -> Vec<f32> {
        let v = self.f[self.pos..self.pos + n].to_vec();
        self.pos += n;
        v
    }
}

fn mean(v: &[f32]) -> f32 {
    v.iter().sum::<f32>() / v.len() as f32
}

impl NnediWeights {
    pub fn load(bytes: &[u8], cfg: NnediConfig) -> Result<Self> {
        if bytes.len() != WEIGHTS_SIZE {
            bail!("nnedi3_weights.bin: unexpected size {} (expected {WEIGHTS_SIZE})", bytes.len());
        }
        let floats: Vec<f32> = bytes.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect();
        let mut r = Reader { f: &floats, pos: 0 };

        // Prescreener "original" (skipped: we use the "new" one, FFmpeg's default).
        r.take(4 * 48 + 4 + 4 * 4 + 4 + 4 * 8 + 4);

        // Three "new" prescreeners; FFmpeg's pscrn=2 uses the first.
        let mut pre: Option<Prescreener> = None;
        for i in 0..3 {
            let k0s = r.take(4 * 64);
            let b0 = r.take(4);
            let k1s = r.take(4 * 4);
            let b1 = r.take(4);
            if i == 0 {
                let mut k0 = vec![0f32; 4 * 64];
                let mut k1 = vec![0f32; 16];
                for n in 0..4 {
                    for k in 0..64 {
                        k0[n * 64 + k] = k0s[(k / 8) * 32 + n * 8 + k % 8];
                    }
                    for k in 0..4 {
                        k1[n * 4 + k] = k1s[k * 4 + n];
                    }
                }
                pre = Some(Prescreener { k0, b0: [b0[0], b0[1], b0[2], b0[3]], k1, b1: [b1[0], b1[1], b1[2], b1[3]] });
            }
        }
        let mut pre = pre.unwrap();

        // Predictors: [etype][nns][nsize].
        let mut chosen: Option<Model> = None;
        for m in 0..2 {
            for (i, &nns) in NNS.iter().enumerate() {
                for j in 0..7 {
                    let nsize = XDIM[j] * YDIM[j];
                    let fsz = nns * nsize;
                    let model = Model {
                        nsize,
                        nns,
                        sm_q1: r.take(fsz),
                        el_q1: r.take(fsz),
                        smb_q1: r.take(nns),
                        elb_q1: r.take(nns),
                        sm_q2: r.take(fsz),
                        el_q2: r.take(fsz),
                        smb_q2: r.take(nns),
                        elb_q2: r.take(nns),
                    };
                    if m == cfg.etype && i == cfg.nns && j == cfg.nsize {
                        chosen = Some(model);
                    }
                }
            }
        }
        if r.pos != floats.len() {
            bail!("nnedi3_weights.bin: layout mismatch ({} of {} floats read)", r.pos, floats.len());
        }
        let mut model = chosen.unwrap();

        // subtract_mean_new with half = 255/2.
        let half = 255.0 / 2.0;
        for n in 0..4 {
            let k = &mut pre.k0[n * 64..(n + 1) * 64];
            let m = mean(k);
            for v in k.iter_mut() {
                *v = (*v - m) / half;
            }
        }
        subtract_mean_predictor(&mut model.sm_q1, &mut model.el_q1, &mut model.smb_q1, model.nns, model.nsize);
        subtract_mean_predictor(&mut model.sm_q2, &mut model.el_q2, &mut model.smb_q2, model.nns, model.nsize);

        let mut data = Vec::new();
        data.extend_from_slice(&pre.k0);
        data.extend_from_slice(&pre.b0);
        data.extend_from_slice(&pre.k1);
        data.extend_from_slice(&pre.b1);
        let model_offset = data.len() as u32;
        for part in [&model.sm_q1, &model.el_q1, &model.smb_q1, &model.elb_q1, &model.sm_q2, &model.el_q2, &model.smb_q2, &model.elb_q2] {
            data.extend_from_slice(part);
        }
        Ok(NnediWeights {
            data,
            xdim: XDIM[cfg.nsize] as u32,
            ydim: YDIM[cfg.nsize] as u32,
            nns: NNS[cfg.nns] as u32,
            model_offset,
        })
    }
}

/// FFmpeg `subtract_mean_predictor` for one quality level.
fn subtract_mean_predictor(sm: &mut [f32], el: &mut [f32], smb: &mut [f32], nns: usize, nsize: usize) {
    let scale = 1.0 / nns as f64;
    let mut sm_means = vec![0f64; nns];
    let mut el_means = vec![0f64; nns];
    let mut mean_filter = vec![0f64; nsize];
    for nn in 0..nns {
        sm_means[nn] = mean(&sm[nn * nsize..(nn + 1) * nsize]) as f64;
        el_means[nn] = mean(&el[nn * nsize..(nn + 1) * nsize]) as f64;
        for k in 0..nsize {
            mean_filter[k] += sm[nn * nsize + k] as f64 - sm_means[nn];
        }
    }
    for v in &mut mean_filter {
        *v *= scale;
    }
    let mean_bias = mean(smb) as f64;
    for nn in 0..nns {
        for k in 0..nsize {
            let i = nn * nsize + k;
            sm[i] = (sm[i] as f64 - (sm_means[nn] + mean_filter[k])) as f32;
            el[i] = (el[i] as f64 - el_means[nn]) as f32;
        }
        smb[nn] = (smb[nn] as f64 - mean_bias) as f32;
    }
}
