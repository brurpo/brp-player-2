//! Builds the ncnn model of Real-ESRGAN x2plus from the official PyTorch weights
//! (xinntao/Real-ESRGAN release v0.2.1, BSD-3-Clause), reusing the x4plus graph:
//! both are RRDBNet (23 blocks); x2plus folds each 2x2 block of pixels into
//! channels first (pixel_unshuffle, done by `upscale.rs` when it builds a tile), so
//! only its first convolution differs (12 input channels instead of 3).
//!
//!   pth2ncnn <extracted .pth folder> <realesrgan-x4plus.param> <out.param> <out.bin>
//!
//! The .pth is a zip (`unzip` it first): `archive/data.pkl` describes the tensors,
//! `archive/data/<key>` holds each one's raw little-endian floats.

use anyhow::{bail, Context, Result};
use std::collections::HashMap;
use std::path::Path;

#[derive(Clone, Debug)]
#[allow(dead_code)] // values are kept whole for debugging
enum V {
    None,
    Bool(bool),
    Int(i64),
    Float(f64),
    Str(String),
    Tuple(Vec<V>),
    List(Vec<V>),
    Dict(Vec<(V, V)>),
    Global(String),
    /// A call (REDUCE) and the items set on its result afterwards (OrderedDict).
    Call(Box<V>, Box<V>, Vec<(V, V)>),
    PersId(Box<V>),
    Mark,
}

/// Minimal pickle machine: enough for torch.save state dicts.
fn unpickle(b: &[u8]) -> Result<V> {
    let mut st: Vec<V> = Vec::new();
    let mut memo: HashMap<u32, V> = HashMap::new();
    let mut p = 0usize;
    let rd = |p: &mut usize, n: usize| -> Result<&[u8]> {
        let s = b.get(*p..*p + n).context("truncated pickle")?;
        *p += n;
        Ok(s)
    };
    let pop_mark = |st: &mut Vec<V>| -> Result<Vec<V>> {
        let i = st.iter().rposition(|v| matches!(v, V::Mark)).context("no mark")?;
        let items = st.split_off(i + 1);
        st.pop();
        Ok(items)
    };
    let set_items = |target: &mut V, items: Vec<V>| -> Result<()> {
        let pairs = items.chunks(2).map(|c| (c[0].clone(), c[1].clone()));
        match target {
            V::Dict(d) | V::Call(_, _, d) => d.extend(pairs),
            _ => bail!("SETITEMS on {target:?}"),
        }
        Ok(())
    };
    loop {
        let op = rd(&mut p, 1)?[0];
        match op {
            0x80 => p += 1,             // PROTO
            0x95 => p += 8,             // FRAME
            b'}' => st.push(V::Dict(Vec::new())),
            b']' => st.push(V::List(Vec::new())),
            b')' => st.push(V::Tuple(Vec::new())),
            b'(' => st.push(V::Mark),
            b'N' => st.push(V::None),
            0x88 => st.push(V::Bool(true)),
            0x89 => st.push(V::Bool(false)),
            b'K' => st.push(V::Int(rd(&mut p, 1)?[0] as i64)),
            b'M' => st.push(V::Int(u16::from_le_bytes(rd(&mut p, 2)?.try_into()?) as i64)),
            b'J' => st.push(V::Int(i32::from_le_bytes(rd(&mut p, 4)?.try_into()?) as i64)),
            0x8a => {
                // LONG1
                let n = rd(&mut p, 1)?[0] as usize;
                let bytes = rd(&mut p, n)?;
                let mut v: i64 = 0;
                for (i, x) in bytes.iter().enumerate().take(8) {
                    v |= (*x as i64) << (8 * i);
                }
                if n > 0 && n < 8 && bytes[n - 1] & 0x80 != 0 {
                    v -= 1 << (8 * n);
                }
                st.push(V::Int(v));
            }
            b'G' => st.push(V::Float(f64::from_be_bytes(rd(&mut p, 8)?.try_into()?))),
            b'X' => {
                let n = u32::from_le_bytes(rd(&mut p, 4)?.try_into()?) as usize;
                st.push(V::Str(String::from_utf8_lossy(rd(&mut p, n)?).into_owned()));
            }
            0x8c | b'U' => {
                let n = rd(&mut p, 1)?[0] as usize;
                st.push(V::Str(String::from_utf8_lossy(rd(&mut p, n)?).into_owned()));
            }
            b'T' => {
                let n = u32::from_le_bytes(rd(&mut p, 4)?.try_into()?) as usize;
                st.push(V::Str(String::from_utf8_lossy(rd(&mut p, n)?).into_owned()));
            }
            b'c' => {
                let end1 = b[p..].iter().position(|&c| c == b'\n').context("GLOBAL")?;
                let module = String::from_utf8_lossy(&b[p..p + end1]).into_owned();
                p += end1 + 1;
                let end2 = b[p..].iter().position(|&c| c == b'\n').context("GLOBAL")?;
                let name = String::from_utf8_lossy(&b[p..p + end2]).into_owned();
                p += end2 + 1;
                st.push(V::Global(format!("{module}.{name}")));
            }
            0x93 => {
                // STACK_GLOBAL
                let (Some(V::Str(name)), Some(V::Str(module))) = (st.pop(), st.pop()) else { bail!("STACK_GLOBAL") };
                st.push(V::Global(format!("{module}.{name}")));
            }
            b'q' => {
                let k = rd(&mut p, 1)?[0] as u32;
                memo.insert(k, st.last().context("BINPUT")?.clone());
            }
            b'r' => {
                let k = u32::from_le_bytes(rd(&mut p, 4)?.try_into()?);
                memo.insert(k, st.last().context("LONG_BINPUT")?.clone());
            }
            0x94 => {
                let k = memo.len() as u32;
                memo.insert(k, st.last().context("MEMOIZE")?.clone());
            }
            b'h' => {
                let k = rd(&mut p, 1)?[0] as u32;
                st.push(memo.get(&k).context("BINGET")?.clone());
            }
            b'j' => {
                let k = u32::from_le_bytes(rd(&mut p, 4)?.try_into()?);
                st.push(memo.get(&k).context("LONG_BINGET")?.clone());
            }
            b't' => {
                let items = pop_mark(&mut st)?;
                st.push(V::Tuple(items));
            }
            0x85..=0x87 => {
                let n = (op - 0x84) as usize;
                let items = st.split_off(st.len().checked_sub(n).context("TUPLEn")?);
                st.push(V::Tuple(items));
            }
            b'R' => {
                let args = st.pop().context("REDUCE")?;
                let f = st.pop().context("REDUCE")?;
                st.push(V::Call(Box::new(f), Box::new(args), Vec::new()));
            }
            b'Q' => {
                let id = st.pop().context("BINPERSID")?;
                st.push(V::PersId(Box::new(id)));
            }
            b'u' => {
                let items = pop_mark(&mut st)?;
                set_items(st.last_mut().context("SETITEMS")?, items)?;
            }
            b's' => {
                let v = st.pop().context("SETITEM")?;
                let k = st.pop().context("SETITEM")?;
                set_items(st.last_mut().context("SETITEM")?, vec![k, v])?;
            }
            b'e' => {
                let items = pop_mark(&mut st)?;
                if let Some(V::List(l)) = st.last_mut() {
                    l.extend(items);
                }
            }
            b'a' => {
                let v = st.pop().context("APPEND")?;
                if let Some(V::List(l)) = st.last_mut() {
                    l.push(v);
                }
            }
            b'b' => {
                // BUILD: state ignored (tensor metadata / OrderedDict attributes).
                st.pop();
            }
            b'.' => return st.pop().context("empty pickle"),
            _ => bail!("unsupported pickle opcode 0x{op:02x} at {}", p - 1),
        }
    }
}

struct Tensor {
    key: String,
    offset: usize,
    shape: Vec<usize>,
    /// Element strides (channels-last tensors are not stored in shape order).
    stride: Vec<usize>,
}

impl Tensor {
    /// The values in shape order (row-major), as little-endian float32 bytes.
    fn bytes(&self, dir: &Path) -> Result<Vec<u8>> {
        let raw = std::fs::read(dir.join("data").join(&self.key)).with_context(|| format!("data/{}", self.key))?;
        let n: usize = self.shape.iter().product();
        let mut out = Vec::with_capacity(n * 4);
        for i in 0..n {
            // Index i in shape order -> element offset through the strides.
            let (mut rest, mut at) = (i, self.offset);
            for d in (0..self.shape.len()).rev() {
                at += (rest % self.shape[d]) * self.stride.get(d).copied().unwrap_or(1);
                rest /= self.shape[d];
            }
            out.extend_from_slice(raw.get(at * 4..at * 4 + 4).context("tensor data short")?);
        }
        Ok(out)
    }
}

/// name -> tensor, from a state dict (or a dict holding one under params_ema/params).
fn tensors(root: &V) -> Result<Vec<(String, Tensor)>> {
    let items = |v: &V| -> Option<Vec<(V, V)>> {
        match v {
            V::Dict(d) | V::Call(_, _, d) => Some(d.clone()),
            _ => None,
        }
    };
    let mut dict = items(root).context("not a dict")?;
    for wrap in ["params_ema", "params"] {
        if let Some((_, inner)) = dict.iter().find(|(k, _)| matches!(k, V::Str(s) if s == wrap)) {
            dict = items(inner).context("inner state dict")?;
            break;
        }
    }
    let mut out = Vec::new();
    for (k, v) in dict {
        let V::Str(name) = k else { continue };
        let V::Call(f, args, _) = v else { continue };
        if !matches!(&*f, V::Global(g) if g.ends_with("_rebuild_tensor_v2")) {
            continue;
        }
        let V::Tuple(a) = *args else { bail!("{name}: tensor args") };
        let (Some(V::PersId(pid)), Some(V::Int(offset)), Some(V::Tuple(size))) = (a.first(), a.get(1), a.get(2)) else { bail!("{name}: tensor args") };
        let V::Tuple(pid) = &**pid else { bail!("{name}: storage id") };
        let (Some(V::Global(ty)), Some(V::Str(key))) = (pid.get(1), pid.get(2)) else { bail!("{name}: storage id") };
        if !ty.ends_with("FloatStorage") {
            // Integer buffers (index tables, masks) are not weights: computed where needed.
            continue;
        }
        let shape: Vec<usize> = size.iter().map(|d| if let V::Int(n) = d { *n as usize } else { 0 }).collect();
        let stride: Vec<usize> = match a.get(3) {
            Some(V::Tuple(s)) => s.iter().map(|d| if let V::Int(n) = d { *n as usize } else { 0 }).collect(),
            _ => Vec::new(),
        };
        let stride = if stride.len() == shape.len() { stride } else { contiguous(&shape) };
        out.push((name, Tensor { key: key.clone(), offset: *offset as usize, shape, stride }));
    }
    Ok(out)
}

/// SRVGGNetCompact (realesr-general-x4v3 and its no-denoise twin, xinntao/Real-ESRGAN
/// release v0.2.5.0, BSD-3-Clause): conv + PReLU, `n` times conv 64 -> 64 + PReLU,
/// conv to 3 x 16 channels, pixel shuffle x4, plus the input enlarged (nearest).
/// The graph is written here (its depth comes from the weights); weights are raw
/// float32 in layer order, so two models of the same shape can be mixed word by word.
fn compact(dir: &Path, out_param: &str, out_bin: &str) -> Result<()> {
    let pkl = std::fs::read(dir.join("data.pkl")).context("data.pkl")?;
    let all = tensors(&unpickle(&pkl)?)?;
    if std::env::var_os("PTH_LIST").is_some() {
        for (n, t) in &all {
            println!("{n}: {:?} key {} offset {}", t.shape, t.key, t.offset);
        }
    }
    let by_name: HashMap<&str, &Tensor> = all.iter().map(|(n, t)| (n.as_str(), t)).collect();
    let read = |name: &str| -> Result<(Vec<u8>, Vec<usize>)> {
        let t = by_name.get(name).with_context(|| format!("{name} missing"))?;
        Ok((t.bytes(dir)?, t.shape.clone()))
    };
    // body.0 conv, body.1 PReLU, ... body.(2n+2) the last conv.
    let convs = (0..).take_while(|i| by_name.contains_key(format!("body.{}.weight", 2 * i).as_str())).count();
    anyhow::ensure!(convs >= 3, "not an SRVGGNetCompact state dict");
    let n = convs - 2;
    // The last convolution makes 3 x scale^2 channels for the pixel shuffle.
    let last = by_name.get(format!("body.{}.weight", 2 * (convs - 1)).as_str()).context("last convolution")?;
    let scale = ((last.shape[0] / 3) as f64).sqrt().round() as usize;
    anyhow::ensure!(3 * scale * scale == last.shape[0], "last convolution has {} channels", last.shape[0]);
    let mut layers = vec!["Input input 0 1 data".to_string(), "Split split 1 2 data in0 in1".to_string()];
    let mut bin: Vec<u8> = Vec::new();
    let mut prev = "in1".to_string();
    for k in 0..convs {
        let (w, shape) = read(&format!("body.{}.weight", 2 * k))?;
        let (b, _) = read(&format!("body.{}.bias", 2 * k))?;
        let (oc, count) = (shape[0], shape.iter().product::<usize>());
        bin.extend(0u32.to_le_bytes());
        bin.extend(w);
        bin.extend(b);
        layers.push(format!("Convolution conv{k} 1 1 {prev} c{k} 0={oc} 1=3 4=1 5=1 6={count}"));
        prev = format!("c{k}");
        if k + 1 < convs {
            let (s, sshape) = read(&format!("body.{}.weight", 2 * k + 1))?;
            bin.extend(s);
            layers.push(format!("PReLU prelu{k} 1 1 {prev} a{k} 0={}", sshape[0]));
            prev = format!("a{k}");
        }
    }
    layers.push(format!("PixelShuffle shuffle 1 1 {prev} up 0={scale}"));
    layers.push(format!("Interp enlarge 1 1 in0 base 0=1 1={scale}.000000e+00 2={scale}.000000e+00"));
    layers.push("BinaryOp add 2 1 up base output".to_string());
    // Blobs: data, in0, in1, c*, a*, up, base, output.
    let blobs = 3 + convs + (convs - 1) + 3;
    let param = format!("7767517\n{} {blobs}\n{}\n", layers.len(), layers.join("\n"));
    std::fs::write(out_param, param)?;
    std::fs::write(out_bin, &bin)?;
    println!("compact {scale}x: {n} inner convolutions, {} layers, {} KB", layers.len(), bin.len() >> 10);
    Ok(())
}

/// SwinIR (JingyunLiang/SwinIR, Apache-2.0), real-world models with the
/// "nearest+conv" upsampler (e.g. 003_realSR_BSRGAN_DFO_s64w8_SwinIR-M_x4_GAN). The
/// graph follows the official forward pass with our layers for the transformer parts:
/// ChannelNorm (LayerNorm over the channels of each pixel) and WindowAttention (on
/// the qkv projection, shift and masks by index); Linear layers are 1x1 convolutions
/// on the planar map. Patches are single pixels (patch_size 1), so "tokens" are pixels.
fn swinir(dir: &Path, out_param: &str, out_bin: &str) -> Result<()> {
    let pkl = std::fs::read(dir.join("data.pkl")).context("data.pkl")?;
    let all = tensors(&unpickle(&pkl)?)?;
    let by_name: HashMap<&str, &Tensor> = all.iter().map(|(n, t)| (n.as_str(), t)).collect();
    let get = |name: &str| -> Result<&Tensor> { by_name.get(name).copied().with_context(|| format!("{name} missing")) };
    let has = |name: &str| by_name.contains_key(name);
    anyhow::ensure!(has("conv_up1.weight"), "only SwinIR with the nearest+conv upsampler is supported");
    let embed = get("conv_first.weight")?.shape[0];
    let groups = (0..).take_while(|r| has(&format!("layers.{r}.conv.weight"))).count();
    let depth = (0..).take_while(|b| has(&format!("layers.0.residual_group.blocks.{b}.norm1.weight"))).count();
    let table = get("layers.0.residual_group.blocks.0.attn.relative_position_bias_table")?;
    let heads = table.shape[1];
    let ws = (((table.shape[0] as f64).sqrt() as usize) + 1) / 2;
    let scale = if has("conv_up2.weight") { 4 } else { 2 };
    println!("swinir: {embed} channels, {groups} groups of {depth} blocks, {heads} heads, window {ws}, x{scale}");

    let mut layers: Vec<String> = Vec::new();
    let mut bin: Vec<u8> = Vec::new();
    let mut n = 0usize;
    let blob = |n: &mut usize| -> String {
        *n += 1;
        format!("b{n}")
    };
    // Convolution from `name` (weight [oc][ic][k][k] or Linear [oc][ic]); act: (type, param).
    let conv = |layers: &mut Vec<String>, bin: &mut Vec<u8>, name: &str, input: &str, output: &str, act: Option<(i32, f32)>| -> Result<()> {
        let w = get(&format!("{name}.weight"))?;
        let b = get(&format!("{name}.bias"))?;
        let k = if w.shape.len() == 4 { w.shape[2] } else { 1 };
        let count: usize = w.shape.iter().product();
        bin.extend(0u32.to_le_bytes());
        bin.extend(w.bytes(dir)?);
        bin.extend(b.bytes(dir)?);
        let a = act.map_or(String::new(), |(t, v)| format!(" 9={t} -23310=1,{v:e}"));
        layers.push(format!("Convolution {name} 1 1 {input} {output} 0={} 1={k} 4={} 5=1 6={count}{a}", w.shape[0], k / 2));
        Ok(())
    };
    let norm = |layers: &mut Vec<String>, bin: &mut Vec<u8>, name: &str, input: &str, output: &str| -> Result<()> {
        bin.extend(get(&format!("{name}.weight"))?.bytes(dir)?);
        bin.extend(get(&format!("{name}.bias"))?.bytes(dir)?);
        layers.push(format!("ChannelNorm {name} 1 1 {input} {output} 0={embed} 1=1.000000e-05"));
        Ok(())
    };
    // SwinIR's input normalisation: the DIV2K mean (img_range 1).
    let mean = [0.4488f32, 0.4371, 0.4040];
    layers.push("Input input 0 1 data".into());
    layers.push("MemoryData mean_in 0 1 mean_in 0=1 1=1 2=3".into());
    mean.iter().for_each(|v| bin.extend(v.to_le_bytes()));
    let x = blob(&mut n);
    layers.push(format!("BinaryOp sub_mean 2 1 data mean_in {x} 0=1"));
    let f0 = blob(&mut n);
    conv(&mut layers, &mut bin, "conv_first", &x, &f0, None)?;
    let (f0a, f0b) = (blob(&mut n), blob(&mut n));
    layers.push(format!("Split split_first 1 2 {f0} {f0a} {f0b}"));
    let mut t = blob(&mut n);
    norm(&mut layers, &mut bin, "patch_embed.norm", &f0b, &t)?;
    for r in 0..groups {
        let (t_res, mut cur) = (blob(&mut n), blob(&mut n));
        layers.push(format!("Split split_rstb{r} 1 2 {t} {t_res} {cur}"));
        for b in 0..depth {
            let pre = format!("layers.{r}.residual_group.blocks.{b}");
            let shift = if b % 2 == 1 { ws / 2 } else { 0 };
            let (sc, cin) = (blob(&mut n), blob(&mut n));
            layers.push(format!("Split split_{r}_{b}_a 1 2 {cur} {sc} {cin}"));
            let n1 = blob(&mut n);
            norm(&mut layers, &mut bin, &format!("{pre}.norm1"), &cin, &n1)?;
            let qkv = blob(&mut n);
            conv(&mut layers, &mut bin, &format!("{pre}.attn.qkv"), &n1, &qkv, None)?;
            let att = blob(&mut n);
            bin.extend(get(&format!("{pre}.attn.relative_position_bias_table"))?.bytes(dir)?);
            layers.push(format!("WindowAttention attn_{r}_{b} 1 1 {qkv} {att} 0={embed} 1={heads} 2={ws} 3={shift}"));
            let pj = blob(&mut n);
            conv(&mut layers, &mut bin, &format!("{pre}.attn.proj"), &att, &pj, None)?;
            let y = blob(&mut n);
            layers.push(format!("BinaryOp add_{r}_{b}_a 2 1 {sc} {pj} {y} 0=0"));
            let (ysc, yin) = (blob(&mut n), blob(&mut n));
            layers.push(format!("Split split_{r}_{b}_b 1 2 {y} {ysc} {yin}"));
            let n2 = blob(&mut n);
            norm(&mut layers, &mut bin, &format!("{pre}.norm2"), &yin, &n2)?;
            let (h1, g, h2) = (blob(&mut n), blob(&mut n), blob(&mut n));
            conv(&mut layers, &mut bin, &format!("{pre}.mlp.fc1"), &n2, &h1, None)?;
            layers.push(format!("GELU gelu_{r}_{b} 1 1 {h1} {g}"));
            conv(&mut layers, &mut bin, &format!("{pre}.mlp.fc2"), &g, &h2, None)?;
            cur = blob(&mut n);
            layers.push(format!("BinaryOp add_{r}_{b}_b 2 1 {ysc} {h2} {cur} 0=0"));
        }
        let rc = blob(&mut n);
        conv(&mut layers, &mut bin, &format!("layers.{r}.conv"), &cur, &rc, None)?;
        // The group's output and the stream after it go past f16's range (2.4e5 in
        // the last group of SwinIR-M): param 30=1 keeps them f32.
        layers.last_mut().unwrap().push_str(" 30=1");
        t = blob(&mut n);
        layers.push(format!("BinaryOp add_rstb{r} 2 1 {t_res} {rc} {t} 0=0 30=1"));
    }
    let nf = blob(&mut n);
    norm(&mut layers, &mut bin, "norm", &t, &nf)?;
    let cab = blob(&mut n);
    conv(&mut layers, &mut bin, "conv_after_body", &nf, &cab, None)?;
    let body = blob(&mut n);
    layers.push(format!("BinaryOp add_body 2 1 {cab} {f0a} {body} 0=0"));
    // conv_before_upsample is followed by nn.LeakyReLU() (slope 0.01); the rest by
    // self.lrelu (slope 0.2).
    let mut u = blob(&mut n);
    conv(&mut layers, &mut bin, "conv_before_upsample.0", &body, &u, Some((2, 0.01)))?;
    for (i, name) in ["conv_up1", "conv_up2"].iter().take(if scale == 4 { 2 } else { 1 }).enumerate() {
        let big = blob(&mut n);
        layers.push(format!("Interp up{i} 1 1 {u} {big} 0=1 1=2.000000e+00 2=2.000000e+00"));
        u = blob(&mut n);
        conv(&mut layers, &mut bin, name, &big, &u, Some((2, 0.2)))?;
    }
    let hr = blob(&mut n);
    conv(&mut layers, &mut bin, "conv_hr", &u, &hr, Some((2, 0.2)))?;
    let last = blob(&mut n);
    conv(&mut layers, &mut bin, "conv_last", &hr, &last, None)?;
    layers.push("MemoryData mean_out 0 1 mean_out 0=1 1=1 2=3".into());
    mean.iter().for_each(|v| bin.extend(v.to_le_bytes()));
    layers.push(format!("BinaryOp add_mean 2 1 {last} mean_out output 0=0"));

    let blobs: std::collections::HashSet<&str> = layers.iter().flat_map(|l| {
        let f: Vec<&str> = l.split_whitespace().collect();
        let (ni, no) = (f[2].parse::<usize>().unwrap_or(0), f[3].parse::<usize>().unwrap_or(0));
        f[4..4 + ni + no].to_vec()
    }).collect();
    std::fs::write(out_param, format!("7767517\n{} {}\n{}\n", layers.len(), blobs.len(), layers.join("\n")))?;
    std::fs::write(out_bin, &bin)?;
    println!("swinir: {} layers, {} MB", layers.len(), bin.len() >> 20);
    Ok(())
}

// ---- ONNX initializers ---------------------------------------------------------------

/// A protobuf message as (field, wire type, payload): varints as their value,
/// length-delimited fields as their bytes.
fn pb_fields(b: &[u8]) -> Result<Vec<(u64, u8, u64, &[u8])>> {
    fn varint(b: &[u8], p: &mut usize) -> Result<u64> {
        let (mut v, mut s) = (0u64, 0);
        loop {
            let c = *b.get(*p).context("protobuf: truncated varint")?;
            *p += 1;
            v |= ((c & 127) as u64) << s;
            if c < 128 {
                return Ok(v);
            }
            s += 7;
        }
    }
    let (mut p, mut out) = (0usize, Vec::new());
    while p < b.len() {
        let k = varint(b, &mut p)?;
        let (f, w) = (k >> 3, (k & 7) as u8);
        match w {
            0 => out.push((f, w, varint(b, &mut p)?, &b[0..0])),
            1 => {
                out.push((f, w, 0, b.get(p..p + 8).context("protobuf: truncated")?));
                p += 8;
            }
            2 => {
                let n = varint(b, &mut p)? as usize;
                out.push((f, w, 0, b.get(p..p + n).context("protobuf: truncated")?));
                p += n;
            }
            5 => {
                out.push((f, w, 0, b.get(p..p + 4).context("protobuf: truncated")?));
                p += 4;
            }
            _ => bail!("protobuf: wire type {w}"),
        }
    }
    Ok(out)
}

/// An ONNX model's float initializers (and Constant nodes' tensors): name ->
/// (shape, values). Integer tensors (index tables) are skipped.
fn onnx_weights(path: &Path) -> Result<HashMap<String, (Vec<usize>, Vec<f32>)>> {
    let data = std::fs::read(path).with_context(|| format!("reading {}", path.display()))?;
    let graph = pb_fields(&data)?.into_iter().find(|f| f.0 == 7).context("no graph in the ONNX file")?.3;
    let tensor = |b: &[u8]| -> Result<Option<(String, Vec<usize>, Vec<f32>)>> {
        let (mut name, mut dims, mut ty, mut raw, mut floats) = (String::new(), Vec::new(), 0, None, Vec::new());
        for (f, w, v, p) in pb_fields(b)? {
            match (f, w) {
                (1, 0) => dims.push(v as usize),
                (1, 2) => {
                    let mut q = 0;
                    while q < p.len() {
                        let (mut x, mut s) = (0u64, 0);
                        loop {
                            let c = p[q];
                            q += 1;
                            x |= ((c & 127) as u64) << s;
                            if c < 128 {
                                break;
                            }
                            s += 7;
                        }
                        dims.push(x as usize);
                    }
                }
                (2, 0) => ty = v,
                (4, 2) => floats.extend(p.chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap()))),
                (4, 5) => floats.push(f32::from_le_bytes(p.try_into().unwrap())),
                (8, 2) => name = String::from_utf8_lossy(p).into_owned(),
                (9, 2) => raw = Some(p),
                (13, _) | (14, _) => bail!("{name}: external tensor data is not supported"),
                _ => {}
            }
        }
        if ty != 1 {
            return Ok(None);
        }
        if let Some(r) = raw {
            floats = r.chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect();
        }
        Ok(Some((name, dims, floats)))
    };
    let mut out = HashMap::new();
    for (f, _, _, b) in pb_fields(graph)? {
        if f == 5 {
            if let Some((n, d, v)) = tensor(b)? {
                out.insert(n, (d, v));
            }
        } else if f == 1 {
            // A Constant node: its output name and its "value" attribute.
            let node = pb_fields(b)?;
            let is_const = node.iter().any(|(f, _, _, p)| *f == 4 && *p == b"Constant");
            let Some(oname) = node.iter().find(|x| x.0 == 2).map(|x| String::from_utf8_lossy(x.3).into_owned()) else { continue };
            if !is_const {
                continue;
            }
            for (_, _, _, attr) in node.iter().filter(|x| x.0 == 5) {
                if let Some(t) = pb_fields(attr)?.into_iter().find(|x| x.0 == 5) {
                    if let Some((_, d, v)) = tensor(t.3)? {
                        out.insert(oname.clone(), (d, v));
                    }
                }
            }
        }
    }
    Ok(out)
}

/// A .safetensors file's float tensors (F32, F16, BF16): name -> (shape, values).
/// The header is a flat JSON object of {"dtype", "shape", "data_offsets"} entries
/// (and "__metadata__", strings only), read here without a JSON library.
fn safetensors_weights(path: &Path) -> Result<HashMap<String, (Vec<usize>, Vec<f32>)>> {
    let data = std::fs::read(path).with_context(|| format!("reading {}", path.display()))?;
    let n = u64::from_le_bytes(data.get(..8).context("safetensors: too short")?.try_into()?) as usize;
    let header = std::str::from_utf8(data.get(8..8 + n).context("safetensors: header past the end")?)?;
    let body = &data[8 + n..];
    let nums = |s: &str, key: &str| -> Vec<usize> {
        s.find(key).map_or(Vec::new(), |i| {
            let rest = &s[i + key.len()..];
            let end = rest.find(']').unwrap_or(0);
            rest[..end].split(',').filter_map(|v| v.trim().parse().ok()).collect()
        })
    };
    let mut out = HashMap::new();
    let mut rest = header.trim().trim_start_matches('{');
    while let Some(q) = rest.find('"') {
        // "name": { ... }
        let after = &rest[q + 1..];
        let end = after.find('"').context("safetensors: bad header")?;
        let name = &after[..end];
        let open = after[end..].find('{').context("safetensors: bad header")? + end;
        let close = after[open..].find('}').context("safetensors: bad header")? + open;
        let entry = &after[open..=close];
        rest = &after[close + 1..];
        if name == "__metadata__" {
            continue;
        }
        let dtype = entry.split("\"dtype\"").nth(1).and_then(|s| s.split('"').nth(1)).unwrap_or("");
        let shape = nums(entry, "\"shape\":[");
        let offs = nums(entry, "\"data_offsets\":[");
        anyhow::ensure!(offs.len() == 2, "{name}: data offsets");
        let raw = body.get(offs[0]..offs[1]).with_context(|| format!("{name}: data past the end"))?;
        let values: Vec<f32> = match dtype {
            "F32" => raw.chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect(),
            "F16" => raw.chunks_exact(2).map(|c| f16_bits_to_f32(u16::from_le_bytes([c[0], c[1]]))).collect(),
            "BF16" => raw.chunks_exact(2).map(|c| f32::from_bits((u16::from_le_bytes([c[0], c[1]]) as u32) << 16)).collect(),
            _ => continue, // integer buffers (index tables, counters)
        };
        out.insert(name.to_string(), (shape, values));
    }
    Ok(out)
}

fn f16_bits_to_f32(h: u16) -> f32 {
    let s = if h & 0x8000 != 0 { -1.0 } else { 1.0 };
    let e = ((h >> 10) & 0x1f) as i32;
    let m = (h & 0x3ff) as f32;
    match e {
        0 => s * m * 2f32.powi(-24),
        31 => if m == 0.0 { s * f32::INFINITY } else { f32::NAN },
        _ => s * (1.0 + m / 1024.0) * 2f32.powi(e - 15),
    }
}

/// DAT (zhengchen1999/DAT, as in neosr's dat_arch: "Dual Aggregation Transformer"),
/// the pixelshuffle upsampler with one convolution per residual group (dat_2, dat_s,
/// dat_m, e.g. Phhofm's 4xNomos2_hq_dat2), read from an ONNX export that kept the
/// PyTorch names. Blocks alternate:
/// - spatial (even): window attention on two halves of the channels, 8x32 and 32x8
///   windows (`DATAttention`; shifted by half a window in some blocks),
/// - channel (odd): attention between the channels over the whole tile
///   (`ChannelAttention`),
/// each beside a depthwise 3x3 convolution of v, the two mixed by learned gates
/// (`AIM`), then the gated feed-forward (SGFN: its second half normalized, depthwise
/// convolved, multiplied with the first). BatchNorms are folded into the layers
/// before them; the position bias MLPs run here (their inputs are constants), giving
/// tables of the same form as Swin's.
fn dat(src: &Path, out_param: &str, out_bin: &str, mean_arg: Option<&str>) -> Result<()> {
    let w = if src.extension().is_some_and(|e| e == "safetensors") { safetensors_weights(src)? } else { onnx_weights(src)? };
    let get = |name: &str| -> Result<&(Vec<usize>, Vec<f32>)> { w.get(name).with_context(|| format!("{name} missing")) };
    let has = |name: &str| w.contains_key(name);
    anyhow::ensure!(has("upsample.0.weight") && has("conv_before_upsample.0.weight"), "only DAT with the pixelshuffle upsampler is supported");
    let embed = get("conv_first.weight")?.0[0];
    let groups = (0..).take_while(|r| has(&format!("layers.{r}.conv.weight"))).count();
    let depth = (0..).take_while(|b| has(&format!("layers.0.blocks.{b}.norm1.weight"))).count();
    let heads = get("layers.0.blocks.1.attn.temperature")?.1.len();
    let hidden = get("layers.0.blocks.0.ffn.fc1.weight")?.0[0];
    // Window sizes from the position bias inputs: (2 s0 - 1)(2 s1 - 1) rows, the
    // first row (1 - s0, 1 - s1).
    let rpe = get("layers.0.blocks.0.attn.attns.0.rpe_biases")?;
    let (s0, s1) = ((1.0 - rpe.1[0]) as usize, (1.0 - rpe.1[1]) as usize);
    anyhow::ensure!((2 * s0 - 1) * (2 * s1 - 1) == rpe.0[0], "position bias rows {} for windows {s0}x{s1}", rpe.0[0]);
    let ups = (0..).take_while(|i| has(&format!("upsample.{}.weight", 2 * i))).count();
    let scale = 1usize << ups;
    println!("dat: {embed} channels, {groups} groups of {depth} blocks, {heads} heads, windows {s0}x{s1}, feed-forward {hidden}, x{scale}");

    let mut layers: Vec<String> = Vec::new();
    let mut bin: Vec<u8> = Vec::new();
    let mut n = 0usize;
    let mut blob = || -> String {
        n += 1;
        format!("b{n}")
    };
    let floats = |bin: &mut Vec<u8>, v: &[f32]| v.iter().for_each(|x| bin.extend(x.to_le_bytes()));
    // A layer's weight and bias, with the BatchNorm `bn` (if any) folded in.
    let folded = |name: &str, bn: Option<&str>| -> Result<(Vec<usize>, Vec<f32>, Vec<f32>)> {
        let (shape, wt) = get(&format!("{name}.weight"))?.clone();
        let mut b = get(&format!("{name}.bias"))?.1.clone();
        let mut wt = wt;
        if let Some(bn) = bn {
            let (g, beta) = (&get(&format!("{bn}.weight"))?.1, &get(&format!("{bn}.bias"))?.1);
            let (mean, var) = (&get(&format!("{bn}.running_mean"))?.1, &get(&format!("{bn}.running_var"))?.1);
            let per = wt.len() / shape[0];
            for o in 0..shape[0] {
                let s = g[o] / (var[o] + 1e-5).sqrt();
                wt[o * per..(o + 1) * per].iter_mut().for_each(|x| *x *= s);
                b[o] = (b[o] - mean[o]) * s + beta[o];
            }
        }
        Ok((shape, wt, b))
    };
    // Convolution (or Linear, as 1x1) with optional BatchNorm and activation.
    let conv = |layers: &mut Vec<String>, bin: &mut Vec<u8>, name: &str, bn: Option<&str>, input: &str, output: &str, act: Option<(i32, f32)>| -> Result<()> {
        let (shape, wt, b) = folded(name, bn)?;
        let k = if shape.len() == 4 { shape[2] } else { 1 };
        bin.extend(0u32.to_le_bytes());
        floats(bin, &wt);
        floats(bin, &b);
        let a = act.map_or(String::new(), |(t, v)| format!(" 9={t} -23310=1,{v:e}"));
        let ty = if shape.len() == 4 && shape[1] == 1 && shape[0] > 1 { "ConvolutionDepthWise".to_string() } else { "Convolution".into() };
        let g = if ty == "ConvolutionDepthWise" { format!(" 7={}", shape[0]) } else { String::new() };
        layers.push(format!("{ty} {name} 1 1 {input} {output} 0={} 1={k} 4={} 5=1 6={}{g}{a}", shape[0], k / 2, wt.len()));
        Ok(())
    };
    // A 1x1 convolution on a 1x1 map (a channel vector) as an inner product.
    let inner = |layers: &mut Vec<String>, bin: &mut Vec<u8>, name: &str, bn: Option<&str>, input: &str, output: &str, act: Option<i32>| -> Result<()> {
        let (shape, wt, b) = folded(name, bn)?;
        bin.extend(0u32.to_le_bytes());
        floats(bin, &wt);
        floats(bin, &b);
        let a = act.map_or(String::new(), |t| format!(" 9={t}"));
        layers.push(format!("InnerProduct {name} 1 1 {input} {output} 0={} 1=1 2={}{a}", shape[0], wt.len()));
        Ok(())
    };
    let norm = |layers: &mut Vec<String>, bin: &mut Vec<u8>, name: &str, input: &str, output: &str| -> Result<()> {
        let g = &get(&format!("{name}.weight"))?.1;
        floats(bin, g);
        floats(bin, &get(&format!("{name}.bias"))?.1);
        layers.push(format!("ChannelNorm {name} 1 1 {input} {output} 0={} 1=1.000000e-05", g.len()));
        Ok(())
    };
    // DynamicPosBias (not residual): pos3(pos2(pos1(pos_proj(b)))), each posN being
    // LayerNorm, ReLU, Linear; for every relative position (dy, dx) of the table.
    // The table rows: (dy, dx) for dy in 1 - wh..wh, dx in 1 - ww..ww (rpe_biases; the
    // export kept one copy of each, in the first block).
    let rows = |wh: usize, ww: usize| -> Vec<f32> {
        let (wh, ww) = (wh as i32, ww as i32);
        (1 - wh..wh).flat_map(|dy| (1 - ww..ww).flat_map(move |dx| [dy as f32, dx as f32])).collect()
    };
    for (k, (wh, ww)) in [(s0, s1), (s1, s0)].into_iter().enumerate() {
        let stored = &get(&format!("layers.0.blocks.0.attn.attns.{k}.rpe_biases"))?.1;
        anyhow::ensure!(*stored == rows(wh, ww), "branch {k}: position bias rows differ");
    }
    let pos_table = |pre: &str, wh: usize, ww: usize| -> Result<Vec<f32>> {
        let rpe = rows(wh, ww);
        let lin = |name: &str, x: &[f32]| -> Result<Vec<f32>> {
            let (shape, wt) = get(&format!("{name}.weight"))?;
            let b = &get(&format!("{name}.bias"))?.1;
            Ok((0..shape[0]).map(|o| b[o] + (0..shape[1]).map(|i| wt[o * shape[1] + i] * x[i]).sum::<f32>()).collect())
        };
        let ln_relu = |name: &str, x: &[f32]| -> Result<Vec<f32>> {
            let (g, b) = (&get(&format!("{name}.weight"))?.1, &get(&format!("{name}.bias"))?.1);
            let m = x.iter().sum::<f32>() / x.len() as f32;
            let v = x.iter().map(|a| (a - m) * (a - m)).sum::<f32>() / x.len() as f32;
            Ok(x.iter().enumerate().map(|(i, a)| ((a - m) / (v + 1e-5).sqrt() * g[i] + b[i]).max(0.0)).collect())
        };
        let mut out = Vec::new();
        for row in rpe.chunks_exact(2) {
            let mut x = lin(&format!("{pre}.pos.pos_proj"), row)?;
            for k in 1..=3 {
                x = ln_relu(&format!("{pre}.pos.pos{k}.0"), &x)?;
                x = lin(&format!("{pre}.pos.pos{k}.2"), &x)?;
            }
            out.extend(x);
        }
        Ok(out)
    };

    // Input normalisation (img_range 1): the mean is a constant of the graph, [1, 3, 1,
    // 1] (neosr: 0.5 each; the original DAT: the DIV2K mean).
    // Given (r,g,b), else from the ONNX graph.
    let mean: Vec<f32> = match mean_arg {
        Some(m) => m.split(',').map(|v| v.trim().parse::<f32>()).collect::<std::result::Result<_, _>>()?,
        None => {
            let means: Vec<&Vec<f32>> = w.iter().filter(|(k, (d, _))| k.starts_with("/Constant") && *d == [1, 3, 1, 1]).map(|(_, (_, v))| v).collect();
            anyhow::ensure!(!means.is_empty() && means.iter().all(|m| *m == means[0]), "input mean not found in the graph: give it (r,g,b)");
            means[0].clone()
        }
    };
    anyhow::ensure!(mean.len() == 3, "the mean needs 3 values");
    println!("dat: input mean {mean:?}");
    layers.push("Input input 0 1 data".into());
    layers.push("MemoryData mean_in 0 1 mean_in 0=1 1=1 2=3".into());
    floats(&mut bin, &mean);
    let x = blob();
    layers.push(format!("BinaryOp sub_mean 2 1 data mean_in {x} 0=1"));
    let f0 = blob();
    conv(&mut layers, &mut bin, "conv_first", None, &x, &f0, None)?;
    let mut t = blob();
    norm(&mut layers, &mut bin, "before_RG.1", &f0, &t)?;
    for r in 0..groups {
        let t_res = t.clone();
        let mut cur = t.clone();
        for b in 0..depth {
            let pre = format!("layers.{r}.blocks.{b}");
            let a = format!("{pre}.attn");
            let n1 = blob();
            norm(&mut layers, &mut bin, &format!("{pre}.norm1"), &cur, &n1)?;
            let qkv = blob();
            conv(&mut layers, &mut bin, &format!("{a}.qkv"), None, &n1, &qkv, None)?;
            // The convolution branch, on v (unpartitioned): depthwise 3x3, BatchNorm, GELU.
            let (v, convx) = (blob(), blob());
            layers.push(format!("ChannelView view_v_{r}_{b} 1 1 {qkv} {v} 0={} 1={embed}", 2 * embed));
            conv(&mut layers, &mut bin, &format!("{a}.dwconv.0"), Some(&format!("{a}.dwconv.1")), &v, &convx, Some((9, 0.0)))?;
            let att = blob();
            // Gates: the channel map from one branch's average, the spatial map per pixel
            // from the other (spatial blocks: channels from the convolution, pixels from
            // the attention; channel blocks the other way round).
            let spatial = b % 2 == 0;
            if spatial {
                let shifted = (r % 2 == 0 && b >= 2 && (b - 2) % 4 == 0) || (r % 2 == 1 && b % 4 == 0);
                floats(&mut bin, &pos_table(&format!("{a}.attns.0"), s0, s1)?);
                floats(&mut bin, &pos_table(&format!("{a}.attns.1"), s1, s0)?);
                layers.push(format!("DATAttention attn_{r}_{b} 1 1 {qkv} {att} 0={embed} 1={} 2={s0} 3={s1} 4={}", heads / 2, shifted as i32));
            } else {
                floats(&mut bin, &get(&format!("{a}.temperature"))?.1);
                layers.push(format!("ChannelAttention attn_{r}_{b} 1 1 {qkv} {att} 0={embed} 1={heads}"));
            }
            let (pooled_src, map_src) = if spatial { (&convx, &att) } else { (&att, &convx) };
            let (gp, c1, cmap) = (blob(), blob(), blob());
            layers.push(format!("Pooling pool_{r}_{b} 1 1 {pooled_src} {gp} 0=1 4=1"));
            inner(&mut layers, &mut bin, &format!("{a}.channel_interaction.1"), Some(&format!("{a}.channel_interaction.2")), &gp, &c1, Some(9))?;
            inner(&mut layers, &mut bin, &format!("{a}.channel_interaction.4"), None, &c1, &cmap, None)?;
            let (s1b, smap) = (blob(), blob());
            conv(&mut layers, &mut bin, &format!("{a}.spatial_interaction.0"), Some(&format!("{a}.spatial_interaction.1")), map_src, &s1b, Some((9, 0.0)))?;
            conv(&mut layers, &mut bin, &format!("{a}.spatial_interaction.3"), None, &s1b, &smap, None)?;
            let mix = blob();
            layers.push(format!("AIM aim_{r}_{b} 4 1 {att} {convx} {cmap} {smap} {mix} 0={}", (!spatial) as i32));
            let pj = blob();
            conv(&mut layers, &mut bin, &format!("{a}.proj"), None, &mix, &pj, None)?;
            let y = blob();
            layers.push(format!("BinaryOp add_{r}_{b}_a 2 1 {cur} {pj} {y} 0=0"));
            // SGFN.
            let f = format!("{pre}.ffn");
            let (n2, h, hg) = (blob(), blob(), blob());
            norm(&mut layers, &mut bin, &format!("{pre}.norm2"), &y, &n2)?;
            conv(&mut layers, &mut bin, &format!("{f}.fc1"), None, &n2, &h, None)?;
            layers.push(format!("GELU gelu_{r}_{b} 1 1 {h} {hg}"));
            let (x1, x2, g, m) = (blob(), blob(), blob(), blob());
            layers.push(format!("ChannelView view_x1_{r}_{b} 1 1 {hg} {x1} 0=0 1={}", hidden / 2));
            layers.push(format!("ChannelView view_x2_{r}_{b} 1 1 {hg} {x2} 0={} 1={}", hidden / 2, hidden / 2));
            norm(&mut layers, &mut bin, &format!("{f}.sg.norm"), &x2, &g)?;
            // The gate (x1 times the convolution) as the depthwise layer's second input.
            conv(&mut layers, &mut bin, &format!("{f}.sg.conv"), None, &g, &m, None)?;
            let l = layers.last_mut().unwrap();
            *l = l.replacen(&format!(" 1 1 {g} {m}"), &format!(" 2 1 {g} {x1} {m}"), 1);
            let fo = blob();
            conv(&mut layers, &mut bin, &format!("{f}.fc2"), None, &m, &fo, None)?;
            cur = blob();
            layers.push(format!("BinaryOp add_{r}_{b}_b 2 1 {y} {fo} {cur} 0=0"));
        }
        let rc = blob();
        conv(&mut layers, &mut bin, &format!("layers.{r}.conv"), None, &cur, &rc, None)?;
        // As SwinIR: the group's output and the stream may pass f16's range.
        layers.last_mut().unwrap().push_str(" 30=1");
        t = blob();
        layers.push(format!("BinaryOp add_rg{r} 2 1 {t_res} {rc} {t} 0=0 30=1"));
    }
    let nf = blob();
    norm(&mut layers, &mut bin, "norm", &t, &nf)?;
    let cab = blob();
    conv(&mut layers, &mut bin, "conv_after_body", None, &nf, &cab, None)?;
    let body = blob();
    layers.push(format!("BinaryOp add_body 2 1 {cab} {f0} {body} 0=0"));
    // conv_before_upsample: convolution and nn.LeakyReLU() (slope 0.01).
    let mut u = blob();
    conv(&mut layers, &mut bin, "conv_before_upsample.0", None, &body, &u, Some((2, 0.01)))?;
    for i in 0..ups {
        let (c, s) = (blob(), blob());
        conv(&mut layers, &mut bin, &format!("upsample.{}", 2 * i), None, &u, &c, None)?;
        layers.push(format!("PixelShuffle shuffle{i} 1 1 {c} {s} 0=2"));
        u = s;
    }
    let last = blob();
    conv(&mut layers, &mut bin, "conv_last", None, &u, &last, None)?;
    layers.push("MemoryData mean_out 0 1 mean_out 0=1 1=1 2=3".into());
    floats(&mut bin, &mean);
    layers.push(format!("BinaryOp add_mean 2 1 {last} mean_out output 0=0"));

    let blobs: std::collections::HashSet<&str> = layers.iter().flat_map(|l| {
        let f: Vec<&str> = l.split_whitespace().collect();
        let (ni, no) = (f[2].parse::<usize>().unwrap_or(0), f[3].parse::<usize>().unwrap_or(0));
        f[4..4 + ni + no].to_vec()
    }).collect();
    std::fs::write(out_param, format!("7767517\n{} {}\n{}\n", layers.len(), blobs.len(), layers.join("\n")))?;
    std::fs::write(out_bin, &bin)?;
    println!("dat: {} layers, {} MB", layers.len(), bin.len() >> 20);
    Ok(())
}

/// A graph being written: layers (param lines), weights, blob names.
struct Graph<'w> {
    w: &'w HashMap<String, (Vec<usize>, Vec<f32>)>,
    layers: Vec<String>,
    bin: Vec<u8>,
    n: usize,
}

impl<'w> Graph<'w> {
    fn new(w: &'w HashMap<String, (Vec<usize>, Vec<f32>)>) -> Self {
        Graph { w, layers: Vec::new(), bin: Vec::new(), n: 0 }
    }
    fn blob(&mut self) -> String {
        self.n += 1;
        format!("b{}", self.n)
    }
    fn get(&self, name: &str) -> Result<&'w (Vec<usize>, Vec<f32>)> {
        self.w.get(name).with_context(|| format!("{name} missing"))
    }
    fn floats(&mut self, v: &[f32]) {
        v.iter().for_each(|x| self.bin.extend(x.to_le_bytes()));
    }
    /// Convolution `name` (Conv2d, or Linear as 1x1) with an optional BatchNorm folded
    /// in, an activation (ncnn type: 1 ReLU, 9 GELU), stride, and a weight scale
    /// (layer scale, constant factors).
    fn conv(&mut self, name: &str, bn: Option<&str>, input: &str, act: i32, stride: usize, scale: f32) -> Result<String> {
        let (shape, wt) = self.get(&format!("{name}.weight"))?.clone();
        let mut wt = wt;
        let oc = shape[0];
        let mut b = match self.w.get(&format!("{name}.bias")) {
            Some((_, b)) => b.clone(),
            None => vec![0.0; oc],
        };
        if let Some(bn) = bn {
            let (g, beta) = (self.get(&format!("{bn}.weight"))?.1.clone(), self.get(&format!("{bn}.bias"))?.1.clone());
            let (mean, var) = (self.get(&format!("{bn}.running_mean"))?.1.clone(), self.get(&format!("{bn}.running_var"))?.1.clone());
            let per = wt.len() / oc;
            for o in 0..oc {
                let s = g[o] / (var[o] + 1e-5).sqrt();
                wt[o * per..(o + 1) * per].iter_mut().for_each(|x| *x *= s);
                b[o] = (b[o] - mean[o]) * s + beta[o];
            }
        }
        if scale != 1.0 {
            wt.iter_mut().for_each(|x| *x *= scale);
            b.iter_mut().for_each(|x| *x *= scale);
        }
        let k = if shape.len() == 4 { shape[2] } else { 1 };
        let dw = shape.len() == 4 && shape[1] == 1 && oc > 1;
        self.bin.extend(0u32.to_le_bytes());
        self.floats(&wt);
        self.floats(&b);
        let out = self.blob();
        let a = if act != 0 { format!(" 9={act}") } else { String::new() };
        let ty = if dw { "ConvolutionDepthWise" } else { "Convolution" };
        let g = if dw { format!(" 7={oc}") } else { String::new() };
        self.layers.push(format!("{ty} {name} 1 1 {input} {out} 0={oc} 1={k} 3={stride} 4={} 5=1 6={}{g}{a}", k / 2, wt.len()));
        Ok(out)
    }
    fn norm(&mut self, name: &str, input: &str, eps: f32) -> Result<String> {
        let g = self.get(&format!("{name}.weight"))?.1.clone();
        let b = self.get(&format!("{name}.bias"))?.1.clone();
        self.floats(&g);
        self.floats(&b);
        let out = self.blob();
        self.layers.push(format!("ChannelNorm {name} 1 1 {input} {out} 0={} 1={eps:e}", g.len()));
        Ok(out)
    }
    fn op(&mut self, line: impl Fn(&str) -> String) -> String {
        let out = self.blob();
        self.layers.push(line(&out));
        out
    }
    fn write(&self, param: &str, bin: &str) -> Result<()> {
        let blobs: std::collections::HashSet<&str> = self.layers.iter().flat_map(|l| {
            let f: Vec<&str> = l.split_whitespace().collect();
            let (ni, no) = (f[2].parse::<usize>().unwrap_or(0), f[3].parse::<usize>().unwrap_or(0));
            f[4..4 + ni + no].to_vec()
        }).collect();
        std::fs::write(param, format!("7767517\n{} {}\n{}\n", self.layers.len(), blobs.len(), self.layers.join("\n")))?;
        std::fs::write(bin, &self.bin)?;
        Ok(())
    }
}

/// SEA-RAFT (princeton-vl/SEA-RAFT, BSD-3-Clause) from its .safetensors, as three
/// graphs for `brp_ai::flow::SeaRaft`:
/// - `<stem>_feat`: a picture (3 channels, -1..1) to its features (1/8 size), cached
///   per frame;
/// - `<stem>_ctx`: both pictures to the hidden state, context and starting flow;
/// - `<stem>_upd`: one refinement step: hidden state, context, correlation lookup and
///   flow to the new hidden state, flow and upsampling weights.
/// BatchNorms are folded; ConvNeXt layer scale and the 0.25 on the upsampling weights
/// go into the weights. The correlation lookup and the convex upsampling are kernels.
fn searaft(src: &Path, out: &str) -> Result<()> {
    let w = safetensors_weights(src)?;
    let has = |n: &str| w.contains_key(n);
    let dim = w.get("init_conv.weight").context("not a SEA-RAFT checkpoint")?.0[0] / 2;
    let blocks = |net: &str, layer: usize| (0..).take_while(|i| has(&format!("{net}.layer{layer}.{i}.conv1.weight"))).count();
    let refine = (0..).take_while(|i| has(&format!("update_block.refine.{i}.final.weight"))).count();
    println!("searaft: dim {dim}, ResNet blocks {:?}, {refine} refine blocks", [1, 2, 3].map(|l| blocks("fnet", l)));

    // ResNetFPN to 1/8: conv1 7x7/2 + BN + ReLU, three stages of BasicBlocks (stride 1,
    // 2, 2), final 1x1.
    let resnet = |g: &mut Graph, net: &str, input: &str| -> Result<String> {
        let mut x = g.conv(&format!("{net}.conv1"), Some(&format!("{net}.bn1")), input, 1, 2, 1.0)?;
        for l in 1..=3 {
            for i in 0..blocks(net, l) {
                let p = format!("{net}.layer{l}.{i}");
                let stride = if l > 1 && i == 0 { 2 } else { 1 };
                let y = g.conv(&format!("{p}.conv1"), Some(&format!("{p}.bn1")), &x, 1, stride, 1.0)?;
                let y = g.conv(&format!("{p}.conv2"), Some(&format!("{p}.bn2")), &y, 1, 1, 1.0)?;
                let short = if has(&format!("{p}.downsample.0.weight")) { g.conv(&format!("{p}.downsample.0"), Some(&format!("{p}.bn3")), &x, 0, stride, 1.0)? } else { x.clone() };
                let s = g.op(|o| format!("BinaryOp {p}.add 2 1 {short} {y} {o} 0=0"));
                x = g.op(|o| format!("ReLU {p}.relu 1 1 {s} {o}"));
            }
        }
        g.conv(&format!("{net}.final_conv"), None, &x, 0, 1, 1.0)
    };
    // Pixels 0..1 to -1..1.
    let normalize = |g: &mut Graph, input: &str| -> String {
        let s = g.op(|o| format!("BinaryOp norm_mul_{input} 1 1 {input} {o} 0=2 1=1 2=2.0"));
        g.op(|o| format!("BinaryOp norm_add_{input} 1 1 {s} {o} 0=0 1=1 2=-1.0"))
    };

    let mut g = Graph::new(&w);
    g.layers.push("Input input 0 1 img".into());
    let x = normalize(&mut g, "img");
    let f = resnet(&mut g, "fnet", &x)?;
    g.layers.push(format!("Split out_fmap 1 1 {f} fmap"));
    g.write(&format!("{out}_feat.param"), &format!("{out}_feat.bin"))?;

    let mut g = Graph::new(&w);
    g.layers.push("Input input1 0 1 img1".into());
    g.layers.push("Input input2 0 1 img2".into());
    let (x1, x2) = (normalize(&mut g, "img1"), normalize(&mut g, "img2"));
    let both = g.op(|o| format!("Concat cat_imgs 2 1 {x1} {x2} {o} 0=0"));
    let c = resnet(&mut g, "cnet", &both)?;
    let c = g.conv("init_conv", None, &c, 0, 1, 1.0)?;
    g.layers.push(format!("ChannelView net_view 1 1 {c} net 0=0 1={dim}"));
    g.layers.push(format!("ChannelView ctx_view 1 1 {c} context 0={dim} 1={dim}"));
    let h = g.conv("flow_head.0", None, "net", 1, 1, 1.0)?;
    let fh = g.conv("flow_head.2", None, &h, 0, 1, 1.0)?;
    g.layers.push(format!("ChannelView flow0_view 1 1 {fh} flow0 0=0 1=2"));
    g.write(&format!("{out}_ctx.param"), &format!("{out}_ctx.bin"))?;

    let mut g = Graph::new(&w);
    for n in ["net", "context", "corr", "flow"] {
        g.layers.push(format!("Input in_{n} 0 1 {n}"));
    }
    let u = "update_block";
    let c1 = g.conv(&format!("{u}.encoder.convc1"), None, "corr", 1, 1, 1.0)?;
    let c2 = g.conv(&format!("{u}.encoder.convc2"), None, &c1, 1, 1, 1.0)?;
    let f1 = g.conv(&format!("{u}.encoder.convf1"), None, "flow", 1, 1, 1.0)?;
    let f2 = g.conv(&format!("{u}.encoder.convf2"), None, &f1, 1, 1, 1.0)?;
    let cf = g.op(|o| format!("Concat cat_cor_flo 2 1 {c2} {f2} {o} 0=0"));
    let m = g.conv(&format!("{u}.encoder.conv"), None, &cf, 1, 1, 1.0)?;
    let motion = g.op(|o| format!("Concat cat_motion 2 1 {m} flow {o} 0=0"));
    let inp = g.op(|o| format!("Concat cat_inp 2 1 context {motion} {o} 0=0"));
    let mut net = "net".to_string();
    for i in 0..refine {
        let p = format!("{u}.refine.{i}");
        let x = g.op(|o| format!("Concat cat_refine{i} 2 1 {net} {inp} {o} 0=0"));
        let d = g.conv(&format!("{p}.dwconv"), None, &x, 0, 1, 1.0)?;
        let n = g.norm(&format!("{p}.norm"), &d, 1e-6)?;
        let a = g.conv(&format!("{p}.pwconv1"), None, &n, 9, 1, 1.0)?;
        // Layer scale (gamma) folded into pwconv2, row by row.
        let gamma = g.get(&format!("{p}.gamma"))?.1.clone();
        let (shape, wt) = g.get(&format!("{p}.pwconv2.weight"))?.clone();
        let mut wt = wt;
        let mut b = g.get(&format!("{p}.pwconv2.bias"))?.1.clone();
        let per = wt.len() / shape[0];
        for o in 0..shape[0] {
            wt[o * per..(o + 1) * per].iter_mut().for_each(|v| *v *= gamma[o]);
            b[o] *= gamma[o];
        }
        g.bin.extend(0u32.to_le_bytes());
        g.floats(&wt);
        g.floats(&b);
        let y = g.blob();
        g.layers.push(format!("Convolution {p}.pwconv2 1 1 {a} {y} 0={} 1=1 4=0 5=1 6={}", shape[0], wt.len()));
        let s = g.op(|o| format!("BinaryOp {p}.add 2 1 {x} {y} {o} 0=0"));
        net = g.conv(&format!("{p}.final"), None, &s, 0, 1, 1.0)?;
    }
    g.layers.push(format!("Split out_net 1 1 {net} net_out"));
    let h = g.conv("flow_head.0", None, &net, 1, 1, 1.0)?;
    let fh = g.conv("flow_head.2", None, &h, 0, 1, 1.0)?;
    let d = g.op(|o| format!("ChannelView delta_view 1 1 {fh} {o} 0=0 1=2"));
    // The flow stays f32 (30=1): f16 holds a few hundred pixels to a quarter pixel only.
    g.layers.push(format!("BinaryOp flow_add 2 1 flow {d} flow_out 0=0 30=1"));
    let h = g.conv("upsample_weight.0", None, &net, 1, 1, 1.0)?;
    let mw = g.conv("upsample_weight.2", None, &h, 0, 1, 0.25)?;
    g.layers.push(format!("Split out_mask 1 1 {mw} mask"));
    g.write(&format!("{out}_upd.param"), &format!("{out}_upd.bin"))?;
    println!("searaft: written {out}_feat / _ctx / _upd");
    Ok(())
}

/// A torch.save state dict (extracted zip: the folder holding data.pkl) as name ->
/// (shape, values), "module." prefixes removed.
fn pth_weights(root: &Path) -> Result<HashMap<String, (Vec<usize>, Vec<f32>)>> {
    let dir = std::fs::read_dir(root)?.flatten().map(|e| e.path()).find(|p| p.join("data.pkl").exists()).context("no data.pkl in the extracted .pth")?;
    let pkl = std::fs::read(dir.join("data.pkl"))?;
    let mut out = HashMap::new();
    for (name, t) in tensors(&unpickle(&pkl)?)? {
        let v: Vec<f32> = t.bytes(&dir)?.chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect();
        out.insert(name.strip_prefix("module.").unwrap_or(&name).to_string(), (t.shape.clone(), v));
    }
    Ok(out)
}

impl Graph<'_> {
    /// ConvTranspose2d with kernel = stride, no padding (PyTorch weights [ic][oc][k][k]):
    /// each input pixel makes a k x k block, so it is a 1x1 convolution to oc k k
    /// channels (row o k k + i k + j = weight [ic][o][i][j]), then a pixel shuffle.
    fn deconv(&mut self, name: &str, input: &str, stride: usize) -> Result<String> {
        let (shape, wt) = self.get(&format!("{name}.weight"))?.clone();
        let (ic, oc, k) = (shape[0], shape[1], shape[2]);
        anyhow::ensure!(k == stride, "{name}: transposed convolution with kernel {k}, stride {stride}");
        let mut t = vec![0f32; wt.len()];
        for i in 0..ic {
            for o in 0..oc {
                for q in 0..k * k {
                    t[(o * k * k + q) * ic + i] = wt[(i * oc + o) * k * k + q];
                }
            }
        }
        let b: Vec<f32> = match self.w.get(&format!("{name}.bias")) {
            Some((_, b)) => b.iter().flat_map(|v| std::iter::repeat_n(*v, k * k)).collect(),
            None => vec![0.0; oc * k * k],
        };
        self.bin.extend(0u32.to_le_bytes());
        self.floats(&t);
        self.floats(&b);
        let c = self.blob();
        self.layers.push(format!("Convolution {name} 1 1 {input} {c} 0={} 1=1 4=0 5=1 6={}", oc * k * k, t.len()));
        Ok(self.op(|o| format!("PixelShuffle {name}.shuffle 1 1 {c} {o} 0={k}")))
    }

    /// A patch embedding (k x k convolution, stride k): pixel unshuffle, then 1x1.
    fn patch_embed(&mut self, name: &str, input: &str) -> Result<String> {
        let k = self.get(&format!("{name}.weight"))?.0[2];
        let u = self.op(|o| format!("PixelUnshuffle {name}.unshuffle 1 1 {input} {o} 0={k}"));
        let (shape, wt) = self.get(&format!("{name}.weight"))?.clone();
        let b = self.w.get(&format!("{name}.bias")).map_or_else(|| vec![0.0; shape[0]], |t| t.1.clone());
        self.bin.extend(0u32.to_le_bytes());
        self.floats(&wt);
        self.floats(&b);
        let out = self.blob();
        self.layers.push(format!("Convolution {name} 1 1 {u} {out} 0={} 1=1 4=0 5=1 6={}", shape[0], wt.len()));
        Ok(out)
    }

    /// A ViT block (timm / DINOv2): x + attn(norm1(x)), then + mlp(norm2(x)); layer
    /// scale (DINOv2: ls1, ls2) folded into proj and fc2.
    fn vit_block(&mut self, p: &str, x: &str, heads: usize) -> Result<String> {
        let ls = |g: &Self, n: &str| g.w.get(&format!("{p}.{n}.gamma")).map(|t| t.1.clone());
        let c = self.get(&format!("{p}.norm1.weight"))?.1.len();
        let n1 = self.norm(&format!("{p}.norm1"), x, 1e-6)?;
        let qkv = self.conv(&format!("{p}.attn.qkv"), None, &n1, 0, 1, 1.0)?;
        let a = self.op(|o| format!("GlobalAttention {p}.attn 1 1 {qkv} {o} 0={c} 1={heads}"));
        let pj = self.conv_scaled_rows(&format!("{p}.attn.proj"), &a, ls(self, "ls1"))?;
        let y = self.op(|o| format!("BinaryOp {p}.add1 2 1 {x} {pj} {o} 0=0"));
        let n2 = self.norm(&format!("{p}.norm2"), &y, 1e-6)?;
        let h = self.conv(&format!("{p}.mlp.fc1"), None, &n2, 9, 1, 1.0)?;
        let f = self.conv_scaled_rows(&format!("{p}.mlp.fc2"), &h, ls(self, "ls2"))?;
        Ok(self.op(|o| format!("BinaryOp {p}.add2 2 1 {y} {f} {o} 0=0")))
    }

    /// A Linear / 1x1 convolution with each output row scaled (layer scale), if given.
    fn conv_scaled_rows(&mut self, name: &str, input: &str, rows: Option<Vec<f32>>) -> Result<String> {
        let (shape, wt) = self.get(&format!("{name}.weight"))?.clone();
        let mut wt = wt;
        let mut b = self.get(&format!("{name}.bias"))?.1.clone();
        if let Some(g) = rows {
            let per = wt.len() / shape[0];
            for o in 0..shape[0] {
                wt[o * per..(o + 1) * per].iter_mut().for_each(|v| *v *= g[o]);
                b[o] *= g[o];
            }
        }
        self.bin.extend(0u32.to_le_bytes());
        self.floats(&wt);
        self.floats(&b);
        let out = self.blob();
        self.layers.push(format!("Convolution {name} 1 1 {input} {out} 0={} 1=1 4=0 5=1 6={}", shape[0], wt.len()));
        Ok(out)
    }

    /// The DPT head of Depth Anything V2 (`return_intermediate`: output_conv1, resized
    /// to `size_ref`'s size, bilinear, align_corners=True), from the four layers' tokens
    /// as maps [C][ph][pw].
    fn dpt(&mut self, p: &str, layers: [&str; 4], size_ref: &str) -> Result<String> {
        let mut r = Vec::new();
        for (i, l) in layers.iter().enumerate() {
            let x = self.conv(&format!("{p}.projects.{i}"), None, l, 0, 1, 1.0)?;
            let x = match i {
                0 => self.deconv(&format!("{p}.resize_layers.0"), &x, 4)?,
                1 => self.deconv(&format!("{p}.resize_layers.1"), &x, 2)?,
                2 => x,
                _ => self.conv(&format!("{p}.resize_layers.3"), None, &x, 0, 2, 1.0)?,
            };
            r.push(self.conv(&format!("{p}.scratch.layer{}_rn", i + 1), None, &x, 0, 1, 1.0)?);
        }
        // ResidualConvUnit: x + conv2(relu(conv1(relu(x)))).
        let rcu = |g: &mut Self, u: &str, x: &str| -> Result<String> {
            let a = g.op(|o| format!("ReLU {u}.relu 1 1 {x} {o}"));
            let c1 = g.conv(&format!("{u}.conv1"), None, &a, 1, 1, 1.0)?;
            let c2 = g.conv(&format!("{u}.conv2"), None, &c1, 0, 1, 1.0)?;
            Ok(g.op(|o| format!("BinaryOp {u}.add 2 1 {c2} {x} {o} 0=0")))
        };
        // FeatureFusionBlock: (prev + rcu1(skip)), rcu2, resize, out_conv.
        let fuse = |g: &mut Self, n: usize, prev: &str, skip: Option<&str>, size: Option<&str>| -> Result<String> {
            let f = format!("{p}.scratch.refinenet{n}");
            let mut x = prev.to_string();
            if let Some(s) = skip {
                let r1 = rcu(g, &format!("{f}.resConfUnit1"), s)?;
                x = g.op(|o| format!("BinaryOp {f}.add 2 1 {x} {r1} {o} 0=0"));
            }
            let x = rcu(g, &format!("{f}.resConfUnit2"), &x)?;
            let up = match size {
                Some(s) => g.op(|o| format!("Interp {f}.resize 2 1 {x} {s} {o} 0=2 6=1")),
                None => g.op(|o| format!("Interp {f}.resize 1 1 {x} {o} 0=2 1=2.0 2=2.0 6=1")),
            };
            g.conv(&format!("{f}.out_conv"), None, &up, 0, 1, 1.0)
        };
        let p4 = fuse(self, 4, &r[3].clone(), None, Some(&r[2].clone()))?;
        let p3 = fuse(self, 3, &p4, Some(&r[2].clone()), Some(&r[1].clone()))?;
        let p2 = fuse(self, 2, &p3, Some(&r[1].clone()), Some(&r[0].clone()))?;
        let p1 = fuse(self, 1, &p2, Some(&r[0].clone()), None)?;
        let o = self.conv(&format!("{p}.scratch.output_conv1"), None, &p1, 0, 1, 1.0)?;
        Ok(self.op(|out| format!("Interp {p}.out_resize 2 1 {o} {size_ref} {out} 0=2 6=1")))
    }
}

/// WAFT a1 (princeton-vl/WAFT, BSD-3-Clause; Depth Anything V2 ViT-S, Apache-2.0) from
/// its checkpoint (extracted .pth), as graphs for `brp_ai::flow::Waft`:
/// - `_dpatch`: picture (normalized, padded to a multiple of 112) to DINOv2's patch
///   tokens (14 x 14 patches) as a map;
/// - `_dblocks`: the token sequence (class token first, position embedding added: built
///   by the caller) through DINOv2's 12 blocks; layers 2, 5, 8, 11 after its norm;
/// - `_ddpt`: those layers (class token removed, as maps) through the DPT head, to the
///   picture's size (32 channels);
/// - `_feat`: that map and the picture to the features at half size (ResNet-18 with
///   deconvolutions; fmap_conv);
/// - `_hidden`: both pictures' features to the starting hidden state;
/// - `_refine`: one refinement step: features, the other picture's features moved by
///   the flow, hidden state, flow and the refinement ViT's position embedding to the
///   new hidden state, flow and upsampling weights.
/// `_pos.bin`: raw f32, DINOv2's class token (384), its position embedding (1370 x
/// 384, token-major) and the refinement ViT's (64 x 384).
fn waft(src: &Path, out: &str) -> Result<()> {
    let w = pth_weights(src)?;
    let has = |n: &str| w.contains_key(n);
    let d = "da_feature.depth_anything";
    anyhow::ensure!(has(&format!("{d}.pretrained.cls_token")) && has("refine_net.pos_embed"), "not a WAFT a1 checkpoint");
    let heads = 6;
    println!("waft: {} tensors", w.len());

    let mut g = Graph::new(&w);
    g.layers.push("Input input 0 1 img".into());
    let t = g.patch_embed(&format!("{d}.pretrained.patch_embed.proj"), "img")?;
    g.layers.push(format!("Split out_tokens 1 1 {t} tokens"));
    g.write(&format!("{out}_dpatch.param"), &format!("{out}_dpatch.bin"))?;

    let mut g = Graph::new(&w);
    g.layers.push("Input input 0 1 x".into());
    let mut x = "x".to_string();
    for i in 0..12 {
        x = g.vit_block(&format!("{d}.pretrained.blocks.{i}"), &x, heads)?;
        if [2, 5, 8, 11].contains(&i) {
            let n = g.norm(&format!("{d}.pretrained.norm"), &x, 1e-6)?;
            g.layers.push(format!("Split out_l{i} 1 1 {n} l{i}"));
        }
    }
    g.write(&format!("{out}_dblocks.param"), &format!("{out}_dblocks.bin"))?;

    let mut g = Graph::new(&w);
    for n in ["l2", "l5", "l8", "l11", "img"] {
        g.layers.push(format!("Input in_{n} 0 1 {n}"));
    }
    let o = g.dpt(&format!("{d}.depth_head"), ["l2", "l5", "l8", "l11"], "img")?;
    g.layers.push(format!("Split out_da 1 1 {o} da"));
    g.write(&format!("{out}_ddpt.param"), &format!("{out}_ddpt.bin"))?;

    let mut g = Graph::new(&w);
    g.layers.push("Input in_da 0 1 da".into());
    g.layers.push("Input in_img 0 1 img".into());
    let x = g.op(|o| format!("Concat cat_da_img 2 1 da img {o} 0=0"));
    // resconv: conv(gelu -> conv k s -> gelu -> conv 3x3) + skip (1x1 stride s).
    let resconv = |g: &mut Graph, p: &str, x: &str, stride: usize| -> Result<String> {
        let a = g.op(|o| format!("GELU {p}.gelu 1 1 {x} {o}"));
        let c1 = g.conv(&format!("{p}.conv.1"), None, &a, 9, stride, 1.0)?;
        let c2 = g.conv(&format!("{p}.conv.3"), None, &c1, 0, 1, 1.0)?;
        let skip = if g.w.contains_key(&format!("{p}.skip_conv.weight")) { g.conv(&format!("{p}.skip_conv"), None, x, 0, stride, 1.0)? } else { x.to_string() };
        Ok(g.op(|o| format!("BinaryOp {p}.add 2 1 {c2} {skip} {o} 0=0")))
    };
    // timm ResNet-18 stages: BasicBlocks, the first of conv2-4 with stride 2.
    let stage = |g: &mut Graph, p: &str, x: &str, stride: usize| -> Result<String> {
        let mut x = x.to_string();
        for i in 0..2 {
            let b = format!("{p}.{i}");
            let s = if i == 0 { stride } else { 1 };
            let y = g.conv(&format!("{b}.conv1"), Some(&format!("{b}.bn1")), &x, 1, s, 1.0)?;
            let y = g.conv(&format!("{b}.conv2"), Some(&format!("{b}.bn2")), &y, 0, 1, 1.0)?;
            let short = if g.w.contains_key(&format!("{b}.downsample.0.weight")) { g.conv(&format!("{b}.downsample.0"), Some(&format!("{b}.downsample.1")), &x, 0, s, 1.0)? } else { x.clone() };
            let sum = g.op(|o| format!("BinaryOp {b}.add 2 1 {y} {short} {o} 0=0"));
            x = g.op(|o| format!("ReLU {b}.relu 1 1 {sum} {o}"));
        }
        Ok(x)
    };
    let o1 = resconv(&mut g, "fnet.ds1", &x, 2)?;
    let o1 = stage(&mut g, "fnet.conv1", &o1, 1)?;
    let o2 = stage(&mut g, "fnet.conv2", &o1, 2)?;
    let o3 = stage(&mut g, "fnet.conv3", &o2, 2)?;
    let o4 = stage(&mut g, "fnet.conv4", &o3, 2)?;
    let u4 = g.deconv("fnet.up_4", &o4, 2)?;
    let s3 = g.op(|o| format!("BinaryOp fnet.sum3 2 1 {o3} {u4} {o} 0=0"));
    let o3 = resconv(&mut g, "fnet.proj_3", &s3, 1)?;
    let u3 = g.deconv("fnet.up_3", &o3, 2)?;
    let s2 = g.op(|o| format!("BinaryOp fnet.sum2 2 1 {o2} {u3} {o} 0=0"));
    let o2 = resconv(&mut g, "fnet.proj_2", &s2, 1)?;
    let u2 = g.deconv("fnet.up_2", &o2, 2)?;
    let s1 = g.op(|o| format!("BinaryOp fnet.sum1 2 1 {o1} {u2} {o} 0=0"));
    let f1 = resconv(&mut g, "fnet.proj_1", &s1, 1)?;
    let da2 = g.op(|o| format!("Interp da_half 2 1 da {f1} {o} 0=2 6=1"));
    let cat = g.op(|o| format!("Concat cat_fmap 2 1 {f1} {da2} {o} 0=0"));
    let fm = g.conv("fmap_conv", None, &cat, 0, 1, 1.0)?;
    g.layers.push(format!("Split out_fmap 1 1 {fm} fmap"));
    g.write(&format!("{out}_feat.param"), &format!("{out}_feat.bin"))?;

    let mut g = Graph::new(&w);
    g.layers.push("Input in_f1 0 1 fmap1".into());
    g.layers.push("Input in_f2 0 1 fmap2".into());
    let c = g.op(|o| format!("Concat cat_hidden 2 1 fmap1 fmap2 {o} 0=0"));
    let n = g.conv("hidden_conv", None, &c, 0, 1, 1.0)?;
    g.layers.push(format!("Split out_net 1 1 {n} net"));
    g.write(&format!("{out}_hidden.param"), &format!("{out}_hidden.bin"))?;

    let mut g = Graph::new(&w);
    for n in ["fmap1", "warp", "net", "flow", "pos"] {
        g.layers.push(format!("Input in_{n} 0 1 {n}"));
    }
    let c = g.op(|o| format!("Concat cat_refine 4 1 fmap1 warp net flow {o} 0=0"));
    let inp = g.conv("warp_linear", None, &c, 0, 1, 1.0)?;
    let t = g.patch_embed("refine_net.patch_embed.proj", &inp)?;
    let mut x = g.op(|o| format!("BinaryOp refine_pos 2 1 {t} pos {o} 0=0"));
    let mut taps = Vec::new();
    for i in 0..12 {
        x = g.vit_block(&format!("refine_net.blks.{i}"), &x, heads)?;
        if [2, 5, 8, 11].contains(&i) {
            taps.push(x.clone());
        }
    }
    let ro = g.dpt("refine_net.dpt_head", [&taps[0], &taps[1], &taps[2], &taps[3]], "fmap1")?;
    let rc = g.op(|o| format!("Concat cat_transform 2 1 {ro} net {o} 0=0"));
    let net = g.conv("refine_transform", None, &rc, 0, 1, 1.0)?;
    g.layers.push(format!("Split out_net 1 1 {net} net_out"));
    let h = g.conv("flow_head.0", None, &net, 1, 1, 1.0)?;
    let fh = g.conv("flow_head.2", None, &h, 0, 1, 1.0)?;
    let dl = g.op(|o| format!("ChannelView delta_view 1 1 {fh} {o} 0=0 1=2"));
    g.layers.push(format!("BinaryOp flow_add 2 1 flow {dl} flow_out 0=0 30=1"));
    let h = g.conv("upsample_weight.0", None, &net, 1, 1, 1.0)?;
    let mw = g.conv("upsample_weight.2", None, &h, 0, 1, 0.25)?;
    g.layers.push(format!("Split out_mask 1 1 {mw} mask"));
    g.write(&format!("{out}_refine.param"), &format!("{out}_refine.bin"))?;

    let mut pos = Vec::new();
    for n in [format!("{d}.pretrained.cls_token"), format!("{d}.pretrained.pos_embed"), "refine_net.pos_embed".to_string()] {
        pos.extend(w.get(&n).with_context(|| format!("{n} missing"))?.1.iter().flat_map(|v| v.to_le_bytes()));
    }
    std::fs::write(format!("{out}_pos.bin"), pos)?;
    println!("waft: written {out}_dpatch / _dblocks / _ddpt / _feat / _hidden / _refine, _pos.bin");
    Ok(())
}

/// NAFNet (megvii-research/NAFNet, MIT) from its checkpoint (extracted .pth, weights
/// under "params"): picture (0..1, padded to a multiple of 16 by the caller) to the
/// restored picture. Width and block counts come from the weights. As the official
/// test setting (NAFNetLocal, train size 256): channel attention averages over a local
/// box of 1.5x the training crop (384 picture pixels, halved per level: `BoxMean`),
/// not the whole picture. Layer scales (beta, gamma) go into conv3 / conv5.
fn nafnet(src: &Path, out: &str) -> Result<()> {
    let w = pth_weights(src)?;
    let has = |n: &str| w.contains_key(n);
    anyhow::ensure!(has("intro.weight") && has("ending.weight"), "not a NAFNet checkpoint");
    let count = |p: &str| (0..).take_while(|i| has(&format!("{p}.{i}.norm1.weight"))).count();
    let levels = (0..).take_while(|i| has(&format!("downs.{i}.weight"))).count();
    let enc: Vec<usize> = (0..levels).map(|i| count(&format!("encoders.{i}"))).collect();
    let dec: Vec<usize> = (0..levels).map(|i| count(&format!("decoders.{i}"))).collect();
    let mid = count("middle_blks");
    println!("nafnet: width {}, encoders {enc:?}, middle {mid}, decoders {dec:?}", w["intro.weight"].0[0]);

    let block = |g: &mut Graph, p: &str, x: &str, level: usize| -> Result<String> {
        let box_k = 384 >> level;
        let gate = |g: &mut Graph, n: &str, x: &str| -> String {
            let c = g.w[&format!("{n}.weight")].0[0] / 2;
            let a = g.op(|o| format!("ChannelView {n}.gate_a 1 1 {x} {o} 0=0 1={c}"));
            let b = g.op(|o| format!("ChannelView {n}.gate_b 1 1 {x} {o} 0={c} 1={c}"));
            g.op(|o| format!("BinaryOp {n}.gate 2 1 {a} {b} {o} 0=2"))
        };
        let n1 = g.norm(&format!("{p}.norm1"), x, 1e-6)?;
        let c1 = g.conv(&format!("{p}.conv1"), None, &n1, 0, 1, 1.0)?;
        let c2 = g.conv(&format!("{p}.conv2"), None, &c1, 0, 1, 1.0)?;
        let m = gate(g, &format!("{p}.conv2"), &c2);
        let pool = g.op(|o| format!("BoxMean {p}.sca.pool 1 1 {m} {o} 0={box_k} 1={box_k}"));
        let s = g.conv(&format!("{p}.sca.1"), None, &pool, 0, 1, 1.0)?;
        let a = g.op(|o| format!("BinaryOp {p}.sca.mul 2 1 {m} {s} {o} 0=2"));
        let beta = g.get(&format!("{p}.beta"))?.1.clone();
        let c3 = g.conv_scaled_rows(&format!("{p}.conv3"), &a, Some(beta))?;
        let y = g.op(|o| format!("BinaryOp {p}.add1 2 1 {x} {c3} {o} 0=0"));
        let n2 = g.norm(&format!("{p}.norm2"), &y, 1e-6)?;
        let c4 = g.conv(&format!("{p}.conv4"), None, &n2, 0, 1, 1.0)?;
        let f = gate(g, &format!("{p}.conv4"), &c4);
        let gamma = g.get(&format!("{p}.gamma"))?.1.clone();
        let c5 = g.conv_scaled_rows(&format!("{p}.conv5"), &f, Some(gamma))?;
        Ok(g.op(|o| format!("BinaryOp {p}.add2 2 1 {y} {c5} {o} 0=0")))
    };

    let mut g = Graph::new(&w);
    g.layers.push("Input input 0 1 img".into());
    let mut x = g.conv("intro", None, "img", 0, 1, 1.0)?;
    let mut skips = Vec::new();
    for (l, &n) in enc.iter().enumerate() {
        for i in 0..n {
            x = block(&mut g, &format!("encoders.{l}.{i}"), &x, l)?;
        }
        skips.push(x.clone());
        x = g.patch_embed(&format!("downs.{l}"), &x)?;
    }
    for i in 0..mid {
        x = block(&mut g, &format!("middle_blks.{i}"), &x, levels)?;
    }
    for (j, &n) in dec.iter().enumerate() {
        let level = levels - 1 - j;
        let u = g.conv(&format!("ups.{j}.0"), None, &x, 0, 1, 1.0)?;
        let u = g.op(|o| format!("PixelShuffle ups.{j}.1 1 1 {u} {o} 0=2"));
        let skip = skips[level].clone();
        x = g.op(|o| format!("BinaryOp skip{j} 2 1 {u} {skip} {o} 0=0"));
        for i in 0..n {
            x = block(&mut g, &format!("decoders.{j}.{i}"), &x, level)?;
        }
    }
    let e = g.conv("ending", None, &x, 0, 1, 1.0)?;
    g.layers.push(format!("BinaryOp residual 2 1 {e} img out 0=0"));
    g.write(&format!("{out}.param"), &format!("{out}.bin"))?;
    println!("nafnet: written {out}.param / .bin ({} MB)", g.bin.len() >> 20);
    Ok(())
}

/// SCUNet (cszn/SCUNet, Apache-2.0) from its checkpoint (extracted .pth): picture
/// (0..1, replicate-padded to a multiple of 64 by the caller) to the restored picture.
/// A U-Net of ConvTransBlocks: a 1x1 splits the channels into a residual conv branch
/// and a Swin branch (8x8 windows, every second block shifted by 4: the model's
/// input_resolution 256 keeps all levels shifted), merged by a 1x1. The relative
/// position table ([heads][2w-1][2w-1] here) is rewritten as SwinIR's
/// [(2w-1)^2][heads], the layout `WindowAttention` reads.
fn scunet(src: &Path, out: &str) -> Result<()> {
    let w = pth_weights(src)?;
    let has = |n: &str| w.contains_key(n);
    anyhow::ensure!(has("m_head.0.weight") && has("m_tail.0.weight"), "not a SCUNet checkpoint");
    let dim = w["m_head.0.weight"].0[0];
    let (head_dim, ws) = (32usize, 8usize);
    let blocks = |p: &str, from: usize| (from..).take_while(|i| has(&format!("{p}.{i}.conv1_1.weight"))).count();
    println!("scunet: dim {dim}, blocks {:?}", [("m_down1", 0), ("m_down2", 0), ("m_down3", 0), ("m_body", 0), ("m_up3", 1), ("m_up2", 1), ("m_up1", 1)].map(|(p, f)| blocks(p, f)));

    let ctb = |g: &mut Graph, p: &str, x: &str, half: usize, shift: usize| -> Result<String> {
        let c = g.conv(&format!("{p}.conv1_1"), None, x, 0, 1, 1.0)?;
        let cx = g.op(|o| format!("ChannelView {p}.conv_view 1 1 {c} {o} 0=0 1={half}"));
        let tx = g.op(|o| format!("ChannelView {p}.trans_view 1 1 {c} {o} 0={half} 1={half}"));
        let r1 = g.conv(&format!("{p}.conv_block.0"), None, &cx, 1, 1, 1.0)?;
        let r2 = g.conv(&format!("{p}.conv_block.2"), None, &r1, 0, 1, 1.0)?;
        let cxo = g.op(|o| format!("BinaryOp {p}.conv_add 2 1 {r2} {cx} {o} 0=0"));
        let t = format!("{p}.trans_block");
        let n1 = g.norm(&format!("{t}.ln1"), &tx, 1e-5)?;
        let qkv = g.conv(&format!("{t}.msa.embedding_layer"), None, &n1, 0, 1, 1.0)?;
        let rp = g.get(&format!("{t}.msa.relative_position_params"))?.clone();
        let (heads, side) = (rp.0[0], rp.0[1]);
        anyhow::ensure!(heads == half / head_dim && side == 2 * ws - 1, "{t}: position table {:?}", rp.0);
        let mut table = vec![0f32; side * side * heads];
        for h in 0..heads {
            for i in 0..side * side {
                table[i * heads + h] = rp.1[h * side * side + i];
            }
        }
        g.floats(&table);
        let a = g.op(|o| format!("WindowAttention {t}.msa 1 1 {qkv} {o} 0={half} 1={heads} 2={ws} 3={shift}"));
        let lin = g.conv(&format!("{t}.msa.linear"), None, &a, 0, 1, 1.0)?;
        let y = g.op(|o| format!("BinaryOp {t}.add1 2 1 {tx} {lin} {o} 0=0"));
        let n2 = g.norm(&format!("{t}.ln2"), &y, 1e-5)?;
        let h = g.conv(&format!("{t}.mlp.0"), None, &n2, 9, 1, 1.0)?;
        let h2 = g.conv(&format!("{t}.mlp.2"), None, &h, 0, 1, 1.0)?;
        let txo = g.op(|o| format!("BinaryOp {t}.add2 2 1 {y} {h2} {o} 0=0"));
        let cat = g.op(|o| format!("Concat {p}.cat 2 1 {cxo} {txo} {o} 0=0"));
        let res = g.conv(&format!("{p}.conv1_2"), None, &cat, 0, 1, 1.0)?;
        Ok(g.op(|o| format!("BinaryOp {p}.add 2 1 {x} {res} {o} 0=0")))
    };
    let stage = |g: &mut Graph, p: &str, x: &str, from: usize, half: usize| -> Result<String> {
        let mut x = x.to_string();
        for i in 0..blocks(p, from) {
            x = ctb(g, &format!("{p}.{}", from + i), &x, half, if i % 2 == 1 { ws / 2 } else { 0 })?;
        }
        Ok(x)
    };

    let mut g = Graph::new(&w);
    g.layers.push("Input input 0 1 img".into());
    let x1 = g.conv("m_head.0", None, "img", 0, 1, 1.0)?;
    let mut skips = vec![x1.clone()];
    let mut x = x1;
    for (l, p) in ["m_down1", "m_down2", "m_down3"].iter().enumerate() {
        x = stage(&mut g, p, &x, 0, (dim << l) / 2)?;
        x = g.patch_embed(&format!("{p}.{}", blocks(p, 0)), &x)?;
        skips.push(x.clone());
    }
    x = stage(&mut g, "m_body", &x, 0, dim * 4)?;
    for (l, p) in ["m_up3", "m_up2", "m_up1"].iter().enumerate() {
        let s = skips[3 - l].clone();
        let sum = g.op(|o| format!("BinaryOp {p}.skip 2 1 {x} {s} {o} 0=0"));
        let u = g.deconv(&format!("{p}.0"), &sum, 2)?;
        x = stage(&mut g, p, &u, 1, (dim << (2 - l)) / 2)?;
    }
    let s = skips[0].clone();
    let sum = g.op(|o| format!("BinaryOp tail.skip 2 1 {x} {s} {o} 0=0"));
    let t = g.conv("m_tail.0", None, &sum, 0, 1, 1.0)?;
    g.layers.push(format!("Split out_img 1 1 {t} out"));
    g.write(&format!("{out}.param"), &format!("{out}.bin"))?;
    println!("scunet: written {out}.param / .bin ({} MB)", g.bin.len() >> 20);
    Ok(())
}

/// Depth Anything V2 Small or Video Depth Anything Small (both Apache-2.0) from their
/// checkpoints (extracted .pth), as graphs for `brp_ai::depth`:
/// - `_dpatch`, `_dblocks`: DINOv2 ViT-S/14 as in `waft` (patch tokens; the 12 blocks,
///   layers 2, 5, 8, 11 after the norm);
/// - the DPT head in four parts, split where Video Depth Anything's temporal modules
///   sit (`_ha`: projections and resizing to r0..r3; `_hb`: r2, r3 to the fused path
///   p4; `_hc`: to p3; `_hd`: to the depth at the input's size, ReLU'd). Per-frame depth
///   feeds each part's outputs straight to the next.
/// - Video Depth Anything only, per temporal module j (0: r2, 1: r3, 2: p4, 3: p3):
///   `_mm{j}_a` (GroupNorm, proj_in, the first attention block's LayerNorm and its
///   q, k, v), `_mm{j}_b` (+ the first attention's output projection; the second
///   block's q, k, v), `_mm{j}_c` (+ the second's output, the GEGLU feed-forward,
///   proj_out, + the module's input); and `_mm{j}_pe.bin`: per attention block, the
///   position encoding (32 window places) through to_q, to_k, to_v (raw f32,
///   [3][32][C]): the projections have no bias, so a cached frame's key at window place
///   i is its own key plus place i's, and the cache keeps projected keys and values.
fn depth(src: &Path, out: &str) -> Result<()> {
    let w = pth_weights(src)?;
    let has = |n: &str| w.contains_key(n);
    let hd = if has("head.projects.0.weight") { "head" } else { "depth_head" };
    anyhow::ensure!(has("pretrained.cls_token") && has(&format!("{hd}.projects.0.weight")), "not a Depth Anything checkpoint");
    let video = has("head.motion_modules.0.temporal_transformer.norm.weight");
    println!("depth: {} ({} tensors)", if video { "Video Depth Anything" } else { "Depth Anything V2" }, w.len());

    let mut g = Graph::new(&w);
    g.layers.push("Input input 0 1 img".into());
    let t = g.patch_embed("pretrained.patch_embed.proj", "img")?;
    g.layers.push(format!("Split out_tokens 1 1 {t} tokens"));
    g.write(&format!("{out}_dpatch.param"), &format!("{out}_dpatch.bin"))?;

    let mut g = Graph::new(&w);
    g.layers.push("Input input 0 1 x".into());
    let mut x = "x".to_string();
    for i in 0..12 {
        x = g.vit_block(&format!("pretrained.blocks.{i}"), &x, 6)?;
        if [2, 5, 8, 11].contains(&i) {
            let n = g.norm("pretrained.norm", &x, 1e-6)?;
            g.layers.push(format!("Split out_l{i} 1 1 {n} l{i}"));
        }
    }
    g.write(&format!("{out}_dblocks.param"), &format!("{out}_dblocks.bin"))?;

    // ResidualConvUnit and FeatureFusionBlock as `Graph::dpt`.
    let rcu = |g: &mut Graph, u: &str, x: &str| -> Result<String> {
        let a = g.op(|o| format!("ReLU {u}.relu 1 1 {x} {o}"));
        let c1 = g.conv(&format!("{u}.conv1"), None, &a, 1, 1, 1.0)?;
        let c2 = g.conv(&format!("{u}.conv2"), None, &c1, 0, 1, 1.0)?;
        Ok(g.op(|o| format!("BinaryOp {u}.add 2 1 {c2} {x} {o} 0=0")))
    };
    let fuse = |g: &mut Graph, n: usize, prev: &str, skip: Option<&str>, size: Option<&str>| -> Result<String> {
        let f = format!("{hd}.scratch.refinenet{n}");
        let mut x = prev.to_string();
        if let Some(s) = skip {
            let r1 = rcu(g, &format!("{f}.resConfUnit1"), s)?;
            x = g.op(|o| format!("BinaryOp {f}.add 2 1 {x} {r1} {o} 0=0"));
        }
        let x = rcu(g, &format!("{f}.resConfUnit2"), &x)?;
        let up = match size {
            Some(s) => g.op(|o| format!("Interp {f}.resize 2 1 {x} {s} {o} 0=2 6=1")),
            None => g.op(|o| format!("Interp {f}.resize 1 1 {x} {o} 0=2 1=2.0 2=2.0 6=1")),
        };
        g.conv(&format!("{f}.out_conv"), None, &up, 0, 1, 1.0)
    };

    let mut g = Graph::new(&w);
    for (i, l) in ["l2", "l5", "l8", "l11"].iter().enumerate() {
        g.layers.push(format!("Input in_{l} 0 1 {l}"));
        let x = g.conv(&format!("{hd}.projects.{i}"), None, l, 0, 1, 1.0)?;
        let x = match i {
            0 => g.deconv(&format!("{hd}.resize_layers.0"), &x, 4)?,
            1 => g.deconv(&format!("{hd}.resize_layers.1"), &x, 2)?,
            2 => x,
            _ => g.conv(&format!("{hd}.resize_layers.3"), None, &x, 0, 2, 1.0)?,
        };
        g.layers.push(format!("Split out_r{i} 1 1 {x} r{i}"));
    }
    g.write(&format!("{out}_ha.param"), &format!("{out}_ha.bin"))?;

    let s = format!("{hd}.scratch");
    let mut g = Graph::new(&w);
    g.layers.push("Input in_r2 0 1 r2".into());
    g.layers.push("Input in_r3 0 1 r3".into());
    let l3 = g.conv(&format!("{s}.layer3_rn"), None, "r2", 0, 1, 1.0)?;
    let l4 = g.conv(&format!("{s}.layer4_rn"), None, "r3", 0, 1, 1.0)?;
    let p4 = fuse(&mut g, 4, &l4, None, Some(&l3))?;
    g.layers.push(format!("Split out_l3rn 1 1 {l3} l3rn"));
    g.layers.push(format!("Split out_p4 1 1 {p4} p4"));
    g.write(&format!("{out}_hb.param"), &format!("{out}_hb.bin"))?;

    let mut g = Graph::new(&w);
    for n in ["p4", "l3rn", "r1"] {
        g.layers.push(format!("Input in_{n} 0 1 {n}"));
    }
    let l2 = g.conv(&format!("{s}.layer2_rn"), None, "r1", 0, 1, 1.0)?;
    let p3 = fuse(&mut g, 3, "p4", Some("l3rn"), Some(&l2))?;
    g.layers.push(format!("Split out_l2rn 1 1 {l2} l2rn"));
    g.layers.push(format!("Split out_p3 1 1 {p3} p3"));
    g.write(&format!("{out}_hc.param"), &format!("{out}_hc.bin"))?;

    let mut g = Graph::new(&w);
    for n in ["p3", "l2rn", "r0", "img"] {
        g.layers.push(format!("Input in_{n} 0 1 {n}"));
    }
    let l1 = g.conv(&format!("{s}.layer1_rn"), None, "r0", 0, 1, 1.0)?;
    let p2 = fuse(&mut g, 2, "p3", Some("l2rn"), Some(&l1))?;
    let p1 = fuse(&mut g, 1, &p2, Some(&l1), None)?;
    let o = g.conv(&format!("{s}.output_conv1"), None, &p1, 0, 1, 1.0)?;
    let o = g.op(|r| format!("Interp {s}.out_resize 2 1 {o} img {r} 0=2 6=1"));
    // The depth's range (tens to hundreds) and its final convolutions stay f32, as
    // the official code (autocast off there).
    let o = g.conv(&format!("{s}.output_conv2.0"), None, &o, 1, 1, 1.0)?;
    let o = g.conv(&format!("{s}.output_conv2.2"), None, &o, 1, 1, 1.0)?;
    g.layers.last_mut().unwrap().push_str(" 30=1");
    g.layers.push(format!("Split out_depth 1 1 {o} depth"));
    g.write(&format!("{out}_hd.param"), &format!("{out}_hd.bin"))?;

    if video {
        for j in 0..4 {
            let tt = format!("head.motion_modules.{j}.temporal_transformer");
            let blk = format!("{tt}.transformer_blocks.0");
            let c = w[&format!("{tt}.norm.weight")].1.len();
            let qkv = |g: &mut Graph, b: usize, y: &str| -> Result<()> {
                let a = format!("{blk}.attention_blocks.{b}");
                let nh = g.norm(&format!("{blk}.norms.{b}"), y, 1e-5)?;
                for (k, n) in ["q", "k", "v"].iter().enumerate() {
                    let r = g.conv(&format!("{a}.to_{n}"), None, &nh, 0, 1, 1.0)?;
                    let _ = k;
                    g.layers.push(format!("Split out_{n} 1 1 {r} {n}"));
                }
                Ok(())
            };
            let mut g = Graph::new(&w);
            g.layers.push("Input in_x 0 1 x".into());
            let (gw, gb) = (w[&format!("{tt}.norm.weight")].1.clone(), w[&format!("{tt}.norm.bias")].1.clone());
            g.floats(&gw);
            g.floats(&gb);
            let n0 = g.op(|o| format!("GroupNorm {tt}.norm 1 1 x {o} 0=32 1={c} 2=1e-6"));
            let y = g.conv(&format!("{tt}.proj_in"), None, &n0, 0, 1, 1.0)?;
            g.layers.push(format!("Split out_y 1 1 {y} y"));
            qkv(&mut g, 0, &y)?;
            g.write(&format!("{out}_mm{j}_a.param"), &format!("{out}_mm{j}_a.bin"))?;

            let mut g = Graph::new(&w);
            g.layers.push("Input in_y 0 1 y".into());
            g.layers.push("Input in_o 0 1 o".into());
            let p = g.conv(&format!("{blk}.attention_blocks.0.to_out.0"), None, "o", 0, 1, 1.0)?;
            let y1 = g.op(|r| format!("BinaryOp {blk}.add0 2 1 y {p} {r} 0=0"));
            g.layers.push(format!("Split out_y1 1 1 {y1} y1"));
            qkv(&mut g, 1, &y1)?;
            g.write(&format!("{out}_mm{j}_b.param"), &format!("{out}_mm{j}_b.bin"))?;

            let mut g = Graph::new(&w);
            for n in ["x", "y1", "o"] {
                g.layers.push(format!("Input in_{n} 0 1 {n}"));
            }
            let p = g.conv(&format!("{blk}.attention_blocks.1.to_out.0"), None, "o", 0, 1, 1.0)?;
            let y2 = g.op(|r| format!("BinaryOp {blk}.add1 2 1 y1 {p} {r} 0=0"));
            let f = g.norm(&format!("{blk}.ff_norm"), &y2, 1e-5)?;
            let pr = g.conv(&format!("{blk}.ff.net.0.proj"), None, &f, 0, 1, 1.0)?;
            let inner = w[&format!("{blk}.ff.net.0.proj.weight")].0[0] / 2;
            let hv = g.op(|r| format!("ChannelView {blk}.ff.h 1 1 {pr} {r} 0=0 1={inner}"));
            let gv = g.op(|r| format!("ChannelView {blk}.ff.gate 1 1 {pr} {r} 0={inner} 1={inner}"));
            let ge = g.op(|r| format!("GELU {blk}.ff.gelu 1 1 {gv} {r}"));
            let m = g.op(|r| format!("BinaryOp {blk}.ff.mul 2 1 {hv} {ge} {r} 0=2"));
            let f2 = g.conv(&format!("{blk}.ff.net.2"), None, &m, 0, 1, 1.0)?;
            let y3 = g.op(|r| format!("BinaryOp {blk}.add2 2 1 {y2} {f2} {r} 0=0"));
            let po = g.conv(&format!("{tt}.proj_out"), None, &y3, 0, 1, 1.0)?;
            g.layers.push(format!("BinaryOp {tt}.residual 2 1 x {po} out 0=0"));
            g.write(&format!("{out}_mm{j}_c.param"), &format!("{out}_mm{j}_c.bin"))?;

            // Position encodings through the (bias-free) projections.
            let mut pe = Vec::new();
            for b in 0..2 {
                let a = format!("{blk}.attention_blocks.{b}");
                let (pshape, pv) = w[&format!("{a}.pos_encoder.pe")].clone();
                let (places, pc) = (pshape[1], pshape[2]);
                anyhow::ensure!(places == 32 && pc == c, "{a}: position encoding {pshape:?}");
                for n in ["q", "k", "v"] {
                    let (ws, wv) = &w[&format!("{a}.to_{n}.weight")];
                    anyhow::ensure!(!w.contains_key(&format!("{a}.to_{n}.bias")) && ws[0] == c, "{a}.to_{n}");
                    for i in 0..32 {
                        for o in 0..c {
                            let s: f32 = (0..c).map(|k| wv[o * c + k] * pv[i * c + k]).sum();
                            pe.extend(s.to_le_bytes());
                        }
                    }
                }
            }
            std::fs::write(format!("{out}_mm{j}_pe.bin"), pe)?;
        }
    }
    println!("depth: written {out}_dpatch / _dblocks / _ha.._hd{}", if video { " / _mm0.._mm3" } else { "" });
    let mut pos = Vec::new();
    for n in ["pretrained.cls_token", "pretrained.pos_embed"] {
        pos.extend(w.get(n).with_context(|| format!("{n} missing"))?.1.iter().flat_map(|v| v.to_le_bytes()));
    }
    std::fs::write(format!("{out}_pos.bin"), pos)?;
    Ok(())
}

/// BiRefNet (ZhengPeng7/BiRefNet, MIT) matting weights (.safetensors) as one graph
/// at a fixed square size: input `img` (RGB 0..1, size x size), output `out` (alpha
/// 0..1, same size). The configuration of the released matting models: Swin v1
/// backbone (T or L, from the weights) run at the size and at half of it, features
/// concatenated; the three finer levels squeezed into the coarsest (BasicDecBlk);
/// BasicDecBlk decoder with ASPPDeformable; the image cut in tiles fed to every
/// level; gradient attention. As tools/airef/birefnet_ref.py.
/// - ImageNet normalisation: a 1x1 convolution first.
/// - Swin blocks pad to whole windows after norm1 (Canvas), as Swin v1, and crop
///   after the projection; patch merging is a pixel unshuffle with the norm and
///   reduction permuted to its channel order.
/// - Deformable convolutions: offsets and modulator in one convolution, the sampling
///   (DeformIm2col), then a 1x1 convolution with the BatchNorm folded in.
/// - ASPP's pooled branch: its share of conv1 is added per channel (broadcast).
fn birefnet(src: &Path, out: &str, size: usize) -> Result<()> {
    let mut w = safetensors_weights(src)?;
    anyhow::ensure!(w.contains_key("bb.patch_embed.proj.weight") && w.contains_key("decoder.conv_out1.0.weight"), "not a BiRefNet checkpoint");
    let embed = w["bb.patch_embed.proj.weight"].0[0];
    let depth = |w: &HashMap<String, (Vec<usize>, Vec<f32>)>, i: usize| (0..).take_while(|j| w.contains_key(&format!("bb.layers.{i}.blocks.{j}.norm1.weight"))).count();
    let depths: Vec<usize> = (0..4).map(|i| depth(&w, i)).collect();
    let win = (((w["bb.layers.0.blocks.0.attn.relative_position_bias_table"].0[0] as f64).sqrt() as usize) + 1) / 2;
    let heads: Vec<usize> = (0..4).map(|i| w[&format!("bb.layers.{i}.blocks.0.attn.relative_position_bias_table")].0[1]).collect();
    anyhow::ensure!(size % 64 == 0, "size {size}: a multiple of 64");
    println!("birefnet: Swin embed {embed}, depths {depths:?}, heads {heads:?}, window {win}, size {size}");

    // ImageNet normalisation as a 1x1 convolution.
    let (mean, std) = ([0.485f32, 0.456, 0.406], [0.229f32, 0.224, 0.225]);
    let mut pw = vec![0f32; 9];
    for c in 0..3 {
        pw[c * 3 + c] = 1.0 / std[c];
    }
    w.insert("prep.weight".into(), (vec![3, 3, 1, 1], pw));
    w.insert("prep.bias".into(), (vec![3], (0..3).map(|c| -mean[c] / std[c]).collect()));
    // Patch merging in pixel-unshuffle order: channel c * 4 + dy * 2 + dx is Swin's
    // (dy + 2 dx) * C + c.
    for i in 0..3 {
        let c = embed << i;
        let s = |u: usize| ((u % 4) / 2 + 2 * (u % 2)) * c + u / 4;
        let p = format!("bb.layers.{i}.downsample");
        for n in ["norm.weight", "norm.bias"] {
            let k = format!("{p}.{n}");
            let v = w[&k].1.clone();
            w.get_mut(&k).unwrap().1 = (0..4 * c).map(|u| v[s(u)]).collect();
        }
        let k = format!("{p}.reduction.weight");
        let (shape, v) = w[&k].clone();
        let mut nv = vec![0f32; v.len()];
        for o in 0..shape[0] {
            for u in 0..4 * c {
                nv[o * 4 * c + u] = v[o * 4 * c + s(u)];
            }
        }
        w.get_mut(&k).unwrap().1 = nv;
    }
    // Deformable convolutions: offsets + modulator in one; the regular weights as 1x1
    // over the sampled [C x K^2]. ASPP conv1 split at the pooled branch, bn1 folded.
    let keys: Vec<String> = w.keys().filter(|k| k.ends_with(".atrous_conv.regular_conv.weight")).cloned().collect();
    for k in keys {
        let p = k.trim_end_matches(".regular_conv.weight").to_string();
        let (s, v) = w[&k].clone();
        w.insert(format!("{p}.reg.weight"), (vec![s[0], s[1] * s[2] * s[3]], v));
        let (os, ov) = w[&format!("{p}.offset_conv.weight")].clone();
        let (ms, mv) = w[&format!("{p}.modulator_conv.weight")].clone();
        let mut ob = w[&format!("{p}.offset_conv.bias")].1.clone();
        ob.extend(&w[&format!("{p}.modulator_conv.bias")].1);
        w.insert(format!("{p}.om.weight"), (vec![os[0] + ms[0], os[1], os[2], os[3]], [ov, mv].concat()));
        w.insert(format!("{p}.om.bias"), (vec![ob.len()], ob));
    }
    let keys: Vec<String> = w.keys().filter(|k| k.ends_with(".dec_att.conv1.weight")).cloned().collect();
    for k in keys {
        let p = k.trim_end_matches(".conv1.weight").to_string();
        let (s, v) = w[&k].clone();
        let (oc, ic) = (s[0], s[1]);
        let bn = |n: &str| w[&format!("{p}.bn1.{n}")].1.clone();
        let (g, b, m, var) = (bn("weight"), bn("bias"), bn("running_mean"), bn("running_var"));
        let sc: Vec<f32> = (0..oc).map(|o| g[o] / (var[o] + 1e-5).sqrt()).collect();
        let split = ic - 256;
        let mut a = Vec::with_capacity(oc * split);
        let mut bb = Vec::with_capacity(oc * 256);
        for o in 0..oc {
            a.extend(v[o * ic..o * ic + split].iter().map(|x| x * sc[o]));
            bb.extend(v[o * ic + split..(o + 1) * ic].iter().map(|x| x * sc[o]));
        }
        w.insert(format!("{p}.conv1a.weight"), (vec![oc, split], a));
        w.insert(format!("{p}.conv1a.bias"), (vec![oc], (0..oc).map(|o| b[o] - m[o] * sc[o]).collect()));
        w.insert(format!("{p}.conv1b.weight"), (vec![oc, 256], bb));
    }

    let mut g = Graph::new(&w);
    g.layers.push("Input input 0 1 img".into());
    let up = |g: &mut Graph, name: &str, x: &str, s: usize| g.op(|o| format!("Interp {name} 1 1 {x} {o} 0=2 3={s} 4={s} 6=1"));
    let add = |g: &mut Graph, name: &str, a: &str, b: &str| g.op(|o| format!("BinaryOp {name} 2 1 {a} {b} {o} 0=0"));
    let x = g.conv("prep", None, "img", 0, 1, 1.0)?;

    // Swin v1: the four stages' outputs (after norm{i}), at size / 4 / 2^i.
    let swin = |g: &mut Graph, x: &str, s: usize| -> Result<Vec<String>> {
        let pe = g.patch_embed("bb.patch_embed.proj", x)?;
        let mut x = g.norm("bb.patch_embed.norm", &pe, 1e-5)?;
        let mut h = s / 4;
        let mut outs = Vec::new();
        for i in 0..4 {
            let dim = embed << i;
            let hp = h.div_ceil(win) * win;
            for j in 0..depths[i] {
                let p = format!("bb.layers.{i}.blocks.{j}");
                let shift = if j % 2 == 1 { win / 2 } else { 0 };
                let mut n1 = g.norm(&format!("{p}.norm1"), &x, 1e-5)?;
                if hp != h {
                    n1 = g.op(|o| format!("Canvas {p}.pad 1 1 {n1} {o} 0={hp} 1={hp}"));
                }
                let qkv = g.conv(&format!("{p}.attn.qkv"), None, &n1, 0, 1, 1.0)?;
                let table = g.get(&format!("{p}.attn.relative_position_bias_table"))?.1.clone();
                g.floats(&table);
                let a = g.op(|o| format!("WindowAttention {p}.attn 1 1 {qkv} {o} 0={dim} 1={} 2={win} 3={shift}", heads[i]));
                let mut pj = g.conv(&format!("{p}.attn.proj"), None, &a, 0, 1, 1.0)?;
                if hp != h {
                    pj = g.op(|o| format!("Canvas {p}.crop 1 1 {pj} {o} 0={h} 1={h}"));
                }
                let y = add(g, &format!("{p}.add1"), &x, &pj);
                let n2 = g.norm(&format!("{p}.norm2"), &y, 1e-5)?;
                let f1 = g.conv(&format!("{p}.mlp.fc1"), None, &n2, 9, 1, 1.0)?;
                let f2 = g.conv(&format!("{p}.mlp.fc2"), None, &f1, 0, 1, 1.0)?;
                x = add(g, &format!("{p}.add2"), &y, &f2);
            }
            outs.push(g.norm(&format!("bb.norm{i}"), &x, 1e-5)?);
            if i < 3 {
                let p = format!("bb.layers.{i}.downsample");
                let u = g.op(|o| format!("PixelUnshuffle {p}.unshuffle 1 1 {x} {o} 0=2"));
                let n = g.norm(&format!("{p}.norm"), &u, 1e-5)?;
                x = g.conv(&format!("{p}.reduction"), None, &n, 0, 1, 1.0)?;
                h /= 2;
            }
        }
        Ok(outs)
    };
    let xh = up(&mut g, "half", &x, size / 2);
    let f = swin(&mut g, &x, size)?;
    let fh = swin(&mut g, &xh, size / 2)?;
    let mut feats = Vec::new();
    for i in 0..4 {
        let s = size / 4 >> i;
        let u = up(&mut g, &format!("cat{i}.up"), &fh[i], s);
        let a = f[i].clone();
        feats.push(g.op(|o| format!("Concat cat{i} 2 1 {a} {u} {o} 0=0")));
    }

    // Decoder blocks.
    let deform = |g: &mut Graph, p: &str, x: &str, k: usize| -> Result<String> {
        let om = g.conv(&format!("{p}.atrous_conv.om"), None, x, 0, 1, 1.0)?;
        let s = g.op(|o| format!("DeformIm2col {p}.sample 2 1 {x} {om} {o} 0={k} 1={}", k / 2));
        g.conv(&format!("{p}.atrous_conv.reg"), Some(&format!("{p}.bn")), &s, 1, 1, 1.0)
    };
    let aspp = |g: &mut Graph, p: &str, x: &str| -> Result<String> {
        let mut br = vec![deform(g, &format!("{p}.aspp1"), x, 1)?];
        for (i, k) in [1usize, 3, 7].into_iter().enumerate() {
            br.push(deform(g, &format!("{p}.aspp_deforms.{i}"), x, k)?);
        }
        let cat = br.join(" ");
        let c = g.op(|o| format!("Concat {p}.cat 4 1 {cat} {o} 0=0"));
        let pool = g.op(|o| format!("Pooling {p}.pool 1 1 {x} {o} 0=1 4=1"));
        let gp = g.conv(&format!("{p}.global_avg_pool.1"), Some(&format!("{p}.global_avg_pool.2")), &pool, 1, 1, 1.0)?;
        let ya = g.conv(&format!("{p}.conv1a"), None, &c, 0, 1, 1.0)?;
        let yb = g.conv(&format!("{p}.conv1b"), None, &gp, 0, 1, 1.0)?;
        let y = g.op(|o| format!("BinaryOp {p}.add 2 1 {ya} {yb} {o} 0=0 20=1"));
        Ok(g.op(|o| format!("ReLU {p}.relu 1 1 {y} {o}")))
    };
    let dec_blk = |g: &mut Graph, p: &str, x: &str| -> Result<String> {
        let c = g.conv(&format!("{p}.conv_in"), Some(&format!("{p}.bn_in")), x, 1, 1, 1.0)?;
        let a = aspp(g, &format!("{p}.dec_att"), &c)?;
        g.conv(&format!("{p}.conv_out"), Some(&format!("{p}.bn_out")), &a, 0, 1, 1.0)
    };
    let simple = |g: &mut Graph, p: &str, x: &str| -> Result<String> {
        let a = g.conv(&format!("{p}.conv1"), None, x, 0, 1, 1.0)?;
        g.conv(&format!("{p}.conv_out"), None, &a, 0, 1, 1.0)
    };
    let tiles = |g: &mut Graph, name: &str, t: usize| -> String {
        if t == size {
            x.clone()
        } else {
            g.op(|o| format!("Tiles {name} 1 1 {x} {o} 0={t} 1={t}"))
        }
    };
    let gdt = |g: &mut Graph, i: usize, p: &str| -> Result<String> {
        let a = g.conv(&format!("decoder.gdt_convs_{i}.0"), Some(&format!("decoder.gdt_convs_{i}.1")), p, 1, 1, 1.0)?;
        let b = g.conv(&format!("decoder.gdt_convs_attn_{i}.0"), None, &a, 0, 1, 1.0)?;
        let s = g.op(|o| format!("Sigmoid decoder.gdt_sig_{i} 1 1 {b} {o}"));
        Ok(g.op(|o| format!("BinaryOp decoder.gdt_mul_{i} 2 1 {p} {s} {o} 0=2")))
    };

    let s4 = size / 32;
    let mut cx = Vec::new();
    for i in 0..3 {
        cx.push(up(&mut g, &format!("cxt{i}"), &feats[i], s4));
    }
    cx.push(feats[3].clone());
    let cxs = cx.join(" ");
    let x4 = g.op(|o| format!("Concat cxt 4 1 {cxs} {o} 0=0"));
    let x4 = dec_blk(&mut g, "squeeze_module.0", &x4)?;

    // Level 4 (size / 32) up to 1 (size / 4), then the full size.
    let mut p = x4;
    for (lvl, (skip, lat)) in [(None, None), (Some(2usize), Some(4usize)), (Some(1), Some(3)), (Some(0), Some(2))].into_iter().enumerate() {
        let s = size / 32 << lvl;
        if let (Some(sk), Some(lb)) = (skip, lat) {
            let u = up(&mut g, &format!("decoder.up{lb}"), &p, s);
            let l = g.conv(&format!("decoder.lateral_block{lb}.conv"), None, &feats[sk], 0, 1, 1.0)?;
            p = add(&mut g, &format!("decoder.lat_add{lb}"), &u, &l);
        }
        let t = tiles(&mut g, &format!("decoder.tiles{lvl}"), s);
        let ib = simple(&mut g, &format!("decoder.ipt_blk{}", 5 - lvl), &t)?;
        let pp = p.clone();
        let c = g.op(|o| format!("Concat decoder.ipt_cat{lvl} 2 1 {pp} {ib} {o} 0=0"));
        p = dec_blk(&mut g, &format!("decoder.decoder_block{}", 4 - lvl), &c)?;
        if lvl < 3 {
            p = gdt(&mut g, 4 - lvl, &p)?;
        }
    }
    let u = up(&mut g, "decoder.up_full", &p, size);
    let ib = simple(&mut g, "decoder.ipt_blk1", &x)?;
    let c = g.op(|o| format!("Concat decoder.ipt_cat_full 2 1 {u} {ib} {o} 0=0"));
    let logit = g.conv("decoder.conv_out1.0", None, &c, 0, 1, 1.0)?;
    let a = g.op(|o| format!("Sigmoid sigmoid 1 1 {logit} {o}"));
    g.layers.push(format!("Split out_split 1 1 {a} out"));
    g.write(&format!("{out}.param"), &format!("{out}.bin"))?;
    println!("birefnet: written {out}.param / .bin ({} layers, {} MB)", g.layers.len(), g.bin.len() >> 20);
    Ok(())
}

fn main() -> Result<()> {
    let a: Vec<String> = std::env::args().skip(1).collect();
    if a.first().map(String::as_str) == Some("birefnet") {
        anyhow::ensure!(a.len() == 3 || a.len() == 4, "usage: pth2ncnn birefnet <model.safetensors> <out stem> [size, default 1024]");
        return birefnet(Path::new(&a[1]), &a[2], a.get(3).map_or(Ok(1024), |s| s.parse())?);
    }
    if a.first().map(String::as_str) == Some("depth") {
        anyhow::ensure!(a.len() == 3, "usage: pth2ncnn depth <extracted .pth folder> <out stem>");
        return depth(Path::new(&a[1]), &a[2]);
    }
    if a.first().map(String::as_str) == Some("scunet") {
        anyhow::ensure!(a.len() == 3, "usage: pth2ncnn scunet <extracted .pth folder> <out stem>");
        return scunet(Path::new(&a[1]), &a[2]);
    }
    if a.first().map(String::as_str) == Some("nafnet") {
        anyhow::ensure!(a.len() == 3, "usage: pth2ncnn nafnet <extracted .pth folder> <out stem>");
        return nafnet(Path::new(&a[1]), &a[2]);
    }
    if a.first().map(String::as_str) == Some("waft") {
        anyhow::ensure!(a.len() == 3, "usage: pth2ncnn waft <extracted .pth folder> <out stem>");
        return waft(Path::new(&a[1]), &a[2]);
    }
    if a.first().map(String::as_str) == Some("searaft") {
        anyhow::ensure!(a.len() == 3, "usage: pth2ncnn searaft <model.safetensors> <out stem>");
        return searaft(Path::new(&a[1]), &a[2]);
    }
    if a.first().map(String::as_str) == Some("dat") {
        anyhow::ensure!(a.len() == 4 || a.len() == 5, "usage: pth2ncnn dat <model.onnx|.safetensors> <out.param> <out.bin> [mean r,g,b]");
        return dat(Path::new(&a[1]), &a[2], &a[3], a.get(4).map(String::as_str));
    }
    // wcmp <a> <b>: tensors of shapes found once in each file, compared by shape
    // (exports that renamed their weights).
    if a.first().map(String::as_str) == Some("wcmp") {
        let load = |p: &str| if p.ends_with(".safetensors") { safetensors_weights(Path::new(p)) } else { onnx_weights(Path::new(p)) };
        let (x, y) = (load(&a[1])?, load(&a[2])?);
        let once = |m: &HashMap<String, (Vec<usize>, Vec<f32>)>| {
            let mut c: HashMap<Vec<usize>, Vec<String>> = HashMap::new();
            for (k, (d, _)) in m {
                c.entry(d.clone()).or_default().push(k.clone());
            }
            c.into_iter().filter(|(_, v)| v.len() == 1).map(|(d, v)| (d, v[0].clone())).collect::<HashMap<_, _>>()
        };
        let (ox, oy) = (once(&x), once(&y));
        for (d, kx) in &ox {
            if let Some(ky) = oy.get(d) {
                let (vx, vy) = (&x[kx].1, &y[ky].1);
                let diff = vx.iter().zip(vy).map(|(p, q)| (p - q).abs()).fold(0f32, f32::max);
                println!("{d:?}: {kx} / {ky}: largest difference {diff:e}");
            }
        }
        return Ok(());
    }
    if a.first().map(String::as_str) == Some("onnxlist") {
        let w = onnx_weights(Path::new(&a[1]))?;
        let mut names: Vec<_> = w.iter().collect();
        names.sort_by(|x, y| x.0.cmp(y.0));
        for (n, (d, v)) in names {
            if v.len() <= 8 { println!("{n}: {d:?} {v:?}") } else { println!("{n}: {d:?} ({} values)", v.len()) }
        }
        return Ok(());
    }
    if a.first().map(String::as_str) == Some("swinir") {
        anyhow::ensure!(a.len() == 4, "usage: pth2ncnn swinir <extracted .pth folder> <out.param> <out.bin>");
        let root = Path::new(&a[1]);
        let dir = std::fs::read_dir(root)?.flatten().map(|e| e.path()).find(|p| p.join("data.pkl").exists()).context("no data.pkl in the extracted .pth")?;
        return swinir(&dir, &a[2], &a[3]);
    }
    if a.first().map(String::as_str) == Some("compact") {
        anyhow::ensure!(a.len() == 4, "usage: pth2ncnn compact <extracted .pth folder> <out.param> <out.bin>");
        // The folder holding data.pkl (usually "archive"; some releases name it otherwise).
        let root = Path::new(&a[1]);
        let dir = std::fs::read_dir(root)?.flatten().map(|e| e.path()).find(|p| p.join("data.pkl").exists()).context("no data.pkl in the extracted .pth")?;
        let _ = root;
        return compact(&dir, &a[2], &a[3]);
    }
    anyhow::ensure!(a.len() == 4, "usage: pth2ncnn <extracted .pth folder> <x4plus.param> <out.param> <out.bin>");
    let dir = Path::new(&a[0]).join("archive");
    let pkl = std::fs::read(dir.join("data.pkl")).context("data.pkl")?;
    let all = tensors(&unpickle(&pkl)?)?;
    let by_name: HashMap<&str, &Tensor> = all.iter().map(|(n, t)| (n.as_str(), t)).collect();

    // Convolutions in graph order (RRDBNet forward order).
    let mut convs = vec!["conv_first".to_string()];
    for b in 0..23 {
        for r in 1..=3 {
            for c in 1..=5 {
                convs.push(format!("body.{b}.rdb{r}.conv{c}"));
            }
        }
    }
    convs.extend(["conv_body", "conv_up1", "conv_up2", "conv_hr", "conv_last"].map(String::from));

    let template = std::fs::read_to_string(&a[1])?;
    let graph_convs: Vec<usize> = template
        .lines()
        .filter(|l| l.starts_with("Convolution"))
        .map(|l| l.split_whitespace().find_map(|f| f.strip_prefix("6=")).and_then(|v| v.parse().ok()).unwrap_or(0))
        .collect();
    anyhow::ensure!(graph_convs.len() == convs.len(), "template has {} convolutions, weights {}", graph_convs.len(), convs.len());

    let mut bin: Vec<u8> = Vec::new();
    let mut param = template.clone();
    for (i, name) in convs.iter().enumerate() {
        let w = by_name.get(format!("{name}.weight").as_str()).with_context(|| format!("{name}.weight missing"))?;
        let b = by_name.get(format!("{name}.bias").as_str()).with_context(|| format!("{name}.bias missing"))?;
        let n: usize = w.shape.iter().product();
        if i == 0 {
            // conv_first reads the folded input: 12 channels.
            anyhow::ensure!(w.shape == [64, 12, 3, 3], "conv_first is {:?}", w.shape);
            param = param.replacen("6=1728", &format!("6={n}"), 1);
        } else {
            anyhow::ensure!(n == graph_convs[i], "{name}: {n} weights, graph expects {}", graph_convs[i]);
        }
        let read = |t: &Tensor, count: usize| -> Result<Vec<u8>> {
            let b = t.bytes(&dir)?;
            anyhow::ensure!(b.len() == count * 4, "tensor size");
            Ok(b)
        };
        // Weights: 4-byte tag 0 = raw float32; bias: raw float32.
        bin.extend(0u32.to_le_bytes());
        bin.extend(read(w, n)?);
        bin.extend(read(b, b.shape.iter().product())?);
    }
    std::fs::write(&a[2], param)?;
    std::fs::write(&a[3], &bin)?;
    println!("{} convolutions, {} MB", convs.len(), bin.len() >> 20);
    Ok(())
}

/// Row-major strides of a shape.
fn contiguous(shape: &[usize]) -> Vec<usize> {
    let mut s = vec![1; shape.len()];
    for d in (0..shape.len().saturating_sub(1)).rev() {
        s[d] = s[d + 1] * shape[d + 1];
    }
    s
}
