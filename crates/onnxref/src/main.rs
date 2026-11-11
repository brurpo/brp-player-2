//! onnxref <model.onnx> <in.png> <out.png> [out.f32]
//!
//! Runs an image model (one input NCHW float 0..1 RGB, one output the same) with ONNX
//! Runtime on the CPU and saves the result: the reference `aicheck` compares our GPU
//! runner against. `out.f32` gets the raw planar output (c, h, w as u32, then floats).

use anyhow::{Context, Result};
use ort::session::Session;
use ort::value::Tensor;

fn main() -> Result<()> {
    let a: Vec<String> = std::env::args().skip(1).collect();
    anyhow::ensure!(a.len() >= 3, "usage: onnxref <model.onnx> <in.png> <out.png> [out.f32]");
    let dec = png::Decoder::new(std::io::BufReader::new(std::fs::File::open(&a[1])?));
    let mut r = dec.read_info()?;
    let mut buf = vec![0; r.output_buffer_size().context("png size")?];
    let info = r.next_frame(&mut buf)?;
    anyhow::ensure!(info.bit_depth == png::BitDepth::Eight, "need an 8-bit PNG");
    let ch = info.color_type.samples();
    let (w, h) = (info.width as usize, info.height as usize);
    let mut input = vec![0f32; 3 * h * w];
    for y in 0..h {
        for x in 0..w {
            for c in 0..3 {
                input[(c * h + y) * w + x] = buf[(y * w + x) * ch + c] as f32 / 255.0;
            }
        }
    }
    let env = ort::init().build()?;
    let mut session = Session::builder(&env)?.commit_from_file(&a[0])?;
    let in_name = session.inputs()[0].name().to_string();
    let tensor = Tensor::from_array(([1usize, 3, h, w], input))?;
    let t = std::time::Instant::now();
    let outputs = session.run(ort::inputs![in_name.as_str() => tensor])?;
    println!("ran in {:?}", t.elapsed());
    let (shape, out) = outputs[0].try_extract_tensor::<f32>()?;
    anyhow::ensure!(shape.len() == 4 && shape[1] == 3, "unexpected output shape {shape:?}");
    let (oh, ow) = (shape[2] as usize, shape[3] as usize);
    let at = |c: usize, y: usize, x: usize| out[(c * oh + y) * ow + x];
    let mut bytes = Vec::with_capacity(oh * ow * 3);
    for y in 0..oh {
        for x in 0..ow {
            for c in 0..3 {
                bytes.push((at(c, y, x).clamp(0.0, 1.0) * 255.0).round() as u8);
            }
        }
    }
    let mut enc = png::Encoder::new(std::io::BufWriter::new(std::fs::File::create(&a[2])?), ow as u32, oh as u32);
    enc.set_color(png::ColorType::Rgb);
    enc.set_depth(png::BitDepth::Eight);
    enc.write_header()?.write_image_data(&bytes)?;
    if let Some(p) = a.get(3) {
        let mut raw = Vec::with_capacity(12 + out.len() * 4);
        for d in [3u32, oh as u32, ow as u32] {
            raw.extend(d.to_le_bytes());
        }
        for v in out {
            raw.extend(v.to_le_bytes());
        }
        std::fs::write(p, raw)?;
    }
    println!("{w}x{h} -> {ow}x{oh}");
    Ok(())
}
