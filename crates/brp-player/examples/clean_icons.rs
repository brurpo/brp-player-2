//! Writes cleaned copies of PNG icons: near-zero alpha (the faint 2% background
//! fill some icons carry) becomes fully transparent; other alpha is rescaled so
//! anti-aliased edges stay smooth. Same rule the player applies at load time.
//!
//!   cargo run -p brp-player --release --example clean_icons -- <in_dir> <out_dir>

use std::path::Path;

const FLOOR: u32 = 12;

fn main() -> anyhow::Result<()> {
    let mut args = std::env::args().skip(1);
    let input = args.next().unwrap_or_else(|| "icons".into());
    let output = args.next().unwrap_or_else(|| "icons/clean".into());
    std::fs::create_dir_all(&output)?;

    let mut done = 0;
    for entry in std::fs::read_dir(&input)? {
        let path = entry?.path();
        if path.extension().is_none_or(|e| !e.eq_ignore_ascii_case("png")) {
            continue;
        }
        let name = path.file_name().unwrap();
        let (w, h, mut rgba) = decode(&path)?;
        let mut cleared = 0usize;
        for p in rgba.chunks_mut(4) {
            let a = p[3] as u32;
            if a <= FLOOR {
                if a > 0 {
                    cleared += 1;
                }
                p[3] = 0;
            } else {
                p[3] = ((a - FLOOR) * 255 / (255 - FLOOR)) as u8;
            }
        }
        let out = Path::new(&output).join(name);
        encode(&out, w, h, &rgba)?;
        println!("{:<32} {w}x{h}  cleared {cleared} faint pixels", name.to_string_lossy());
        done += 1;
    }
    println!("{done} icons written to {output}");
    Ok(())
}

fn decode(path: &Path) -> anyhow::Result<(u32, u32, Vec<u8>)> {
    let mut dec = png::Decoder::new(std::io::BufReader::new(std::fs::File::open(path)?));
    dec.set_transformations(png::Transformations::EXPAND | png::Transformations::STRIP_16);
    let mut reader = dec.read_info()?;
    let mut buf = vec![0; reader.output_buffer_size().unwrap_or(0)];
    let info = reader.next_frame(&mut buf)?;
    let (w, h) = (info.width, info.height);
    let n = (w * h) as usize;
    let rgba = match info.color_type {
        png::ColorType::Rgba => buf[..n * 4].to_vec(),
        png::ColorType::Rgb => buf[..n * 3].chunks(3).flat_map(|p| [p[0], p[1], p[2], 255]).collect(),
        png::ColorType::GrayscaleAlpha => buf[..n * 2].chunks(2).flat_map(|p| [p[0], p[0], p[0], p[1]]).collect(),
        png::ColorType::Grayscale => buf[..n].iter().flat_map(|&g| [g, g, g, 255]).collect(),
        other => anyhow::bail!("{}: unsupported color type {other:?}", path.display()),
    };
    Ok((w, h, rgba))
}

fn encode(path: &Path, w: u32, h: u32, rgba: &[u8]) -> anyhow::Result<()> {
    let file = std::io::BufWriter::new(std::fs::File::create(path)?);
    let mut enc = png::Encoder::new(file, w, h);
    enc.set_color(png::ColorType::Rgba);
    enc.set_depth(png::BitDepth::Eight);
    enc.set_compression(png::Compression::High);
    enc.write_header()?.write_image_data(rgba)?;
    Ok(())
}
