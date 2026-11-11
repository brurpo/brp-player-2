//! OpenEXR sequences (DESIGN.md §7c): the numbered files of a sequence, their layers
//! (Blender's `ViewLayer.Combined.R`…, multi-part files' parts), and one layer decoded
//! into a half-float RGBA picture.
//!
//! Only the chosen layer's channels are converted; the file part that holds them is
//! decompressed block by block on all cores (the `exr` crate), other parts are skipped.

use anyhow::{Context, Result};
use exr::block::reader::ChunksReader;
use exr::meta::attribute::SampleType;
use exr::prelude::f16;
use std::path::{Path, PathBuf};

/// One layer as the picker shows it.
#[derive(Clone, Debug)]
pub struct ExrLayer {
    /// "ViewLayer.Combined", "ViewLayer.Depth", or "" for plain R, G, B, A.
    pub name: String,
    /// Its channels' own names, e.g. "RGBA", "Z".
    pub channels: String,
    part: usize,
    /// Index in the part's channel list for R, G, B, A (one channel: grey).
    map: [Option<usize>; 4],
    /// A depth channel (Z): shown normalised, see `decode`.
    pub depth: bool,
}

impl ExrLayer {
    /// Name for the interface: the layer's name, or "RGBA" for the plain channels.
    pub fn label(&self) -> String {
        if self.name.is_empty() { self.channels.clone() } else { format!("{} ({})", self.name, self.channels) }
    }

    pub fn has_alpha(&self) -> bool {
        self.map[3].is_some()
    }
}

/// A sequence of EXR files (or one file).
#[derive(Clone, Debug)]
pub struct ExrSequence {
    pub files: Vec<PathBuf>,
    /// The number in the first file's name (frame 0), if numbered.
    pub first_number: Option<i64>,
    pub layers: Vec<ExrLayer>,
    /// Display window size and the data window's place in it, per part.
    pub width: u32,
    pub height: u32,
    origin: Vec<(i32, i32)>,
    pub pixel_aspect: f32,
    /// Frames per second from the file (`framesPerSecond`), if it says.
    pub fps: Option<(i32, i32)>,
    pub compression: String,
    /// Some channel is 32-bit float.
    pub float32: bool,
    /// Rec.2020 primaries (from `chromaticities`); otherwise Rec.709.
    pub bt2020: bool,
    /// Near and far of each depth layer, from the first frame decoded (one range for
    /// the whole sequence, so it does not flicker).
    depth_range: std::sync::Arc<parking_lot::Mutex<std::collections::HashMap<usize, (f32, f32)>>>,
}

/// Is `path` something `scan` reads: an `.exr` file, or a folder with some?
pub fn is_exr(path: &str) -> bool {
    let p = Path::new(path);
    if path.contains('%') {
        return false; // a pattern for FFmpeg's image sequence reader
    }
    if p.is_dir() {
        return std::fs::read_dir(p).is_ok_and(|mut it| it.any(|e| e.is_ok_and(|e| is_exr_name(&e.path()))));
    }
    is_exr_name(p)
}

fn is_exr_name(p: &Path) -> bool {
    p.extension().is_some_and(|x| x.eq_ignore_ascii_case("exr"))
}

/// Read the sequence `path` belongs to and the first file's layers.
pub fn scan(path: &str) -> Result<ExrSequence> {
    let (files, first_number) = crate::sequence::files(Path::new(path), is_exr_name)?;
    let meta = exr::meta::MetaData::read_from_file(&files[0], false).with_context(|| format!("reading {}", files[0].display()))?;
    let h0 = meta.headers.first().context("EXR without headers")?;
    let display = h0.shared_attributes.display_window;
    let (width, height) = (display.size.0 as u32, display.size.1 as u32);
    let mut layers = Vec::new();
    let mut origin = Vec::new();
    let mut float32 = false;
    for (pi, h) in meta.headers.iter().enumerate() {
        if h.deep {
            origin.push((0, 0));
            continue;
        }
        origin.push((h.own_attributes.layer_position.0 - display.position.0, h.own_attributes.layer_position.1 - display.position.1));
        let part_name = h.own_attributes.layer_name.as_ref().map(|t| t.to_string()).unwrap_or_default();
        // Channels grouped by the name before the last dot.
        let mut groups: Vec<(String, Vec<(String, usize)>)> = Vec::new();
        for (ci, ch) in h.channels.list.iter().enumerate() {
            float32 |= ch.sample_type == SampleType::F32;
            let full = ch.name.to_string();
            let (g, c) = full.rsplit_once('.').map_or((String::new(), full.clone()), |(g, c)| (g.to_string(), c.to_string()));
            match groups.iter_mut().find(|(n, _)| *n == g) {
                Some((_, v)) => v.push((c, ci)),
                None => groups.push((g, vec![(c, ci)])),
            }
        }
        for (g, chans) in groups {
            let find = |names: &[&str]| chans.iter().find(|(c, _)| names.iter().any(|n| c.eq_ignore_ascii_case(n))).map(|(_, i)| *i);
            let (r, gr, b, a) = (find(&["R", "red"]), find(&["G", "green"]), find(&["B", "blue"]), find(&["A", "alpha"]));
            let map = if r.is_some() || gr.is_some() || b.is_some() {
                [r, gr, b, a]
            } else {
                // Not colour (Z, normals, vectors…): the first channels as R, G, B; one
                // channel as grey.
                let rest: Vec<usize> = chans.iter().map(|(_, i)| *i).filter(|i| Some(*i) != a).collect();
                match rest.len() {
                    0 => [a, a, a, None],
                    1 => [Some(rest[0]), Some(rest[0]), Some(rest[0]), a],
                    _ => [rest.first().copied(), rest.get(1).copied(), rest.get(2).copied(), a],
                }
            };
            let name = match (part_name.is_empty(), g.is_empty()) {
                (true, _) => g,
                (false, true) => part_name.clone(),
                (false, false) => format!("{part_name}.{g}"),
            };
            let colour = r.is_some() || gr.is_some() || b.is_some();
            let channels = if colour {
                [(r, "R"), (gr, "G"), (b, "B"), (a, "A")].iter().filter(|(c, _)| c.is_some()).map(|(_, n)| *n).collect::<String>()
            } else {
                chans.iter().map(|(c, _)| c.as_str()).collect::<Vec<_>>().join(if chans.iter().all(|(c, _)| c.len() == 1) { "" } else { "," })
            };
            // Depth: shown from near (black) to far (white) over the first frame's range.
            let depth = !colour && chans.len() == 1 && ["z", "depth"].contains(&chans[0].0.to_ascii_lowercase().as_str());
            layers.push(ExrLayer { name, channels, part: pi, map, depth });
        }
    }
    anyhow::ensure!(!layers.is_empty(), "no picture layers in {}", files[0].display());
    let fps = h0.shared_attributes.other.iter().chain(h0.own_attributes.other.iter()).find_map(|(k, v)| {
        if k.to_string() != "framesPerSecond" {
            return None;
        }
        match v {
            exr::meta::attribute::AttributeValue::Rational(r) => (r.0 > 0 && r.1 > 0).then_some((r.0, r.1 as i32)),
            _ => None,
        }
    });
    let bt2020 = h0.shared_attributes.chromaticities.is_some_and(|c| (c.red.0 - 0.708).abs() < 0.01 && (c.green.1 - 0.797).abs() < 0.01);
    Ok(ExrSequence {
        files,
        first_number,
        layers,
        width,
        height,
        origin,
        pixel_aspect: h0.shared_attributes.pixel_aspect,
        fps,
        compression: format!("{:?}", h0.compression),
        float32,
        bt2020,
        depth_range: Default::default(),
    })
}

impl ExrSequence {
    /// The layer shown first: the plain RGBA channels, else a "Combined" one (Blender's
    /// beauty pass), else the first with colour.
    pub fn default_layer(&self) -> usize {
        let colour = |l: &ExrLayer| l.map[0] != l.map[1];
        self.layers
            .iter()
            .position(|l| l.name.is_empty() && colour(l))
            .or_else(|| self.layers.iter().position(|l| l.name.to_ascii_lowercase().ends_with("combined") && colour(l)))
            .or_else(|| self.layers.iter().position(colour))
            .unwrap_or(0)
    }

    /// The layer called `name` (as `ExrLayer::name`).
    pub fn layer_named(&self, name: &str) -> Option<usize> {
        self.layers.iter().position(|l| l.name == name)
    }

    /// File `i`'s layer `layer` as half-float RGBA (alpha 1 when the layer has none),
    /// colour un-premultiplied (EXR stores it premultiplied).
    pub fn decode(&self, i: usize, layer: usize) -> Result<Vec<f16>> {
        let l = &self.layers[layer];
        let path = &self.files[i];
        let (w, h) = (self.width as usize, self.height as usize);
        let mut px = vec![0f32; w * h * 4];
        if l.map[3].is_none() {
            px.chunks_exact_mut(4).for_each(|p| p[3] = 1.0);
        }
        let (ox, oy) = self.origin[l.part];
        let file = std::io::BufReader::new(std::fs::File::open(path).with_context(|| format!("opening {}", path.display()))?);
        let reader = exr::block::read(file, false)?;
        let part = l.part;
        let headers = reader.headers().to_vec();
        anyhow::ensure!(headers.len() > part, "{}: no part {part}", path.display());
        let types: Vec<SampleType> = headers[part].channels.list.iter().map(|c| c.sample_type).collect();
        // Which output slots each channel of the part fills.
        let mut slots: Vec<Vec<usize>> = vec![Vec::new(); types.len()];
        for (s, c) in l.map.iter().enumerate() {
            if let Some(c) = c {
                if *c < slots.len() {
                    slots[*c].push(s);
                }
            }
        }
        let mut row = Vec::<f32>::new();
        reader
            .filter_chunks(false, |_, _, block| block.layer == part && block.level == exr::math::Vec2(0, 0))?
            .decompress_parallel(false, |meta, block| {
                for line in block.lines(&meta.headers[part].channels) {
                    let loc = line.location;
                    let targets = &slots[loc.channel];
                    if targets.is_empty() {
                        continue;
                    }
                    let y = loc.position.1 as i32 + oy;
                    if y < 0 || y >= h as i32 {
                        continue;
                    }
                    row.clear();
                    match types[loc.channel] {
                        SampleType::F16 => row.extend(line.read_samples::<f16>().map(|v| v.map_or(0.0, f32::from))),
                        SampleType::F32 => row.extend(line.read_samples::<f32>().map(|v| v.unwrap_or(0.0))),
                        SampleType::U32 => row.extend(line.read_samples::<u32>().map(|v| v.unwrap_or(0) as f32)),
                    }
                    let base = y as usize * w;
                    for (k, &v) in row.iter().enumerate() {
                        let x = loc.position.0 as i32 + ox + k as i32;
                        if x < 0 || x >= w as i32 {
                            continue;
                        }
                        for &s in targets {
                            px[(base + x as usize) * 4 + s] = v;
                        }
                    }
                }
                Ok(())
            })?;
        // Depth: near..far of the first frame read to 0..1 (the background, at 1e10 in
        // Blender's files, beyond far: white).
        if l.depth {
            let range = *self.depth_range.lock().entry(layer).or_insert_with(|| {
                let (mut lo, mut hi) = (f32::MAX, f32::MIN);
                for p in px.chunks_exact(4) {
                    let z = p[0];
                    if z.is_finite() && z < 1e9 {
                        lo = lo.min(z);
                        hi = hi.max(z);
                    }
                }
                if lo < hi { (lo, hi) } else { (0.0, 1.0) }
            });
            let k = 1.0 / (range.1 - range.0);
            for p in px.chunks_exact_mut(4) {
                let v = ((p[0] - range.0) * k).clamp(0.0, 1.0);
                p[0] = v;
                p[1] = v;
                p[2] = v;
            }
        }
        // Half float out (within its range: no infinities), colour divided by alpha (on
        // all cores: ~2M pixels).
        let to_half = |v: f32| f16::from_f32(if v.is_nan() { 0.0 } else { v.clamp(-65504.0, 65504.0) });
        let mut out = vec![f16::ZERO; w * h * 4];
        let alpha = l.has_alpha();
        let threads = std::thread::available_parallelism().map_or(4, |n| n.get()).min(16);
        let per = (w * h).div_ceil(threads).max(1) * 4;
        std::thread::scope(|sc| {
            for (src, dst) in px.chunks(per).zip(out.chunks_mut(per)) {
                sc.spawn(move || {
                    for (p, o) in src.chunks_exact(4).zip(dst.chunks_exact_mut(4)) {
                        let a = p[3];
                        let k = if alpha && a > 1e-6 && a < 1.0 { 1.0 / a } else { 1.0 };
                        for c in 0..3 {
                            o[c] = to_half(p[c] * k);
                        }
                        o[3] = to_half(a);
                    }
                });
            }
        });
        Ok(out)
    }
}

