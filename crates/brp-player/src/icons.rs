//! UI icons (embedded PNGs from /icons), decoded once at startup.

use egui::{Color32, ColorImage, TextureHandle, TextureOptions};
use std::collections::HashMap;

/// Size icons are stored at (they are drawn at ~18–24 points).
const SIZE: usize = 64;

macro_rules! icons {
    ($($name:literal),* $(,)?) => {
        &[$(($name, include_bytes!(concat!("../assets/icons/", $name, ".png")) as &[u8])),*]
    };
}

const ICONS: &[(&str, &[u8])] = icons![
    "play", "pause", "step_back", "step_foward", "toStart", "toEnd", "repeat_icon", "clear",
    "originalSize", "restoreZoom", "fullscreen_icon", "folder_expanded", "config", "video",
    "fx_enabled", "audio", "muted",
];

pub const APP_ICON: &[u8] = include_bytes!("../assets/icons/app_icon.png");

pub struct Icons {
    map: HashMap<&'static str, TextureHandle>,
}

impl Icons {
    pub fn load(ctx: &egui::Context) -> Self {
        let mut map = HashMap::new();
        for (name, bytes) in ICONS {
            match decode(bytes) {
                Ok((w, h, rgba)) => {
                    let rgba = key_out_tile(rgba);
                    let (sw, sh, small) = downscale(w, h, &rgba, SIZE);
                    let img = ColorImage::from_rgba_unmultiplied([sw, sh], &small);
                    map.insert(*name, ctx.load_texture(*name, img, TextureOptions::LINEAR));
                }
                Err(e) => log::warn!("icon {name}: {e}"),
            }
        }
        Icons { map }
    }

    pub fn get(&self, name: &str) -> Option<&TextureHandle> {
        self.map.get(name)
    }
}

/// RGBA8 pixels of a PNG.
pub fn decode(bytes: &[u8]) -> anyhow::Result<(usize, usize, Vec<u8>)> {
    let mut dec = png::Decoder::new(std::io::Cursor::new(bytes));
    dec.set_transformations(png::Transformations::EXPAND | png::Transformations::STRIP_16);
    let mut reader = dec.read_info()?;
    let mut buf = vec![0; reader.output_buffer_size().unwrap_or(0)];
    let info = reader.next_frame(&mut buf)?;
    let (w, h) = (info.width as usize, info.height as usize);
    let rgba = match info.color_type {
        png::ColorType::Rgba => buf[..w * h * 4].to_vec(),
        png::ColorType::Rgb => buf[..w * h * 3].chunks(3).flat_map(|p| [p[0], p[1], p[2], 255]).collect(),
        png::ColorType::GrayscaleAlpha => buf[..w * h * 2].chunks(2).flat_map(|p| [p[0], p[0], p[0], p[1]]).collect(),
        png::ColorType::Grayscale => buf[..w * h].iter().flat_map(|&g| [g, g, g, 255]).collect(),
        _ => anyhow::bail!("unsupported png color type"),
    };
    Ok((w, h, rgba))
}

/// The icon files fill their background with light grey at alpha 5/255 (about 2%).
/// Invisible over a checkerboard, but it shows as a lighter square on dark UI.
/// Treat near-zero alpha as fully transparent and rescale the rest, so
/// anti-aliased edges stay smooth.
fn key_out_tile(mut rgba: Vec<u8>) -> Vec<u8> {
    const FLOOR: u32 = 12;
    for p in rgba.chunks_mut(4) {
        let a = p[3] as u32;
        p[3] = if a <= FLOOR { 0 } else { ((a - FLOOR) * 255 / (255 - FLOOR)) as u8 };
    }
    rgba
}

/// Area-average downscale to fit `size` (box filter; icons are square).
fn downscale(w: usize, h: usize, rgba: &[u8], size: usize) -> (usize, usize, Vec<u8>) {
    if w <= size && h <= size {
        return (w, h, rgba.to_vec());
    }
    let (sw, sh) = (size, (h * size / w).max(1));
    let mut out = vec![0u8; sw * sh * 4];
    for y in 0..sh {
        for x in 0..sw {
            let (x0, x1) = (x * w / sw, ((x + 1) * w / sw).max(x * w / sw + 1));
            let (y0, y1) = (y * h / sh, ((y + 1) * h / sh).max(y * h / sh + 1));
            // Premultiplied average so transparent pixels don't darken edges.
            let mut acc = [0u32; 4];
            for yy in y0..y1 {
                for xx in x0..x1 {
                    let p = &rgba[(yy * w + xx) * 4..][..4];
                    let a = p[3] as u32;
                    acc[0] += p[0] as u32 * a;
                    acc[1] += p[1] as u32 * a;
                    acc[2] += p[2] as u32 * a;
                    acc[3] += a;
                }
            }
            let n = ((x1 - x0) * (y1 - y0)) as u32;
            let o = &mut out[(y * sw + x) * 4..][..4];
            if acc[3] > 0 {
                o[0] = (acc[0] / acc[3]) as u8;
                o[1] = (acc[1] / acc[3]) as u8;
                o[2] = (acc[2] / acc[3]) as u8;
            }
            o[3] = (acc[3] / n) as u8;
        }
    }
    (sw, sh, out)
}

/// Window icon for winit (RGBA, 64x64).
pub fn window_icon() -> Option<winit::window::Icon> {
    let (w, h, rgba) = decode(APP_ICON).ok()?;
    let rgba = key_out_tile(rgba);
    let (sw, sh, small) = downscale(w, h, &rgba, 64);
    winit::window::Icon::from_rgba(small, sw as u32, sh as u32).ok()
}

/// A padlock drawn in the icon style (the icon set has none).
pub fn paint_lock(painter: &egui::Painter, rect: egui::Rect, locked: bool, color: Color32) {
    let c = rect.center();
    let s = rect.height().min(rect.width()) * 0.5;
    let body = egui::Rect::from_center_size(c + egui::vec2(0.0, s * 0.28), egui::vec2(s * 1.25, s * 0.95));
    painter.rect_filled(body, s * 0.18, color);
    // Shackle: an arch above the body; shifted up and to the side when unlocked.
    let r = s * 0.38;
    let lift = if locked { 0.0 } else { s * 0.28 };
    let cx = c.x + if locked { 0.0 } else { s * 0.3 };
    let top = body.top() - s * 0.42 - lift;
    let stroke = egui::Stroke::new(s * 0.2, color);
    let n = 12;
    let pts: Vec<egui::Pos2> = (0..=n)
        .map(|i| {
            let a = std::f32::consts::PI * (i as f32 / n as f32);
            egui::pos2(cx - r * a.cos(), top - r * a.sin() + r)
        })
        .collect();
    painter.add(egui::Shape::line(pts, stroke));
    let left_end = if locked { body.top() } else { top + r + s * 0.12 };
    painter.line_segment([egui::pos2(cx - r, top + r), egui::pos2(cx - r, left_end)], stroke);
    painter.line_segment([egui::pos2(cx + r, top + r), egui::pos2(cx + r, body.top())], stroke);
    // Keyhole.
    painter.circle_filled(body.center() - egui::vec2(0.0, s * 0.06), s * 0.12, Color32::from_gray(25));
}
