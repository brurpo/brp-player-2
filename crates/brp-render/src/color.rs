//! Builds the video shader's push constants from frame layout and color metadata.

use brp_media::{ColorInfo, FrameLayout, Matrix, Primaries, Transfer};

#[repr(C)]
#[derive(Clone, Copy, Default, Debug)]
pub struct VideoPush {
    pub rect: [f32; 4],
    pub m: [[f32; 4]; 3],
    pub off: [f32; 4],
    pub misc: [f32; 4],
    pub mode: [i32; 4],
    pub swz: [i32; 4],
}

const _: () = assert!(std::mem::size_of::<VideoPush>() == 128);

impl VideoPush {
    pub fn bytes(&self) -> &[u8] {
        unsafe { std::slice::from_raw_parts(self as *const Self as *const u8, std::mem::size_of::<Self>()) }
    }
}

fn kr_kb(m: Matrix) -> (f32, f32) {
    match m {
        Matrix::Bt601 => (0.299, 0.114),
        Matrix::Bt709 | Matrix::Rgb => (0.2126, 0.0722),
        Matrix::Bt2020Ncl => (0.2627, 0.0593),
        Matrix::Smpte240 => (0.212, 0.087),
    }
}

/// YUV (U, V centred on 0) to RGB matrix rows.
fn yuv_matrix(m: Matrix) -> [[f32; 4]; 3] {
    let (kr, kb) = kr_kb(m);
    let kg = 1.0 - kr - kb;
    [
        [1.0, 0.0, 2.0 * (1.0 - kr), 0.0],
        [1.0, -2.0 * kb * (1.0 - kb) / kg, -2.0 * kr * (1.0 - kr) / kg, 0.0],
        [1.0, 2.0 * (1.0 - kb), 0.0, 0.0],
    ]
}

/// HDR peak brightness assumed when the file does not say (nits).
const DEFAULT_PEAK_NITS: f32 = 1000.0;

pub fn video_push(layout: FrameLayout, color: &ColorInfo, checker_px: f32) -> VideoPush {
    let mut p = VideoPush::default();
    let transfer = match color.transfer {
        Transfer::Sdr => 0,
        Transfer::Pq => 1,
        Transfer::Hlg => 2,
        Transfer::Linear => 3,
    };
    p.mode = [0, transfer, layout.has_alpha() as i32, (color.primaries == Primaries::Bt2020) as i32];
    p.misc = [1.0, DEFAULT_PEAK_NITS / 203.0, checker_px, 0.0];
    match layout {
        FrameLayout::Yuv { bits, .. } => {
            let max = ((1u32 << bits) - 1) as f32;
            let unit = (1u32 << (bits.max(8) - 8)) as f32;
            p.misc[0] = if bits > 8 { 65535.0 / max } else { 1.0 };
            p.m = yuv_matrix(color.matrix);
            p.off = if color.full_range {
                [0.0, (1u32 << (bits - 1)) as f32 / max, 1.0, 1.0]
            } else {
                [16.0 * unit / max, 128.0 * unit / max, max / (219.0 * unit), max / (224.0 * unit)]
            };
            p.swz = [0, 1, 2, -1];
        }
        FrameLayout::Rgba8 { offsets, alpha } => {
            p.mode[0] = 1;
            p.swz = [offsets[0] as i32, offsets[1] as i32, offsets[2] as i32, alpha.map_or(-1, |a| a as i32)];
        }
        FrameLayout::Rgba16 => {
            p.mode[0] = 1;
            p.swz = [0, 1, 2, 3];
        }
    }
    p
}
