//! Working colour spaces of the effects (DESIGN.md §7a): which space each effect
//! works in, and when two spaces are the same picture for a source.
//!
//! Conversions go through linear light, Rec.2020, 1.0 = SDR white, in one GPU pass
//! (`shaders/space.glsl`, shared by the renderer's chain and the AI worker).

use brp_media::{ColorInfo, Primaries, Transfer};

/// The GLSL of the conversions (`brp_convert(from, to, rgb, transfer, src2020, peak)`).
pub const SPACE_GLSL: &str = include_str!("../shaders/space.glsl");

/// HDR peak assumed when the file does not say, relative to SDR white (the viewer's
/// tone mapping uses the same).
pub const HDR_PEAK: f32 = 1000.0 / 203.0;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Space {
    /// The file's own encoding and primaries (mpv `MAIN`).
    Source,
    /// Linear light, Rec.2020, 1.0 = SDR white (203 nits).
    Linear,
    /// The SDR viewer's picture (tone mapped, gamma), extended invertibly beyond 0..1.
    Display,
    /// PQ code values, Rec.2020.
    Pq,
    /// Linear light in the source primaries (libplacebo `LINEAR`).
    MpvLinear,
    /// libplacebo's sigmoidised linear light (`SIGMOID`).
    Sigmoid,
}

impl Space {
    /// Number used by `space.glsl`.
    pub fn id(self) -> i32 {
        match self {
            Space::Source => 0,
            Space::Linear => 1,
            Space::Display => 2,
            Space::Pq => 3,
            Space::MpvLinear => 4,
            Space::Sigmoid => 5,
        }
    }

    /// Name in `// @brp space <name>`.
    pub fn from_name(s: &str) -> Option<Space> {
        Some(match s.trim().to_ascii_lowercase().as_str() {
            "source" | "main" => Space::Source,
            "linear" => Space::Linear,
            "display" | "sdr" => Space::Display,
            "pq" => Space::Pq,
            "mpv-linear" => Space::MpvLinear,
            "sigmoid" => Space::Sigmoid,
            _ => return None,
        })
    }

    /// The space a pass hooking `stage` works in (`file_space`: the file's
    /// `// @brp space`, for MAIN/RGB). None for stages before RGB (LUMA, CHROMA…)
    /// and the ones we do not run.
    pub fn of_hook(stage: &str, file_space: Option<Space>) -> Option<Space> {
        match stage {
            "MAIN" | "RGB" => Some(file_space.unwrap_or(Space::Source)),
            "LINEAR" => Some(Space::MpvLinear),
            "SIGMOID" => Some(Space::Sigmoid),
            _ => None,
        }
    }

    /// Order of the stages after RGB in mpv (MAIN/RGB, LINEAR, SIGMOID).
    pub fn hook_rank(stage: &str) -> Option<u8> {
        match stage {
            "MAIN" | "RGB" => Some(0),
            "LINEAR" => Some(1),
            "SIGMOID" => Some(2),
            _ => None,
        }
    }
}

/// The source as the conversions need it.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SourceColor {
    /// 0 SDR, 1 PQ, 2 HLG, 3 linear.
    pub transfer: i32,
    pub bt2020: bool,
    /// HDR peak relative to SDR white (1 for SDR and linear sources).
    pub peak: f32,
}

impl SourceColor {
    pub fn of(c: &ColorInfo) -> SourceColor {
        let transfer = match c.transfer {
            Transfer::Sdr => 0,
            Transfer::Pq => 1,
            Transfer::Hlg => 2,
            Transfer::Linear => 3,
        };
        SourceColor { transfer, bt2020: c.primaries == Primaries::Bt2020, peak: if c.is_hdr() { HDR_PEAK } else { 1.0 } }
    }

    /// What a space's numbers mean for this source: (curve, Rec.2020 primaries).
    /// Two spaces with the same description are the same picture.
    fn describe(&self, s: Space) -> (u8, bool) {
        const G24: u8 = 0;
        const PQ: u8 = 1;
        const HLG: u8 = 2;
        const LIN: u8 = 3;
        const G22: u8 = 4;
        const TONE: u8 = 5;
        const SIG: u8 = 6;
        let hdr = matches!(self.transfer, 1 | 2);
        match s {
            Space::Source => ([G24, PQ, HLG, LIN][self.transfer as usize], self.bt2020),
            Space::Linear => (LIN, true),
            Space::Pq => (PQ, true),
            Space::MpvLinear => (LIN, self.bt2020),
            Space::Sigmoid => (SIG, self.bt2020),
            Space::Display if hdr => (TONE, false),
            Space::Display if self.transfer == 3 => (G22, self.bt2020),
            Space::Display => (G24, self.bt2020),
        }
    }

    /// `a` and `b` hold the same numbers for this source (no conversion needed).
    pub fn same(&self, a: Space, b: Space) -> bool {
        a == b || self.describe(a) == self.describe(b)
    }

    /// Push constants of `space.comp` / the tensor kernel: from, to, transfer, src2020.
    pub fn words(&self, from: Space, to: Space) -> [i32; 4] {
        [from.id(), to.id(), self.transfer, self.bt2020 as i32]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use brp_media::Matrix;

    fn color(transfer: Transfer, primaries: Primaries) -> ColorInfo {
        ColorInfo { matrix: Matrix::Bt709, full_range: false, transfer, primaries }
    }

    #[test]
    fn same_spaces() {
        let sdr = SourceColor::of(&color(Transfer::Sdr, Primaries::Bt709));
        assert!(sdr.same(Space::Source, Space::Display));
        assert!(!sdr.same(Space::Source, Space::Linear));
        let pq = SourceColor::of(&color(Transfer::Pq, Primaries::Bt2020));
        assert!(pq.same(Space::Source, Space::Pq));
        assert!(!pq.same(Space::Source, Space::Display));
        assert!(pq.same(Space::Linear, Space::MpvLinear));
        let exr = SourceColor::of(&color(Transfer::Linear, Primaries::Bt709));
        assert!(exr.same(Space::Source, Space::MpvLinear));
        assert!(!exr.same(Space::Source, Space::Linear));
        assert!(!exr.same(Space::Source, Space::Display));
    }
}
