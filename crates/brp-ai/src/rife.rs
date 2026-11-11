//! RIFE v4 frame interpolation models (ncnn-format files from
//! TNTwise/rife-ncnn-vulkan, MIT). The network, including its `rife.Warp` layer, runs
//! on our own Vulkan runner: see [`crate::gpu`].

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum RifeModel {
    /// RIFE 4.26: best quality.
    V4_26,
    /// RIFE 4.25 lite: faster.
    V4_25Lite,
}

impl RifeModel {
    pub const ALL: [RifeModel; 2] = [Self::V4_26, Self::V4_25Lite];

    pub fn dir_name(self) -> &'static str {
        match self {
            Self::V4_26 => "rife-v4.26",
            Self::V4_25Lite => "rife-v4.25-lite",
        }
    }

    /// Inputs are padded to a multiple of this.
    pub fn padding(self) -> usize {
        match self {
            Self::V4_26 => 64,
            Self::V4_25Lite => 128,
        }
    }
}

