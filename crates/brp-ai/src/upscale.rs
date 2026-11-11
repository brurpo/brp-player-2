//! Real-ESRGAN super-resolution models (ncnn-format files from xinntao/Real-ESRGAN,
//! BSD-3-Clause). The network runs on our own Vulkan runner: see [`crate::gpu`].

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum UpscaleModel {
    /// realesrgan-x4plus: general photos and video, best quality, slowest.
    General4x,
    /// realesr-animevideov3: small, fast; made for animation but fine on video.
    Fast2x,
    Fast3x,
    Fast4x,
    /// realesrgan-x4plus-anime: illustrations and anime.
    Anime4x,
    /// realesrgan-x2plus: like General4x at 2x; runs at half resolution inside, so
    /// about 4x faster than General4x.
    General2x,
    /// realesr-general-x4v3: small and fast like Fast4x, but trained on real-world
    /// pictures; its denoising strength is set by mixing it with
    /// realesr-general-wdn-x4v3 (the same network trained without denoising).
    GeneralV3,
    /// 2xNomosUni_compact_otf_medium (Helaman, CC-BY-4.0): the compact network at 2x,
    /// trained on real-world pictures with realistic degradations (OpenModelDB).
    Natural2x,
    /// SwinIR-M real-world x4 GAN (JingyunLiang/SwinIR, Apache-2.0): a transformer
    /// (window attention), the finest detail, slowest. Runs on the same runner
    /// (ChannelNorm, WindowAttention, GELU layers); converted by `pth2ncnn swinir`.
    Swin4x,
    /// 4xNomos2_hq_dat2 (Philip Hofmann, CC-BY-4.0; DAT, Chen et al. 2023): a
    /// transformer alternating window attention (8x32 and 32x8 windows) with attention
    /// between channels, trained on clean (non-degraded) pictures. Finer than Swin4x on
    /// clean footage, about 1.6x its time. Converted by `pth2ncnn dat`.
    Dat4x,
    /// 4xRealWebPhoto_v4_dat2 (Philip Hofmann, CC-BY-4.0): the same DAT network trained
    /// on degraded pictures (blur, noise, JPEG/WebP compression, rescaling), so it
    /// restores real footage instead of keeping its flaws. Converted by `pth2ncnn dat`
    /// from the .safetensors (the ONNX export lost the weight names).
    DatReal4x,
}

impl UpscaleModel {
    pub const ALL: [UpscaleModel; 11] = [Self::General4x, Self::General2x, Self::Fast2x, Self::Fast3x, Self::Fast4x, Self::Anime4x, Self::GeneralV3, Self::Natural2x, Self::Swin4x, Self::Dat4x, Self::DatReal4x];

    /// Folder under the models directory.
    pub fn dir(self) -> &'static str {
        match self {
            Self::Swin4x => "swinir",
            Self::Dat4x | Self::DatReal4x => "dat",
            _ => "realesrgan",
        }
    }

    /// Network input sides must be a multiple of this (SwinIR: its 8 px windows; DAT:
    /// its 32 px ones).
    pub fn align(self) -> usize {
        match self {
            Self::Swin4x => 8,
            Self::Dat4x | Self::DatReal4x => 32,
            _ => 2,
        }
    }

    pub fn file_stem(self) -> &'static str {
        match self {
            Self::General4x => "realesrgan-x4plus",
            Self::Fast2x => "realesr-animevideov3-x2",
            Self::Fast3x => "realesr-animevideov3-x3",
            Self::Fast4x => "realesr-animevideov3-x4",
            Self::Anime4x => "realesrgan-x4plus-anime",
            Self::General2x => "realesrgan-x2plus",
            Self::GeneralV3 => "realesr-general-x4v3",
            Self::Natural2x => "2xNomosUni_compact_otf_medium",
            Self::Swin4x => "SwinIR-M_x4_GAN",
            Self::Dat4x => "4xNomos2_hq_dat2",
            Self::DatReal4x => "4xRealWebPhoto_v4_dat2",
        }
    }

    pub fn scale(self) -> usize {
        match self {
            Self::Fast2x | Self::General2x | Self::Natural2x => 2,
            Self::Fast3x => 3,
            _ => 4,
        }
    }

    /// The network sees the picture folded 2x2 into channels (pixel_unshuffle).
    pub fn folded(self) -> bool {
        self == Self::General2x
    }
}

