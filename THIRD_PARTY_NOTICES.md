# Third-party components

BRP Player's own code is under LICENSE.md. These components keep their own
licenses, which take precedence for them.

## Shipped in the app zip

| Component | License | Notes |
|---|---|---|
| FFmpeg 7.1.5 libraries (avcodec, avformat, avfilter, avdevice, avutil, swresample, swscale) | LGPL-3.0-or-later | LGPL build by BtbN/FFmpeg-Builds (`n7.1.5-12-g1fdbca85aa`, win64-lgpl-shared-7.1), dynamically linked and replaceable. Source: https://github.com/FFmpeg/FFmpeg/tree/n7.1.5 , build scripts: https://github.com/BtbN/FFmpeg-Builds . Its license texts are in `licenses/ffmpeg-lgpl/`. |
| `shaders/Builtin_Temporal_Denoiser.glsl` | LGPL-3.0-or-later | Copyright (c) 2022 an3223, (c) 2016 Clément Bœsch |
| `shaders/upscale4x.glsl` | MIT | Copyright (c) 2019-2021 bloc97 (Anime4K) |
| Rust crates | MIT / Apache-2.0 / BSD / Zlib / MPL-2.0 (see `Cargo.lock`) | winit, egui, ash, gpu-allocator, cpal, ureq, zip, sha2 and others |

## Downloaded on first start (Google Drive)

| Component | License | Notes |
|---|---|---|
| FFmpeg 7.1.5 `ffmpeg.exe`, GPL build | GPL-3.0-or-later | A separate program, used only to encode H.264/HEVC with x264/x265 on export. Source: https://github.com/FFmpeg/FFmpeg/tree/n7.1.5 , x264: https://code.videolan.org/videolan/x264 , x265: https://bitbucket.org/multicoreware/x265_git . Build: BtbN/FFmpeg-Builds `n7.1.5-12-g1fdbca85aa` win64-gpl-shared-7.1. |
| RIFE v4.25 lite / v4.26 (interpolation) | MIT | hzwer/ECCV2022-RIFE, ncnn by TNTwise/rife-ncnn-vulkan (c) 2020 nihui |
| Real-ESRGAN x2plus, general-x4v3, animevideov3 | BSD-3-Clause | xinntao/Real-ESRGAN |
| 2xNomosUni_compact_otf_medium | CC-BY-4.0 | Helaman / Philip Hofmann |
| SwinIR-M x4 GAN | Apache-2.0 | JingyunLiang/SwinIR |
| 4xNomos2_hq_dat2, 4xRealWebPhoto_v4_dat2 | CC-BY-4.0 | Philip Hofmann; DAT architecture (zhengchen1999/DAT, Apache-2.0) |
| SEA-RAFT M | BSD-3-Clause | princeton-vl/SEA-RAFT |
| WAFT a1 | BSD-3-Clause | princeton-vl/WAFT |
| NAFNet REDS w64 (deblur) | MIT | megvii-research/NAFNet |
| SCUNet real PSNR (clean-up) | Apache-2.0 | cszn/SCUNet |
| Video Depth Anything Small, Depth Anything V2 Small | Apache-2.0 | DepthAnything (the Base/Large models are CC-BY-NC-4.0 and are not used) |
| BiRefNet lite matting | MIT | ZhengPeng7/BiRefNet |
| NNEDI3 weights (NNEDI3 deinterlacing) | GPL-2.0-or-later | tritical's NNEDI3 (as in znedi3 / FFmpeg's `nnedi`). Data only, downloaded separately into `nnedi3/`; without it the other deinterlacers still work. |
| RobustVideoMatting MobileNetV3 (optional "RVM" matting) | GPL-3.0 | PeterL1n/RobustVideoMatting. Model weights only, loaded as data; downloaded separately. |

All models were converted to ncnn format by this project's tools (`tools/fetch_ai.sh`,
`crates/brp-ai/examples/pth2ncnn.rs`).
