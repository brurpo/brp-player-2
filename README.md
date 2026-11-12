# BRP Player 2

A native video player for Windows, written in Rust with Vulkan: frame-accurate
playback, GPU decoding, deinterlacing, GPU shader effects (mpv/libplacebo style) and
AI effects (upscaling, interpolation, denoising, deblurring, matting, depth,
stabilisation), plus export and batch export.

## Download

Get the portable zip from [Releases](https://github.com/brurpo/brp-player-2/releases),
unzip it anywhere and run `brp-player.exe`. On first start it downloads the AI models
and the x264/x265 encoder (about 1.3 GB) in the background. New versions are offered
inside the app.

**Requirements:** Windows 10/11 x64 and a graphics card with Vulkan 1.3, with an
up-to-date driver:

- **NVIDIA:** GeForce GTX 900 series or newer (e.g. GTX 1060, RTX 2060, RTX 3060,
  RTX 4060 and up).
- **AMD:** Radeon RX 400 series or newer (e.g. RX 580, RX 5700, RX 6600, RX 7600 and
  up).
- **Intel:** Arc graphics cards, or the integrated graphics of 6th-generation Core
  processors or newer.

Playback and shader effects run on all of these. The AI effects (upscaling,
interpolation, denoising, matting...) are much heavier: a recent dedicated card is
recommended, and on integrated graphics they will be slow.

## Building

- Rust (stable), the Vulkan SDK, LLVM (for bindgen).
- FFmpeg 7.1 **LGPL** shared build (BtbN/FFmpeg-Builds `win64-lgpl-shared-7.1`)
  unpacked in `third_party/ffmpeg-lgpl/7.1` (see `.cargo/config.toml`).
- `cargo build --release -p brp-player`
- AI models: `tools/fetch_ai.sh` (converts them into `third_party/models`).
- x264/x265 export: a GPL FFmpeg build's `ffmpeg.exe` in `<exe dir>/ffmpeg-gpl/`, or
  `BRP_FFMPEG_GPL` pointing to one.

Releases are packaged by `tools/release/package.py`.

## License

BRP Player is free for individuals and for companies with less than US$1 million in
annual revenue, including for commercial work. Larger companies need a commercial
license: contact **contact@brpapp.com**. Everything you make with it is yours. It may
be shared and modified, for free and with attribution, but not sold. See
[LICENSE.md](LICENSE.md). Third-party components keep their own
licenses: [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
