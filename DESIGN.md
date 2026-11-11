# BRP Player 2 — Design

Rewrite of the Python/PyQt6/mpv player in `old/` as a native app.

## 1. Requirements (agreed)

| Requirement | Decision |
|---|---|
| No Python | Rust app, native libraries only |
| Extremely performant, efficient | Own decode + GPU render engine, zero-copy where possible |
| Exact frame seek, including long-GOP codecs (H.264/HEVC/AV1…) | Frame index + sync-point seek + decoded-frame cache (§4) |
| Existing shaders work | Own implementation of the mpv/libplacebo user-shader format (§5) |
| **Temporal effects must work** (they did not in the old app) | Deterministic temporal model (§6) |
| All old features | Parity checklist (§10) |
| Fast startup, no resident server | Target window <150 ms, first frame <300 ms (§8) |
| Platforms | Windows, macOS, Linux |
| **Multi-vendor GPU** (NVIDIA, AMD, Intel, Apple) | Vulkan everywhere, no CUDA/NVDEC-only paths, vendor features are optional extras (§3.1) |
| Second file while open | Per-window **Lock** toggle (§8) |
| UI language | English + Portuguese (pt-BR), switchable at runtime |
| Out of scope | `keyer.py`, `project_manager.py`, `main_window.py`, `playback_logic.py`, `player_thread.py` (unused) |

## 2. Why temporal effects failed in the old app

From reading the code (likely causes, all structural to the mpv approach):

1. **Shaders ran on every redraw, not once per frame.** Pausing, resizing, zooming or OSD updates re-ran the chain, so `STORAGE` history textures (`PREV1`, `PREV2`, `LastFrameTex`) were overwritten with the *current* frame.
2. **History was never rebuilt after a seek.** After a seek, frame step back or loop, the stored "previous frame" came from a different position.
3. **Fixed-size storage.** `temporalNL.glsl` hardcodes `3840x3840 r16f` and `MB.glsl` `3840x2160`; other resolutions index wrongly.
4. **FFmpeg temporal filters (`atadenoise`, `hqdn3d`, `fftdnoiz`, `deshake`) were flushed on every seek** by mpv and only get past frames, so stepping and scrubbing showed un-filtered or wrong frames.
5. **Export used a different pipeline** (ffmpeg CLI + `libplacebo` filter, with shader params rewritten as defaults), so preview and export did not match.

The fix is to make every rendered frame a pure function of *(frame index, effect settings)*, which needs control of decoding. mpv cannot provide that, so we own the pipeline.

## 3. Architecture

```
            ┌──────────────── UI thread (winit + egui) ────────────────┐
            │ windows, timeline, effects panel, export dialogs, i18n  │
            └───────────────┬─────────────────────────────┬────────────┘
                            │ commands                    │ frames to present
┌──────────── Media engine ─▼───────────┐   ┌──────────── Render engine (Vulkan) ──────────┐
│ Demux + index (libavformat)           │   │ Upload / hw-frame import (zero-copy)         │
│ Decode workers (libavcodec, HW/SW)    │──▶│ Color: YUV→RGB, HDR, linear, OCIO (GPU)      │
│ Image sequences / EXR (exr crate)     │   │ Effect graph: user shaders + builtins        │
│ Frame cache (VRAM + RAM, LRU by GOP)  │   │ Temporal history manager (§6)                │
│ Audio decode → cpal (clock master)    │   │ Viewer: zoom/pan/1:1, scopes, output to swap │
└───────────────────────────────────────┘   └──────────────────────────────────────────────┘
                            │                                │
                            └──── Export: same engine offscreen → libavcodec encode ─┘
```

### 3.1 GPU and vendor neutrality

- **Vulkan 1.3** via `ash` on all platforms (MoltenVK on macOS). No CUDA, no NVAPI, no vendor SDKs required.
- **Hardware decode** through FFmpeg hwcontexts, picked at runtime, all vendor-neutral:
  - Windows: D3D11VA (NVIDIA/AMD/Intel), Vulkan Video where supported.
  - Linux: Vulkan Video, VAAPI (AMD/Intel), with Vulkan import.
  - macOS: VideoToolbox.
  - Always falls back to multithreaded software decode.
- **Hardware encode** is optional: NVENC, AMF, QSV, VideoToolbox, VAAPI, Vulkan encode are offered only if probing succeeds on the actual device. Software encoders always available.
- **GPU selection** lists all Vulkan devices (discrete preferred). Replaces the old "GPU API" combo: no OpenGL/D3D11 renderer switch needed.
- CI and test matrix must include at least one AMD, one Intel and one NVIDIA GPU (plus Apple Silicon).

### 3.2 Main crates / libraries

| Area | Choice |
|---|---|
| Language | Rust (stable) |
| Window/input | `winit` (multi-window) |
| UI | `egui` with custom theme, rendered by our own Vulkan pass into the same swapchain |
| GPU | `ash`, `gpu-allocator`; GLSL → SPIR-V with `shaderc` |
| Decode/encode | FFmpeg 7.x libraries (`rsmpeg`/`ffmpeg-sys`) built via vcpkg, shipped with the app (no `ffmpeg.exe`/`ffprobe` in PATH) |
| EXR | `exr` crate (layers read directly, no pre-render to temp files) |
| Color | OpenColorIO 2.x through a small C++ shim (`cxx`), GPU shader path |
| Audio | `cpal` (WASAPI / CoreAudio / ALSA-Pulse-PipeWire), `libswresample`, `atempo` for speed changes |
| Dialogs | `rfd` |
| Single instance IPC | `interprocess` local sockets (named pipe / Unix socket) |
| i18n | `fluent` (`en`, `pt-BR` files) |
| Settings/presets | `serde` JSON in the OS config dir |
| Logging | `tracing` → in-app log window + rotating file |

## 4. Exact seeking engine

1. **Index on open.** Build a frame table from the container index (MP4/MOV `stts`/`stss`/`ctts`: instant; MKV cues; MXF index). Formats without a usable index get a background packet scan (no decoding). Each entry: presentation index → pts, dts, keyframe flag, packet position.
2. **Frame numbers are table indices**, never `time × fps`. Works for VFR, drop-frame and odd timebases. Timecode derived from the table and container timecode track.
3. **Seek to frame N:** find the sync point that can produce N (latest keyframe in decode order; one GOP earlier for open-GOP streams), decode forward, select the frame whose pts equals `table[N].pts`. Exact by construction.
4. **Decoded frame cache:** GPU memory budget (configurable, default by VRAM size) with RAM spill, LRU by GOP. Whole GOPs are kept after decode, so:
   - step back / reverse play inside a cached GOP is instant;
   - a background worker prefetches the next GOP ahead and the previous GOP behind the playhead.
5. **Scrubbing:** show the nearest cached or keyframe immediately, then refine to the exact frame (exact frame always wins when the drag stops).
6. **Intra codecs and sequences** (ProRes, DNxHR, PNG/EXR/TIFF…): several decoder workers on different frames in parallel.
7. **Validation:** test that random seeks return the same frame hash as sequential decode, for H.264/HEVC (closed and open GOP, B-pyramids), AV1, VP9 with alpha, ProRes, DNxHR, MXF, interlaced material, and image sequences.

## 5. Shader compatibility (mpv/libplacebo user-shader format)

We implement the format ourselves so all files in `old/shaders/` run unchanged.

- **Directives:** `HOOK` (`MAIN`, `LUMA`, `CHROMA`, `RGB`, `NATIVE`, `OUTPUT`…), `BIND`, `SAVE`, `DESC`, `WIDTH`/`HEIGHT`/`WHEN` (RPN expressions), `COMPONENTS`, `COMPUTE`, `TEXTURE` (`SIZE`, `FORMAT`, `FILTER`, `BORDER`, `STORAGE`, inline data), `PARAM` (`TYPE` float/int/uint/`DEFINE`, `MINIMUM`, `MAXIMUM`, `DESC`).
- **GLSL environment:** generate the same preamble mpv provides (`HOOKED_pos/_tex/_texOff/_size/_pt`, `<NAME>_*` for each bound texture, `frame`, `random`, `input_size`, `target_size`, `tex_offset`, `linearize`/`delinearize`), compile to SPIR-V, cache by content hash.
- **Pipeline stages** match mpv's order (native → luma/chroma → RGB → linear → main → scale → output) so `LUMA`/`CHROMA` hooks (FSRCNNX, nlmeans) behave the same.
- **Params UI** keeps the old conventions: `xxxR/xxxG/xxxB` triplets become a color picker; `(cat: Name)` in `DESC` groups params; uint → checkbox.
- **Param changes are live uniforms** (no file rewrite, no temp copies). Export uses the same values.
- **Validation:** golden-frame tests comparing our output against mpv `--vo=gpu-next` for each shader on fixed frames (within a small tolerance).

## 6. Temporal model (the key fix)

Rule: **the effect chain runs once per (frame index, settings).** Redraws (pause, resize, zoom, UI) only re-composite the cached result.

Two ways a shader can see time:

1. **Explicit frame inputs (new, preferred).** Extension directives bind neighbour frames from the frame cache:
   `//!BIND SRC_PREV1` … `SRC_PREVn`, `SRC_NEXT1` … `SRC_NEXTn` (decoded source frames) and `OUT_PREV1` (processed output of the previous frame, for recursive filters). The engine guarantees these are the right frames for any seek, step or direction.
2. **Legacy `STORAGE` history (existing shaders: `temporalNL`, `Builtin_Temporal_Denoiser`, `MB`).**
   - Storage textures are resized to the actual frame size (ignoring hardcoded `SIZE`).
   - After any discontinuity (seek, step back, loop, param change, file change) storage is cleared and the chain is **silently replayed over a warm-up window** N−w … N−1 before showing N. `w` defaults per shader (2 for 2-frame history, 8 for recursive) and can be set with `//!TEMPORAL_WARMUP w`.
   - During forward playback the chain simply continues; no extra cost.
   - Reverse playback / scrubbing: processed outputs are cached alongside decoded frames, so the warm-up cost is paid once per frame.

**Builtin temporal filters** (`atadenoise`, `hqdn3d`, `fftdnoiz`, `deshake`) and **deinterlacers** (`bwdif`/`yadif`, field and frame modes):
- Phase 1: run in libavfilter fed with an explicit window of frames around N from the cache (deterministic, exact).
- Phase 2: GPU compute ports using explicit frame inputs (`atadenoise`, `hqdn3d`, `bwdif`, `yadif`, deband first; `fftdnoiz`/`deshake` later if worth it).

Preview and export share this code, so exported temporal effects match the viewer frame for frame.

## 6b. Deinterlacing (automatic, with neural NNEDI)

**Detection (automatic):**
1. Container/codec field order (`AVCodecParameters.field_order`) and per-frame flags (`AV_FRAME_FLAG_INTERLACED`, top-field-first). Already read (Info panel shows Interlaced/Progressive).
2. **Content analysis**, because flags are often wrong (progressive material flagged interlaced and the reverse): a GPU "idet"-style pass over a few frames spread through the file. It measures combing between the two fields vs. within each field, and gives TFF / BFF / progressive with a confidence. It runs in the background after open, using frames already in the cache.
3. Auto mode deinterlaces only when (1)+(2) agree it's interlaced; the Info panel shows what was detected and why.

**Methods** (all GPU compute, all vendors):
- **NNEDI3** (neural edge-directed interpolation): rebuilds the missing field lines with the NNEDI3 neural network (predictor network + prescreener, the standard 16x6 / 32x4 … neuron sets, `nnedi3_weights.bin`). Default for quality.
- **NNEDI3 + motion-adaptive temporal** (QTGMC-like "best" mode): static areas are woven from the neighbouring fields (no detail loss); moving areas use NNEDI3. The decision uses the bwdif-style temporal check on fields N−1, N, N+1.
- **bwdif** and **yadif** (fast), as before.
- Output: **field rate** ("bob": 50i → 50p, smooth motion; default) or **frame rate** (50i → 25p).

**Exactness:** deinterlacing frame N needs neighbouring fields, so it uses the explicit frame inputs of the temporal model (§6). After any seek or step the result is identical to sequential playback, and export matches the viewer. In field-rate mode the timeline steps by field (frame N, field 1/2), and frame numbers stay those of the source.

**Licensing note:** the NNEDI3 weights come from tritical's NNEDI3 (GPL-family ports such as znedi3/ffmpeg `nnedi`). Check the weights' licence before shipping; it ties in with the GPL decision in §12.

## 7. Color

- Correct YUV→RGB per stream metadata (matrix, range, primaries, transfer), chroma siting.
- HDR (PQ/HLG) tone-mapped for SDR displays; HDR output where the OS/display supports it.
- Linear EXR handled in float end to end (no 8/16-bit intermediate).
- **OCIO on the GPU:** use OCIO's GPU shader generator (exact transform + its own LUT textures) instead of baking a 64³ `.cube`. Input/output color space pickers as before, plus display/view.
- Effect chain runs in 16-bit float.

### 7a. Working colour space for effects

**No effects (or the effects switch off): nothing changes.** The decoded planes go to the window through one shader (`video.frag`: YUV→RGB, transfer, tone mapping), and no float working picture exists. Export packs from the same direct conversion (`pack.rs`).

**With effects**, each effect says which colour space it works in, and the chain converts at its edges:

| Space | What the effect sees | Who uses it |
|---|---|---|
| `source` | RGB in the file's own encoding and primaries (SDR gamma, PQ, HLG, linear EXR): mpv's `MAIN`/`RGB` | every mpv/libplacebo `MAIN`/`RGB` shader (unchanged behaviour), `LUMA`/`CHROMA` hooks (they run on the planes, before RGB) |
| `linear` | linear light, **Rec.2020** primaries, 1.0 = SDR white (203 nits), no upper limit | effects that need light to add up (the planned learned denoiser, `tools/denoise_train/PLAN.md`), shaders with `// @brp space linear` |
| `display` | the picture as an SDR screen shows it, 0–1, gamma encoded; extended invertibly above 1 and below 0 | the AI models (trained on display pictures); shaders with `// @brp space display` |
| `pq` | PQ (ST 2084) code values, Rec.2020 | effects made for HDR code values (`// @brp space pq`) |
| `mpv-linear` | linear light in the source primaries (libplacebo's `LINEAR` stage) | shaders hooking `LINEAR` |
| `sigmoid` | libplacebo's sigmoidised linear light (centre 0.75, slope 6.5), over all the light the source can hold (PQ: 10,000 nits; HLG: its 1,000-nit peak; SDR: 0–1) | shaders hooking `SIGMOID` |

The hub of every conversion is linear Rec.2020: a conversion is "space A → linear → space B" in one compute pass (`shaders/space.glsl`, shared by the renderer's chain and the AI worker's tensor kernel). Spaces that encode the same thing for a source are recognised and never converted: for an SDR Rec.709 file `display` *is* `source`, for a PQ Rec.2020 file `pq` *is* `source`, for a linear EXR `mpv-linear` *is* `source`.

**Lazy conversion.** The chain keeps the space the picture is currently in, starting at `source`. Before an effect, it converts only if the effect's space differs; after it, the picture stays in that effect's space. So a stack of `source` (mpv) or `display` effects on SDR never converts at all, the first `linear` effect converts to linear and following `linear` effects share that conversion, and a `display` effect after a `linear` one converts once each way.

**The end of the chain** converts back to the source's encoding (only if the picture is not in it already), and the display shader then applies the one display transform it always applies: tone mapping (HDR on an SDR screen), gamma, alpha over the checkerboard. The source's encoding keeps HDR whole in float16: PQ reaches 10,000 nits, HLG and linear are unbounded, and SDR gamma is extended above 1 (sign-preserving powers). Ending in the source's encoding is what keeps "no effects", "effects that change nothing" and export identical: the display shader and the export packer stay exactly as they are and see the same kind of picture in both cases. On an HDR screen the same display shader writes light instead of a tone-mapped picture (§7b).

**The invertible mapping for HDR through SDR-trained models (`display`).** From linear Rec.2020 (L, 1 = 203 nits):
1. Rec.2020 → the display's primaries (Rec.709 for HDR sources, as the viewer does; SDR files keep their own primaries, so `display` = `source` for them). Out-of-gamut colours become negative channels and are kept, not clipped.
2. Tone curve on the largest channel `m`, scaling the colour by `t(m)/m` (hue kept): extended Reinhard `t(m) = m (1 + m/P²) / (1 + m)` with `P` = HDR peak relative to SDR white (1000/203, the viewer's), so `t(P) = 1`. It is the viewer's tone map, so SDR-trained models see the picture exactly as the SDR viewer shows it. `t` rises for all `m ≥ 0`, so it inverts exactly: `m = 2t / ((1 − t) + √((1 − t)² + 4t/P²))`. For SDR sources and linear EXR `P = 1` and `t(m) = m`: no curve at all.
3. Gamma: `sign(x)·|x|^(1/2.4)` (2.2 for linear EXR, as the viewer shows them).

Back: undo the gamma, take the largest channel, invert the curve, scale, back to Rec.2020. Highlights above `P` map above 1 and come back; a model that leaves a pixel alone returns it to the same HDR value (within float rounding: the curve's slope is 2/(P(1+P)) ≈ 0.07 at the peak, so an error of 0.001 in the model's output becomes ≈ 0.015 of SDR white at 1000 nits, invisible after the display's tone map).

**Where the working space lives.**
- *Renderer* (`brp-render/src/effects.rs`, `EffectChain`): the chain's pictures are `rgba16f` images from its pool; a conversion is one more pass (`space.comp`), and the picture it replaced goes back to the pool at once, so conversions add no memory. `ChainFrame` says what space its input is in and which space the caller wants back (`Some(Source)` for the window). The pool and the stateful effects' history textures are freed when no effect is active (stack emptied, all disabled, effects switch off); live playback through the direct path then holds no working picture.
- *AI worker* (`brp-player/src/ai.rs`): pictures between stages are float32 tensors. Each stage's output is stored already in the space the next stage wants (`Job::spaces`, computed once from the stack: AI models want `display`, a group of shader effects takes any input), so neighbouring stages with the same space share one conversion, and the cache keeps converted pictures. The output of the last AI stage is always brought back to `source`: that is where the window's live shader chain takes over, and export (which runs the whole stack in the worker) converts at the same point, so viewer and export stay bit-identical.
- *Export* (`pack.rs`) is unchanged: it receives the source's encoding as before.

**Why `source` for mpv shaders.** libplacebo's `MAIN` is "RGB, before linearisation", in the source's transfer and primaries; existing shaders (ours too: Glow, Deblur, Keyer call `linearize()` themselves) depend on that. Hooks are honoured: `LINEAR` and `SIGMOID` get their libplacebo meaning; within one file passes run in mpv's stage order (`MAIN`/`RGB` → `LINEAR` → `SIGMOID`), file after file in stack order. Our extension `// @brp space <name>` (a comment, so mpv still reads the file) moves a file's `MAIN` passes to another space.

**Tests.** `cargo run -p brp-render --release --example spacecheck -- --exr <sequence>` runs every space's pass-through effect, alone and stacked, on SDR, PQ, HLG (a PQ frame re-tagged) and EXR frames, and compares with no effect after the display transform. The export pass-through test exports with no effect, then `--verify`s that file against the player's picture *with* pass-through effects (and the other way round); results below (§16 "Working colour space").

### 7b. HDR screens

When the window is on a screen Windows shows in HDR, the window is HDR10: an `HDR10_ST2084` swapchain (`A2B10G10R10`, PQ, Rec.2020), so HDR video, HDR renders (linear EXR above 1) and effects' highlights reach the screen instead of being tone mapped down. On an SDR screen nothing changes.

- **Detection** (`brp-render/src/hdr.rs`): the surface must offer HDR10 (`VK_EXT_swapchain_colorspace`), and DXGI must report the window's monitor in `G2084_P2020` (Windows HDR on). DXGI also gives the screen's peak (`MaxLuminance`), and the display configuration API gives Windows' "SDR content brightness" (the SDR white level). 0.3 ms; at start, then every 2 s on a thread, so moving the window to another screen or switching HDR in Windows rebuilds the swapchain. `BRP_HDR=0` never uses it, `BRP_HDR=1` always does where the surface allows.
- **Two layers, one composite** (`composite.rs`, `composite.frag`): the video is drawn by the same `video.frag` (HDR output bit) into a float *scene* picture as light, Rec.2020, 1.0 = SDR white; the UI is drawn exactly as for an SDR window (egui, 8-bit) into a transparent layer. One pass puts them into the swapchain: the UI blends in sRGB over the video's SDR part, so it looks as it does on an SDR window (and as other SDR apps do on the HDR desktop), and light above SDR white is added back dimmed by the UI's coverage, so highlights stay HDR under translucent glass.
- **Brightness:** SDR white (UI, SDR video, linear 1.0) at Windows' SDR white level, like every other SDR app; PQ and HLG at their absolute brightness (HDR reference white, 203 nits, is not lifted to the SDR slider); highlights above the screen's peak rolled off on the largest channel (hue kept): unchanged to ¾ of the peak, then a curve reaching the peak at 1,000 nits (PQ, HLG) or 10,000 nits (linear).
- **Popups' glass** samples an SDR picture of the window: while popups are open the composite also writes one (UI over the video clipped at SDR white), and the backdrop is made from it as before.
- **Cost:** two window-sized layers (4K window: 63 MB float scene + 31 MB UI; +31 MB while popups are open) and one full-screen pass. The effect chain, export and the SDR path are not affected.
- **Checked** on this machine (RTX 5070 Ti, 4K screen in HDR: peak 1,499 nits, SDR white 240): detection, playback of the PQ clip with and without effects, glass popups, the effects panel; validation layers clean (only the known VUID-06265 from other kernels). Windows' screenshots clip at SDR white, so the highlights were judged on the screen itself by the user: they look right.

### 7c. Image sequences, EXR layers

**Any image sequence opens from one of its files.** Picking a numbered still image (PNG, JPEG, TIFF, DPX, Cineon, TGA, BMP, WebP, JPEG 2000, JPEG XL, PSD, Radiance HDR, PFM, SGI, PNM, QOI…; `sequence.rs`) opens every file named the same apart from the last run of digits (any padding, gaps allowed), in number order; frame N is the Nth file, shown as `shot_####.png`, 24 fps (no audio). The picked file only decides the sequence; playback starts at its first frame. A lone image, or a name without a number, opens as before (one frame); an FFmpeg pattern with `%` still goes to FFmpeg's image2 reader. Each file is decoded by FFmpeg on its own (`decode_image`, the same conversion as video frames), three files at a time around the playhead (`Source::open_frames`, shared with EXR: same frame cache, requests and eviction as video, no index scan, any frame at once). Checked with 30-frame PNG, TIFF, DPX and JPEG sequences made by our export (`examples/seqcheck.rs`): every frame byte-identical to FFmpeg's own sequence reader; 70–460 fps at 1080p (DPX 69, PNG 88, TIFF 134, JPEG 463); export + `--verify` of PNG and DPX sequences opened from their 12th file: all 30 frames, mean 0.11 / 0.08 levels. DPX tagged "linear" (FFmpeg's DPX writer and many others tag everything so) is shown as video, as other DPX viewers do; float TIFF/PFM go through the 16-bit conversion (values above 1 clipped; EXR keeps them).

EXR files and sequences are read by our own reader (`brp-media/src/exr_seq.rs`, the `exr` crate), not FFmpeg, which reads single-layer files only. `Source::open` sends a `.exr` file or a folder of them there (an FFmpeg pattern with `%` still goes to FFmpeg, as `--verify` of an EXR export uses).

- **Sequence:** the files named like the one opened apart from the last run of digits (any padding), in number order; a folder: its largest such set. Frame N is the Nth file. Shown as `chap_####.exr`. Rate: the files' `framesPerSecond`, else 24.
- **Layers:** channels grouped by the name before the last dot (Blender: `ViewLayer.Combined.R`), per part in multi-part files (part name first). R/G/B/A map by name; layers that are not colour show their first channels as R, G, B, one channel as grey. A depth layer (`Z`, `depth`) is shown from near (black) to far (white) over the first frame's range, one range for the whole sequence so it does not flicker; Blender's 10¹⁰ background is far. Default layer: the plain RGBA channels, else Blender's `Combined`, else the first colour layer.
- **Decoding** only the chosen layer: the part holding it is decompressed block by block on all cores; only its channels are converted, to half-float RGBA (a new `FrameLayout::RgbaF16`, uploaded as `R16G16B16A16_SFLOAT`, so values above 1 survive), colour divided by alpha (EXR stores it premultiplied), anything beyond half float's range clamped. Three decode threads around the playhead, sharing `Source`'s frame cache, index and eviction rules, so playback, seeking, cache bars, effects and export work as for video. Measured: one 55 MB 1080p float ZIP16 file, 49–57 ms; through `Source`, 34.5 fps (enough for 24/30 fps at 1080p; a 4K float sequence would be about 4× slower). Compressions: all the crate reads (none, RLE, ZIP, ZIPS, PIZ, PXR24, B44, B44A); not DWAA/DWAB.
- **Colour:** linear light, 1.0 = SDR white, Rec.709 (Rec.2020 when the `chromaticities` say so): the working space (§7a) takes it as it is, an SDR screen shows it at gamma 2.2 clipped as before, an HDR screen (§7b) shows above 1.0 as HDR.
- **Layer picker:** a chip at the start of the title bar's footage chips (only with more than one layer) shows the layer; click it for the list. Choosing one opens the sequence again with that layer (`Player::set_layer`, `Source::open_exr(path, Some(layer))`), so every cache starts over for the new pictures (the background effects worker included: its keys carry the source, `Ai::set_source`, so neither its stage pictures nor its shader runners' uploaded frames, kept by frame number, are reused across layers or files; before, they were, and effects kept showing the old layer), while the position, In/Out, view and playback stay. Export of the open file uses the layer shown (`ExportJob::layer`; CLI `--layer=<name>`); batch items use the default layer.
- **Checked:** `footage/exr` (3 layers) plays at 24 fps, layers switch by clicking the chip (real mouse clicks) and with `BRP_TEST_EXR_LAYER=secs:name`; export + `--verify` of Combined and Depth (FFV1 10-bit: mean 0.10 / 0.17 levels), and of Noisy Image with a linear gain and AI Denoise (mean 0.12). `--verify` now applies the export's linear→gamma conversion to the player's picture before comparing (linear sources never reached export before). `cargo run -p brp-media --release --example exrcheck -- <file|folder> [layer] [--source]` lists layers and times decoding.

## 8. Startup, windows and the Lock toggle

- No splash, no resident server, no tray requirement.
- Startup budget: window <150 ms, first frame <300 ms for typical H.264 on a mid-range GPU. Techniques: lazy init (encoders, OCIO, audio device list), Vulkan pipeline cache and SPIR-V cache on disk, index read in parallel with first decode.
- **Single process, multiple windows.** The first launch owns a local socket while any window is open (gone when the last window closes).
- A new launch with a file sends the path to the running process and exits:
  - goes to the most recently focused **unlocked** window, replacing its media;
  - if every window is **locked**, opens a new window.
- **Lock toggle** in each window's top bar (padlock icon). Locked windows never get replaced.

## 9. UI

Modern, minimal, dark by default:
- Full-bleed viewer; one slim bottom bar: play/pause, frame step, timeline with In/Out range, frame field, timecode, loop, volume.
- Controls fade during playback, come back on mouse move. Same behaviour in fullscreen.
- Secondary items move to compact popovers/side panels: Effects, Color (OCIO), Info (metadata), Export, Log, Settings.
- Drag & drop anywhere (media files, or `.glsl` onto the Effects panel).
- Keyboard: old shortcuts kept (Space, ←/→, I/O, G, R, F, F11, Esc) plus Shift+←/→ ±10 frames, Home/End to In/Out, J/K/L shuttle, Ctrl+Z/Y in effects.
- Language switch (English / Português) in Settings, applied without restart.

### 9a. Interface overhaul (glass popups, timeline, devices)

**Bars.** In a window the player draws its own title bar (no system frame): app mark,
file name, footage chips (size, frame rate with i/p and field order — interlaced in
amber —, codec, pixel format and bit depth, SDR/HDR, Alpha (green) or Opaque, audio,
length and frame count), then lock, Info (circled i), Settings, Effects (wand) and Open,
then minimise, maximise and close. Dragging the bar moves the window, a double-click
maximises, the 6-point border resizes it (system resize). Title and bottom bar are solid
and stay; the picture is fitted into the space between them (`Player::video_rect` takes
that area). In fullscreen both bars float over the picture as gradients (black →
transparent) and fade out during playback; the window buttons become one exit button.
Play/pause and these icons are drawn as shapes (the triangle placed by its centroid),
not PNGs, so they sit exactly centred.
The transport (set In `[`, to In, step back, **play**, step forward, to Out, set Out `]`)
is centred in the bottom bar whatever the window width; the position (editable: a frame
number or `m:ss` / `h:mm:ss:ff`, the file's own timecode understood too), length, the
TC/F# display toggle and the deinterlacing chip sit to its left; loop, clear range,
volume, playback fps and the view buttons to its right. The play button is an accent
disc with a soft glow and a slow "breathing" ring while playing.

**Timeline** (`ui::timeline`): In value at the left end, Out value at the right end
(click: go there, double-click: clear the range). Above the track a ruler whose labels
pick their own precision: the tick step is the smallest "nice" interval ≥ ~74 px
(1, 2, 5, 10 frames, then 1 s … 4 h); labels show frames (`m:ss:ff`) only when the step
is under a second, hours only for clips over an hour, plain frame numbers in F# mode.
The current position and the In/Out values drop the hours the same way. Wheel zooms
around the pointer (min. 8 frames in view), Shift+wheel / horizontal wheel / middle
drag scroll, `+`/`-` zoom around the playhead, `\` or a double-click on the ruler shows
the whole clip. While zoomed in a thin overview under the track shows the visible part,
playback pages the view to follow the playhead, and frames wide enough to see get a
highlight under the playhead. The In/Out handles (amber brackets) are draggable — the
picture follows the handle like in an NLE — and outside the range the track is red with
hatching. Hovering shows a ghost line and the time in a pill above the track. Decoded
(cached) frames are a thin line along the bottom of the track.

**Feedback.** Toasts (a pill at the top centre that fades) confirm In/Out, volume,
mute, loop, range cleared and GPU changes. The empty window shows a breathing ring
inviting a drop.

**Popups as glass windows** (`brp-player/src/popup.rs`, `brp-render/src/popup.rs`).
Info, Settings and Effects are borderless OS windows of their own, so they can be moved
anywhere, also off the player window. Title bar: a small accent orb, the name and a
close button; Esc closes. Dragging the title bar moves the window by our own loop, not the
system one: each frame reads the pointer (`GetCursorPos`), moves the window there with a
*synchronous* `SetWindowPos` (winit posts its moves for later, `SWP_ASYNCWINDOWPOS`),
immediately presents the glass drawn for that position, then waits for the compositor
(`DwmFlush`), so the move and its frame reach the screen in the same refresh; the wait
also paces the drag to the display. The blur needs no per-frame work: the eighth-size
copy of the player picture does not depend on where the popup is, moving it only
changes where it is read. `BRP_GLASS_TEST=1` turns the glass into a plain
pass-through of the background (no tint, bend or content) for alignment checks: with it,
a popup should look like a blurred window onto exactly what is behind it, also while
being dragged. The
system move loop shifts the window first and shows the old frame (drawn for the old
place) until a new one arrives, which made the refraction trail and flicker. Popup
swapchains use MAILBOX (newest frame shown, no queue of stale ones), FIFO otherwise;
while dragging a popup redraws at most ~240 times a second. Their
height follows the content. On Windows they are *owned* by the player window (stay above
it, minimise with it, no taskbar entry), with system rounded corners and a thin border.
Each popup has its own swapchain (`PopupRenderer`) on the player window's device and its
own egui context; nothing is created at start-up, so launch time is unchanged.

The glass technique, cheapest first:
1. *Backdrop.* While any popup is open, the player renderer, right after drawing a
   frame, blits its swapchain image to a half-size image, that to a quarter-size one and
   that to an eighth-size one (three linear 2:1 blits = a box-filtered downsample,
   < 0.1 ms at 4K). Popups sample the eighth image. It is recreated with the window size (after the
   wait-idle the swapchain rebuild already does) and gets a generation number popups
   use to rebind their descriptor. Ordering needs no semaphores: the player and the
   popups submit to the same queue, and the barrier before the next blit waits for
   earlier fragment-shader reads.
2. *Glass pass.* A popup draws one full-window triangle before egui
   (`shaders/glass.frag`). For each pixel it maps its screen position into the player
   window (popup origin − player origin) and samples the eighth image with five cubic
   B-spline reads in a small tent (centre + four diagonals 0.9 texel away; each B-spline
   read is 4 bilinear reads). Plain bilinear magnified 8x showed the texel grid as
   blocks; the B-spline is smooth across texels. Together ~8 px of Gaussian-like blur
   at full resolution for 20 reads per pixel. The result is mixed 46% toward a
   dark blue-grey tint. Near the rounded border (a signed-distance rounded rectangle,
   16-point bevel) the sample point moves inward by up to 14 points along the border
   normal with a cubic falloff — the picture swells toward the edge like a lens rim —
   and the three channels bend by slightly different amounts (dispersion). A specular
   rim brighter toward the top-left and a faint top sheen finish it. Pixels outside
   the player window are written as premultiplied translucent tint instead.
3. *Outside the player.* The popup swapchain uses `PRE_MULTIPLIED` composite alpha
   (supported on this machine; `OPAQUE` fallback makes the tint solid) and the window
   asks the compositor to blur what is behind its translucent pixels
   (`SetWindowCompositionAttribute`, accent "blur behind", looked up at run time; it
   keeps working when the popup is not the active window, unlike the system acrylic,
   which turns flat — `BRP_ACRYLIC=1` uses acrylic anyway). The system drop shadow
   (`BRP_POPUP_SHADOW=1`) is off: it extends the frame into the client area, which
   turned the translucent part flat grey.

Cost per popup frame: one pass of 9–27 texture reads per pixel over a few hundred
thousand pixels, plus egui. Popups redraw when they change, when moved, and — only if
they overlap the player window — when the player presents a new frame. If a popup
window cannot be made (no presentation support), popups fall back to egui windows
inside the player window. `--show=info,settings,fx` opens popups at start (testing).

**Devices and display details.** The bottom bar has two chips, audio output and GPU
(short name, e.g. "RTX 5070 Ti"); a click opens a menu of devices (the audio list is read
when it opens). Positions default to frame numbers, zero-padded to the digits of the
last frame so In, Out and the playhead have one width; the In/Out areas are as wide as
that text needs. The app icon (colour) is in the title bar and is the window and
taskbar icon (64 and 256 px). Fullscreen: a borderless window on the current monitor,
kept on top (`WindowLevel::AlwaysOnTop`) so it covers the taskbar; a maximised window is
un-maximised first and maximised again on exit. `--fullscreen` starts that way.

**Settings store.** `%APPDATA%\BRP Player\settings.json` keeps language, GPU, audio
output, frame cache size, the TC/F# choice, volume, mute, loop, deinterlacing mode and
output, and the window's place (position, size as a normal window, maximised). Changes
are written a second after they stop (and when a window closes); the first window opens
where the last one was, if that place is still on a screen. New windows start from
these settings.

**DPI.** The process is per-monitor DPI aware (V2; winit sets it when the event loop
starts, so no manifest is needed). egui follows each window's scale factor, including
when a window or popup moves to a monitor with another scale; popup sizes follow
their content in points, and the glass works in physical pixels of the virtual screen.

**Glass on other GPUs.** The popup surface may not support per-pixel alpha (the AMD
iGPU here only offers OPAQUE composite alpha). The glass over the player still works
(it is drawn from the backdrop); the part outside the player is then a plain dark
panel. If the player swapchain cannot be copied from, popups are plain panels; both
cases are logged at start.

**Effects panel.** Each effect is a card with fixed parts: a grip (drag to reorder; an
accent line shows where it lands; one undo step), on/off, the name (left-aligned, cut
with an ellipsis), status, then up, down and remove. Icons are drawn, not font glyphs
(the font lacks ▲▼↶↷). Ctrl+Z / Ctrl+Y (and Ctrl+Shift+Z) work in the player window and
in a popup window that has the keyboard, unless a text field is being edited.

**Add effect menu.** Two labelled groups: "AI effects" (a drawn sparkle before each; a
tooltip says they run in the background and can take a moment per frame) and "Shader
effects" (live). It opens as a list inside the Effects window (not a dropdown), so the
window simply grows with it: a dropdown inside a small popup window is cut to the
window, and growing the window for it fed back into its own size. `BRP_TEST_FX_MENU=1`
starts with the list open. The UI font has no arrows or decorative symbols:
`cargo run --release -p brp-player --example glyphcheck` lists every non-ASCII character
in the UI strings and whether the font can draw it (all can, as of this change).

**Scrubbing.** While the playhead or an In/Out handle is dragged, the picture is the raw
decoded frame (deinterlaced if on): no live effects (`VideoDraw::raw`), no history replay
for temporal effects, no AI requests and no decoding ahead for them. The frame let go on
is processed normally; a click on the timeline (no drag) processes that frame. The
device chips sit at the left of the bottom bar and shrink to icons when there is no
room; title-bar info chips that do not fit are left out.

**Devices.** Settings lists every Vulkan GPU (dedicated / integrated) and the audio
outputs. Choosing a GPU (`Win::switch_gpu`) closes the popups, clears the AI cache and
detaches the AI from the old device, drops the renderer (swapchain first), creates a new
instance + renderer on the chosen GPU (~150 ms), starts a fresh egui context (its
textures lived on the old device), gives the AI the new device (its worker rebuilds
itself on the next job) and re-applies the effect chain; on failure it returns to the
previous GPU. `BRP_GPU` still overrides the saved choice at start.

**Frame cache.** The decoded-frame cache of each open file defaults to 70% of total
RAM (`brp_media::mem`, from `GlobalMemoryStatusEx` / `/proc/meminfo`), adjustable in
Settings (0.5 GB … 90% of RAM) and saved. With several files open each has its own budget.

**Filling while paused.** Paused, the decoder used to stop after the frame and a few
neighbours (3 ahead, 8 behind). It now keeps decoding ahead of the playhead in the
background (`Request::fill`; off while scrubbing):
- while the cache is under 80% of its budget;
- once full, while the next frame ahead is nearer the playhead than the farthest one
  kept. The eviction then drops that far one, so after a jump the cache moves to the new
  place and decoding stops when it settles, instead of churning.

Forward only (no seeks); a request is still answered within a batch of 8 frames.
`fillcheck` (brp-media example), stage clip with a 512 MB budget: paused at frame 100,
the cache filled to frame 217 in under 0.5 s and stopped. After a jump to 600 it held
592–718 within 0.5 s and stopped.

## 10. Feature parity checklist (from `old/`)

**Media**
- [ ] Open video: mp4, mkv, mov, avi, mxf, webm (incl. VP9 alpha), gif
- [ ] Open image sequence from one file: png, jpg, exr, tiff, tga, bmp
- [ ] EXR layer/AOV selection (direct, no pre-render)
- [ ] Drag & drop, command-line file argument, OS "open with"
- [ ] Automatic interlace detection (flags + content analysis); deinterlace Auto / Off / NNEDI3 / NNEDI3 + motion-adaptive / bwdif / yadif, field or frame rate (§6b)
- [ ] Metadata panel: file, field order, codec, alpha, HDR, audio, fps, color space, resolution

**Playback**
- [ ] Play/pause (button, Space, click on video), stop → In point
- [ ] Frame step forward/back, go to In/Out
- [ ] In/Out points (I/O, editable fields), clear range, loop on/off within range
- [ ] Playback FPS override (speed), frame number field seek, timecode
- [ ] Timeline slider with In/Out markers
- [ ] Volume, mute, audio output device switch (no player restart)

**View**
- [ ] Zoom at cursor (wheel), pan (middle drag), 1:1 (G), reset (R)
- [ ] Window fullscreen (F11), video fullscreen (F / double-click), Esc to exit
- [ ] GPU device selection

**Effects**
- [ ] Ordered chain of GLSL shaders + builtins, drag to reorder, enable toggles
- [ ] Builtins: temporal denoiser (atadenoise), 3D denoiser (hqdn3d), deband, FFT denoiser, deshake
- [ ] Param widgets: float/int sliders + numeric entry, color pickers, categories, checkboxes, choices, spinboxes
- [ ] Undo/redo, presets save/load (new format; old presets are not loaded), shader folder, import by drag & drop / file picker
- [ ] All shaders in `old/shaders/` including temporal ones working

**Color**
- [ ] Load `.ocio` config, input/output transforms, applied in viewer and export
- [x] HDR (PQ/HLG) and linear EXR handling (§7a–7c)

**Export**
- [x] Containers/codecs from `ui_export_dialog.py` (H.264/H.265, ProRes family, DNxHR family, XDCAM, AV1, VP9, GIF, JPG/PNG/EXR sequences), plus FFV1, Ut Video, CineForm, QuickTime Animation, HAP, DXV3, MJPEG, TIFF/DPX sequences, and any other encoder of the bundled FFmpeg ("Other") (§16, Export)
- [ ] DXV3 and DXV3 with alpha (DXT5, CVExplorer patch), playback and export (§11b) — export: DXV3 (DXT1) done; the alpha preset appears by itself when the FFmpeg build has the patch (`dxv` `format=dxt5`); this build does not
- [x] Hardware encoders when available (any vendor), software fallback — NVENC, AMF, Quick Sync, MediaFoundation probed at run time; Vulkan/VAAPI/D3D12 encoders (hardware frames only) not yet
- [x] Resolution presets/custom, Fit/Fill/Stretch, quality/bitrate, alpha export, HDR export (HDR kept, or tone mapped to SDR)
- [ ] Export In/Out range exactly, with deinterlace, effects and OCIO baked in (same engine as viewer) — done except OCIO (not in the app yet)
- [x] Export presets (new format), progress + ETA, cancel
- [x] Batch export queue with per-item status

**App**
- [ ] Log window
- [ ] Single instance + Lock toggle + multi-window
- [ ] English / Portuguese
- [ ] New: real scopes (waveform, vectorscope, histogram); the old scopes dialog was a stub

## 11. Milestones

1. **Skeleton:** Vulkan + winit + egui shell, open file, software decode, display, startup time measured.
2. **Seek engine:** index, exact seek, GOP cache, step/reverse, In/Out/loop, audio sync. Seek hash tests green.
3. **Shader engine:** format parser, stages, params UI, storage textures, temporal model. Golden tests vs mpv for every shader in `old/shaders/`.
4. **Color:** OCIO GPU, HDR, linear EXR, EXR layers, image sequences.
5. **Builtins + deinterlace**: automatic interlace detection (flags + content analysis), NNEDI3 and NNEDI3 + motion-adaptive on the GPU, bwdif/yadif, field- or frame-rate output (§6b); temporal builtins (libavfilter windowed, then GPU ports).
6. **Export + batch + presets.**
7. **HW decode per platform/vendor, multi-window + Lock, i18n, scopes, polish, packaging** (Windows installer, macOS .app with MoltenVK, Linux AppImage/Flatpak).

## 11b. DXV3 with alpha (from CVExplorer)

Stock FFmpeg's `dxv` encoder only writes DXT1 (no alpha). CVExplorer (`github.com/brurpo/CVExplorer`) carries `vendor/ffmpeg/0001-dxvenc-dxt5-alpha.patch`, which adds `-format dxt5`: DXT5 textures with **premultiplied** alpha, MOV tag `DXD3`, PCM audio.

- **Playback:** the stock `dxv` decoder reads DXT5 as DXT4 and **un-premultiplies** (colour × 255 / alpha, capped at 255; colour unchanged where alpha is 0). Checked against FFmpeg 7.1's libavcodec: colours are not double-premultiplied.
- **GPU decoding (done, §16 "GPU decoding"):** the LZ-style unpacking runs on the CPU, the DXT blocks are expanded on the GPU with FFmpeg's integer arithmetic (not hardware BC sampling, whose interpolation differs per vendor), bit-exact with FFmpeg for DXT1 and DXT5.
- **Export:** "MOV / DXV3" and "MOV / DXV3 (alpha)" presets. DXV3 alpha is written by our own GPU encoder (DXT5, premultiplied, tag `DXD3`; §16 "GPU encoders"), so the stock FFmpeg is enough (an FFmpeg with the CVExplorer patch is used instead only when GPU acceleration is off).
- Test: round-trip export → decode must give the same alpha (CVExplorer reports mean error 0.00).

## 12. Risks and open points

- **Licensing:** linking x264/x265 makes the app GPL. Options: ship as GPL, or LGPL FFmpeg + use vendor/OS encoders and optional external GPL encoders. Needs a decision before milestone 6.
- **MoltenVK limits** on macOS (some storage image formats, subgroup ops) may need shader fallbacks; validate early with the heavy shaders (`upscale4x`, `temporalNL`).
- **Vulkan Video** maturity varies by driver; D3D11VA/VAAPI/VideoToolbox remain the default HW paths.
- **Reverse playback of long-GOP 4K** is bounded by decode speed and cache size; prefetch + HW decode should cover it, measured in milestone 2.

## 13. Test footage (`footage/`)

Read from the file headers (no ffprobe on the dev machine).

| File | What it is | What it tests |
|---|---|---|
| `20250725_164717.mp4` | HEVC 1920x1080, **HDR10** (BT.2020 / PQ), **variable frame rate** (~59.7 fps), B-frames, GOP ~57, AAC audio, 128 frames | HDR tone mapping, VFR frame indexing (no `time × fps`), HEVC seek, A/V sync |
| `interlaced.mp4` | H.264 1280x720 29.97, B-frames, GOP 30, 245 frames, no audio | Interlace detection, deinterlace modes, normal long-GOP seek |
| `interlaced_prog.mp4` | H.264 1280x720 29.97, B-frames, **one keyframe for all 245 frames** | Worst-case seek: last frame needs 245 decodes. Step back and reverse play must still feel instant (GOP cache + prefetch) |
| `test.mp4` | H.264 1920x1080 30 fps, B-frames, GOP 64, **timecode track**, 74 frames | Container timecode display, seek across a GOP boundary |
| `McCartney-MPEG2_1080-interlaced_sample.mkv` | MPEG-2 1080i in MKV, AC3 audio, 135 MB | Interlaced broadcast material, likely **open-GOP** (sync point one GOP earlier), MKV cues index, AC3 |
| `PASSAGEM_ALPHA_LOGO_SBT_MANHA.mov` | QuickTime Animation (RLE) 1920x1080, **32-bit with alpha**, intra-only, 30 fps, 74 frames, ~6.6 MB/frame, timecode track, `fiel` progressive | Alpha display (checkerboard), alpha export, parallel intra decode, high bandwidth |
| `exr/chap_0001…0009.exr` | Blender multi-layer EXR 1920x1080, 32-bit float, ZIP, ~55 MB/frame, **no plain RGBA layer**: `ViewLayer.Combined` (RGBA), `ViewLayer.Depth` (Z only), `ViewLayer.Noisy Image` (RGBA); layer names contain spaces | Layer picker, single-channel layer display, decode only the selected layer's channels, parallel float decode, linear float pipeline |

`gpudec/` (made from the files above with FFmpeg 7.1's encoders and our export, 24 frames each, for the GPU decoders): ProRes Proxy/LT/Standard/HQ (`prores_ks` and `prores_aw`), HQ interlaced, HQ 4K, 1000×562, 4444 XQ, 4444 with 16-bit alpha; DNxHD 120 (8-bit), 185x (10-bit), both interlaced, 720p 60/90x; DNxHR LB/SQ/HQ/HQX/444, HQ 4K, SQ 1000×562; DXV3 DXT1 1080p/4K; DXV3 DXT5 alpha (our export); XDCAM HD422 1080i MXF. (`CRV CHA EITA LUCAS BASE.mov` is itself DNxHD.)

Not covered yet (would be good to add later): **a DXV3-alpha file made by CVExplorer or Resolume** (§11b), DXV3 high-quality (YCoCg) files, 12-bit DNxHR, VP9 with alpha (WebM), HLG, AV1, a long real-world clip (minutes, to test indexing and cache limits).

## 14. Development environment (current machine)

- Windows 11, GPUs: **NVIDIA RTX 5070 Ti + AMD Radeon iGPU** (two vendors for multi-vendor testing on one box).
- Installed: Rust 1.99.1 (`~/.cargo/bin`, not on the shell PATH yet), Visual Studio Build Tools 2022 (MSVC).
- Missing, needed for milestone 1: Vulkan SDK (validation layers, shaderc), FFmpeg 7.x shared dev build, LLVM/clang (bindgen), CMake.

## 16. Status

### Milestone 1 (skeleton) — done

- Workspace: `crates/brp-media` (FFmpeg demux, frame index, decode worker, GOP cache), `crates/brp-render` (Vulkan 1.3, dynamic rendering, video pass, egui overlay), `crates/brp-player` (app, UI, i18n EN/PT).
- **Exact seek verified** with `cargo run -p brp-media --release --example seekcheck -- <file>`: every frame reached by random seeks and backward steps is byte-identical to sequential decode. All 6 sample videos pass (0 mismatches, ~300 seeks).
  - Fixes found by it: open-GOP leading pictures before the first keyframe are not counted as frames (MPEG-2 `.mkv`: 1798 real frames); edit-list frames are excluded (phone HEVC: 126, not 128); decoded frames are kept and re-numbered when the index arrives (HEVC first frame 250 → 77 ms).
  - Worst seek on long-GOP HEVC 1080p 10-bit (software decode): ~300 ms; stepping back inside a decoded GOP: microseconds.
- Display: YUV (any matrix/range/bit depth), packed RGBA with alpha over checkerboard, PQ/HLG tone mapping, display rotation (phone footage), zoom at cursor, pan, 1:1, fullscreen, auto-hiding controls, icons from `/icons`, window icon.
- Works on **NVIDIA RTX 5070 Ti and AMD Radeon iGPU** (`BRP_GPU=AMD` selects it).

### Milestone 2 (playback) — done

- **Audio:** decode + resample (`swresample`, downmix e.g. AC3 5.1 → stereo) into a lock-free ring, played by cpal (WASAPI / CoreAudio / ALSA-Pulse). Sample-accurate start at the current frame after every seek/loop. Plays only at native speed forward; silent while scrubbing, at other speeds or reversed.
- **A/V sync:** video clock slews toward the audio clock (hard resync above 250 ms). Measured drift on the HEVC/AAC and MPEG-2/AC3 samples: within ±0.3 ms, including across loops. (This proves the clocks agree; the driver-reported output latency still needs a listening check.)
- Volume, mute (M, ↑/↓), audio output device in Settings (switches without losing the position).
- Loop works without In/Out (end detected by time, not frame number).
- **Single instance + Lock:** a second launch hands its file over a local socket (named pipe on Windows) and exits in ~50 ms. The file goes to the most recently used unlocked window, or to a new window of the same process if all are locked. `--new-window` forces a separate process, `--lock` opens the first window locked. Nothing stays running after the last window closes.

### Deinterlacing — done

- **GPU frame cache + compute stage** in `brp-render` (frames uploaded once, reused; deferred destruction). This is the base the shader engine builds on.
- **NNEDI3** (compute shader, port of FFmpeg `vf_nnedi.c`, weights `nnedi3_weights.bin` shipped next to the exe) and **bwdif** (integer port of FFmpeg `bwdifdsp.c`), plus **Best** = NNEDI3 spatial estimate inside bwdif's motion-adaptive check.
- **Golden tests vs FFmpeg** (`cargo run -p brp-render --release --example deintcheck -- <file> <frame> <ffmpeg.exe>`, needs the GPL FFmpeg): bwdif **100.00% identical** on every plane; NNEDI3 **100.00%** (luma has a handful of ±1 float-rounding ties). Same results on NVIDIA and AMD. Tested on 720p H.264 and 1080i MPEG-2.
- Speed, 1080 frame: RTX 5070 Ti 0.8 ms (shared-memory variant), AMD iGPU ~24 ms with the full network. On integrated GPUs playback uses a lighter network (16 neurons, NNEDI3 on luma, bwdif on chroma); paused frames, stepping and export always use the full FFmpeg-equivalent network. Measured 1080i field-rate playback: NVIDIA 59.5–60 pictures/s, AMD iGPU ~59 pictures/s steady (43–51 in the first seconds while indexing).
- **Auto detection from content** (port of FFmpeg `idet` metric, 24 sampled frame triples, majority vote) overrides wrong flags: `interlaced_prog.mp4` is flagged progressive but detected as interlaced TFF and deinterlaced automatically.
- Field-rate output (50i → 50p) with field-by-field stepping, or frame rate; mode and output in Settings; a badge in the bottom bar shows the active method; `--deint=off|auto|best|nnedi|bwdif`.
- Project license: GPL-3.0-or-later; dev FFmpeg switched to the GPL 7.1 build (has `nnedi`, x264/x265 for export).

### Milestone 3 (shader engine) — working

- **Parser + codegen** (`brp-render/src/usershader/`): mpv/libplacebo format (HOOK, BIND, SAVE, WIDTH/HEIGHT/WHEN RPN, COMPONENTS, COMPUTE, TEXTURE/STORAGE, PARAM float/int/uint/DEFINE, our `TEMPORAL_WARMUP`). Every pass runs as a compute shader with the libplacebo environment (`NAME_raw/_pos/_size/_pt/_mul/_rot/_off/_tex/_texOff/_gather`, stage-name aliases, `frame`, `random`, `input_size`, `target_size`, `linearize`). Params live in a uniform buffer (no recompile on slider moves); constant-expression users fall back to consts.
- **Compile check**: `cargo run -p brp-render --release --example shadercheck -- shaders` → all 24 shipped shaders compile (261 passes). Of the 29 old files, 3 were broken in the files themselves (`MB.glsl`, `_test_mb.glsl`, `_test_temporal2.glsl`); mpv rejects them too.
- **Effect chain** (`effects.rs`): planes → LUMA/CHROMA hooks → MAIN (merge) → MAIN/RGB hooks → display. Background compiles, per-effect persistent storage textures, shared output (a stateful shader never processes a picture twice).
- **Temporal determinism**: after any jump the history is cleared and the previous pictures are replayed silently (default 2, `//!TEMPORAL_WARMUP n`), including deinterlacing; `frame` counts from the reset (0 = no history, as in libplacebo); `random` is a hash of the picture number. Paused frames are reused, not recomputed.
- **UI**: Effects panel (fx icon): add from library (`<exe>/shaders`, `%APPDATA%\BRP Player\shaders`) or browse, drop a `.glsl` on the window, enable, reorder, remove, undo/redo (Ctrl+Z/Y), parameter sliders, R/G/B triples as colour pickers, `(cat: …)` groups, presets (JSON, by parameter name). `--fx=A.glsl,B.glsl` starts with effects.
- **Verified against libplacebo** (FFmpeg `libplacebo` filter, same frames): Gaussian blur, spatial denoiser, Invert match; Painterly, Edges, Glow, FSRCNN (WHEN logic) look right; nlmeans CHROMA path matches.
- **Fixed — global initializers**: `NAME_pos/_size/_pt` are now expressions, not globals filled in `main()`. Shaders that compute in global initializers (e.g. nlmeans' `vec4 poi_ = GET_(vec3(0));`) read pixel (0,0) before; that broke the temporal denoiser (cyan fringes, red lines) and others. MAIN from YUV is clamped to [0, 1] like libplacebo's. All shipped shaders now match libplacebo within ~0.002 mean (random-based ones excepted).
- **Tools**: `examples/fxcheck.rs` runs effects offline on real frames (NaN/black counts, diff to source, crops, `--frames`/`--step` for temporal effects, `--ref` to compare with `tools/plref.sh`, a libplacebo reference render via FFmpeg).
- **Shader changes**: ColorCorrection saturation/vibrance no longer go negative in linear light (was NaN -> black). Deblur rewritten (Richardson-Lucy on linear luminance, separable Gaussian PSF). temporalNL replaced by our own temporal NL-means (the original hung the GPU: a `switch` falling off a non-void function, 2-channel history in r16f, unbound texture). MB rewritten: pyramidal Lucas-Kanade flow + tile-max/McGuire reconstruction. Upscalers always run and have an `Upscale` switch (off: Lanczos downscale back to source size; on: output the larger picture, shown in Info and used by 1:1). Invert has a single intensity. Removed `invert.glsl` and `Builtin_Sharpen.glsl` (ColorCorrection sharpens).
- **Expressions** can use parameter names (`//!WHEN Upscale !`). Our warm-up hint can be a comment: `// @brp temporal_warmup n`. libplacebo compatibility is not a goal.

### AI (our own Vulkan runner, all on the GPU) — working: upscaling, interpolation, motion blur, background removal

- **No CPU in the loop**: everything runs on the window's own GPU device and stays there. Decoded frames are uploaded once and converted to RGB by the effect chain's merge pass (`OfflineEffects::begin/record` into the worker's command buffer). From then on they are GPU tensors: shader stages read them as pictures, models and temporal processing work on tensors. The result is an RGBA16F `Picture` that the renderer draws directly (`VideoDraw::picture`); effects after the last AI entry run live on it. One command buffer per output picture, submitted on a second queue of the same family (`compute_queue`), so the renderer's queue is never blocked. ncnn is gone (no DLL, no CPU warps, no conversions).
- **Network runner** (`brp-ai/src/vk.rs`): reads ncnn `.param`/`.bin` files (the model files stay in that format). It covers every layer the 7 models use: conv (plain, grouped, depthwise), deconv, the inner product, eltwise/binary/unary ops, interp (nearest/bilinear/bicubic), pixel shuffle, pooling, channel mean, and RIFE's warp. Tensors are planar f32 (or f16, below) in pooled buffers (best-fit reuse) addressed by buffer device address. Concats are planned away: producers write into slots of one buffer, which goes back to the pool once its last reader has run (keeping them to the end of the network took 8 GB per tile on the big upscaler). All 7 models matched ncnn (max difference 0.0016–0.006 with fp16 tensor cores; RIFE and RVM near exact) before ncnn was removed.
  - **3×3 stride-1 convolutions** run on tensor cores (`VK_KHR_cooperative_matrix`, fp16×fp16→fp32, 16×16×16) when available, otherwise on a tiled fp32 kernel. The input block is staged once in shared memory, pixel-major (16 input channels of a pixel side by side), so the 9 taps are the same block shifted by whole pixels; this replaced three shifted copies and halved the time. Weights are fed to the tensor cores straight from memory. The next 16 input channels load into registers while the current ones are multiplied. Workgroup: 4 subgroups of 3 pixel rows; 64 output channels × 16×12 pixels, or 32 × 16×12 (`BRP_COOP64` / `BRP_COOP32` = "tiles,rows,subgroups" for tuning). 3 rows beat 2 or 4 by 8–16%; fewer channel tiles or 8 subgroups were slower.
  - **Fusion at load**: a tensor-core convolution whose output only feeds a PReLU or a two-input Eltwise sum does that layer in its output step (17 PReLUs in the fast upscalers: 93 → 71 ms; 69 residual sums in the big ones). `BRP_NN_NO_FUSE` turns it off.
  - **4×4 stride-2 transposed convolutions** (RIFE) use their own kernel: an output pixel's parity picks its 4 taps, and each invocation makes 8 channels from weights rearranged to [ic][tap][oc]. RIFE went from 124 to 24 ms (its largest deconvolution from 49 to 1 ms).
  - **fp16 activations** (`Net::set_half`, on for the upscalers only): tensor-core convolutions, element-wise ops and zero-copy concats read and write f16 tensors. Any other layer gets an f32 copy, and network inputs and outputs stay f32. Kernel variants per element type are compiled on first use; a branch per load serialized the loads. The convolution reads f16 input as 32-bit words, because 16-bit loads made it twice as slow (found with `BRP_PIPELINE_STATS`, the driver's statistics per kernel, and by timing each variant). Gain: 4–11% (best 4× 2.10 → 1.86 s), with output within 62–81 dB of f32. The gain is small because the f32 kernel already hides most of its reads behind the products: the planes are per channel, so f16 halves are read in 32-bit words and the bytes read stay the same. Storing channel pairs side by side would halve them, but every f16 kernel would need the new layout. `BRP_NN_FP32` turns fp16 activations off.
  - **Winograd F(2×2, 3×3)** (`CONV_WINO`, opt-in with `BRP_NN_WINO`; weights transformed at load only then). It does 2.25× fewer products: per 16 input channels the input block is transformed in shared memory into 16 position matrices, each subgroup multiplies 2 of the 16 positions on tensor cores, and the output is transformed back with the same output step as the direct kernel. It is accurate (60–68 dB from the direct kernel) but 15–25% slower on the RTX 5070 Ti: fast 2× 76 vs 61 ms, best 2× 0.63 vs 0.54 s, best 4× 2.16 vs 1.74 s. The transforms are cheap (4% of its time). The loss is in the products: each subgroup owns its positions, so each weight block it reads (from L2, never shared with another subgroup) feeds only 2 products. In the direct kernel 4 subgroups share each weight block (mostly L1 hits) and each feeds 3 products. Its 256-thread workgroups at 231–255 registers also leave one workgroup per SM, so the phases (stage, transform, multiply) do not overlap across workgroups. Loading the weights before the transform gained 4%; other shapes were slower. Making it win would need the weights shared across subgroups (e.g. staged in shared memory), which costs registers this kernel does not have.
  - **fp16 sums** (`Net::set_fp16_sums`, on for best 2×/4× and anime 4×): GeForce tensor cores multiply-add at twice the rate when the sums stay in fp16 (98 against 49 TMAC/s here). Summing a whole convolution in fp16 is 10–13% faster on the big upscalers, and halves the accumulator registers; the output stays within 56–64 dB of fp32 sums (ncnn also ran these models with fp16 arithmetic). The compact models are 6% slower with it, so they keep fp32. It is used only when the GPU offers fp16-accumulating cooperative matrices. An earlier try that summed each 16-channel step in fp16 and then added it to fp32 totals was slower: the conversions cost more than the faster products saved. `BRP_NN_FP32_SUMS` turns it off.
  - Measured ceilings on the RTX 5070 Ti (`coopbench`): 49 TMAC/s with fp32 accumulation, 98 with fp16. Accumulating in fp16 per 16-channel step was slower, and smaller image tiles did not help. The tensor-core kernels use all 255 registers (`BRP_PIPELINE_STATS`), which limits each SM to 8 warps. Partial unrolling of the 9 taps cut that to 170–220 registers but was 15–40% slower: the full unroll's overlap of loads and products matters more than occupancy here. In fp16 mode, the register that tracks which half of each 32-bit word holds the value is recomputed when staging instead of kept, to save registers.
  - `BRP_NO_COOPMAT`, `BRP_NN_NAIVE`, `BRP_GPU_PROFILE` (per-dispatch timings by layer type and size) and `BRP_PIPELINE_STATS` are for debugging; `BRP_UPSCALE_TILE` sets the upscaler's tile size.
- **Upscaler tiling and GPU memory** (`gpu.rs`):
  - **Equal tiles.** The picture is split evenly (1920 = 3 × 640, not 768 + 768 + 384), and every network input has the same size (edge-replicated past the picture), so one set of buffers serves every tile. Tiles of different sizes had needed buffers of their own: best 4× held 8.8 GB, now 4.4 GB at the same tile size.
  - **Tile size from memory.** The size is the largest that fits a budget, chosen once per picture size: a third of the GPU's memory, at most 5 GB, and at most half of what is free (the driver's figure through `VK_EXT_memory_budget`, other programs' use counted) plus what the upscaler already holds. Memory per tile pixel was measured with the allocator's report (`aicheck upscale` prints it): about 8 KB for best/anime 4×, 1.6 KB for best 2× and 0.6 KB for the compact models, margin included. The output picture and 256 MB come off the budget first. The largest tensor stays under 2³⁰ elements (32-bit index arithmetic). `BRP_AI_MEMORY_MB` sets the budget and `BRP_UPSCALE_TILE` the size.
  - **Larger tiles are faster:** best 2× 470 → 380 ms from 384 to 1024 px, best 4× 1.75 → 1.58 s from 512 to 640 px. On this 16 GB card best 4× uses 640 px tiles and 4.9 GB; a 2.7 GB budget gives 384 px and 2.6 GB.
  - **Border.** The large networks see far (300+ layers of 3×3). With 10 px of context, a tiled picture differed from the whole one by 32.5 dB along edges everywhere, not only at the seams; with 32 px of context it differs by 48 dB, for 5–13% more work. The compact models keep 10 px (45 dB).
  - **Short submissions.** Each tile is its own GPU submission (`Recorder::flush`), so no submission runs for long: Windows resets a GPU that stays busy for about 2 seconds.
  - **Bug found with GPU-assisted validation** (`VK_INSTANCE_LAYERS=VK_LAYER_KHRONOS_validation VK_LAYER_GPUAV_ENABLE=1`). The f16 output tensor was dropped right after its conversion back to f32 had been recorded. That freed its memory before the last layer had written into it, so the GPU later wrote into memory that other buffers could be using. The `aicheck` tool happened to survive this; best 4× in the player lost the device after a few frames. The tensor now goes back to the pool, which keeps it alive. The validation run also flagged the missing `vulkanMemoryModel` feature, now enabled.
- **Models as picture operations** (`brp-ai/src/gpu.rs`): `Upscaler` (equal tiles sized by GPU memory, see below; `General2x` folded 2×2 into the x4plus graph; alpha resized), `Rife` (pad, network, crop, alpha mixed), `Matting` (recurrent state kept on the GPU). `upscale.rs`, `rife.rs` and `matting.rs` only describe the files, sizes and padding (and rewrite RVM's baked-in sizes per input).
- **Temporal processing** (`brp-ai/src/gpu_temporal.rs`): motion (pyramidal Lucas-Kanade), the temporal window, the fast motion blur and the background-removal outputs, as compute kernels. The CPU version (`temporal.rs`) stays as the reference: `aicheck gtemp` gave the same motion (mean |Δ| 0.004 px), window (8e-5) and blur (5e-5) until the GPU motion pyramid got its anti-aliasing filter (see the motion notes below); the CPU one still averages 2×2. The scene-cut and noise decisions are made on the GPU too: reductions write the mean agreement, and a histogram median gives the typical difference. These go into a small stats buffer that the merge kernel reads. 1080p: motion 0.7 ms (CPU 76 ms), window with its 4 motions 6–8 ms (CPU 0.5 s), motion blur 1.8 ms (CPU 0.3 s).
- **AI optical flow, evaluated** (`tools/flowref`: `run_flow.py` runs the official PyTorch code with CUDA from a private Python in `third_party/pyref`; `score.py` scores; `particles.py` makes a test with known motion; `aicheck flowdump` dumps ours). Four open models, 1080p, RTX 5070 Ti, PyTorch fp32 per pair:
  - WAFT a1 (BSD-3, ICLR 2026): 1.56 s.
  - MEMFOF Tartan-T-TSKH (BSD-3, ICCV 2025; 3 frames, both directions): 0.5 s (0.29 s fp16).
  - SEA-RAFT M Spring (BSD-3, ECCV 2024; at half size): 50 ms.
  - MemFlow things (Apache-2.0, CVPR 2024): at half size, 0.27 s; at 1080p its all-pairs correlation needs about 8 GB and runs out of memory.

  Each flow moves the next frame onto the current one. Scores are the share of detailed pixels off by more than 0.1 (lower is better):

| | Ours | MEMFOF | SEA-RAFT | WAFT | MemFlow |
|---|---|---|---|---|---|
| Stage clip | **9.5%** | 13.1% | 14.1% | 11.3% | 19.1% |
| Concert | **8.3%** | 11.6% | 10.1% | 11.5% | 14.0% |
| Motion graphics (thin fast streaks) | **41%** | 61% | 68% | 50% | 61% |

  Against true motion (a real picture panning, with particles each moving their own way), end-point error:

| | Ours | MEMFOF | SEA-RAFT | WAFT | MemFlow |
|---|---|---|---|---|---|
| Background (all pixels) | 1.5–2.0 px | 0.14–0.23 | 0.15–0.22 | **0.15–0.20** | 0.15–0.27 |
| Particles 8–20 px, 3–15 px/frame: error / share > 3 px | 5.7 / 67% | 4.9 / 56% | 4.6 / 52% | **3.7 / 36%** | 5.4 / 66% |
| Particles 2–7 px, 4–30 px/frame: share > 3 px | 81% | 97% | 83% | 97% | 95% |

  - The networks' motion is about ten times more accurate on whole surfaces.
  - Ours fits each pixel's colour best (it minimises exactly that, grain included). That is what the merge needs.
  - With particles they can see, WAFT tracks them best. None tracks particles smaller than its feature grid (8–16 px) that move farther than their own size each frame.
- **AI optical flow in the player** (`brp-ai/src/flow.rs`). Upscale, Motion Blur and Denoise have a **Motion** drop-down: Built-in (ours, the default), SEA-RAFT (AI), WAFT (AI). Each AI model loads the first time a stage asks for it, and its motions are cached like ours (the cache key includes the method). `tools/fetch_ai.sh` downloads and converts both (`pth2ncnn searaft` / `pth2ncnn waft`); the start-up warm-up builds their shaders.
  - **SEA-RAFT M** (Spring fine-tune): three graphs (features, context, one refinement step), run 4 times at half size. The correlation is computed on the fly from pixel-major f16 features (no all-pairs volume). The flow is upsampled ×8 with the learned convex weights. 1080p: **39 ms** per pair, 315 MB. Against PyTorch: 0.006 px mean, 0.055 px at the 99th percentile.
  - **WAFT a1**: DINOv2 ViT-S (class token, position embedding resized bicubically with PyTorch's +0.1 scale), a DPT head and a ResNet-18 give the features. A ViT refines the flow 5 times on 8 px patches at half size, with each step warping the second frame's features. The flow is upsampled ×2 convexly. The picture is padded to a multiple of 112 (zeros after ImageNet normalization). Six graphs plus the position embeddings. 1080p: **0.70 s** per pair, 3.1 GB (PyTorch: 1.56 s). Against PyTorch: 0.028 px mean, 0.37 px at the 99th percentile.
  - **Making WAFT faster** (1080p pair, RTX 5070 Ti: 1.70 s at first, then 0.70 s). Global attention (84 calls over 8,800-11,500 tokens) took 65% of the time:
    - Keys and values in shared memory were token-major rows of 64 halves, so a subgroup's writes all hit one bank. Channel-major: 1114 -> 976 ms.
    - One pass instead of two. The second Q·Kᵀ is gone: scores stay f32, and a row's reference maximum is raised only when a block passes it by more than 8 (e^8 still fits the f16 probabilities). Only then is the output so far scaled, through shared memory, which is rare after the first blocks. The sums divide the output at the end: 976 -> 766 ms.
    - Keys and values re-laid once per call (`KV_T`: [head][token][64] f16, tokens padded to 64 with zeros). Each subgroup reads its tiles straight from it as matrix loads (the cache serves the workgroup's other subgroups), with no staging and no workgroup barriers: 766 -> 309 ms, plus 5 ms for the re-layout.
    - Tried, no gain: summing the products in f16 (same 320 ms, so the tensor cores are not the limit), and 8 subgroups per workgroup with f16 scores.
    - The ResNet stem (7×7, stride 2, 35 -> 64 channels at 1008×560) ran on the scalar kernel: 87 ms for two calls. Convolutions with more than 9 taps now unfold their patches (`IM2COL`, channel = (input channel, tap)) and run as a 1×1 GEMM on tensor cores, in strips of rows that keep the patches under 256 MB: about 25 ms. SEA-RAFT's 7×7 convolutions use it too (42 -> 39 ms).
    - Left: the ViT's 1×1 GEMMs (token counts not a multiple of 8: 200 ms together) and attention itself (about 3.7 ms per 11,500-token call).
    - On the AMD iGPU (64-wide subgroups) the 1×1 GEMM shader had never compiled: its shape gave each thread 0 loads. The tile count now grows with the subgroup width. WAFT runs there and is correct (0.026 px), but it is far too slow to use (72 s for 336×224).
  - **Runner additions for these:** `GlobalAttention` (one-pass flash on tensor cores; scalar fallback), `PixelUnshuffle`, `Interp` sized from a second input, 1×1 GEMM on tensor cores for sizes not a multiple of 8, stride-2 convolutions on tensor cores (a 1×1 subsamples its input first; a 3×3 runs at stride 1, then is subsampled), unfolded large-kernel convolutions, and a `ChannelNorm` over 32 pixels × 8 channel groups per workgroup. `BRP_ATTN_NSG` sets the attention's subgroups per workgroup (tuning).
  - **Bug: device lost at 1600×900 and above.** The flow code let intermediate tensors drop while recording: the DINOv2 tokens and layer outputs, and the previous hidden state and flow each iteration. That freed their memory before the GPU ran. Small sizes survived because the allocator kept the memory blocks; at 1600×900 large buffers get their own allocations, which really were freed. Every temporary now goes back to the pool (`Ops::recycle`). `BRP_NN_SYNC=1` (waits for the GPU before each layer, `Recorder::sync`) found it: the failure went away with waits, but not with submissions alone.
  - **One pool:** the flow nets share the motion code's buffer pool (`Net::share_pool`). Each net had kept its own peak working set, so WAFT held 13 GB at 1080p; it now holds 2.9 GB.
- **Best motion blur and Slow Motion along the chosen motion** (`Temporal::interpolate`, `aicheck interp`). Best motion blur used to average RIFE's in-between pictures, so its Motion setting did nothing. Now it moves the two neighbouring frames along the chosen motion (Built-in, SEA-RAFT or WAFT) to each in-between moment, and RIFE's picture is used only where that motion proves wrong.
  - **How:** each frame's motion is splatted forward to the in-between time (integer atomics in 32-bit fixed point: AMD's iGPU has no float atomic add). Each pixel is weighted by its trust: colour match in the other frame, and the forward-backward motion check. Then both frames are sampled backward along that one motion (Catmull-Rom), weighted by how much of the motion came from each, which handles what only one frame shows. Pixels on a motion boundary (the motion changes within 3 px) are distrusted too.
  - **Where RIFE takes over:** where the two frames sampled along the motion disagree (mean difference 0.03 -> 0.10, mixed), and where the result is far from RIFE's (0.05 -> 0.15). The second matters most: a small object the motion missed shows as background in both samples, so they agree; RIFE places whole objects more reliably.
  - **Tests** (middle frame from its two neighbours, PSNR against the real one). The particle test now renders true in-between moments (`particles.py`, `TSTEP=0.5`); the real clips skip a frame, so their motion is doubled:

| Middle frame | RIFE | Ours + RIFE | SEA-RAFT + RIFE | WAFT + RIFE |
|---|---|---|---|---|
| Particles f2 / f4 / f6 | 33.2 / 32.9 / 32.6 | 33.8 / 33.5 / 33.2 | 34.5 / 34.0 / 33.7 | **34.5 / 34.1 / 33.8** |
| Stage f2 / f4 | 29.4 / 29.4 | **29.7 / 29.7** | 29.6 / 29.7 | 29.6 / 29.7 |
| Concert f2 / f4 | 28.9 / 27.0 | 28.8 / 27.0 | 28.9 / 27.0 | 28.9 / 27.0 |

  - On the particle test, split by area:
    - Background: AI motion 49.9 dB, RIFE 38.7. RIFE softens texture and grain, and the hybrid keeps them.
    - The particles themselves: all about 22.1 dB, RIFE's picture in effect.
  - **Ghost outlines (fixed).** With SEA-RAFT and WAFT, moving objects got thin outlines at their place in the other frame (an arm in the CRV clip, frames 405-406). The networks' motion blends both sides of an edge over a few pixels, and the soft edge pixels just outside an object get the background's motion. They stay behind as the object moves on and draw its edge. Two things made it visible, both now removed:
    - A depth test (what moves differently from the mean motion around it is in front) took the background next to a fast arm for foreground. It hid the other frame's clean background there. It had gained only 0.1 dB on particles.
    - Those edge pixels kept full trust. They are now distrusted (motion change within 3 px / 1 px divides the trust by e).
    - Measured on the strip above the arm against RIFE's clean background: the worst row was off by 4.3 levels, now 1.2 (the grain). The test scores are unchanged (within 0.03 dB).
    - Tried and dropped: taking one frame's sample instead of mixing where the two disagree. It gave jagged edges and speckled hands, and lost 0.3-0.5 dB.
  - Splatting alone was worse than RIFE on particles (16 dB against 22). SEA-RAFT and WAFT are still several pixels off on particles 8-20 px moving 3-15 px per frame. Splatting the colours of both frames put two half-transparent copies a few pixels apart; splatting the motion and sampling along it gives one.
  - **Slow Motion** makes its new frames the same way, with its own Motion setting: Built-in (the default), SEA-RAFT (AI), WAFT (AI), or RIFE only (the network alone, as before). Effects saved before this have no Motion value and get Built-in.
- **Scene cuts in motion blur and slow motion.** The merge (and so Denoise) already left out neighbours of another shot: their mean agreement after motion is below `CUT` (0.3). Motion blur and slow motion did not, so they blurred along, or blended between, two different shots at a cut.
  - `Temporal::same_shot` gives that measure for any pair: the neighbour is moved by the motion and its brightness matched per channel (gain from the pixels the motion filled; the warp leaves the others NaN), then the merge's agreement mean is taken.
  - The result stays on the GPU, in a 1-value tensor read by the kernels that need it: no readback, no stall.
  - **Fast motion blur:** a neighbour across a cut counts as absent (`MB_VEL`), so only the other side's motion is used, or none.
  - **Slow motion** (and best motion blur, built from it): across a cut the in-between picture is replaced by the nearer real one (`SHOT_PICK`), instead of a morph between two scenes.
  - Stage clip, frames 50–58 and 68–75 (`aicheck shots`): the two cuts score 0.03 and 0.04; every same-shot pair scores 0.75–0.94, a 10-frame jump 0.80. At the cut, the picked picture equals the real frame and the blur leaves it untouched (difference 0.0000).
  - A strobe flash or a large lighting jump within a shot also scores low (concert clip): the motion estimate fails there too (it tracks brightness), so leaving that pair out is right. Within a normal shot, nothing is flagged.
- **Models**: Real-ESRGAN (`realesrgan-x4plus`; `realesrgan-x2plus`, built by the `pth2ncnn` example from the official PyTorch weights into the x4plus graph with the input folded 2×2 into 12 channels; `realesr-animevideov3-x2/x3/x4`; `-x4plus-anime`). RIFE 4.26 and 4.25-lite (TNTwise/rife-ncnn-vulkan). RobustVideoMatting MobileNetV3 (GPL-3.0 weights), from the official ONNX: `onnxfix` fixes the input sizes and bakes the downsample ratio, then pnnx converts it. Recurrent: after a jump it replays the 4 previous pictures. Output: transparent, green or blue screen, or the matte. `tools/fetch_ai.sh` puts the models in `third_party/models` (not in git); the player build copies them next to the exe.
- **Natural 4×** (`realesr-general-x4v3`, Real-ESRGAN v0.2.5.0, BSD-3-Clause): the compact network (32 inner 3×3 convolutions, PReLU) trained on real-world pictures. The Fast models are `animevideov3`, trained on anime, and look flat and painterly on live action. `pth2ncnn compact` writes the graph and the weights from the official `.pth`. Its **Denoise** parameter mixes its weights with those of `realesr-general-wdn-x4v3` (same network trained without denoising) when the model loads, as Real-ESRGAN's `-dn` does: 0 keeps grain and fine texture, 1 is cleanest (0.5 by default). It runs at 117 ms per 1080p frame (8K out), against 66 ms for Fast 4× and 1.58 s for Best 4×. It matches the ONNX export of the same model to 57 dB.
- **Natural 2×** (`2xNomosUni_compact_otf_medium`, Helaman, CC-BY-4.0, OpenModelDB): the compact network at 2× (16 inner convolutions), trained on real-world pictures with JPEG, noise and blur. On live action it was the sharpest 2× in the survey and the fastest: 48 ms at 1080p input, about 80 ms in the player with the temporal window, against 380 ms for Best 2×. Only the medium variant is published (weak and strong are not), so it has no strength slider. Its `.pth` stores tensors channels-last; `pth2ncnn` now reads every tensor through its strides (the official Real-ESRGAN files were in plain order, so their conversions are byte-identical). The converter's compact mode takes the scale from the last layer (3 × scale² channels). Menu: Fast 2×, Fast 3×, Fast 4×, Best 2×, Best 4×, Anime 4×, Natural 2×, Natural 4×. Natural 2× and 4× are for real footage; Fast and Anime are made for drawn pictures.
- **Model survey** (ONNX exports from `notaneimu/onnx-image-models`, converted with pnnx, run by `aicheck custom`; the runner gained `Swish`). Speeds are at 1080p input:

  | Model | License | Speed | Result |
  |---|---|---|---|
  | NomosUni compact otf medium 2× (Helaman, compact) | CC-BY-4.0 | 48 ms | The fastest and sharpest 2× on live action, a little over-sharpened |
  | LiveAction V1 SPAN 2× | CC-BY-NC-SA, non-commercial | 120 ms | The most faithful; keeps JPEG blocking |
  | Nomos8k SPAN otf weak / strong 4× | CC-BY-4.0 | 120 ms | Strong is the sharpest on hair, with artifacts; weak is soft |
  | ClearReality V1 4× (SPAN) | non-commercial | 120 ms | Good detail, some blocking left |
  | PurePhoto SPAN 4× | — | 120 ms | Blurry |
  | LSDIR Compact v2 4× (compact) | CC-BY-4.0 | 57 ms | Soft |

  - Natural 4× at denoise 0 kept the most real texture of the 4× models, at a 13th of Best 4×'s cost.
  - Not supported: transformers (DAT2, DRCT, HAT, SwinIR) need attention layers. RealPLKSR needs large-kernel convolutions and its own normalization. The NomosWebPhoto ESRGAN export failed in pnnx; its `.pth` would go through `pth2ncnn`.
  - SPAN's Swish and attention multiplies are not fused into the convolutions yet: it runs at about twice the compact models' time for similar arithmetic.
- **Speed (RTX 5070 Ti, 1080p)**:
  - In the player: background removal ~7 ms per picture (real time); fast motion blur ~2.5 ms; slow-motion in-between ~27 ms; Real-ESRGAN fast 2× with the temporal window ~69 ms; best 2× ~0.43 s; anime 4× ~0.6 s; best 4× ~1.66 s.
  - Standalone (`aicheck upscale`): fast 2× 62 ms, fast 4× 69 ms, best 2× 0.38 s, anime 4× 0.54 s, best 4× 1.58 s, RIFE 25 ms. ncnn took 132 ms, 278 ms, 0.78 s, 1.09 s, 4.4 s and 374 ms.
  - Compared with others: the only figure found for this card was about 3 fps for x4plus with TensorRT FP16 from 960×540 to 4K (aivideoupscale.com, not a controlled benchmark). That input is a quarter of 1080p, where we take about 0.4 s (2.5 fps).
  - Timings drift by about ±5% between runs on this machine; compare variants side by side.
- **One effect stack**: AI effects (Upscale, Slow Motion, Motion Blur, Background Removal) are ordinary entries of the Effects list, in any order with shader effects. Their parameters show like a shader's ("(options: a|b)" in a parameter description makes a drop-down). Presets store them as `{"ai": "<name>"}`; `--fx=A.glsl,ai:matting:Output=1,B.glsl:Param=v` builds a stack from the command line.
- **How it runs** (`brp-player/src/ai.rs`): the stack is split at its last enabled AI entry. Everything up to it runs on a worker thread as stages; shader effects after it run live in the renderer on the result.
  - Pull model: each stage asks the previous one for pictures at a time in ticks of 1/subs frame (subs = product of interpolation factors, at most 64), so interpolation and motion blur work on whatever comes before them. The latest request wins.
  - A job records everything it needs into one command buffer. If it fails before submitting (a missing frame, an error), the cache entries it added are dropped again.
  - Caches, all in GPU memory: stage outputs as tensors (1.5 GB), motions (32), final pictures (1 GB). Working ahead is limited to what half of the picture cache holds.
  - **Deinterlacing comes first.** Interlaced sources are deinterlaced before the whole stack, with the same method, field order and field rate as the live path (`OfflineEffects` has its own deinterlacer). Each source frame is read with its neighbours, so every job asks for one more frame each way. With field rate each field is a picture: 2 per frame times the slow-motion factor (`Ai::subs`); the clock only follows slow motion (`Ai::slowdown`), so 29.97i plays at 59.94 pictures/s. The deinterlacer updates its descriptor sets on every run, so one recording uses up to 16 slots (each with its own sets and output images). The setting is passed every frame, since auto-detection may change it. `BRP_AI_DUMP=dir` writes each finished picture as a PPM, for debugging (it reads the picture back to the CPU).
- **Temporal upscaling** (Upscale's *Temporal* option, on by default): Real-ESRGAN invents detail from each frame alone, so grain and compression noise (different every frame) become shimmering detail. Each frame is first merged with its neighbours, moved onto it along the motion, and weighted per pixel against ghosting (as TAA does):
  1. Agreement over 5×5: the signed difference is averaged, so grain cancels while occlusions and wrong motion do not.
  2. A per-pixel test: the difference must stay within 2.5–5× the neighbour's typical (median) difference, which is grain when aligned. A misplaced thin detail is rejected even where the 5×5 average agrees.
  3. Values clamped to mean ± 1.25 sd of the current 3×3 (variance clipping).

  *Window* sets how many pictures are merged each way (1–5, default 2). Each picture weighs 0.85 of the one a step nearer (`DECAY`). Then the frame is upscaled. There is no state: every frame gets the same result whatever was shown before, and consecutive frames share most of their window.

  The merge works with any window size, in two passes. Pass 1 moves each neighbour and measures its agreement and typical difference. It keeps the moved picture and the agreement map, 4 channels per neighbour, so 200 MB at 1080p with 3 each way. Pass 2 (`MERGE_ALL`) is one kernel, one invocation per pixel. It computes the current picture's 3x3 statistics once, adds every neighbour in registers and writes the result. Before, pass 2 moved and measured each neighbour again and read and wrote the sums once per neighbour. Measured at 1080p with 3 each way on the integrated Radeon: 299 ms for adding and finishing became 55 ms, and the whole frame went from 1041 ms to 805 ms. The output is identical. What remains is about 43% motion estimation (2 motions per new frame), 24% measuring and 20% the DCT filter. A neighbour is used only if it and every nearer one on its side are of the same shot (mean agreement ≥ 0.3). The last usable one on a side also takes the weights of those beyond it (cut, or past the clip's end), so a side keeps its share. *Strength* blends from the frame alone to the full merge.

  **Grain against window size** (`aicheck temporal` with `BRP_PAN=1 BRP_GRAIN=0.03`, Natural 4×, flicker as mean |Δ| of consecutive outputs):

  | Each way | Grain left before upscaling | Flicker |
  |---|---|---|
  | 0 (off) | 0.0258 | 0.0057 |
  | 1 | 0.0127 | 0.0033 |
  | 2 | 0.0100 | 0.0030 |
  | 3 | 0.0088 | 0.0029 |
  | 5 | 0.0077 | 0.0027 |

  The merge removes grain as an ideal average would (0.0258/√11 = 0.0078 at 5 each way). With no grain the flicker is 0.0005: the upscalers are stable on clean pictures, and all the flicker comes from leftover grain. But the response is not linear: the models erase strong grain and draw faint grain as texture, so a third of the grain left still gives half the flicker. By model, at 2 and then 5 each way:
  - Best 4×: 0.0038 → 0.0034
  - Fast 2×: 0.0046 → 0.0036
  - Natural 2×: 0.046 → 0.035 (it keeps the most grain: 0.084 without the window)

  A thin moving line shows no ghosting at 5 each way (error beside it 0.0109 at 2, 0.0084 at 5, as grain falls). The merge at 1080p costs about 17 ms more at 5 each way than at 1, even with every motion recomputed.

  **Grain smoothing** (0–2, default 1; `SMOOTH`): after the merge, a 5×5 bilateral filter whose range is the slider × the grain measured on this picture (the typical difference to the nearest neighbours, from the stats buffer). Differences of grain size are averaged and stronger edges kept. On a clean picture it changes almost nothing (difference from the input 0.0009 → 0.0013, flicker unchanged at 0.0005). A thin moving line stays: at Window 5 the error beside it falls from 0.0084 to 0.0049, and on it from 0.0207 to 0.0162 (closer to the clean line).

  Natural 4×'s flicker grows roughly with the square root of the grain: grain 0.002 → flicker 0.0011, 0.005 → 0.0019, 0.01 → 0.0027, 0.03 → 0.0057. So halving the grain only cuts flicker by about 30%, and smoothing leaves the rest as soft blotches that still change. Flicker with grain 0.03:

  | Model | Window 2 | Window 5 | Window 5 + smoothing 1 |
  |---|---|---|---|
  | Natural 4× | 0.0029 | 0.0027 | 0.0025 |
  | Best 4× | 0.0038 | 0.0034 | 0.0030 |
  | Natural 2× | 0.046 | 0.036 | 0.019 |

  **Tried and left out: a second merge after upscaling** (the upscaled pictures merged at the output size, with the input motion scaled up; `aicheck temporal` with `BRP_POST=r`). With Window 5 and smoothing:
  - Natural 4×: 0.0025 → 0.0018, but 20% of the fine detail lost (mean gradient).
  - Best 4×: 0.0030 → 0.0019, 26% of the detail lost.
  - Natural 2×: worse (0.019 → 0.031), 41% lost.

  Opening up all its per-pixel tests changed nothing. The upscaled pictures already share most of their inputs, so averaging them adds little new information. A paused picture would also need its neighbours upscaled (slow with Transformer 4×).

  Motion: pyramidal Lucas-Kanade on luma with candidate testing (below), at half size above 1280 px wide. It is estimated only between consecutive pictures, chained outward and cached (32 kept). `BRP_RADIUS` and `BRP_DECAY` set the window in `aicheck temporal`; `BRP_LINE=1` measures ghosting of a thin moving line.

  **Motion pyramid.** Each halving uses the 5×5 binomial filter (1 4 6 4 1 each way). It used 2×2 averaging, so fine textures aliased at the coarse levels and seemed to move the wrong way (waves of 3–5 px moved by 1 px came out 4–6 px wrong in `aicheck lkcheck`).

  **Motion estimator.** Measured by what the merge depends on: the clean previous picture moved by the estimate, against the clean current one, on synthetic pans (`BRP_FLOW_ERR=1 BRP_FLOW_ONLY=1`). The error against the true motion overstates the problem: motion along a straight edge is invisible, so getting it wrong there costs nothing. Four changes, each from a failure seen in the error maps or the flicker:
  - *Candidate testing* (`CAND`), twice per level before the refinement steps: each pixel tries its own motion and those of the pixels 4 and 12 px away in 8 directions, and keeps the one whose 9×9 patch (every other pixel) matches best, by at least 3%. Regions a wrong coarse guess set off, and thin lines Lucas-Kanade cannot reach, take the motion of what is around them where it fits.
  - *Direction-aware filling*: refinement and filling carry each pixel's 2×2 structure tensor, not only its smallest eigenvalue. A line determines the motion across it and nothing along it; it used to count as undetermined, so its motion was skipped and then overwritten from the surroundings.
  - *Grain-aware solve*: the grain level (median high-pass of the luma) is taken out of the structure and added to the solve (`GRAIN_REG` 0.25 of what independent grain gives), so grain alone does not move the motion. Above the finest level a step also needs a least structure (`COARSE_FLOOR`): faint remains of a pattern the pyramid blurred away otherwise drove it (16 px waves moved 2.5 px came out 4 px wrong).
  - *Wide fill* (`WIDE_RADIUS` 32 at the motion's size), last: pixels without detail take the motion of the detail around them; detail keeps its own. What the merge needs in flat areas is not the right motion but the same motion from frame to frame: with the true pan plus a constant 0.25 px error, flicker is unchanged, while motion that wanders in flat grainy areas makes the leftover grain boil.

  Share of detailed pixels warped more than 8/255 off, old → new (with the wide fill, the grain cases fall a further 30% in mean error) (pictures 960×540 except the shirt, 1200×700; panned):

  | Picture | No grain, 1 / 3 / 6 px per frame | Grain 0.03, 1 / 3 / 6 px |
  |---|---|---|
  | Stage, person, curtain | 1.1 / 8.1 / 9.8% → 0 / 0 / 0% | 1.7 / 8.9 / 11.3% → 0.07 / 0.11 / 0.05% |
  | Glass on a table | 0.02 / 0.23 / 0.16% → 0 / 0 / 0% | 0.16 / 0.42 / 0.63% → 0.05 / 0.02 / 0.03% |
  | Thin lines on gradients | 31 / 33 / 35% → 7.7 / 7.1 / 3.5% | 32 / 35 / 36% → 27 / 26 / 23% |
  | Checked shirt, close up | 5.9 / 16 / 16% → 0 / 0 / 0% | 8.1 / 18 / 19% → 0.22 / 0.13 / 0.06% |

  Thin lines under heavy grain are the case left: only the lines carry motion and the grain covers them. Waves (`lkcheck`) are within 0.01–0.1 px at shifts of 0.3, 1 and 2.5 px, except where a shift is ambiguous (half a period, or more than a period of 3 px). Tried and left out: a slight blur of the luma first (no gain), a cost for candidates far from the pixel's own motion (worse on pictures, did not fix waves), and a candidate pass after the final smoothing (helps clean lines, worse with grain).

  Flicker with the merge (Window 2, grain 0.03, 1 px pan; perfect motion in brackets): Natural 4× 0.0029 → 0.0022 (0.0021), Best 4× 0.0037 → 0.0031 (0.0030). A moving thin line still shows no ghosting: the error beside it is 0.0103 at Window 2 and 0.0079 at 5 (was 0.0109 and 0.0084), and the line itself is kept better (0.0137 against 0.0207). Motion costs about 2.7 ms per pair of 1080p pictures (was 0.7 ms).

  **Tried and left out: the neighbouring pair's motion as a candidate** (`Temporal::motion_hinted`, `aicheck temporal` with `BRP_HINT=1`). It makes clean thin lines exact (7.7% → 0%) and helps them under grain (23–27% → 16–24%), but flicker did not move (thin lines with Natural 2×: 0.0115 → 0.0113; perfect motion 0.0084, out of reach there as every line is parallel and motion along them shows only at their ends). To keep each frame the same however playback reached it, the hint must be the plain motion of the next pair, so it doubles the motion work; the player does not use it.

  The CPU reference (`temporal.rs`) still has the old motion, so `aicheck gtemp` differs on motion.

  A first, recursive version (carry the previous output forward) was dropped: its result depended on where playback started. RIFE's internal flow was tried as motion and is unusable (wrong for 1–2 px motions).
- **Denoise** (AI stage, `brp-ai/src/denoise.rs`; put it before Upscale, or use it alone). Our own, in the manner of Neat Video, with nothing to set up:
  1. **Temporal:** the motion-compensated merge above (*Window* 1–5 each way, default 3). It also writes the noise left at each pixel as a fraction of one picture's: the current picture weighs (1 − s) + s/W and each neighbour s·k/W, so the variance left is the sum of their squares.
  2. **Noise profile, measured on every picture.** Where a neighbour moved onto the picture agrees with it (agreement ≥ 0.9 over a whole 8×8 block), their difference is noise, at twice its variance.
     - Its 8×8 DCT energies give the noise per frequency, per brightness level (8 bins) and per channel.
     - The channels are an orthonormal opponent space (brightness and two colour differences), in which RGB noise stays independent and transforms exactly.
     - Levels with fewer than 16 blocks take the nearest level's values.
     - It runs on the GPU: a thread per block, then a workgroup per (channel, frequency).
  3. **Spatial:** overlapping 8×8 DCT blocks of the merged picture, one every 2 pixels.
     - Each coefficient except DC is kept only if its square exceeds 2.7² × the expected noise: the profile × the noise left there × *Spatial* (*Colour* for the colour channels).
     - Each pixel is the weighted mean of its 16 blocks, with weight 1/(1 + coefficients kept): blocks that kept less are smoother, so more reliable.
     - A workgroup makes a 16×16 tile from the 11×11 blocks covering it, one channel at a time, in shared memory.

  Hard thresholding beat the Wiener gain c²/(c² + N): with Natural 2× and Window 5, flicker 0.015 against 0.029.

  **Results** (synthetic pan with grain 0.03, flicker):

  | Model | Window 5 + Grain smoothing | Window 5 + Denoise |
  |---|---|---|
  | Natural 2× | 0.019 | 0.015 |
  | Natural 4× | 0.0025 | 0.0026 |
  | Best 4× | 0.0030 | 0.0033 |

  Without grain it changes the picture less than Grain smoothing does (0.0012 against 0.0015 from the clean picture). Visually it cleans flat areas well.

  Flicker in the models that remove grain themselves (Natural 4×, Best 4×) barely moves. What changes from frame to frame there is in textures and edges: grain inside texture (no spatial filter can tell it from texture) and motion errors. With the exact motion, Denoise brings Natural 4× to 0.0020.

  Cost per 1080p frame: about 5 ms for the profile and the filter, 13–17 ms in all with the extra moves of the nearest neighbours (fastest frames; times vary as the GPU changes clock between frames).

  **Tried and left out:** a merge after upscaling lowered flicker (Natural 4×) to about 0.0018 but lost 20–26% of the fine detail and made Natural 2× worse; see the grain-smoothing notes above.

  **Real footage (the merge's clamp, fixed).** On the stage clip (`footage/CRV CHA EITA LUCAS BASE.mov`, the shot of frames 55–71, a 960×540 crop at full resolution) the window size made almost no difference, little noise went and what was left flickered. The cause was the clamp of each neighbour to the current picture's 3×3 mean ± 1.25 deviations: that mean carries the current picture's coarse noise (colour blotches of 3–6 px), so the merge kept it whatever the window, fresh every frame, and the spatial filter, aiming at a fixed noise level, evened the windows out. The clamp is now widened by twice the typical difference between the pictures (`CLAMP_NOISE`): changes from noise pass, real changes are still held. No ghosting on the moving line or the performer's fist.

  Measured with `aicheck temporal` and `BRP_STEADY=x0,y0,x1,y1` (no reference needed): the change from the previous output moved along the motion, and the fine noise, in a rectangle. Curtain, frames 60–66 (input change 0.027):

  | Window | Merge, old clamp | Merge, new | Denoise, new |
  |---|---|---|---|
  | 1 | 0.0137 | 0.0114 | 0.0088 |
  | 3 | 0.0113 | 0.0070 | 0.0054 |
  | 5 | 0.0108 | 0.0057 | 0.0045 |

  Against the user's earlier TAA denoiser (history carried forward along the motion, clamped to the blended 3×3 min/max in YCoCg, per-pixel rate 0.05–0.5; `BRP_TAA=k` in `aicheck temporal`), Window 5, all of the shot's history for TAA: on the static curtain Denoise is steadier (0.0042 against 0.0061, fine noise 0.0017 against 0.0033); on the moving performer TAA is (0.0119 against 0.0147). A cut 17 frames apart gives TAA little history there; the window also uses the pictures after.

  **Moving areas.** On the performer the window merges about 6.6 pictures' worth at Window 5, against 10 on the curtain (`aicheck` prints it with `BRP_STEADY`, and saves the map as `_fac`): the checked shirt, the face and the white T-shirt change shape and shading between pictures, so their neighbours are rightly left out. TAA's lower change there is smearing, not steadiness: the checks of the shirt come out dragged and doubled, old detail held in place, while the window keeps them sharp. Tried and left out: refining each chained motion against its far picture at full detail (merged amount 6.63 → 6.69, change 0.0152 → 0.0159: the chained motion was not what limited it).


- **Playback**: with AI effects in the stack, every picture (frame or in-between) is shown in order after the whole stack, never skipped. The playhead moves to the next picture only once its processed version is ready: real time when processing keeps up, slower otherwise (it carries on from the late picture's time rather than catching up). The worker makes pictures in order from the playhead; the decoder reads 60 frames ahead. Audio plays while processing keeps real time (it joins once the next 0.4 s of pictures is ready) and stops when playback falls behind. Matting keeps its recurrent state across small gaps.
- **Names**: the panel shows plain names (Upscale, Slow Motion, Motion Blur, Background Removal) and Fast/Best choices, never model names.
- **Slow motion**: the clip gets k times longer, and each new picture lasts a whole frame. The frame counter, timeline and timecode count output frames; In/Out stay in source frames. Audio is off in slow motion.
- **Motion blur**: *Fast* (default) uses the cached motion to the previous and next frames and gathers along it, as MB.glsl does (McGuire-style: 16 px tile max, neighbour max, cone/cylinder weights, faster counts as nearer, fixed per-pixel jitter). *Best* makes the in-between moments over the shutter with frame interpolation and averages them (one per 4 samples, ~0.2 s per frame); it handles motion too fast for the vectors.
- **Tools** (`brp-ai/examples/aicheck.rs`, GPU only): `upscale`, `rife`, `matting`, `flow`, `temporal`, `gtemp` (GPU temporal vs the CPU reference) and `vkop` (one layer).
- **Limits**: Large upscalers are not real time; playback slows down to the processing speed. When paused, the decoded frame shows until the result is ready.


### Transformer upscalers: SwinIR-M ("Transformer 4×")

SwinIR-M real-world ×4 GAN (JingyunLiang/SwinIR, Apache-2.0) runs on our runner. It has 180 channels, 6 groups × 6 blocks, 6 heads, 8×8 windows and a nearest + conv upsampler. pnnx's conversion was unusable: 851 layers of Reshape, Permute and Slice chains with shapes known only at run time, and DAT2 crashed it. So `pth2ncnn swinir` writes the graph itself: 468 layers on our `[c][h][w]` tensors, with no reshapes.

- **Layers.**
  - ChannelNorm: LayerNorm over the channels of each pixel.
  - WindowAttention: one fused layer per attention. It reads the qkv projection and does window partition, cyclic shift and shift mask by index arithmetic.
  - GELU: op 9, exact erf.
  - Linear layers: 1×1 convolutions.
- **1×1 convolutions: `CONV1X1_COOP`.** On f16 tensors, a 1×1 convolution is a plain matrix product, with the planar input as the right-hand matrix.
  - Each step reads 32 input channels × 32 pixels in 16-byte pieces and stages them pixel-major in shared memory: a pixel's channels side by side, the layout the tensor cores read fastest. Staging row-major instead was 2× slower.
  - The output step writes 8 pixels of a channel in one 16-byte store.
  - Small workgroups won (2 subgroups × 16 pixels × 64 channels, `GEMM_SHAPE`, swept with `BRP_GEMM`). Larger ones hit the register limit: 16 accumulator tiles spilled about 1 KB per thread, and 256 threads × 234 registers left one workgroup per multiprocessor.
  - Other cases (f32 input, ≤ 32 output channels, misaligned tensors) use `CONV_COOP` with `KS` = 1: one tap, no border.
- **Fusion.** The load-time fusion into tensor-core convolutions now also takes:
  - a following GELU, as the convolution's activation (act 9 in `ACT`);
  - a following BinaryOp add of two tensors, like an Eltwise sum (not a broadcast MemoryData constant);
  - this helps Best 4× too (1.70 → 1.60 s per 1080p frame; output within 58.5 dB of before, fp16 rounding).
- **f16 in and out** for ChannelNorm and WindowAttention, so no conversion copies around them.
- **Window attention on tensor cores** (`WINDOW_ATTN_COOP`, for windows of 16 to 64 tokens).
  - One workgroup per (window, head), one subgroup per 16 query tokens.
  - Q, K and V are gathered into shared memory as fp16, with the head's channels padded to 16.
  - S = QKᵀ and O = PV are fp16 matrix products with f32 sums. S is kept in fp16, and the softmax runs in place on it. O goes over the subgroup's own Q rows. That makes 20 KB of shared memory, against 36 KB with f32 scores: more workgroups at once, 28 → 15–18 ms.
  - Softmax is f32, with subgroup reductions; the position bias and shift mask are added there.
  - The scalar kernel remains for other window sizes and GPUs without cooperative matrices.
- **fp16 range.** The residual stream of the last group reaches 2.4·10⁵ (its closing 3×3 convolution turns inputs of about 3·10³ into that). fp16 overflowed (27 dB). Our own param key `30=1` keeps a layer's output f32. The converter sets it on each group's closing convolution and the add after it; fusion requires both layers to agree on it. fp16 *sums* also overflow, so SwinIR always sums in f32.
- **Accuracy.** Compared with ONNX Runtime on the CPU (`onnxref` crate; the ONNX export accepts only 64×64 input): 62.7 dB in f32, 61.5 dB with fp16 activations. Tiled against whole picture: 55–56 dB.

| SwinIR-M, RTX 5070 Ti, 256×256 input | Time |
|---|---|
| First version (scalar 1×1 and attention, f32) | 823 ms |
| 1×1 convolutions on the 3×3 tensor-core kernel | 373 ms |
| Attention on tensor cores | ~240 ms |
| fp16 activations | ~170 ms |
| GELU and adds fused, f16 norm and attention (no copies) | ~129 ms |
| Attention scores in fp16 (20 KB shared) | ~115–125 ms |
| `CONV1X1_COOP` | **~77 ms** (1×1 about 50 ms, attention 15–18 ms) |

For comparison, the SwinIR README gives 0.539 s for a 256×256 input (PyTorch; GPU not stated).

- **In the upscaler.**
  - Tiles are a multiple of 8 (`UpscaleModel::align`), with a 32 px border.
  - Memory: 8.6 KB per network input pixel measured, budgeted at 10 KB (mostly the 4× upsampler's tensors). So tiles are 512 px for 1080p, using 2.9 GB.
  - A tile takes about 1.2 µs per input pixel (0.4 s for 512 px). `Net::set_flush_every` splits it into GPU submissions of about 0.1 s, so slower cards stay under Windows' 2 s reset too.
  - A 1080p frame takes **3.4 s** at 4× (was 7.2 s; Best 4× ESRGAN: 1.6 s). This is for paused frames and export, not playback.
- **Next.**
  1. Interp (the upsampler's nearest resize) in f16: fewer copies and less memory at the output size.
  2. HAT and DRCT (Swin plus channel attention, dense connections).
  3. RealPLKSR.

### DAT2 ("Transformer HQ 4×")

4xNomos2_hq_dat2 (Philip Hofmann, CC-BY-4.0; DAT architecture, Chen et al. 2023, Apache-2.0), trained on clean pictures. `pth2ncnn dat` reads the author's ONNX export directly: it kept the PyTorch names, so no `.pth` is needed. It writes 856 layers (43 MB).

- **Architecture.** 6 groups × 6 blocks, 180 channels. Blocks alternate:
  - *spatial*: window attention on two halves of the channels, 8×32 and 32×8 windows (256 tokens), 3 heads each, some blocks shifted by half a window;
  - *channel*: attention between channels (30×30 per head) over the whole tile.

  Each runs beside a depthwise 3×3 convolution of v; learned gates mix the two (a channel map from a global average, a per-pixel map). The feed-forward is gated (half of it normalized, depthwise-convolved, multiplied with the other half).
- **Conversion.** BatchNorms are folded into the layers before them. The position bias MLPs run in the converter: their inputs are constants, which gives Swin-style tables. The export deduplicated identical buffers (only block 0 keeps `rpe_biases`), so the converter generates the rows and checks them against block 0. The input mean is read from the graph: neosr uses 0.5, the original DAT the DIV2K mean. Assuming DIV2K's gave 47 dB.
- **New layers.**
  - `DATAttention`: rectangular windows, per-branch tables, masks across the shift.
  - `ChannelAttention`: Gram matrix per pixel chunk, then softmax, then apply. Fixed-order sums, so runs are deterministic.
  - `AIM`: the gates.
  - `ChannelView`: zero-copy channel slices.
  - Global average in two stages.
  - Depthwise 3×3 (`DW3`); grouped convolutions now read and write f16.
- **Window attention on tensor cores** (`DAT_ATTN_COOP`). 256-token windows are too big to keep all scores, so each subgroup takes 16 queries and streams the keys in blocks of 32: one pass for each row's max and sum, a second for P·V. K and V take 32 KB of shared memory; each subgroup needs only 1 KB (Q, then S and P in place, then O, all fp16), so 16 subgroups fit, one per 16 queries. The count follows the device's shared-memory limit (16, 8, 4…). The scalar kernel (running softmax) is the fallback and the reference.
- **Accuracy** (ONNX Runtime, CPU): 75 dB in f32, 73 dB with f16 activations (64×64 and 128×96 inputs).
- **Speed** (RTX 5070 Ti, 256×256 input, whole picture):

| Step | Time |
|---|---|
| First version (scalar attention, 1-thread-per-channel average) | 345 ms |
| Window attention on tensor cores (200 → 30 ms) | 162 ms |
| Two-stage global average (29 → 5 ms) | 145 ms |
| Channel-attention sums on tensor cores, DW3 kernel | 126 ms |
| Channel-attention apply on tensor cores (7.7 → 2.6 ms), softmax sums in parallel, gate multiply inside the depthwise kernel, shift mask only in the last window row and column | 112 ms |
| Window attention with 16 subgroups (26 → 19 ms) | **107 ms** |

  A 1080p frame takes about 4.9 s at 4×, warm (SwinIR 3.5 s; about 5.9 s before the last two steps), in 512 px tiles, 4 GB.

  Tried and left out:
  - fp16 sums in the 1×1 layers: slower here (139 vs 127 ms), and the Restore model's sums overflow.
  - A one-pass attention (running maximum, O rescaled in shared memory): only 8% faster at 4 subgroups, and its f32 O leaves room for no more.
  - The position bias lookup costs nothing measurable; the mask about 3 ms (now skipped where it cannot apply).
- **Tiles.** Channel attention and the gates average over the tile, so tiled output differs from the whole picture: 40 dB at 256 or 128 px tiles (SwinIR 55 dB). The difference is grain-level texture spread over whole tiles. At 1:1 no seam is visible.
- **Is it working?** It looked like it barely changed the stage clip. Checks: shrinking a sharp test pattern 4× and upscaling it back, it rebuilt the fine checkers and noise texture almost exactly, where bicubic blurred them and SwinIR wiped them out. On a 1024 px crop of `a.png`: bicubic 34.6 dB, DAT2 HQ 36.2, SwinIR 30.1, Best 4× 30.3. The network works; it was trained on cleanly shrunk pictures. Real video is softer than that (lens, compression, noise), so it adds almost nothing and keeps the flaws. With Upscale off (scaled back to size) the result is nearly the original.
- **Transformer Restore 4×**: 4xRealWebPhoto_v4_dat2 (same author and license), the same network trained on degraded pictures (blur, noise, JPEG/WebP, rescaling).
  - Its ONNX export folded the BatchNorms and renamed the weights (`onnx::Conv_…`), so `pth2ncnn dat` also reads `.safetensors` (header parsed without a JSON library; F32/F16/BF16). The input mean is the DIV2K one, given on the command line. Unique-shape tensors match the ONNX exactly (`pth2ncnn wcmp`).
  - Against ONNX Runtime: 64 dB at 64×64, 59 dB at 128×96, the same in pure f32 without tensor cores. The differences are tiny and spread evenly (below 8-bit rounding). Its residual stream reaches 4.3·10⁵ (the HQ model's far less), which likely accounts for both runners' rounding.
  - LayerNorm (`CHANNEL_NORM`) now subtracts the mean before squaring (E[x²] − mean² cancels in f32 at such magnitudes). This was not the cause here, but it is safer.
  - On the stage clip it removes the compression blocks and noise and keeps shapes true (the checked shirt stays as filmed). It adds little new fine detail. SwinIR stays the crispest, with invented geometric edges.
- **Next:** the 1×1 layers are now the largest part (about 46 ms of 107: qkv, proj, fc1, fc2), already on the shared GEMM kernel; fusing LayerNorm and the gates into their input staging would save passes over memory.


### Startup (release build, this machine)

| Step | Time |
|---|---|
| Window visible | 30–60 ms |
| Vulkan instance (driver load, on a background thread from process start) | 80–120 ms |
| Device | ~40 ms |
| Swapchain creation (driver) | 150–250 ms |
| **First frame with video** | **~380–410 ms** (NVIDIA), ~435–475 ms (AMD iGPU) |

Target was < 300 ms. The remaining time is inside the GPU drivers. Next steps:
1. Create the logical device on the background thread too (needs only a late present-support check).
2. Investigate presenting through a DXGI flip-model swapchain on Windows (Vulkan renders into a shared image); DXGI swapchain creation is typically ~10–20 ms.
3. Paint the window background immediately so it never shows blank while the driver starts.

Even now, launch-to-video is well under half a second, so the old resident server / tray workaround is not needed.

### Shader cache and start-up build

Adding an effect used to wait for its shaders. Measured on first use (RTX 5070 Ti):

| Effect | Shaders | Compile time |
|---|---|---|
| Transformer Restore 4× | 44 | 2.4 s (first run 1.3 s instead of 0.2 s, plus 0.6 s loading) |
| Denoise | 63 | **37.5 s**: two kernels, the 8×8 DCTs, took 22 s and 15 s |

Almost all of it is GLSL to SPIR-V in shaderc, one kernel at a time. The driver's own step took a few ms in total, because NVIDIA caches it on disk itself.

- **The DCT kernels** were fully unrolled (64-value arrays, 512 multiply-adds each way). spirv-opt's scalar passes choke on that. Without spirv-opt they compile in 2 ms each (the driver needs 0.3–0.4 s once), and Denoise runs at the same speed (45–49 ms per frame either way). Without spirv-opt everywhere, frames were 2–3 ms slower, so only these two kernels skip it, marked by `NO_OPT` in their source. Denoise's first use: 37.5 → 0.8 s.
- **SPIR-V on disk** (`shader_cache`), keyed by a hash of the source and compile settings. Every shader goes through it: AI kernels, the motion/denoise kernels, effect shaders. A shader is compiled again only when its source changes. Entries are marked when used; those unused for 60 days are removed after a warm-up (earlier builds' versions).
- **The driver's pipeline cache** (`pipelines.bin`) is created with the device and saved after the warm-up and when the context goes. Its header (vendor, device, cache UUID) must match, so another driver's cache is ignored: a driver update rebuilds it. 2.2 MB, read in about 1.3 ms at start-up.
- **Start-up build** ("Compiling shaders… n/m" on the controls bar and in the effects panel). One second after the window opens, a background thread builds the shared objects (operations, motion, denoiser: they compile everything when made) and records each network once on a blank 64 px picture. Each network compiles its own kernel variants while it records; the recording is dropped unsubmitted. Fresh: 85 shaders in 4.5 s (most of it compiling, the rest loading every model). After that, a real first use compiles nothing: Restore 4× 0 of 44 (first run 0.21 s), SwinIR 0 of 41, Best 4× 0 of 36, Denoise 0 of 63.
- **When it runs:** `warm.stamp` records this executable (path, size, time) and the GPU with its driver.
  - Same build and driver: nothing runs.
  - A new build: everything is checked, but only changed shaders compile (unchanged ones are hits: 0.7 s).
  - A new driver: the driver's step is redone.
- `BRP_SHADER_CACHE=<folder>` uses another folder, `off` turns caching off. `BRP_WARM_UP=1` forces the build. `aicheck warm` runs it with counts; `BRP_CACHE_STATS=1` prints compiled/cached counts after any `aicheck` command.
- **Effect shaders (`.glsl`)** are built at start-up too, every run: each file in the shader folders, with its default parameters, for each colour transfer (SDR, PQ, HLG, linear: the generated code depends on it). Unchanged files are cache hits; new or edited ones compile. The 24 bundled files make 1,180 SPIR-V modules: 9.1 s on one thread. Each thread now has its own shaderc compiler (it was one behind a lock), and the build uses all cores but two: 1.7 s (22 threads); 0.07 s from the cache (`fxprecompile` example). A parameter that a shader uses as a constant still compiles when it changes (once per value, then cached).
- **Progress bar** ("Compiling shaders… n/m") on the controls bar and in the effects panel, shown only once the build has taken over 0.3 s (a run that finds everything cached shows nothing). AI kernels follow the effect files, when needed.

**Changing a setting** (AI stack, background worker) compiles nothing: no AI kernel's source depends on a setting. The wait came from redoing work:
- Every stage's results were dropped, with all motion and matting, so changing the last effect (an upscaler's Grain smoothing) redid the ones before it (Denoise). Now each stage's output is keyed by a hash of what it depends on (the stages up to it, the colour transfer, deinterlacing, pictures per frame). Results, motion and matting of the stages before the first changed one stay.
- A shader stage built a new effect runner on any change, compiling all its passes again. It now updates the existing runner, which recompiles only a shader whose parameters are built in as constants.
- While paused, the unprocessed frame showed until the new picture was ready: the held picture was dropped on every change. It now stays on screen until the new one replaces it (same frame only; during playback, as before).

### Export and batch export — working

**One engine, the player's.** An export runs the whole enabled effect stack (shader effects, AI stages, deinterlacing) through the same worker the player uses for its AI path (`ai::Worker`, `export.rs`): the same stages, caches, scene-cut handling and recurrent state, so each exported picture is the picture the viewer shows. Shader effects after the last AI entry, which the window runs live, become one more stage (`Effects::all_stages`; same `EffectChain` code). The worker gained `run` (set up, record, submit, wait), `export` (a picture packed for the encoder), `warm` and `reference`.

- **Decoding:** the export opens its own frame-exact `Source` (its own decoder thread, cache capped at about 48 frames with `Source::set_cache_limit`, not the viewer's 70% of RAM), waits for the index, and asks for each picture's window of source frames (`ai::Reach`: deinterlace neighbours, temporal windows, matting replay, interpolation).
- **On the GPU to the end** (`brp-ai/src/pack.rs`): the finished RGBA tensor is scaled (separable Catmull-Rom, widened when shrinking; premultiplied alpha), placed on the output canvas (fit with bars, fill with crop, stretch), converted when needed (HDR→SDR with the viewer's tone mapping, linear→gamma 2.2 as the viewer shows linear files, →linear for EXR), and written straight into the encoder's pixel format: any planar, semi-planar (NV12/P010) or packed layout, 8–16-bit, half or full float, any chroma subsampling, Y'CbCr with the stream's matrix and range (`encode::pack_layout` reads FFmpeg's pixel-format descriptors). One thread per 32-bit output word. Only those bytes come back. It rounds through half float first, as the window's pictures are, so export and viewer agree bit for bit. Formats it cannot write (big-endian, bit-packed, palette) are written in a wide format and converted by swscale, so every FFmpeg encoder stays usable.
- **Encoding** (`brp-media/src/encode.rs`, `Writer`): libavformat/libavcodec on an encoder thread fed by a channel of 3 pictures (GPU and encoder overlap). Rate control per encoder family (CRF, NVENC CQ, AMF CQP, QSV global quality, qscale, CBR for XDCAM). Colour tags, sample aspect ratio and the display rotation are written. Audio is read from the source, cut to the exported range by libavfilter (`atrim`, silence where the source has none, `apad`/`atrim` to the exact length), retimed when the speed changes (`atempo` keeping pitch, or `asetrate` shifting it), converted and encoded alongside (AAC, PCM, FLAC, Opus…). GIF goes through a per-picture palette (`palettegen`/`paletteuse`). A cancelled or failed export removes what it wrote.
- **Codecs:** the panel offers families (H.264, HEVC 10-bit, AV1, VP9, ProRes, DNxHR / DNxHD, XDCAM HD422, FFV1, Ut Video, CineForm, QuickTime Animation, HAP, DXV3, MJPEG, GIF, PNG/JPEG/TIFF/EXR/DPX sequences), each with software and hardware encoders. Hardware encoders are probed once in the background (opened on a small picture); "GPU" picks this card's vendor first; for ProRes, DNx and DXV3 it is our own GPU encoder (see "GPU encoders" below). "Other" lists every video encoder of the bundled FFmpeg that the container takes. Size multiples per format (HAP 4, CineForm width 16, 4:2:0 even); sizes are rounded down with the few pixels cropped, centred. Source rates like 2997/100 are written as 30000/1001 (MXF takes only standard fractions).
- **Output size:** "Same as the effects" (after an upscale: the upscaled size; anamorphic sources keep their pixel aspect), presets (4K, QHD, 1080, 720, PAL, NTSC, vertical, square), custom W×H (0 = from the aspect), or a percentage; Fit / Fill / Stretch.
- **Timing.** The stack makes `subs` pictures per source frame (fields × slow-motion factors) at `stack rate` = frame rate × fields. Output frame *i* maps to a stack position (`Timing`): its time *i*/rate is turned into source time through the frame index (exact for variable frame rates), slowed by slow motion, into a fractional tick. Then:
  - *Same rate* or *Retime*: every picture is one frame (Slow Motion makes the clip k times longer; field-rate deinterlacing gives 50i → 50p). A different rate changes the speed; audio follows with or without pitch.
  - *Repeat / drop*: the picture on screen at that time (25 → 50 repeats each picture, 29.97 → 25 drops one in six).
  - *Blend*: the two nearest pictures mixed by the fraction.
  - *Motion (AI)*: the picture is made at the fractional tick by the same code as Slow Motion (`picture_at` on the stack's output: motion splatting with RIFE where the motion fails, scene cuts respected), with the stack's own Slow Motion settings if it has one.
  - Interlaced sources are deinterlaced as in the player (Auto detection from content; field or frame rate). Batch items decide it from their own file (picture analysis over the first ~125 frames). Stateful shader effects get their warm-up pictures before the first exported one.
- **Presets:** 19 built-ins (H.264 delivery / GPU / web, HEVC 10-bit keeping HDR, AV1 WebM, ProRes 422 HQ, ProRes 4444 alpha, DNxHR HQX MXF, DNxHR HQ MOV, XDCAM HD422 MXF, FFV1 lossless alpha, VP9 alpha, QuickTime Animation alpha, HAP Alpha, DXV3, DXV3 alpha when available, PNG alpha, EXR linear, GIF 480 px 15 fps). User presets (named, with every setting) and the last settings used are saved in `settings.json` (`export_presets`, `export_last`, `batch_dir`).
- **UI** (reworked twice after user feedback). Two title-bar buttons, each with a ring of progress while it runs:
  - **Export** (one file): Preset (with Save… / delete), Format (one choice for codec and container, e.g. "H.264 / AVC · .mp4"), Quality as named levels (Draft, Good, High, Best: CRF/CQ values per codec) or the codec's profiles (ProRes, DNxHR) or "Lossless", then **toggles that appear only where the codec supports them**: Alpha, HDR (HDR sources, 10-bit codecs; off or an SDR-only codec: tone mapped to SDR), GPU acceleration (when a hardware encoder works here; not with interlaced output), Audio. Then Resolution (+ Fit/Fill/Stretch for a fixed size), **Frame rate with p/i** (23.976p … 60p, 50i, 59.94i, 60i; "i" counts fields per second and appears only for codecs that store interlaced video), and, when the rate needs pictures the stack does not make, **"New pictures by"**: Motion (AI) by default, Repeat / drop, Blend, or Retime (speed change). Then In–Out range (when set) and File. The pixel format and encoder follow from the codec and the toggles. A collapsed **Advanced** section holds container, encoder, field order (top or bottom first, when interlaced), exact quality or bit rate, pixel format, audio codec and bit rate, audio speed (pitch), extra encoder options. Summary line (size · rate · frames · length · speed) and Export / Cancel with progress and ETA.
  - **Batch export** (its own workflow, laid out like the original app's batch dialog): the files on the left (Add files…, add current, drop on the window; click, Ctrl-click, Shift-click to select; Select all / none; each row shows its format and rate, status, progress, error, time and Show), and on the right **the same settings panel as Export**, editing the selected files; **with none selected it edits every file in the list and the files added later** (saved as `batch_defaults`). A note warns when the files it edits have different settings, and each row shows its format, frame rate (p/i) and size. (Fixed after a user report: the panel used to change only the settings of files added later, so a file already in the list exported at its old progressive rate while the panel showed 59.94i.) With a selection, "Use these settings for all files" gives the rest the same. Below the panel, for all files: apply the current effects (saved), output folder (or next to each file). Overall progress, Start (n), Stop, Skip; a failing file does not stop the queue. Every file is exported with its own settings.
  - English and Portuguese. `BRP_TEST_EXPORT_ADVANCED` opens Advanced; `BRP_TEST_BATCH`, `BRP_TEST_BATCH_SELECT=n`, `BRP_TEST_BATCH_DIR`, `BRP_TEST_BATCH_START` fill, select and start the batch (screenshots).
- **Interlaced output.** A target like 59.94i or 50i is 2 × rate pictures per second, each a field: output frame *i* takes the stack's pictures at fields 2*i* and 2*i*+1 (their own moments, through the same timing and the same "new pictures by" choice), and the GPU weaves them (the first in time on the top lines for top field first, on the bottom lines for bottom field first) before packing; interlaced 4:2:0 chroma averages lines of one field only. The encoder gets `field_order` (TT/BB), interlaced DCT and motion estimation, and every frame is flagged interlaced with its field order, so the codec and the container (MOV `fiel`, MXF, MP4/H.264 SEI) say so. Codecs: H.264 (x264), XDCAM HD422, ProRes, DNxHD (1080i CIDs), FFV1, and "Other"; DNxHR is refused by this FFmpeg, VP9/AV1/HEVC are not offered. An interlaced source deinterlaced at field rate and exported at its own field rate (29.97i → 59.94i) re-interlaces with no new pictures.
- **Command line** (also the test tool): `brp-player <file> --export=<out> [--preset=… --container=… --codec=… --encoder=… --profile=… --pix-fmt=… --quality=… --kbps=… --alpha --audio=… --pitch=keep|shift --size=WxH|50%|source --fit=… --fps=25 --conform=nearest|blend|motion|retime --tonemap --range=a-b --deint=… --field-rate=0|1 --fx=…]`; `brp-player <files…> --batch-dir=<folder> [same options]`; `brp-player <file> --verify=<exported> [the export's options] [--frames=…]` decodes listed output frames of an export and compares them with the picture the player makes for that moment, decoded through the player's own YUV→RGB path, and checks frame count, rate and size. `BRP_TEST_BATCH`, `BRP_TEST_BATCH_DIR`, `BRP_TEST_BATCH_START` fill and start the batch window; `BRP_TEST_CANCEL_AFTER` cancels a command-line export; `BRP_TEST_PLAY_AFTER=secs` presses play that long after opening (with `--frame=n`: timing how playback starts; `RUST_LOG=brp_player=debug` logs when each picture is shown).

**Tests** (RTX 5070 Ti; `--verify` against the player's pictures; levels are 8-bit):

| Case | Result |
|---|---|
| `test.mp4` → H.264 MP4 (defaults) | 1920×1080, 30/1, 74 frames; vs viewer 43–47 dB (x264 CRF 18, 4:2:0) |
| Packing exactness: FFV1 `gbrp16le` / `yuv444p16le`, PNG `rgb48be` (swscale path) | max 0.02 / 0.12 / 0.02 levels (0.12 = the viewer's own half float); `yuv444p` 8-bit: max 1.7 (quantisation) |
| CRV clip, shader → AI Denoise → AI Upscale Fast 2× → Film Grain, FFV1 16-bit | 3840×2160, 8 frames, max 0.12 (exact) |
| Slow Motion 2× + Motion Blur, 16 frames, FLAC audio | 32 frames at 29.97, audio stretched to the same 1.068 s, exact |
| Background Removal (transparent) → PNG sequence | RGBA, exact within 8-bit (matting replayed in order, as playback) |
| PASSAGEM (alpha) → ProRes 4444 | `yuva444p12le`, alpha compared too: 60 dB |
| **29.97 → 25** (CRV, 71 frames, repeat/drop, ProRes + PCM) | 59 frames at 25/1, video and audio both 2.360 s; frame 19 = source 73 (one in six dropped) |
| **25 → 50** (that 25p file, 59 frames) repeat / blend / motion, FFV1 16-bit | 118 frames at 50/1 each; repeat 0,0,1,1…; blend and motion in-betweens at x.5; all exact vs the player |
| `interlaced_prog.mp4` (flagged progressive): Auto → NNEDI3 + motion-adaptive, field rate | detected interlaced, 30 frames → 60 at 59.94, exact; conformed to 25: 25 frames, exact |
| `interlaced.mp4`, bwdif, frame rate | 30 frames at 30000/1001, exact |
| Batch: 4 files (one missing) to H.264 1280×720, command line and GUI | 3 done (the interlaced one 245 → 490 frames at 59.94), the missing one failed alone; GUI shows per-item progress and the title-bar ring |
| Codec sweep (15 frames each) | h264_nvenc, hevc_nvenc 10-bit, DNxHR HQX MXF + PCM 24, XDCAM HD422 MXF at 25, DXV3, HAP Alpha, VP9 alpha (`alpha_mode=1`, alpha kept), QuickTime Animation `argb`, AV1 (SVT), EXR half float RGBA, GIF, HEVC 10-bit HDR (tags kept; also on NVENC), HDR → SDR (HDR off, or H.264 which cannot hold HDR), MJPEG AVI, Ut Video `gbrap`, CineForm |
| Fit / Fill / Stretch to 600×600, 1001×auto, 50% | bars / crop / distort correct; 1000×562; 960×540 |
| User preset in `settings.json` (`--preset="my 25p"`) | ProRes LT 1280×720, 60 frames 29.97 → 50 at 25, audio 2.000 s |
| Cancel after 2 s | stops; the partial file is removed |
| **Interlaced**: CRV 29.97p → 59.94i, AI motion fields, FFV1 16-bit | 10 frames at 30000/1001, `field_order=tt`, frames flagged interlaced TFF; each frame = its two fields' pictures woven, exact. Split into fields with FFmpeg (`separatefields`): 20 fields, each bit-identical (PSNR ∞) to the same lines of a 59.94p export of the same moments; the two fields of a frame differ (27.8 dB: two moments) |
| Interlaced BFF (59.94i), 50i by blend, 29.97i (deinterlaced at field rate) → 59.94i | all exact; the last needs no new pictures (Retime) |
| Interlaced delivery codecs | H.264 (x264) `tt`; ProRes 422 HQ `bb`; XDCAM HD422 50i in MXF (25 frames, `tt`); GPU encoder asked: falls back to x264; VP9 refused with a message |
| Batch from the command line, ProRes 50i, 3 files (one missing) | 2 done, the missing one failed alone |
| User's case, whole CRV clip 29.97p → 59.94i H.264, Motion AI: command line and the batch window's path | both: 1215 frames at 30000/1001 (59.94 fields/s), `field_order=tt`, all 1215 frames flagged interlaced TFF, video and audio 40.54 s; the two fields of a frame differ more (25.6 dB luma) than alternate lines of the progressive source (30.0 dB): the second field is the in-between moment. Unit test `panel_without_selection_edits_every_file` covers the trap (file added, then 59.94i set with nothing selected) |
| Unit tests (`cargo test -p brp-player export::`) | preset JSON round trip, presets saved and loaded with the settings, rate parsing, geometry (fit/fill/stretch, rounding, CineForm, anamorphic 64:45) |

Speed (1080p, no effects): x264 ~130 pictures/s, ProRes HQ ~105; FFV1 16-bit ~11 (encoder-bound). With AI stages the stack's own speed sets the pace (e.g. Upscale Fast 2× + Denoise to 4K: ~3 pictures/s including FFV1).

**Not done / limits:** OCIO (not in the app yet); hardware encoders that take only hardware frames (Vulkan, VAAPI, D3D12) are hidden; DXV3 alpha is written by our GPU encoder (also with GPU acceleration off, unless FFmpeg has the CVExplorer patch); the Export and Batch windows were checked in screenshots and scripted runs, but not clicked by hand (no mouse control here); per-file batch settings are exercised by the code path every batch uses, not by a hand-made mixed batch; the `target_size` shaders see is the picture's size, not the export size.

### GPU encoders (ProRes, DNxHD/DNxHR, DXV3) — working

No GPU has hardware encoders for the post-production intra codecs, so we wrote our own, in Vulkan compute (`crates/brp-ai/src/gpuenc/`), any vendor. They run right after the export packer (`pack.rs`), which already wrote the picture in the codec's pixel format into a GPU buffer; the encoder turns it into one finished packet on the GPU and only the packet is copied back (exactly its size: a kernel copies `packet bytes` from the GPU buffer to host memory and zeroes what it copied for the next frame). FFmpeg only muxes it (`encode::Precoded`, `Writer` pre-encoded mode), so MOV/MXF muxing, audio, metadata and colour tags stay as before. Every piece the format codes independently runs in parallel, and even entropy coding is parallel: exact code lengths are computed first, prefix sums give every slice/row/block (and inside a slice, every chunk of the coding order) its bit offset, and each thread writes its codewords with byte-granular `atomicOr`s (no 8-bit storage needed). Quantisation is integer arithmetic so the size estimate and the writer (different shaders) always agree; a mismatch is flagged and treated as a failure.

- **ProRes** (`prores.rs`; 422 Proxy/LT/Standard/HQ, 4444 and 4444 XQ, alpha, interlaced as two field pictures). Same bitstream, profile matrices, bit budgets and per-slice quantiser trellis as FFmpeg's `prores_ks` (its tables, LGPL). Per slice of 8 macroblocks (a workgroup): 8x8 DCTs into shared memory, then the exact coded size and error of every plane at every quantiser of the profile at once (each thread a chunk of the coding order; a scan over the chunks passes the adaptive codebook context — previous run and level — from chunk to chunk), coarser quantisers by binary search when even the coarsest is over budget; a workgroup per slice row runs `prores_ks`'s trellis (the NQ x NQ transitions in parallel); one workgroup lays out slices (prefix sum, picture header, slice index); a workgroup per slice writes its planes (DC by one thread, AC chunks in parallel at their bit offsets) and the alpha plane (run-length). With FFmpeg's truncating quantiser the output is the same as `prores_ks` (4,530,172 vs 4,529,464 bytes, 45.05 dB both, LT on the hard 1080i clip); we round instead (0.33 of a step for AC, 0.5 for DC), which gives +1.1 dB for the same size where the budget binds, and a better picture (with 4–9% larger files) on easy content where both stay at the profile's finest quantiser.
- **DNxHD / DNxHR** (`dnxhd.rs`, tables generated from FFmpeg's `dnxhddata.c`, shared with the GPU decoders in `brp_render::gpudec::dnxhd_tables`; DNxHR LB/SQ/HQ/HQX/444 at any size, DNxHD 36/120/145/185/220 (8 and 10-bit, CIDs 1235, 1237, 1238, 1241, 1242, 1243, 1244, 1250–1253, 1259) at 1920x1080, 1280x720 and 1440x1080, **1080i as two coding units**). The frame size is fixed by the profile. A thread per 8x8 block (DCT, DC, luma variance), DC prediction codes per row, coded size of every macroblock at 16 quantisers (a ladder, then the 16 values between the two that bracket the budget), the smallest quantiser that fits, then like FFmpeg's "fast" rate control the least detailed macroblocks get the next finer one while the budget lasts (a variance histogram instead of a sort). Exact bits per block → row prefix sums (the header's row index) → a thread per block writes its codewords. Quantisation as FFmpeg's encoder (floor), the same frame sizes. DNxHD profiles appear in the DNxHR family's profile list ("DNxHD HQ (1080 or 720)" …); the export picks the CID from size, depth and scan, sets the 1080/720 size, and gives FFmpeg's encoder (the CPU fallback) the bit rate it wants for that CID and frame rate. Interlaced DNxHD is now also possible on the CPU path.
- **DXV3** (`dxv.rs`; DXT1, and DXT5 with premultiplied alpha as CVExplorer writes it — the stock FFmpeg decoder reads it as DXT4 and un-premultiplies). A thread per 4x4 block compresses it with the stb_dxt method FFmpeg uses (principal axis, one least-squares refinement, single-colour tables from `texturedspenc.c`): the same textures as FFmpeg's encoder (PSNR identical to 4 decimals). The LZ stage: for each kind of match (whole block or half block by hash, colour word, index word) a GPU radix sort of (value, block) gives every block the most recent earlier block with the same value (what FFmpeg's hash table finds sequentially), checked, with a neighbourhood search as fallback; each block picks its ops; two prefix sums give every op its index (ops are packed 16 to a 32-bit word that sits in the stream where its first op is) and every block its byte offset; the blocks write their ops and data in parallel. When the LZ stream would not be smaller, the raw texture is stored (DXV's raw flag).
- **XDCAM HD422: not done, on purpose.** Measured first: FFmpeg's CPU encoder makes XDCAM HD422 1080i50 long-GOP at about 5 ms per frame (1080p: 1.8 ms) on this machine, faster than the effect stack ever delivers pictures, so a GPU encoder would not make exports faster. At 50 Mbit/s on our footage, long GOP was no better than intra-only (hard 1080i clip: 42.6 dB long GOP vs 44.0 dB intra-only; clean 1080p: 54.4 vs 54.7 dB), but an intra-only stream is not XDCAM HD422 (RDD 9 defines it as long GOP, which broadcast ingest checks), and a long-GOP GPU encoder (motion search, a reconstruction loop bit-exact with decoders' IDCT, VBV-exact CBR, B-frame reordering, field/frame-adaptive coding for 50i) is a large, risky piece for no speed benefit. XDCAM keeps FFmpeg's `mpeg2video`.
- **In the app:** our encoders are named like hardware encoders (`prores_gpu`, `dnxhd_gpu`, `dxv_gpu`) in the families' hardware lists, so the Export and Batch panels' **GPU acceleration** toggle offers them (with its own tooltip), also with interlaced output; Advanced lists them as "… (BRP, GPU)". `--encoder=gpu` (or the name) on the command line. They need no probe: Vulkan compute is all they use. DXV3 alpha always uses ours unless FFmpeg has the CVExplorer patch and GPU acceleration is off. **Fallback:** if our encoder fails (kernels, a packet over its buffer, an internal size mismatch), the export starts again with the CPU encoder (`GpuEncodeFailed` in `export::start`; `BRP_TEST_GPUENC_FAIL=1` forces it for tests); `BRP_NO_GPU_ENCODERS=1` hides them. `--verify` now compares colours weighted by alpha when both pictures have alpha (premultiplied formats keep no colour under transparent pixels).
- **Cross-vendor:** NVIDIA RTX 5070 Ti and the AMD iGPU (64-wide subgroups, 32 KB shared memory: the ProRes slice kernel uses about 26 KB; no subgroup operations, no float atomics; `shaderInt64` for the bit writer's accumulator). On the AMD iGPU (640x360, 3 frames each) every codec decodes cleanly with the same PSNR as on NVIDIA; DXV files are bit-identical between the two, ProRes/DNx differ only by float DCT rounding (PSNR equal to 0.01 dB).
- **Bench:** `cargo run --release -p brp-ai --example gpuenc -- --raw=in.yuv --size=1920x1080 --pix=yuv422p10le --codec=prores:hq|dnx:<cid>|dxv[:alpha] [--tff|--bff] [--alpha] --out=out.mov [--repeat=3]` (raw frames in, muxed file out, GPU time per frame).

**Results** (RTX 5070 Ti; CPU = FFmpeg 7.1 on a Ryzen 9 7900X, all 24 threads; 10 frames each; GPU time per frame = all the encoder's GPU work including the packet copy to host memory, median; CPU = wall time per frame over 100 frames. PSNR Y / average of the planes, decoded output vs the encoder's input, 10-bit scale for 10-bit formats and 8-bit for 8-bit ones. Sources: 1080p = `test.mp4` (clean), 1080i = the McCartney MPEG-2 broadcast clip (grain, motion), 4K = `test.mp4` scaled, alpha = `PASSAGEM`. Every GPU file decodes with FFmpeg without a single warning.)

| Case | Size | GPU ms/frame | CPU ms/frame | GPU KB/frame | CPU KB/frame | GPU PSNR Y / avg | CPU PSNR Y / avg | GPU decode messages |
|---|---|---|---|---|---|---|---|---|
| ProRes proxy | 1920x1080 | 1.00 | 8.2 | 147 | 139 | 43.57 / 44.54 | 42.92 / 43.88 | none |
| ProRes lt | 1920x1080 | 1.53 | 10.0 | 340 | 325 | 53.96 / 54.92 | 52.94 / 53.88 | none |
| ProRes standard | 1920x1080 | 1.15 | 7.2 | 446 | 428 | 59.56 / 60.13 | 58.32 / 58.93 | none |
| ProRes hq | 1920x1080 | 0.96 | 7.0 | 605 | 580 | 67.48 / 67.52 | 65.65 / 65.75 | none |
| ProRes proxy, 1080i (hard) | 1920x1080 | 1.75 | 14.4 | 200 | 199 | 35.14 / 37.42 | 34.46 / 36.77 | none |
| ProRes lt, 1080i (hard) | 1920x1080 | 2.13 | 12.5 | 444 | 444 | 42.84 / 44.91 | 41.64 / 43.79 | none |
| ProRes standard, 1080i (hard) | 1920x1080 | 1.74 | 10.3 | 633 | 633 | 46.94 / 48.66 | 45.52 / 47.36 | none |
| ProRes hq, 1080i (hard) | 1920x1080 | 1.60 | 10.9 | 948 | 946 | 52.36 / 53.51 | 50.70 / 52.02 | none |
| ProRes 4444 + alpha | 1920x1080 | 2.06 | 12.1 | 2037 | 1784 | 66.23 / 67.61 | 64.76 / 66.27 | none |
| ProRes 4444 XQ | 1920x1080 | 1.03 | 10.5 | 836 | 781 | 72.01 / 71.92 | 70.48 / 70.55 | none |
| ProRes HQ 4K | 3840x2160 | 2.39 | 26.6 | 1519 | 1390 | 75.91 / 76.09 | 73.49 / 73.55 | none |
| ProRes 4444 + alpha 4K | 3840x2160 | 7.42 | 43.9 | 5418 | 4693 | 67.24 / 68.76 | 66.24 / 67.77 | none |
| DNxHR lb | 1920x1080 | 0.43 | 2.5 | 184 | 184 | 51.12 / 51.33 | 51.17 / 51.38 | none |
| DNxHR sq | 1920x1080 | 0.45 | 2.4 | 592 | 592 | 58.47 / 57.52 | 58.54 / 57.56 | none |
| DNxHR hq | 1920x1080 | 0.47 | 2.3 | 896 | 896 | 60.75 / 59.11 | 60.84 / 59.17 | none |
| DNxHR hqx | 1920x1080 | 0.50 | 3.8 | 896 | 896 | 70.15 / 69.81 | 70.26 / 69.83 | none |
| DNxHR 444 | 1920x1080 | 0.73 | 5.0 | 1792 | 1792 | 70.60 / 71.40 | 70.67 / 71.44 | none |
| DNxHR hq 4K | 3840x2160 | 1.26 | 8.1 | 3556 | 3556 | 61.44 / 60.50 | 61.46 / 60.52 | none |
| DNxHR hqx 4K | 3840x2160 | 1.34 | 14.4 | 3556 | 3556 | 72.39 / 71.96 | 72.44 / 72.01 | none |
| DNxHD 185 (1238), 1080p | 1920x1080 | 0.54 | 2.6 | 896 | 896 | 60.75 / 59.11 | 60.84 / 59.17 | none |
| DNxHD 185x (1235), 1080p | 1920x1080 | 0.49 | 3.8 | 896 | 896 | 62.00 / 60.40 | 62.09 / 60.45 | none |
| DNxHD 185 (1243), 1080i (hard) | 1920x1080 | 0.68 | 2.3 | 896 | 896 | 53.21 / 53.55 | 53.38 / 53.71 | none |
| DNxHD 185x (1241), 1080i (hard) | 1920x1080 | 0.69 | 4.4 | 896 | 896 | 53.95 / 54.67 | 54.14 / 54.84 | none |
| DNxHD 120 (1242), 1080i (hard) | 1920x1080 | 0.65 | 2.5 | 592 | 592 | 49.31 / 50.03 | 49.49 / 50.18 | none |
| DXV3 DXT1 | 1920x1080 | 0.59 | 2.0 | 174 | 178 | 40.40 / 40.87 | 40.40 / 40.87 | none |
| DXV3 DXT1 4K | 3840x2160 | 1.01 | 7.1 | 565 | 581 | 42.81 / 44.24 | 42.81 / 44.24 | none |
| DXV3 DXT5 alpha (premultiplied) | 1920x1080 | 0.80 | – (no FFmpeg encoder) | 347 | – | 43.4 / 44.9 premultiplied RGB, alpha 50.4 | – | none |

- **Speed:** ProRes 6–11x faster than `prores_ks` on all 24 cores (1080p HQ 0.96 vs 7.0 ms, 4K HQ 2.4 vs 26.6 ms), DNxHD/DNxHR 3–11x (4K HQX 1.34 vs 14.4 ms), DXV3 3–7x (4K 1.0 vs 7.1 ms), and the CPU stays free for decoding and the effects. On the AMD iGPU (small frames only) the same work takes about 10–40x longer than on the RTX, still correct.
- **ProRes quality/size:** where the profile's budget binds (the 1080i clip) the GPU files are the same size and 0.65–1.5 dB better (rounding instead of truncating, same rate control). On easy content both encoders sit at the profile's finest quantiser: the GPU files are 4–9% larger (14% for 4444 with alpha, which `prores_ks` codes almost losslessly: its alpha budget is 20x) and 0.7–2.5 dB better.
- **DNx:** sizes are fixed by the profile (equal by construction); PSNR 0.02–0.17 dB below FFmpeg (its rate control searches every quantiser; ours a 16 + 16 ladder and a variance histogram).
- **DXV3:** the same textures as FFmpeg's (PSNR equal), files 2–3% smaller (the sort finds whole-block matches FFmpeg's word-by-word hash does not). DXT5 has no FFmpeg encoder to compare with; PSNR is on premultiplied colour (what DXT5 stores).
- Slowest part left: the ProRes 4444 alpha plane's run-length coder runs on one thread per slice (4K with alpha 7.4 ms; still 6x the CPU).

### GPU decoding (DXV3, ProRes, DNxHD/DNxHR) — working

Our own decoders on the GPU (Vulkan compute, any vendor) for the intra-only production codecs, in playback, scrubbing and export. Every picture is **bit-exact with FFmpeg 7.1** (same integer arithmetic), so the frame cache, deinterlacing, effects and export see the same pixels whichever decoder made them.

- **Where it plugs in** (`brp-media/src/gpudec.rs`, `source.rs`): the media layer defines the hook (`GpuDecoders` factory, `GpuDecoder`, `GpuPlanes`); `brp-render/src/gpudec/` implements it and the player registers it once the GPU exists (also after a GPU switch, and in command-line export). The decode worker sends each packet to the GPU decoder instead of FFmpeg; frame numbering, exact seeking, the cache, the paused fill and interlace flags are unchanged. A file opened before the GPU was up (the player opens it while the GPU starts) restarts at its current frame with the GPU decoder. `BRP_GPU_DECODE=0` turns it off.
- **Pictures stay on the GPU.** A decoded picture is a GPU buffer of planes in FFmpeg's layout (`yuv422p10le`, `yuv444p12le`, `rgba`…); `VideoFrame` holds it and makes a CPU copy only when asked (`plane()`). The renderer's frame cache (`frames.rs`) copies it GPU to GPU into its textures (no upload). Sources decoding on the GPU keep only a window of pictures around the playhead in GPU memory: the frames read ahead (60 with AI effects, 12 without) plus 24 behind and 16 spare, about 400 MB at 1080p. That is at most a quarter of the card. It shrinks further, keeping at least 256 MB, when the driver's budget (`VK_EXT_memory_budget`, other programs counted) leaves less than 1 GB free. The compressed packets of frames already decoded stay in system memory, under the RAM cache budget. A frame that has left GPU memory is decoded again from its packet, with no seek and no file read: GPU decoding takes a few ms, and packets are about 5 times smaller than pictures. Measured on a 40 s DNxHD 1080p clip with Denoise: 390 MB of decoded pictures, where the old cap of a third of the card held the whole clip at 4.8 GB, plus 540 MB of packets in RAM. `gpudeccheck --source` checks re-decoded frames against FFmpeg (exact). Trimming the cache never drops the frame just stored. A frame the decoder made is never marked unreachable, even if it is dropped afterwards. With the smaller window, the paused read-ahead used to store a frame, drop it at once while trimming to 90%, and mark it unreachable ("passed without producing it"). Playback with an AI effect then waited forever for any picture needing that frame: 88 pictures after play, then nothing. Until the frame index is ready (~150 ms after opening), frames are numbered from the file's first decoded picture. A seek made before any picture was decoded used to take the first picture after the seek as frame 0, so the frame returned was wrong (`gpudeccheck --scrub` showed it on seek 0). Now such a seek first decodes the file's first picture, which fixes the origin and is cached as frame 0, and then seeks: one extra decode, with no wait for the index. Numbering from the stream's start time was tried instead and made it worse: the CRV clip's start time is not its first frame's. Measured: exporting 24 frames reads back none for DXV; for YUV files only the 12 frames the interlace detector samples (`detect.rs` still analyses on the CPU).
- **Fallback, never less support:** a packet the GPU decoder refuses (a profile it does not take, damaged data, an RGB DNx macroblock…) is decoded by a slice-threaded FFmpeg decoder, so output is identical; 8 refusals in a row hand the whole stream back to FFmpeg. Taken by FFmpeg: DXV3 high-quality (YCoCg: YCG6/YG10) and old headerless DXV2, ProRes 4:2:2 with alpha, DNx 12-bit, alpha, MBAFF (CID 1260), CIDs 1256/1258/1260 and RGB (adaptive colour transform) 4:4:4, MPEG-2/XDCAM.
- **DXV3** (`dxv.rs`): the LZ-style op stream is sequential, so the CPU unpacks it into system memory (reading it back from the write-combined upload buffer was 50x slower), then one GPU thread per 4×4 block expands DXT1/DXT5 with FFmpeg's `texturedsp` arithmetic; DXT5 is un-premultiplied as FFmpeg does. Details found by testing against libavcodec: block-aligned size is 4 (not 16); DXT5 op 2 jumps back 8 + 4·n words.
- **ProRes** (`prores.rs`): the CPU reads the frame/picture/slice headers; one workgroup per slice: four threads decode Y, Cb, Cr (adaptive Rice/exp-Golomb, codes up to 31 bits as FFmpeg's long bit reader) and alpha (run-length) into shared memory, then 64 threads run the IDCT. FFmpeg's 10-bit IDCT constants differ from the 8-bit ones (W3 = 19265, W4 = 16384); 4:4:4 decodes as 12-bit like FFmpeg.
- **DNxHD/DNxHR** (`dnxhd.rs`, tables in `dnxhd_tables.rs`, shared with the encoders): each macroblock row is its own bitstream, decoded by one GPU thread (one per workgroup: rows in one warp diverged at every branch, 3–4x slower), with a 64-bit bit cache in registers and 10-bit first-level code tables in shared memory; then one thread per 8×8 block runs the IDCT. DNxHR HQX dequantises like 4:4:4 (bias 32, shift 6), and the frame's colour-transform flag only allows RGB macroblocks (bit clear = RGB): both found against libavcodec.
- **Integrated GPUs:** correct (bit-exact on the AMD iGPU too) but ProRes/DNx entropy decoding is serial per slice/row and an iGPU has few compute units: 45–150 ms per 1080p frame, far slower than the CPU. So on an integrated GPU only DXV decodes on the GPU (`BRP_GPU_DECODE=all` forces all), and on any GPU a stream whose first frames take more than 25 ms goes back to FFmpeg.
- **XDCAM HD422 (MPEG-2 4:2:2 long-GOP): not done, on purpose.** FFmpeg decodes it in 2.5 ms per 1080i frame on one CPU thread and 0.44 ms on all threads. A GPU decoder would still be bound by per-slice serial entropy decoding (DNx, the closest case, takes ~3 ms on the RTX), and would add motion compensation, field/frame prediction modes and reference pictures kept across the GOP for exact seeking; hardware video decoders do not take 4:2:2 MPEG-2 either. It cannot beat the CPU, and its decoded frames are small (8-bit 4:2:2, 4 MB) to upload.

**Correctness** (`brp-render/examples/gpudeccheck`: every plane of every frame against FFmpeg's decode of the same file; "Source" = through the player's frame-exact source; "scrub" = random seeks; `examples/dxvfuzz`: generated valid DXV packets using every op):

| Files | RTX 5070 Ti | AMD iGPU |
|---|---|---|
| ProRes Proxy, LT, Standard, HQ (`prores_ks`, `prores_aw`), HQ interlaced, HQ 4K, 1000×562, 4444 XQ, 4444 + 16-bit alpha; our export's ProRes HQ and 4444 alpha | bit-exact, 24 frames each, Source and 30–60 random seeks too | bit-exact (HQ, interlaced, 4444 alpha; 3 frames each) |
| DNxHD 120 / 185x (8/10-bit), both interlaced, 720p 60 / 90x; DNxHR LB, SQ, HQ, HQX, 444, HQ 4K, SQ 1000×562; `CRV CHA EITA LUCAS BASE.mov`; our export's DNxHR HQX (MXF) and HQ | bit-exact, Source and random seeks too | bit-exact (DNxHD 120, 185x interlaced, DNxHR HQX, 444) |
| DXV3 DXT1 1080p / 4K (FFmpeg encoder), DXT1 and DXT5-alpha from our export | bit-exact, Source and random seeks too | bit-exact (DXT1) |
| `dxvfuzz`: DXT1 / DXT5 × LZ-packed / raw, 20 packets each at 1920×1080 and 724×404 (every op, long runs) | identical to FFmpeg | – |
| Through the whole player (export to FFV1 16-bit, GPU decoding on vs off) | identical files (ProRes interlaced, DXV, ProRes 4444 alpha) | – |

**Speed** (ms per frame; GPU = upload, decode and wait, one frame at a time, RTX 5070 Ti; FFmpeg = `ffmpeg -benchmark` decoding only, 1 thread / all 24 threads):

| Clip | GPU | FFmpeg 1 thread | FFmpeg 24 threads | AMD iGPU |
|---|---|---|---|---|
| DXV3 DXT1 1080p | 1.49 | 2.54 | 0.88 | 3.08 |
| DXV3 DXT1 4K | 6.61 | 10.50 | 3.12 | – |
| ProRes Proxy 1080p | 0.30 | 3.33 | 1.04 | – |
| ProRes HQ 1080p | 0.82 | 10.79 | 1.50 | 90.6 |
| ProRes HQ 4K | 2.80 | 42.92 | 5.33 | – |
| ProRes 4444 XQ 1080p | 1.70 | 31.54 | 3.75 | – |
| ProRes 4444 + alpha 1080p | 2.03 | 31.88 | 4.17 | 153 |
| DNxHD 120 1080p | 2.67 | 6.79 | 0.92 | 44.5 |
| DNxHD 185x 1080i | 3.03 | 6.83 | 1.21 | 54.6 |
| DNxHR HQ 1080p | 3.79 | 9.54 | 1.21 | – |
| DNxHR HQX 1080p | 3.76 | 9.33 | 1.58 | 62.6 |
| DNxHR 444 1080p | 6.67 | 18.29 | 2.62 | 116 |
| DNxHR HQ 4K | 5.66 | 34.33 | 21.79 | – |

ProRes is 2x faster than FFmpeg on all 24 threads and DNxHR 4K 4x; DXV and 1080p DNx are about 2–4x slower than 24 CPU threads but use none of them, are real-time at 4K (DXV 6.6 ms, DNxHR 5.7 ms), and the picture is already on the GPU (no 4–66 MB upload per frame). The GPU numbers wait for each frame; pipelining packets would hide the upload and the CPU's DXV unpacking (most of DXV's time).

**Not done / limits:** XDCAM (above); DXV3 YCoCg and DNx 12-bit (no test files); pipelining several packets in flight; the interlace detector still reads its sample frames back to the CPU; a source keeps its decoder's GPU after a GPU switch until it is reopened (its pictures are then copied through the CPU); the Info panel does not yet show which decoder is used (`Source::decoder_name()`). The player was exercised by command-line exports and the check tools, not by clicking through playback.

### Deblur, Compression Clean-up, Depth, Stabilise — working

Four more AI entries (`AiKind::Deblur`, `CleanUp`, `Depth`, `Stabilise`). Every model is Apache-2.0 or MIT (the app is open source: the CC-BY-NC Depth Anything Base/Large, CodeFormer and ProPainter were left out on purpose). Each network was checked against a plain-PyTorch version of the official one written in `tools/airef/` (loading weights with `weights_only`; no downloaded code runs).

| Effect | Model | Matches the official network | Speed (RTX 5070 Ti) |
|---|---|---|---|
| Deblur | NAFNet REDS width 64 (MIT): video frames with motion blur and compression | 71 dB | 120 ms at 720p, ~0.3 s at 1080p |
| Compression Clean-up | SCUNet real PSNR (Apache-2.0) | 76 dB | 107 ms at 1280x640 |
| Depth | Video Depth Anything Small / Depth Anything V2 Small (Apache-2.0) | 0.02 % / 0.04 % mean difference | 18 / 14 ms per frame (any picture size: the network runs at 518 lines) |
| Stabilise | (no network: the motion) | — | ~3 ms per new frame (built-in motion) |

- **Deblur** (`brp_ai::restore`, `pth2ncnn nafnet`): NAFNet's official test setting replaces the global average of its channel attention with a local box of 1.5x the training crop (TLC: 384 picture pixels, halved per level): a new runner layer `BoxMean` (column running sums, then a row prefix sum in shared memory; the box clamped inside the picture as the official replicate padding). The GoPro-trained NAFNet was tried and dropped: it barely sharpens real footage and breaks up dark flat areas. Time goes to the 28 blocks at 1/8 size (1x1 GEMMs, 50 ms at 720p), then the gates (16 ms) and the box averages (13 ms); fusing the gates into the convolutions is the next speed-up.
- **Compression Clean-up** (SCUNet, `pth2ncnn scunet`): Swin windows on our `WindowAttention` (SCUNet's position table [heads][2w-1][2w-1] rewritten as SwinIR's layout; all its blocks shift, as its input_resolution 256 keeps every level above the window). Honest result: on the stage clip at 0.8 / 1.5 Mbit/s H.264 (NVENC, scored against a 50 Mbit/s decode) +0.6 / +0.4 dB; it removes blocking and mosquito noise and smooths a little texture. JPEG q20: +0.4 dB.
- **Depth** (`brp_ai::depth`, `pth2ncnn depth`): DINOv2 and the DPT head as WAFT's; the head is split in four graphs at the places Video Depth Anything's temporal modules sit, so per-frame depth chains them and the video model runs a module between them. A temporal module is three graphs (GroupNorm: a new runner layer; LayerNorms; projections; GEGLU feed-forward) around our attention-over-time kernel. The projections have no bias, so a cached frame's key at window place i is its own projected key plus place i's position encoding projected (precomputed at conversion): the cache keeps projected keys and values (f16, 44 slots, ~0.5 GB), streamed exactly as video_depth_stream.py (the first frame after a reset kept, then the window's list). Unlike the official stream, the first frame runs twice (alone, then against the filled window), so the depth does not jump at the second frame (official: median 5.1 then 6.3). In the player, the stream continues while pictures follow closely and are of one shot (a cut test reads one value back), else it starts again after replaying 4 earlier pictures (as matting); export runs in order.
  - Effects (`DepthFx`): depth scaled 0 (far) .. 1 (near) by its 2nd/98th percentiles (a histogram on the GPU, nothing read back); Depth map, Lens blur (gather, 64 samples on a golden-angle disk, up to 24 px at 1080 lines; a sample counts where its own blur reaches this pixel, and a farther one only as far as this pixel's own blur, so far things do not bleed over sharp near ones), Fog, Depth key (alpha). 1080p depth + lens blur exports at 15.6 fps.
- **Which depth** (`tools/airef/depth_stability.py`): change of each pixel's depth along the motion between frames (WAFT motion, pixels that pass a forward-backward check, each clip's depth divided by its own median), and the frame-to-frame scale "breathing"; from frame 4 (after Video Depth Anything's start):

| | Phone clip (slow camera, glass bowl) | Three people gesturing |
|---|---|---|
| Depth Anything V2, each frame | 0.37 % / 0.40 % | 1.95 % / 1.81 % |
| Video Depth Anything | 0.21 % / 0.19 % | 0.93 % / 0.78 % |
| V2 steadied along the motion (SEA-RAFT, 0.8 of the history) | 0.09 % / 0.06 % | 0.55 % / 0.34 % |

  Steadying per-frame depth along the motion scores best but leaves trails and doubled edges around fast hands (the score cannot see them: it measures along the same kind of motion); Video Depth Anything keeps clean shapes and halves the flicker, so it is the default ("Video (steady)"), with V2 alone as "Per frame".
- **Stabilise** (`brp_ai::stabilise`): per pair of neighbouring pictures, the camera's movement as a similarity (shift, rotation, zoom) fitted to a 96-column grid of the motion read back: RANSAC from two-cell hypotheses (fixed sequence, 1.5 px), then Tukey-weighted least squares, so people and cars moving on their own count less; none across a cut. A picture's path composes those over ±Smoothness frames (stopping at cuts and the clip's ends), the steady camera is the Gaussian-weighted mean, and the picture is warped onto it (Catmull-Rom) with a zoom (default 5 %). The fits are 4 numbers each, kept for 16384 pairs: playing on, one new fit per frame. Added shake (±8 px, ±0.4°, with a slow pan that must stay): jitter (mean change of the frame-to-frame movement) 1.25 px, 1.04 px, 1.28 mrad → 0.05 px, 0.04 px, 0.08 mrad on the gesturing clip; the same through a player export (0.04 px).

**Not done / limits:** Deblur and Clean-up run the whole picture at once (about 1.5 GB of activations at 1080p; 4K would need tiles) and are export-speed (1.8 fps together at 1080p); Stabilise's zoom is fixed (no automatic crop per shot) and uncovered edges repeat; the depth range follows each frame's percentiles (Video Depth Anything's steady values make that steady too); the new kernels are not in the start-up warm-up; checked through command-line exports and the check tools, not by clicking through playback.

**Deblur runs away on dense, sharp detail.** On aerial shots of the CRV clip (frames 148–152, 400, 700: crowds, foliage), NAFNet REDS's activations grow from about 9 to 1e5–1e6 by the middle blocks. The output then reaches ±1000 over 18–54% of the picture, which shows as coloured noise. The official network does the same (`tools/airef/nafnet_ref.py`, with or without TLC), and at half size the same frame is fine: a deblur model amplifies fine detail, and footage that is already sharp sends that gain out of range. `Restorer` therefore falls back to the original picture where the output moved a pixel by more than 0.35, or is not finite. On a normal frame that is 0.002% of pixels. Over the 17 × 17 around each pixel, 2% of such pixels or more shows the original alone, and fewer fade it in. After this, no pixel of those frames moves by more than 0.35. Those areas were sharp already, so they keep the original. The same check runs after Clean-up.

**Glass popups' animation redraws wait for the next display refresh.** The popups present in MAILBOX mode. While the AI worked, the "Processing…" spinner asked for redraws "at once", about 900 a second, and the player window got none. Playback with the Effects panel open then waited about 7 s for its first picture. Now it is 0.3 s with Deblur.

### Steady Detail — working

The **Steady Detail** AI effect calms fine detail that comes and goes from frame to frame. That covers aliasing on thin lines, compression, and in-camera sharpening that varies per frame. The opening drone shot of the CRV clip is the test: its sharpness jumps 9.9% between consecutive frames, and the outlines of the stage's LED tower change thickness and colour. Denoise cannot help there. It only merges a neighbour where it differs by about the noise level, and a crisp-versus-soft difference is far more.

How it works (`Temporal::steady_detail`): each frame is split into a broad picture (Gaussian blur of *Detail size* px) and the detail above it. The neighbours' detail (*Frames* each way) is moved onto the frame along the motion and averaged with its own. A neighbour is used where its blurred luma is within 0.03 of the current one, so it is lined up on the same thing, weighted 0.85 per frame of distance, and a side stops at a cut as in Denoise. The broad picture stays the current frame's.

Averaging loses detail that does not line up exactly, so *Keep sharpness* brings some of it back. The gain is at most 1 + keep/2, towards the frames' mean detail energy over a 31 × 31 box, which is itself steady. A gain above 1.5 was never needed on the test, so the setting stops there. *Strength* mixes the result with the original.

Measured on frames 0–18 (`tools/airef/shimmer_score.py`, change along SEA-RAFT motion where it passes a forward-backward check; `tools/airef/deshimmer_proto.py` is the PyTorch prototype):

| | Tower (60 × 110 px) | Whole frame | Sharpness | Its jumps |
|---|---|---|---|---|
| Original | 0.0218 | 0.0183 | 751 | 9.9% |
| 4 frames, 2.5 px, keep 1 | 0.0098 | 0.0083 | 470 | 1.8% |
| 4 frames, 8 px | 0.0075 | 0.0065 | 425 | 2.2% |
| 8 frames, 4 px, keep 0.6 | 0.0070 | 0.0067 | 404 | 1.6% |
| Defaults, SEA-RAFT motion | 0.0090 | 0.0076 | 477 | 1.7% |

The defaults are 4 frames each way and 4 px. It plays in real time at 1080p on the RTX 5070 Ti with the built-in motion: 29.5 pictures/s at 4 frames, 2.5 px and at 8 frames, 8 px. The cost is softness: about 40% less Laplacian energy at the defaults. Lost aliased detail cannot come back, and averaging also smooths what did not shimmer. **Only where it shimmers: tried, not kept.** Three measures were tried on the clip to steady only the areas that shimmer, so steady ones would keep their sharpness (`deshimmer_proto.py`, `QMAP`, `LEVEL`):
- **Disagreement of the lined-up detail** (relative, then absolute): it marks every edge, because a sub-pixel misalignment of a steady edge looks like disagreement; only flat areas (sky, sea, plain grass) stay unmarked. Built in the effect, it reached 0.0121 on the tower at sharpness 501. Plain Strength 0.85 everywhere gives 0.0107 at 484, which is better, so it was taken out again.
- **How much each frame's local detail strength varies over the window:** about the same everywhere textured (tower 0.14, grass 0.15, palms 0.14), because this footage pumps between crisp and soft as a whole frame.
- **Levelling each frame's own detail strength to the window's mean**, without averaging: sharpness stays (741), and its frame-to-frame jumps fall from 9.9% to 5.9%. But the change along the motion barely moves (0.0214 against 0.0218): what is seen is the detail itself changing (aliasing), which only averaging steadies.

**Two AI-pipeline bugs found through it** (wrong or banded pictures after Go to In, "not constant but often"), both fixed. `BRP_TEST_SEEK_AFTER=secs:frame` jumps there, `BRP_TEST_MAXIMIZED` opens maximized, and dumps (`BRP_AI_DUMP`) compared with an export of the same frames find them:
- **Source pictures were kept across a change of the tick grid.** At the start, the interlace detection briefly doubled the pictures per frame and then restored them. The worker keeps stage 0 (the source pictures, keyed by tick) when a setting changes, so frame n then showed frame n / 2 (16 → 8, 18 → 9, …). Now they are kept only if their colour, deinterlacing and grid are unchanged (`source_key`).
- **Unwritten frame images stayed in the AI runners' frame cache.** A job that stops at a frame no longer decoded is dropped without running. The frames it had added to its `OfflineEffects` cache kept images that were never written, holding other memory's contents: bands of other frames, wrong colours. This often happened after a jump, and more with the smaller decoder window. `OfflineEffects::abort` (`FrameCache::forget`) now drops them.

Five maximized play-then-jump-to-0 runs: every picture is identical to the export (64.8 dB, rounding only). Before the fixes, two of three runs had wrong pictures from frame 16 on.

### Matting with BiRefNet — working

The Matting effect has a *Model* setting:
- **BiRefNet** is the default. It finds the alpha one picture at a time. Edges are finer, including hair and see-through things like glass, and it works on any subject, not only people.
- **RVM** is the earlier recurrent model. It is made for people, steadier from frame to frame and has softer edges.

The weights are BiRefNet lite matting (`ZhengPeng7/BiRefNet_lite-matting`, MIT, Swin-T backbone, 44 M parameters). They are converted by `pth2ncnn birefnet` into one graph at 1024 × 1024, the size the model was trained at. `brp_ai::birefnet` squeezes the picture to that square, as the official inference does, and stretches the alpha back. The picture's own colour goes with the alpha into the existing outputs: transparent, green, blue or the matte.

| | |
|---|---|
| Match with the official network | Alpha within 0.0045 and 0.0083 (0..1) on two test frames, 0.00004–0.00006 on average. The reference is `tools/airef/birefnet_ref.py`, plain PyTorch loading the weights strictly; activations here are f16. |
| Speed | 52 ms per 1024² picture in `aicheck custom`, and 72 ms per frame in a 1080p export (61 frames in 4.4 s); both on the RTX 5070 Ti while a game was using it, so the speed with the GPU to itself is not measured yet. |
| Memory | About 2.5 GB while it runs (1024², f16). |

How the graph is built:
- **ImageNet normalisation** is a 1 × 1 convolution at the start.
- **The Swin v1 backbone runs twice**, at 1024 and at 512, and the features are concatenated per level. A Swin block pads to whole windows after `norm1` with zeros (`Canvas`) and crops back after the projection, as Swin v1 does. The shift mask covers the padded size, which is what `WindowAttention` already does. Patch merging is a pixel unshuffle, with the norm and the reduction permuted to its channel order.
- **The image goes to every decoder level as tiles** (`b c (hg h) (wg w) -> b (c hg wg) h w`), by the `Tiles` layer.
- **Deformable convolutions (ASPPDeformable, 1×1, 3×3 and 7×7)** are split in three steps. The offsets and modulator are one convolution. `DeformIm2col` samples the input as torchvision's `deform_conv2d` does: bilinear, zero outside, times 2·sigmoid(modulator), into `[C·K²][H][W]`. Then a 1 × 1 convolution runs on tensor cores with the BatchNorm folded in. The 7 × 7 at 256² makes 3136 channels (411 MB in f16).
- **ASPP's pooled branch** adds its share of `conv1` per channel, a broadcast. BinaryOp `20=1` keeps that add out of the convolution fusion.

**Steadiness** (default 2 frames each way, 0 to 4) averages each frame's alpha with its neighbours'. BiRefNet still runs once per frame: alphas are cached per (stage input, tick), 32 at most. Each neighbour's alpha is moved along the built-in motion, chained step by step as in Denoise. It is weighted by how well its moved picture agrees with the current one (Denoise's agreement map) and by 0.85 per frame of distance. A side stops at a cut, by the same rule as `MERGE_ALL` (`ALPHA_MERGE`, `Temporal::alpha_window`).

A neighbour's alpha is clamped to the range the current alpha has over the 5 × 5 around the pixel. Where the current matte is flat, a wrong motion cannot bring in a ghost; near an edge the range is wide enough to average. Steadying costs about 10% more time per frame (4.9 s against 4.4 s for 61 frames).

Measured on 61 frames of the CRV clip at 1080p (`tools/airef/matte_stability.py`). Shimmer is the mean |alpha_t − alpha_t−1 moved along SEA-RAFT| over soft edges, where the motion passes a forward-backward check. Ghost is the share of pixels with alpha above 0.02 where the unsteadied matte has none within 2 px.

| | Shimmer | Ghost |
|---|---|---|
| RVM | 0.136 | |
| BiRefNet, steadiness 0 | 0.129 | |
| Steadiness 2 (the default: 5 × 5 range, no margin) | 0.115 | 0.0003% |
| Steadiness 4, the same | 0.112 | 0.003% |
| Steadiness 2, 3 × 3 range + 0.1 margin | 0.098 | 0.25% |
| Steadiness 4, 3 × 3 range + 0.3 margin | 0.075 | 0.76% |

The rows with a margin score lower partly by pulling alpha into the background beside fast movement: faint trails, visible beside a moving thumb. Even a steady matte scores well above 0, because moving a sharp edge along a slightly wrong motion counts as change, and RVM, made to be steady, scores worst. So most of what is left is that floor rather than shimmer.

Limits:
- **No colour clean-up yet:** the colour at soft edges is the picture's own, so the background bleeds into hair.
- **The full BiRefNet matting model** (Swin-L, 885 MB, same graph code) is not wired in yet.

### Window, effects switch, cache bars, RAM cache of effect pictures, translated parameters — working

- **The window stays where it is when a file opens.** No open path (menu, drop, a file from another launch) moves, resizes or un-maximises the window; checked from maximised, normal, fullscreen and minimised windows with `BRP_TEST_DROP_AFTER=secs:path` (opens a file as a drop would and logs the window's place before and after) and with a second launch. Two safeguards: the Open dialog is owned by the player window (in fullscreen the window is always on top and an unowned dialog opened behind it), and a file from another launch only restores the window if it is minimised (`bring_to_front`).
- **All effects on/off.** A switch at the left of the Effects panel's toolbar (and the E key) sets `Effects::bypass`. The stack, its settings and undo history are untouched; `split()` then gives no AI stages and the live shaders switched off (they keep their compiled pipelines), and `all_stages()` is empty, so export matches what is shown. Not an undo step. Turning it back on finds the AI pictures still cached (same stack key). `BRP_TEST_FX_BYPASS=off:on` (seconds) flips it for tests.
- **Cache bars on the timeline.** Green: decoded frames, ready to show (in video memory for GPU-decoded files, the window around the playhead). Thin grey above it: what is in system RAM — compressed packets kept for re-decoding (`Source::cache_map`, a copy of the worker's packet keys) and FFmpeg's decoded frames. Thin accent bar above that: processed effect pictures of the current stack in RAM. Frames are bucketed per pixel, not sorted (there can be ~100k packets). The Info panel counts green and grey.
- **Processed effect pictures in RAM** (`Ai::ram`). The worker copies each finished picture (RGBA16F) into a host-visible buffer in the same submission and hands the pixels back with it; they are kept by (stack key, position) — the key that identifies the exact stack and settings — and never dropped because the stack changed: going back to the same stack (undo, the effects switch, the same values again) finds them. Least recently used go first over `mem::fx_cache_budget()` (25% of RAM, `BRP_FX_RAM_MB`), or when less than 1.5 GB is free. Playback asks for a picture kept in RAM to be uploaded again (its own thread, `restorer`, a few ms at 1080p) instead of queuing it for computation, so `paces()`/`ready()` work as before: pictures still go through the 1 GB GPU cache, ahead of the playhead by the same rule. Measured (DNxHD 1080p, denoise): computing ~25 ms a picture, from RAM ~5 ms; after the effects were switched off and on and playback jumped back, 279 pictures came from RAM and only those never made were computed. Cleared when another file opens (positions would mean other pictures); kept on a GPU switch.
- **Effect parameters in Portuguese.** Shader files carry translations as plain comments, so mpv and libplacebo still read them: `// @brp pt.title <name>` and `// @brp pt.param <NAME> <label> | <description>` (`ShaderFile::l10n`, `param_label`, `param_desc`). A translated description without "(cat: …)" or "(options: …)" keeps the English one's tags; a colour (R/G/B) uses its R parameter's label. Every bundled shader has Portuguese (test `bundled_shaders_translated` checks each parameter the panel shows). AI parameters are translated in `i18n.rs` (`Lang::ai_param`, by effect id and parameter name, choice lists included; test `ai_params_translated` checks the number of choices matches). Stored names (presets, `--fx`) stay English.

### Working colour space for effects — working

Design in §7a. Code: `brp-render/shaders/space.glsl` (+ `space.comp`), `src/workspace.rs` (`Space`, `SourceColor::same`), `effects.rs` (`ChainFrame.space/want`, `ChainOutput.space`, `run_main`, `main_passes`, `out_space`, `convert`, `release_dead`, `release_idle`), `usershader/parse.rs` (`// @brp space`), `offline_fx.rs` (`record(.., space, want)`); `brp-ai/src/gpu_temporal.rs` (`Temporal::convert_space`, the same GLSL on tensors); `brp-player/src/ai.rs` (`AiKind::space`, `stage_spaces`, `Job::spaces`), `effects.rs` (`--fx` takes a relative path to a `.glsl` file too). Test effects: `crates/brp-render/tests/spaces/` (pass-through in each space, linear gain).

**No effects: unchanged.** The direct path (`prepare_video` without a chain, `video.frag`, `pack.rs`) is untouched. Measured GPU memory with no effect, playing (`BRP_MEM_LOG`): 1080p 73 MB, 4K ProRes 870 MB, and no working picture at all. The effects switch now also frees the chain's pictures and history textures (`release_idle`): 1080p with a linear effect 141 MB → switched off 73 MB (before, the pool stayed allocated until the window closed).

**With effects (measured, RTX 5070 Ti):**

| | 1080p | 4K |
|---|---|---|
| Working pictures, one MAIN effect | 2 × rgba16f, 33 MB | 2, 127 MB |
| Same with a `linear` effect, or linear + display + linear | 2, 33 MB (unchanged) | 2, 127 MB |
| One conversion (GPU) | 0.006 ms | 0.03 ms |

A converted or processed picture that nothing else names is given back to the pool within the run (`release_dead`), so conversions cost no memory and a chain ping-pongs between two pictures (it used one per pass: a linear effect took 4 pictures, 255 MB at 4K, before this). All 24 bundled shaders give identical output with and without that reuse (`fxcheck`, 3 frames each, PNG and statistics). AI stages convert float32 tensors in place of the next stage's input (one kernel, same maths); their pictures are cached already converted.

**Pass-through: no shift** (`cargo run -p brp-render --release --example spacecheck -- --exr <seq> --4k`): effects that change nothing, compared after the display transform with no conversion. 8-bit levels, mean / 99.99th percentile / max:

| Source | linear | display | pq | mpv LINEAR | all stacked |
|---|---|---|---|---|---|
| SDR Rec.709 (test.mp4) | 0.18 / 4.7 / 5.2 | 0 (same space) | 0.30 / 7.7 / 9.1 | 0.06 / 0.22 / 0.26 | 0.36 / 7.5 / 10.3 |
| PQ Rec.2020 (phone HDR) | 0.11 / 0.60 / 0.80 | 0.11 / 0.42 / 0.77 | 0 (same space) | 0.11 / 0.60 / 0.80 | 0.22 / 0.76 / 1.12 |
| HLG (the PQ frame re-tagged) | 0.07 / 0.25 / 0.34 | 0.08 / 0.30 / 0.42 | 0.09 / 0.52 / 0.83 | 0.07 / 0.25 / 0.34 | 0.16 / 0.70 / 0.97 |
| Linear EXR (our export of the PQ clip) | 0.04 / 0.16 / 0.20 | 0.07 / 0.18 / 0.18 | 0.18 / 0.85 / 0.94 | 0 (same space) | 0.25 / 1.02 / 1.22 |

HDR sources and EXR stay under about one level everywhere. **SDR through Rec.2020 float16** is the one visible-in-numbers case: about 5% of channel values move by more than a level, up to 5 (linear) or 9 (pq). It is float16 rounding of the Rec.2020 values (half an ulp ≈ 2.4e-4) leaking through the 2020→709 matrix into a channel near zero of a saturated colour (G and B of a bright red), where the 2.4 gamma's steep start turns 6e-5 of light into 4–5 levels. In light it is ~1e-4 of the pixel's brightest channel, and the mean stays at 0.2 levels; without the gamut change (mpv LINEAR, linear in Rec.709) the same storage gives at most 0.26. Float32 linear pictures, or Rec.709 linear for Rec.709 sources, would remove it (twice the memory for those pictures, or a second linear space for effects to handle).

**Through export** (FFV1 10-bit 4:4:4, 12 frames, `--verify`; stack: linear, display, pq, mpv LINEAR pass-throughs): SDR plain export vs the player *with* the stack, mean 0.04–0.36 levels (max 10, as above); PQ: mean 0.17, max 0.37. Export with the stack vs the player with the same stack: max 0.50 (10-bit quantisation of the export). PQ with AI Denoise (a `display` model): export vs player mean 0.12, max 0.50; vs the plain export mean ≤ 0.55, max 9 (the denoising): highlights survive (2,400-nit pixels clipped at SDR white would differ by ~70 levels).

**HDR through linear effects:** a gain of ¼ in linear light on the PQ frame takes its brightest channel from 2398 to 597 nits; all 417,425 pixels above SDR white scale with an error ≤ 0.5% of the pixel's light (HLG: 354 → 88 nits, 0.3%).

**Not done / open:**
- HDR screens: done since, see §7b.
- The multi-layer EXRs in `footage/exr` did not open then (EXR was tested with a sequence our export made); they do now, see §7c.
- No model declares `linear` yet (the planned denoiser will, through `AiKind::space`); linear in the AI worker was exercised through shader stages and the conversion kernel only.

### Dev notes

- SDK paths are in `.cargo/config.toml` (FFmpeg 7.1 LGPL shared, Vulkan SDK, LLVM). FFmpeg DLLs are copied next to the exe by `crates/brp-player/build.rs`.
- Debug builds enable the Vulkan validation layer (`BRP_NO_VALIDATION=1` to turn off). `RUST_LOG=info,brp_render=debug` prints start-up timings.
- `brp-player <file> --frame=N` opens paused on frame N (used for visual checks).

## 15. Inputs

- The `old/` sources are enough; the full Python environment is not needed.
- Old effect/export presets do **not** need to load in the new app.
- Sample footage: provided in `footage/` (§13).
