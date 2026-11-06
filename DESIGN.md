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
- [ ] HDR (PQ/HLG) and linear EXR handling

**Export**
- [ ] Containers/codecs from `ui_export_dialog.py` (H.264/H.265, ProRes family, DNxHR family, XDCAM, AV1, VP9, GIF, JPG/PNG/EXR sequences)
- [ ] DXV3 and DXV3 with alpha (DXT5, CVExplorer patch), playback and export (§11b)
- [ ] Hardware encoders when available (any vendor), software fallback
- [ ] Resolution presets/custom, Fit/Fill/Stretch, quality/bitrate, alpha export, HDR export
- [ ] Export In/Out range exactly, with deinterlace, effects and OCIO baked in (same engine as viewer)
- [ ] Export presets (new format), progress + ETA, cancel
- [ ] Batch export queue with per-item status

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

- **Playback:** the stock `dxv` decoder already reads DXT5 (it treats it as DXT4 and un-premultiplies), so these files decode with the normal FFmpeg libraries. Must be verified with a real CVExplorer DXV3-alpha file: alpha exact, colours not double-premultiplied.
- **Faster path (later):** DXV frames are DXT blocks after LZ-style unpacking. We can upload them straight to the GPU as BC1/BC3 compressed textures (supported on every desktop vendor) and skip CPU DXT decompression. Premultiplied alpha is then handled in the shader.
- **Export:** "MOV / DXV3" and "MOV / DXV3 (alpha)" presets. Needs our bundled FFmpeg to be built **with the CVExplorer patch applied** (we build FFmpeg ourselves for export anyway, see licensing below). Alternative if the patch stops applying to newer FFmpeg: our own DXT5 + DXV packer in Rust feeding the MOV muxer.
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

Not covered yet (would be good to add later): **a DXV3-alpha file made by CVExplorer** (§11b), VP9 with alpha (WebM), ProRes / DNxHR, HLG, AV1, a long real-world clip (minutes, to test indexing and cache limits).

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

### Dev notes

- SDK paths are in `.cargo/config.toml` (FFmpeg 7.1 LGPL shared, Vulkan SDK, LLVM). FFmpeg DLLs are copied next to the exe by `crates/brp-player/build.rs`.
- Debug builds enable the Vulkan validation layer (`BRP_NO_VALIDATION=1` to turn off). `RUST_LOG=info,brp_render=debug` prints start-up timings.
- `brp-player <file> --frame=N` opens paused on frame N (used for visual checks).

## 15. Inputs

- The `old/` sources are enough; the full Python environment is not needed.
- Old effect/export presets do **not** need to load in the new app.
- Sample footage: provided in `footage/` (§13).
