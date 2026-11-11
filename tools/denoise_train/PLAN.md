# Stable video denoiser: training plan

Goal: a learned denoiser for the player that removes grain, noise, blotches and compression damage and is **steady from frame to frame**. It should run at 1080p in real time if possible (a target, not a hard limit), as a model choice in the Denoise effect. Training runs on the RTX 5070 Ti, which is also used for other things, so it must pause and resume at any point.

Status: plan only. Nothing here is built yet.

**Two requirements shape everything below:**
- **Self-contained.** The model is the whole pipeline, alignment included. It must not rely on the player: anyone can run it from the PyTorch reference, an ONNX export, or our own Vulkan runner.
- **Resolution-independent.** One set of weights for any size, from SD to 4K and above; no whole-frame operations. Big frames run in overlapping tiles for memory.

---

## 1. What the network sees and returns

**Input window.** The current frame plus one neighbour on each side (t−1, t, t+1). At the clip's start, after a cut and at the end there is no neighbour on one side, so the network uses two on the other side instead: t, t+1, t+2 at the start, t−2, t−1, t at the end. The network always has three slots: current, "before" and "after". Each neighbour slot also gets a **validity map**, so it knows when a slot holds the second neighbour on the other side, or nothing at all. Training samples all of these layouts: normal, start, end, and a cut where one side is from another shot.

**Alignment inside the model: coarse to fine, learned.** A small motion network is part of the model, coarse-to-fine like SpyNet: a few layers per level, with as many levels as the frame size needs. It moves the neighbours (and the previous output) onto the current frame for the large motion. The denoiser behind it then fixes what is left:
- small misalignments (a few pixels, within its reach of about ±16 px);
- what to merge, pixel by pixel;
- what to ignore: occlusions, failed matches, other shots.

Both parts are trained together, so the alignment learns what helps denoising, which is not always the true motion. To keep the denoiser robust, training sometimes disturbs the alignment on purpose: small random offsets, wrong patches, a neighbour from another shot. Warping is bilinear sampling, which is `GridSample` in ONNX and our warp kernel in the player.

Why not let the denoiser align everything by itself: to match movement of tens of pixels it would need far more reach, meaning more layers or deformable convolutions, beyond the real-time budget. Without that, fast pans ghost or blur, the known weakness of window-only video denoisers. The coarse motion stage costs a few percent of the budget instead.

The player's own motion (`Temporal::motion`) is **not** used: the model would depend on the player. Two motion stages per frame would also be wasteful.

**The previous output, moved along the motion (recommended).** A ±1 window alone limits how steady the result can be: each output is still a fresh guess from three noisy frames, and that is where flicker comes from. Feeding the network its own previous output, moved onto the current frame, makes it recurrent:
- The result settles over many frames, so it is far steadier.
- It costs almost nothing.
- **The catch:** after a jump, a few earlier frames have to be replayed so it settles, which the player already does for RVM matting (`Reach`). The recurrent state (previous output, and the motion to it) is part of the model's documented interface, so other users can do the same.

Training then uses sequences of 6–10 frames, run through the network in order with gradients through the chain, not single windows.

**Noise strength input.** A per-pixel map of how much to remove, like FFDNet's. It starts from the noise level the player already measures (`denoise: noise profile`), and the user's Strength setting scales it. This gives one control from "light clean-up" to "remove everything", and lets real film grain be kept on purpose.

**Output.** The current frame cleaned, predicted as a correction added to the noisy frame (residual). Alpha passes through untouched.

## 2. Network size for real time

The budget comes from what we measured on our own Vulkan runner. NAFNet (Deblur) runs at about 970 thousand multiply-adds per pixel, and took about 250 ms at 1080p in f16 on tensor cores. That is about 8 trillion multiply-adds per second in practice. The pipeline itself already took 17–23 ms of a 33 ms frame with Steady Detail. Leaving about 12 ms for the network gives **about 100 billion multiply-adds per 1080p frame, roughly 40–50 thousand per pixel**.

A shape that fits:

| Part | Resolution | Channels | Cost per output pixel |
|---|---|---|---|
| Inputs, about 19 channels: 3 frames + 2 validity + 2 confidence + previous output + its validity + noise map | full | → pixel-unshuffle ×2 | — |
| Encoder level 1, 2 convolutions | ½ | 32 | ~5 k |
| Encoder level 2, 4 convolutions | ¼ | 64 | ~9 k |
| Bottleneck, 6 convolutions | ⅛ | 128 | ~14 k |
| Decoder (mirror, skips added) | ¼, ½ | 64, 32 | ~12 k |
| Pixel-shuffle ×2, add to the noisy frame | full | 3 | — |
| **Total** | | | **~40 k** |

Plus the motion stage: about 2–5 thousand per pixel, at 1/4 resolution and coarser.

Blocks use only layers our runner already has, so `pth2ncnn` converts them and they run on tensor cores: plain 3×3 convolutions, or NAFNet-style blocks (depthwise convolution, gating, 1×1). The same layers export to ONNX unchanged. No deformable or attention layers.

**Resolution independence:**
- **No whole-picture operations.** NAFNet's channel attention averages over the entire frame, so its behaviour changes with size; that is why it needs "TLC" (seen with Deblur). Here any channel attention uses a local box average.
- **Size padded to a multiple of 16.** The motion stage adds pyramid levels for bigger frames.
- **Tiles for memory at 4K and above,** overlapping by the network's reach (about 64 px), so there are no seams. The recurrent state is tiled the same way.
- **Training mixes scales:** native footage, footage downscaled 2–4×, noise and blotches of many sizes, motion from still to fast. Grain at 4K is bigger in pixels than at 720p, and both must work.

**Step 0 of the work:** build this graph with random weights in the runner and time it at 1080p, before any training. That fixes the channel widths to what actually fits 12 ms.

### Speed probe results (step 1, done)

`tools/denoise_train/model.py` is the model, `probe.py` the probe, both with random weights. Timed on the RTX 5070 Ti; times are per step (one output frame), and a step needs two motion estimates (before and after).

| Version | Multiply-adds per pixel | Our runner, 1080p | PyTorch fp16, 1080p | PyTorch fp16, 4K |
|---|---|---|---|---|
| Merge 32/64/128, motion 5×5 | 48 k | 6.5 ms + 2 × 3.5 ms = **13.5 ms** | 11.3 ms | 37.2 ms |
| Merge 32/64/128, motion 3×3 | 45 k | 6.5 ms + 2 × 0.6 ms = **7.7 ms** | 11.8 ms | 34.2 ms |
| Merge 48/96/192, motion 5×5 | 98 k | 11.5 ms + 2 × 3.5 ms = **18.5 ms** | 15.1 ms | 57.6 ms |
| Merge 48/96/192, motion 3×3 | ~95 k | 11.5 ms + 2 × 0.6 ms = **12.7 ms** | — | — |

- **Runner merge times** were measured at 1080p and 4K: 32/64/128 takes 6.5 ms at 1080p and 24.8 ms at 4K. The 4K motion time in the runner is an estimate, scaled by pixel count (about 12 ms per estimate with 5×5 filters).
- **All versions fit 1080p real time** in our runner. The pipeline around a picture took 1–8 ms more in earlier measurements, against 33 ms per frame at 30 fps.
- **4K** is about 30 ms with 32/64/128 and 3×3 motion, around real time; and about 50 ms with 5×5 motion.
- **3×3 motion is 6× cheaper** in our runner, whose 3×3 path is much faster than its 5×5. Smaller filters reach less per pyramid level, so the motion may need an extra refinement pass at each level, about 0.3 ms more. Training decides.
- **Budget:** about 30 fps, so 33 ms per frame for everything, as a target, not a hard limit. With 1–8 ms of pipeline that leaves about 25 ms for the model: room for 48/96/192 with 5×5 motion (18.5 ms), or wider.
- **Plan:** train 48/96/192 with 3×3 motion first (12.7 ms), and 32/64/128 alongside it as the fast option. Try 5×5 motion or wider merge levels where quality asks for it. Keep whichever is clearly better per millisecond.
- **ONNX:** the export matches PyTorch to 1e-7 and runs at other sizes. But tracing froze the number of pyramid levels and the warp's size scaling at the export size, so as exported it aligns wrongly at other sizes. Fix for step 7: export the motion level and the merge as separate graphs and run the pyramid loop in the calling code (`denoise_video.py`, the player). Or export with size-dependent operations computed from the tensor shapes.

## 2b. Two models: a teacher and a real-time student

The real-time budget limits what one network can learn. So there are two:

**Teacher (offline, no speed limit).** A large modern video restorer, trained first on the same data and losses:
- **Motion-guided deformable alignment,** as in BasicVSR++ and RVRT: the motion stage gives coarse offsets, a deformable convolution learns the fine sampling per pixel and per group of channels. This is the biggest single quality gain in recent video restoration.
- **Two-way propagation over the whole sequence** (past and future), not just a ±1 window.
- **NAFNet-style blocks**, wide (64–128 channels), with optional window attention at low resolution if it measurably helps.
- **It ships too:** as **"Best quality" for export**, and in ONNX. Expect seconds per 1080p frame, not milliseconds.

**Student (the real-time model).** The architecture of §1–2, made modern where it pays per millisecond:
- **Re-parameterised blocks** (RepVGG-style): trained with parallel 3×3, 1×1 and identity branches, folded into one 3×3 for inference. Better quality at no runtime cost.
- **NAFNet-style gating** where it runs fast in our runner. Depthwise convolutions save multiply-adds but are memory-bound on GPUs; on our runner dense 3×3 on tensor cores is often faster for the same quality (speed probe). Measure both.
- **Recurrent,** plus **deformable alignment at 1/4 resolution**: our runner already has the sampling layer (`DeformIm2col`, from BiRefNet).
- **Optional short look-ahead** (t+1, t+2) where playback works ahead, as it already does for AI effects.
- **Trained to imitate the teacher** (distillation): its loss compares the student's output with both the clean frame and the teacher's output. The teacher's outputs are steadier and more consistent targets than the clean frames alone. This usually beats training the small model alone.
- **After training:** prune channels that contribute little, and try int8 or fp8 on tensor cores where quality holds (fp16 is the baseline).

**Choosing.** Every option is timed in our runner at 1080p and scored on the test set (quality, and stability with the warp-error measure). The student keeps what gives the best quality per millisecond under the 33 ms budget.

**Training recipe for both.** bf16, AdamW, cosine schedule with warm-up, EMA weights, `torch.compile`, crops growing from 256² to 512², Charbonnier plus frequency loss, the stability loss, and a small perceptual term late in training.

**Cost.** The teacher adds roughly 1–2 nights of GPU time before the student, measured once the teacher's step speed is known; see §6.

## 3. Training data

**Clean sources.** Footage that is clean, or becomes clean when downscaled. Scaling a 4K source to 1080p or 720p averages most of its own noise away.

| Source | What it covers | Licence |
|---|---|---|
| REDS (NTIRE 2019) | 270 day-time sequences, 720p, real camera motion | CC BY 4.0 (author's page) |
| Blender Studio open films and their raw renders: Sintel, Tears of Steel (live action plus CG, 4K), Spring, Charge… | 3D renders, film-look live action, night-like scenes | CC BY (check each title) |
| **Our own Blender renders**: open demo scenes at low samples and high samples | Real path-tracing noise with exact clean pairs | Our renders of CC BY scenes |
| Pexels / Pixabay stock video (free licences) | Night, crowds, fast sport, drone, low light | Free to use; check redistribution terms. We would keep frames locally, not redistribute them |
| DAVIS 2017, YouTube-VOS | Varied motion | **Check before use**: the terms weren't confirmed |
| Netflix Open Content | High-end camera footage | **Per title**: Meridian is CC BY-NC-ND (no), others may be CC BY |
| Inter4K, Vimeo-90K | — | **Avoid unless confirmed**: possibly non-commercial or unclear |

Every source goes into a `manifest.csv` (file, source, licence, link). The model card lists them, since the weights ship with an open-source app. Nothing non-commercial is used.

**Preparation (offline).**
- Downscale 4K to 1080p and 720p.
- Split into shots with our cut detection, keeping cuts recorded so cut samples can be made on purpose.
- Store short clips of 8–12 frames as PNG or lossless FFV1.
- Drop clips that are already noisy (our noise profile above a threshold) or static.
- Night and low-light are the hardest to get clean. Use the cleanest stock clips there, downscaled, plus renders.

**Targets must be clean in time as well as in each frame.** A model learns to reproduce its targets. Checked by eye:
- **Tears of Steel has artificial grain added**, and downscaling removes only part of it.
- **Sintel flickers in some scenes**, from a Monte Carlo denoiser (OIDN/OptiX-like). That flicker would train the stability loss to flicker.

So every source gets a look plus measurements before use:
- **Noise per clip:** Immerkær, already in `prepare.py`.
- **Flicker per clip:** change of the broad picture (blurred) from frame to frame along the motion, where the motion is reliable.

Clips over either threshold are not targets. Grainy sources can be scaled further down (720p, 540p) until they measure clean. If a source's flicker can't be separated reliably, it is kept out of the stability loss (per-frame loss only) or dropped. The thresholds are checked against scenes picked out by eye.

**Noise is checked per patch, not only per clip.** A whole-frame noise figure hides grain in the shadows: the Netflix clips measured cleaner than REDS overall, yet their flat shadows carry 2–4× REDS's luma and chroma noise (`check/netflix_noise_sheet.png`). So:
- **Dropped:** Netflix BarScene1, DinnerScene and FoodMarket, grainy almost everywhere (43 clips, moved to `excluded/netflix`, marked in `sources.csv`).
- **Gate per patch** (`data/patchnoise.py`): every clip has a map of luma and chroma noise per 16×16 block, from three of its frames. A patch's score is the noise of its flat blocks only; a patch with too few flat blocks is all detail, which hides grain, and passes. The thresholds are REDS's 97th percentile, REDS being the clean reference (`noise_maps/gate.json`: luma 3.46, chroma 0.62 in 8-bit levels; chroma does most of the work). The sampler draws crops and drops those over either threshold, so a clip keeps its clean highlights and loses its grainy shadows.
  - Check: REDS keeps 98% of its 256×256 patches, the dropped grainy clips would keep 17%. Netflix keeps 66–100% per sequence (DrivingPOV 72%, RollerCoaster 66%, BoxingPractice 66%, the rest ≥ 90%). Random rejects viewed high-passed (`check/noise_gate_rejected_random.png`) show sensor speckle in their flat areas.
  - A first try scored the flattest quarter of every patch instead; it rejected fine detail (gravel, foliage, water spray) as noise.

**Mix of content to aim for.** Day, night, indoor, crowds, foliage, water, sky, text and graphics, animation, and 3D renders. Motion from still to very fast, and fades and flashes. Held out as test material, never trained on: the CRV drone clip, the black-curtain clip, and other real footage of yours.

**Colour space: linear light outside, perceptual inside.** HDR must survive (render highlights, PQ/HLG video, EXR sequences), so the model works with linear light, not display-referred values:
- **Interface:** linear light, values above 1.0 allowed. The caller declares what it has: linear (EXR, renders), PQ, HLG, or SDR Rec.709/sRGB. Display encodings are decoded to display-linear light in absolute terms, with SDR reference white at 203 nits. For SDR the grade is unknown but irrelevant: we need one consistent space, not scene-linear light. Decoding the encoding itself is exact.
- **Inside the network:** a perceptual encoding of that light (PQ, or a log curve), as OIDN's HDR mode and KPCN do. Raw linear values span many orders of magnitude: losses and gradients would be dominated by highlights, and the shadows, where noise shows most, would be neglected.
- **Losses** in the same perceptual encoding (picture, frequency and stability alike).
- **Output** goes back to linear and is re-encoded as the input was: SDR in, SDR out; HDR in, HDR out.
- **Training mixes all of it:** renders and Noisebase as linear HDR at randomised exposure (± 1–2 stops around each scene's level, from the `scene.txt` hint or Noisebase's `exposure`); Netflix and REDS decoded from SDR. Some SDR clips are also brightened into the HDR range, so highlights above SDR white are not only ever renders.
- **Camera footage is clean only down to its noise floor.** Netflix clips viewed at normal brightness are clean, but boosted their shadows show sensor noise (the worst three sequences are dropped, the rest gated per patch, see above), about 1–3 levels out of 255 at 4K, half that at 1080p. So for camera sources:
  - at most +0.5 stop of exposure change;
  - HDR highlights made only by stretching the bright range, never by lifting shadows;
  - the loss weighted down where the clean frame is near black (below ~2% luminance).

  Renders and Noisebase, whose references are clean into the shadows, keep the full ± 1–2 stops.
- **Where each degradation is made:** sensor-like noise (Poisson–Gaussian, read noise) in linear light; grain added in grading, compression and banding in the display encoding where they happen, then decoded. Monte Carlo noise comes as rendered, in linear.
- **The player** passes its frames with their transfer (it already tracks it per job), so SDR, HDR and EXR all go through unchanged in range.

## 4. Degradations (generated during training)

Each training clip gets a random chain of these. The same noise *settings* hold across the clip, while the noise itself changes per frame, except where noted.

| Kind | How | Varies |
|---|---|---|
| Sensor noise | Poisson–Gaussian on linear light, then the camera curve | Level, ISO-like gain, read noise |
| Grain, luma and colour | Gaussian or Laplacian, filtered for size (1–4 px) and clumping; per-channel correlation sets how colourful it is | Size, strength, colour vs monochrome, sparseness |
| Film grain | Grain particle model, stronger in the midtones | Size, density |
| **Blotches** (as on the black curtain) | Low-frequency colour blobs 8–64 px, strongest in shadows, partly kept between frames (heavy in-camera NR plus compression) | Size, strength, how long they stay |
| Path-tracing noise | Real: low-sample Blender renders. Synthetic: speckle that depends on brightness, plus fireflies (rare bright pixels) | Samples, firefly rate |
| Compression | Real encoders through ffmpeg/PyAV: MPEG-2, H.264, HEVC, VP9, AV1, JPEG, DV, ProRes Proxy, each at random quality and GOP. Applied *after* noise, as in a camera | Codec, bitrate/QP, chroma 4:2:0 |
| Aliasing | Decimate without filtering, then rescale; nearest-neighbour scaling; field problems as they reach a denoiser after deinterlacing (below) | Factor, field order |

**Field problems: leftovers, not raw interlacing.** Heavy combing (every other line from a different picture on fast motion) is the deinterlacer's job, and the player has one that runs before effects; rebuilding it would spend the real-time model on deinterlacing. A first version wove in the *next frame's* lines (twice a field's motion) and produced exactly that, so it was replaced by:
- **weave, motion-adaptive:** the other field from the picture half a frame later (optical flow, half way), woven in only where that motion is under 1–4 px (a light comb on edges); elsewhere the lines are rebuilt from one field (jaggies), as a motion-adaptive deinterlacer does;
- **bob:** one field per frame, alternating, lines rebuilt (averaged or repeated): jaggies plus a one-line up/down jitter on fine horizontal detail;
- **line doubling:** one field, lines repeated, the same field every frame.
| Fixed-pattern noise | Static pattern, hot pixels | Strength. Open question: remove it, or leave it as "content"? |
| Banding | Quantise to 6–8 bits, then compress | Bit depth |
| **Salt and pepper** | Isolated white/black pixels (1–2 px), fresh each frame; sometimes one channel only; for HDR renders the salt is brighter than white. Likelier on renders (35%) than footage (12%) | Rate 1e-4–1e-2 |

No film damage (scratches, dust, line dropouts): decided against.

**Implemented: `degrade.py`** (`degrade(clean, rng, kind)` → noisy clip, noise map, list of steps), **`gallery.py`** (writes `check/gallery_<seed>.png`).
- Kinds: `camera` (sensor noise 75%, grain, film grain, blotches, aliasing, banding, compression 65%) and `render` (render noise 70%, fireflies 50%, salt and pepper, lighter compression). 5% of clips come back clean or only lightly compressed.
- Render noise is modelled as a renderer averages: N samples per pixel (1–64), each missing the light with some probability or carrying a heavy-tailed weight, unbiased. A first version (soft multiplicative speckle) looked nothing like Noisebase at low samples; this one does.
- Hot pixels (fixed position) are in at 3%; whether to keep them is still the open question above.
- Compression: real encoders through PyAV 19 (bundles x264, x265, VP9, SVT-AV1, MPEG-2/4, MJPEG, DV, ProRes); libaom is not in it, SVT-AV1 covers AV1. x265 fails on about 1 in 40 encodes (an argument error from the muxer, not reproducible with the same seed); such encodes fall back to H.264 and are counted in `degrade.FAILURES`.
- **Speed: about 0.2 s per clip of 8 frames at 256×256 on one CPU core**, about half of it compression. At ~10 steps/s with batches of 8 that is 16 of the 24 threads. Options when the training script exists: move the noise steps to the GPU (torch), keep only compression on CPU; or reuse each compressed clip for a few steps.
- Gallery checked by eye (seeds 0 and 1): camera noise, blotches, weave combing, banding and compression look right; render noise at 2–7 spp matches Noisebase at 1–4 spp.

## 5. Losses

- **Picture:** Charbonnier (smooth L1) on the output against the clean frame, plus a small frequency loss on the FFT magnitude, so the model doesn't blur texture to win on L1.
- **Stability**, the point of the project: each output is moved onto the next along the motion of the *clean* frames. The difference between consecutive outputs is compared with that between consecutive clean frames, only where the motion is reliable:
  `|(out_t − W(out_t−1)) − (gt_t − W(gt_t−1))|`, masked by the clean frames' forward–backward check.
  This penalises changes that the clean video doesn't have (flicker) without forcing still pictures.
- **Optional later:** a perceptual (LPIPS-type) term at low weight, if outputs look plastic. Check its licence: LPIPS is BSD.

## 6. Training run

- **PyTorch**, bf16 mixed precision, AdamW, cosine schedule with warm-up, an EMA copy of the weights that is the one exported.
- **Batches:** 256×256 crops of 8-frame sequences, batch 8–16 on 16 GB (tune). Later fine-tune on 512×512 crops so the bottleneck sees bigger structures.
- **Curriculum:** start with single-step windows (no recurrence), then switch to recurrent sequences of 6–10 frames.
- **Order of runs:**
  1. Window only.
  2. Plus the previous output (recurrent).
  3. Plus the noise map.

  Measure each against the test set before adding the next, so we know what each part buys.
- **Measured on the RTX 5070 Ti** (`bench_train.py`, random data, while another program used 95% of the GPU):
  - 48/96/192 with 3×3 motion, batch 8 × 8-frame recurrent sequences of 256²: **10 steps/s, 1.7 GB of GPU memory**.
  - 32/64/128: 8.7 steps/s, 1.3 GB. At this size the fixed overhead per step dominates.
  - So 300k steps take about **8–10 h of GPU time**: a couple of nights with pause and resume. Memory leaves room for bigger batches or crops.
- **The data feed is the real limit:** 10 steps/s consumes about 800 frames per second.
  - Training patches are cut once from the clips into 512×512 uncompressed arrays read straight from disk (no PNG decoding while training).
  - Codec damage is made in worker processes with PyAV (no ffmpeg process per sample), plus a pre-generated bank of compressed variants.
  - The machine has 24 CPU threads.

## 7. Pause and resume

- **Checkpoints** go out every N minutes and on Ctrl+C or a `pause` file. They are written to a temporary file and then renamed, so a crash never leaves a broken one. The last K are kept, plus the best on validation.
- **A checkpoint holds** the model and EMA weights, optimiser, schedule, mixed-precision scaler, step and epoch, the data sampler's position and seed, all RNG states (Python, NumPy, torch CPU and CUDA), and the config with a hash of the code.
- **Resuming** continues from exactly the same point; `--resume` picks the latest. Changing the config on resume is refused unless `--force`.
- **Quiet mode** (`--low-priority`) lowers the process priority and adds a sleep between steps, so the GPU stays usable for other things. A `pause` file in the run folder makes training stop at the next step and wait: drop it in to free the GPU for a game, delete it to continue.
- Logs go to CSV plus TensorBoard: losses, validation metrics, and sample pictures every few thousand steps.

## 8. Testing

| Measure | How |
|---|---|
| Quality | PSNR and SSIM against clean frames, per kind of degradation and per content type |
| Stability | Temporal warp error, as in `tools/airef/shimmer_score.py` and `matte_stability.py`: change along the motion where it is reliable, minus the clean video's own change |
| Real footage | CRV drone shot, the black-curtain clip and others: side-by-side clips plus the stability score (no clean reference there) |
| Speed | The exported graph in the player at 1080p: ms per frame, playback rate |
| Comparison | The current built-in Denoise on the same material |

## 9. Deliverables and integration

The model ships as a package that works without the player:
- **Reference:** a plain-PyTorch `model.py`. One call takes the window frames, the validity flags, the recurrent state and an optional strength map; it returns the cleaned frame and the new state. Tiling is a helper.
- **Weights:** safetensors, with a model card holding the data manifest, licences and measured results.
- **ONNX export** in two graphs: one step of the model, and the motion stage on its own (for tiling and caching). Standard operations only, so it runs in ONNX Runtime, TensorRT and OpenVINO.
- **`denoise_video.py`:** cleans a video file through ffmpeg (frames in, frames out, state carried, cuts detected). It is the usage example and the test outside the player.

In the player:
1. `pth2ncnn` converts the same weights. Warping uses our existing warp kernel; it is matched to the PyTorch reference within f16 accuracy, as for the other models.
2. `brp_ai::denoise_net` runs the model's own motion stage and network, fills the window slots at clip ends and cuts, keeps the recurrent state, and replays after a jump.
3. Denoise gets **Model: Built-in | AI (temporal)**, with Strength feeding the noise map.
4. `tools/fetch_ai.sh` gets the weights.

## 10. Decisions to make first

1. **Settled:** alignment is a learned coarse-to-fine motion stage inside the model; the denoiser fixes what is left and chooses what to merge.
2. **Recurrent** (previous output fed back, recommended): much steadier, needs replay after jumps. Or **window only**.
3. **Film grain:** remove it always, or keep an amount set by the user through the noise map?
4. **Fixed-pattern noise and dust:** remove, or leave alone?
5. **Where it trains:** this PC (in quiet mode, around your use), or rented cloud GPUs for the long run. Measured: the student trains at about 10 steps/s here (§6). The teacher will be slower, to be measured.
6. **Settled:** two models: a large teacher (offline, "Best quality" export) and a real-time student distilled from it (§2b).

## 11. Order of work

1. ~~Speed probe~~: done, see the results in §2. ONNX Runtime here has only the CPU backend, so it was used to check the export, not to time it.
2. Data: download, licence manifest, preparation into clips.
3. Degradation chain and its gallery (§4).
4. The motion stage: pre-train it alone on clean pairs, against flow from an existing estimator, with a photometric loss, so joint training starts from sensible alignment.
5. Training script with pause and resume (§6–7); a short run to check everything moves.
6. **Teacher:** build it (§2b), measure its step speed, train it, test it (§8).
7. **Student:** runs 1–3 (§6), distilled from the teacher, with re-parameterised blocks and deformable alignment at 1/4 compared by quality per millisecond; then pruning and lower precision.
8. Exports (ONNX, our runner) of both models, `denoise_video.py`, reference match, player integration (§9): the student as "AI (temporal)" in Denoise, the teacher as "Best quality" for export.
