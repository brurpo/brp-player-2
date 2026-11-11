# Blender renders for the denoiser: what to render

Each scene gives a **clean plate** (the training target) and several **degraded versions of the same frames** (training inputs). Same camera, same frames, same everything except sampling and denoising, so every pair lines up pixel for pixel.

## The clean plate (target)

- **Cycles, 4096 samples, adaptive sampling off.** There must be no visible noise on a 200% zoom, in shadows, glass, volumes and depth of field too: those are where low-sample noise hides last.
- **Denoised and raw, both saved.** At 4096 samples the denoiser has almost nothing left to remove, so its flicker (the Sintel problem) should be negligible. It can still decide fine details frame by frame: hair, foliage edges, small highlights. Keep the raw beauty and the denoised layer in the same multilayer EXR. Per scene we measure the difference and the flicker of each, and use the denoised one where it is steady.
- **Anti-aliasing:** the default pixel filter (Blackman-Harris, 1.5 px) is fine. Don't drop the filter width below 1.0: that makes aliasing, which the targets must not have.
- **Motion blur on** for most scenes (cameras have it), off for some. Vary the shutter (0.25–0.5).
- **Light paths:** clamp indirect light at a high value (or not at all). Strong clamping changes the image between the clean and noisy renders.
- **Linear, as Blender writes EXR:** the model works in linear light (HDR kept) with a perceptual encoding inside, so no view transform is applied. Highlights above 1.0 are welcome; don't clamp them. Note in `scene.txt` the exposure you'd use to look at the scene: training varies exposure around it.

## The degraded versions (inputs)

Same scene, same frames, same output settings, only these changed:

| Version | Samples | Seed | Denoiser |
|---|---|---|---|
| Low samples | several levels per scene: 1, 4, 16, 64 (and 256 for subtle noise) | **animated** (Render > Sampling > Advanced > Seed: the clock icon), so the noise changes every frame as on a camera | off |
| Stuck noise | 16 or 64 | **fixed** (not animated): the noise stays on the screen while the picture moves. Real renders sometimes have this | off |
| Denoiser flicker | 4–64, a couple of levels | animated | **OIDN and/or OptiX** on, as artists use them. Its flicker is what the model must learn to remove |

Adaptive sampling **off** for the low-sample versions, or the noise level depends on its threshold, not the sample count.

## Output

- **Resolution:** 1920×1080 (or 3840×2160 if render time allows: we can downscale; we can't add detail).
- **Format:** OpenEXR half float (RGBA, 16-bit) is best. 16-bit PNG is fine. No JPEG, no video codecs.
- **Passes:**
  - Clean plate: beauty (raw), the denoised beauty, and *Vector* (32-bit float if possible).
  - Low-sample versions: beauty only, plus the denoised beauty for the denoiser-flicker versions.
  - No Z-depth: the model only sees colour, as on real footage. The vector pass gives the exact motion for measuring stability.
  - Layer names are read from the EXR headers, so Blender's own names are fine.
- **Length:** at least 24 frames per shot. 48–120 is better: training takes 10-frame windows from anywhere in a shot. Several short shots beat one long one.
- **Folders:**
  ```
  <scene>/clean/0001.exr ...
  <scene>/spp1/0001.exr ...
  <scene>/spp4/...
  <scene>/spp16_fixedseed/...
  <scene>/spp16_oidn/...
  <scene>/scene.txt   (source .blend and its licence, author, samples, anything unusual)
  ```

## What to vary across scenes

- **Light:** day exterior, interior with practical lights, night with few bright sources, candle or firelight, neon, and volumes (fog, god rays).
- **Materials:** glass, water, hair or fur, foliage, metal and other glossy surfaces, subsurface (skin, wax).
- **Motion:** locked camera, slow pan or dolly, fast move, moving objects, depth of field racks.
- **Content:** characters, architecture, nature, fine regular patterns (fences, grilles, fabric). Those alias, and the clean plate must not.

## Licences

The scene files you start from must allow derivative use (CC0, CC BY, or your own). Note the source and licence of each in its `scene.txt`; they go into `sources.csv`. The Blender demo files page lists each file's licence.
