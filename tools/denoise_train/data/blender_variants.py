"""Render the noisy training variants of the open .blend (inside Blender):

  blender -b scene.blend --python blender_variants.py -- <out dir> [variant ...] [--frames A-B]

Variants (as the classroom renders): 16, 64 and 128 samples with adaptive sampling
off, and 1024 samples with the adaptive threshold at 0.1 ("1024T"), each with a
static seed and with an animated (random per frame) seed:

  16_static_seed 16_random_seed 64_static_seed 64_random_seed
  128_static_seed 128_random_seed 1024T_static_seed 1024T_random_seed

Everything else stays as saved in the .blend: camera, frames, resolution, light
paths, OIDN (so the EXR holds both the denoised Combined and the Noisy Image),
the Vector pass, multilayer half-float EXR with ZIP, and every view layer as saved
(some scenes build the picture in the compositor from several layers: Blender-3.0's
"Full" layer alone is only the backdrop). Frames already written are skipped
(resumable). The .blend is not saved.
"""
import bpy, os, sys, time, json

argv = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
out_root = argv[0]
frames = None
names = []
i = 1
while i < len(argv):
    if argv[i] == "--frames":
        a, b = argv[i + 1].split("-")
        frames = (int(a), int(b))
        i += 2
    else:
        names.append(argv[i])
        i += 1
ALL = [f"{s}_{k}_seed" for s in ("16", "64", "128", "1024T") for k in ("static", "random")]
names = names or ALL

s = bpy.context.scene
c = s.cycles
r = s.render
if frames:
    s.frame_start, s.frame_end = frames
r.use_overwrite = False
# Keep the scene loaded between frames (otherwise every frame re-syncs: 6 of 8.5 s per
# frame on barbershop at 16 samples).
r.use_persistent_data = True
r.use_placeholder = True
r.image_settings.file_format = "OPEN_EXR_MULTILAYER"
r.image_settings.color_depth = "16"
r.image_settings.exr_codec = "ZIP"
c.use_denoising = True

log = {"blend": bpy.data.filepath, "layers": [vl.name for vl in s.view_layers if vl.use], "variants": {}}
for name in names:
    spp, seed = name.split("_")[0], name.split("_")[1]
    if spp.endswith("T"):
        c.samples = int(spp[:-1])
        c.use_adaptive_sampling = True
        c.adaptive_threshold = 0.1
    else:
        c.samples = int(spp)
        c.use_adaptive_sampling = False
    c.seed = 0
    c.use_animated_seed = seed == "random"
    d = os.path.join(out_root, name)
    os.makedirs(d, exist_ok=True)
    r.filepath = d + os.sep
    t = time.time()
    bpy.ops.render.render(animation=True)
    log["variants"][name] = round(time.time() - t, 1)
    print(f"VARIANT {name}: {log['variants'][name]} s", flush=True)
    json.dump(log, open(os.path.join(out_root, "render_log.json"), "w"), indent=1)
