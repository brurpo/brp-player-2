"""Print the render setup of the open .blend (run: blender -b scene.blend --python this).
Read-only: changes nothing, saves nothing."""
import bpy, json

s = bpy.context.scene
c = s.cycles
r = s.render
info = {
    "scene": s.name,
    "scenes": [x.name for x in bpy.data.scenes],
    "engine": r.engine,
    "camera": s.camera.name if s.camera else None,
    "frames": [s.frame_start, s.frame_end, s.frame_step],
    "resolution": [r.resolution_x, r.resolution_y, r.resolution_percentage],
    "fps": r.fps / r.fps_base,
    "samples": c.samples,
    "adaptive": [c.use_adaptive_sampling, c.adaptive_threshold],
    "seed": [c.seed, c.use_animated_seed],
    "denoise": [c.use_denoising, getattr(c, "denoiser", None)],
    "device": c.device,
    "motion_blur": r.use_motion_blur,
    "format": [r.image_settings.file_format, r.image_settings.color_depth, getattr(r.image_settings, "exr_codec", None)],
    "filepath": r.filepath,
    "view_layers": [
        {
            "name": vl.name,
            "use": vl.use,
            "denoising_data": getattr(vl.cycles, "denoising_store_passes", None),
            "noisy": getattr(vl.cycles, "pass_debug_sample_count", None),
            "vector": vl.use_pass_vector,
            "z": vl.use_pass_z,
        }
        for vl in s.view_layers
    ],
    "use_nodes": s.use_nodes,
    "view_transform": s.view_settings.view_transform,
}
import sys
out = sys.argv[sys.argv.index("--") + 1] if "--" in sys.argv else None
if out:
    open(out, "w").write(json.dumps(info, indent=1))
else:
    print("INSPECT " + json.dumps(info))
