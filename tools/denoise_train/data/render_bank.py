"""Decode the Blender render pairs once into raw half-float frames for fast training
reads (decoding a crop from the EXRs costs ~0.15 s; from these, a few ms).

  python render_bank.py <data dir> [--workers N]

<data>/src/3D_RENDERS/<scene>/{GT,<variant>}/*.exr  ->
<data>/bank/renders/<scene>/<name>.f16   (frames, height, width, 3) float16, raw
<data>/bank/renders/<scene>/meta.json    names, frame numbers, shape, layer

Per scene: GT (the farm's denoised Combined), the 8 noisy variants' "Noisy Image"
(raw path-tracing noise), and the denoised "Combined" of the 16 and 64 sample variants
(Blender's denoiser, with its flicker: inputs too, cleaned towards GT). Frames are
matched by the number at the end of the file name (the farm prefixes GT names).
Resumable: finished files are kept (written to .part, then renamed).
"""
import json, os, re, sys
from concurrent.futures import ProcessPoolExecutor
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from exr import Exr

VARIANTS = [f"{s}_{k}_seed" for s in ("16", "64", "128", "1024T") for k in ("static", "random")]
DENOISED = [f"{s}_{k}_seed" for s in ("16", "64") for k in ("static", "random")]


def frames(d):
    out = {}
    for n in os.listdir(d):
        m = re.search(r"(\d+)\.exr$", n)
        if m:
            out[int(m.group(1))] = os.path.join(d, n)
    return out


def main_layer(path):
    """The view layer with the Vector pass (the one the pairs share)."""
    e = Exr(path)
    names = [n for n, _ in e.channels]
    for n in names:
        if n.endswith(".Vector.X") or n.endswith(".Vector.x"):
            return n[: -len(".Vector.X")]
    raise ValueError(f"{path}: no Vector pass")


def job(args):
    paths, layer, pas, out = args
    if os.path.exists(out):
        return out
    first = Exr(paths[0])
    arr = np.lib.format.open_memmap(out + ".part.npy", "w+", np.float16, (len(paths), first.height, first.width, 3))
    for i, p in enumerate(paths):
        arr[i] = Exr(p).read(f"{layer}.{pas}").astype(np.float16)
    arr.flush()
    del arr
    os.replace(out + ".part.npy", out)
    return out


def main():
    data = sys.argv[1]
    a = sys.argv[2:]
    workers = int(a[a.index("--workers") + 1]) if "--workers" in a else 8
    src = os.path.join(data, "src", "3D_RENDERS")
    bank = os.path.join(data, "bank", "renders")
    jobs, metas = [], {}
    for scene in sorted(os.listdir(src)):
        d = os.path.join(src, scene)
        if not os.path.isdir(os.path.join(d, "GT")):
            continue
        gt = frames(os.path.join(d, "GT"))
        common = set(gt)
        for v in VARIANTS:
            common &= set(frames(os.path.join(d, v))) if os.path.isdir(os.path.join(d, v)) else set()
        nums = sorted(common)
        if not nums:
            print(f"{scene}: no complete frames, skipped")
            continue
        layer = main_layer(gt[nums[0]])
        os.makedirs(os.path.join(bank, scene), exist_ok=True)
        entries = [("GT", "Combined", [gt[n] for n in nums])]
        for v in VARIANTS:
            fv = frames(os.path.join(d, v))
            entries.append((f"{v}.noisy", "Noisy Image", [fv[n] for n in nums]))
            if v in DENOISED:
                entries.append((f"{v}.denoised", "Combined", [fv[n] for n in nums]))
        for name, pas, paths in entries:
            jobs.append((paths, layer, pas, os.path.join(bank, scene, f"{name}.f16.npy")))
        h, w = Exr(gt[nums[0]]).height, Exr(gt[nums[0]]).width
        metas[scene] = {"layer": layer, "frames": nums, "shape": [len(nums), h, w, 3], "names": [e[0] for e in entries]}
        print(f"{scene}: layer {layer}, {len(nums)} frames, {len(entries)} arrays")
    with ProcessPoolExecutor(workers) as ex:
        for i, out in enumerate(ex.map(job, jobs)):
            print(f"[{i + 1}/{len(jobs)}] {os.path.relpath(out, bank)}", flush=True)
    for scene, m in metas.items():
        json.dump(m, open(os.path.join(bank, scene, "meta.json"), "w"), indent=1)
    print("done")


if __name__ == "__main__":
    main()
