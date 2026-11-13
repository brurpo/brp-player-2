"""Cut the camera clips (clips.csv) once into 512x512 crops for fast training reads:
decoding a clip's PNGs costs 0.3-0.9 s, a cached crop a few ms.

  python camera_bank.py <data dir> [--crops 2] [--size 512] [--workers 12]

Per clip, `--crops` crops of `--size` pixels (all its frames) at random places that pass
the noise gate (patchnoise.py) on the whole crop. Each frame of a crop is one chunk,
compressed with Blosc (zstd level 1, byte shuffle; lossless), into
<data>/bank/camera/crops.bin, with <data>/bank/camera/index.json listing, per crop:
clip, source, split, y, x (in the clip), frames, dtype (uint8 or uint16, the clip's
own depth) and each frame's offset and size in crops.bin. Rebuilt from scratch.
"""
import csv, json, os, sys
from concurrent.futures import ProcessPoolExecutor

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import patchnoise as pn

cv2.setNumThreads(1)


def codec():
    from numcodecs import Blosc

    return Blosc(cname="zstd", clevel=1, shuffle=Blosc.SHUFFLE)


def job(args):
    row, data, crops, size, seed = args
    r = np.random.default_rng(seed)
    gate = json.load(open(os.path.join(data, "noise_maps", "gate.json")))
    m = pn.load_maps(data, row["clip"])
    H, W = int(row["height"]), int(row["width"])
    size = min(size, H, W) // 16 * 16
    places = []
    for _ in range(40):
        if len(places) == crops:
            break
        y, x = int(r.integers(0, (H - size) // 16 + 1)) * 16, int(r.integers(0, (W - size) // 16 + 1)) * 16
        if pn.patch_ok(m, y, x, size, gate) and all(abs(y - a) + abs(x - b) > size // 2 for a, b in places):
            places.append((y, x))
    if not places:
        return row["clip"], []
    frames = []
    for f in row["frames"].split(";"):
        a = cv2.imread(os.path.join(row["dir"], f), cv2.IMREAD_UNCHANGED)
        frames.append(a[..., ::-1])
    c = codec()
    out = []
    for y, x in places:
        chunks = [c.encode(np.ascontiguousarray(a[y : y + size, x : x + size])) for a in frames]
        out.append({"y": y, "x": x, "size": size, "dtype": str(frames[0].dtype), "chunks": chunks})
    return row["clip"], out


def main():
    data = sys.argv[1]
    a = sys.argv[2:]
    opt = lambda k, d: int(a[a.index(k) + 1]) if k in a else d
    crops, size, workers = opt("--crops", 2), opt("--size", 512), opt("--workers", 12)
    rows = [r for r in csv.DictReader(open(os.path.join(data, "clips.csv"), encoding="utf-8")) if os.path.exists(os.path.join(data, "noise_maps", r["clip"] + ".npz"))]
    rowmap = {r["clip"]: r for r in rows}
    out_dir = os.path.join(data, "bank", "camera")
    os.makedirs(out_dir, exist_ok=True)
    binp = os.path.join(out_dir, "crops.bin.part")
    index = []
    pos = 0
    with open(binp, "wb") as f, ProcessPoolExecutor(workers) as ex:
        jobs = [(r, data, crops, size, i) for i, r in enumerate(rows)]
        for i, (clip, cs) in enumerate(ex.map(job, jobs, chunksize=4)):
            row = rowmap[clip]
            for c in cs:
                offs = []
                for ch in c["chunks"]:
                    f.write(ch)
                    offs.append([pos, len(ch)])
                    pos += len(ch)
                index.append({"clip": clip, "source": row["source"], "split": row["split"], "y": c["y"], "x": c["x"], "size": c["size"], "dtype": c["dtype"], "frames": offs})
            if i % 200 == 0:
                print(f"{i}/{len(rows)} clips, {len(index)} crops, {pos / 1e9:.1f} GB", flush=True)
    os.replace(binp, os.path.join(out_dir, "crops.bin"))
    json.dump(index, open(os.path.join(out_dir, "index.json"), "w"))
    print(f"done: {len(index)} crops from {len(rows)} clips, {pos / 1e9:.1f} GB")


if __name__ == "__main__":
    main()
