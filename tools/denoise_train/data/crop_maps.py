"""The noise-gate maps (patchnoise.py) of every cached camera crop, in one small file
each training worker loads once (loading a clip's map from disk per sample cost a file
open, ~3 ms on Windows, most of the time).

  python crop_maps.py <data dir>

<data>/bank/camera/index.json -> <data>/bank/camera/crop_maps.npy:
(crops, 3 (luma noise, chroma noise, flatness), size/16, size/16) float16, the
blocks covering each crop, in the crop's own block coordinates.
"""
import json, os, sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import patchnoise as pn


def main():
    data = sys.argv[1]
    bank = os.path.join(data, "bank", "camera")
    index = json.load(open(os.path.join(bank, "index.json")))
    n = max(e["size"] for e in index) // pn.B
    out = np.zeros((len(index), 3, n, n), np.float16)
    cache = {}
    for i, e in enumerate(index):
        m = cache.get(e["clip"]) or pn.load_maps(data, e["clip"])
        cache = {e["clip"]: m}
        by, bx, k = e["y"] // pn.B, e["x"] // pn.B, e["size"] // pn.B
        for j, key in enumerate(("y", "c", "grad")):
            out[i, j, :k, :k] = m[key][by : by + k, bx : bx + k]
    np.save(os.path.join(bank, "crop_maps.npy"), out)
    print(f"{len(index)} crops, {out.nbytes / 1e6:.0f} MB")


if __name__ == "__main__":
    main()
