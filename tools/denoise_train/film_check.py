"""Film grain look check: a few clips, each clean and with several grain settings
(gpu_degrade.film_grain), 2x zoom, to <data>/check/film_grain.png.

  python film_check.py [--data D:/denoise_data]
"""
import os, sys

import cv2
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import dataset as ds
import degrade as D
import gpu_degrade as G

SETTINGS = [
    ("clean", None),
    ("negative fine", {"r": 0.07, "spread": 0.2, "mono": False, "blur": 0.6, "strength": 0.6, "chroma": 0.3, "negative": True}),
    ("negative medium", {"r": 0.12, "spread": 0.3, "mono": False, "blur": 0.8, "strength": 0.8, "chroma": 0.35, "negative": True}),
    ("negative coarse", {"r": 0.22, "spread": 0.3, "mono": False, "blur": 1.1, "strength": 1.0, "chroma": 0.5, "negative": True}),
    ("negative b&w", {"r": 0.12, "spread": 0.3, "mono": True, "blur": 0.8, "strength": 0.9, "negative": True}),
    ("print medium", {"r": 0.12, "spread": 0.3, "mono": False, "blur": 0.8, "strength": 0.8, "chroma": 0.35, "negative": False}),
]


def main():
    data = sys.argv[sys.argv.index("--data") + 1] if "--data" in sys.argv else "D:/denoise_data"
    cam = ds.Camera(ds.DataConfig(data=data), "train")
    g = torch.Generator(device="cuda")
    g.manual_seed(0)
    rows = []
    for seed in (11, 23, 5):
        clip, _, _, origin = cam.sample(np.random.default_rng(seed), 2, 256)
        v = torch.from_numpy(np.ascontiguousarray(clip.transpose(0, 3, 1, 2))).cuda().float()  # sRGB values
        tiles = []
        for label, p in SETTINGS:
            out = v if p is None else G.film_grain(v, p, g)
            a = (out[0].clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255 + 0.5).astype(np.uint8)
            a = cv2.resize(a[64:192, 64:192], (256, 256), interpolation=cv2.INTER_NEAREST)[..., ::-1].copy()
            cv2.putText(a, label, (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)
            tiles.append(a)
        rows.append(np.concatenate(tiles, 1))
        print(origin)
    path = os.path.join(data, "check", "film_grain.png")
    cv2.imwrite(path, np.concatenate(rows, 0))
    print("sheet:", path)


if __name__ == "__main__":
    main()
