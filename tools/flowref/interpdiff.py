"""Where an in-between picture differs from the real one.

    interpdiff.py truth.png out.png [out2.png ...]

Per output: PSNR over everything, without a 48 px border, and the share of the
error in that border; then the error per 4x4 block of the picture area (row-major)
for the worst-placed part.
"""
import sys
import numpy as np
from PIL import Image


def load(p):
    return np.asarray(Image.open(p).convert("RGB"), dtype=np.float64) / 255.0


truth = load(sys.argv[1])
h, w, _ = truth.shape
b = 48
for p in sys.argv[2:]:
    o = load(p)
    se = ((o - truth) ** 2).mean(axis=2)
    inner = se[b:h - b, b:w - b]
    psnr = lambda m: 10 * np.log10(1 / max(m, 1e-12))
    border_share = 1 - inner.sum() / se.sum()
    gy, gx = 4, 4
    blocks = [[se[y * h // gy:(y + 1) * h // gy, x * w // gx:(x + 1) * w // gx].mean() for x in range(gx)] for y in range(gy)]
    print(f"{p}: all {psnr(se.mean()):.2f} dB, inner {psnr(inner.mean()):.2f} dB, border holds {border_share * 100:.0f}% of the error")
    print("   blocks (dB): " + " | ".join(" ".join(f"{psnr(v):5.1f}" for v in row) for row in blocks))
