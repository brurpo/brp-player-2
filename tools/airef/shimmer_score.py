"""Shimmer of a frame sequence: how much the picture changes along the motion, and
how much its sharpness jumps from frame to frame.

  python shimmer_score.py <flows fwd> <flows bwd> <x0,y0,x1,y1 | all> <dir>...

Each dir holds the frames as PNGs in order (any names, sorted); flows as `aicheck
searaft` writes them for the same frames (fwd in order, bwd reversed). Reports, per
dir: mean |frame_t - frame_t-1 moved along the motion| over the region (where the
motion passes a forward-backward check), and the mean frame-to-frame change of the
sharpness (variance of the luma Laplacian over the whole frame) with its mean.
"""
import sys, os
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

fw, bw, reg = sys.argv[1], sys.argv[2], sys.argv[3]
dirs = sys.argv[4:]
dev = "cuda"


def ld(p, H, W):
    raw = open(p, "rb").read()
    c, h, w = np.frombuffer(raw[:12], "<u4")
    f = torch.from_numpy(np.frombuffer(raw[12:], "<f4").reshape(c, h, w).copy()).to(dev)
    return F.interpolate(f[None], size=(H, W), mode="bilinear", align_corners=False)[0] * (W / w) if (h, w) != (H, W) else f


for d in dirs:
    names = sorted(f for f in os.listdir(d) if f.endswith(".png"))
    img = [torch.from_numpy(np.asarray(Image.open(os.path.join(d, f)).convert("RGB"), dtype=np.float32) / 255).permute(2, 0, 1).to(dev) for f in names]
    n = len(img)
    H, W = img[0].shape[1:]
    ys, xs = torch.meshgrid(torch.arange(H, device=dev, dtype=torch.float32), torch.arange(W, device=dev, dtype=torch.float32), indexing="ij")

    def sample(a, flow):
        x, y = xs + flow[0], ys + flow[1]
        g = torch.stack([x / (W - 1) * 2 - 1, y / (H - 1) * 2 - 1], -1)[None]
        return F.grid_sample(a[None], g, mode="bicubic", align_corners=True, padding_mode="border")[0], (x >= 0) & (x <= W - 1) & (y >= 0) & (y <= H - 1)

    x0, y0, x1, y1 = (0, 0, W, H) if reg == "all" else map(int, reg.split(","))
    ch = []
    for t in range(1, n):
        b = ld(f"{bw}/{n - 1 - t:03d}.f32", H, W)
        f = ld(f"{fw}/{t - 1:03d}.f32", H, W)
        prev, inside = sample(img[t - 1], b)
        f_at, _ = sample(f, b)
        tr = (inside & ((b + f_at).norm(dim=0) < 1.0))[y0:y1, x0:x1]
        dd = (img[t] - prev).abs().mean(0)[y0:y1, x0:x1]
        ch.append(dd[tr].mean().item())
    sh = []
    for a in img:
        l = 0.299 * a[0] + 0.587 * a[1] + 0.114 * a[2]
        lap = 4 * l[1:-1, 1:-1] - l[:-2, 1:-1] - l[2:, 1:-1] - l[1:-1, :-2] - l[1:-1, 2:]
        sh.append((lap * 255).var().item())
    sh = np.array(sh)
    print("%-28s change along the motion %.4f; sharpness %.0f, jumps %.1f%%" % (os.path.basename(d.rstrip("/")), np.mean(ch), sh.mean(), np.abs(np.diff(sh)).mean() / sh.mean() * 100))
