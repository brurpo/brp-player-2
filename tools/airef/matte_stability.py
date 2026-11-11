"""Temporal stability of alpha mattes (exported videos of the matte, gray).

  python matte_stability.py <frames dir> <flows fwd> <flows bwd> <w> <h> <matte.gray>...

Frames: the source pictures (000.png ...); flows as `aicheck searaft` writes them:
fwd from the frames in order (file t: t -> t+1), bwd from them reversed (file k:
N-1-k -> N-2-k). A .gray file is the matte as raw 8-bit frames (ffmpeg -f rawvideo
-vf format=gray).

Score per consecutive pair, over pixels the motion trusts (inside the picture,
forward-backward check under 1 px): the mean |a_t(p) - a_t-1(p + b(p))|, the matte
compared with the previous one moved along the motion: what changed that the
motion does not explain. Reported over the soft edges (0.02 < a < 0.98 in either)
and over all trusted pixels; lower is steadier.
"""
import sys, os
import numpy as np
import torch
import torch.nn.functional as F


def ld(p):
    raw = open(p, "rb").read()
    c, h, w = np.frombuffer(raw[:12], "<u4")
    return np.frombuffer(raw[12:], "<f4").reshape(c, h, w).copy()


frames, fw, bw, W, H = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4]), int(sys.argv[5])
mattes = sys.argv[6:]
n = len([f for f in os.listdir(frames) if f.endswith(".png")])
dev = "cuda" if torch.cuda.is_available() else "cpu"


def sample(img, flow):
    """img [C,h,w] at p + flow(p) (flow [2,h,w] in pixels), bilinear; and inside mask."""
    C, h, w = img.shape
    ys, xs = torch.meshgrid(torch.arange(h, device=dev, dtype=torch.float32), torch.arange(w, device=dev, dtype=torch.float32), indexing="ij")
    x, y = xs + flow[0], ys + flow[1]
    inside = (x >= 0) & (x <= w - 1) & (y >= 0) & (y <= h - 1)
    g = torch.stack([x / (w - 1) * 2 - 1, y / (h - 1) * 2 - 1], -1)[None]
    return F.grid_sample(img[None], g, mode="bilinear", align_corners=True)[0], inside


def up(f):
    """A flow at its own size to the picture's (values scaled)."""
    f = torch.from_numpy(f).to(dev)
    if f.shape[1:] == (H, W):
        return f
    k = W / f.shape[2]
    return F.interpolate(f[None], size=(H, W), mode="bilinear", align_corners=False)[0] * k


fwd = [up(ld(f"{fw}/{t:03d}.f32")) for t in range(n - 1)]  # t -> t+1
bwd = [None] + [up(ld(f"{bw}/{n - 1 - t:03d}.f32")) for t in range(1, n)]  # t -> t-1
trust = [None]
for t in range(1, n):
    b = bwd[t]
    f_at, inside = sample(fwd[t - 1], b)  # forward flow of t-1 at where p lands
    err = (b + f_at).norm(dim=0)
    trust.append(inside & (err < 1.0))

for m in mattes:
    a = np.fromfile(m, dtype=np.uint8).reshape(-1, H, W)
    assert a.shape[0] >= n, (m, a.shape)
    a = torch.from_numpy(a[:n].astype(np.float32) / 255.0).to(dev)
    soft, all_ = [], []
    for t in range(1, n):
        prev, _ = sample(a[t - 1][None], bwd[t])
        prev = prev[0]
        d = (a[t] - prev).abs()
        tr = trust[t]
        edge = tr & (((a[t] > 0.02) & (a[t] < 0.98)) | ((prev > 0.02) & (prev < 0.98)))
        if edge.sum() > 0:
            soft.append(d[edge].mean().item())
        all_.append(d[tr].mean().item())
    print(f"{os.path.basename(m)}: soft edges {np.mean(soft):.4f}, all trusted {np.mean(all_):.5f} (trusted {np.mean([t.float().mean().item() for t in trust[1:]]) * 100:.0f}%)")
