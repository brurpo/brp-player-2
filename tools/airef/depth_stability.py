"""Temporal stability of depth sequences (from depth_ref.py), and a prototype of
per-frame depth stabilised along the motion ("da+flow").

  python depth_stability.py <clip dir> <stab flows fwd> <stab flows bwd> <eval flows fwd> <eval flows bwd> <method dir>...

Flow dirs as `aicheck searaft|waft` writes them: fwd from frames in order (file t:
t -> t+1), bwd from the frames reversed (file k: N-1-k -> N-2-k).

Score (lower = steadier), per consecutive pair, over pixels the evaluation motion
trusts (inside the picture, forward-backward check): the mean |d_t(p) - d_t-1(p +
b(p))| relative to the mean depth, each sequence first divided by its own median
over the whole clip (one number per clip, so per-frame scale jumps count). Also the
spread (std) of the per-frame log scale ratio: the "breathing" of the whole map.
The stabiliser uses one motion (SEA-RAFT) and the score another (WAFT), so it is not
graded with the motion it was built on.
"""
import sys, os
import numpy as np
import torch
import torch.nn.functional as F


def ld(p):
    raw = open(p, "rb").read()
    c, h, w = np.frombuffer(raw[:12], "<u4")
    return np.frombuffer(raw[12:], "<f4").reshape(c, h, w).copy()


clip, sfw, sbw, efw, ebw = sys.argv[1:6]
methods = sys.argv[6:]
n = len([f for f in os.listdir(clip) if f.endswith(".png")])
dev = "cuda"


def flows(fw, bw):
    f = [torch.from_numpy(ld(f"{fw}/{t:03d}.f32")).to(dev) for t in range(n - 1)]  # t -> t+1
    b = [None] + [torch.from_numpy(ld(f"{bw}/{n - 1 - t:03d}.f32")).to(dev) for t in range(1, n)]  # t -> t-1
    return f, b


def sample(img, flow):
    """img [C][h][w] sampled at p + flow(p) (bilinear); flow at img's size."""
    c, h, w = img.shape
    ys, xs = torch.meshgrid(torch.arange(h, device=dev), torch.arange(w, device=dev), indexing="ij")
    gx = (xs + flow[0]) / (w - 1) * 2 - 1
    gy = (ys + flow[1]) / (h - 1) * 2 - 1
    g = torch.stack([gx, gy], -1)[None]
    return F.grid_sample(img[None], g, mode="bilinear", padding_mode="border", align_corners=True)[0]


def trust(fwd_prev, bwd):
    """Pixels of frame t whose backward motion lands inside frame t-1 and agrees with
    t-1's forward motion there."""
    _, h, w = bwd.shape
    ys, xs = torch.meshgrid(torch.arange(h, device=dev), torch.arange(w, device=dev), indexing="ij")
    inside = (xs + bwd[0] >= 0) & (xs + bwd[0] <= w - 1) & (ys + bwd[1] >= 0) & (ys + bwd[1] <= h - 1)
    back = sample(fwd_prev, bwd)
    err = ((back + bwd) ** 2).sum(0)
    mag = (back ** 2).sum(0) + (bwd ** 2).sum(0)
    return inside & (err < 0.01 * mag + 0.5)


def score(seq, ef, eb):
    med = torch.median(torch.stack(seq))
    seq = [d / med for d in seq]
    errs, logs = [], []
    # START: skip the first pairs (Video Depth Anything's stream starts from one frame
    # alone, then jumps once its cache holds copies of it).
    for t in range(int(os.environ.get("START", "1")), n):
        m = trust(ef[t - 1], eb[t])
        if m.float().mean() < 0.2:  # a cut: nothing to compare
            continue
        prev = sample(seq[t - 1][None], eb[t])[0]
        cur = seq[t]
        errs.append(((cur - prev).abs()[m].mean() / cur[m].mean()).item())
        logs.append(np.log((cur[m].median() / prev[m].median()).item()))
    return np.mean(errs) * 100, np.std(logs) * 100


def stabilise(seq, sf, sb, keep=0.8):
    """Each new frame's depth fitted (scale, shift) to the previous result moved along
    the motion where the motion is trusted, then blended with it there (`keep` of the
    history); elsewhere the fitted frame alone."""
    out = [seq[0]]
    for t in range(1, n):
        m = trust(sf[t - 1], sb[t])
        if m.float().mean() < 0.2:  # a cut: start again from this frame
            out.append(seq[t])
            continue
        hist = sample(out[-1][None], sb[t])[0]
        d = seq[t]
        x, y = d[m], hist[m]
        a = torch.stack([x, torch.ones_like(x)], 1)
        sol = torch.linalg.lstsq(a, y[:, None]).solution[:, 0]
        fit = d * sol[0] + sol[1]
        out.append(torch.where(m, keep * hist + (1 - keep) * fit, fit))
    return out


ef, eb = flows(efw, ebw)
sf, sb = flows(sfw, sbw)
for md in methods:
    seq = [torch.from_numpy(ld(f"{md}/d{t:04d}.f32")[0]).to(dev) for t in range(n)]
    e, s = score(seq, ef, eb)
    print(f"{os.path.basename(md):>14}: change along the motion {e:.2f}%, scale breathing {s:.2f}%")
    if os.path.basename(md).endswith("_da"):
        for keep in (0.5, 0.8):
            st = stabilise(seq, sf, sb, keep)
            e, s = score(st, ef, eb)
            print(f"{'da+flow ' + str(keep):>14}: change along the motion {e:.2f}%, scale breathing {s:.2f}%")
            od = md + f"_flow{int(keep * 10)}"
            os.makedirs(od, exist_ok=True)
            for t, d in enumerate(st):
                with open(f"{od}/d{t:04d}.f32", "wb") as fh:
                    fh.write(np.array([1, *d.shape], "<u4").tobytes()); fh.write(d.cpu().numpy().astype("<f4").tobytes())
