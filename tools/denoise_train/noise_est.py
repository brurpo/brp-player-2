"""The noise-strength map estimated from the noisy frame alone, as the player can
compute it (no clean reference): the model's "strength" input in training and in use.

    sigma = estimate(v_pq)   # (B, 3, H, W) PQ -> (B, 1, H, W), the training map's units

Noise in flat 16x16 blocks (Immerkaer's Laplacian-difference estimate on luma and the
two colour differences) is taken per brightness level (8 levels; the median of the
flat blocks of each level, the others interpolated), applied to every block by its
brightness, upsampled and smoothed: a noise profile by brightness, as the player's
Denoise measures footage. The same as realtest.noise_map_estimate's first version
(NumPy), on the GPU.
"""
import math

import torch
import torch.nn.functional as F

B = 16  # block size
LEVELS = 8
FLAT = 3.0  # block gradient (8-bit levels) below which a block counts as flat
_M = torch.tensor([[1.0, -2.0, 1.0], [-2.0, 4.0, -2.0], [1.0, -2.0, 1.0]])
LUMA = (0.2126, 0.7152, 0.0722)


def _gauss(x, s):
    rad = max(1, int(math.ceil(3 * s)))
    t = torch.arange(-rad, rad + 1, device=x.device, dtype=torch.float32)
    k = torch.exp(-t * t / (2 * s * s))
    k = k / k.sum()
    y = F.pad(x, (rad, rad, 0, 0), mode="reflect")
    y = F.conv2d(y, k.view(1, 1, 1, -1))
    y = F.pad(y, (0, 0, rad, rad), mode="reflect")
    return F.conv2d(y, k.view(1, 1, -1, 1))


@torch.no_grad()
def estimate(v):
    """v: (B, 3, H, W) PQ values 0..1 -> (B, 1, H, W) noise map (local RMS of the noise
    over the 3 channels, PQ)."""
    v = v.float() * 255.0  # the block statistics work in 8-bit levels
    N, _, H, W = v.shape
    y = LUMA[0] * v[:, 0:1] + LUMA[1] * v[:, 1:2] + LUMA[2] * v[:, 2:3]
    cb, cr = v[:, 2:3] - y, v[:, 0:1] - y
    m = _M.to(v.device).view(1, 1, 3, 3)
    hp = lambda x: F.avg_pool2d(F.conv2d(x, m, padding=1).abs(), B) * math.sqrt(math.pi / 2) / 6
    ny = hp(y)
    nc = torch.hypot(hp(cb), hp(cr))
    g = _gauss(y, 3.0)
    gx = F.conv2d(g, torch.tensor([[-1.0, 0, 1], [-2, 0, 2], [-1, 0, 1]], device=v.device).view(1, 1, 3, 3), padding=1)
    gy = F.conv2d(g, torch.tensor([[-1.0, -2, -1], [0, 0, 0], [1, 2, 1]], device=v.device).view(1, 1, 3, 3), padding=1)
    grad = F.avg_pool2d(torch.hypot(gx, gy), B)
    mean = F.avg_pool2d(y, B)
    sig = torch.sqrt(ny * ny + nc * nc) / 255.0
    lvl = (mean / 255.0 * LEVELS).floor().clamp(0, LEVELS - 1).long()
    flat = grad < FLAT
    out = torch.empty_like(sig)
    for i in range(N):
        prof = torch.full((LEVELS,), float("nan"), device=v.device)
        s_i, l_i, f_i = sig[i, 0], lvl[i, 0], flat[i, 0]
        for k in range(LEVELS):
            sel = s_i[f_i & (l_i == k)]
            if sel.numel() >= 4:
                prof[k] = sel.median()
        good = ~torch.isnan(prof)
        if not good.any():
            prof[:] = s_i.median()
        elif not good.all():
            idx = torch.arange(LEVELS, device=v.device, dtype=torch.float32)
            gi, gv = idx[good], prof[good]
            # Linear interpolation, constant beyond the ends (as numpy.interp).
            pos = torch.searchsorted(gi, idx).clamp(1, max(len(gi) - 1, 1))
            if len(gi) == 1:
                prof = gv.expand(LEVELS).clone()
            else:
                x0, x1 = gi[pos - 1], gi[pos]
                y0, y1 = gv[pos - 1], gv[pos]
                t = ((idx - x0) / (x1 - x0)).clamp(0, 1)
                prof = y0 + t * (y1 - y0)
                prof = torch.where(idx < gi[0], gv[0], torch.where(idx > gi[-1], gv[-1], prof))
        out[i, 0] = prof[l_i]
    m_up = F.interpolate(out, size=(H, W), mode="bilinear", align_corners=False)
    return _gauss(m_up, 8.0)
