"""Stable video denoiser (see PLAN.md): a learned coarse-to-fine motion stage that moves
the neighbours and the previous output onto the current frame, and a U-Net that
merges them. Plain PyTorch, resolution-independent (any size padded to a multiple of
16), only operations that export to ONNX and convert to our Vulkan runner:
convolutions, ReLU, pixel (un)shuffle, bilinear resize, bilinear sampling (warp).

One step:

    out, state = model(cur, before, after, valid, prev_out, prev_valid, strength)

- cur, before, after: [B, 3, H, W] RGB 0..1. `before` / `after` are the neighbour
  slots: t-1 / t+1, or at a clip's start t+1 / t+2, at its end t-2 / t-1 (see PLAN.md).
- valid: [B, 2, 1, 1] or [B, 2, H, W]: whether each slot holds a usable frame.
- prev_out: the previous step's output (or zeros), prev_valid: [B, 1, ...] (0 at a
  jump or the first frame).
- strength: [B, 1, H, W] noise to remove (0..1 relative), or a scalar map.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def conv(cin, cout, k=3, stride=1):
    return nn.Conv2d(cin, cout, k, stride, k // 2)


class FlowLevel(nn.Module):
    """One pyramid level's refinement (SpyNet-like): from the current picture, the
    neighbour moved by the flow so far, and that flow, a correction to the flow. The
    same weights at every level, so any frame size works (more levels)."""

    def __init__(s, c=(16, 32, 16), k=5):
        super().__init__()
        s.net = nn.Sequential(conv(4, c[0], k), nn.ReLU(), conv(c[0], c[1], k), nn.ReLU(), conv(c[1], c[2], k), nn.ReLU(), conv(c[2], 2, k))

    def forward(s, a, b_warped, flow):
        return flow + s.net(torch.cat([a, b_warped, flow], 1))


def warp(img, flow):
    """`img` sampled at p + flow(p) (flow in pixels, [B, 2, H, W]); bilinear, edges
    repeat. Exports as GridSample."""
    B, _, H, W = img.shape
    ys = torch.arange(H, device=img.device, dtype=img.dtype).view(1, H, 1).expand(B, H, W)
    xs = torch.arange(W, device=img.device, dtype=img.dtype).view(1, 1, W).expand(B, H, W)
    gx = (xs + flow[:, 0]) * (2.0 / max(W - 1, 1)) - 1.0
    gy = (ys + flow[:, 1]) * (2.0 / max(H - 1, 1)) - 1.0
    return F.grid_sample(img, torch.stack([gx, gy], -1), mode="bilinear", padding_mode="border", align_corners=True)


def luma(x):
    return 0.299 * x[:, 0:1] + 0.587 * x[:, 1:2] + 0.114 * x[:, 2:3]


class Motion(nn.Module):
    """Coarse-to-fine flow from `a` to `b` (where each pixel of `a` is in `b`), at
    1/4 of the size and up to it in pixels of the full picture. Levels until the
    coarsest side is <= 32 px."""

    def __init__(s, c=(16, 32, 16), k=5):
        super().__init__()
        s.level = FlowLevel(c, k)

    def forward(s, a, b):
        a, b = luma(a), luma(b)
        pa, pb = [F.avg_pool2d(a, 4)], [F.avg_pool2d(b, 4)]
        while min(pa[-1].shape[2:]) > 32:
            pa.append(F.avg_pool2d(pa[-1], 2))
            pb.append(F.avg_pool2d(pb[-1], 2))
        flow = torch.zeros_like(pa[-1]).repeat(1, 2, 1, 1)
        for la, lb in zip(reversed(pa), reversed(pb)):
            if flow.shape[2:] != la.shape[2:]:
                flow = F.interpolate(flow, size=la.shape[2:], mode="bilinear", align_corners=False) * 2.0
            flow = s.level(la, warp(lb, flow), flow)
        return F.interpolate(flow, scale_factor=4, mode="bilinear", align_corners=False) * 4.0


class UNet(nn.Module):
    """The merge: full-size inputs unshuffled to 1/2, levels at 1/2, 1/4, 1/8 of widths
    `w`, blocks `n` per level (encoder; the decoder mirrors the first two)."""

    def __init__(s, cin, w=(32, 64, 128), n=(2, 3, 6)):
        super().__init__()
        relu = nn.ReLU
        def stack(c, k):
            return nn.Sequential(*[m for _ in range(k) for m in (conv(c, c), relu())])
        s.inp = nn.Sequential(conv(cin * 4, w[0]), relu())
        s.e1 = stack(w[0], n[0] - 1)
        s.d1 = nn.Sequential(conv(w[0], w[1], 3, 2), relu())
        s.e2 = stack(w[1], n[1])
        s.d2 = nn.Sequential(conv(w[1], w[2], 3, 2), relu())
        s.mid = stack(w[2], n[2])
        s.u2 = nn.Sequential(nn.Conv2d(w[2], w[1] * 4, 1), nn.PixelShuffle(2))
        s.dec2 = stack(w[1], n[1])
        s.u1 = nn.Sequential(nn.Conv2d(w[1], w[0] * 4, 1), nn.PixelShuffle(2))
        s.dec1 = stack(w[0], n[0])
        s.out = conv(w[0], 3 * 4)

    def forward(s, x):
        x = F.pixel_unshuffle(x, 2)
        a = s.e1(s.inp(x))
        b = s.e2(s.d1(a))
        c = s.mid(s.d2(b))
        b = s.dec2(s.u2(c) + b)
        a = s.dec1(s.u1(b) + a)
        return F.pixel_shuffle(s.out(a), 2)


class StableDenoiser(nn.Module):
    def __init__(s, w=(32, 64, 128), n=(2, 3, 6), motion=(16, 32, 16), motion_k=5):
        super().__init__()
        s.motion = Motion(motion, motion_k)
        # cur, before, after, prev_out (12) + valid x2, prev_valid, strength (4)
        s.unet = UNet(16, w, n)

    def forward(s, cur, before, after, valid, prev_out, prev_valid, strength):
        H, W = cur.shape[2:]
        fb = s.motion(cur, before)
        fa = s.motion(cur, after)
        wb, wa = warp(before, fb), warp(after, fa)
        # The previous output lines up with t-1: moved along the same motion.
        wp = warp(prev_out, fb)
        ones = torch.ones_like(cur[:, :1])
        x = torch.cat([cur, wb, wa, wp, valid[:, 0:1] * ones, valid[:, 1:2] * ones, prev_valid * ones, strength * ones], 1)
        out = cur + s.unet(x)
        return out, (fb, fa)


def pad16(x):
    H, W = x.shape[2:]
    return F.pad(x, (0, (16 - W % 16) % 16, 0, (16 - H % 16) % 16), mode="replicate")
