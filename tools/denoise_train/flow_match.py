"""Motion corrected by matching against the previous result (no learned part).

Why: on faint detail under heavy noise the motion stage cannot see what moves and
takes the motion of the surroundings (the classroom's chalk writing: true motion
0.1 px, estimate up to 1.1 px, on strokes 1.5 px wide). The frame before is as noisy
as the frame, but the model's PREVIOUS RESULT is clean and at the frame before's time:
matching the noisy frame against it finds the motion where two noisy frames cannot.

    flow = match(flow, cur, ref)

flow: (B, 2, H, W), where each pixel of `cur` is in `ref`, full-size pixels; H and W
multiples of 4. cur: the noisy frame, ref: the previous result (both PQ, (B, 3, H, W)).

The same steps as the player's kernels (crates/brp-ai/src/learned.rs, MATCH_*):
1. Both lumas minus their 5 x 5 mean (brightness differences between a noisy and a
   clean picture do not count).
2. For the 25 offsets d in {-1, -0.5, 0, 0.5, 1}^2: the squared difference between
   the frame and the reference moved by flow + d, averaged over 4 x 4 blocks.
3. Those costs averaged over 5 x 5 blocks (so over 20 px).
4. Per block the best offset, refined by a parabola along x and y, times a share that
   grows with how much better it is than no correction (0 at equal cost, 1 from
   SOFT of the cost of no correction on).
5. The correction averaged over 3 x 3 blocks, resized to the motion's own size
   (1/base of the picture; the player adds it to its motion there), enlarged to full
   size like the motion, added.
"""
import torch
import torch.nn.functional as F

OFFS = (-1.0, -0.5, 0.0, 0.5, 1.0)
SOFT = 0.03


def _box(x, k):
    """k x k mean, the border repeated."""
    p = k // 2
    return F.avg_pool2d(F.pad(x, (p, p, p, p), mode="replicate"), k, 1)


def _luma(x):
    return x.float().mean(1, keepdim=True)


def correction(flow, cur, ref, warp, soft=SOFT, base=2):
    """The correction (B, 2, H, W) to add to `flow`. warp(x, flow): the model's.
    base: the motion stage's finest level is 1/base of the picture."""
    flow = flow.float()
    a = _luma(cur)
    a = a - _box(a, 5)
    b = _luma(ref)
    b = b - _box(b, 5)
    n = len(OFFS)
    # The 25 moved references in one call (the offsets as part of the batch): the same
    # numbers as 25 separate warps, a fraction of the operations.
    B_, _, H_, W_ = flow.shape
    offs = torch.tensor([[dx, dy] for dy in OFFS for dx in OFFS], device=flow.device, dtype=flow.dtype)  # index = iy * n + ix
    fl = (flow[:, None] + offs[None, :, :, None, None]).flatten(0, 1)
    moved = warp(b[:, None].expand(B_, n * n, 1, H_, W_).flatten(0, 1), fl).view(B_, n * n, H_, W_)
    C = _box(F.avg_pool2d((a - moved) ** 2, 4), 5)  # (B, 25, H/4, W/4)
    i = C.argmin(1, keepdim=True)
    iy, ix = i // n, i % n
    c0 = C.gather(1, i)

    def sub(k, stride):
        ok = (k > 0) & (k < n - 1)
        cm = C.gather(1, (i - stride).clamp(0, n * n - 1))
        cp = C.gather(1, (i + stride).clamp(0, n * n - 1))
        return torch.where(ok, (0.5 * (cm - cp) / (cm + cp - 2 * c0).clamp_min(1e-12)).clamp(-0.5, 0.5), torch.zeros_like(c0))

    step = OFFS[1] - OFFS[0]
    ox = (ix.float() + sub(ix, 1)) * step + OFFS[0]
    oy = (iy.float() + sub(iy, n)) * step + OFFS[0]
    c_none = C[:, (n // 2) * n + n // 2 : (n // 2) * n + n // 2 + 1]
    share = ((c_none - c0) / (c_none * soft + 1e-12)).clamp(0, 1)
    d = _box(torch.cat([ox, oy], 1) * share, 3)
    H, W = flow.shape[2:]
    if base != 4:
        d = F.interpolate(d, size=(H // base, W // base), mode="bilinear", align_corners=False)
    return F.interpolate(d, scale_factor=base, mode="bilinear", align_corners=False)


def match(flow, cur, ref, warp, soft=SOFT, base=2):
    return flow.float() + correction(flow, cur, ref, warp, soft, base)
