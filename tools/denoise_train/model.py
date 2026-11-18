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

import flow_match


def conv(cin, cout, k=3, stride=1):
    return nn.Conv2d(cin, cout, k, stride, k // 2)


class FlowLevel(nn.Module):
    """One pyramid level's refinement (SpyNet-like): from the current picture, the
    neighbour moved by the flow so far, and that flow, a correction to the flow. The
    same weights at every level, so any frame size works (more levels)."""

    def __init__(s, c=(16, 32, 16), k=5):
        super().__init__()
        # Any number of widths (three: the layers' names are those of older checkpoints).
        layers, cin = [], 4
        for w in c:
            layers += [conv(cin, w, k), nn.ReLU()]
            cin = w
        s.net = nn.Sequential(*layers, conv(cin, 2, k))

    def forward(s, a, b_warped, flow):
        return flow + s.net(torch.cat([a, b_warped, flow], 1))


def warp(img, flow, mode="bilinear"):
    """`img` sampled at p + flow(p) (flow in pixels, [B, 2, H, W]); bilinear (or
    bicubic: keeps fine detail at fractional shifts, where bilinear averages
    neighbouring pixels), edges repeat. Exports as GridSample."""
    B, _, H, W = img.shape
    ys = torch.arange(H, device=img.device, dtype=img.dtype).view(1, H, 1).expand(B, H, W)
    xs = torch.arange(W, device=img.device, dtype=img.dtype).view(1, 1, W).expand(B, H, W)
    gx = (xs + flow[:, 0]) * (2.0 / max(W - 1, 1)) - 1.0
    gy = (ys + flow[:, 1]) * (2.0 / max(H - 1, 1)) - 1.0
    return F.grid_sample(img, torch.stack([gx, gy], -1), mode=mode, padding_mode="border", align_corners=True)


def luma(x):
    return 0.299 * x[:, 0:1] + 0.587 * x[:, 1:2] + 0.114 * x[:, 2:3]


class Motion(nn.Module):
    """Coarse-to-fine flow from `a` to `b` (where each pixel of `a` is in `b`), at
    1/`base` of the size and up to it in pixels of the full picture. Levels until the
    coarsest side is <= `min_side` px."""

    def __init__(s, c=(16, 32, 16), k=5, min_side=32, iters=1, base=4, mix=False):
        super().__init__()
        s.level = FlowLevel(c, k)
        # mix: the estimate from one noisy pair is noise-limited (0.85 px from the true
        # motion on the classroom at 16 samples, median), and motion mostly changes
        # slowly: the frame before's motion, carried over, is a far better guess on a
        # camera move (0.18 px) and a wrong one where motion changes. A small network
        # sees both guesses and what each makes of the neighbour, and gives per pixel
        # the share of the prior (and a correction). Without a prior: the estimate.
        s.mix = None
        if mix:
            s.mix = nn.Sequential(conv(5, 16), nn.ReLU(), conv(16, 32), nn.ReLU(), conv(32, 16), nn.ReLU(), conv(16, 3))
            nn.init.zeros_(s.mix[-1].weight)
            nn.init.zeros_(s.mix[-1].bias)
            s.mix[-1].bias.data[0] = -2.0  # a share of 0.12 at the start
        # Pyramid levels are added until the coarsest side is <= min_side px; each level
        # refines the flow `iters` times (same weights).
        # base: the finest level's size, as a divisor of the picture (4 or 2).
        s.min_side, s.iters, s.base = min_side, iters, base

    def raw(s, a, b):
        """The estimate at 1/base of the size, in that level's pixels, and the two
        lumas at that level."""
        # Always in float32: the stage is tiny, and half precision costs flow accuracy.
        with torch.autocast("cuda", enabled=False):
            a, b = luma(a.float()), luma(b.float())
            pa, pb = [F.avg_pool2d(a, s.base)], [F.avg_pool2d(b, s.base)]
            while min(pa[-1].shape[2:]) > s.min_side:
                pa.append(F.avg_pool2d(pa[-1], 2))
                pb.append(F.avg_pool2d(pb[-1], 2))
            flow = torch.zeros_like(pa[-1]).repeat(1, 2, 1, 1)
            for la, lb in zip(reversed(pa), reversed(pb)):
                if flow.shape[2:] != la.shape[2:]:
                    flow = F.interpolate(flow, size=la.shape[2:], mode="bilinear", align_corners=False) * 2.0
                for _ in range(s.iters):
                    flow = s.level(la, warp(lb, flow), flow)
            return flow, pa[0], pb[0]

    def up(s, flow):
        with torch.autocast("cuda", enabled=False):
            return F.interpolate(flow.float(), scale_factor=s.base, mode="bilinear", align_corners=False) * float(s.base)

    def mixed(s, flow, prior, valid, la, lb):
        """The estimate `flow` and the `prior` (both at 1/base size) mixed per pixel;
        valid (B, 1, 1, 1): 0 where there is no prior (then the estimate as it is)."""
        with torch.autocast("cuda", enabled=False):
            d = prior.float() - flow
            o = s.mix(torch.cat([la, warp(lb, flow), warp(lb, prior.float()), d], 1))
            return flow + (torch.sigmoid(o[:, :1]) * d + o[:, 1:]) * valid.float()

    def forward(s, a, b):
        return s.up(s.raw(a, b)[0])


def mem_in(cin, cout):
    """The convolution that reads the memory: no bias, zero at the start."""
    m = nn.Conv2d(cin, cout, 3, 1, 1, bias=False)
    nn.init.zeros_(m.weight)
    return m


class UNet(nn.Module):
    """The merge: full-size inputs unshuffled to 1/2, levels at 1/2, 1/4, 1/8 of widths
    `w`, blocks `n` per level (encoder; the decoder mirrors the first two)."""

    def __init__(s, cin, w=(32, 64, 128), n=(2, 3, 6), state=0):
        super().__init__()
        relu = nn.ReLU
        def stack(c, k):
            return nn.Sequential(*[m for _ in range(k) for m in (conv(c, c), relu())])
        # state: channels of the memory carried from frame to frame (at 1/2 size): read
        # by a convolution of its own beside the first one (starting at zero, so older
        # weights compute what they did), written from the last features, kept within
        # -1..1 so it cannot grow over a long run.
        s.inp = nn.Sequential(conv(cin * 4, w[0]), relu())
        s.inp_h = mem_in(state, w[0]) if state else None
        s.state = conv(w[0], state) if state else None
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

    def forward(s, x, h=None):
        a = s.inp[0](F.pixel_unshuffle(x, 2))
        if s.state is not None:
            a = a + s.inp_h(h.to(a.dtype))
        a = s.e1(s.inp[1](a))
        b = s.e2(s.d1(a))
        c = s.mid(s.d2(b))
        b = s.dec2(s.u2(c) + b)
        a = s.dec1(s.u1(b) + a)
        y = F.pixel_shuffle(s.out(a), 2)
        return (y, s.state(a).float().clamp(-1.0, 1.0)) if s.state is not None else y


class LayerNorm2d(nn.Module):
    """Normalisation over the channels of each pixel (nothing spatial: the same at any
    picture size)."""

    def __init__(s, c, eps=1e-6):
        super().__init__()
        s.weight, s.bias, s.eps = nn.Parameter(torch.ones(1, c, 1, 1)), nn.Parameter(torch.zeros(1, c, 1, 1)), eps

    def forward(s, x):
        x = x.float()
        mu = x.mean(1, keepdim=True)
        return (x - mu) / ((x - mu).pow(2).mean(1, keepdim=True) + s.eps).sqrt() * s.weight + s.bias


class ResBlock(nn.Module):
    """x + conv(relu(conv(x))): the plain design with a skip, so that many blocks train
    (the second convolution starts at zero: the block starts as nothing)."""

    def __init__(s, c):
        super().__init__()
        s.a, s.b = conv(c, c), conv(c, c)
        nn.init.zeros_(s.b.weight)
        nn.init.zeros_(s.b.bias)

    def forward(s, x):
        return x + s.b(F.relu(s.a(x)))


class NAFBlock(nn.Module):
    """NAFNet's block without its global channel attention (that one depends on the
    picture size): normalise, widen 1x1, depthwise 3x3, gate (one half times the
    other), 1x1; then normalise, widen, gate, 1x1. Both parts added with a learned
    scale starting at zero."""

    def __init__(s, c):
        super().__init__()
        s.n1, s.n2 = LayerNorm2d(c), LayerNorm2d(c)
        s.c1, s.dw, s.c2 = nn.Conv2d(c, 2 * c, 1), nn.Conv2d(2 * c, 2 * c, 3, 1, 1, groups=2 * c), nn.Conv2d(c, c, 1)
        s.c3, s.c4 = nn.Conv2d(c, 2 * c, 1), nn.Conv2d(c, c, 1)
        s.beta, s.gamma = nn.Parameter(torch.zeros(1, c, 1, 1)), nn.Parameter(torch.zeros(1, c, 1, 1))

    def forward(s, x):
        a, b = s.dw(s.c1(s.n1(x))).chunk(2, 1)
        y = x + s.c2(a * b) * s.beta
        a, b = s.c3(s.n2(y)).chunk(2, 1)
        return y + s.c4(a * b) * s.gamma


class TrBlock(nn.Module):
    """Restormer's transformer block: attention between CHANNELS, computed over the
    whole picture (each channel's map against each other's: cost linear in the picture
    size), then a gated depthwise feed-forward. Heads of about 48 channels."""

    def __init__(s, c, ffn=2.66):
        super().__init__()
        s.heads = 1 << max(0, (c // 48).bit_length() - 1)  # 48: 1, 96: 2, 192: 4, 384: 8 (64: 1, 128: 2, ...)
        s.n1, s.n2 = LayerNorm2d(c), LayerNorm2d(c)
        s.temp = nn.Parameter(torch.ones(s.heads, 1, 1))
        s.qkv, s.qkv_dw, s.proj = nn.Conv2d(c, 3 * c, 1, bias=False), nn.Conv2d(3 * c, 3 * c, 3, 1, 1, groups=3 * c, bias=False), nn.Conv2d(c, c, 1, bias=False)
        h = int(c * ffn)
        s.f1, s.f_dw, s.f2 = nn.Conv2d(c, 2 * h, 1, bias=False), nn.Conv2d(2 * h, 2 * h, 3, 1, 1, groups=2 * h, bias=False), nn.Conv2d(h, c, 1, bias=False)

    def forward(s, x):
        B, C, H, W = x.shape
        q, k, v = s.qkv_dw(s.qkv(s.n1(x))).chunk(3, 1)
        sh = lambda t: t.reshape(B, s.heads, C // s.heads, H * W)
        q, k, v = F.normalize(sh(q).float(), dim=-1), F.normalize(sh(k).float(), dim=-1), sh(v).float()
        a = ((q @ k.transpose(-2, -1)) * s.temp).softmax(-1)
        x = x + s.proj((a @ v).reshape(B, C, H, W))
        a, b = s.f_dw(s.f1(s.n2(x))).chunk(2, 1)
        return x + s.f2(F.gelu(a) * b)


BLOCKS = {"res": ResBlock, "naf": NAFBlock, "tr": TrBlock}


class UNet2(nn.Module):
    """The merge with residual / NAFNet / transformer blocks (`block`), for the quality
    model. As UNet: full-size inputs unshuffled to 1/2, one level per width (1/2, 1/4,
    1/8, ...), `n` blocks per level in the encoder and again in the decoder, the levels
    joined by addition; the memory (`state`) read beside the first convolution and
    written from the last features. The output starts at zero."""

    def __init__(s, cin, w, n, block="res", state=0):
        super().__init__()
        # One kind of block for every level, or one per level ("res,res,tr,tr").
        kinds = block.split(",") if "," in block else [block] * len(w)
        assert len(kinds) == len(w), (block, w)
        stack = lambda c, k, i=0: nn.Sequential(*[BLOCKS[kinds[i]](c) for _ in range(k)])
        s.inp = conv(cin * 4, w[0])
        s.inp_h = mem_in(state, w[0]) if state else None
        s.state = conv(w[0], state) if state else None
        s.enc = nn.ModuleList([stack(c, k, i) for i, (c, k) in enumerate(zip(w, n))])
        s.down = nn.ModuleList([nn.Conv2d(w[i], w[i + 1], 2, 2) for i in range(len(w) - 1)])
        s.up = nn.ModuleList([nn.Sequential(nn.Conv2d(w[i + 1], w[i] * 4, 1), nn.PixelShuffle(2)) for i in range(len(w) - 1)])
        s.dec = nn.ModuleList([stack(w[i], n[i], i) for i in range(len(w) - 1)])
        s.out = conv(w[0], 3 * 4)
        nn.init.zeros_(s.out.weight)
        nn.init.zeros_(s.out.bias)

    def forward(s, x, h=None):
        a = s.inp(F.pixel_unshuffle(x, 2))
        if s.state is not None:
            a = a + s.inp_h(h.to(a.dtype))
        skips = []
        for i, e in enumerate(s.enc):
            a = e(a)
            if i < len(s.down):
                skips.append(a)
                a = s.down[i](a)
        for i in reversed(range(len(s.dec))):
            a = s.dec[i](s.up[i](a) + skips[i])
        y = F.pixel_shuffle(s.out(a), 2)
        return (y, s.state(a).float().clamp(-1.0, 1.0)) if s.state is not None else y


class Fuse(nn.Module):
    """The pre-merge, before the U-Net: the frame, its aligned neighbours and the aligned
    previous output merged over time (a few plain convolutions at 1/2 size).

    - blend (the one to use): per pixel, weights for the four pictures (softmax; slots
      without a picture get none); the result is their weighted mix. "Take this pixel
      from the neighbour" is then a smooth map to predict, which a network learns fast,
      and the mix cannot invent anything. It starts as mostly the frame itself.
    - residual: a correction added to the frame, starting at zero.

    Why it exists: the U-Net alone did not learn to merge frames, pre-trained or from
    scratch: given the clean frame itself as both neighbours it gained 0.2-0.3 dB.
    Copying fine detail from one input through a stack of convolutions is slow to
    learn; choosing between aligned pictures is not."""

    def __init__(s, cin, w=32, n=4, blend=True, start=3.0, state=0, pics=4):
        super().__init__()
        s.blend = blend
        # pics: how many pictures are mixed (4, or 5 with the accumulated one).
        s.pics = pics
        # The memory (StableDenoiser `state`), read beside the first convolution: how
        # far the history can be trusted is decided here, so it has to see it.
        s.in_h = mem_in(state, w) if state else None
        layers = [conv(cin * 4, w), nn.ReLU()]
        for _ in range(n - 2):
            layers += [conv(w, w), nn.ReLU()]
        last = conv(w, (pics if blend else 3) * 4)
        nn.init.zeros_(last.weight)
        nn.init.zeros_(last.bias)
        if blend:
            with torch.no_grad():
                last.bias[:4] = start  # the frame's own weight (channels 0-3 -> weight 0 after the shuffle)
        s.net = nn.Sequential(*layers, last)
        s.k, s.kp, s.width = 0, None, w

    def set_kp(s, k):
        """Kernel prediction: per pixel a k x k kernel for each picture (k odd) instead
        of one weight. The taps around the centre start with next to no weight, so
        the result starts as the plain mix."""
        s.k = k
        s.kp = conv(s.width, s.pics * (k * k - 1) * 4)
        nn.init.zeros_(s.kp.weight)
        nn.init.constant_(s.kp.bias, -8.0)

    def forward(s, x, frames=None, mask=None, h=None):
        """Returns the merged picture, each picture's share of it (B, pics, H, W) and
        each picture's part of the sum (their shares times the pictures; with
        kernels, each picture filtered by its own)."""
        y = s.net[0](F.pixel_unshuffle(x, 2))
        if s.in_h is not None:
            y = y + s.in_h(h.to(y.dtype))
        feat = s.net[1:-1](y)
        y = F.pixel_shuffle(s.net[-1](feat), 2).float()
        if not s.blend:
            return frames[0] + y, None, None
        y = y.masked_fill(mask < 0.5, -1e4)
        if not s.k:
            w = torch.softmax(y, 1)
            parts = [w[:, k : k + 1] * frames[k].float() for k in range(s.pics)]
            return sum(parts), w, parts
        B, P, H, W = y.shape
        kk, r = s.k * s.k, s.k // 2
        off = F.pixel_shuffle(s.kp(feat), 2).float().view(B, P, kk - 1, H, W).masked_fill((mask < 0.5)[:, :, None], -1e4)
        # All taps of all pictures share one softmax; the centre tap goes in the
        # middle of each picture's kernel (unfold's order).
        logits = torch.cat([off[:, :, : kk // 2], y[:, :, None], off[:, :, kk // 2 :]], 2)
        wk = torch.softmax(logits.flatten(1, 2), 1).view(B, P, kk, H, W)
        parts = []
        for p in range(P):
            f = frames[p].float()
            taps = F.unfold(F.pad(f, (r, r, r, r), mode="replicate"), s.k).view(B, f.shape[1], kk, H, W)
            parts.append((taps * wk[:, p : p + 1]).sum(2))
        return sum(parts), wk.sum(2), parts


def mismatch(cur, w):
    """How far an aligned picture is from the frame, as two maps (means of 4 px and of
    16 px blocks: noise averages out, a wrong picture does not), x8 to be of order 1.
    Handed to the merge so that "this neighbour does not belong here" (a cut, something
    passing in front, a failed motion estimate) is something it sees, not something it
    has to work out: on its own it learned that slowly and ghosts were the result."""
    out = []
    for k in (4, 16):
        d = (F.avg_pool2d(cur.float(), k) - F.avg_pool2d(w.float(), k)).abs().mean(1, keepdim=True)
        out.append(F.interpolate(d, size=cur.shape[2:], mode="bilinear", align_corners=False))
    return torch.cat(out, 1) * 8.0


def mismatch_many(cur, ws):
    """`mismatch(cur, w)` for each w of `ws` (same shapes), in one pass: the frame's
    block means once, the pictures as one batch. The same numbers."""
    n, B = len(ws), cur.shape[0]
    allw = torch.cat([w.float() for w in ws], 0)
    out = []
    for k in (4, 16):
        d = (F.avg_pool2d(cur.float(), k).repeat(n, 1, 1, 1) - F.avg_pool2d(allw, k)).abs().mean(1, keepdim=True)
        out.append(F.interpolate(d, size=cur.shape[2:], mode="bilinear", align_corners=False))
    m = torch.cat(out, 1) * 8.0
    return [m[i * B : (i + 1) * B] for i in range(n)]


class NoiseEstimator(nn.Module):
    """The noise-strength map predicted from the frame and its two aligned neighbours
    (CBDNet-style): what differs between them over time is noise, what they share is
    picture. Works at 1/4 size; output in the map's units (damage RMS in PQ x 10),
    upsampled. Trained against the true map (known in training only) as a target."""

    def __init__(s, c=(32, 48)):
        super().__init__()
        relu = nn.ReLU
        s.net = nn.Sequential(
            nn.PixelUnshuffle(2), conv(9 * 4, c[0]), relu(),
            conv(c[0], c[1], 3, 2), relu(),
            conv(c[1], c[1]), relu(), conv(c[1], c[1]), relu(), conv(c[1], c[1]), relu(),
            conv(c[1], 1),
        )

    PAD = 16  # mirrored margin (pixels, a multiple of 4)

    def forward(s, cur, wb, wa):
        # The frame is mirrored out by PAD (and to a multiple of 4) before the network
        # and the margin cut off after: with the convolutions' zero padding at the
        # edges, the map had a grid of lines there, which the denoiser drew.
        H, W = cur.shape[2:]
        P = s.PAD
        eh, ew = -H % 4, -W % 4
        x = F.pad(torch.cat([cur, wb, wa], 1), (P, P + ew, P, P + eh), mode="reflect")
        y = s.net(x)[..., P // 4 : P // 4 + (H + eh) // 4, P // 4 : P // 4 + (W + ew) // 4]
        y = F.interpolate(y, scale_factor=4, mode="bilinear", align_corners=False)[..., :H, :W]
        return F.softplus(y)


class StableDenoiser(nn.Module):
    def __init__(s, w=(32, 64, 128), n=(2, 3, 6), motion=(16, 32, 16), motion_k=5, motion_min=32, motion_iters=1, fuse=0, fuse_mode="blend", keep_cur=False, motion_base=4, mismatch=False, state=0, acc=False, flow_mix=False, arch="plain"):
        super().__init__()
        s.motion = Motion(motion, motion_k, motion_min, motion_iters, motion_base, mix=flow_mix)
        s.last_fstate = None
        s.flow_override = None
        s.flow_match = False  # the motion corrected against the previous result (flow_match.py)
        s.dis_flow = None  # "clean" / "noisy": train.sequence hands in DIS motion (study)
        # The pre-merge (width `fuse`; 0: none, as in models before it existed).
        # mismatch: the merge also gets the mismatch maps of the three aligned pictures
        # (6 channels, after everything else, so older weights load with them at zero).
        s.mismatch = bool(mismatch)
        mm = 6 if s.mismatch else 0
        # acc: an honest history. The previous output carries what the merge drew,
        # blur and inventions included, so the pre-merge distrusts it exactly where
        # there is fine detail and the detail is redrawn from noise every frame. The
        # accumulated picture is a running mix of INPUT frames only (the pre-merge's
        # own weights, the previous output left out), moved along the motion: a fifth
        # picture to choose from, in which real detail can build up. The pre-merge
        # reads it and its mismatch maps (5 more channels, last).
        s.acc = bool(acc and fuse and fuse_mode == "blend")
        s.fuse = Fuse(16 + mm + (5 if s.acc else 0), fuse, blend=fuse_mode == "blend", state=state, pics=5 if s.acc else 4) if fuse else None
        s.last_acc = None
        s.last_base = s.last_weights = None
        # cur, before, after, prev_out (12) + valid x2, prev_valid, strength (4)
        # keep_cur (with the pre-merge): the U-Net also gets the frame as it came (3 more
        # channels, last): the merged picture replaces it in the first slot, and where the
        # merge went wrong (misaligned neighbours) only the frame itself can repair it.
        s.keep_cur = bool(keep_cur and fuse)
        # state: a learned memory beside the previous output (UNet.state): what the
        # merge wrote at the previous frame, moved along the motion like the previous
        # output and dropped with it (clip start, jump, cut), read by the pre-merge and
        # the merge. The previous output alone carries only what was drawn; this
        # carries what the merge chooses to keep (how far to trust the history, say).
        s.unet = UNet((19 if s.keep_cur else 16) + mm, w, n, state=state) if arch == "plain" else UNet2((19 if s.keep_cur else 16) + mm, w, n, arch, state=state)
        s.last_state = s._last_out = None
        s.estimator = NoiseEstimator()
        # How the neighbours are moved onto the frame (not a weight: any checkpoint).
        s.warp_mode = "bilinear"
        # strength=None: use the estimator's map (scaled by strength_scale, the user's
        # Strength) when use_estimator, else a constant 1 (blind).
        s.use_estimator = False
        s.strength_scale = 1.0
        s.last_sigma = None
        # Cut detector, for playing (not a weight; 0 = off, as in training): a neighbour
        # whose aligned picture differs from the frame by more than this on average
        # (means of 16 px blocks, PQ) is another shot and is dropped, with the previous
        # output when it is the frame before. On validation true neighbours reach 0.019
        # at most and other clips 0.035 at least: 0.026 separates them. The network also
        # learns to reject what does not match (train.py --cut-prob); this is the net
        # under it, for the cuts it would half-believe.
        s.cut_thr = 0.0
        s.last_cut = None
        # flow_prev (config; the motion stage must be trained for it): the motion
        # towards the frame before is estimated against the previous OUTPUT, not the
        # noisy frame before. The history is what gets moved, so this lines the history
        # itself up with the new frame: errors in where it sits are corrected at every
        # frame instead of adding up along a chain of frame-to-frame estimates (the
        # clean frame carried 4 frames along the chain matches at 34.5 dB on the
        # classroom, below the output's own 37.7), and one of the two pictures is
        # clean. Where there is no previous output: the frame before, as always.
        s.flow_prev = False

    def flows(s, cur, nb, after, valid, prev_valid, fstate=None, gap=1.0, raw=None):
        """The motion towards the frame before (`nb`) and after, at full size, and the
        flow state for the next frame (None without the mix).

        With the mix (Motion.mix): fstate is the previous frame's motion towards THIS
        frame (its `after` motion, at 1/base size, on its own grid). Read where each
        pixel came from and negated, it is this frame's motion towards the frame
        before: the prior for it. The prior for the motion towards the frame after is
        minus the motion towards the frame before (motion going on as it was).

        raw: the two estimates worked out beforehand, (towards before, luma, the
        frame before's luma, towards after, the frame after's luma) as Motion.raw
        gives them: training runs Motion.raw once for a whole clip (train.sequence),
        many frames in one call, instead of twice per frame."""
        mo = s.motion
        if mo.mix is None:
            return mo(cur, nb), mo(cur, after), None
        if raw is None:
            fb0, la, lb = mo.raw(cur, nb)
        else:
            fb0, la, lb, fa0_, lb2_ = raw
        if fstate is None:
            fb = fb0
        else:
            pv = prev_valid if prev_valid.shape[-1] == 1 else prev_valid.float().amax((1, 2, 3), keepdim=True)
            # gap: `nb` is this many frames back (the player, when playback skipped
            # frames: the last picture made). The flow state is the motion over ONE
            # frame, so the prior is that many times it.
            fb = mo.mixed(fb0, -gap * warp(fstate.float(), fb0), (pv > 0).float() * (valid[:, 0:1] > 0).float(), la, lb)
        if raw is None:
            fa0, la, lb = mo.raw(cur, after)
        else:
            fa0, lb = fa0_, lb2_
        fa = mo.mixed(fa0, -fb / gap, (valid[:, 0:1] > 0).float() * (valid[:, 1:2] > 0).float(), la, lb)
        return mo.up(fb), mo.up(fa), fa

    def forward(s, cur, before, after, valid, prev_out, prev_valid, strength, state=None, acc=None, fstate=None, gap=1.0, raw=None, future=None):
        """state: the memory returned by the previous step (`last_state`), for models
        that have one. Left out when not training: the one kept from the last call is
        used if `prev_out` is that call's output (so code that only feeds the previous
        output back runs these models unchanged), else none."""
        H, W = cur.shape[2:]
        nb = before
        if s.flow_prev:
            nb = torch.where(prev_valid > 0, prev_out.detach().to(before.dtype), before)
        if fstate is None and not s.training and s._last_out is not None and s._last_out.shape == prev_out.shape and (prev_out is s._last_out or torch.equal(prev_out, s._last_out)):
            fstate = s.last_fstate  # as for the memory below: code that only feeds the output back
        if getattr(s, "flow_override", None) is not None:
            # A study switch (train.py --dis-flow): the motion handed in from outside.
            (fb, fa), s.last_fstate = s.flow_override, None
        else:
            fb, fa, s.last_fstate = s.flows(cur, nb, after, valid, prev_valid, fstate, gap, None if s.flow_prev else raw)
        if s.flow_match:
            # The estimate cannot see faint detail under heavy noise and takes the
            # motion of the surroundings there; the previous result is clean and at
            # the frame before's time, so the noisy frame is matched against it
            # (flow_match.py; no learned part, no gradient).
            pm = prev_valid if prev_valid.shape[-1] == 1 else prev_valid.float().amax((1, 2, 3), keepdim=True)
            with torch.no_grad():
                fix = flow_match.correction(fb.detach(), cur, prev_out.detach(), warp, base=s.motion.base) * ((pm > 0) & (valid[:, 0:1] > 0)).float()
            fb = fb.float() + fix
        s.last_flows = (fb, fa)  # for the motion loss and checks
        if before.dtype == after.dtype and fb.dtype == fa.dtype:
            # Both neighbours in one warp (the same numbers, half the operations).
            wb, wa = warp(torch.cat([before, after], 0), torch.cat([fb, fa], 0), s.warp_mode).chunk(2, 0)
        else:
            wb, wa = warp(before, fb, s.warp_mode), warp(after, fa, s.warp_mode)
        if future is not None:
            # Forward and backward (train.py --future): in the frame-after's place, the
            # result a backward pass made for THIS frame: clean, already aligned, and
            # holding what the frames after it have to give. The motion towards the
            # frame after is still estimated (the next frame's prior needs it).
            wa = future.to(wa.dtype)
        # The previous output lines up with t-1: moved along the same motion.
        wp = warp(prev_out, fb, s.warp_mode)
        fbr = fb
        if getattr(s, "refine", None) is not None:
            # That motion corrected, from the frame and the previous result moved onto
            # it (the previous result is clean: what two noisy frames cannot show can
            # be matched against it). The correction starts at zero and is learned from
            # the denoising loss; it moves what the model carries along (previous
            # result, accumulated picture, memory), not the noisy frame before.
            d = s.refine(F.pixel_unshuffle(torch.cat([cur, wp.to(cur.dtype)], 1), 2)).float()
            fbr = fb + F.interpolate(d, scale_factor=2, mode="bilinear", align_corners=False) * (prev_valid > 0).float()
            wp = warp(prev_out, fbr, s.warp_mode)
        if s.cut_thr > 0:
            off = lambda w_: (F.avg_pool2d(cur.float(), 16) - F.avg_pool2d(w_.float(), 16)).abs().mean((1, 2, 3), keepdim=True) > s.cut_thr
            cb, ca = off(wb) & (valid[:, 0:1] > 0), off(wa) & (valid[:, 1:2] > 0)
            # As at a clip's ends in training: the frame itself in the neighbour's
            # place, no previous output, flags off.
            wb, wa, wp = torch.where(cb, cur, wb), torch.where(ca, cur, wa), torch.where(cb, torch.zeros_like(wp), wp)
            valid = torch.cat([valid[:, 0:1] * (~cb), valid[:, 1:2] * (~ca)], 1)
            prev_valid = prev_valid * (~cb)
            s.last_cut = (cb, ca)
        ones = torch.ones_like(cur[:, :1])
        if strength is None:
            if s.use_estimator:
                s.last_sigma = s.estimator(cur, wb, wa)
                strength = s.last_sigma * s.strength_scale
            else:
                strength = ones
        x = torch.cat([cur, wb, wa, wp, valid[:, 0:1] * ones, valid[:, 1:2] * ones, prev_valid * ones, strength * ones], 1)
        base = cur
        h = None
        if s.unet.state is not None:
            if state is None and not s.training and s._last_out is not None and s._last_out.shape == prev_out.shape and (prev_out is s._last_out or torch.equal(prev_out, s._last_out)):
                state = s.last_state
                acc = s.last_acc if acc is None else acc
            if state is None:
                state = torch.zeros(cur.shape[0], s.unet.state.out_channels, H // 2, W // 2, device=cur.device)
            # Moved like the previous output (the flow at 1/2 size), nothing where
            # there is no previous frame.
            pv = prev_valid if prev_valid.shape[-1] == 1 else F.avg_pool2d(prev_valid.float(), 2)
            h = warp(state.float(), F.avg_pool2d(fbr, 2) * 0.5) * pv
        wacc = av = macc = None
        if s.fuse is not None and s.acc:
            # The accumulated picture, moved like the previous output; none at a
            # start or after a cut.
            av = prev_valid * (0.0 if acc is None else 1.0)
            wacc = warp(acc.float(), fbr, s.warp_mode) * av if acc is not None else torch.zeros_like(cur)
        if s.mismatch:
            # The mismatch maps of all the aligned pictures in one go.
            mall = mismatch_many(cur, [wb, wa, wp] + ([wacc] if wacc is not None else []))
            mm = [torch.cat(mall[:3], 1).to(x.dtype)]
            macc = mall[3] if wacc is not None else None
        else:
            mm = []
        if s.fuse is not None:
            # The frames merged over time first; the U-Net then works on that picture.
            mask = torch.cat([ones, valid[:, 0:1] * ones, valid[:, 1:2] * ones, prev_valid * ones], 1)
            if s.acc:
                if macc is None:
                    macc = mismatch(cur, wacc)
                fx = torch.cat([x] + mm + [wacc.to(x.dtype), macc.to(x.dtype)], 1)
                base, w, parts = s.fuse(fx, (cur, wb, wa, wp, wacc), torch.cat([mask, av * ones], 1), h)
                # Its next value: the same mix without the previous output.
                den = (w[:, 0:1] + w[:, 1:2] + w[:, 2:3] + w[:, 4:5]).clamp_min(1e-4)
                s.last_acc = (parts[0] + parts[1] + parts[2] + parts[4]) / den
                s.last_weights = w
            else:
                base, s.last_weights, _ = s.fuse(torch.cat([x] + mm, 1), (cur, wb, wa, wp), mask, h)
            x = torch.cat([base.to(x.dtype), x[:, 3:]] + ([cur] if s.keep_cur else []) + mm, 1)
        elif mm:
            x = torch.cat([x] + mm, 1)
        s.last_base = base
        if s.unet.state is None:
            return base + s.unet(x), (fb, fa)
        r, s.last_state = s.unet(x, h)
        out = base + r
        s._last_out = None if s.training else out
        return out, (fb, fa)


def pad16(x):
    H, W = x.shape[2:]
    return F.pad(x, (0, (16 - W % 16) % 16, 0, (16 - H % 16) % 16), mode="replicate")


def build_model(cfg):
    """The model of a run's config.json (older runs lack the motion settings)."""
    t = lambda v: tuple(int(x) for x in v.split(","))
    m = StableDenoiser(w=t(cfg["width"]), n=t(cfg["blocks"]), motion=t(cfg.get("motion_w") or "16,32,16"), motion_k=cfg["motion_k"],
                       motion_min=cfg.get("motion_min") or 32, motion_iters=cfg.get("motion_iters") or 1, fuse=cfg.get("fuse") or 0, fuse_mode=cfg.get("fuse_mode") or "residual", keep_cur=cfg.get("keep_cur") or False, motion_base=cfg.get("motion_base") or 4, mismatch=cfg.get("mismatch") or False, state=cfg.get("state") or 0, acc=cfg.get("acc") or False, flow_mix=cfg.get("flow_mix") or False, arch=cfg.get("arch") or "plain")
    m.warp_mode = cfg.get("warp", "bilinear")
    m.flow_prev = bool(cfg.get("flow_prev"))
    m.flow_match = bool(cfg.get("flow_match"))
    m.use_estimator = cfg.get("map") == "learned"
    if cfg.get("kp"):
        m.fuse.set_kp(int(cfg["kp"]))
    if cfg.get("refine"):
        w_ = int(cfg["refine"])
        m.refine = nn.Sequential(conv(24, w_), nn.ReLU(), conv(w_, w_), nn.ReLU(), conv(w_, w_), nn.ReLU(), conv(w_, 2))
        nn.init.zeros_(m.refine[-1].weight)
        nn.init.zeros_(m.refine[-1].bias)
    return m


def load_matching(model, state, prefix=""):
    """Load the weights of `state` whose names and shapes match (others are reported):
    for a checkpoint from before a part was resized or added. A convolution that only
    gained input channels keeps its weights, the new channels starting at zero (the
    model then computes what it did)."""
    own = model.state_dict()
    ok, grown = {}, []
    for k, v in state.items():
        if not k.startswith(prefix) or k not in own:
            continue
        if own[k].shape == v.shape:
            ok[k] = v
        elif ".fuse.net." in "." + k and own[k].shape[0] == 20 and v.shape[0] == 16 and own[k].shape[1:] == v.shape[1:]:
            # The pre-merge given a fifth picture (the accumulated one): its weight map
            # starts as a copy of the previous output's, 1.5 lower (about a fifth of
            # that one's share), so the model starts close to what it did.
            w = torch.cat([v, v[12:16] - (1.5 if v.dim() == 1 else 0.0)], 0)
            ok[k] = w
            grown.append(k)
        elif v.dim() == 4 and own[k].dim() == 4 and own[k].shape[0] == v.shape[0] and own[k].shape[2:] == v.shape[2:] and own[k].shape[1] > v.shape[1]:
            w = torch.zeros_like(own[k])
            w[:, : v.shape[1]] = v
            ok[k] = w
            grown.append(k)
    model.load_state_dict(ok, strict=False)
    if grown:
        print(f"inputs added (new channels start at zero): {grown}", flush=True)
    return sorted({k.split(".")[0 if not k.startswith("module.") else 1] for k in own if k.startswith(prefix) and k not in ok})
