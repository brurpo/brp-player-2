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

    def __init__(s, c=(16, 32, 16), k=5, min_side=32, iters=1, base=4):
        super().__init__()
        s.level = FlowLevel(c, k)
        # Pyramid levels are added until the coarsest side is <= min_side px; each level
        # refines the flow `iters` times (same weights).
        # base: the finest level's size, as a divisor of the picture (4 or 2).
        s.min_side, s.iters, s.base = min_side, iters, base

    def forward(s, a, b):
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
            return F.interpolate(flow, scale_factor=s.base, mode="bilinear", align_corners=False) * float(s.base)


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

    def forward(s, x, frames=None, mask=None, h=None):
        y = s.net[0](F.pixel_unshuffle(x, 2))
        if s.in_h is not None:
            y = y + s.in_h(h.to(y.dtype))
        y = F.pixel_shuffle(s.net[1:](y), 2).float()
        if not s.blend:
            return frames[0] + y, None
        w = torch.softmax(y.masked_fill(mask < 0.5, -1e4), 1)
        return sum(w[:, k : k + 1] * frames[k].float() for k in range(s.pics)), w


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
    def __init__(s, w=(32, 64, 128), n=(2, 3, 6), motion=(16, 32, 16), motion_k=5, motion_min=32, motion_iters=1, fuse=0, fuse_mode="blend", keep_cur=False, motion_base=4, mismatch=False, state=0, acc=False):
        super().__init__()
        s.motion = Motion(motion, motion_k, motion_min, motion_iters, motion_base)
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
        s.unet = UNet((19 if s.keep_cur else 16) + mm, w, n, state=state)
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

    def forward(s, cur, before, after, valid, prev_out, prev_valid, strength, state=None, acc=None):
        """state: the memory returned by the previous step (`last_state`), for models
        that have one. Left out when not training: the one kept from the last call is
        used if `prev_out` is that call's output (so code that only feeds the previous
        output back runs these models unchanged), else none."""
        H, W = cur.shape[2:]
        fb = s.motion(cur, before)
        fa = s.motion(cur, after)
        s.last_flows = (fb, fa)  # for the motion loss and checks
        wb, wa = warp(before, fb, s.warp_mode), warp(after, fa, s.warp_mode)
        # The previous output lines up with t-1: moved along the same motion.
        wp = warp(prev_out, fb, s.warp_mode)
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
            h = warp(state.float(), F.avg_pool2d(fb, 2) * 0.5) * pv
        mm = [torch.cat([mismatch(cur, wb), mismatch(cur, wa), mismatch(cur, wp)], 1).to(x.dtype)] if s.mismatch else []
        if s.fuse is not None:
            # The frames merged over time first; the U-Net then works on that picture.
            mask = torch.cat([ones, valid[:, 0:1] * ones, valid[:, 1:2] * ones, prev_valid * ones], 1)
            if s.acc:
                # The accumulated picture, moved like the previous output; none at a
                # start or after a cut.
                av = prev_valid * (0.0 if acc is None else 1.0)
                wacc = warp(acc.float(), fb, s.warp_mode) * av if acc is not None else torch.zeros_like(cur)
                fx = torch.cat([x] + mm + [wacc.to(x.dtype), mismatch(cur, wacc).to(x.dtype)], 1)
                base, w = s.fuse(fx, (cur, wb, wa, wp, wacc), torch.cat([mask, av * ones], 1), h)
                # Its next value: the same mix without the previous output.
                den = (w[:, 0:1] + w[:, 1:2] + w[:, 2:3] + w[:, 4:5]).clamp_min(1e-4)
                s.last_acc = (w[:, 0:1] * cur.float() + w[:, 1:2] * wb.float() + w[:, 2:3] * wa.float() + w[:, 4:5] * wacc.float()) / den
                s.last_weights = w
            else:
                base, s.last_weights = s.fuse(torch.cat([x] + mm, 1), (cur, wb, wa, wp), mask, h)
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
                       motion_min=cfg.get("motion_min") or 32, motion_iters=cfg.get("motion_iters") or 1, fuse=cfg.get("fuse") or 0, fuse_mode=cfg.get("fuse_mode") or "residual", keep_cur=cfg.get("keep_cur") or False, motion_base=cfg.get("motion_base") or 4, mismatch=cfg.get("mismatch") or False, state=cfg.get("state") or 0, acc=cfg.get("acc") or False)
    m.warp_mode = cfg.get("warp", "bilinear")
    m.use_estimator = cfg.get("map") == "learned"
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
