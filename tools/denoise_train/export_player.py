"""Write a checkpoint as the files the player loads (crates/brp-ai/src/learned.rs):

  export_player.py RUN_DIR [CKPT=best] [OUT_DIR=<repo>/target/release/models/brpdn]

Three small networks in ncnn's format (.param text + .bin weights), run by the
player's own Vulkan runner, and `config.txt` (the settings that are not weights):

- flow:  one refinement of the motion stage (the same weights at every pyramid level).
         Input `x` = [frame luma, moved neighbour luma, flow x, flow y]; output `out` =
         the correction to the flow.
- fuse:  the pre-merge (with a memory it also reads `h`, as `unet` does). Input `x` =
         22 channels at full size (frame, the two moved
         neighbours, the moved previous output, 4 flags, 6 mismatch maps); output
         `out` = 4 weight maps before the softmax.
- unet:  the merge. Inputs `x` (the same 22 channels) and `base` (the pre-merged
         picture); output `out` = the denoised picture (base + correction). A model
         with a memory (config `state` N) also reads `h` (N channels at 1/2 size: the
         memory written at the frame before, moved onto this one) and writes `state`
         (the new memory, before its clamp to -1..1).

Everything between them (pyramid, warps, mismatch maps, cut test, softmax mix, PQ) is
compute kernels in learned.rs. The U-Net's first convolution is written with its input
channels reordered to [x, base], so the player builds the 22 channels once for both.
"""
import json, os, struct, sys

import numpy as np
import torch

from model import build_model


class Graph:
    def __init__(s):
        s.lines, s.blobs, s.bin = [], [], bytearray()

    def add(s, ty, name, ins, outs, params=""):
        for b in outs:
            assert b not in s.blobs, b
            s.blobs.append(b)
        s.lines.append(f"{ty:<16} {name:<12} {len(ins)} {len(outs)} {' '.join(ins + outs)} {params}".rstrip())

    def conv(s, name, src, dst, weight, bias, stride=1, relu=False):
        w = weight.detach().float().cpu().numpy()
        b = bias.detach().float().cpu().numpy()
        oc, ic, kh, kw = w.shape
        assert kh == kw
        p = f"0={oc} 1={kw} 3={stride} 4={kw // 2} 5=1 6={w.size}" + (" 9=1" if relu else "")
        s.add("Convolution", name, [src], [dst], p)
        s.bin += struct.pack("<I", 0) + w.astype("<f4").tobytes() + b.astype("<f4").tobytes()

    def seq(s, prefix, src, convs, last_relu=True):
        """Convolutions one after another ((module, stride) pairs), ReLU after each."""
        for i, (m, stride) in enumerate(convs):
            dst = f"{prefix}{i}"
            s.conv(dst, src, dst, m.weight, m.bias, stride, relu=last_relu or i < len(convs) - 1)
            src = dst
        return src

    def write(s, path):
        with open(path + ".param", "w", newline="\n") as f:
            f.write("7767517\n%d %d\n%s\n" % (len(s.lines), len(s.blobs), "\n".join(s.lines)))
        with open(path + ".bin", "wb") as f:
            f.write(bytes(s.bin))


def convs_of(seq):
    return [(m, m.stride[0]) for m in seq if isinstance(m, torch.nn.Conv2d)]


def export(run, ckpt, out):
    cfg = json.load(open(os.path.join(run, "config.json")))
    ck = torch.load(os.path.join(run, "ckpt", ckpt + ".pt"), map_location="cpu", weights_only=False)
    m = build_model(cfg).eval()
    sd = {k[7:]: v for k, v in ck["ema"].items() if k.startswith("module.")}
    missing = [k for k in m.state_dict() if k not in sd and not k.startswith("estimator.")]
    assert not missing, missing
    m.load_state_dict({k: v for k, v in sd.items() if k in m.state_dict()}, strict=False)
    # What learned.rs implements.
    assert m.fuse is not None and m.fuse.blend and m.keep_cur and m.mismatch and not m.use_estimator and m.warp_mode == "bilinear", cfg
    os.makedirs(out, exist_ok=True)

    g = Graph()
    g.add("Input", "x", [], ["x"])
    g.seq("f", "x", convs_of(m.motion.level.net), last_relu=False)
    g.lines[-1] = g.lines[-1].replace(g.blobs[-1], "out")
    g.blobs[-1] = "out"
    g.write(os.path.join(out, "flow"))

    u = m.unet
    mem = u.state.out_channels if u.state is not None else 0
    assert mem % 4 == 0 and (not mem or m.fuse.in_h is not None), mem
    g = Graph()
    g.add("Input", "x", [], ["x"])
    if mem:
        g.add("Input", "h", [], ["h"])
    g.add("PixelUnshuffle", "unsh", ["x"], ["xu"], "0=2")
    fc, src = convs_of(m.fuse.net), "xu"
    if mem:
        # The memory's convolution beside the first one = one convolution over both.
        g.add("Concat", "cat_h", ["xu", "h"], ["xh"], "0=0")
        both = torch.nn.Conv2d(fc[0][0].in_channels + mem, fc[0][0].out_channels, 3, 1, 1)
        both.weight.data, both.bias.data = torch.cat([fc[0][0].weight.detach(), m.fuse.in_h.weight.detach()], 1), fc[0][0].bias.detach().clone()
        fc, src = [(both, 1)] + fc[1:], "xh"
    last = g.seq("m", src, fc, last_relu=False)
    g.add("PixelShuffle", "sh", [last], ["out"], "0=2")
    g.write(os.path.join(out, "fuse"))

    # Input channels as the player gives them: x (22: frame, moved before / after /
    # previous output, 4 flags, 6 mismatch maps) then base (3). The model's order is
    # base, x[3:16], frame, mismatch maps.
    order = list(range(16, 19)) + list(range(3, 16)) + list(range(19, 25)) + [0, 1, 2]
    w0 = u.inp[0].weight.detach()
    idx = torch.tensor([c * 4 + k for c in order for k in range(4)])
    assert w0.shape[1] == 100, w0.shape
    first = torch.nn.Conv2d(100 + mem, w0.shape[0], 3, 1, 1)
    # The memory's convolution beside the first one = one convolution over both.
    first.weight.data, first.bias.data = torch.cat([w0[:, idx]] + ([u.inp_h.weight.detach()] if mem else []), 1).clone(), u.inp[0].bias.detach().clone()
    g = Graph()
    g.add("Input", "x", [], ["x"])
    g.add("Input", "base", [], ["base"])
    if mem:
        g.add("Input", "h", [], ["h"])
    g.add("Split", "base_s", ["base"], ["base_a", "base_b"])
    g.add("Concat", "cat", ["x", "base_a"], ["xin"], "0=0")
    g.add("PixelUnshuffle", "unsh", ["xin"], ["xu"], "0=2")
    src = "xu"
    if mem:
        g.add("Concat", "cat_h", ["xu", "h"], ["xh"], "0=0")
        src = "xh"
    a = g.seq("e1_", src, [(first, 1)] + convs_of(u.e1))
    g.add("Split", "a_s", [a], ["a_a", "a_b"])
    b = g.seq("e2_", "a_a", convs_of(u.d1) + convs_of(u.e2))
    g.add("Split", "b_s", [b], ["b_a", "b_b"])
    c = g.seq("mid_", "b_a", convs_of(u.d2) + convs_of(u.mid))
    g.conv("u2", c, "u2", u.u2[0].weight, u.u2[0].bias)
    g.add("PixelShuffle", "u2s", ["u2"], ["u2s"], "0=2")
    g.add("BinaryOp", "add2", ["u2s", "b_b"], ["s2"], "0=0")
    b = g.seq("dec2_", "s2", convs_of(u.dec2))
    g.conv("u1", b, "u1", u.u1[0].weight, u.u1[0].bias)
    g.add("PixelShuffle", "u1s", ["u1"], ["u1s"], "0=2")
    g.add("BinaryOp", "add1", ["u1s", "a_b"], ["s1"], "0=0")
    a = g.seq("dec1_", "s1", convs_of(u.dec1))
    if mem:
        g.add("Split", "last_s", [a], ["last_a", "last_b"])
        g.conv("state", "last_b", "state", u.state.weight, u.state.bias)
        a = "last_a"
    g.conv("res", a, "res", u.out.weight, u.out.bias)
    g.add("PixelShuffle", "ress", ["res"], ["resid"], "0=2")
    g.add("BinaryOp", "sum", ["resid", "base_b"], ["out"], "0=0")
    g.write(os.path.join(out, "unet"))

    with open(os.path.join(out, "config.txt"), "w", newline="\n") as f:
        f.write(f"motion_min {m.motion.min_side}\nmotion_iters {m.motion.iters}\nmotion_base {m.motion.base}\ncut_thr 0.026\nstate {mem}\n")
        f.write(f"source {os.path.basename(os.path.normpath(run))}/{ckpt} step {ck['step']}\n")
    print(f"{run} {ckpt} (step {ck['step']}) -> {out}")
    for n in ("flow", "fuse", "unet"):
        print(f"  {n}: {os.path.getsize(os.path.join(out, n + '.bin')) / 1e6:.2f} MB")


if __name__ == "__main__":
    here = os.path.dirname(os.path.abspath(__file__))
    run = sys.argv[1]
    ckpt = sys.argv[2] if len(sys.argv) > 2 else "best"
    out = sys.argv[3] if len(sys.argv) > 3 else os.path.join(here, "..", "..", "target", "release", "models", "brpdn")
    export(run, ckpt, os.path.normpath(out))
