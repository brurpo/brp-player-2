"""Speed probe for the stable denoiser (PLAN.md step 1): random weights, no training.

  python probe.py <out dir> [widths, e.g. 32,64,128] [blocks, e.g. 2,3,6]

- Counts multiply-adds per output pixel (convolutions; warps and resizes are small).
- Times one step in PyTorch (CUDA, fp16) at 1920x1088 and 3840x2160.
- Exports ONNX (opset 17) at 256x256 with dynamic size, checks ONNX Runtime (CPU)
  against PyTorch, and that it runs at another size.
- Writes graphs in our runner's format (.param/.bin, as pth2ncnn) for `aicheck
  custom`: `unet` (the merge; a 1x1 convolution makes its 16 inputs from the
  picture) and `motion` (one flow: the level network at 1/4, 1/8, ... of the
  picture, as many levels as the model uses at that size; warps left out).
"""
import os, sys, struct, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model import StableDenoiser, UNet, FlowLevel

out_dir = sys.argv[1]
w = tuple(int(v) for v in (sys.argv[2] if len(sys.argv) > 2 else "32,64,128").split(","))
n = tuple(int(v) for v in (sys.argv[3] if len(sys.argv) > 3 else "2,3,6").split(","))
mk = int(sys.argv[4]) if len(sys.argv) > 4 else 5
os.makedirs(out_dir, exist_ok=True)
torch.manual_seed(0)
dev = "cuda"
model = StableDenoiser(w, n, motion_k=mk).eval().to(dev)


def macs(m, shape):
    total = [0]
    hooks = []
    def hook(mod, inp, out):
        k = mod.kernel_size[0] * mod.kernel_size[1] * mod.in_channels // mod.groups
        total[0] += out.numel() * k
    for mod in m.modules():
        if isinstance(mod, nn.Conv2d):
            hooks.append(mod.register_forward_hook(hook))
    with torch.no_grad():
        m(*inputs(shape, torch.float32))
    for h in hooks:
        h.remove()
    return total[0]


def inputs(shape, dt):
    B, H, W = shape
    r = lambda c: torch.rand(B, c, H, W, device=dev, dtype=dt)
    return r(3), r(3), r(3), torch.ones(B, 2, 1, 1, device=dev, dtype=dt), r(3), torch.ones(B, 1, 1, 1, device=dev, dtype=dt), torch.full((B, 1, 1, 1), 0.5, device=dev, dtype=dt)


H, W = 1088, 1920
m_all = macs(model, (1, H, W))
m_unet = macs(model.unet, None) if False else None
px = H * W
print(f"widths {w}, blocks {n}, motion filters {mk}x{mk}")
print(f"multiply-adds per output pixel: {m_all / px / 1e3:.1f} k (whole step at 1920x1088: {m_all / 1e9:.1f} G)")

# PyTorch fp16 timing
mh = model.half().to(memory_format=torch.channels_last)
for (h, ww) in [(1088, 1920), (2160, 3840)]:
    x = [t.contiguous(memory_format=torch.channels_last) if t.dim() == 4 and t.shape[2] > 1 else t for t in inputs((1, h, ww), torch.float16)]
    with torch.no_grad():
        for _ in range(3):
            mh(*x)
        torch.cuda.synchronize()
        t = time.time()
        k = 10
        for _ in range(k):
            mh(*x)
        torch.cuda.synchronize()
    print(f"PyTorch fp16 {ww}x{h}: {(time.time() - t) / k * 1000:.1f} ms per step, memory {torch.cuda.max_memory_allocated() >> 20} MB")
model = model.float()

# ONNX export and check
try:
    import onnxruntime as ort
    m32 = StableDenoiser(w, n, motion_k=mk).eval()
    m32.load_state_dict({k: v.float().cpu() for k, v in model.state_dict().items()})
    names = ["cur", "before", "after", "valid", "prev_out", "prev_valid", "strength"]
    xi = [t.float().cpu() for t in inputs((1, 256, 256), torch.float32)]
    path = os.path.join(out_dir, "probe.onnx")
    dyn = {nm: {2: "h", 3: "w"} for nm in ["cur", "before", "after", "prev_out"]}
    dyn["out"] = {2: "h", 3: "w"}
    torch.onnx.export(m32, tuple(xi), path, input_names=names, output_names=["out", "flow_b", "flow_a"], dynamic_axes=dyn, opset_version=17, dynamo=False)
    sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    ref = m32(*xi)[0].detach().numpy()
    got = sess.run(["out"], {nm: t.numpy() for nm, t in zip(names, xi)})[0]
    print(f"ONNX: exported ({os.path.getsize(path) >> 10} KB); ONNX Runtime vs PyTorch at 256x256: max diff {np.abs(ref - got).max():.2e}")
    x2 = [t.float().cpu() for t in inputs((1, 384, 512), torch.float32)]
    got2 = sess.run(["out"], {nm: t.numpy() for nm, t in zip(names, x2)})[0]
    print(f"ONNX: runs at 512x384 too -> {got2.shape}")
except Exception as e:
    print(f"ONNX: failed: {e}")


# Our runner's format
class G:
    def __init__(s):
        s.layers, s.bin, s.n = ["Input input 0 1 img"], bytearray(), 0
    def blob(s):
        s.n += 1
        return f"b{s.n}"
    def conv(s, c: nn.Conv2d, x, act):
        wt = c.weight.detach().float().cpu().numpy().ravel()
        b = c.bias.detach().float().cpu().numpy() if c.bias is not None else np.zeros(c.out_channels, np.float32)
        s.bin += struct.pack("<I", 0) + wt.astype("<f4").tobytes() + b.astype("<f4").tobytes()
        o = s.blob()
        k = c.kernel_size[0]
        a = f" 9={act}" if act else ""
        s.layers.append(f"Convolution c{s.n} 1 1 {x} {o} 0={c.out_channels} 1={k} 3={c.stride[0]} 4={k // 2} 5=1 6={wt.size}{a}")
        return o
    def op(s, line):
        o = s.blob()
        s.layers.append(line.format(o=o))
        return o
    def seq(s, sq, x):
        mods = list(sq)
        i = 0
        while i < len(mods):
            m = mods[i]
            if isinstance(m, nn.Conv2d):
                relu = i + 1 < len(mods) and isinstance(mods[i + 1], nn.ReLU)
                x = s.conv(m, x, 1 if relu else 0)
                i += 2 if relu else 1
            elif isinstance(m, nn.PixelShuffle):
                x = s.op(f"PixelShuffle ps{s.n} 1 1 {x} {{o}} 0=2")
                i += 1
            else:
                i += 1
        return x
    def write(s, stem, out):
        s.layers.append(f"Split out_split 1 1 {out} out")
        blobs = set()
        for l in s.layers:
            f = l.split()
            ni, no = int(f[2]), int(f[3])
            blobs.update(f[4:4 + ni + no])
        open(stem + ".param", "w").write(f"7767517\n{len(s.layers)} {len(blobs)}\n" + "\n".join(s.layers) + "\n")
        open(stem + ".bin", "wb").write(bytes(s.bin))


u: UNet = model.unet
g = G()
make16 = nn.Conv2d(3, 16, 1).to(dev)
x = g.conv(make16, "img", 0)
x = g.op(f"PixelUnshuffle pu 1 1 {x} {{o}} 0=2")
a = g.seq(u.e1, g.seq(u.inp, x))
b = g.seq(u.e2, g.seq(u.d1, a))
c = g.seq(u.mid, g.seq(u.d2, b))
up = g.seq(u.u2, c)
b2 = g.op(f"BinaryOp add2 2 1 {up} {b} {{o}} 0=0")
b2 = g.seq(u.dec2, b2)
up = g.seq(u.u1, b2)
a2 = g.op(f"BinaryOp add1 2 1 {up} {a} {{o}} 0=0")
a2 = g.seq(u.dec1, a2)
o = g.conv(u.out, a2, 0)
o = g.op(f"PixelShuffle psout 1 1 {o} {{o}} 0=2")
o = g.op(f"BinaryOp res 2 1 {o} img {{o}} 0=0")
g.write(os.path.join(out_dir, "unet"), o)

# One flow: the level network at 1/4, 1/8, ... (levels as Motion makes them at 1088x1920).
lv: FlowLevel = model.motion.level
g = G()
four = nn.Conv2d(3, 4, 1).to(dev)
hh, ww = H // 4, W // 4
outs = []
while True:
    r = g.op(f"Interp down{s if (s := len(outs)) else 0} 1 1 img {{o}} 0=2 3={hh} 4={ww}")
    r = g.conv(four, r, 0)
    r = g.seq(lv.net, r)
    outs.append(r)
    if min(hh, ww) <= 32:
        break
    hh, ww = hh // 2, ww // 2
# Keep every level's output alive: add them after resizing to the first.
acc = outs[0]
for i, r in enumerate(outs[1:]):
    up = g.op(f"Interp up{i} 1 1 {r} {{o}} 0=2 3={H // 4} 4={W // 4}")
    acc = g.op(f"BinaryOp acc{i} 2 1 {acc} {up} {{o}} 0=0")
g.write(os.path.join(out_dir, "motion"), acc)
print(f"runner graphs written to {out_dir} (unet, motion: {len(outs)} levels)")
