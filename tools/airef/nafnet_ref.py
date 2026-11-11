"""Reference for our GPU NAFNet: NAFNet (megvii-research/NAFNet, MIT) written out in
plain PyTorch, as its official test setting (NAFNetLocal: channel attention averages
over a local box of 1.5x the 256 training crop, clamped inside the picture), run on a
picture. Writes the output as raw f32 (c, h, w as u32, then planar floats), the
format `aicheck custom` compares against (BRP_REF). Weights load with
weights_only=True; no downloaded code runs.

  python nafnet_ref.py <model.pth> in.png out.f32 [out.png]
"""
import sys, struct, time
import numpy as np, torch
import torch.nn.functional as F
from PIL import Image

path, src, out = sys.argv[1], sys.argv[2], sys.argv[3]
W = {k.removeprefix("module."): v.cuda() for k, v in torch.load(path, map_location="cpu", weights_only=True)["params"].items()}


def conv(x, n, stride=1, groups=1):
    w, b = W[n + ".weight"], W.get(n + ".bias")
    return F.conv2d(x, w, b, stride=stride, padding=w.shape[-1] // 2 if stride == 1 else 0, groups=groups)


def norm(x, n, eps=1e-6):
    mu = x.mean(1, keepdim=True)
    var = (x - mu).pow(2).mean(1, keepdim=True)
    return W[n + ".weight"].view(1, -1, 1, 1) * (x - mu) / (var + eps).sqrt() + W[n + ".bias"].view(1, -1, 1, 1)


def local_mean(x, k):
    # The official AvgPool2d (fast_imp=False, auto_pad): box sums from an integral
    # image, replicate-padded to the input size; the global mean if the box covers it.
    n, c, h, w = x.shape
    if (k >= h and k >= w) or __import__("os").environ.get("NOTLC"):
        return x.mean((2, 3), keepdim=True)
    s = F.pad(x.cumsum(-1).cumsum(-2), (1, 0, 1, 0))
    k1, k2 = min(h, k), min(w, k)
    o = s[:, :, k1:, k2:] + s[:, :, :-k1, :-k2] - s[:, :, :-k1, k2:] - s[:, :, k1:, :-k2]
    o = o / (k1 * k2)
    _h, _w = o.shape[2:]
    return F.pad(o, ((w - _w) // 2, (w - _w + 1) // 2, (h - _h) // 2, (h - _h + 1) // 2), mode="replicate")


def gate(x):
    a, b = x.chunk(2, 1)
    return a * b


def block(x, p, level):
    y = conv(norm(x, p + ".norm1"), p + ".conv1")
    y = gate(conv(y, p + ".conv2", groups=y.shape[1]))
    y = y * conv(local_mean(y, 384 >> level), p + ".sca.1")
    y = x + conv(y, p + ".conv3") * W[p + ".beta"]
    z = conv(gate(conv(norm(y, p + ".norm2"), p + ".conv4")), p + ".conv5")
    return y + z * W[p + ".gamma"]


def count(p):
    i = 0
    while f"{p}.{i}.norm1.weight" in W:
        i += 1
    return i


levels = sum(1 for k in W if k.startswith("downs.") and k.endswith(".weight"))


def net(inp):
    h, w = inp.shape[2:]
    m = 2 ** levels
    x0 = F.pad(inp, (0, (m - w % m) % m, 0, (m - h % m) % m))
    x = conv(x0, "intro")
    skips = []
    for l in range(levels):
        for i in range(count(f"encoders.{l}")):
            x = block(x, f"encoders.{l}.{i}", l)
        skips.append(x)
        x = conv(x, f"downs.{l}", stride=2)
    for i in range(count("middle_blks")):
        x = block(x, f"middle_blks.{i}", levels)
    for j in range(levels):
        x = F.pixel_shuffle(conv(x, f"ups.{j}.0"), 2) + skips[levels - 1 - j]
        for i in range(count(f"decoders.{j}")):
            x = block(x, f"decoders.{j}.{i}", levels - 1 - j)
    return (conv(x, "ending") + x0)[:, :, :h, :w]


img = np.asarray(Image.open(src).convert("RGB"), dtype=np.float32) / 255.0
x = torch.from_numpy(img).permute(2, 0, 1)[None].cuda()
with torch.inference_mode():
    y = net(x)
    torch.cuda.synchronize(); t = time.time()
    y = net(x)
    torch.cuda.synchronize()
print(f"{img.shape[1]}x{img.shape[0]}: {(time.time() - t) * 1000:.1f} ms (PyTorch fp32)")
f = y[0].float().cpu().numpy()
with open(out, "wb") as fh:
    fh.write(struct.pack("<3I", *f.shape)); fh.write(f.astype("<f4").tobytes())
if len(sys.argv) > 4:
    Image.fromarray((f.clip(0, 1).transpose(1, 2, 0) * 255 + 0.5).astype(np.uint8)).save(sys.argv[4])
