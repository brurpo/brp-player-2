"""Reference for our GPU SCUNet: SCUNet (cszn/SCUNet, Apache-2.0) written out in plain
PyTorch, run on a picture (replicate-padded to a multiple of 64, as the official
forward). Writes raw f32 (c, h, w as u32, then planar floats) for `aicheck custom`
(BRP_REF). Weights load with weights_only=True; no downloaded code runs.

  python scunet_ref.py <model.pth> in.png out.f32 [out.png]
"""
import sys, struct, time
import numpy as np, torch
import torch.nn.functional as F
from PIL import Image

path, src, out = sys.argv[1], sys.argv[2], sys.argv[3]
W = {k: v.cuda() for k, v in torch.load(path, map_location="cpu", weights_only=True).items()}
WS, HD = 8, 32


def lin(x, n):  # x: b h w c
    return F.linear(x, W[n + ".weight"], W.get(n + ".bias"))


def conv(x, n, stride=1):
    w = W[n + ".weight"]
    return F.conv2d(x, w, W.get(n + ".bias"), stride=stride, padding=w.shape[-1] // 2 if stride == 1 else 0)


def wmsa(x, n, shifted):
    b, h, w, c = x.shape
    heads, p = c // HD, WS
    if shifted:
        x = torch.roll(x, (-(p // 2), -(p // 2)), (1, 2))
    hw, ww = h // p, w // p
    x = x.view(b, hw, p, ww, p, c).permute(0, 1, 3, 2, 4, 5).reshape(b, hw * ww, p * p, c)
    qkv = lin(x, n + ".embedding_layer").view(b, hw * ww, p * p, 3 * heads, HD).permute(3, 0, 1, 2, 4)
    q, k, v = qkv.chunk(3, 0)
    sim = torch.einsum("hbwpc,hbwqc->hbwpq", q, k) * HD ** -0.5
    cord = torch.tensor([[i, j] for i in range(p) for j in range(p)])
    rel = cord[:, None, :] - cord[None, :, :] + p - 1
    sim = sim + W[n + ".relative_position_params"][:, rel[..., 0], rel[..., 1]][:, None, None]
    if shifted:
        s = p - p // 2
        m = torch.zeros(hw, ww, p, p, p, p, dtype=torch.bool, device=x.device)
        m[-1, :, :s, :, s:, :] = True
        m[-1, :, s:, :, :s, :] = True
        m[:, -1, :, :s, :, s:] = True
        m[:, -1, :, s:, :, :s] = True
        sim = sim.masked_fill(m.reshape(1, 1, hw * ww, p * p, p * p), float("-inf"))
    o = torch.einsum("hbwij,hbwjc->hbwic", sim.softmax(-1), v)
    o = o.permute(1, 2, 3, 0, 4).reshape(b, hw * ww, p * p, c)
    o = lin(o, n + ".linear").view(b, hw, ww, p, p, c).permute(0, 1, 3, 2, 4, 5).reshape(b, h, w, c)
    if shifted:
        o = torch.roll(o, (p // 2, p // 2), (1, 2))
    return o


def ctb(x, n, shifted):
    y = conv(x, n + ".conv1_1")
    half = y.shape[1] // 2
    cx, tx = y[:, :half], y[:, half:]
    cx = conv(F.relu(conv(cx, n + ".conv_block.0")), n + ".conv_block.2") + cx
    t = tx.permute(0, 2, 3, 1)
    tb = n + ".trans_block"
    t = t + wmsa(F.layer_norm(t, (half,), W[tb + ".ln1.weight"], W[tb + ".ln1.bias"]), tb + ".msa", shifted)
    t = t + lin(F.gelu(lin(F.layer_norm(t, (half,), W[tb + ".ln2.weight"], W[tb + ".ln2.bias"]), tb + ".mlp.0")), tb + ".mlp.2")
    return x + conv(torch.cat([cx, t.permute(0, 3, 1, 2)], 1), n + ".conv1_2")


def count(p, start):
    i = start
    while f"{p}.{i}.conv1_1.weight" in W:
        i += 1
    return i - start


def stage(x, p, start):
    for i in range(count(p, start)):
        x = ctb(x, f"{p}.{start + i}", i % 2 == 1)
    return x


def net(x0):
    h, w = x0.shape[2:]
    x0 = F.pad(x0, (0, (64 - w % 64) % 64, 0, (64 - h % 64) % 64), mode="replicate")
    x1 = conv(x0, "m_head.0")
    skips, x = [x1], x1
    for p in ["m_down1", "m_down2", "m_down3"]:
        x = conv(stage(x, p, 0), f"{p}.{count(p, 0)}", stride=2)
        skips.append(x)
    x = stage(x, "m_body", 0)
    for l, p in enumerate(["m_up3", "m_up2", "m_up1"]):
        x = F.conv_transpose2d(x + skips[3 - l], W[p + ".0.weight"], stride=2)
        x = stage(x, p, 1)
    return conv(x + skips[0], "m_tail.0")[:, :, :h, :w]


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
