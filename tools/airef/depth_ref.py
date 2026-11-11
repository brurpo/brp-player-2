"""Reference depth for our GPU port and the temporal-stability comparison, written out
in plain PyTorch (weights load with weights_only=True; no downloaded code runs):

- `da`:  Depth Anything V2 Small (Apache-2.0), one frame at a time;
- `vda`: Video Depth Anything Small (Apache-2.0), streamed one frame at a time as its
         video_depth_stream.py: the temporal head attends over a rolling cache of 31
         earlier frames' hidden states (the first two kept, then the latest 29).

Both resize as the official transform (cv2 cubic, the short side to 518, multiples of
14), normalize (ImageNet), and return relative inverse depth at the frame's size.

  python depth_ref.py da|vda <model.pth> <frames dir (png, sorted)> <out dir> [max frames]
Writes <out dir>/dNNNN.f32 per frame (c=1, h, w as u32, then floats) and timings.
"""
import sys, os, struct, math, time
import numpy as np, torch, cv2
import torch.nn.functional as F

mode, path, src, dst = sys.argv[1:5]
limit = int(sys.argv[5]) if len(sys.argv) > 5 else 10**9
W = {k: v.float().cuda() for k, v in torch.load(path, map_location="cpu", weights_only=True).items()}
H = "head" if mode == "vda" else "depth_head"
dev = "cuda"


def lin(x, n, bias=True):
    return F.linear(x, W[n + ".weight"], W.get(n + ".bias") if bias else None)


def conv(x, n, stride=1, act=None):
    w = W[n + ".weight"]
    y = F.conv2d(x, w, W.get(n + ".bias"), stride=stride, padding=w.shape[-1] // 2)
    return F.relu(y) if act == "relu" else y


def ln(x, n, eps):
    return F.layer_norm(x, (x.shape[-1],), W[n + ".weight"], W[n + ".bias"], eps)


# ---- DINOv2 ViT-S/14 -------------------------------------------------------------
def dinov2(x):
    b, _, h, w = x.shape
    t = F.conv2d(x, W["pretrained.patch_embed.proj.weight"], W["pretrained.patch_embed.proj.bias"], stride=14)
    ph, pw = t.shape[2:]
    t = t.flatten(2).transpose(1, 2)
    pos = W["pretrained.pos_embed"]
    n = pos.shape[1] - 1
    s = int(math.sqrt(n))
    pp = F.interpolate(pos[:, 1:].reshape(1, s, s, -1).permute(0, 3, 1, 2), scale_factor=((ph + 0.1) / s, (pw + 0.1) / s), mode="bicubic", antialias=False)
    pp = pp.permute(0, 2, 3, 1).reshape(1, -1, pos.shape[-1])
    t = torch.cat([W["pretrained.cls_token"].expand(b, -1, -1) + pos[:, :1], t + pp], 1)
    outs = []
    for i in range(12):
        p = f"pretrained.blocks.{i}"
        y = ln(t, p + ".norm1", 1e-6)
        q, k, v = lin(y, p + ".attn.qkv").reshape(b, -1, 3, 6, 64).permute(2, 0, 3, 1, 4)
        a = F.scaled_dot_product_attention(q, k, v).transpose(1, 2).reshape(b, -1, 384)
        t = t + lin(a, p + ".attn.proj") * W[p + ".ls1.gamma"]
        y = lin(F.gelu(lin(ln(t, p + ".norm2", 1e-6), p + ".mlp.fc1")), p + ".mlp.fc2")
        t = t + y * W[p + ".ls2.gamma"]
        if i in (2, 5, 8, 11):
            o = ln(t, "pretrained.norm", 1e-6)[:, 1:]
            outs.append(o.transpose(1, 2).reshape(b, 384, ph, pw))
    return outs, ph, pw


# ---- Temporal module (VDA) -----------------------------------------------------------
def temporal(x, n, cache):
    """x: [C][h][w] of the current frame. cache: per attention block, the list of
    earlier frames' normalized hidden states ([h*w][C] each) or None. Returns the new
    map and this frame's hidden states (per attention block)."""
    c, h, w = x.shape
    tt = n + ".temporal_transformer"
    y = F.group_norm(x[None], 32, W[tt + ".norm.weight"], W[tt + ".norm.bias"], 1e-6)[0]
    y = lin(y.permute(1, 2, 0).reshape(h * w, c), tt + ".proj_in")
    blk = tt + ".transformer_blocks.0"
    states = []
    for i in range(2):
        a = f"{blk}.attention_blocks.{i}"
        nh = ln(y, f"{blk}.norms.{i}", 1e-5)  # [hw][C]
        states.append(nh)
        seq = torch.stack((cache[i] if cache else []) + [nh], 1)  # [hw][T][C]
        seq = seq + W[a + ".pos_encoder.pe"][:, : seq.shape[1]]
        q = lin(seq[:, -1:], a + ".to_q", False)
        k = lin(seq, a + ".to_k", False)
        v = lin(seq, a + ".to_v", False)
        heads = 8
        sp = lambda z: z.reshape(z.shape[0], z.shape[1], heads, -1).transpose(1, 2)
        o = F.scaled_dot_product_attention(sp(q), sp(k), sp(v)).transpose(1, 2).reshape(h * w, 1, -1)[:, 0]
        y = y + lin(o, a + ".to_out.0")
    g, gate = lin(ln(y, f"{blk}.ff_norm", 1e-5), f"{blk}.ff.net.0.proj").chunk(2, -1)
    y = y + lin(g * F.gelu(gate), f"{blk}.ff.net.2")
    y = lin(y, tt + ".proj_out").reshape(h, w, c).permute(2, 0, 1)
    return x + y, states


# ---- DPT head ------------------------------------------------------------------------
def rcu(x, n):
    return x + conv(F.relu(conv(F.relu(x), n + ".conv1")), n + ".conv2")


def fuse(n, prev, skip=None, size=None):
    x = prev if skip is None else prev + rcu(skip, n + ".resConfUnit1")
    x = rcu(x, n + ".resConfUnit2")
    x = F.interpolate(x, size=size, mode="bilinear", align_corners=True) if size else F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=True)
    return conv(x, n + ".out_conv")


def head(feats, ph, pw, cache=None):
    r = []
    for i, f in enumerate(feats):
        x = conv(f, f"{H}.projects.{i}")
        if i == 0:
            x = F.conv_transpose2d(x, W[f"{H}.resize_layers.0.weight"], W[f"{H}.resize_layers.0.bias"], stride=4)
        elif i == 1:
            x = F.conv_transpose2d(x, W[f"{H}.resize_layers.1.weight"], W[f"{H}.resize_layers.1.bias"], stride=2)
        elif i == 3:
            x = F.conv2d(x, W[f"{H}.resize_layers.3.weight"], W[f"{H}.resize_layers.3.bias"], stride=2, padding=1)
        r.append(x)
    new = []
    if mode == "vda":
        for j, li in enumerate([2, 3]):
            o, st = temporal(r[li][0], f"{H}.motion_modules.{j}", cache[j] if cache else None)
            r[li] = o[None]
            new.append(st)
    s = f"{H}.scratch"
    rn = [conv(r[i], f"{s}.layer{i + 1}_rn") for i in range(4)]
    p4 = fuse(f"{s}.refinenet4", rn[3], None, rn[2].shape[2:])
    if mode == "vda":
        o, st = temporal(p4[0], f"{H}.motion_modules.2", cache[2] if cache else None)
        p4 = o[None]
        new.append(st)
    p3 = fuse(f"{s}.refinenet3", p4, rn[2], rn[1].shape[2:])
    if mode == "vda":
        o, st = temporal(p3[0], f"{H}.motion_modules.3", cache[3] if cache else None)
        p3 = o[None]
        new.append(st)
    p2 = fuse(f"{s}.refinenet2", p3, rn[1], rn[0].shape[2:])
    p1 = fuse(f"{s}.refinenet1", p2, rn[0])
    o = conv(p1, f"{s}.output_conv1")
    o = F.interpolate(o, (ph * 14, pw * 14), mode="bilinear", align_corners=True)
    o = conv(conv(o, f"{s}.output_conv2.0", act="relu"), f"{s}.output_conv2.2", act="relu")
    return F.relu(o), new


def prep(img):
    h, w = img.shape[:2]
    sc = max(518 / h, 518 / w)
    m = lambda v: max(int(np.round(v / 14) * 14), 518) if np.round(v / 14) * 14 >= 518 else int(np.ceil(v / 14) * 14)
    nh, nw = m(h * sc), m(w * sc)
    x = cv2.resize(img.astype(np.float32) / 255.0, (nw, nh), interpolation=cv2.INTER_CUBIC)
    x = (x - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
    return torch.from_numpy(x.transpose(2, 0, 1).astype(np.float32))[None].to(dev)


frames = sorted(f for f in os.listdir(src) if f.endswith(".png"))[:limit]
os.makedirs(dst, exist_ok=True)
caches = []  # per frame: [module][block] hidden states
times = []
with torch.inference_mode():
    for fi, f in enumerate(frames):
        img = cv2.cvtColor(cv2.imread(os.path.join(src, f)), cv2.COLOR_BGR2RGB)
        h, w = img.shape[:2]
        torch.cuda.synchronize(); t0 = time.time()
        x = prep(img)
        feats, ph, pw = dinov2(x)
        if mode == "vda":
            if fi == 0:
                d, st = head(feats, ph, pw)
                caches = [st] * 32
            else:
                cur = caches[0:2] + caches[-29:]
                cache = [[[c[m][b] for c in cur] for b in range(2)] for m in range(4)]
                d, st = head(feats, ph, pw, cache)
                caches.append(st)
            if fi + 32 > 42:
                del caches[1]
        else:
            d, _ = head(feats, ph, pw)
        d = F.interpolate(d, (h, w), mode="bilinear", align_corners=True)[0, 0]
        torch.cuda.synchronize(); times.append(time.time() - t0)
        a = d.cpu().numpy().astype("<f4")
        with open(os.path.join(dst, f"d{fi:04d}.f32"), "wb") as fh:
            fh.write(struct.pack("<3I", 1, h, w)); fh.write(a.tobytes())
print(f"{mode}: {len(frames)} frames, {np.median(times[1:] or times) * 1000:.1f} ms per frame (PyTorch fp32, median)")
