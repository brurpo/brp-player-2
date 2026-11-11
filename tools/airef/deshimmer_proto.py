"""Prototype: steady fine detail that shimmers (comes and goes from frame to frame)
by averaging only the high-frequency band over neighbours aligned along the motion,
keeping the current frame's low band.

  python deshimmer_proto.py <dir with s000.png...> <flows fwd> <flows bwd> <radius> <sigma> <out dir>

Flows as `aicheck searaft` writes them (fwd from the frames in order, bwd from them
reversed). Prints per-frame sharpness (variance of the Laplacian of luma) before and
after, its frame-to-frame spread, and the motion-compensated change of the high band.
"""
import sys, os
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

src, fw, bw, R, sigma, out = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4]), float(sys.argv[5]), sys.argv[6]
os.makedirs(out, exist_ok=True)
dev = "cuda"
names = sorted(f for f in os.listdir(src) if f.startswith("s") and f.endswith(".png"))
n = len(names)
img = [torch.from_numpy(np.asarray(Image.open(os.path.join(src, f)).convert("RGB"), dtype=np.float32) / 255).permute(2, 0, 1).to(dev) for f in names]
H, W = img[0].shape[1:]


def ld(p):
    raw = open(p, "rb").read()
    c, h, w = np.frombuffer(raw[:12], "<u4")
    f = torch.from_numpy(np.frombuffer(raw[12:], "<f4").reshape(c, h, w).copy()).to(dev)
    return F.interpolate(f[None], size=(H, W), mode="bilinear", align_corners=False)[0] * (W / w) if (h, w) != (H, W) else f


fwd = [ld(f"{fw}/{t:03d}.f32") for t in range(n - 1)]  # t -> t+1
bwd = [None] + [ld(f"{bw}/{n - 1 - t:03d}.f32") for t in range(1, n)]  # t -> t-1
ys, xs = torch.meshgrid(torch.arange(H, device=dev, dtype=torch.float32), torch.arange(W, device=dev, dtype=torch.float32), indexing="ij")


def sample(a, flow):
    x, y = xs + flow[0], ys + flow[1]
    g = torch.stack([x / (W - 1) * 2 - 1, y / (H - 1) * 2 - 1], -1)[None]
    inside = (x >= 0) & (x <= W - 1) & (y >= 0) & (y <= H - 1)
    return F.grid_sample(a[None], g, mode="bicubic", align_corners=True, padding_mode="border")[0], inside


def blur(a, s):
    k = int(3 * s) * 2 + 1
    g = torch.exp(-((torch.arange(k, device=dev) - k // 2) ** 2) / (2 * s * s))
    g = g / g.sum()
    c = a.shape[0]
    a = F.conv2d(F.pad(a[None], (k // 2, k // 2, 0, 0), mode="replicate"), g.view(1, 1, 1, k).repeat(c, 1, 1, 1), groups=c)
    a = F.conv2d(F.pad(a, (0, 0, k // 2, k // 2), mode="replicate"), g.view(1, 1, k, 1).repeat(c, 1, 1, 1), groups=c)
    return a[0]


def flow_to(t, d):
    """Motion from frame t to t + d (d != 0), chained, and where it is trusted."""
    step = 1 if d > 0 else -1
    tot = torch.zeros(2, H, W, device=dev)
    ok = torch.ones(H, W, dtype=torch.bool, device=dev)
    cur = t
    for _ in range(abs(d)):
        f = fwd[cur] if step > 0 else bwd[cur]
        back = bwd[cur + 1] if step > 0 else fwd[cur - 1]
        f_at, inside = sample(f, tot)
        # forward-backward check of this step, where the chain has got to
        b_at, _ = sample(back, tot + f_at)
        ok &= inside & ((f_at + b_at).norm(dim=0) < 1.0)
        tot = tot + f_at
        cur += step
    return tot, ok


def luma(a):
    return 0.299 * a[0] + 0.587 * a[1] + 0.114 * a[2]


def sharp(a):
    l = luma(a)
    lap = 4 * l[1:-1, 1:-1] - l[:-2, 1:-1] - l[2:, 1:-1] - l[1:-1, :-2] - l[1:-1, 2:]
    return (lap * 255).var().item()


res = []
for t in range(n):
    low = blur(img[t], sigma)
    hp_sum = img[t] - low
    wsum = torch.ones(H, W, device=dev)
    # Energy of each frame's own detail (for the gain), weighted as the detail.
    e_sum = (hp_sum ** 2).sum(0)
    bx = lambda a: F.avg_pool2d(a[None, None], 9, 1, 4, count_include_pad=False)[0, 0]
    E0 = bx(e_sum)
    S1, S2 = E0.clone(), E0 ** 2
    for d in [x for r in range(1, R + 1) for x in (-r, r)]:
        if not 0 <= t + d < n:
            continue
        fl, ok = flow_to(t, d)
        nb, _ = sample(img[t + d], fl)
        nlow = blur(nb, sigma)
        agree = (luma(nlow) - luma(low)).abs() < 0.03
        w = (ok & agree).float() * (0.85 ** (abs(d) - 1))
        hp_sum = hp_sum + w * (nb - nlow)
        e_sum = e_sum + w * ((nb - nlow) ** 2).sum(0)
        Ek = bx(((nb - nlow) ** 2).sum(0))
        S1 = S1 + w * Ek
        S2 = S2 + w * Ek ** 2
        wsum = wsum + w
    if os.environ.get("REGION"):
        x0, y0, x1, y1 = map(int, os.environ["REGION"].split(","))
        print("frame %d: neighbours used in region %.2f of %d" % (t, (wsum[y0:y1, x0:x1] - 1).mean().item(), 2 * R))
    hp = hp_sum / wsum
    if os.environ.get("LEVEL"):
        # Keep this frame's own detail; only its strength, per area, goes to the
        # window's mean (the crisp/soft pumping), mixed with the average by LEVEL_MIX.
        g = (S1 / wsum / E0.clamp_min(1e-10)).sqrt().clamp(0.5, 2.0)
        own = (img[t] - low) * g
        mix = float(os.environ.get("LEVEL_MIX", "0"))
        hp = own * (1 - mix) + hp * mix
    if os.environ.get("QMAP"):
        box = lambda a: F.avg_pool2d(a[None, None], 9, 1, 4, count_include_pad=False)[0, 0]
        m1 = S1 / wsum
        q = ((S2 / wsum - m1 ** 2).clamp_min(0).sqrt() / m1.clamp_min(1e-10)) * (m1 * 255 * 255 / 3 > 4).float()
        if t == 4:
            Image.fromarray((q / 0.6).clamp(0, 1).mul(255).byte().cpu().numpy()).save(os.path.join(out, "qmap.png"))
        for nm, (x0, y0, x1, y1) in {"tower": (960, 520, 1020, 630), "grass": (100, 420, 500, 560), "sky": (100, 50, 900, 200), "cars": (1500, 760, 1700, 900), "palms": (0, 820, 300, 1060)}.items():
            r = q[y0:y1, x0:x1]
            print("frame %d %-6s disagreement median %.2f, 90%% %.2f" % (t, nm, r.median().item(), r.flatten().kthvalue(int(r.numel() * 0.9)).values.item()))
    if os.environ.get("GAIN"):
        # Back to the window's average detail strength, measured over a box.
        box = lambda a: F.avg_pool2d(a[None, None], 31, 1, 15, count_include_pad=False)[0, 0]
        g = (box(e_sum / wsum) / box((hp ** 2).sum(0)).clamp_min(1e-8)).sqrt().clamp(1.0, float(os.environ["GAIN"]))
        hp = hp * g
    o = (low + hp).clamp(0, 1)
    res.append(o)
    Image.fromarray((o.permute(1, 2, 0).cpu().numpy() * 255 + 0.5).astype(np.uint8)).save(os.path.join(out, names[t]))

a = np.array([sharp(x) for x in img])
b = np.array([sharp(x) for x in res])
print("sharpness before:", " ".join("%.0f" % x for x in a))
print("sharpness after: ", " ".join("%.0f" % x for x in b))
for nm, s in (("before", a), ("after", b)):
    print("%s: mean %.0f, frame-to-frame change %.1f%%" % (nm, s.mean(), np.abs(np.diff(s)).mean() / s.mean() * 100))


def mc_change(seq):
    v = []
    for t in range(1, n):
        prev, inside = sample(seq[t - 1] - blur(seq[t - 1], sigma), bwd[t])
        f_at, _ = sample(fwd[t - 1], bwd[t])
        tr = inside & ((bwd[t] + f_at).norm(dim=0) < 1.0)
        d = ((seq[t] - blur(seq[t], sigma)) - prev).abs().mean(0)
        v.append(d[tr].mean().item())
    return np.mean(v)


print("detail change along the motion: before %.5f, after %.5f" % (mc_change(img), mc_change(res)))

if os.environ.get("REGION"):
    x0, y0, x1, y1 = map(int, os.environ["REGION"].split(","))

    def region_change(seq):
        v = []
        for t in range(1, n):
            prev, inside = sample(seq[t - 1], bwd[t])
            d = (seq[t] - prev).abs().mean(0)[y0:y1, x0:x1]
            v.append(d.mean().item())
        return np.mean(v)

    print("region change along the motion (whole picture): before %.4f, after %.4f" % (region_change(img), region_change(res)))
