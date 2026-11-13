"""The degradation chain of degrade.py, split so the GPU does the heavy part.

- `recipes.recipe(r, kind, hdr)` (CPU, data workers, NumPy only): draws every
  setting the chain will use, with degrade.degrade's choices and distributions.
- `GpuDegrader.apply` (GPU, training process): runs a batch of recipes on clean clips
  with PyTorch: sensor and render noise, fireflies, salt and pepper, hot pixels, the
  display encoding, grain, film grain, blotches, aliasing, banding, and the noise map.
- Compression is the one step the GPU cannot do (real encoders): samples whose recipe
  ends with it are handed, encoded, to a pool of CPU processes running
  degrade.compress, and come back to finish (decode, noise map).

Real noisy inputs (renders, Noisebase) arrive with their noisy clip; their recipe only
holds the extra steps (compression, salt and pepper).

Tensors are (T, 3, H, W) per sample, float32, linear light unless noted.
"""
import math

import numpy as np
import torch
import torch.nn.functional as F

import degrade as D
from recipes import describe, needs_flow, recipe  # noqa: F401 (re-exported)

LUMA = torch.tensor([0.2126, 0.7152, 0.0722])
_M1, _M2, _C1, _C2, _C3 = D._M1, D._M2, D._C1, D._C2, D._C3
WHITE = D.WHITE_NITS / 10000.0


# --- GPU helpers -------------------------------------------------------------------


def srgb_oetf(x):
    x = x.clamp_min(0)
    return torch.where(x <= 0.0031308, 12.92 * x, 1.055 * x.pow(1 / 2.4) - 0.055)


def srgb_eotf(v):
    v = v.clamp_min(0)
    return torch.where(v <= 0.04045, v / 12.92, ((v + 0.055) / 1.055).pow(2.4))


def pq_oetf(x):
    y = (x * WHITE).clamp(0, 1).pow(_M1)
    return ((_C1 + _C2 * y) / (1 + _C3 * y)).pow(_M2)


def pq_eotf(v):
    p = v.clamp(0, 1).pow(1 / _M2)
    return ((p - _C1).clamp_min(0) / (_C2 - _C3 * p)).pow(1 / _M1) / WHITE


_KERNELS = {}


def blur(x, s):
    """Gaussian blur over the last two dims (reflect padding), any leading dims."""
    if s <= 0.05:
        return x
    rad = max(1, int(math.ceil(3 * s)))
    key = (round(s, 3), rad, x.device)  # rad too: s near a step of ceil(3s) rounds onto a kernel of another size
    k = _KERNELS.get(key)
    if k is None:
        t = torch.arange(-rad, rad + 1, device=x.device, dtype=torch.float32)
        k = torch.exp(-t * t / (2 * s * s))
        k = k / k.sum()
        if len(_KERNELS) > 512:
            _KERNELS.clear()
        _KERNELS[key] = k
    sh = x.shape
    y = x.reshape(-1, 1, sh[-2], sh[-1])
    H, W = sh[-2], sh[-1]
    ry, rx = min(rad, H - 1), min(rad, W - 1)
    y = F.pad(y, (rx, rx, 0, 0), mode="reflect")
    y = F.conv2d(y, k[rad - rx : rad + rx + 1].view(1, 1, 1, -1))
    y = F.pad(y, (0, 0, ry, ry), mode="reflect")
    y = F.conv2d(y, k[rad - ry : rad + ry + 1].view(1, 1, -1, 1))
    return y.reshape(sh)


def shaped_noise(g, shape, size, laplace, device):
    """Unit-variance noise with grain size `size` px, shape (..., H, W)."""
    if laplace:
        n = torch.empty(shape, device=device).exponential_(generator=g) * (torch.randint(0, 2, shape, device=device, generator=g) * 2 - 1) / math.sqrt(2)
    else:
        n = torch.randn(shape, device=device, generator=g)
    if size > 0.6:
        n = blur(n, size * 0.5)
        n = n / (n.flatten(-2).std(-1)[..., None, None] + 1e-8)
    return n


def colour_mix(g, T, H, W, size, colour, laplace, device):
    """(T, 3, H, W) noise: `colour` 0 = monochrome, 1 = independent per channel."""
    m = shaped_noise(g, (T, 1, H, W), size, laplace, device)
    if colour <= 0:
        return m.expand(T, 3, H, W).clone()
    e = shaped_noise(g, (T, 3, H, W), size, laplace, device)
    return math.sqrt(1 - colour) * m + math.sqrt(colour) * e


def luma(x):
    return (x * LUMA.to(x.device).view(1, 3, 1, 1)).sum(1, keepdim=True)


def resize(x, size, mode):
    if mode == "nearest":
        return F.interpolate(x, size=size, mode="nearest")
    return F.interpolate(x, size=size, mode=mode, align_corners=False)


def warp_half(img, flow):
    """img (3, H, W) moved by half of flow (2, H, W) (picture half way)."""
    _, H, W = img.shape
    gy, gx = torch.meshgrid(torch.arange(H, device=img.device, dtype=torch.float32), torch.arange(W, device=img.device, dtype=torch.float32), indexing="ij")
    x = gx - flow[0] / 2
    y = gy - flow[1] / 2
    grid = torch.stack([2 * x / (W - 1) - 1, 2 * y / (H - 1) - 1], -1)[None]
    return F.grid_sample(img[None], grid, mode="bilinear", padding_mode="reflection", align_corners=True)[0]


def fill_field(f, parity, smooth):
    """Keep one field's lines (rows of parity), rebuild the others (avg or repeat)."""
    H = f.shape[-2]
    out = f.clone()
    rows = torch.arange(1 - parity, H, 2, device=f.device)
    up = (rows - 1).clamp(0, H - 1)
    up = torch.where(rows - 1 < 0, rows + 1, up).clamp(0, H - 1)
    dn = torch.where(rows + 1 >= H, rows - 1, rows + 1).clamp(0, H - 1)
    out[..., rows, :] = 0.5 * (f[..., up, :] + f[..., dn, :]) if smooth else f[..., up, :]
    return out


# --- the steps (GPU) ---------------------------------------------------------------


# A colour-correction matrix's typical shape (rows sum to 0, so white stays white):
# camera RGB -> output RGB = I + ccm * CCM_K.
CCM_K = ((0.6, -0.4, -0.2), (-0.2, 0.5, -0.3), (0.0, -0.5, 0.5))


def sensor(x, p, g):
    """Camera noise through a camera's processing: shot + read noise per raw channel
    (signal in raw units: the picture divided by the white balance gains), times the
    gains, mixed by a colour-correction matrix, softened by demosaicing."""
    T, _, H, W = x.shape
    wb = torch.tensor([p["wb_r"], 1.0, p["wb_b"]], device=x.device).view(1, 3, 1, 1)
    raw = x.clamp_min(0) / wb
    sd = (raw * p["gain"] + p["read"] ** 2).sqrt()
    fresh = torch.randn((T, 3, H, W), device=x.device, generator=g)
    k = p.get("persist", 0.0)
    if k > 0:  # temporal NR leaves part of the last frame's noise
        for t in range(1, T):
            fresh[t] = k * fresh[t - 1] + math.sqrt(1 - k * k) * fresh[t]
    n = fresh * sd * wb
    ccm = torch.eye(3, device=x.device) + p["ccm"] * torch.tensor(p.get("ccm_k", CCM_K), device=x.device, dtype=torch.float32)
    n = torch.einsum("ij,tjhw->tihw", ccm, n)
    y = luma(n)
    n = y + p.get("sat", 1.0) * (n - y)  # the picture style's saturation
    if p["spread"] > 0.1:
        n = blur(n, p["spread"]) * (1 + p["spread"])
    if p["nr"]:
        y = luma(n)
        n = y + blur(n - y, p["nr_s"])
    return x + n


def render(x, p, g):
    T, _, H, W = x.shape
    n, miss, s, colour = p["n"], p["miss"], p["s"], p["colour"]
    if n <= 8:
        acc = torch.zeros_like(x)
        for _ in range(n):
            hit = (torch.rand((T, 1, H, W), device=x.device, generator=g) >= miss).float()
            gg = colour_mix(g, T, H, W, 0.0, colour, False, x.device)
            acc += hit * torch.exp(s * gg - s * s / 2) / (1 - miss)
        w = acc / n
    else:
        m2 = math.exp(s * s) / (1 - miss)
        w = (1 + math.sqrt((m2 - 1) / n) * colour_mix(g, T, H, W, 0.0, colour, False, x.device)).clamp_min(0)
    return x * w


def fireflies(x, p, g):
    T, _, H, W = x.shape
    base = max(float(x.mean()), 0.05)
    m = (torch.rand((T, 1, H, W), device=x.device, generator=g) < p["rate"]).float()
    v = base * torch.exp(torch.empty((T, 1, H, W), device=x.device).uniform_(math.log(10), math.log(1000), generator=g))
    tint = torch.empty((T, 3, H, W), device=x.device).uniform_(0.5, 1.5, generator=g)
    return x + m * v * tint


def salt_pepper(x, p, g):
    T, _, H, W = x.shape
    m = torch.rand((T, 1, H, W), device=x.device, generator=g) < p["rate"]
    if p["cluster"]:
        extra = m & (torch.rand((T, 1, H, W), device=x.device, generator=g) < 0.5)
        m = m | torch.roll(extra, 1, int(torch.randint(2, 4, (1,), generator=g, device=x.device)))
    salt = m & (torch.rand((T, 1, H, W), device=x.device, generator=g) < p["salt_frac"])
    pep = m & ~salt
    if p["coloured"]:
        ch = torch.randint(0, 3, (T, 1, H, W), device=x.device, generator=g)
        sel = torch.arange(3, device=x.device).view(1, 3, 1, 1) == ch
        x = torch.where(salt & sel, torch.full_like(x, p["hi"]), x)
        return torch.where(pep & sel, torch.zeros_like(x), x)
    x = torch.where(salt.expand_as(x), torch.full_like(x, p["hi"]), x)
    return torch.where(pep.expand_as(x), torch.zeros_like(x), x)


def hot_pixels(x, p, g):
    T, _, H, W = x.shape
    m = (torch.rand((1, 1, H, W), device=x.device, generator=g) < p["rate"]).float()
    v = torch.empty((1, 3, H, W), device=x.device).uniform_(0.5, 1.0, generator=g) * p["hi"]
    return torch.maximum(x, m * v)


def grain(v, p, g):
    T, _, H, W = v.shape
    n = colour_mix(g, T, H, W, p["size"], p["colour"], p["laplace"], v.device)
    if p["sparse"]:
        n = n * (n[:, :1].abs() > p["sparse"]) * 2.0
    if p["clump"]:
        u = blur(torch.rand((T, 1, H, W), device=v.device, generator=g), p["size"] * 6)
        n = n * (0.4 + 1.2 * (u * 3 - 1).clamp(0, 1))
    return v + p["amp"] * n


# Film grain: the Boolean model (Newson, Delon, Galerne, "A Stochastic Film Grain
# Model for Resolution-Independent Rendering", 2017). Grains are opaque discs (far
# smaller than a pixel) dropped
# by a Poisson process whose density follows the picture, so a pixel's expected
# coverage equals its value: lambda = -ln(1 - u) / (disc area). Rendered at SS times
# the resolution and averaged down (the scan), then softened by the scan's blur. The
# grain is added as the difference to the equally blurred picture, so the picture
# keeps its sharpness. Colour film: one layer per dye (R, G, B), blue's grains largest;
# black and white: one layer on luma. Grain is strongest in the mid-tones and fades
# at black and white by itself; new grains every frame.
SS = 4
RADIUS_SCALE = (1.0, 0.9, 1.3)  # R, G, B dye layers
_DISCS = {}


def _disc(radius, device):
    """Disc kernel of `radius` (super-sampled pixels) and its area."""
    key = (round(radius, 3), max(1, int(math.ceil(radius))), device)  # radius bucket and size together
    if key not in _DISCS:
        R = max(1, int(math.ceil(radius)))
        t = torch.arange(-R, R + 1, device=device, dtype=torch.float32)
        k = ((t[:, None] ** 2 + t[None, :] ** 2) <= max(radius, 0.5) ** 2).float()
        _DISCS[key] = (k[None, None], float(k.sum()), R)
    return _DISCS[key]


def boolean_grain(u, radius, spread, g):
    """One grain layer: u (T, 1, H, W) in [0, 1] -> the covered fraction per pixel."""
    T, _, H, W = u.shape
    up = F.interpolate(u, scale_factor=SS, mode="bilinear", align_corners=False).clamp(0, 0.995)
    radii = [radius * SS] if spread <= 0.02 else [radius * SS * (1 - spread), radius * SS * (1 + spread)]
    hits = None
    for rr in radii:
        k, area, R = _disc(rr, u.device)
        lam = -torch.log1p(-up) / (area * len(radii))
        pts = torch.poisson(lam, generator=g)
        c = F.conv2d(pts, k, padding=R)
        hits = c if hits is None else hits + c
    return F.avg_pool2d((hits > 0).float(), SS)


def film_grain(v, p, g):
    """Grains are much smaller than a pixel (many per pixel, as in a scan): the visible
    grain is their uneven clumping, sized by the scan blur. Colour film mixes each dye
    layer's own grain with a shared (luma) grain; `chroma` sets how colourful.
    `negative` (most motion-picture film): the grains respond to light, so the model
    runs in linear light, and on screen the grain is strongest in the shadows, fading
    in the highlights. Otherwise (prints, reversal) on the display values: strongest
    in the mid-tones."""
    neg = p.get("negative", False)
    x = v.clamp(0, 1)
    if neg:
        x = srgb_eotf(x)
    y = luma(x)
    gy = blur(boolean_grain(y, p["r"], p["spread"], g), p["blur"]) - blur(y, p["blur"])
    if p["mono"]:
        out = x + p["strength"] * gy
    else:
        out = torch.empty_like(x)
        c = p["chroma"]
        for ch in range(3):
            u = x[:, ch : ch + 1]
            gc = blur(boolean_grain(u, p["r"] * RADIUS_SCALE[ch], p["spread"], g), p["blur"]) - blur(u, p["blur"])
            out[:, ch : ch + 1] = u + p["strength"] * ((1 - c) * gy + c * gc)
    if neg:
        out = srgb_oetf(out)
    # Values outside 0..1 (HDR, overshoots) keep their own offset from the clamp.
    return out + (v - v.clamp(0, 1))


def blotches(v, p, g):
    T, _, H, W = v.shape
    out = torch.empty_like(v)
    prev = None
    keep = p["keep"]
    for t in range(T):
        n = colour_mix(g, 1, H, W, p["size"] * 2, 1.0, False, v.device)[0]
        n = n - (1 - p["luma_part"]) * luma(n[None])[0]
        n = n / (n.std() + 1e-8)
        prev = n if prev is None else keep * prev + math.sqrt(1 - keep * keep) * n
        y = luma(v[t : t + 1])[0].clamp(0, 1)
        out[t] = v[t] + p["amp"] * (1 - y).pow(p["shadow_pow"]) * prev
    return out


def aliasing(v, p, g, flow):
    T, _, H, W = v.shape
    mode = p["mode"]
    if mode == "decimate":
        f = p["f"]
        small = v[:, :, ::f, ::f]
        return resize(small, (H, W), "bicubic" if p["cubic"] else "bilinear")
    if mode == "nearest":
        s = p["s"]
        small = resize(v, (max(8, int(H * s)), max(8, int(W * s))), "nearest")
        return resize(small, (H, W), "nearest")
    first = p["first"]
    if mode == "weave":
        out = torch.empty_like(v)
        for t in range(T):
            nb = v[t + 1] if t + 1 < T else v[t - 1]
            fl = flow[t]  # flow of frame t towards that neighbour (CPU, DIS)
            mid = warp_half(v[t], fl)
            motion = fl.pow(2).sum(0).sqrt() / 2
            bobbed = fill_field(v[t], first, p["smooth"])
            m = blur((motion <= p["lim"]).float(), 1.0)[None]
            woven = v[t].clone()
            woven[:, 1 - first :: 2] = mid[:, 1 - first :: 2]
            out[t] = m * woven + (1 - m) * bobbed
        return out
    if mode == "bob":
        return torch.stack([fill_field(v[t], (t + first) % 2, p["smooth"]) for t in range(T)])
    return fill_field(v, first, False)


def banding(v, p, g):
    q = 2 ** p["bits"] - 1
    return torch.round(v.clamp(0, 1) * q) / q


def noise_map(clean, noisy):
    """Local RMS of the damage in PQ (as degrade.py's), (T, 1, H, W)."""
    d = pq_oetf(noisy) - pq_oetf(clean)
    return blur(d.pow(2).mean(1, keepdim=True), 4.0).sqrt()


# --- running batches ---------------------------------------------------------------


def _compress_job(args):
    """CPU pool: degrade.compress on one clip (T, H, W, 3) uint8 display values; with
    `twice`, a camera codec first, then an editing one (as footage often goes)."""
    u8, p = args
    r = np.random.default_rng(p["seed"])
    v = u8.astype(np.float32) / 255.0
    if p.get("twice"):
        v, l1 = D.compress(r, v, str(r.choice(D.CAMERA_CODECS)))
        out, l2 = D.compress(r, v, str(r.choice(D.EDIT_CODECS)))
        return out.astype(np.float16), f"{l1} + {l2}"
    out, label = D.compress(r, v)
    return out.astype(np.float16), label


class GpuDegrader:
    def __init__(self, device="cuda", compress_workers=8, seed=0):
        self.device = device
        self.g = torch.Generator(device=device)
        self.g.manual_seed(seed)
        self.pool = None
        self.compress_workers = compress_workers

    def _pool(self):
        if self.pool is None:
            import multiprocessing as mp
            from concurrent.futures import ProcessPoolExecutor

            # Workers are replaced every 200 clips, so a slow leak in an encoder library
            # cannot build up over a long run.
            self.pool = ProcessPoolExecutor(self.compress_workers, mp_context=mp.get_context("spawn"), max_tasks_per_child=200)
        return self.pool

    def reset_pool(self):
        """After an encoder process died (the pool is then unusable): a fresh pool."""
        if self.pool is not None:
            try:
                self.pool.shutdown(wait=False, cancel_futures=True)
            except Exception:
                pass
            self.pool = None

    def close(self):
        if self.pool is not None:
            self.pool.shutdown(wait=False, cancel_futures=True)
            self.pool = None

    @torch.no_grad()
    def start(self, clean, noisy_given, flows, recipes):
        """Run every step up to compression. clean / noisy_given: (B, T, 3, H, W)
        linear (noisy_given used where a recipe says real_noisy); flows: (B, T, 2, H,
        W) or None. Returns a job to `finish` (compression submitted to the CPU pool)."""
        B = clean.shape[0]
        noisy = torch.empty_like(clean)
        futures = {}
        for i, rec in enumerate(recipes):
            hdr = rec["hdr"]
            enc, dec = (pq_oetf, pq_eotf) if hdr else (srgb_oetf, srgb_eotf)
            if rec.get("decode") == "srgb":  # camera clips arrive as stored (sRGB values)
                clean[i] = srgb_eotf(clean[i]) * rec["gain"]
            x = noisy_given[i] if rec.get("real_noisy") else clean[i]
            v = None
            for name, p in rec["steps"]:
                if name == "clean":
                    continue
                if name == "encode":
                    v = enc(x)
                    continue
                if name == "compress":
                    vv = enc(x) if v is None else v
                    u8 = (vv.clamp(0, 1) * 255 + 0.5).to(torch.uint8).permute(0, 2, 3, 1).cpu().numpy()
                    futures[i] = self._pool().submit(_compress_job, (u8, p))
                    v = vv
                    break
                if v is None:  # linear-light steps
                    x = {"sensor": sensor, "render": render, "fireflies": fireflies, "salt_pepper": salt_pepper, "hot_pixels": hot_pixels}[name](x, p, self.g)
                else:
                    if name == "aliasing":
                        v = aliasing(v, p, self.g, flows[i] if flows is not None else None)
                    else:
                        v = {"grain": grain, "film_grain": film_grain, "blotches": blotches, "banding": banding}[name](v, p, self.g)
            if v is not None:
                noisy[i] = dec(v)
            else:
                noisy[i] = x
        return {"clean": clean, "noisy": noisy, "futures": futures, "recipes": recipes}

    def ready(self, job):
        return all(f.done() for f in job["futures"].values())

    @torch.no_grad()
    def finish(self, job):
        """Bring the compressed clips back; noise maps. Returns clean, noisy, sigma, labels."""
        noisy = job["noisy"]
        labels = [describe(r) for r in job["recipes"]]
        for i, f in job["futures"].items():
            out, label = f.result()
            hdr = job["recipes"][i]["hdr"]
            dec = pq_eotf if hdr else srgb_eotf
            v = torch.from_numpy(out).to(self.device).float().permute(0, 3, 1, 2)
            noisy[i] = dec(v)
            labels[i] = [l if l != "compress" else " + ".join(x.split(" q ")[0] for x in label.split(" + ")) for l in labels[i]]
        sigma = torch.stack([noise_map(c, n) for c, n in zip(job["clean"], noisy)])
        return job["clean"], noisy, sigma, labels


class Feeder:
    """Finished batches for training, prepared in a background thread: loader batch ->
    GPU damage -> (compression on the CPU pool, a few batches in flight) -> noise maps.

        f = Feeder(loader, degrader, "cuda"); b = f.next()  # clean, noisy, sigma, labels
    """

    def __init__(self, loader, degrader, device, depth=4):
        import queue
        import threading

        self.loader, self.degr, self.device, self.depth = loader, degrader, device, depth
        self.q = queue.Queue(maxsize=depth)
        self.err = None
        self.stopping = False
        self.thread = threading.Thread(target=self._run, name="feeder", daemon=True)
        self.thread.start()

    def _run(self):
        from collections import deque

        pending = deque()
        failures = 0  # in a row: a bad batch is skipped, many in a row stop training
        try:
            while not self.stopping:
                if len(pending) < self.depth:
                    b = self.loader.next(self.device)
                    try:
                        job = self.degr.start(b["clean"], b["noisy"], b["flow"], b["recipe"])
                    except Exception as e:
                        failures += 1
                        self._skip(e, b["recipe"], failures)
                        continue
                    job["kind"], job["origin"] = b["kind"], b["origin"]
                    pending.append(job)
                while pending and (self.degr.ready(pending[0]) or len(pending) >= self.depth):
                    job = pending.popleft()
                    try:
                        clean, noisy, sigma, labels = self.degr.finish(job)
                    except Exception as e:
                        failures += 1
                        self._skip(e, job["recipes"], failures)
                        continue
                    failures = 0
                    item = {"clean": clean, "noisy": noisy, "sigma": sigma, "labels": labels, "kind": job["kind"], "origin": job["origin"]}
                    while not self.stopping:
                        try:
                            self.q.put(item, timeout=1)
                            break
                        except Exception:
                            pass
        except Exception as e:  # handed to the training thread
            import traceback

            traceback.print_exc()
            self.err = e

    def _skip(self, e, recipes, failures):
        import traceback
        from concurrent.futures.process import BrokenProcessPool

        print(f"feeder: batch skipped ({type(e).__name__}: {e}); recipes: {recipes}", flush=True)
        traceback.print_exc()
        if isinstance(e, BrokenProcessPool):
            # An encoder crashed and took the pool down: start a new one. The other
            # batches waiting on the old pool fail too and are skipped as they come.
            print("feeder: encoder pool broken, starting a new one", flush=True)
            self.degr.reset_pool()
        if failures >= 20:
            raise RuntimeError(f"20 batches in a row failed, last: {e}")

    def next(self):
        import queue

        while True:
            if self.err is not None:
                raise RuntimeError(f"data feeder failed: {self.err}")
            try:
                return self.q.get(timeout=2)
            except queue.Empty:
                pass

    def close(self):
        self.stopping = True
        self.thread.join(timeout=10)


@torch.no_grad()
def prepare_val(degrader, batches, device, seed=12345):
    """Damage the fixed validation clips once (fixed seed), kept on the CPU (float16)."""
    g_old = degrader.g
    degrader.g = torch.Generator(device=device)
    degrader.g.manual_seed(seed)
    out = []
    try:
        for b in batches:
            t = lambda k: torch.from_numpy(b[k]).to(device).float()
            job = degrader.start(t("clean"), t("noisy"), t("flow"), b["recipe"])
            clean, noisy, sigma, labels = degrader.finish(job)
            out.append({"clean": clean.half().cpu(), "noisy": noisy.clamp(-65000, 65000).half().cpu(), "sigma": sigma.half().cpu(), "kind": b["kind"]})
    finally:
        degrader.g = g_old
    return out
