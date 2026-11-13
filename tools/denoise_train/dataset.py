"""Training samples for the stable denoiser (PLAN.md §3-4).

    s = Sampler(DataConfig(data="D:/denoise_data"), split="train").one(rng)
    s["clean"], s["noisy"], s["sigma"]  # (T, 3|1, C, C) float32, linear light

NumPy only (see Sampler); loader.py runs it in worker processes.

Each sample is a short clip (T frames, C x C crop) in linear light, from one of:

- camera   REDS and Netflix clips, from the crops cached by data/camera_bank.py, that
           pass the noise gate (data/patchnoise.py); random scale, flips, exposure (at
           most +0.5 stop, §3);
           damage made by degrade.py (kind "camera").
- render   our Blender pairs (bank built by data/render_bank.py): GT against a real
           noisy variant (raw noise, or Blender's denoiser output with its flicker), or
           against synthetic render damage; exposure +-1.5 stops; sometimes compressed.
- noisebase  Noisebase sequences: the mean of the first k samples per pixel (k from 1
           to what was fetched, 16)
           against the reference; exposure +-1 stop.

`sigma` is the local RMS of the damage in the model's encoding (PQ, degrade.pq_oetf),
as degrade.py computes it. Validation holds out REDS val, a Netflix sequence, one
render scene and the last Noisebase sequences, and draws a fixed set (same seed).
"""
import glob, json, os, random
from dataclasses import dataclass, field

import cv2
import numpy as np

import degrade as D
from recipes import needs_flow, recipe
from data import patchnoise as pn

cv2.setNumThreads(1)
try:  # Blosc (camera cache) would start a thread per CPU core in every worker.
    from numcodecs import blosc as _blosc

    _blosc.use_threads = False
    _blosc.set_nthreads(1)
except ImportError:
    pass

VAL_RENDER = "classroom"
VAL_NETFLIX = "nf_ElFuente-Crosswalk"


@dataclass
class DataConfig:
    data: str = "D:/denoise_data"
    frames: int = 8
    crop: int = 256
    # Source mix.
    weights: dict = field(default_factory=lambda: {"camera": 0.55, "render": 0.3, "noisebase": 0.15})
    # Netflix clips drawn this many times more often than a REDS clip.
    netflix_boost: float = 4.0
    seed: int = 0
    val_size: int = 256
    # Renders: share of clips with synthetic damage, then (cumulative) with Blender's
    # denoised pass as the input; the rest are real noisy passes. render_low: share of
    # the real ones (noisy or denoised) taken from the 16 and 64 sample renders only.
    render_mix: tuple = (0.25, 0.45)
    render_low: float = 0.0


class Camera:
    """Crops cached by data/camera_bank.py (512x512, every frame of the clip)."""

    def __init__(self, cfg, split):
        bank = os.path.join(cfg.data, "bank", "camera")
        index = json.load(open(os.path.join(bank, "index.json")))
        val = lambda e: e["split"] == "val" or e["clip"].startswith(VAL_NETFLIX)
        for i, e in enumerate(index):
            e["i"] = i
        self.crops = [e for e in index if val(e) == (split == "val")]
        # Noise-gate blocks of every crop (data/crop_maps.py), loaded once.
        self.cmaps = np.load(os.path.join(bank, "crop_maps.npy"), mmap_mode="r")
        w = np.array([cfg.netflix_boost if e["source"] == "netflix" else 1.0 for e in self.crops])
        self.p = w / w.sum()
        self.gate = json.load(open(os.path.join(cfg.data, "noise_maps", "gate.json")))
        self.data, self.bin = cfg.data, os.path.join(bank, "crops.bin")
        self.f = None

    def __bool__(self):
        return bool(self.crops)

    def _frame(self, e, i):
        from numcodecs import Blosc

        if self.f is None:
            self.f = open(self.bin, "rb")
        off, n = e["frames"][i]
        self.f.seek(off)
        a = np.frombuffer(Blosc().decode(self.f.read(n)), e["dtype"]).reshape(e["size"], e["size"], 3)
        return a.astype(np.float32) / (65535.0 if e["dtype"] == "uint16" else 255.0)

    def sample(self, r, T, C):
        for _ in range(20):
            e = self.crops[r.choice(len(self.crops), p=self.p)]
            if len(e["frames"]) < T:
                continue
            s = float(r.choice([1.0, 1.0, 1.25, 1.5, 2.0]))  # read s*C pixels, scale to C
            size = int(C * s)
            if size > e["size"]:
                s, size = 1.0, C
            y, x = int(r.integers(0, e["size"] - size + 1)), int(r.integers(0, e["size"] - size + 1))
            cm = self.cmaps[e["i"]]
            m = {"y": cm[0], "c": cm[1], "grad": cm[2]}
            if not pn.patch_ok(m, y, x, size, self.gate):
                continue
            t0 = int(r.integers(0, len(e["frames"]) - T + 1))
            clip = np.stack([self._frame(e, i)[y : y + size, x : x + size] for i in range(t0, t0 + T)])
            if size != C:
                clip = np.stack([cv2.resize(f, (C, C), interpolation=cv2.INTER_AREA) for f in clip])
            rec = recipe(r, "camera", hdr=False)
            rec["decode"], rec["gain"] = "srgb", float(2 ** r.uniform(-1.0, 0.5))
            return clip, None, rec, f"camera:{e['clip']}"
        raise RuntimeError("camera: no crop passed the noise gate in 20 tries")


class Render:
    def __init__(self, cfg, split):
        bank = os.path.join(cfg.data, "bank", "renders")
        scenes = sorted(d for d in os.listdir(bank) if os.path.exists(os.path.join(bank, d, "meta.json"))) if os.path.isdir(bank) else []
        scenes = [s for s in scenes if (s == VAL_RENDER) == (split == "val")]
        self.scenes = []
        for s in scenes:
            meta = json.load(open(os.path.join(bank, s, "meta.json")))
            arrays = {n: np.load(os.path.join(bank, s, f"{n}.f16.npy"), mmap_mode="r") for n in meta["names"]}
            self.scenes.append((s, arrays))
        self.mix, self.low = tuple(cfg.render_mix), cfg.render_low

    def __bool__(self):
        return bool(self.scenes)

    def sample(self, r, T, C):
        name, arrays = self.scenes[r.integers(len(self.scenes))]
        gt = arrays["GT"]
        F, H, W, _ = gt.shape
        t0 = int(r.integers(0, F - T + 1))
        y, x = int(r.integers(0, H - C + 1)), int(r.integers(0, W - C + 1))
        sl = np.s_[t0 : t0 + T, y : y + C, x : x + C]
        clean = np.asarray(gt[sl], np.float32)
        gain = np.float32(2 ** r.uniform(-1.5, 1.5))
        clean = np.maximum(clean, 0) * gain
        u = r.random()
        if u < self.mix[0]:
            # Synthetic damage on the clean render.
            return clean, None, recipe(r, "render", hdr=True), f"render:{name}:synthetic"
        keys = [k for k in arrays if k.endswith(".denoised")] if u < self.mix[1] else [k for k in arrays if k.endswith(".noisy")]
        if self.low > 0 and r.random() < self.low:
            keys = [k for k in keys if k.startswith(("16_", "64_"))] or keys
        k = keys[r.integers(len(keys))]
        noisy = np.maximum(np.asarray(arrays[k][sl], np.float32), 0) * gain
        return clean, noisy, recipe(r, "render", hdr=True, real_noisy=True), f"render:{name}:{k}"


def _nb_chunk(d, name, idx):
    from numcodecs import get_codec

    meta = _nb_meta(d, name)
    raw = open(os.path.join(d, name, ".".join(map(str, idx))), "rb").read()
    buf = get_codec(meta["compressor"]).decode(raw) if meta["compressor"] else raw
    return np.frombuffer(buf, np.dtype(meta["dtype"])).reshape(meta["chunks"])


_NB_META = {}


def _nb_meta(d, name):
    k = (d, name)
    if k not in _NB_META:
        _NB_META[k] = json.load(open(os.path.join(d, name, ".zarray")))
    return _NB_META[k]


class Noisebase:
    """Noisebase: color [F, 4 (RGBE), H, W, S] uint8 in chunks (1, 3, H, W, 4);
    reference [F, 3, H, W] float32; exposure [F, 2] (RGBE range, log)."""

    def __init__(self, cfg, split):
        seqs = sorted(d for d in glob.glob(os.path.join(cfg.data, "noisebase", "scene*")) if os.path.exists(os.path.join(d, ".done")))
        n_val = max(1, len(seqs) // 16)
        self.seqs = seqs[-n_val:] if split == "val" else seqs[:-n_val]
        self.spp = {}
        # Cached means at a few sample counts (data/noisebase_bank.py): far fewer files.
        bank = os.path.join(cfg.data, "bank", "noisebase")
        meta = os.path.join(bank, "meta.json")
        self.bank = None
        if os.path.exists(meta):
            spp = json.load(open(meta))["spp"]
            arrays = {}
            for d in self.seqs:
                n = os.path.basename(d)
                rp, np_ = os.path.join(bank, f"{n}.ref.f16.npy"), os.path.join(bank, f"{n}.noisy.f16.npy")
                if os.path.exists(rp) and os.path.exists(np_):
                    arrays[d] = (rp, np_)
            if len(arrays) == len(self.seqs):
                self.bank, self.bank_spp, self.open = arrays, spp, {}

    def __bool__(self):
        return bool(self.seqs)

    def _spp(self, d):
        """Samples per pixel fetched for this sequence (4 per chunk file)."""
        if d not in self.spp:
            self.spp[d] = 4 * sum(1 for n in os.listdir(os.path.join(d, "color")) if n.startswith("0.0.0.0."))
        return self.spp[d]

    def _frame(self, d, f, k, exposure):
        # Channel chunks: 0 holds R,G,B (3 of the 4), 1 holds E (and padding).
        parts = []
        for c in range((k + 3) // 4):
            rgb = _nb_chunk(d, "color", (f, 0, 0, 0, c))[0]  # (3, H, W, 4)
            e = _nb_chunk(d, "color", (f, 1, 0, 0, c))[0][:1]  # (1, H, W, 4)
            parts.append(np.concatenate([rgb, e], 0))
        rgbe = np.concatenate(parts, -1)[..., :k]
        lo, hi = exposure[f]
        ex = np.exp((rgbe[3].astype(np.float32) + 1) / 256 * (hi - lo) + lo)
        return (rgbe[:3].astype(np.float32) / 255 * ex[None]).mean(-1).transpose(1, 2, 0)

    def sample_bank(self, r, T, C):
        d = self.seqs[r.integers(len(self.seqs))]
        if d not in self.open:
            rp, np_ = self.bank[d]
            self.open[d] = (np.load(rp, mmap_mode="r"), np.load(np_, mmap_mode="r"))
        ref, noisy_all = self.open[d]
        F, H, W, _ = ref.shape
        t0 = int(r.integers(0, F - T + 1))
        i = int(r.integers(len(self.bank_spp)))
        y, x = int(r.integers(0, H - C + 1)), int(r.integers(0, W - C + 1))
        gain = np.float32(2 ** r.uniform(-1, 1))
        clean = np.asarray(ref[t0 : t0 + T, y : y + C, x : x + C], np.float32) * gain
        noisy = np.asarray(noisy_all[t0 : t0 + T, i, y : y + C, x : x + C], np.float32) * gain
        return clean, noisy, recipe(r, "render", hdr=True, real_noisy=True), f"noisebase:{os.path.basename(d)}:{self.bank_spp[i]}spp"

    def sample(self, r, T, C):
        if self.bank:
            return self.sample_bank(r, T, C)
        d = self.seqs[r.integers(len(self.seqs))]
        F = _nb_meta(d, "reference")["shape"][0]
        H, W = _nb_meta(d, "reference")["shape"][2:]
        exposure = np.concatenate([_nb_chunk(d, "exposure", (0, 0))])
        t0 = int(r.integers(0, F - T + 1))
        k = int(round(np.exp(r.uniform(0, np.log(self._spp(d))))))
        y, x = int(r.integers(0, H - C + 1)), int(r.integers(0, W - C + 1))
        gain = np.float32(2 ** r.uniform(-1, 1))
        clean = np.stack([_nb_chunk(d, "reference", (f, 0, 0, 0))[0].transpose(1, 2, 0) for f in range(t0, t0 + T)])[:, y : y + C, x : x + C]
        noisy = np.stack([self._frame(d, f, k, exposure) for f in range(t0, t0 + T)])[:, y : y + C, x : x + C]
        clean, noisy = np.maximum(clean, 0) * gain, np.maximum(noisy, 0) * gain
        return clean, noisy, recipe(r, "render", hdr=True, real_noisy=True), f"noisebase:{os.path.basename(d)}:{k}spp"


def flows(clip, linear):
    """Optical flow of each frame towards its neighbour (the next; the previous for the
    last), (T, H, W, 2), on the display picture: for the GPU's weave step."""
    v = D.srgb_oetf(clip) if linear else clip
    g = [cv2.cvtColor((np.clip(f, 0, 1) * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY) for f in v]
    dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_ULTRAFAST)
    T = len(g)
    return np.stack([dis.calc(g[t], g[t + 1] if t + 1 < T else g[t - 1], None) for t in range(T)])


def noise_map(clean, noisy):
    """Local RMS of the damage in the model's encoding (as degrade.py's)."""
    d = D.pq_oetf(noisy) - D.pq_oetf(clean)
    return np.stack([np.sqrt(D.blur(np.mean(f * f, 2), 4.0)) for f in d])[..., None].astype(np.float32)


def augment(r, *clips):
    """The same flips / transpose / time reversal on every array (T, H, W, c)."""
    k = int(r.integers(8))
    rev = r.random() < 0.3
    out = []
    for c in clips:
        if k & 1:
            c = c[:, :, ::-1]
        if k & 2:
            c = c[:, ::-1]
        if k & 4:
            c = c.transpose(0, 2, 1, 3)
        if rev:
            c = c[::-1]
        out.append(np.ascontiguousarray(c))
    return out


class Sampler:
    """Draws training samples (NumPy only: the loader's worker processes must not
    import PyTorch, whose CUDA libraries reserve ~4 GB per process on Windows)."""

    def __init__(self, cfg: DataConfig, split="train"):
        c = cfg
        self.cfg, self.split = cfg, split
        self.sources = {k: v for k, v in (("camera", Camera(c, split)), ("render", Render(c, split)), ("noisebase", Noisebase(c, split))) if v}
        w = np.array([c.weights.get(k, 0) for k in self.sources], float)
        self.names, self.p = list(self.sources), w / w.sum()

    def one(self, r):
        """A sample: clean, noisy (T, 3, C, C) and sigma (T, 1, C, C), linear light."""
        kind = self.names[r.choice(len(self.names), p=self.p)]
        clean, noisy, rec, origin = self.sources[kind].sample(r, self.cfg.frames, self.cfg.crop)
        if noisy is None:
            (clean,) = augment(r, clean)
        else:
            clean, noisy = augment(r, clean, noisy)
        t = lambda a: np.ascontiguousarray(a.transpose(0, 3, 1, 2))
        flow = t(flows(clean, rec.get("decode") != "srgb")) if needs_flow(rec) else None
        return {"clean": t(clean), "noisy": None if noisy is None else t(noisy), "flow": flow, "recipe": rec, "kind": kind, "origin": origin}

    def safe(self, r):
        """`one`, retried: a bad file must not stop training."""
        for _ in range(10):
            try:
                return self.one(r)
            except Exception as e:
                print(f"dataset: {type(e).__name__}: {e}", flush=True)
        raise RuntimeError("dataset: 10 failures in a row")
