"""Check the GPU damage (gpu_degrade.py) against the CPU reference (degrade.py): the
same clean clip and the same step settings through both; prints the strength of the
damage (RMS of noisy - clean in the display encoding) and writes a sheet of
clean | CPU | GPU per step to <data>/check/gpu_vs_cpu.png.

  python gpu_check.py [--data D:/denoise_data]
"""
import os, sys

import cv2
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import degrade as D
import dataset as ds
import gpu_degrade as G
import recipes as R

STEPS = ["sensor", "render", "fireflies", "salt_pepper", "grain", "film_grain", "blotches", "aliasing:decimate", "aliasing:nearest", "aliasing:bob", "aliasing:weave", "banding"]


def cpu_apply(name, p, lin, r, hdr):
    """The CPU reference for one step with these settings (mirrors degrade.py)."""
    enc, dec = (D.pq_oetf, D.pq_eotf) if hdr else (D.srgb_oetf, D.srgb_eotf)
    rr = _Fixed(r, p)
    if name in ("sensor", "render", "fireflies", "salt_pepper"):
        f = {"sensor": D.sensor_noise, "render": D.render_noise, "fireflies": D.fireflies, "salt_pepper": lambda a, b: D.salt_pepper(a, b, hdr)}[name]
        out, _ = f(rr, lin)
        return enc(out)
    v = enc(lin)
    f = {"grain": D.grain, "film_grain": D.film_grain, "blotches": D.blotches, "aliasing": D.aliasing, "banding": D.banding}[name]
    out, _ = f(rr, v)
    return out


class _Fixed:
    """A NumPy Generator whose first draws return the recipe's settings, in the order
    degrade.py draws them, then real random numbers (for the noise itself)."""

    def __init__(self, r, p):
        self.r, self.vals = r, list(_order(p))

    def _next(self, *a, **k):
        if self.vals:
            return self.vals.pop(0)
        return None

    def uniform(self, a=0.0, b=1.0, size=None):
        if size is None and self.vals:
            return self.vals.pop(0)
        return self.r.uniform(a, b, size)

    def random(self, size=None):
        if size is None and self.vals:
            return self.vals.pop(0)
        return self.r.random(size)

    def integers(self, *a, **k):
        if "size" not in k and len(a) <= 2 and self.vals:
            return self.vals.pop(0)
        return self.r.integers(*a, **k)

    def choice(self, a, *x, **k):
        if self.vals and not x and not k:
            return self.vals.pop(0)
        return self.r.choice(a, *x, **k)

    def __getattr__(self, n):
        return getattr(self.r, n)


def _order(p):
    """Settings in the order degrade.py's functions draw them (log-uniforms as the
    uniform of their log)."""
    import math

    L = math.log
    k = p.get("_name")
    if k == "sensor":
        base = [[0.0, -0.4, -0.2], [-0.2, 0.0, -0.3], [0.0, -0.5, 0.0]]
        jit = [p["ccm_k"][i][j] - base[i][j] for i in range(3) for j in range(3) if i != j]
        return [L(p["gain"]), L(p["read"])] + jit + [p["spread"], 0.0 if p["nr"] else 1.0, p["nr_s"], p["wb_r"], p["wb_b"], p["ccm"], p["sat"], p["persist"]]
    if k == "render":
        return [L(p["n"]), p["miss"], p["s"], p["colour"]]
    if k == "fireflies":
        return [L(p["rate"])]
    if k == "salt_pepper":
        return [L(p["rate"]), p["salt_frac"], 0.0 if p["coloured"] else 1.0, 0.0 if p["cluster"] else 1.0] + ([L(p["hi"])] if p["hi"] != 1.0 else [])
    if k == "grain":
        # colour = r.choice([0.0, r.uniform(...)]): the uniform is drawn before the choice.
        return [L(p["size"]), L(p["amp"]), p["colour"] or 0.5, p["colour"], 0.0 if p["laplace"] else 1.0, 0.0 if p["sparse"] else 1.0, 0.0 if p["clump"] else 1.0] + ([p["sparse"]] if p["sparse"] else [])
    if k == "film_grain":
        return [L(p["r"]), p["spread"], 0.0 if p["mono"] else 1.0, p["blur"], p["strength"], p["chroma"], 0.0 if p.get("negative") else 1.0]
    if k == "blotches":
        return [L(p["size"]), L(p["amp"]), p["keep"], p["luma_part"], p["shadow_pow"]]
    if k == "aliasing":
        m = p["mode"]
        if m == "decimate":
            return [m, p["f"], 0.9 if p["cubic"] else 0.1]
        if m == "nearest":
            return [m, p["s"]]
        if m == "weave":
            return [m, p["first"], p["lim"], 0.1 if p["smooth"] else 0.9]
        if m == "bob":
            return [m, p["first"], 0.1 if p["smooth"] else 0.9]
        return [m, p["first"]]
    if k == "banding":
        return [p["bits"]]
    return []


def main():
    data = sys.argv[sys.argv.index("--data") + 1] if "--data" in sys.argv else "D:/denoise_data"
    r = np.random.default_rng(3)
    cam = ds.Camera(ds.DataConfig(data=data), "train")
    clip_enc, _, _, origin = cam.sample(r, 8, 256)
    lin = D.srgb_eotf(clip_enc) * 0.8
    degr = G.GpuDegrader("cuda", compress_workers=1, seed=1)
    tiles, rows = [], []
    for spec in STEPS:
        name, _, mode = spec.partition(":")
        # Settings: a real recipe draw for this step (strong enough to see).
        rr = np.random.default_rng(abs(hash(spec)) % (1 << 31))
        if name == "sensor":
            p = {"gain": 4e-3, "read": 2e-3, "spread": 0.7, "nr": False, "nr_s": 2.0, "wb_r": 1.9, "wb_b": 2.2, "ccm": 0.8, "sat": 1.5, "persist": 0.2,
                 "ccm_k": [[0.6, -0.4, -0.2], [-0.2, 0.5, -0.3], [0.0, -0.5, 0.5]]}
        elif name == "render":
            p = {"n": 4, "miss": 0.5, "s": 0.8, "colour": 0.5}
        elif name == "fireflies":
            p = {"rate": 1e-3}
        elif name == "salt_pepper":
            p = {"rate": 3e-3, "salt_frac": 0.5, "coloured": False, "cluster": False, "hi": 1.0}
        elif name == "aliasing":
            p = R._display(rr, "aliasing")
            while p["mode"] != mode:
                p = R._display(rr, "aliasing")
        elif name == "banding":
            p = {"bits": 5}
        else:
            p = R._display(rr, name)
        hdr = False
        flow = None
        if name == "aliasing" and mode == "weave":
            flow = ds.flows(lin, True).transpose(0, 3, 1, 2)
        # CPU reference.
        cpu_v = cpu_apply(name, dict(p, _name=name), lin, np.random.default_rng(7), hdr)
        # GPU.
        enc_step = [] if name in ("sensor", "render", "fireflies", "salt_pepper") else [("encode", {})]
        rec = {"kind": "camera", "hdr": hdr, "steps": ([(name, p)] + [("encode", {})]) if not enc_step else (enc_step + [(name, p)])}
        t = lambda a: torch.from_numpy(np.ascontiguousarray(a.transpose(0, 3, 1, 2))).cuda().float()[None]
        fl = torch.from_numpy(flow).cuda().float()[None] if flow is not None else None
        job = degr.start(t(lin), t(lin), fl, [rec])
        _, noisy, _, _ = degr.finish(job)
        gpu_v = G.srgb_oetf(noisy[0]).permute(0, 2, 3, 1).cpu().numpy()
        base = D.srgb_oetf(lin)
        rc = float(np.sqrt(np.mean((cpu_v - base) ** 2)))
        rg = float(np.sqrt(np.mean((gpu_v - base) ** 2)))
        rows.append((spec, rc, rg))
        z = lambda a: cv2.resize((np.clip(a[0, 64:192, 64:192], 0, 1) * 255).astype(np.uint8), (256, 256), interpolation=cv2.INTER_NEAREST)
        tile = np.concatenate([z(base), z(cpu_v), z(gpu_v)], 1)[..., ::-1].copy()
        cv2.putText(tile, f"{spec}: CPU {rc:.4f} GPU {rg:.4f}", (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)
        tiles.append(tile)
    degr.close()
    sheet = np.concatenate([np.concatenate(tiles[i : i + 2], 1) if i + 1 < len(tiles) else np.concatenate([tiles[i], np.zeros_like(tiles[i])], 1) for i in range(0, len(tiles), 2)], 0)
    path = os.path.join(data, "check", "gpu_vs_cpu.png")
    cv2.imwrite(path, sheet)
    print(f"clip {origin}")
    print(f"{'step':18s} {'CPU rms':>9s} {'GPU rms':>9s}  ratio")
    for spec, rc, rg in rows:
        print(f"{spec:18s} {rc:9.4f} {rg:9.4f}  {rg / max(rc, 1e-9):.2f}")
    print("sheet:", path)


if __name__ == "__main__":
    main()
