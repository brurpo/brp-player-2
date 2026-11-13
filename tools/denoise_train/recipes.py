"""Recipes for the GPU degradation chain (gpu_degrade.py): every random choice of a
clip's damage, drawn on the CPU by the data workers with the same choices and
distributions as degrade.degrade, without touching pixels. NumPy only (the workers
must not import PyTorch)."""
import degrade as D

lu = D.lu


def recipe(r, kind="any", hdr=False, strength=1.0, real_noisy=False):
    """Every random choice of one clip's chain, as a list of (step, settings). For a
    real noisy input (`real_noisy`): only extra compression / salt and pepper."""
    P = lambda p: r.random() < min(1.0, p * strength)
    steps = []
    if real_noisy:
        if r.random() < 0.15:
            steps.append(("salt_pepper", _sp(r, hdr)))
        if r.random() < 0.2:
            steps.append(("compress", _compress(r)))
        return {"kind": kind, "hdr": hdr, "steps": steps, "real_noisy": True}
    if kind == "any":
        kind = "render" if r.random() < 0.3 else "camera"
    if r.random() < 0.05:  # nearly clean: untouched, or only lightly compressed
        steps.append(("clean", {}))
        if r.random() < 0.5:
            steps.append(("compress", _compress(r)))
        return {"kind": kind, "hdr": hdr, "steps": steps}
    if kind == "camera":
        if P(0.75):
            steps.append(("sensor", sensor_settings(r)))
    else:
        if P(0.7):
            steps.append(("render", {"n": int(round(lu(r, 1, 64))), "miss": r.uniform(0.0, 0.85), "s": r.uniform(0.3, 1.5), "colour": r.uniform(0.1, 0.8)}))
        if P(0.5):
            steps.append(("fireflies", {"rate": lu(r, 2e-5, 2e-3)}))
    if P(0.35 if kind == "render" else 0.12):
        steps.append(("salt_pepper", _sp(r, hdr)))
    if P(0.03):
        steps.append(("hot_pixels", {"rate": lu(r, 2e-5, 5e-4), "hi": lu(r, 1, 20) if hdr else 1.0}))
    steps.append(("encode", {}))
    disp = [("grain", 0.45 if kind == "camera" else 0.15), ("film_grain", 0.12 if kind == "camera" else 0.03), ("blotches", 0.3 if kind == "camera" else 0.05), ("aliasing", 0.15)]
    for i in r.permutation(len(disp)):
        name, p = disp[i]
        if P(p):
            steps.append((name, _display(r, name)))
    if P(0.08):
        steps.append(("banding", {"bits": int(r.integers(5, 9))}))
    if P(0.65 if kind == "camera" else 0.3):
        steps.append(("compress", _compress(r)))
    if not any(s[0] not in ("encode",) for s in steps):
        steps.append(("grain", _display(r, "grain")))
    return {"kind": kind, "hdr": hdr, "steps": steps}


def sensor_settings(r):
    """A camera's noise: per raw channel (shot + read noise), scaled by the white
    balance gains (red and blue 1.2-2.8x), mixed by a colour-correction matrix (its
    negative off-diagonals spread red/blue noise into strong chroma noise along a
    magenta-teal axis, as measured on the black-curtain clip: chroma as strong as
    luma, 1.6-1.8x more along Cr), boosted by the picture style's saturation (1-2.2x on
    the noise's colour), softened by demosaicing (0.2-0.9 px; the curtain's noise is 0.8 px wide), partly carried from
    frame to frame (temporal NR, 0-30%; the curtain kept ~20%); sometimes in-camera
    chroma NR."""
    # The colour matrix's shape varies by camera: the typical one, off-diagonals
    # jittered by +-0.25, diagonals keeping each row's sum at 0 (white stays white).
    k = [[0.0, -0.4, -0.2], [-0.2, 0.0, -0.3], [0.0, -0.5, 0.0]]
    for i in range(3):
        for j in range(3):
            if i != j:
                k[i][j] += r.uniform(-0.25, 0.25)
        k[i][i] = -sum(k[i][j] for j in range(3) if j != i)
    return {"gain": lu(r, 2e-4, 1e-2), "read": lu(r, 1e-4, 8e-3), "spread": r.uniform(0.2, 0.9), "nr": bool(r.random() < 0.3), "nr_s": r.uniform(1.5, 4), "ccm_k": k,
            "wb_r": r.uniform(1.0, 3.0), "wb_b": r.uniform(1.0, 3.0), "ccm": r.uniform(0.3, 1.5), "sat": r.uniform(1.0, 2.2), "persist": r.uniform(0.0, 0.3)}


def _compress(r):
    """Compression settings: the codec is drawn in the CPU pool from `seed`; `twice`:
    a camera codec then an editing one (DNxHR, ProRes, MPEG-2), as footage often goes."""
    return {"seed": int(r.integers(1 << 31)), "twice": bool(r.random() < 0.25)}


def _sp(r, hdr):
    return {"rate": lu(r, 1e-4, 1e-2), "salt_frac": r.uniform(0.2, 0.8), "coloured": bool(r.random() < 0.3), "cluster": bool(r.random() < 0.3), "hi": lu(r, 1.0, 50.0) if hdr else 1.0}


def _display(r, name):
    if name == "grain":
        return {"size": lu(r, 0.5, 4.0), "amp": lu(r, 0.004, 0.06), "colour": float(r.choice([0.0, r.uniform(0.1, 1.0)])), "laplace": bool(r.random() < 0.4),
                "sparse": float(r.uniform(1.0, 2.0)) if r.random() < 0.25 else 0.0, "clump": bool(r.random() < 0.3)}
    if name == "film_grain":
        # Boolean-model grain (gpu_degrade.film_grain): grain radius in pixels, radius
        # spread, black and white or colour (dye layers), scan blur, strength, chroma.
        # Grains far smaller than a pixel; the scan blur sizes the visible clumps.
        return {"r": lu(r, 0.05, 0.25), "spread": r.uniform(0, 0.4), "mono": bool(r.random() < 0.3), "blur": r.uniform(0.5, 1.2), "strength": r.uniform(0.4, 1.0), "chroma": r.uniform(0.15, 0.6), "negative": bool(r.random() < 0.7)}
    if name == "blotches":
        return {"size": lu(r, 4, 32), "amp": lu(r, 0.01, 0.06), "keep": r.uniform(0, 0.95), "luma_part": r.uniform(0, 0.5), "shadow_pow": r.uniform(1, 4)}
    mode = str(r.choice(["decimate", "nearest", "weave", "weave", "bob", "linedouble"]))
    p = {"mode": mode, "first": int(r.integers(2))}
    if mode == "decimate":
        p.update(f=int(r.integers(2, 5)), cubic=bool(r.random() >= 0.5))
    elif mode == "nearest":
        p.update(s=r.uniform(0.4, 0.8))
    elif mode == "weave":
        p.update(lim=r.uniform(1.0, 4.0), smooth=bool(r.random() < 0.7))
    elif mode == "bob":
        p.update(smooth=bool(r.random() < 0.7))
    return p


def describe(rec):
    """Short labels of a recipe's steps (for the dashboard and sample sheets)."""
    out = []
    for name, p in rec["steps"]:
        if name == "encode":
            continue
        if name == "aliasing":
            out.append(f"aliasing {p['mode']}")
        elif name == "sensor":
            out.append(f"sensor {p['gain']:.0e}")
        elif name == "render":
            out.append(f"render {p['n']} spp")
        else:
            out.append(name)
    return out


def needs_flow(rec):
    return any(n == "aliasing" and p["mode"] == "weave" for n, p in rec["steps"])
