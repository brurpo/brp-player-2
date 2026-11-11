"""Gallery of the degradation chain (PLAN.md §4): random degraded samples next to their
clean crops, and real noisy material for comparison, so the synthetic damage can be
checked by eye before any long training run.

  python gallery.py <data dir> [--n 18] [--seed 0] [--zoom 2]

Writes <data>/check/gallery_<seed>.png. Each tile: clean | degraded (frame 0), at 1:1
then zoomed, labelled with the steps taken. Camera samples come from REDS and Netflix
crops that pass the noise gate; render samples from Noisebase references (linear HDR,
shown tone-mapped). The last row is real noise: Netflix DinnerScene shadows (dropped
from training for its sensor noise) and Noisebase at 1 and 4 samples per pixel.
"""
import csv, json, os, sys, glob, textwrap
import cv2
import numpy as np

here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, here)
sys.path.insert(0, os.path.join(here, "data"))
import degrade as D
import patchnoise as pn

C = 256  # crop
T = 8  # frames


def show(x, hdr):
    """Linear -> 8-bit display (HDR tone-mapped)."""
    if hdr:
        x = np.maximum(x, 0)
        x = x / (1 + x) * 1.6
    return (np.clip(D.srgb_oetf(x), 0, 1) * 255 + 0.5).astype(np.uint8)


def camera_clip(rows, gate, r, data):
    while True:
        row = rows[r.integers(len(rows))]
        m = pn.load_maps(data, row["clip"])
        H, W = m["grad"].shape[0] * pn.B, m["grad"].shape[1] * pn.B
        y, x = int(r.integers(0, (H - C) // 16)) * 16, int(r.integers(0, (W - C) // 16)) * 16
        if not pn.patch_ok(m, y, x, C, gate):
            continue
        fr = row["frames"].split(";")[:T]
        clip = np.stack([pn.load(os.path.join(row["dir"], f))[y : y + C, x : x + C] for f in fr]) / 255.0
        return D.srgb_eotf(clip.astype(np.float32)), row["clip"]


# Noisebase (Zarr v2, blosc): the same reader as data/noisebase.py, which runs on import.
def nb_array(d, name):
    from numcodecs import get_codec

    meta = json.load(open(os.path.join(d, name, ".zarray")))
    shape, chunks, dtype = meta["shape"], meta["chunks"], np.dtype(meta["dtype"])
    codec = get_codec(meta["compressor"]) if meta["compressor"] else None
    out = np.full(shape, meta["fill_value"] or 0, dtype)
    for f in os.listdir(os.path.join(d, name)):
        if f.startswith("."):
            continue
        idx = [int(v) for v in f.split(".")]
        buf = codec.decode(open(os.path.join(d, name, f), "rb").read()) if codec else open(os.path.join(d, name, f), "rb").read()
        arr = np.frombuffer(buf, dtype).reshape(chunks)
        sl = tuple(slice(i * c, min((i + 1) * c, s)) for i, c, s in zip(idx, chunks, shape))
        out[sl] = arr[tuple(slice(0, s.stop - s.start) for s in sl)]
    return out


def nb_decode(rgbe, exposure):
    e = np.exp((rgbe[3].astype(np.float32) + 1) / 256 * (exposure[1] - exposure[0]) + exposure[0])
    return rgbe[:3].astype(np.float32) / 255 * e[None]


def render_clip(seqs, r):
    d = seqs[r.integers(len(seqs))]
    ref = nb_array(d, "reference")  # [F, 3, H, W]
    F, _, H, W = ref.shape
    f0 = int(r.integers(0, F - T))
    y, x = int(r.integers(0, H - C + 1)), int(r.integers(0, W - C + 1))
    clip = ref[f0 : f0 + T, :, y : y + C, x : x + C].transpose(0, 2, 3, 1).astype(np.float32)
    clip *= np.float32(2 ** r.uniform(-1, 1))  # exposure, as in training (§3)
    return np.ascontiguousarray(clip), os.path.basename(d)


def label(img, text, w):
    lines = textwrap.wrap(text, w // 7) or [""]
    pad = np.zeros((14 * len(lines) + 4, img.shape[1], 3), np.uint8)
    for i, l in enumerate(lines):
        cv2.putText(pad, l, (3, 13 + 14 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (220, 220, 220), 1, cv2.LINE_AA)
    return np.concatenate([img, pad], 0)


def tile(clean, noisy, hdr, text, zoom):
    a, b = show(clean, hdr), show(noisy, hdr)
    z = C // (2 * zoom)
    za = cv2.resize(a[C // 2 - z : C // 2 + z, C // 2 - z : C // 2 + z], (C, C), interpolation=cv2.INTER_NEAREST)
    zb = cv2.resize(b[C // 2 - z : C // 2 + z, C // 2 - z : C // 2 + z], (C, C), interpolation=cv2.INTER_NEAREST)
    img = np.concatenate([np.concatenate([a, b], 1), np.concatenate([za, zb], 1)], 0)
    return label(img, text, img.shape[1])


def grid(tiles, cols):
    h = max(t.shape[0] for t in tiles)
    tiles = [np.pad(t, ((0, h - t.shape[0]), (0, 0), (0, 0))) for t in tiles]
    while len(tiles) % cols:
        tiles.append(np.zeros_like(tiles[0]))
    sep = lambda t: np.pad(t, ((0, 6), (0, 6), (0, 0)), constant_values=60)
    return np.concatenate([np.concatenate([sep(t) for t in tiles[i : i + cols]], 1) for i in range(0, len(tiles), cols)], 0)


def main():
    data = sys.argv[1]
    a = sys.argv[2:]
    opt = lambda k, d: type(d)(a[a.index(k) + 1]) if k in a else d
    n, seed, zoom = opt("--n", 18), opt("--seed", 0), opt("--zoom", 2)
    r = np.random.default_rng(seed)
    rows = [x for x in csv.DictReader(open(os.path.join(data, "clips.csv"), encoding="utf-8")) if x["split"] == "train"]
    gate = json.load(open(os.path.join(data, "noise_maps", "gate.json")))
    seqs = sorted(d for d in glob.glob(os.path.join(data, "noisebase", "scene*")) if os.path.exists(os.path.join(d, ".done")))
    tiles = []
    for i in range(n):
        render = seqs and i % 3 == 2
        clean, name = render_clip(seqs, r) if render else camera_clip(rows, gate, r, data)
        out = D.degrade(clean, r, "render" if render else "camera")
        hdr = bool(clean.max() > 1)
        tiles.append(tile(clean[0], out.noisy[0], hdr, f"{name}: " + "; ".join(out.steps), zoom))
        print(f"{i + 1}/{n} {name}: {out.steps}", flush=True)
    # real noise
    ex = sorted(glob.glob(os.path.join(data, "excluded", "netflix", "nf_Chimera-DinnerScene_*")))
    if ex:
        im = pn.load(os.path.join(ex[len(ex) // 2], "01.png")) / 255.0
        Y = im @ D.LUMA
        best = min(((abs(Y[y : y + C, x : x + C].mean() - 0.15), y, x) for y in range(0, im.shape[0] - C, 64) for x in range(0, im.shape[1] - C, 64)))
        crop = D.srgb_eotf(im[best[1] : best[1] + C, best[2] : best[2] + C].astype(np.float32))
        tiles.append(tile(crop, crop, False, "REAL camera noise: Netflix DinnerScene shadows (same crop twice)", zoom))
    if seqs:
        d = seqs[0]
        ref, col, exp = nb_array(d, "reference"), nb_array(d, "color"), nb_array(d, "exposure")
        c = nb_decode(col[8], exp[8])
        H, W = ref.shape[2:]
        y, x = (H - C) // 2, (W - C) // 2
        cl = ref[8, :, y : y + C, x : x + C].transpose(1, 2, 0)
        for s in (1, 4):
            nz = c[:, y : y + C, x : x + C, :s].mean(-1).transpose(1, 2, 0)
            tiles.append(tile(cl, nz, True, f"REAL render noise: Noisebase {os.path.basename(d)}, {s} spp vs reference", zoom))
    img = grid(tiles, 3)
    path = os.path.join(data, "check", f"gallery_{seed}.png")
    cv2.imwrite(path, img[..., ::-1])
    print("gallery:", path)


if __name__ == "__main__":
    main()
