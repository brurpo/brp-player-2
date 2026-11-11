"""Noise gate for training patches (PLAN.md §3): clean targets must not carry grain the
network would learn to keep. Some sources are clean in their highlights and grainy in
the shadows (Netflix low light), so this works per patch, not per clip.

  python patchnoise.py <data dir> [--workers N] [--sheet]

1. Noise maps. For every clip in clips.csv, three frames (first, middle, last) are
   cut into 16x16 blocks. Per block: luma noise and chroma noise (Immerkaer's
   Laplacian-difference estimate, 8-bit levels), how flat it is (gradient of a blurred
   luma) and its brightness; the noise is the maximum of the three frames. Saved as
   <data>/noise_maps/<clip>.npz. Resumable: maps already written are kept.
2. Gate. A patch's score is the median noise of its flat blocks (blurred-luma
   gradient under FLAT): in texture the estimate reads detail as noise, in flat areas
   it reads grain. A patch with fewer than MIN_FLAT flat blocks is all detail, which
   hides grain, and passes. REDS is taken as the clean reference: the thresholds (luma
   and chroma) are its 97th percentile over 256x256 patches. Saved as
   <data>/noise_maps/gate.json, with how many patches each source keeps. Checked
   against the dropped grainy Netflix clips: REDS keeps 98% of its patches, those 17%.
   (Scoring the flattest quarter instead rejected detailed patches: gravel, foliage.)
3. --sheet: <data>/check/noise_gate_rejected.png, the rejected patches with the
   highest scores (plain and contrast-boosted), for checking by eye.

The sampler calls patch_ok(maps, y, x, size, gate) with crop coordinates in the clip's
own pixels.
"""
import csv, json, os, sys
import numpy as np

B = 16  # block size, pixels
PATCH = 256  # patch size used for calibration and the report
QUANT = 0.97  # REDS percentile that sets the thresholds
FLAT = 3.0  # block gradient below which a block counts as flat
MIN_FLAT = 16  # flat blocks a patch needs to be scored
_M = np.array([[1, -2, 1], [-2, 4, -2], [1, -2, 1]], np.float32)
_LUMA = np.array([0.2126, 0.7152, 0.0722], np.float32)


def load(path):
    """RGB float in 8-bit levels (8- or 16-bit PNG)."""
    import cv2

    a = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if a is None:
        raise IOError(path)
    a = a[..., ::-1].astype(np.float32)
    return a / 257.0 if a.max() > 255 else a


def block_stats(img):
    """Per 16x16 block: luma noise, chroma noise, flatness (gradient), brightness."""
    import cv2

    Y = img @ _LUMA
    cb, cr = img[..., 2] - Y, img[..., 0] - Y
    h, w = Y.shape[0] // B * B, Y.shape[1] // B * B

    def blocks(x):
        return x[:h, :w].reshape(h // B, B, w // B, B).mean((1, 3))

    hp = lambda x: blocks(np.abs(cv2.filter2D(x, -1, _M))) * np.sqrt(np.pi / 2) / 6
    g = cv2.GaussianBlur(Y, (0, 0), 3)
    grad = blocks(np.hypot(cv2.Sobel(g, cv2.CV_32F, 1, 0), cv2.Sobel(g, cv2.CV_32F, 0, 1)))
    return hp(Y), np.hypot(hp(cb), hp(cr)), grad, blocks(Y)


def clip_map(row):
    frames = row["frames"].split(";")
    ys, cs = [], []
    for f in (frames[0], frames[len(frames) // 2], frames[-1]):
        y, c, grad, mean = block_stats(load(os.path.join(row["dir"], f)))
        ys.append(y)
        cs.append(c)
    return dict(y=np.max(ys, 0), c=np.max(cs, 0), grad=grad, mean=mean)


def patch_score(maps, y, x, size):
    """(luma, chroma) noise of the flat blocks in the crop, or None if too few are flat."""
    nb = max(1, size // B)
    sl = np.s_[y // B : y // B + nb, x // B : x // B + nb]
    flat = maps["grad"][sl] < FLAT
    if flat.sum() < min(MIN_FLAT, flat.size // 4 + 1):
        return None
    return float(np.median(maps["y"][sl][flat])), float(np.median(maps["c"][sl][flat]))


def patch_ok(maps, y, x, size, gate):
    s = patch_score(maps, y, x, size)
    return s is None or (s[0] <= gate["luma"] and s[1] <= gate["chroma"])


def load_maps(data, clip):
    return dict(np.load(os.path.join(data, "noise_maps", clip + ".npz")))


def patch_grid(maps, size=PATCH):
    nb = size // B
    H, W = maps["grad"].shape
    for by in range(0, H - nb + 1, nb // 2):
        for bx in range(0, W - nb + 1, nb // 2):
            yield by * B, bx * B


def _work(row):
    out = os.path.join(_out, row["clip"] + ".npz")
    if not os.path.exists(out):
        m = clip_map(row)
        np.savez_compressed(out + ".tmp.npz", **m)
        os.replace(out + ".tmp.npz", out)
    return row["clip"]


def _init(out):
    global _out
    _out = out


def main():
    data = sys.argv[1]
    args = sys.argv[2:]
    workers = int(args[args.index("--workers") + 1]) if "--workers" in args else 4
    out = os.path.join(data, "noise_maps")
    os.makedirs(out, exist_ok=True)
    rows = list(csv.DictReader(open(os.path.join(data, "clips.csv"), encoding="utf-8")))

    from multiprocessing import Pool

    with Pool(workers, _init, (out,)) as p:
        for i, _ in enumerate(p.imap_unordered(_work, rows, chunksize=4)):
            if i % 200 == 0:
                print(f"maps {i}/{len(rows)}", flush=True)

    scores = {}  # source -> list of (clip, y, x, sy, sc)
    for r in rows:
        m = load_maps(data, r["clip"])
        key = r["clip"].rsplit("_", 1)[0] if r["source"] == "netflix" else r["source"]
        for y, x in patch_grid(m):
            sc = patch_score(m, y, x, PATCH)
            scores.setdefault(key, []).append((r["clip"], y, x, *(sc or (0.0, 0.0))))
    ref = np.array([s[3:] for s in scores["reds"] if s[3] > 0])
    gate = {"luma": round(float(np.quantile(ref[:, 0], QUANT)), 3), "chroma": round(float(np.quantile(ref[:, 1], QUANT)), 3), "patch": PATCH, "block": B, "flat": FLAT, "min_flat": MIN_FLAT, "reference": f"REDS {QUANT:.0%} percentile"}
    gate["kept"] = {}
    for k, v in sorted(scores.items()):
        a = np.array([s[3:] for s in v])
        ok = (a[:, 0] <= gate["luma"]) & (a[:, 1] <= gate["chroma"])
        gate["kept"][k] = f"{ok.mean():.0%} of {len(v)}"
        print(f"{k:36s} keeps {ok.mean():5.0%} of {len(v):6d} patches", flush=True)
    json.dump(gate, open(os.path.join(out, "gate.json"), "w"), indent=1)
    print(f"thresholds: luma {gate['luma']}, chroma {gate['chroma']}")

    if "--sheet" in args:
        sheet(data, rows, scores, gate)


def sheet(data, rows, scores, gate):
    """Rejected patches with the highest scores (up to 3 clips per source), plain | 6x contrast."""
    import cv2

    dirs = {r["clip"]: r for r in rows}
    bad = []
    for k, v in scores.items():
        rej = [s for s in v if s[3] > gate["luma"] or s[4] > gate["chroma"]]
        rej.sort(key=lambda s: -max(s[3] / gate["luma"], s[4] / gate["chroma"]))
        seen = set()
        for s in rej:
            if s[0] not in seen and len(seen) < 3:
                seen.add(s[0])
                bad.append((k, s))
    tiles = []
    for k, (clip, y, x, sy, sc) in bad[:24]:
        r = dirs[clip]
        a = load(os.path.join(r["dir"], r["frames"].split(";")[0]))[y : y + PATCH, x : x + PATCH]
        boost = np.clip((a - a.mean((0, 1))) * 6 + 128, 0, 255)
        t = np.concatenate([np.clip(a, 0, 255), boost], 1).astype(np.uint8)[..., ::-1].copy()
        cv2.putText(t, f"{k} {sy:.2f}/{sc:.2f}", (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        tiles.append(t)
    while len(tiles) % 3:
        tiles.append(np.zeros_like(tiles[0]))
    img = np.concatenate([np.concatenate(tiles[i : i + 3], 1) for i in range(0, len(tiles), 3)], 0)
    path = os.path.join(data, "check", "noise_gate_rejected.png")
    cv2.imwrite(path, img)
    print("sheet:", path)


if __name__ == "__main__":
    main()
