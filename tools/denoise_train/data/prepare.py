"""Clean training clips for the stable denoiser (PLAN.md §3), from what fetch.sh put in
$DENOISE_DATA/src. Writes $DENOISE_DATA/clips/... and the index $DENOISE_DATA/clips.csv.

  python prepare.py <data dir> [--every SECONDS] [--length FRAMES]

- Films (Tears of Steel, Sintel, 4K masters): a low-resolution pass finds the shot
  cuts; inside each shot a clip of LENGTH frames is taken every SECONDS, never across
  a cut, skipping near-black frames (titles, credits). Clips are scaled to 1920 wide
  (Lanczos; halving a 4K master averages its own noise away) and saved as PNG.
- REDS: its sequences (100 frames, 1280x720 PNG) are indexed as clips of LENGTH
  frames in place, not copied.
- Every clip gets its measured noise (Immerkaer's estimate on luma, 8-bit levels) and
  motion (mean absolute change between frames, 8-bit levels at 1/8 size): training
  leaves out noisy clips and balances still against moving ones.

Resumable: clips already written (with their .done marker) are kept.
"""
import csv, os, subprocess, sys, json
import numpy as np
from PIL import Image

data = sys.argv[1]
args = sys.argv[2:]
opt = lambda k, d: float(args[args.index(k) + 1]) if k in args else d
EVERY = opt("--every", 3.0)
LENGTH = int(opt("--length", 10))
src = os.path.join(data, "src")
out_root = os.path.join(data, "clips")
os.makedirs(out_root, exist_ok=True)


def noise_sigma(gray):
    """Immerkaer (1996): noise standard deviation from a Laplacian-difference mask."""
    g = gray.astype(np.float64)
    m = (g[:-2, :-2] - 2 * g[:-2, 1:-1] + g[:-2, 2:] - 2 * g[1:-1, :-2] + 4 * g[1:-1, 1:-1] - 2 * g[1:-1, 2:] + g[2:, :-2] - 2 * g[2:, 1:-1] + g[2:, 2:])
    h, w = gray.shape
    return np.sqrt(np.pi / 2) * np.abs(m).sum() / (6 * (w - 2) * (h - 2))


def clip_stats(paths):
    grays = [np.asarray(Image.open(p).convert("L"), dtype=np.float32) for p in paths]
    sig = float(np.median([noise_sigma(g) for g in grays[:: max(1, len(grays) // 3)]]))
    small = [g[::8, ::8] for g in grays]
    motion = float(np.mean([np.abs(a - b).mean() for a, b in zip(small, small[1:])]))
    return sig, motion, float(np.mean([g.mean() for g in grays]))


def probe(path):
    r = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=width,height,r_frame_rate,nb_frames", "-show_entries", "format=duration", "-of", "json", path], capture_output=True, text=True, check=True)
    j = json.loads(r.stdout)
    s = j["streams"][0]
    num, den = map(int, s["r_frame_rate"].split("/"))
    return s["width"], s["height"], num / den, float(j["format"]["duration"])


def low_res_frames(path, w=160):
    """All frames at w px wide, grey (for cuts and black frames)."""
    _, _, _, _ = probe(path)
    p = subprocess.run(["ffmpeg", "-v", "error", "-i", path, "-vf", f"scale={w}:-2,format=gray", "-f", "rawvideo", "-"], capture_output=True, check=True)
    W0, H0, _, _ = probe(path)
    h = int(round(H0 * w / W0 / 2)) * 2
    a = np.frombuffer(p.stdout, np.uint8)
    return a[: a.size // (w * h) * (w * h)].reshape(-1, h, w).astype(np.float32)


def cuts_of(frames):
    """Frame indices where a new shot starts: a jump in the grey histogram well above
    the change around it."""
    hist = np.stack([np.histogram(f, 32, (0, 256))[0] / f.size for f in frames])
    d = np.abs(np.diff(hist, axis=0)).sum(1)
    med = np.array([np.median(d[max(0, i - 12) : i + 12]) for i in range(len(d))])
    return [0] + [i + 1 for i in range(len(d)) if d[i] > 0.5 and d[i] > 4 * med[i] + 0.1] + [len(frames)]


def film(name, path, writer):
    W0, H0, fps, dur = probe(path)
    print(f"{name}: {W0}x{H0} {fps:.3f} fps {dur:.0f} s; finding shots", flush=True)
    lo = low_res_frames(path)
    cuts = cuts_of(lo)
    print(f"  {len(lo)} frames, {len(cuts) - 2} cuts", flush=True)
    step = int(round(EVERY * fps))
    n = 0
    for a, b in zip(cuts, cuts[1:]):
        for s0 in range(a + 2, b - LENGTH - 2, step):
            seg = lo[s0 : s0 + LENGTH]
            if seg.mean() < 12 or (seg.max(axis=(1, 2)) < 30).any():
                continue
            cid = f"{name}_{s0:06d}"
            d = os.path.join(out_root, name, cid)
            if not os.path.exists(os.path.join(d, ".done")):
                os.makedirs(d, exist_ok=True)
                t = s0 / fps
                subprocess.run(["ffmpeg", "-v", "error", "-y", "-ss", f"{t:.4f}", "-i", path, "-frames:v", str(LENGTH), "-vf", "scale=1920:-2:flags=lanczos", "-pix_fmt", "rgb24", os.path.join(d, "%02d.png")], check=True)
                open(os.path.join(d, ".done"), "w").close()
            paths = sorted(os.path.join(d, f) for f in os.listdir(d) if f.endswith(".png"))
            if len(paths) < LENGTH:
                continue
            writer(cid, name, d, paths)
            n += 1
    print(f"  {n} clips", flush=True)


def reds(name, root, split, writer):
    seqs = sorted(dirs for dirs in os.listdir(root) if os.path.isdir(os.path.join(root, dirs)))
    n = 0
    for sq in seqs:
        d = os.path.join(root, sq)
        frames = sorted(os.path.join(d, f) for f in os.listdir(d) if f.endswith(".png"))
        for s0 in range(0, len(frames) - LENGTH + 1, LENGTH):
            writer(f"{name}_{sq}_{s0:03d}", name, d, frames[s0 : s0 + LENGTH], split)
            n += 1
    print(f"{name}: {len(seqs)} sequences, {n} clips", flush=True)


def find(pattern_dir, exts):
    for r, _, fs in os.walk(pattern_dir):
        for f in fs:
            if f.lower().endswith(exts):
                return os.path.join(r, f)
    return None


index_path = os.path.join(data, "clips.csv")
done = set()
if os.path.exists(index_path):
    done = {r["clip"] for r in csv.DictReader(open(index_path, encoding="utf-8"))}
fresh = not os.path.exists(index_path)
out = open(index_path, "a", newline="", encoding="utf-8")
wr = csv.writer(out)
if fresh:
    wr.writerow(["clip", "source", "split", "dir", "frames", "width", "height", "noise", "motion", "brightness"])


def writer(cid, source, d, paths, split="train"):
    if cid in done:
        return
    sig, motion, bright = clip_stats(paths)
    w, h = Image.open(paths[0]).size
    wr.writerow([cid, source, split, d, ";".join(os.path.basename(p) for p in paths), w, h, f"{sig:.3f}", f"{motion:.3f}", f"{bright:.1f}"])
    out.flush()
    done.add(cid)


# Sources marked "excluded" in sources.csv are skipped (folder names as fetch.sh makes them).
here = os.path.dirname(os.path.abspath(__file__))
excluded = set()
for r in csv.DictReader(open(os.path.join(here, "sources.csv"), encoding="utf-8")):
    if r["use"] == "excluded":
        excluded.add("_".join(filter(None, r["name"].replace("(", " ").replace(")", " ").split())))

for entry in sorted(os.listdir(src)):
    base = os.path.join(src, entry)
    if entry.strip("_") in excluded:
        print(f"{entry}: excluded in sources.csv, skipped")
        continue
    if entry.startswith("REDS"):
        # train_sharp/train/train_sharp/000/00000000.png (the zip's layout)
        seq_root = None
        for r, ds, _ in os.walk(base):
            if any(d.isdigit() and len(d) == 3 for d in ds):
                seq_root = r
                break
        if seq_root:
            reds("reds_val" if "validation" in entry else "reds", seq_root, "val" if "validation" in entry else "train", writer)
    else:
        movie = find(base, (".mov", ".mkv", ".mp4"))
        if movie:
            film(entry.lower().split("_")[0], movie, writer)
out.close()
rows = list(csv.DictReader(open(index_path, encoding="utf-8")))
print(f"index: {len(rows)} clips in {index_path}")
for s in sorted({r['source'] for r in rows}):
    rs = [r for r in rows if r["source"] == s]
    print(f"  {s}: {len(rs)} clips, noise median {np.median([float(r['noise']) for r in rs]):.2f}, motion median {np.median([float(r['motion']) for r in rs]):.2f}")
