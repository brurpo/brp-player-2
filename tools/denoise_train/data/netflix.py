"""Netflix Open Content test-condition clips (CC BY 4.0, Netflix_test_conditions/ in the
public bucket): uncompressed 10-bit 4:2:0 4K YUV. One file at a time: download
(resumable, size-checked), cut clips, measure them, delete the raw file.

  python netflix.py <data dir> <name substring>... [--every SECONDS] [--length N] [--keep]

Clips: LENGTH frames every SECONDS, scaled to 1920 wide (Lanczos, BT.709 limited range
in), saved as 16-bit RGB PNG in <data>/clips/netflix/<clip>/. Each is added to
<data>/clips.csv with its noise and motion, as prepare.py does. Also writes
<data>/check/<name>_*.png: a full frame and 1:1 crops (4K and 1080p) of its darkest
and its most detailed area, for checking by eye.
"""
import csv, os, re, subprocess, sys, urllib.request
import numpy as np
from PIL import Image

data = sys.argv[1]
args = sys.argv[2:]
opt = lambda k, d: float(args[args.index(k) + 1]) if k in args else d
EVERY, LENGTH, KEEP = opt("--every", 0.5), int(opt("--length", 10)), "--keep" in args
names = [a for i, a in enumerate(args) if not a.startswith("--") and (i == 0 or args[i - 1] not in ("--every", "--length"))]
B = "https://s3.amazonaws.com/download.opencontent.netflix.com"
raw_dir, out_root, check = (os.path.join(data, d) for d in ("raw", "clips/netflix", "check"))
for d in (raw_dir, out_root, check):
    os.makedirs(d, exist_ok=True)


def listing():
    x = urllib.request.urlopen(f"{B}/?list-type=2&prefix=Netflix_test_conditions/&delimiter=/", timeout=60).read().decode()
    return [(k, int(s)) for k, s in re.findall(r"<Key>([^<]*)</Key>.*?<Size>(\d+)</Size>", x) if k.endswith(".yuv")]


def noise_sigma(g):
    g = g.astype(np.float64)
    m = g[:-2, :-2] - 2 * g[:-2, 1:-1] + g[:-2, 2:] - 2 * g[1:-1, :-2] + 4 * g[1:-1, 1:-1] - 2 * g[1:-1, 2:] + g[2:, :-2] - 2 * g[2:, 1:-1] + g[2:, 2:]
    h, w = g.shape
    return np.sqrt(np.pi / 2) * np.abs(m).sum() / (6 * (w - 2) * (h - 2))


def gray8(path):
    a = np.asarray(Image.open(path))
    if a.dtype == np.uint16 or a.max() > 255:
        a = a.astype(np.float32) / 257.0
    a = a.astype(np.float32)
    return 0.299 * a[..., 0] + 0.587 * a[..., 1] + 0.114 * a[..., 2] if a.ndim == 3 else a


def load16(path):
    """16-bit RGB PNG (PIL can't): through ffmpeg."""
    w, h = Image.open(path).size
    p = subprocess.run(["ffmpeg", "-v", "error", "-i", path, "-f", "rawvideo", "-pix_fmt", "rgb48le", "-"], capture_output=True, check=True)
    return np.frombuffer(p.stdout, "<u2").reshape(h, w, 3).astype(np.float32) / 257.0


index_path = os.path.join(data, "clips.csv")
done = {r["clip"] for r in csv.DictReader(open(index_path, encoding="utf-8"))} if os.path.exists(index_path) else set()
fresh = not os.path.exists(index_path)
idx = open(index_path, "a", newline="", encoding="utf-8")
wr = csv.writer(idx)
if fresh:
    wr.writerow(["clip", "source", "split", "dir", "frames", "width", "height", "noise", "motion", "brightness"])

for key, size in listing():
    base = key.split("/")[-1]
    if "Sintel" in base or (names and not any(n.lower() in base.lower() for n in names)):
        continue
    m = re.match(r"(.+?)_(\d+)x(\d+)_(\d+)fps_10b(?:it)?_420\.yuv", base)
    title, W, H, fps_s = m.group(1), int(m.group(2)), int(m.group(3)), m.group(4)
    fps = {"2997": 30000 / 1001, "2398": 24000 / 1001}.get(fps_s, float(fps_s))
    raw = os.path.join(raw_dir, base)
    print(f"== {title}: {size / 1e9:.1f} GB, {W}x{H} @ {fps:.3f}", flush=True)
    have = os.path.getsize(raw) if os.path.exists(raw) else 0
    if have != size:
        subprocess.run(["curl", "-L", "--fail", "--retry", "20", "--retry-all-errors", "--retry-delay", "10", "-C", "-", "-s", "-S", "-o", raw, f"{B}/{key}"], check=True)
    assert os.path.getsize(raw) == size, "incomplete download: run again to resume"
    frame_bytes = W * H * 3 // 2 * 2
    nframes = size // frame_bytes
    rawin = ["-f", "rawvideo", "-pix_fmt", "yuv420p10le", "-s", f"{W}x{H}", "-r", str(fps)]
    conv = "scale=1920:-2:flags=lanczos:in_range=tv:in_color_matrix=bt709,format=rgb48le"
    # Check images: a 4K frame from the middle as 16-bit PNG.
    mid = nframes // 2
    full4k = os.path.join(check, f"{title}_4k.png")
    subprocess.run(["ffmpeg", "-v", "error", "-y", *rawin, "-ss", f"{mid / fps:.4f}", "-i", raw, "-frames:v", "1", "-vf", "scale=in_range=tv:in_color_matrix=bt709,format=rgb48le", full4k], check=True)
    step = max(int(round(EVERY * fps)), LENGTH + 2)
    n = 0
    for s0 in range(0, nframes - LENGTH + 1, step):
        cid = f"nf_{title}_{s0:05d}"
        d = os.path.join(out_root, cid)
        if not os.path.exists(os.path.join(d, ".done")):
            os.makedirs(d, exist_ok=True)
            subprocess.run(["ffmpeg", "-v", "error", "-y", *rawin, "-ss", f"{s0 / fps:.4f}", "-i", raw, "-frames:v", str(LENGTH), "-vf", conv, os.path.join(d, "%02d.png")], check=True)
            open(os.path.join(d, ".done"), "w").close()
        paths = sorted(os.path.join(d, f) for f in os.listdir(d) if f.endswith(".png"))
        if cid not in done and len(paths) == LENGTH:
            grays = [load16(p) for p in paths]
            grays = [0.299 * g[..., 0] + 0.587 * g[..., 1] + 0.114 * g[..., 2] for g in grays]
            sig = float(np.median([noise_sigma(g) for g in grays[::3]]))
            small = [g[::8, ::8] for g in grays]
            motion = float(np.mean([np.abs(a - b).mean() for a, b in zip(small, small[1:])]))
            w, h = Image.open(paths[0]).size
            wr.writerow([cid, "netflix", "train", d, ";".join(os.path.basename(p) for p in paths), w, h, f"{sig:.3f}", f"{motion:.3f}", f"{np.mean([g.mean() for g in grays]):.1f}"])
            idx.flush()
            done.add(cid)
        n += 1
    # Crops at 1:1 from the 4K frame and from the matching 1080p clip frame: the
    # darkest and the most detailed 256x256 (by local variance) areas.
    a4 = load16(full4k)
    g4 = 0.299 * a4[..., 0] + 0.587 * a4[..., 1] + 0.114 * a4[..., 2]
    blocks = [(y, x) for y in range(0, g4.shape[0] - 512, 256) for x in range(0, g4.shape[1] - 512, 256)]
    dark = min(blocks, key=lambda b: g4[b[0]:b[0] + 512, b[1]:b[1] + 512].mean())
    busy = max(blocks, key=lambda b: g4[b[0]:b[0] + 512, b[1]:b[1] + 512].std())
    for tag, (y, x) in (("dark", dark), ("detail", busy)):
        Image.fromarray(np.clip(a4[y:y + 512, x:x + 512], 0, 255).astype(np.uint8)).save(os.path.join(check, f"{title}_{tag}_4k.png"))
    print(f"  {n} clips; check images in {check}", flush=True)
    if not KEEP:
        os.remove(raw)
idx.close()
rows = [r for r in csv.DictReader(open(index_path, encoding="utf-8")) if r["source"] == "netflix"]
for t in sorted({r["clip"].rsplit("_", 1)[0] for r in rows}):
    rs = [r for r in rows if r["clip"].rsplit("_", 1)[0] == t]
    print(f"{t}: {len(rs)} clips, noise median {np.median([float(r['noise']) for r in rs]):.2f}, motion median {np.median([float(r['motion']) for r in rs]):.2f}, brightness {np.median([float(r['brightness']) for r in rs]):.0f}")
