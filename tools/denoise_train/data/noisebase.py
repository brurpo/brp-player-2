"""Noisebase SampleSet v1 (Balint et al., "Neural Partitioning Pyramids for Denoising
Monte Carlo Renderings", SIGGRAPH 2023): path-traced 64-frame sequences of 256x256,
32 samples per pixel stored separately, references at 6144 spp (median of three
OIDN-denoised 2048 spp renders). Licence: none stated for the data; used by the
project owner's decision (see sources.csv).

  python noisebase.py <data dir> [--every 8] [--samples 16] [--jobs 4]

Each sequence is a Zarr ZipStore of ~2.4 GB on the server. Only what we use is
fetched, with HTTP range requests on the zip: `color` (RGBE, the first SAMPLES
samples: chunks of 4), `reference`, `exposure`, `crop_offset`. The pieces stay in
their compressed Zarr form, in <data>/noisebase/scene####/ (a Zarr directory store).
About 300 MB per sequence at 16 samples. Sequences 0, EVERY, 2*EVERY, ... (the scenes
are spread at random over the sequences).

Also writes <data>/check/noisebase_scene####.png: reference, 1 sample and 16 samples
of frame 32, tone-mapped side by side.
"""
import http.client, json, os, struct, sys, zlib, threading, queue, urllib.parse
from concurrent.futures import ThreadPoolExecutor
import numpy as np

data = sys.argv[1]
args = sys.argv[2:]
opt = lambda k, d: int(args[args.index(k) + 1]) if k in args else d
EVERY, SAMPLES, JOBS = opt("--every", 8), opt("--samples", 16), opt("--jobs", 4)
HOST = "neural-partitioning-pyramids.mpi-inf.mpg.de"
PATH = "/data/sampleset_v1/train/scene{:04d}.zip"
out_root = os.path.join(data, "noisebase")
check = os.path.join(data, "check")
os.makedirs(out_root, exist_ok=True)
os.makedirs(check, exist_ok=True)
local = threading.local()


def conn():
    if getattr(local, "c", None) is None:
        local.c = http.client.HTTPSConnection(HOST, timeout=120)
    return local.c


def get(path, a=None, b=None, tries=8):
    for t in range(tries):
        try:
            c = conn()
            h = {"Range": f"bytes={a}-{b}"} if a is not None else {}
            c.request("GET" if a is not None else "HEAD", path, headers=h)
            r = c.getresponse()
            body = r.read()
            if r.status in (200, 206):
                return r, body
            raise IOError(f"HTTP {r.status}")
        except Exception as e:
            local.c = None
            if t == tries - 1:
                raise
            import time
            time.sleep(5 * (t + 1))


def members(path):
    r, _ = get(path)
    size = int(r.getheader("Content-Length"))
    _, tail = get(path, size - 65536, size - 1)
    i = tail.rfind(b"PK\x05\x06")
    cd_size, cd_off = struct.unpack("<II", tail[i + 12 : i + 20])
    if cd_off == 0xFFFFFFFF:
        j = tail.rfind(b"PK\x06\x06")
        cd_size, cd_off = struct.unpack("<QQ", tail[j + 40 : j + 56])
    _, cd = get(path, cd_off, cd_off + cd_size - 1)
    p, out = 0, []
    while p < len(cd) and cd[p : p + 4] == b"PK\x01\x02":
        method = struct.unpack("<H", cd[p + 10 : p + 12])[0]
        comp, unc = struct.unpack("<II", cd[p + 20 : p + 28])
        n, e, c = struct.unpack("<HHH", cd[p + 28 : p + 34])
        lho = struct.unpack("<I", cd[p + 42 : p + 46])[0]
        name = cd[p + 46 : p + 46 + n].decode()
        extra = cd[p + 46 + n : p + 46 + n + e]
        q = 0
        while q < len(extra):
            hid, hs = struct.unpack("<HH", extra[q : q + 4])
            body, k = extra[q + 4 : q + 4 + hs], 0
            if hid == 1:
                if unc == 0xFFFFFFFF:
                    unc = struct.unpack("<Q", body[k : k + 8])[0]; k += 8
                if comp == 0xFFFFFFFF:
                    comp = struct.unpack("<Q", body[k : k + 8])[0]; k += 8
                if lho == 0xFFFFFFFF:
                    lho = struct.unpack("<Q", body[k : k + 8])[0]
            q += 4 + hs
        out.append((name, comp, lho, method))
        p += 46 + n + e + c
    return out


def wanted(name):
    top = name.split("/")[0]
    if name == ".zgroup" or top in ("reference", "exposure", "crop_offset"):
        return True
    if top == "color":
        if name.endswith(".zarray") or name.endswith(".zattrs"):
            return True
        s = int(name.split(".")[-1])  # chunk index along samples (4 per chunk)
        return s * 4 < SAMPLES
    return False


def fetch(index):
    seq = f"scene{index:04d}"
    d = os.path.join(out_root, seq)
    if os.path.exists(os.path.join(d, ".done")):
        return seq, "kept"
    path = PATH.format(index)
    ms = sorted((m for m in members(path) if wanted(m[0])), key=lambda m: m[2])
    # Coalesce neighbouring members into ranges of up to 64 MB (local header 30 bytes +
    # name + extra before each member's data; read generously and parse).
    groups, cur = [], []
    for m in ms:
        if cur and (m[2] - cur[0][2] > 64 << 20 or m[2] - (cur[-1][2] + cur[-1][1] + 4096) > 1 << 20):
            groups.append(cur)
            cur = []
        cur.append(m)
    if cur:
        groups.append(cur)
    total = 0
    for g in groups:
        a = g[0][2]
        b = g[-1][2] + g[-1][1] + 30 + 1024
        _, blob = get(path, a, b)
        for name, comp, lho, method in g:
            o = lho - a
            n, e = struct.unpack("<HH", blob[o + 26 : o + 30])
            raw = blob[o + 30 + n + e : o + 30 + n + e + comp]
            if method == 8:
                raw = zlib.decompress(raw, -15)
            f = os.path.join(d, *name.split("/"))
            os.makedirs(os.path.dirname(f), exist_ok=True)
            open(f, "wb").write(raw)
            total += len(raw)
    open(os.path.join(d, ".done"), "w").close()
    try:
        preview(d, seq)
    except Exception as e:
        print(f"  {seq}: preview failed: {e}", flush=True)
    return seq, f"{total / 1e6:.0f} MB"


def read_array(d, name):
    """A Zarr v2 array from a directory store (whole array; blosc via numcodecs)."""
    from numcodecs import get_codec
    meta = json.load(open(os.path.join(d, name, ".zarray")))
    shape, chunks, dtype = meta["shape"], meta["chunks"], np.dtype(meta["dtype"])
    codec = get_codec(meta["compressor"]) if meta["compressor"] else None
    out = np.full(shape, meta["fill_value"] or 0, dtype)
    for f in os.listdir(os.path.join(d, name)):
        if f.startswith("."):
            continue
        idx = [int(v) for v in f.split(".")]
        raw = open(os.path.join(d, name, f), "rb").read()
        buf = codec.decode(raw) if codec else raw
        arr = np.frombuffer(buf, dtype).reshape(chunks)
        sl = tuple(slice(i * c, min((i + 1) * c, s)) for i, c, s in zip(idx, chunks, shape))
        out[sl] = arr[tuple(slice(0, s.stop - s.start) for s in sl)]
    return out


def decode_color(rgbe, exposure):
    """Noisebase's decompress_RGBE: rgbe [4, H, W, S] uint8, exposure [min, max] (log)."""
    e = np.exp((rgbe[3].astype(np.float32) + 1) / 256 * (exposure[1] - exposure[0]) + exposure[0])
    return rgbe[:3].astype(np.float32) / 255 * e[None]


def tonemap(x):
    x = np.maximum(x, 0)
    return (np.clip(x / (1 + x), 0, 1) ** (1 / 2.2) * 255 + 0.5).astype(np.uint8)


def preview(d, seq):
    from PIL import Image
    ref = read_array(d, "reference")
    exp = read_array(d, "exposure")
    col = read_array(d, "color")
    f = 32
    c = decode_color(col[f], exp[f])  # [3, H, W, S]
    s = min(SAMPLES, c.shape[-1])
    panels = [ref[f], c[..., 0], c[..., :s].mean(-1)]
    img = np.concatenate([tonemap(p.transpose(1, 2, 0)) for p in panels], 1)
    Image.fromarray(img).save(os.path.join(check, f"noisebase_{seq}.png"))


indices = list(range(0, 1024, EVERY))
print(f"{len(indices)} sequences, {SAMPLES} samples each, {JOBS} at a time", flush=True)
with ThreadPoolExecutor(JOBS) as ex:
    for i, (seq, what) in enumerate(ex.map(fetch, indices)):
        print(f"[{i + 1}/{len(indices)}] {seq}: {what}", flush=True)
print("done")
