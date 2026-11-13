"""Real-footage check (no ground truth): the model on real noisy video, never trained
on, judged without a clean reference.

  python realtest.py prepare                      # build the set (once)
  python realtest.py run --run <run dir> [--ckpt best]   # report for a checkpoint

Set (<data>/realtest/*.npy, sRGB display values, (T, 512, 512, 3) float16):
- the CRV clip's dark noisy shots (the black curtain at 2 s, 7.5 s, 34 s), held out
  as test material from the start;
- the Netflix sequences dropped from training for their grain (BarScene1,
  DinnerScene, FoodMarket), two clips each.
Each crop is placed on the clip's darkest flat area, where noise shows most.

Measures per clip (no reference needed):
- noise left: Immerkaer noise in flat blocks of the output vs the input (luma and
  chroma, 8-bit levels);
- flicker: mean absolute change from one frame to the next in flat static areas,
  output vs input;
- detail: fine-detail energy (0.7-1.5 px band) in textured blocks, output vs input.
The noise map the model needs is estimated from the frame itself (noise_map_estimate):
noise in flat blocks per brightness level, as the player's Denoise profiles footage.
Writes <run>/realtest/<step>.png (input | output, middle frame, 2x zoom on a dark
area) and appends <run>/realtest.jsonl.
"""
import argparse, glob, json, os, subprocess, sys

for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "data"))
import patchnoise as pn

DATA = "D:/denoise_data"
OUT = os.path.join(DATA, "realtest")
CRV = "D:/BRP_PLAYER_2/footage/CRV CHA EITA LUCAS BASE.mov"
CRV_TIMES = {"crv_curtain": 2.05, "crv_7s": 7.55, "crv_34s": 34.0}
NETFLIX = ("nf_Chimera-BarScene1", "nf_Chimera-DinnerScene", "nf_ElFuente-FoodMarket")
FFMPEG = os.path.join(os.environ.get("LOCALAPPDATA", ""), r"Microsoft\WinGet\Packages\BtbN.FFmpeg.GPL.Shared.7.1_Microsoft.Winget.Source_8wekyb3d8bbwe\ffmpeg-n7.1.5-12-g1fdbca85aa-win64-gpl-shared-7.1\bin\ffmpeg.exe")
C = 512


def dark_crop(frame):
    """(y, x) of the C x C crop with the most dark flat blocks (noise shows there),
    on the 16-pixel block grid."""
    y_, c_, grad, mean = pn.block_stats(frame * 255)
    score = ((mean < 80) & (grad < 3)).astype(np.float32)
    n = C // pn.B
    best, at = -1, (0, 0)
    for by in range(0, score.shape[0] - n + 1, 2):
        for bx in range(0, score.shape[1] - n + 1, 2):
            s = score[by : by + n, bx : bx + n].sum()
            if s > best:
                best, at = s, (by * pn.B, bx * pn.B)
    return at


def prepare():
    os.makedirs(OUT, exist_ok=True)
    meta = {}
    for name, t in CRV_TIMES.items():
        p = subprocess.run([FFMPEG, "-v", "error", "-ss", f"{t:.3f}", "-i", CRV, "-frames:v", "12", "-vf", "scale=in_range=tv:in_color_matrix=bt709:out_range=pc,format=rgb48le", "-f", "rawvideo", "-"], capture_output=True, check=True)
        frames = np.frombuffer(p.stdout, "<u2").reshape(-1, 1080, 1920, 3).astype(np.float32) / 65535
        y, x = dark_crop(frames[len(frames) // 2])
        np.save(os.path.join(OUT, f"{name}.npy"), frames[:, y : y + C, x : x + C].astype(np.float16))
        meta[name] = {"source": f"CRV {t:.2f} s", "y": y, "x": x, "frames": len(frames)}
        print(name, meta[name])
    for seq in NETFLIX:
        dirs = sorted(glob.glob(os.path.join(DATA, "excluded", "netflix", seq + "_*")))
        for k, d in enumerate([dirs[len(dirs) // 3], dirs[2 * len(dirs) // 3]] if len(dirs) > 2 else dirs[:2]):
            fs = sorted(glob.glob(os.path.join(d, "*.png")))
            frames = np.stack([cv2.imread(f, cv2.IMREAD_UNCHANGED)[..., ::-1].astype(np.float32) / 65535 for f in fs])
            y, x = dark_crop(frames[len(frames) // 2])
            name = f"{seq[3:]}_{k}"
            np.save(os.path.join(OUT, f"{name}.npy"), frames[:, y : y + C, x : x + C].astype(np.float16))
            meta[name] = {"source": os.path.basename(d), "y": y, "x": x, "frames": len(frames)}
            print(name, meta[name])
    json.dump(meta, open(os.path.join(OUT, "meta.json"), "w"), indent=1)


# --- measures (NumPy, 8-bit levels on display values) --------------------------------


def flat_noise(frames):
    """Median Immerkaer noise of the flat blocks: (luma, chroma)."""
    ys, cs = [], []
    for f in frames:
        y, c, grad, mean = pn.block_stats(np.clip(f, 0, 1) * 255)
        m = grad < 3
        if m.sum() > 10:
            ys.append(np.median(y[m]))
            cs.append(np.median(c[m]))
    return (float(np.median(ys)), float(np.median(cs))) if ys else (float("nan"), float("nan"))


def flicker(frames, ref):
    """Mean |frame t+1 - frame t| (8-bit levels) where `ref` (the input) is flat and
    static: blurred input changes little between frames and has little texture."""
    out = []
    for t in range(len(frames) - 1):
        a, b = ref[t], ref[t + 1]
        ga, gb = cv2.GaussianBlur(a, (0, 0), 3), cv2.GaussianBlur(b, (0, 0), 3)
        static = (np.abs(ga - gb).mean(2) < 2 / 255)
        g = cv2.GaussianBlur(a.mean(2), (0, 0), 3)
        flat = np.hypot(cv2.Sobel(g, cv2.CV_32F, 1, 0), cv2.Sobel(g, cv2.CV_32F, 0, 1)) < 0.01
        m = static & flat
        if m.sum() > 500:
            d = np.abs(cv2.GaussianBlur(frames[t + 1], (0, 0), 1) - cv2.GaussianBlur(frames[t], (0, 0), 1)).mean(2)
            out.append(float(d[m].mean() * 255))
    return float(np.mean(out)) if out else float("nan")


def textured_detail(frame, ref):
    """Fine-detail energy (0.7-1.5 px) of `frame` in the textured part of `ref`."""
    y = np.clip(frame, 0, 1).mean(2)
    g = cv2.GaussianBlur(ref.mean(2), (0, 0), 3)
    tex = np.hypot(cv2.Sobel(g, cv2.CV_32F, 1, 0), cv2.Sobel(g, cv2.CV_32F, 0, 1)) > 0.03
    band = cv2.GaussianBlur(y, (0, 0), 0.7) - cv2.GaussianBlur(y, (0, 0), 1.5)
    return float(band[tex].std()) if tex.sum() > 500 else float("nan")


# --- noise map for real footage ---------------------------------------------------------


def noise_map_estimate(noisy_pq):
    """The model's noise-strength input for footage without a reference: noise of the
    flat blocks per brightness level (8 levels), applied by each block's brightness,
    smoothed; in the training map's units (local RMS of the damage in PQ, over the 3
    channels). noisy_pq: (T, 3, H, W) torch, PQ. Returns (T, 1, H, W) sigma."""
    import torch

    out = []
    for f in noisy_pq:
        a = f.permute(1, 2, 0).float().cpu().numpy()
        H, W, _ = a.shape
        y, c, grad, mean = pn.block_stats(a * 255)
        # Immerkaer per channel group -> RMS over 3 channels (as the training map).
        sig = np.sqrt((y**2 + c**2) / 3) / 255 * np.sqrt(3)
        lvl = np.clip((mean / 255 * 8).astype(int), 0, 7)
        flat = grad < 3 * 255 / 255
        prof = np.full(8, np.nan)
        for k in range(8):
            m = flat & (lvl == k)
            if m.sum() >= 4:
                prof[k] = np.median(sig[m])
        if np.isnan(prof).all():
            prof[:] = np.median(sig)
        idx = np.arange(8)
        good = ~np.isnan(prof)
        prof = np.interp(idx, idx[good], prof[good])
        blk = prof[lvl]
        m = cv2.resize(blk.astype(np.float32), (W, H), interpolation=cv2.INTER_LINEAR)
        m = cv2.GaussianBlur(m, (0, 0), 8)
        out.append(torch.from_numpy(m)[None])
    return torch.stack(out)


def run(args):
    import torch

    from model import StableDenoiser
    from train import noise_strength, pq, run_model, SIGMA_SCALE  # noqa: F401
    import gpu_degrade as G

    cfg = json.load(open(os.path.join(args.run, "config.json")))
    ck = torch.load(os.path.join(args.run, "ckpt", f"{args.ckpt}.pt"), map_location="cpu", weights_only=False)
    from model import build_model

    model = build_model(cfg).cuda().eval()
    model.load_state_dict({k[len("module.") :]: v for k, v in ck["ema"].items() if k.startswith("module.")})
    model.warp_mode = cfg.get("warp", "bilinear")
    model.use_estimator = cfg.get("map") == "learned"
    import train

    train.MAP_MODE = cfg.get("map", "true")
    report(model, cfg.get("noise_map", False), args.run, ck["step"], args.map_scale, args.tag, cfg.get("stage", "window"))


def report(model, noise_map, run_dir, step, map_scale=1.0, tag="", stage="window"):
    """Run `model` on the real set; sheet and measures to the run folder."""
    import torch

    from train import noise_strength, pq, run_model, SIGMA_SCALE
    import gpu_degrade as G

    learned = getattr(model, "use_estimator", False)  # the model's own estimator makes the map
    if learned:
        model.strength_scale = map_scale

    meta = json.load(open(os.path.join(OUT, "meta.json")))
    rec, tiles = {"step": step, "map_scale": map_scale}, []
    with torch.no_grad():
        for name in meta:
            clip = np.load(os.path.join(OUT, f"{name}.npy")).astype(np.float32)  # sRGB display
            lin = G.srgb_eotf(torch.from_numpy(clip).cuda().permute(0, 3, 1, 2))
            v = pq(lin)
            import noise_est

            sig = noise_est.estimate(v) * map_scale if noise_map and not learned else None
            maps = []
            if stage == "recurrent":
                # The whole clip in order, the previous output fed back (as in the
                # player); the first and last frames are left out of the measures.
                from train import sequence

                seq, _ = sequence(model, v[None], torch.zeros_like(v[None, :, :1]), "learned" if learned else "estimated", noise_map or learned)
                if learned:
                    maps.append(float(model.seq_sigma.float().mean()) / SIGMA_SCALE)
                o = seq[0, 1:-1].float().clamp(0, 1)
            else:
                outs = []
                for t in range(1, len(v) - 1):
                    st = noise_strength(sig[t : t + 1], False) if sig is not None else None
                    o = run_model(model, v[t : t + 1], v[t - 1 : t], v[t + 1 : t + 2], st).float().clamp(0, 1)
                    if learned:
                        maps.append(float(model.last_sigma.float().mean()) / SIGMA_SCALE)
                    outs.append(o[0])
                o = torch.stack(outs)
            # Back to display values for the measures and the sheet.
            disp = lambda x: G.srgb_oetf(G.pq_eotf(x)).clamp(0, 1).permute(0, 2, 3, 1).cpu().numpy()
            out_d, in_d = disp(o), clip[1:-1]
            ni, no = flat_noise(in_d), flat_noise(out_d)
            fi, fo = flicker(in_d, in_d), flicker(out_d, in_d)
            m = len(out_d) // 2
            di, do = textured_detail(in_d[m], in_d[m]), textured_detail(out_d[m], in_d[m])
            rec[name] = {"noise_in": ni, "noise_out": no, "flicker_in": fi, "flicker_out": fo, "detail_in": di, "detail_out": do, "map": float(np.mean(maps)) if maps else (float(sig.mean()) if sig is not None else None)}
            y, x = dark_zoom(in_d[m])
            # Display only: dark clips brightened so noise and detail can be judged
            # (the same gain on input and output: up to 4x, to a mean of ~0.35).
            gain = float(np.clip(0.35 / max(in_d[m][y : y + 192, x : x + 192].mean(), 1e-3), 1.0, 4.0))
            z = lambda a: cv2.resize((np.clip(a[y : y + 192, x : x + 192] * gain, 0, 1) * 255).astype(np.uint8), (384, 384), interpolation=cv2.INTER_NEAREST)
            t_ = np.concatenate([z(in_d[m]), z(out_d[m])], 1)[..., ::-1].copy()
            cv2.putText(t_, f"{name}  (view x{gain:.1f})", (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(t_, f"noise {ni[0]:.1f}/{ni[1]:.1f} -> {no[0]:.1f}/{no[1]:.1f}  flicker {fi:.1f}->{fo:.1f}", (6, 376), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1, cv2.LINE_AA)
            tiles.append(t_)
            print(f"{name:24s} noise luma/chroma {ni[0]:4.1f}/{ni[1]:4.1f} -> {no[0]:4.1f}/{no[1]:4.1f}   flicker {fi:4.2f} -> {fo:4.2f}   detail {di * 255:4.1f} -> {do * 255:4.1f}", flush=True)
    while len(tiles) % 2:
        tiles.append(np.zeros_like(tiles[0]))
    sheet = np.concatenate([np.concatenate(tiles[i : i + 2], 1) for i in range(0, len(tiles), 2)], 0)
    d = os.path.join(run_dir, "realtest")
    os.makedirs(d, exist_ok=True)
    cv2.imwrite(os.path.join(d, f"{step:08d}{tag}.png"), sheet)
    with open(os.path.join(run_dir, "realtest.jsonl"), "a") as f:
        f.write(json.dumps(rec) + "\n")
    return rec


def dark_zoom(frame):
    """A 192 px square on the darkest flat part, for the zoomed sheet."""
    g = cv2.GaussianBlur(frame.mean(2), (0, 0), 4)
    best, at = 1e9, (0, 0)
    for y in range(0, frame.shape[0] - 192 + 1, 32):
        for x in range(0, frame.shape[1] - 192 + 1, 32):
            s = g[y : y + 192, x : x + 192].mean()
            if s < best:
                best, at = s, (y, x)
    return at


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["prepare", "run"])
    ap.add_argument("--run")
    ap.add_argument("--ckpt", default="best")
    ap.add_argument("--map-scale", type=float, default=1.0, help="multiply the estimated noise map")
    ap.add_argument("--tag", default="", help="suffix of the sheet's file name")
    a = ap.parse_args()
    prepare() if a.cmd == "prepare" else run(a)
