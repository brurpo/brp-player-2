"""Per-sample validation report for a checkpoint: input and output PSNR (in PQ, as
training measures), gain, and each sample's damage, grouped by source and by how
damaged the input was. Uses the run's fixed validation set (same seeds as train.py).

  python val_report.py --run D:/denoise_data/runs/stage1_window [--ckpt best] [--n 128]
"""
import argparse, json, os, sys

for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dataset import DataConfig
from gpu_degrade import GpuDegrader, describe
from loader import val_batches
from model import StableDenoiser
from train import detail, noise_strength, pq, psnr, run_model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--ckpt", default="best")
    ap.add_argument("--n", type=int, default=128)
    a = ap.parse_args()
    cfg = json.load(open(os.path.join(a.run, "config.json")))
    ck = torch.load(os.path.join(a.run, "ckpt", f"{a.ckpt}.pt"), map_location="cpu", weights_only=False)
    w = tuple(int(x) for x in cfg["width"].split(","))
    n = tuple(int(x) for x in cfg["blocks"].split(","))
    model = StableDenoiser(w=w, n=n, motion_k=cfg["motion_k"]).cuda().eval()
    ema = {k[len("module."):]: v for k, v in ck["ema"].items() if k.startswith("module.")}
    model.load_state_dict(ema)
    model.warp_mode = cfg.get("warp", "bilinear")
    vcfg = DataConfig(frames=cfg["frames"], crop=cfg["crop"], seed=cfg["seed"], val_size=a.n)
    raw = val_batches(vcfg, a.n, 16, 8)
    degr = GpuDegrader("cuda", compress_workers=8, seed=cfg["seed"] + 12345)
    rows = []
    with torch.no_grad():
        for b in raw:
            t = lambda k: torch.from_numpy(b[k]).cuda().float()
            job = degr.start(t("clean"), t("noisy"), t("flow"), b["recipe"])
            clean, noisy, sigma, labels = degr.finish(job)
            n_, c_ = pq(noisy), pq(clean)
            m = n_.shape[1] // 2
            strength = noise_strength(sigma[:, m], False) if cfg.get("noise_map") else None
            out = run_model(model, n_[:, m], n_[:, m - 1], n_[:, m + 1], strength).float()
            pi, po = psnr(n_[:, m], c_[:, m]).tolist(), psnr(out.clamp(0, 1), c_[:, m]).tolist()
            df = (detail(out.clamp(0, 1), 0.7, 1.5) / detail(c_[:, m], 0.7, 1.5).clamp_min(1e-6)).tolist()
            dm = (detail(out.clamp(0, 1), 1.5, 3.0) / detail(c_[:, m], 1.5, 3.0).clamp_min(1e-6)).tolist()
            di = (detail(n_[:, m], 0.7, 1.5) / detail(c_[:, m], 0.7, 1.5).clamp_min(1e-6)).tolist()
            for k, rec, l, x, y, f_, m_, i_ in zip(b["kind"], b["recipe"], labels, pi, po, df, dm, di):
                rows.append((k, x, y, ", ".join(l) or "-", f_, m_, i_))
    degr.close()
    print(f"checkpoint {a.ckpt} (step {ck['step']}), {len(rows)} samples\n")
    for kind in sorted({r[0] for r in rows}):
        rs = [r for r in rows if r[0] == kind]
        print(f"== {kind}: {len(rs)} samples, mean gain {np.mean([r[2] - r[1] for r in rs]):+.2f} dB")
        for lo, hi in ((0, 30), (30, 36), (36, 42), (42, 99)):
            g = [r for r in rs if lo <= r[1] < hi]
            if g:
                print(f"   input PSNR {lo:2d}-{hi:2d} dB: {len(g):3d} samples, gain {np.mean([r[2] - r[1] for r in g]):+.2f} dB, detail kept fine {100 * np.median([r[4] for r in g]):3.0f}% medium {100 * np.median([r[5] for r in g]):3.0f}%")
        light = [r for r in rs if 36 <= r[1] < 60]
        if light:
            print("   lightly damaged inputs, fine detail: input / output (vs clean), damage:")
            for r in sorted(light, key=lambda r: r[4])[:14]:
                print(f"     in {100 * r[6]:4.0f}%  out {100 * r[4]:4.0f}%   {r[3]}")
        for r in sorted(rs, key=lambda r: r[2] - r[1])[:6]:
            print(f"     worst: in {r[1]:5.1f} out {r[2]:5.1f} ({r[2] - r[1]:+.2f})  {r[3]}")
        print()


if __name__ == "__main__":
    main()
