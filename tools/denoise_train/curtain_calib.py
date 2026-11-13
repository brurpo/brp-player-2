"""Calibrate the camera-noise damage against the black-curtain clip (held out, only
measured): a clean curtain (12 frames averaged, smoothed) goes through the GPU sensor
noise and a camera codec at random settings drawn from the recipe; the noise
statistics (noise_stats.py) of each are printed next to the real curtain's, and a
sheet real | synthetic is written to <data>/check/curtain_vs_synthetic.png.

  python curtain_calib.py <dir with c01.png..c12.png (16-bit RGB frames)> [--n 10]
"""
import glob, os, sys

import cv2
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import degrade as D
import gpu_degrade as G
import noise_stats as N
import recipes as R

Y0, X0, S = 192, 224, 256


def main():
    d = sys.argv[1]
    n = int(sys.argv[sys.argv.index("--n") + 1]) if "--n" in sys.argv else 10
    fs = [cv2.imread(p, -1)[..., ::-1].astype(np.float32) / 65535 for p in sorted(glob.glob(os.path.join(d, "c[0-9][0-9].png")))]
    real = np.stack([f[Y0 : Y0 + S, X0 : X0 + S] for f in fs[:8]])
    st = N.stats(list(real))
    keys = ["std_Y", "std_Cb", "std_Cr", "chroma_axis_deg", "chroma_axis_ratio", "size_Y_px", "size_C_px", "frame_to_frame_C"]
    fmt = lambda s: "  ".join(f"{s[k]:6.2f}" for k in keys)
    print(f"{'':28s}" + "  ".join(f"{k[:6]:>6s}" for k in keys))
    print(f"{'REAL curtain':28s}{fmt(st)}")
    clean_disp = cv2.GaussianBlur(np.mean(np.stack(fs), 0), (0, 0), 3)[Y0 : Y0 + S, X0 : X0 + S]
    clean = D.srgb_eotf(np.repeat(clean_disp[None], 8, 0))
    degr = G.GpuDegrader("cuda", compress_workers=4, seed=3)
    r = np.random.default_rng(5)
    sheet = [((np.clip(real[0] * 3, 0, 1)) * 255).astype(np.uint8)]
    cands = []
    for i in range(n):
        p = R.sensor_settings(r)
        p["nr"] = False
        steps = [("sensor", p), ("encode", {}), ("compress", R._compress(r))]
        t = torch.from_numpy(np.ascontiguousarray(clean.transpose(0, 3, 1, 2))).cuda().float()[None]
        job = degr.start(t.clone(), t, None, [{"kind": "camera", "hdr": False, "steps": steps}])
        _, noisy, _, labels = degr.finish(job)
        v = G.srgb_oetf(noisy[0]).clamp(0, 1).permute(0, 2, 3, 1).cpu().numpy()
        s2 = N.stats(list(v))
        print(f"{'gain %.0e wb %.1f/%.1f ccm %.1f' % (p['gain'], p['wb_r'], p['wb_b'], p['ccm']):28s}{fmt(s2)}  {labels[0][-1]}")
        dist = sum(((s2[k] - st[k]) / sc) ** 2 for k, sc in (("std_Y", 2), ("std_Cb", 2), ("std_Cr", 2), ("size_Y_px", 1), ("size_C_px", 1)))
        dist += (min(abs(s2["chroma_axis_deg"] - st["chroma_axis_deg"]), 180 - abs(s2["chroma_axis_deg"] - st["chroma_axis_deg"])) / 30) ** 2
        cands.append((dist, (np.clip(v[0] * 3, 0, 1) * 255).astype(np.uint8)))
    degr.close()
    sheet += [c[1] for c in sorted(cands, key=lambda c: c[0])[:5]]
    z = [cv2.resize(a[:128, :128], (256, 256), interpolation=cv2.INTER_NEAREST)[..., ::-1] for a in sheet]
    out = os.path.join("D:/denoise_data/check", "curtain_vs_synthetic.png")
    cv2.imwrite(out, np.concatenate(z, 1))
    print("sheet (real first, then the 5 closest draws; brightened 3x):", out)


if __name__ == "__main__":
    main()
