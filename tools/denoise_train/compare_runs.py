"""Compare runs on their last validation: quality gain (all, per source), clean
pass-through, detail kept; and a sheet of the same validation crops' outputs side by
side (clean first) to <data>/check/compare_runs.png.

  python compare_runs.py <run dir> <run dir> ...
"""
import json, os, sys

import cv2
import numpy as np


def last_val(run):
    rows = [json.loads(l) for l in open(os.path.join(run, "val.jsonl"))]
    return rows[-1]


def main():
    runs = sys.argv[1:]
    print(f"{'run':28s} {'step':>6s} {'gain':>6s} {'camera':>7s} {'render':>7s} {'nbase':>6s} {'clean':>6s} {'fine':>6s} {'medium':>7s}")
    sheets = []
    for r in runs:
        v = last_val(r)
        g = lambda k: v[f"psnr_{k}"] - v[f"psnr_in_{k}"] if f"psnr_{k}" in v else float("nan")
        print(f"{os.path.basename(r):28s} {v['step']:6d} {v['psnr'] - v['psnr_in']:+6.2f} {g('camera'):+7.2f} {g('render'):+7.2f} {g('noisebase'):+6.2f} {v.get('clean_psnr') or float('nan'):6.1f} {100 * v.get('detail_fine', float('nan')):5.0f}% {100 * v.get('detail_med', float('nan')):6.0f}%")
        sd = os.path.join(r, "samples")
        f = sorted(os.listdir(sd))[-1]
        sheets.append((os.path.basename(r), cv2.imread(os.path.join(sd, f))))
    # Columns of every sheet: noisy | output | clean. Show clean once, then each output.
    H, W = sheets[0][1].shape[:2]
    C = W // 3
    cols = [sheets[0][1][:, 2 * C :]] + [s[1][:, C : 2 * C] for s in sheets]
    labels = ["clean"] + [s[0] for s in sheets]
    for c, l in zip(cols, labels):
        cv2.putText(c, l[:22], (4, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 255, 255), 1, cv2.LINE_AA)
    out = np.concatenate(cols, 1)
    path = "D:/denoise_data/check/compare_runs.png"
    cv2.imwrite(path, out)
    print("sheet:", path)


if __name__ == "__main__":
    main()
