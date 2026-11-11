"""Scores optical flows by how well they move each next frame onto the current one.

  python score.py <seq_dir> method1 method2 ...

<seq_dir> holds the frames f1.png, f2.png, ... and one folder per method with 000.f32,
001.f32, ... (forward flow of each consecutive pair, from run_flow.py or aicheck
flowdump; a flow smaller than the frame is scaled up). For each method and pair, the
next frame is sampled at x + flow (bilinear) and compared with the current one, over the
pixels whose sample lands inside the picture:
  mean   mean absolute RGB error (0..1)
  bad    share of pixels off by more than 0.1
  detail the same share among pixels with detail (local gradient above 0.04), where a
         wrong motion shows; flat areas match whatever the motion
Also writes <seq_dir>/<method>_vis.png: the flow (hue = direction, saturation = size).
"""
import os, struct, sys
import numpy as np, cv2

seq = sys.argv[1]
frames = sorted([f for f in os.listdir(seq) if f.startswith("f") and f.endswith(".png")], key=lambda s: int(s[1:-4]))
img = [cv2.cvtColor(cv2.imread(os.path.join(seq, f)), cv2.COLOR_BGR2RGB).astype(np.float32) / 255 for f in frames]
H, W = img[0].shape[:2]


def read(path):
    with open(path, "rb") as fh:
        c, h, w = struct.unpack("<3I", fh.read(12))
        f = np.frombuffer(fh.read(), dtype="<f4").reshape(c, h, w)
    if (h, w) != (H, W):
        k = W / w
        f = np.stack([cv2.resize(f[i], (W, H), interpolation=cv2.INTER_LINEAR) * k for i in range(2)])
    return f[:2]


def vis(f):
    mag = np.hypot(f[0], f[1])
    hsv = np.zeros((H, W, 3), np.uint8)
    hsv[..., 0] = ((np.arctan2(f[1], f[0]) + np.pi) / (2 * np.pi) * 180).astype(np.uint8)
    hsv[..., 1] = np.clip(mag / max(np.percentile(mag, 99), 1e-3) * 255, 0, 255).astype(np.uint8)
    hsv[..., 2] = 255
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


gy, gx = np.mgrid[0:H, 0:W].astype(np.float32)
print(f"{seq}: {len(img) - 1} pairs, {W}x{H}")
has_gt = os.path.isdir(os.path.join(seq, "gt"))
print(f"{'method':10} {'mean':>7} {'bad':>7} {'detail':>7}" + (f" {'EPE':>7} {'EPE prt':>8} {'prt>3px':>8}" if has_gt else ""))
for m in sys.argv[2:]:
    means, bads, dets, epes, epps, outs = [], [], [], [], [], []
    for i in range(len(img) - 1):
        p = os.path.join(seq, m, f"{i:03d}.f32")
        if not os.path.exists(p):
            continue
        f = read(p)
        mx, my = gx + f[0], gy + f[1]
        inside = (mx >= 0) & (mx <= W - 1) & (my >= 0) & (my <= H - 1)
        warped = cv2.remap(img[i + 1], mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        err = np.abs(warped - img[i]).mean(axis=2)
        g = cv2.cvtColor(img[i], cv2.COLOR_RGB2GRAY)
        grad = np.hypot(cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3), cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)) / 8
        detail = inside & (grad > 0.04)
        means.append(err[inside].mean())
        bads.append((err[inside] > 0.1).mean())
        dets.append((err[detail] > 0.1).mean() if detail.any() else 0)
        if has_gt:
            gt = read(os.path.join(seq, "gt", f"{i:03d}.f32"))
            pm = cv2.imread(os.path.join(seq, "gt", f"mask_{i:03d}.png"), 0) > 127
            e = np.hypot(f[0] - gt[0], f[1] - gt[1])
            epes.append(e.mean())
            epps.append(e[pm].mean())
            outs.append((e[pm] > 3).mean())
        if i == len(img) // 2 - 1:
            cv2.imwrite(os.path.join(seq, f"{m}_vis.png"), vis(f))
    if means:
        extra = f" {np.mean(epes):7.2f} {np.mean(epps):8.2f} {np.mean(outs) * 100:7.1f}%" if has_gt else ""
        print(f"{m:10} {np.mean(means):7.4f} {np.mean(bads) * 100:6.2f}% {np.mean(dets) * 100:6.2f}%" + extra)
