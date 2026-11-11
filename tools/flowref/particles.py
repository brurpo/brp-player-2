"""Synthetic test with known motion: a real picture panning slowly, with small particles
(dots and short streaks, 2-7 px) on top, each moving its own way (up to 30 px per
frame), some over each other. Writes f1.png ... and gt/000.f32 ... (true forward flow
of each pair, as run_flow.py writes flows), and gt/mask_000.png (particle pixels).

  python particles.py <background.png> <out_dir> [frames] [particles]
  PART=size_min,size_max,speed_min,speed_max (default 2,7,4,30)
  TSTEP=frame time step (default 1; 0.5 renders in-between moments too, for
  scoring frame interpolation: f1 and f3 are a frame apart, f2 the true middle)
"""
import os, struct, sys
import numpy as np, cv2

bg_path, out = sys.argv[1], sys.argv[2]
n_frames = int(sys.argv[3]) if len(sys.argv) > 3 else 6
n_part = int(sys.argv[4]) if len(sys.argv) > 4 else 400
rng = np.random.default_rng(7)
bg = cv2.imread(bg_path).astype(np.float32)
H, W = 1080, 1920
pan = np.array([2.5, 1.0])  # background motion per frame (x, y)
pos = rng.uniform([0, 0], [W, H], (n_part, 2))
vel = rng.normal(0, 1, (n_part, 2))
SMIN, SMAX, VMIN, VMAX = [float(v) for v in os.environ.get("PART", "2,7,4,30").split(",")]
vel = vel / np.linalg.norm(vel, axis=1, keepdims=True) * rng.uniform(VMIN, VMAX, (n_part, 1))
size = rng.uniform(SMIN, SMAX, n_part)
streak = rng.random(n_part) < 0.4
color = rng.uniform(150, 255, (n_part, 3))
os.makedirs(os.path.join(out, "gt"), exist_ok=True)


def render(t):
    m = np.float32([[1, 0, -pan[0] * t - 100], [0, 1, -pan[1] * t - 100]])
    img = cv2.warpAffine(bg, m, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    flow = np.zeros((2, H, W), np.float32)
    flow[0], flow[1] = pan[0], pan[1]
    flow[0] *= -1
    flow[1] *= -1
    # the background moves by -pan: content at x shows at x - pan next frame
    mask = np.zeros((H, W), np.uint8)
    for i in range(n_part):
        p = pos[i] + vel[i] * t
        x, y = p[0] % W, p[1] % H
        layer = np.zeros((H, W), np.uint8)
        if streak[i]:
            d = vel[i] / np.linalg.norm(vel[i]) * size[i] * 1.5
            cv2.line(layer, (int(x - d[0]), int(y - d[1])), (int(x + d[0]), int(y + d[1])), 255, max(1, int(size[i] / 3)), cv2.LINE_AA)
        else:
            cv2.circle(layer, (int(x), int(y)), int(size[i] / 2), 255, -1, cv2.LINE_AA)
        a = layer.astype(np.float32)[..., None] / 255
        img = img * (1 - a) + color[i] * a
        hit = layer > 127
        flow[0][hit], flow[1][hit] = vel[i][0], vel[i][1]
        mask[hit] = 255
    return np.clip(img, 0, 255).astype(np.uint8), flow, mask


TSTEP = float(os.environ.get("TSTEP", "1"))
for t in range(n_frames):
    img, flow, mask = render(t * TSTEP)
    cv2.imwrite(os.path.join(out, f"f{t + 1}.png"), img)
    if t < n_frames - 1:
        with open(os.path.join(out, "gt", f"{t:03d}.f32"), "wb") as fh:
            fh.write(struct.pack("<3I", 2, H, W))
            fh.write(flow.astype("<f4").tobytes())
        cv2.imwrite(os.path.join(out, "gt", f"mask_{t:03d}.png"), mask)
print(f"{n_frames} frames, {n_part} particles, background pan {pan}")
