"""Reference for our GPU MEMFOF: runs the official model (PyTorch, CUDA) on three frames
and writes the middle frame's backward and forward flows as raw f32 (c, h, w as u32,
then planar floats: dx21, dy21, dx23, dy23), plus timings.

  python memfof_ref.py <model> prev.png cur.png next.png out.f32 [iters]
  model: a Hugging Face id (egorchistov/optical-flow-MEMFOF-Tartan-T-TSKH)
Run with third_party/flowsrc/memfof on PYTHONPATH.
"""
import sys, time, struct
import numpy as np, torch, cv2
from memfof.model import MEMFOF

name, a, b, c, out = sys.argv[1:6]
iters = int(sys.argv[6]) if len(sys.argv) > 6 else 8
load = lambda p: cv2.cvtColor(cv2.imread(p, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
frames = np.stack([load(a), load(b), load(c)])            # 3, H, W, 3
x = torch.from_numpy(frames).permute(0, 3, 1, 2).float()[None].cuda()  # 1, 3, 3, H, W
model = MEMFOF.from_pretrained(name).eval().cuda()
with torch.inference_mode():
    res = model(x, iters=iters)
    flow = res["flow"][-1][0]                              # 2 (21, 23), 2, H, W
    for mode in ["fp32", "fp16"]:
        ctx = torch.autocast("cuda", dtype=torch.float16) if mode == "fp16" else torch.autocast("cuda", enabled=False)
        with ctx:
            for _ in range(2):
                model(x, iters=iters)
            torch.cuda.synchronize(); t = time.time()
            for _ in range(5):
                model(x, iters=iters)
            torch.cuda.synchronize()
        print(f"{mode}: {x.shape[-1]}x{x.shape[-2]}, {iters} iterations: {(time.time() - t) / 5 * 1000:.1f} ms")
f = flow.reshape(4, flow.shape[-2], flow.shape[-1]).float().cpu().numpy()
with open(out, "wb") as fh:
    fh.write(struct.pack("<3I", 4, f.shape[1], f.shape[2])); fh.write(f.astype("<f4").tobytes())
m = np.hypot(f[2], f[3])
print(f"forward flow: median {np.median(m):.2f} px, 99th percentile {np.percentile(m, 99):.2f} px")
