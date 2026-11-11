"""Training speed on this machine (no data: random tensors), for planning.

  python bench_train.py [widths 48,96,192] [batch 8] [seq 8] [crop 256] [motion_k 3]

One step = a recurrent sequence of `seq` frames through the model (each step's
output fed to the next), L1 loss, backward, AdamW, bf16 autocast. Prints steps/s,
peak GPU memory, and the time for 300k steps.
"""
import sys, os, time
import torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model import StableDenoiser

a = sys.argv[1:]
w = tuple(int(v) for v in (a[0] if len(a) > 0 else "48,96,192").split(","))
B = int(a[1]) if len(a) > 1 else 8
T = int(a[2]) if len(a) > 2 else 8
C = int(a[3]) if len(a) > 3 else 256
mk = int(a[4]) if len(a) > 4 else 3
dev = "cuda"
m = StableDenoiser(w, (2, 3, 6), motion_k=mk).to(dev).to(memory_format=torch.channels_last)
opt = torch.optim.AdamW(m.parameters(), 2e-4)
print(f"model {sum(p.numel() for p in m.parameters()) / 1e6:.2f} M parameters; batch {B} x {T} frames x {C}x{C}")
clip = torch.rand(B, T + 2, 3, C, C, device=dev)
gt = torch.rand(B, T, 3, C, C, device=dev)
valid = torch.ones(B, 2, 1, 1, device=dev)
strength = torch.full((B, 1, 1, 1), 0.5, device=dev)


def step():
    opt.zero_grad(set_to_none=True)
    prev = torch.zeros(B, 3, C, C, device=dev)
    pv = torch.zeros(B, 1, 1, 1, device=dev)
    loss = 0
    with torch.autocast("cuda", dtype=torch.bfloat16):
        for t in range(T):
            out, _ = m(clip[:, t + 1], clip[:, t], clip[:, t + 2], valid, prev, pv, strength)
            loss = loss + (out.float() - gt[:, t]).abs().mean()
            prev, pv = out, torch.ones_like(pv)
    loss.backward()
    opt.step()


for _ in range(3):
    step()
torch.cuda.synchronize()
torch.cuda.reset_peak_memory_stats()
n = 10
t0 = time.time()
for _ in range(n):
    step()
torch.cuda.synchronize()
dt = (time.time() - t0) / n
print(f"{1 / dt:.2f} steps/s ({dt * 1000:.0f} ms per step), peak memory {torch.cuda.max_memory_allocated() >> 20} MB")
print(f"300k steps: {300000 * dt / 3600:.0f} h of GPU time")
