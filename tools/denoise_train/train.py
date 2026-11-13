"""Train the stable denoiser (PLAN.md §5-7).

  python train.py --run runs/first [--resume] [--steps 300000] [--batch 8] ...
  python dashboard.py            # the statistics window

Everything a run produces is in its folder:
  config.json        settings (a resumed run must match unless --force)
  metrics.jsonl      one line per log interval: losses, PSNR, speed, data wait, memory
  val.jsonl          one line per validation: PSNR per source, input PSNR (the baseline)
  samples/<step>.png noisy | output | clean of fixed validation crops
  status.json        state, step, speed, ETA (read by the dashboard)
  ckpt/              checkpoints (every --ckpt-min minutes, on pause/stop; last 3 + best)
  pause              create it to pause (checkpoint, wait), delete it to continue
  stop               create it to stop (checkpoint, exit)

Curriculum stage 1 (window): each step takes one frame with its neighbours (t-1, t+1)
from the clip, no recurrence (the previous-output input is zero). The model works in
PQ (the perceptual encoding inside, §3); the loss is Charbonnier in PQ plus a small
frequency term.
"""
import argparse, json, math, os, random, shutil, sys, time

for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")  # before NumPy loads; see loader.py
from dataclasses import asdict

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dataset import DataConfig
from loader import BatchLoader, val_batches
from gpu_degrade import Feeder, GpuDegrader, prepare_val
from model import StableDenoiser, build_model, load_matching

M1, M2, C1, C2, C3 = 0.1593017578125, 78.84375, 0.8359375, 18.8515625, 18.6875
WHITE = 203.0 / 10000.0


def pq(x):
    """Linear light (1.0 = SDR white, 203 nits) -> PQ (0..1)."""
    y = torch.clamp(x * WHITE, 0, 1) ** M1
    return ((C1 + C2 * y) / (1 + C3 * y)) ** M2


def charbonnier(a, b, eps=1e-3):
    return torch.sqrt((a - b) ** 2 + eps * eps).mean()


def fft_loss(a, b):
    fa, fb = torch.fft.rfft2(a.float()), torch.fft.rfft2(b.float())
    return (fa.abs() - fb.abs()).abs().mean() / (a.shape[-1] * a.shape[-2]) ** 0.5


# Inputs this close to the clean frame are the "leave clean footage alone" samples:
# their PSNR is near infinite, so they are kept out of the gain averages (one of them
# moved the camera average by -0.6 dB) and reported on their own.
CLEAN_DB = 60.0
NOISE_MAP = False  # set from --noise-map (validation uses it too)
MAP_MODE = "true"  # set from --map
STAGE = "window"  # set from --stage


def grad_loss(a, b):
    """Edges: Charbonnier on the horizontal and vertical differences, at full and half
    resolution. A softened output loses its edge strength and pays for it here."""
    total = 0.0
    for k in range(2):
        if k:
            a, b = F.avg_pool2d(a, 2), F.avg_pool2d(b, 2)
        total = total + charbonnier(a[..., :, 1:] - a[..., :, :-1], b[..., :, 1:] - b[..., :, :-1]) + charbonnier(a[..., 1:, :] - a[..., :-1, :], b[..., 1:, :] - b[..., :-1, :])
    return total / 2


def detail_loss(a, b, win=16):
    """Detail energy, patch by patch: the fine (0.7-1.5 px) and medium (1.5-3 px) band
    energy of output and clean in 16x16 patches must match. Pixel losses reward the
    average of all plausible textures, which is smoother than any of them; this
    penalises a smoothed patch directly, without asking for each texel's position."""
    from gpu_degrade import blur

    total = 0.0
    for lo, hi in ((0.7, 1.5), (1.5, 3.0)):
        ea = F.avg_pool2d((blur(a, lo) - blur(a, hi)) ** 2, win)
        eb = F.avg_pool2d((blur(b, lo) - blur(b, hi)) ** 2, win)
        total = total + ((ea + 1e-6).sqrt() - (eb + 1e-6).sqrt()).abs().mean()
    return total


def detail(x, lo, hi):
    """Band-pass energy of the luma (std of a difference of Gaussians), per sample."""
    from gpu_degrade import blur

    y = 0.2126 * x[:, 0:1] + 0.7152 * x[:, 1:2] + 0.0722 * x[:, 2:3]
    return (blur(y, lo) - blur(y, hi)).flatten(1).std(1)


def psnr(a, b):
    mse = ((a.float() - b.float()) ** 2).flatten(1).mean(1).clamp_min(1e-10)
    return -10 * torch.log10(mse)


# The noise-strength map (FFDNet-style): how much damage there is at each pixel, as
# the model's "strength" input. Without it the model is blind and smooths light noise
# as much as the heavy noise it mostly saw, losing real detail. The map is the local
# RMS of the damage in PQ (sigma), scaled to about 0..1, with a random error per
# clip (x0.7-1.4) so the model tolerates the player's estimate.
SIGMA_SCALE = 10.0


def noise_strength(sigma, jitter=True, lo=0.7, hi=1.4):
    s = sigma * SIGMA_SCALE
    if jitter:
        s = s * torch.empty(s.shape[0], 1, 1, 1, device=s.device).uniform_(lo, hi)
    return s.clamp(0, 4)


def strength_for(cur, true_sigma, mode, jitter):
    """The model's noise-map input: "true" (the damage's own RMS, known only in
    training) or "estimated" (from the noisy frame, as the player computes it, with a
    small random error for differences between estimators)."""
    if mode == "estimated":
        import noise_est

        return noise_strength(noise_est.estimate(cur), jitter, 0.85, 1.2)
    if mode == "learned":
        return None  # the model's own estimator (model.use_estimator)
    return noise_strength(true_sigma, jitter)


def estimator_loss(pred, target, alpha=0.3):
    """The estimator against the true map (a target here, never an input): under-
    estimating costs 1 - alpha, over-estimating alpha (CBDNet): too low a map leaves
    noise, too high only smooths a little."""
    d = pred - target
    return (torch.where(d < 0, 1 - alpha, alpha) * d.abs()).mean()


def window(batch, device):
    """One frame and its neighbours from each clip (PQ), the clean frame, and the noise
    map of that frame."""
    noisy, clean, sigma = batch["noisy"], batch["clean"], batch["sigma"]
    B, T = noisy.shape[:2]
    t = torch.randint(1, T - 1, (B,), device=device)
    i = torch.arange(B, device=device)
    n, c = pq(noisy), pq(clean)
    window.clean_nb = (c[i, t - 1], c[i, t + 1])  # for the motion loss
    return n[i, t], n[i, t - 1], n[i, t + 1], c[i, t], sigma[i, t]


def flow_loss(flows, cur_c, nb_c, smooth=0.02, weights=None):
    """The motion stage's own loss: each clean neighbour moved by the flow (estimated
    from the noisy frames, as in use) should match the clean current frame
    (photometric, robust to occlusions), plus an edge-aware smoothness term so flat
    areas get no wild flow. Without it the motion stage learned nothing: trained only
    through the denoising loss from a random start, its early warps scrambled the
    neighbours, the merge learned to ignore them, and no gradient came back.
    `weights`: per flow, (B,) 0/1: slots without a real neighbour do not count."""
    from model import warp

    gx = lambda x: x[..., :, 1:] - x[..., :, :-1]
    gy = lambda x: x[..., 1:, :] - x[..., :-1, :]
    lp, ls = 0.0, 0.0
    for j, (f, nb) in enumerate(zip(flows, nb_c)):
        f = f.float()
        w = 1.0 if weights is None else weights[j].view(-1, 1, 1, 1).float()
        lp = lp + (torch.sqrt((warp(nb, f) - cur_c) ** 2 + 1e-6) * w).mean()
        wx = torch.exp(-10 * gx(cur_c).abs().mean(1, keepdim=True))
        wy = torch.exp(-10 * gy(cur_c).abs().mean(1, keepdim=True))
        ls = ls + (wx * gx(f).abs() * w).mean() + (wy * gy(f).abs() * w).mean()
    return lp + smooth * ls


def flow_loss_seq(extra, clean, same=None):
    """flow_loss over a whole clip run by `sequence`: clean (B, T, 3, H, W) PQ.
    same (B, T): 0 where frame t starts another shot (with_cuts): no motion to learn
    across a cut."""
    prev_c = torch.cat([clean[:, :1], clean[:, :-1]], 1).flatten(0, 1)
    next_c = torch.cat([clean[:, 1:], clean[:, -1:]], 1).flatten(0, 1)
    wb, wa = extra["vb"].flatten(), extra["va"].flatten()
    if same is not None:
        wb = wb * same.flatten()
        wa = wa * torch.cat([same[:, 1:], same[:, -1:]], 1).flatten()
    return flow_loss([extra["fb"].flatten(0, 1), extra["fa"].flatten(0, 1)], clean.flatten(0, 1), [prev_c, next_c], weights=[wb, wa])


def with_occluders(tensors, p, seed=None):
    """Fast things passing in front: with probability `p` a soft-edged ellipse crosses
    the clip at 5-80 px a frame (at 256 px), showing the batch's next clip through it.
    Damaged and clean clips get the same one, so the target stays exact. It makes what
    real footage has and the training clips rarely do: large areas covered and
    uncovered from one frame to the next, where the aligned neighbours and the
    previous output are simply wrong and must be left out (ghosts otherwise).
    tensors: each (B, T, C, H, W). Returns them and the mask (B, T, 1, H, W).
    seed: fixed shapes (validation)."""
    B, Tn, _, H, W = tensors[0].shape
    dev = tensors[0].device
    g = None
    if seed is not None:
        g = torch.Generator(device=dev)
        g.manual_seed(seed)
    rnd = lambda *sh: torch.rand(*sh, device=dev, generator=g)
    on = ((rnd(B) < p) & (B > 1)).float()
    rad = (0.08 + 0.27 * rnd(B, 2)) * H  # radii, 20-90 px at 256
    speed = (0.02 + 0.29 * rnd(B)) * H
    ang = 2 * math.pi * rnd(B)
    through = rnd(B, 2) * torch.tensor([W, H], device=dev)  # where it is at time t0
    t0 = rnd(B) * (Tn - 1)
    soft = 0.04 + 0.4 * rnd(B)  # edge width, share of the radius (motion blur)
    t = torch.arange(Tn, device=dev)[None] - t0[:, None]  # (B, T)
    cx = through[:, :1] + speed[:, None] * torch.cos(ang)[:, None] * t
    cy = through[:, 1:] + speed[:, None] * torch.sin(ang)[:, None] * t
    yy, xx = torch.meshgrid(torch.arange(H, device=dev, dtype=torch.float32), torch.arange(W, device=dev, dtype=torch.float32), indexing="ij")
    d = (((xx - cx[..., None, None]) / rad[:, 0, None, None, None]) ** 2 + ((yy - cy[..., None, None]) / rad[:, 1, None, None, None]) ** 2).sqrt()
    m = (((1 - d) / soft[:, None, None, None]) + 0.5).clamp(0, 1) * on[:, None, None, None]
    m = m[:, :, None]
    return [x.roll(1, 0) * m.to(x.dtype) + x * (1 - m).to(x.dtype) if x.dim() == 5 and x.shape[-2:] == (H, W) else x for x in tensors], m


def with_cuts(tensors, p, at=None):
    """Scene cuts nobody announces: with probability `p` a clip goes on, from a random
    frame, as the next clip of the batch. The neighbour flags stay "valid", as in a
    player without cut detection: the model has to see that the other pictures do not
    match and leave them out. tensors: each (B, T, ...). Returns them cut alike, and
    same (B, T): 0 at the first frame of the new shot. at: the cut's frame (fixed, for
    validation; then every clip is cut)."""
    B, Tn = tensors[0].shape[:2]
    dev = tensors[0].device
    cut = torch.rand(B, device=dev) < p if at is None else torch.ones(B, dtype=torch.bool, device=dev)
    pos = torch.randint(1, Tn, (B,), device=dev) if at is None else torch.full((B,), at, device=dev)
    t = torch.arange(Tn, device=dev)[None]
    swap = cut[:, None] & (t >= pos[:, None]) & (B > 1)
    out = [torch.where(swap.view(B, Tn, *[1] * (x.dim() - 2)), x.roll(1, 0), x) for x in tensors]
    same = 1.0 - (cut[:, None] & (t == pos[:, None]) & (B > 1)).float()
    return out, same


def single(model, cur):
    """The frame denoised alone: no neighbours, no previous output (all marked invalid)."""
    B = cur.shape[0]
    z = torch.zeros(B, 1, 1, 1, device=cur.device)
    out, _ = model(cur, cur, cur, torch.zeros(B, 2, 1, 1, device=cur.device), torch.zeros_like(cur), z, torch.ones(B, 1, 1, 1, device=cur.device))
    return out


def run_model(model, cur, before, after, strength=None):
    """strength: the noise map (B, 1, H, W), or None for the constant 1 (blind)."""
    B = cur.shape[0]
    z = torch.zeros_like(cur)
    one = torch.ones(B, 1, 1, 1, device=cur.device)
    valid = torch.ones(B, 2, 1, 1, device=cur.device)
    out, _ = model(cur, before, after, valid, z, 0 * one, strength)
    return out


def sequence(model, n, true_sig, map_mode, noise_map, jitter=False, drop=0.0):
    """Stage 2: the model run frame by frame over a clip, its previous output fed back
    (recurrence). n: (B, T, 3, H, W) PQ. At the clip's ends the missing neighbour is
    the frame itself, marked invalid; `drop` also drops a neighbour at random (cuts).
    Returns the outputs (B, T, 3, H, W) and each frame's flow towards the one before
    (B, T, 2, H, W; zero at t = 0)."""
    B, T = n.shape[:2]
    dev = n.device
    prev = torch.zeros_like(n[:, 0])
    prev_valid = torch.zeros(B, 1, 1, 1, device=dev)
    outs, flows, sigs = [], [], []
    h = acc = None  # the model's memory and accumulated picture, if it has them
    has_acc = bool(getattr(model, "acc", False))
    ex = {"fb": [], "fa": [], "vb": [], "va": [], "base": []}
    if has_acc:
        ex["acc"] = []
    for t in range(T):
        cur = n[:, t]
        vb = torch.full((B, 1, 1, 1), float(t > 0), device=dev)
        va = torch.full((B, 1, 1, 1), float(t < T - 1), device=dev)
        if drop > 0:
            vb = vb * (torch.rand(B, 1, 1, 1, device=dev) >= drop)
            va = va * (torch.rand(B, 1, 1, 1, device=dev) >= drop)
        before = torch.where(vb > 0, n[:, max(t - 1, 0)], cur)
        after = torch.where(va > 0, n[:, min(t + 1, T - 1)], cur)
        st = strength_for(cur, true_sig[:, t], map_mode, jitter) if noise_map else torch.ones(B, 1, 1, 1, device=dev)
        # No frame before (clip start, or a cut): the previous output is another shot's.
        out, (fb, fa) = model(cur, before, after, torch.cat([vb, va], 1), prev, prev_valid * vb, st, state=h, acc=acc)
        h = model.last_state
        if has_acc:
            acc = model.last_acc
            ex["acc"].append(acc)
        sigs.append(model.last_sigma if getattr(model, "use_estimator", False) else None)
        outs.append(out)
        flows.append(fb if t > 0 else torch.zeros_like(fb))
        for k_, v_ in (("fb", fb), ("fa", fa), ("vb", vb.flatten()), ("va", va.flatten()), ("base", model.last_base)):
            ex[k_].append(v_)
        prev = out
        prev_valid = torch.ones(B, 1, 1, 1, device=dev)
    model.seq_sigma = torch.stack(sigs, 1) if sigs and sigs[0] is not None else None
    sequence.extra = {k_: torch.stack(v_, 1) for k_, v_ in ex.items()}  # for the motion loss and checks
    return torch.stack(outs, 1), torch.stack(flows, 1)


def stability(outs, clean, motion, thr=0.03):
    """Flicker: how each output changes from the one before (moved along the motion),
    against how the clean frames change. Penalises change the clean video does not
    have. The motion is that of the CLEAN frames (the motion stage run on them, not
    trained by this loss), and only pixels where it is reliable count: where the clean
    frames match along it (within `thr`, PQ) and it stays inside the frame. (Before,
    the model's own flow from the noisy frames was used unmasked: with a poor flow the
    loss and the measure meant little.) Returns (loss, mean absolute excess change in
    8-bit levels, share of pixels counted)."""
    from model import warp

    c1, c0 = clean[:, 1:].flatten(0, 1).float(), clean[:, :-1].flatten(0, 1).float()
    with torch.no_grad():
        f = motion(c1, c0).float()
        wc = warp(c0, f)
        H, W = f.shape[2:]
        ys = torch.arange(H, device=f.device, dtype=f.dtype).view(1, H, 1)
        xs = torch.arange(W, device=f.device, dtype=f.dtype).view(1, 1, W)
        px, py = xs + f[:, 0], ys + f[:, 1]
        inside = (px >= 0) & (px <= W - 1) & (py >= 0) & (py <= H - 1)
        m = (((c1 - wc).abs().mean(1) < thr) & inside)[:, None].float()
        d_clean = c1 - wc
    o = outs.float()
    d_out = o[:, 1:].flatten(0, 1) - warp(o[:, :-1].flatten(0, 1), f)
    e = d_out - d_clean
    den = m.sum().clamp_min(1.0) * e.shape[1]
    loss = (torch.sqrt(e * e + 1e-6) * m).sum() / den
    # The same change over blocks of 4 and 16 px (coarse flicker: boiling, blotches
    # that come and go; per pixel it is small and the loss above hardly sees it), and
    # its RMS inside 16 px blocks (flicker concentrated in a few places counts more
    # than the same amount spread thin). Blocks at least half reliable.
    coarse, lv = 0.0, []
    for k in (4, 16):
        mk = F.avg_pool2d(m, k)
        ek = F.avg_pool2d(e * m, k) / mk.clamp_min(0.25)
        wk = (mk > 0.5).float()
        dk = wk.sum().clamp_min(1.0) * e.shape[1]
        coarse = coarse + (torch.sqrt(ek * ek + 1e-8) * wk).sum() / dk
        lv.append(float((ek.detach().abs() * wk).sum() / dk) * 255)
    m16 = F.avg_pool2d(m, 16)
    w16 = (m16 > 0.5).float()
    rk = torch.sqrt(F.avg_pool2d((e * e).mean(1, keepdim=True) * m, 16) / m16.clamp_min(0.25) + 1e-8)
    peak = (rk * w16).sum() / w16.sum().clamp_min(1.0)
    stability.extra = {"coarse": coarse, "peak": peak, "c4": lv[0], "c16": lv[1], "peak_levels": float(peak.detach()) * 255}
    return loss, float((e.detach().abs() * m).sum() / den) * 255, float(m.mean())


def coarse_loss(a, b):
    """The error over blocks of 8 and 32 px: level and colour blotches. Per pixel
    they are small against fine error, so the pixel loss leaves them; the eye does not."""
    d = a - b
    return sum(torch.sqrt(F.avg_pool2d(d, k) ** 2 + 1e-8).mean() for k in (8, 32))


class Run:
    def __init__(self, a):
        self.a = a
        self.dir = a.run
        os.makedirs(os.path.join(self.dir, "samples"), exist_ok=True)
        os.makedirs(os.path.join(self.dir, "ckpt"), exist_ok=True)
        self.cfg = {
            "width": a.width, "blocks": a.blocks, "motion_k": a.motion_k, "motion_w": a.motion_w, "motion_min": a.motion_min, "motion_iters": a.motion_iters, "motion_base": a.motion_base, "fuse": a.fuse, "fuse_mode": a.fuse_mode, "keep_cur": a.keep_cur, "mismatch": a.mismatch, "state": a.state, "state_lr": a.state_lr, "acc": a.acc, "acc_weight": a.acc_weight, "stab_coarse": a.stab_coarse, "stab_peak": a.stab_peak, "coarse_weight": a.coarse_weight, "long_every": a.long_every, "long_frames": a.long_frames, "long_batch": a.long_batch, "long_low": a.long_low, "premerge_weight": a.premerge_weight, "cut_prob": a.cut_prob, "occl_prob": a.occl_prob, "batch": a.batch, "frames": a.frames, "crop": a.crop,
            "lr": a.lr, "warmup": a.warmup, "steps": a.steps, "fft_weight": a.fft_weight, "stab_weight": a.stab_weight, "grad_weight": a.grad_weight, "warp": a.warp, "noise_map": a.noise_map, "map": a.map, "train_only": a.train_only, "est_weight": a.est_weight, "flow_weight": a.flow_weight, "detail_weight": a.detail_weight, "ema": a.ema, "seed": a.seed, "stage": a.stage,
        }
        p = os.path.join(self.dir, "config.json")
        if os.path.exists(p) and a.resume:
            old = json.load(open(p))
            diff = {k: (old.get(k), v) for k, v in self.cfg.items() if old.get(k) != v}
            if diff and not a.force:
                sys.exit(f"config differs from the run's ({diff}); use --force to accept")
        json.dump(self.cfg, open(p, "w"), indent=1)
        self.status = {"state": "starting", "run": os.path.basename(os.path.abspath(self.dir)), "steps": a.steps}
        self.write_status()

    def write_status(self, **kw):
        """Never fatal: if the dashboard holds the file open at that instant (Windows
        then refuses the swap), retry briefly, else skip this update."""
        self.status.update(kw, updated=time.time())
        tmp = os.path.join(self.dir, "status.json.tmp")
        try:
            with open(tmp, "w") as f:
                json.dump(self.status, f, indent=1)
            replace_retry(tmp, os.path.join(self.dir, "status.json"))
        except OSError as e:
            print(f"status not written: {e}", flush=True)

    def log(self, name, rec):
        with open(os.path.join(self.dir, name), "a") as f:
            f.write(json.dumps(rec) + "\n")


def replace_retry(src, dst, tries=20):
    """os.replace, retried: on Windows it fails while another program (the dashboard,
    an antivirus scan) has the target open."""
    for i in range(tries):
        try:
            os.replace(src, dst)
            return
        except PermissionError:
            if i == tries - 1:
                raise
            time.sleep(0.05 * (i + 1))


def save_ckpt(run, state, tag):
    d = os.path.join(run.dir, "ckpt")
    path = os.path.join(d, f"{tag}.pt")
    torch.save(state, path + ".tmp")
    replace_retry(path + ".tmp", path)
    steps = sorted((int(f[5:-3]) for f in os.listdir(d) if f.startswith("step_") and f.endswith(".pt")))
    for s in steps[:-3]:
        os.remove(os.path.join(d, f"step_{s}.pt"))
    return path


def latest_ckpt(run):
    """The checkpoint with the most steps: the periodic ones, or the best-validation one
    when it is newer (a run that stopped before its first periodic checkpoint)."""
    d = os.path.join(run.dir, "ckpt")
    cands = [(int(f[5:-3]), os.path.join(d, f)) for f in os.listdir(d) if f.startswith("step_") and f.endswith(".pt")]
    best = os.path.join(d, "best.pt")
    if os.path.exists(best):
        cands.append((torch.load(best, map_location="cpu", weights_only=False)["step"], best))
    return max(cands)[1] if cands else None


@torch.no_grad()
def validate(model, val_batches, device, run, step):
    model.eval()
    per = {}
    clean_out = []  # output PSNR of the clean ("leave alone") samples
    detail_r = {}  # output / clean band energy (fine, medium detail), damaged samples
    flicks = []  # stage 2: excess frame-to-frame change against the clean clip
    flick_x = []  # ... over 4 px blocks, 16 px blocks, and its RMS inside 16 px blocks
    est_ratio = []  # learned map: predicted / true, mean per damaged sample
    align, fusion = [], []  # motion alignment (warped, unwarped); output loss with wrong neighbours
    base_p = []  # the pre-merged picture alone (before the U-Net)
    flicks_in, pos, longrun, counted = [], [], [], []  # stage 2: input flicker; PSNR first / last frame; long run; pixels counted
    cuts = []  # stage 2: an unannounced cut in the middle: dB lost on the frames around it
    occl = []  # stage 2: a fast occluder: dB lost where it covers / uncovers, against the frame denoised alone
    rows = []
    for b in val_batches:
        if STAGE == "recurrent":
            n_ = pq(b["noisy"].to(device).float())
            c_ = pq(b["clean"].to(device).float())
            from model import warp as _warp

            sg = b["sigma"].to(device).float()
            T_ = n_.shape[1]
            with torch.autocast("cuda", torch.bfloat16):
                outs, flows = sequence(model, n_, sg, MAP_MODE, NOISE_MAP)
            ex = sequence.extra
            outs = outs.float()
            _, flick, cnt = stability(outs, c_, model.motion)
            flick_x.append([stability.extra[k] for k in ("c4", "c16", "peak_levels")])
            flicks.append(flick)
            counted.append(cnt)
            flicks_in.append(stability(n_, c_, model.motion)[1])
            m_ = T_ // 2
            cur, out, clean = n_[:, m_], outs[:, m_], c_[:, m_]
            dmg = psnr(cur, clean) < CLEAN_DB
            align.append((psnr(_warp(c_[:, m_ - 1], ex["fb"][:, m_].float()), clean).mean().item(), psnr(c_[:, m_ - 1], clean).mean().item()))
            if dmg.any():
                # What time buys: against the same frame denoised alone.
                with torch.autocast("cuda", torch.bfloat16):
                    o1 = single(model, cur).float()
                fusion.extend((psnr(out.clamp(0, 1), clean) - psnr(o1.clamp(0, 1), clean))[dmg].tolist())
                pos.append((psnr(outs[:, 0].clamp(0, 1), c_[:, 0])[dmg].mean().item(), psnr(outs[:, -1].clamp(0, 1), c_[:, -1])[dmg].mean().item()))
                # Long run: the clip played forward, backward and forward again (3T - 2
                # frames, state carried). The last pass against the first: gain or drift.
                idx = list(range(T_)) + list(range(T_ - 2, -1, -1)) + list(range(1, T_))
                with torch.autocast("cuda", torch.bfloat16):
                    ol, _ = sequence(model, n_[:, idx], sg[:, idx], MAP_MODE, NOISE_MAP)
                ol = ol.float()[:, 2 * T_ - 2 :]  # frames 0 .. T-1 of the last pass
                longrun.append((psnr(ol[:, m_].clamp(0, 1), clean)[dmg].mean().item() - psnr(out.clamp(0, 1), clean)[dmg].mean().item(), stability(ol, c_, model.motion)[1]))
                # A cut in the middle that nobody announces (the clip goes on as the
                # batch's next one), against the two halves run as separate clips: dB
                # lost on the last frame before the cut and the first two after it.
                if n_.shape[0] > 1:
                    (nc, cc, sc), _ = with_cuts([n_, c_, sg], 1.0, at=m_)
                    with torch.autocast("cuda", torch.bfloat16):
                        oc, _ = sequence(model, nc, sc, MAP_MODE, NOISE_MAP)
                        oa, _ = sequence(model, nc[:, :m_], sc[:, :m_], MAP_MODE, NOISE_MAP)
                        ob, _ = sequence(model, nc[:, m_:], sc[:, m_:], MAP_MODE, NOISE_MAP)
                    ref = torch.cat([oa, ob], 1).float()
                    fr = [m_ - 1, m_, m_ + 1]
                    d = torch.stack([psnr(ref[:, t].clamp(0, 1), cc[:, t]) - psnr(oc[:, t].float().clamp(0, 1), cc[:, t]) for t in fr], 1)
                    ok = torch.stack([psnr(nc[:, t], cc[:, t]) < CLEAN_DB for t in fr], 1)
                    if ok.any():
                        cuts.append([(d[:, k][ok[:, k]].mean().item() if ok[:, k].any() else 0.0) for k in range(3)])
                    # A fast occluder crossing (with_occluders, fixed shapes): on the
                    # pixels it covers or uncovers at the middle frame, the clip run
                    # in order against that frame denoised alone. Above 0: the
                    # neighbours or the previous output leak in (ghosts).
                    (no, co, so), mo = with_occluders([n_, c_, sg], 1.0, seed=777 + len(occl))
                    band = ((mo[:, m_] - mo[:, m_ - 1]).abs() > 0.1).float()
                    with torch.autocast("cuda", torch.bfloat16):
                        oo, _ = sequence(model, no, so, MAP_MODE, NOISE_MAP)
                        o1o = single(model, no[:, m_]).float()
                    bp = lambda x: -10 * torch.log10((((x.clamp(0, 1) - co[:, m_]) ** 2) * band).flatten(1).sum(1) / (band.flatten(1).sum(1) * 3).clamp_min(1) + 1e-10)
                    use = (band.flatten(1).sum(1) > 300) & dmg
                    if use.any():
                        occl.append(((bp(o1o) - bp(oo[:, m_].float()))[use].mean().item(), int(use.sum())))
        else:
            cur, before, after, clean, sig = window_fixed(b, device)
            with torch.autocast("cuda", torch.bfloat16):
                out = run_model(model, cur, before, after, strength_for(cur, sig, MAP_MODE, False) if NOISE_MAP else None).float()
            # Temporal checks. Alignment: the clean t-1 moved by the motion stage's flow
            # onto the clean frame, against not moving it. Fusion: the output with
            # neighbours from another clip; a model that merges frames loses clearly.
            from model import warp as _warp

            if getattr(model, "fuse", None) is not None:
                base_p.extend(psnr(model.last_base.float().clamp(0, 1), clean)[psnr(cur, clean) < CLEAN_DB].tolist())
            cb = window_fixed.clean_nb[0]
            fb_ = model.last_flows[0].float()
            align.append((psnr(_warp(cb, fb_), clean).mean().item(), psnr(cb, clean).mean().item()))
            with torch.autocast("cuda", torch.bfloat16):
                o_w = run_model(model, cur, before.roll(1, 0), after.roll(1, 0), strength_for(cur, sig, MAP_MODE, False) if NOISE_MAP else None).float()
            dmg = psnr(cur, clean) < CLEAN_DB
            if dmg.any():
                fusion.extend((psnr(out.clamp(0, 1), clean) - psnr(o_w.clamp(0, 1), clean))[dmg].tolist())
            if MAP_MODE == "learned":
                tgt = noise_strength(sig, False)
                pm, tm = model.last_sigma.float().flatten(1).mean(1), tgt.flatten(1).mean(1)
                est_ratio.extend((pm / tm.clamp_min(1e-4))[tm > 0.02].tolist())
        p_out, p_in = psnr(out.clamp(0, 1), clean), psnr(cur, clean)
        for name, lo, hi in (("fine", 0.7, 1.5), ("med", 1.5, 3.0)):
            r_ = (detail(out.clamp(0, 1), lo, hi) / detail(clean, lo, hi).clamp_min(1e-6)).tolist()
            detail_r.setdefault(name, []).extend(x for x, pi in zip(r_, p_in.tolist()) if pi < CLEAN_DB)
        for k, po, pi in zip(b["kind"], p_out.tolist(), p_in.tolist()):
            if pi >= CLEAN_DB:
                clean_out.append(po)
                continue
            e = per.setdefault(k, [[], []])
            e[0].append(po)
            e[1].append(pi)
        if len(rows) < 6:
            for j in range(min(6 - len(rows), cur.shape[0])):
                rows.append(torch.cat([cur[j], out[j].clamp(0, 1), clean[j]], 2))
    rec = {"step": step, "time": time.time()}
    for k, (po, pi) in per.items():
        rec[f"psnr_{k}"] = float(np.mean(po))
        rec[f"psnr_in_{k}"] = float(np.mean(pi))
    allo = [x for v in per.values() for x in v[0]]
    alli = [x for v in per.values() for x in v[1]]
    rec["psnr"], rec["psnr_in"] = float(np.mean(allo)), float(np.mean(alli))
    rec["clean_psnr"] = float(np.mean(clean_out)) if clean_out else None
    for k, v in detail_r.items():
        rec[f"detail_{k}"] = float(np.median(v))
    if flicks:
        rec["flicker"] = float(np.mean(flicks))
        rec["flicker_in"] = float(np.mean(flicks_in))
        rec["flicker_counted"] = float(np.mean(counted))
        rec["flicker_c4"], rec["flicker_c16"], rec["flicker_peak"] = (float(v) for v in np.mean(flick_x, 0))
    if pos:
        rec["psnr_first"], rec["psnr_last"] = float(np.mean([x[0] for x in pos])), float(np.mean([x[1] for x in pos]))
    if longrun:
        rec["long_gain"], rec["long_flicker"] = float(np.mean([x[0] for x in longrun])), float(np.mean([x[1] for x in longrun]))
    if occl:
        rec["occl_cost"] = float(sum(x[0] * x[1] for x in occl) / sum(x[1] for x in occl))
    if cuts:
        rec["cut_before"], rec["cut_first"], rec["cut_second"] = (float(v) for v in np.mean(cuts, 0))
    if align:
        rec["align_psnr"] = float(np.mean([x[0] for x in align]))
        rec["align_psnr_nowarp"] = float(np.mean([x[1] for x in align]))
    if fusion:
        rec["fusion_gain"] = float(np.mean(fusion))
    if base_p:
        rec["psnr_premerge"] = float(np.mean(base_p))
    if est_ratio:
        rec["est_ratio_median"] = float(np.median(est_ratio))
        rec["est_ratio_q25"], rec["est_ratio_q75"] = (float(x) for x in np.quantile(est_ratio, [0.25, 0.75]))
    run.log("val.jsonl", rec)
    # Sample sheet: rows of noisy | output | clean (PQ shown as is: a sensible view).
    import cv2

    img = torch.cat(rows, 1).clamp(0, 1).permute(1, 2, 0).cpu().numpy()
    cv2.imwrite(os.path.join(run.dir, "samples", f"{step:08d}.png"), (img[..., ::-1] * 255 + 0.5).astype(np.uint8))
    model.train()
    return rec


def window_fixed(b, device):
    """Validation: always the middle frame (and its noise map, without random error)."""
    noisy, clean = pq(b["noisy"].to(device).float()), pq(b["clean"].to(device).float())
    t = noisy.shape[1] // 2
    window_fixed.clean_nb = (clean[:, t - 1], clean[:, t + 1])
    return noisy[:, t], noisy[:, t - 1], noisy[:, t + 1], clean[:, t], b["sigma"][:, t].to(device).float()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--data", default="D:/denoise_data")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--steps", type=int, default=300_000)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--frames", type=int, default=8)
    ap.add_argument("--crop", type=int, default=256)
    ap.add_argument("--width", default="48,96,192")
    ap.add_argument("--blocks", default="2,3,6")
    ap.add_argument("--motion-k", type=int, default=3)
    ap.add_argument("--motion-w", default="16,32,16", help="motion stage: channels of its three hidden layers")
    ap.add_argument("--motion-min", type=int, default=32, help="motion stage: pyramid levels until the coarsest side is <= this (px)")
    ap.add_argument("--motion-iters", type=int, default=1, help="motion stage: refinements per pyramid level")
    ap.add_argument("--mismatch", action="store_true", help="the merge also gets mismatch maps of the aligned pictures (model.mismatch)")
    ap.add_argument("--state", type=int, default=0, help="channels of the learned memory carried between frames (model.UNet.state; 0: none)")
    ap.add_argument("--state-lr", type=float, default=1.0, help="learning rate of the memory's own convolutions, as a multiple of --lr")
    ap.add_argument("--stab-coarse", type=float, default=0.0, help="stage 2: weight of the flicker loss over 4 and 16 px blocks")
    ap.add_argument("--stab-peak", type=float, default=0.0, help="stage 2: weight of the flicker loss as RMS inside 16 px blocks (local flicker)")
    ap.add_argument("--coarse-weight", type=float, default=0.0, help="stage 2: weight of the loss on the error over 8 and 32 px blocks")
    ap.add_argument("--long-every", type=int, default=0, help="stage 2: every Nth step is a batch of long render clips (0: none)")
    ap.add_argument("--long-frames", type=int, default=20)
    ap.add_argument("--long-batch", type=int, default=3)
    ap.add_argument("--long-low", type=float, default=0.6, help="long render clips: share taken from the 16 and 64 sample renders only")
    ap.add_argument("--acc", action="store_true", help="carry the accumulated pre-merge (input frames only) as a fifth picture for the pre-merge (model.acc)")
    ap.add_argument("--acc-weight", type=float, default=1.0, help="loss on the accumulated picture itself")
    ap.add_argument("--val-first", action="store_true", help="validate before the first step (the starting weights under this run's validation)")
    ap.add_argument("--occl-prob", type=float, default=0.0, help="stage 2: share of clips crossed by a fast soft-edged occluder (another clip seen through a moving ellipse)")
    ap.add_argument("--cut-prob", type=float, default=0.0, help="stage 2: share of clips with an unannounced scene cut (continues as another clip, flags left valid)")
    ap.add_argument("--motion-base", type=int, default=4, choices=[2, 4], help="motion stage: its finest level is the picture divided by this")
    ap.add_argument("--fuse", type=int, default=0, help="pre-merge stage before the U-Net: its width (0: none)")
    ap.add_argument("--fuse-mode", default="blend", choices=["blend", "residual"], help="pre-merge: per-pixel weighted mix of the aligned pictures, or a correction added to the frame")
    ap.add_argument("--keep-cur", action="store_true", help="with the pre-merge: the U-Net also gets the frame as it came")
    ap.add_argument("--premerge-weight", type=float, default=0.0, help="loss on the pre-merged picture itself (before the U-Net): behind a denoiser the pre-merge gets almost no gradient for removing noise, so it is taught directly")
    ap.add_argument("--init-motion", help="with --init-weights: take the motion stage from this checkpoint instead")
    ap.add_argument("--lr", type=float, default=4e-4)
    ap.add_argument("--warmup", type=int, default=2000)
    ap.add_argument("--fft-weight", type=float, default=0.05)
    ap.add_argument("--grad-weight", type=float, default=0.0, help="edge (gradient) loss")
    ap.add_argument("--warp", default="bilinear", choices=["bilinear", "bicubic"], help="how neighbours are moved onto the frame")
    ap.add_argument("--detail-weight", type=float, default=0.0, help="detail-energy loss (keeps texture)")
    ap.add_argument("--real-map-scale", type=float, default=1.0, help="scale of the estimated noise map in the real-footage check (4.45 for a model trained on the true map)")
    ap.add_argument("--stage", default="window", choices=["window", "recurrent"], help="window: one frame and its neighbours; recurrent: whole clips, output fed back, stability loss")
    ap.add_argument("--stab-weight", type=float, default=1.0, help="stage 2: flicker (stability) loss")
    ap.add_argument("--est-weight", type=float, default=0.1, help="learned map: weight of the estimator's loss in joint training")
    ap.add_argument("--train-only", choices=["estimator", "motion"], help="train only this part, the rest frozen (estimator: its map loss; motion: the photometric flow loss alone)")
    ap.add_argument("--flow-weight", type=float, default=0.0, help="weight of the motion stage's photometric loss on the clean frames")
    ap.add_argument("--map", default="true", choices=["true", "estimated", "learned"], help="noise map: the damage's own (training only) or estimated from the noisy frame (as in the player)")
    ap.add_argument("--init-weights", help="start from this checkpoint's weights only (new optimiser and schedule, step 0)")
    ap.add_argument("--noise-map", action="store_true", help="give the model the per-pixel noise strength (not blind)")
    ap.add_argument("--init", help="start from this checkpoint (weights, optimiser, step) in a new run")
    ap.add_argument("--stop-at", type=int, help="stop (validate, checkpoint) at this step; the schedule still spans --steps")
    ap.add_argument("--ema", type=float, default=0.999)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=6, help="data reading processes")
    ap.add_argument("--compress-workers", type=int, default=12, help="processes running the video encoders")
    ap.add_argument("--feed-depth", type=int, default=8, help="batches in flight between the loader and training (hides compression time)")
    ap.add_argument("--log-every", type=int, default=25)
    ap.add_argument("--val-every", type=int, default=2000)
    ap.add_argument("--val-size", type=int, default=128)
    ap.add_argument("--ckpt-min", type=float, default=15)
    ap.add_argument("--low-priority", action="store_true", help="lower process priority and sleep between steps")
    a = ap.parse_args()
    run = Run(a)
    global NOISE_MAP, MAP_MODE, STAGE
    NOISE_MAP, MAP_MODE, STAGE = a.noise_map, a.map, a.stage
    device = "cuda"
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    if a.low_priority and os.name == "nt":
        import ctypes

        ctypes.windll.kernel32.SetPriorityClass(ctypes.windll.kernel32.GetCurrentProcess(), 0x4000)  # below normal

    model = build_model(run.cfg).to(device)
    # EMA with a warm-up: early on it follows the model closely (decay (1+n)/(10+n)),
    # so validation is meaningful from the first checks, then settles at --ema.
    def ema_avg(avg, p, n):
        d = min(a.ema, (1 + int(n)) / (10 + int(n)))
        return d * avg + (1 - d) * p

    ema = torch.optim.swa_utils.AveragedModel(model, avg_fn=ema_avg)
    # The memory's own convolutions (new: zero or random at the start) learn at
    # --state-lr times the rate of the rest.
    fresh = [p_ for n_, p_ in model.named_parameters() if ".in_h." in n_ or ".inp_h." in n_ or n_.startswith("unet.state.")]
    rest = [p_ for p_ in model.parameters() if all(p_ is not q_ for q_ in fresh)]
    groups = [{"params": rest}] + ([{"params": fresh, "lr": a.lr * a.state_lr}] if fresh else [])
    opt = torch.optim.AdamW(groups, lr=a.lr, betas=(0.9, 0.99), weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / a.warmup) * (0.5 * (1 + math.cos(math.pi * min(s, a.steps) / a.steps)) * 0.99 + 0.01))
    step, best = 0, -1.0
    for m in (model, ema.module):
        m.warp_mode = a.warp
        m.use_estimator = a.map == "learned"
    if a.train_only:
        for n_, p_ in model.named_parameters():
            p_.requires_grad = n_.startswith(a.train_only + ".")
    p = latest_ckpt(run) if a.resume else a.init
    if a.init_weights and not a.resume:
        ck = torch.load(a.init_weights, map_location="cpu", weights_only=False)
        # Weights whose shape matches (a resized or new part starts fresh).
        missing = load_matching(model, ck["model"])
        load_matching(ema, {k: v for k, v in ck["ema"].items() if k != "n_averaged"})
        if missing:
            print(f"new parts (not in that checkpoint): {missing}", flush=True)
        if a.init_motion:
            cm = torch.load(a.init_motion, map_location="cpu", weights_only=False)
            # The motion stage's EMA weights for both: the raw ones are noisier.
            mo = {k[len("module."):]: v for k, v in cm["ema"].items() if k.startswith("module.motion.")}
            bad = load_matching(model, mo, "motion.")
            load_matching(ema, {"module." + k: v for k, v in mo.items()}, "module.motion.")
            print(f"motion stage from {a.init_motion} (step {cm['step']}){'; NOT matching: ' + str(bad) if bad else ''}", flush=True)
        torch.manual_seed(a.seed)
        random.seed(a.seed)
        print(f"weights from {a.init_weights} (step {ck['step']}); new optimiser and schedule", flush=True)
        p = None
    elif p:
        ck = torch.load(p, map_location="cpu", weights_only=False)  # RNG states must stay on the CPU
        model.load_state_dict(ck["model"])
        ema.load_state_dict(ck["ema"])
        opt.load_state_dict(ck["opt"])
        sched.load_state_dict(ck["sched"])
        step, best = ck["step"], ck.get("best", -1.0) if a.resume else -1.0
        random.setstate(ck["rng"]["py"])
        np.random.set_state(ck["rng"]["np"])
        torch.set_rng_state(ck["rng"]["torch"])
        torch.cuda.set_rng_state(ck["rng"]["cuda"])
        print(f"{'resumed' if a.resume else 'started from'} {p} at step {step}", flush=True)
    else:
        torch.manual_seed(a.seed)
        random.seed(a.seed)
    params = sum(p.numel() for p in model.parameters())

    # Validation: the same clips and damage every time (fixed seeds, also on resume).
    vcfg = DataConfig(data=a.data, frames=a.frames, crop=a.crop, seed=a.seed, val_size=a.val_size)
    dcfg = DataConfig(data=a.data, frames=a.frames, crop=a.crop, seed=a.seed + step, val_size=a.val_size)
    degr = GpuDegrader(device, compress_workers=a.compress_workers, seed=a.seed + step)
    vb = prepare_val(degr, val_batches(vcfg, a.val_size, 16, min(a.workers, 8)), device, seed=a.seed + 12345)
    loader = BatchLoader(dcfg, "train", a.batch, a.workers, seed=a.seed + step)
    feeder = Feeder(loader, degr, device, depth=a.feed_depth)
    extra = []
    a.long_feeder = None
    if a.long_every > 0 and a.stage == "recurrent":
        # Long render clips (the renders have 25 frames, the camera crops 10): fewer
        # synthetic ones, more of Blender's denoised pass and of the low sample counts.
        lcfg = DataConfig(data=a.data, frames=a.long_frames, crop=a.crop, seed=a.seed + step + 7, val_size=a.val_size, weights={"render": 1.0}, render_mix=(0.1, 0.45), render_low=a.long_low)
        ldeg = GpuDegrader(device, compress_workers=4, seed=a.seed + step + 7)
        lload = BatchLoader(lcfg, "train", a.long_batch, 3, seed=a.seed + step + 7)
        a.long_feeder = Feeder(lload, ldeg, device, depth=4)
        extra = [a.long_feeder, lload, ldeg]
    print(f"model {params / 1e6:.2f} M parameters; {len(vb)} validation batches", flush=True)
    try:
        train(a, run, model, ema, opt, sched, feeder, vb, device, step, best, params)
    except BaseException as e:
        # Visible in the dashboard instead of a stale "running".
        run.write_status(state="crashed" if not isinstance(e, KeyboardInterrupt) else "stopped", error=f"{type(e).__name__}: {e}")
        raise
    finally:
        feeder.close()
        loader.close()
        degr.close()
        for x in extra:
            x.close()


def train(a, run, model, ema, opt, sched, loader, val_batches, device, step, best, params):

    def state():
        return {"model": model.state_dict(), "ema": ema.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(), "step": step, "best": best, "config": run.cfg,
                "rng": {"py": random.getstate(), "np": np.random.get_state(), "torch": torch.get_rng_state(), "cuda": torch.cuda.get_rng_state()}}

    t_start, t_ckpt = time.time(), time.time()
    acc = {"loss": 0.0, "charb": 0.0, "fft": 0.0, "psnr": 0.0, "psnr_in": 0.0, "n": 0, "wait": 0.0, "gpu": 0.0, "kinds": {}}
    t_log = time.time()
    run.write_status(state="running", step=step, params=params, gpu=torch.cuda.get_device_name(0), started=t_start)
    model.train()
    end = min(a.steps, a.stop_at or a.steps)
    if a.val_first and step == 0:
        v0 = validate(ema.module, val_batches, device, run, 0)
        print("start: " + " ".join(f"{k} {v0[k]:.3f}" for k in ("psnr", "flicker", "flicker_c4", "flicker_c16", "flicker_peak", "fusion_gain", "psnr_first", "psnr_last", "long_gain", "occl_cost", "cut_first") if v0.get(k) is not None), flush=True)
    while step < end:
        # Pause / stop files.
        if os.path.exists(os.path.join(run.dir, "stop")):
            save_ckpt(run, state(), f"step_{step}")
            os.remove(os.path.join(run.dir, "stop"))
            run.write_status(state="stopped", step=step)
            print(f"stopped at step {step}", flush=True)
            return
        if os.path.exists(os.path.join(run.dir, "pause")):
            save_ckpt(run, state(), f"step_{step}")
            run.write_status(state="paused", step=step)
            torch.cuda.empty_cache()
            while os.path.exists(os.path.join(run.dir, "pause")) and not os.path.exists(os.path.join(run.dir, "stop")):
                time.sleep(2)
            run.write_status(state="running")
            t_log = time.time()
            continue
        t0 = time.time()
        long = a.long_feeder is not None and step % a.long_every == a.long_every - 1
        batch = (a.long_feeder if long else loader).next()
        t1 = time.time()
        if a.train_only == "motion":
            # The motion stage alone, on every frame of the clip that has both neighbours
            # (B x (T - 2) frames a step instead of B): the merge is not run at all.
            n_, c_ = pq(batch["noisy"]), pq(batch["clean"])
            mid = lambda x: x[:, 1:-1].flatten(0, 1)
            cur_a, cc = mid(n_), mid(c_)
            fb_ = model.motion(cur_a, n_[:, :-2].flatten(0, 1))
            fa_ = model.motion(cur_a, n_[:, 2:].flatten(0, 1))
            loss = flow_loss([fb_, fa_], cc, [c_[:, :-2].flatten(0, 1), c_[:, 2:].flatten(0, 1)])
            acc["flow"] = acc.get("flow", 0.0) + loss.item()
            lc = lf = torch.zeros((), device=device)
            m_ = n_.shape[1] // 2
            cur, out, clean = n_[:, m_], n_[:, m_], c_[:, m_]  # for the log only: nothing is denoised
        elif a.stage == "recurrent":
            n_, c_ = pq(batch["noisy"]), pq(batch["clean"])
            sg_, same = batch["sigma"], None
            if a.occl_prob > 0:
                (n_, c_, sg_), _ = with_occluders([n_, c_, sg_], a.occl_prob)
            if a.cut_prob > 0:
                (n_, c_, sg_), same = with_cuts([n_, c_, sg_], a.cut_prob)
            with torch.autocast("cuda", torch.bfloat16):
                outs, flows = sequence(model, n_, sg_, a.map, a.noise_map, jitter=True, drop=0.05)
            outs = outs.float()
            o2, c2 = outs.flatten(0, 1), c_.flatten(0, 1)
            lc = charbonnier(o2, c2)
            lf = fft_loss(o2, c2)
            loss = lc + a.fft_weight * lf
            if a.grad_weight > 0:
                loss = loss + a.grad_weight * grad_loss(o2, c2)
            if a.detail_weight > 0:
                loss = loss + a.detail_weight * detail_loss(o2, c2)
            ls, flick, _ = stability(outs, c_, model.motion)
            loss = loss + a.stab_weight * ls + a.stab_coarse * stability.extra["coarse"] + a.stab_peak * stability.extra["peak"]
            if a.coarse_weight > 0:
                loss = loss + a.coarse_weight * coarse_loss(o2, c2)
            if a.flow_weight > 0:
                lfl = flow_loss_seq(sequence.extra, c_, same)
                acc["flow"] = acc.get("flow", 0.0) + lfl.item()
                loss = loss + a.flow_weight * lfl
            if a.premerge_weight > 0:
                loss = loss + a.premerge_weight * charbonnier(sequence.extra["base"].float(), c_)
            if a.acc and a.acc_weight > 0:
                loss = loss + a.acc_weight * charbonnier(sequence.extra["acc"].float(), c_)
            if a.map == "learned":
                le = estimator_loss(model.seq_sigma.float().flatten(0, 1), noise_strength(sg_.flatten(0, 1), False))
                acc["est"] = acc.get("est", 0.0) + le.item()
                loss = le if a.train_only == "estimator" else loss + a.est_weight * le
            acc["flicker"] = acc.get("flicker", 0.0) + flick
            m_ = n_.shape[1] // 2
            cur, out, clean = n_[:, m_], outs[:, m_], c_[:, m_]
        else:
            cur, before, after, clean, sig = window(batch, device)
            with torch.autocast("cuda", torch.bfloat16):
                out = run_model(model, cur, before, after, strength_for(cur, sig, a.map, True) if a.noise_map else None)
            out = out.float()
            lc = charbonnier(out, clean)
            lf = fft_loss(out, clean)
            loss = lc + a.fft_weight * lf
            if a.grad_weight > 0:
                loss = loss + a.grad_weight * grad_loss(out, clean)
            if a.detail_weight > 0:
                loss = loss + a.detail_weight * detail_loss(out, clean)
            if a.map == "learned":
                le = estimator_loss(model.last_sigma.float(), noise_strength(sig, False))
                acc["est"] = acc.get("est", 0.0) + le.item()
                loss = le if a.train_only == "estimator" else loss + a.est_weight * le
            if a.premerge_weight > 0:
                loss = loss + a.premerge_weight * charbonnier(model.last_base.float(), clean)
            if a.flow_weight > 0 or a.train_only == "motion":
                lfl = flow_loss(model.last_flows, clean, window.clean_nb)
                acc["flow"] = acc.get("flow", 0.0) + lfl.item()
                loss = lfl if a.train_only == "motion" else loss + a.flow_weight * lfl
        opt.zero_grad(set_to_none=True)
        loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        ema.update_parameters(model)
        step += 1
        with torch.no_grad():
            po, pi = psnr(out.clamp(0, 1), clean), psnr(cur, clean)
        acc["n"] += 1
        acc["loss"] += loss.item()
        acc["charb"] += lc.item()
        acc["fft"] += lf.item()
        dmg = pi < CLEAN_DB
        acc["psnr"] += po[dmg].mean().item() if dmg.any() else 0.0
        acc["psnr_in"] += pi[dmg].mean().item() if dmg.any() else 0.0
        acc["n_dmg"] = acc.get("n_dmg", 0) + int(dmg.any())
        if (~dmg).any():
            acc["clean_psnr"] = acc.get("clean_psnr", 0.0) + po[~dmg].sum().item()
            acc["n_clean"] = acc.get("n_clean", 0) + int((~dmg).sum())
        for k, x, y in zip(batch["kind"], po.tolist(), pi.tolist()):
            if y >= CLEAN_DB:
                continue
            e = acc["kinds"].setdefault(k, [0.0, 0.0, 0])
            e[0] += x - y
            e[2] += 1
        acc["wait"] += t1 - t0
        acc["gpu"] += time.time() - t1
        acc["gn"] = gn.item()
        if a.low_priority:
            time.sleep(0.02)
        if step % a.log_every == 0:
            dt = time.time() - t_log
            k = acc["n"]
            rec = {
                "step": step, "time": time.time(), "lr": sched.get_last_lr()[0], "loss": acc["loss"] / k, "charb": acc["charb"] / k, "fft": acc["fft"] / k,
                "psnr": acc["psnr"] / max(acc.get("n_dmg", 0), 1), "psnr_in": acc["psnr_in"] / max(acc.get("n_dmg", 0), 1),
                "clean_psnr": acc["clean_psnr"] / acc["n_clean"] if acc.get("n_clean") else None,
                "flicker": acc["flicker"] / k if acc.get("flicker") is not None else None,
                "est_loss": acc["est"] / k if acc.get("est") is not None else None,
                "flow_loss": acc["flow"] / k if acc.get("flow") is not None else None, "steps_per_s": k / dt, "samples_per_s": k * a.batch / dt,
                "data_wait": acc["wait"] / dt, "grad_norm": acc["gn"], "gpu_mem_gb": torch.cuda.max_memory_allocated() / 1e9,
                "gain": {kk: v[0] / max(v[2], 1) for kk, v in acc["kinds"].items()},
            }
            run.log("metrics.jsonl", rec)
            left = (a.steps - step) / max(rec["steps_per_s"], 1e-6)
            run.write_status(state="running", step=step, steps_per_s=rec["steps_per_s"], eta_s=left, loss=rec["loss"], psnr_gain=rec["psnr"] - rec["psnr_in"], data_wait=rec["data_wait"])
            print(f"step {step} loss {rec['loss']:.4f} gain {rec['psnr'] - rec['psnr_in']:+.2f} dB  {rec['steps_per_s']:.1f} it/s  data wait {100 * rec['data_wait']:.0f}%", flush=True)
            acc = {"loss": 0.0, "charb": 0.0, "fft": 0.0, "psnr": 0.0, "psnr_in": 0.0, "n": 0, "wait": 0.0, "gpu": 0.0, "kinds": {}}
            torch.cuda.reset_peak_memory_stats()
            t_log = time.time()
        if step % a.val_every == 0 or step == end:
            v = validate(ema.module, val_batches, device, run, step)
            # Real noisy footage (no ground truth): sheet and measures for the window.
            if True:
                try:
                    import realtest

                    ema.module.eval()
                    realtest.report(ema.module, a.noise_map, run.dir, step, a.real_map_scale, stage=a.stage)
                    ema.module.train()
                except Exception as e:
                    print(f"realtest skipped: {e}", flush=True)
            run.write_status(val_psnr=v["psnr"], val_gain=v["psnr"] - v["psnr_in"], val_detail=v.get("detail_fine"), last_val=step)
            if v["psnr"] > best:
                best = v["psnr"]
                save_ckpt(run, state(), "best")
            t_log = time.time()
        if time.time() - t_ckpt > a.ckpt_min * 60:
            save_ckpt(run, state(), f"step_{step}")
            run.write_status(last_ckpt=step)
            t_ckpt = time.time()
    save_ckpt(run, state(), f"step_{step}")
    run.write_status(state="finished", step=step)


if __name__ == "__main__":
    main()
