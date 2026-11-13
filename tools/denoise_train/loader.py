"""Batches from worker processes through shared memory (no PyTorch in the workers).

    with BatchLoader(cfg, "train", batch=8, workers=12) as ld:
        b = ld.next(device)   # clean, noisy (real noisy inputs), flow: CUDA tensors;
                              # recipe, kind, origin: per sample

Each worker runs dataset.Sampler (NumPy) and fills whole batches into slots of shared
memory (float16: clean and real noisy (B, T, 3, C, C), flow (B, T, 2, C, C)), with
each sample's damage recipe; gpu_degrade.GpuDegrader then makes the damage on the
GPU. The training process copies a ready slot to the GPU and hands it back. PyTorch's own
DataLoader would import PyTorch in every worker: on Windows that reserves ~4 GB of
memory each for its CUDA libraries, which ran a 31 GB machine out of memory.

    val_batches(cfg, n, batch)  # the fixed validation set, as NumPy batches
"""
import multiprocessing as mp
import os

# One BLAS thread per process, set before NumPy loads (workers inherit it): OpenBLAS
# otherwise reserves a buffer per CPU thread (0.8 GB per process here) and its idle
# threads slow the samplers down 3x.
for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import queue
import sys
import time
from multiprocessing import shared_memory

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def _shapes(cfg, batch):
    T, C = cfg.frames, cfg.crop
    return {"clean": (batch, T, 3, C, C), "noisy": (batch, T, 3, C, C), "flow": (batch, T, 2, C, C)}


def _views(buf, shapes):
    out, off = {}, 0
    for k, sh in shapes.items():
        n = int(np.prod(sh))
        out[k] = np.ndarray(sh, np.float16, buf, off)
        off += n * 2
    return out


def _slot_bytes(shapes):
    return sum(int(np.prod(s)) * 2 for s in shapes.values())


def _worker(cfg, split, batch, names, free_q, ready_q, seed):
    import dataset

    s = dataset.Sampler(cfg, split)
    r = np.random.default_rng(seed)
    shapes = _shapes(cfg, batch)
    shms = [shared_memory.SharedMemory(name=n) for n in names]
    views = [_views(m.buf, shapes) for m in shms]
    while True:
        slot = free_q.get()
        if slot is None:
            break
        v = views[slot]
        meta = []
        for i in range(batch):
            x = s.safe(r)
            v["clean"][i] = np.clip(x["clean"], -65000, 65000)
            if x["noisy"] is not None:
                v["noisy"][i] = np.clip(x["noisy"], -65000, 65000)
            if x["flow"] is not None:
                v["flow"][i] = np.clip(x["flow"], -65000, 65000)
            meta.append({"kind": x["kind"], "origin": x["origin"], "recipe": x["recipe"], "flow": x["flow"] is not None})
        ready_q.put((slot, meta))
    del views
    for m in shms:
        m.close()


class BatchLoader:
    def __init__(self, cfg, split="train", batch=8, workers=12, slots=None, seed=0):
        self.cfg, self.batch = cfg, batch
        self.shapes = _shapes(cfg, batch)
        n = slots or workers * 2 + 2
        size = _slot_bytes(self.shapes)
        self.shms = [shared_memory.SharedMemory(create=True, size=size) for _ in range(n)]
        self.views = [_views(m.buf, self.shapes) for m in self.shms]
        ctx = mp.get_context("spawn")
        self.free_q, self.ready_q = ctx.Queue(), ctx.Queue()
        for i in range(n):
            self.free_q.put(i)
        names = [m.name for m in self.shms]
        self.procs = [ctx.Process(target=_worker, args=(cfg, split, batch, names, self.free_q, self.ready_q, [seed, w]), daemon=True, name=f"data{w}") for w in range(workers)]
        for p in self.procs:
            p.start()
        self.wait = 0.0  # seconds spent waiting for the next batch

    def next(self, device):
        import torch

        t = time.time()
        while True:
            try:
                slot, meta = self.ready_q.get(timeout=5)
                break
            except queue.Empty:
                if not any(p.is_alive() for p in self.procs):
                    raise RuntimeError("all data workers died")
        self.wait += time.time() - t
        v = self.views[slot]
        out = {k: torch.from_numpy(v[k]).to(device).float() for k in ("clean", "noisy")}
        out["flow"] = torch.from_numpy(v["flow"]).to(device).float() if any(m["flow"] for m in meta) else None
        self.free_q.put(slot)
        out["kind"] = [m["kind"] for m in meta]
        out["origin"] = [m["origin"] for m in meta]
        out["recipe"] = [m["recipe"] for m in meta]
        return out

    def close(self):
        for _ in self.procs:
            self.free_q.put(None)
        for p in self.procs:
            p.join(timeout=5)
            if p.is_alive():
                p.terminate()
        self.views = None
        for m in self.shms:
            try:
                m.close()
                m.unlink()
            except Exception:
                pass

    def __enter__(self):
        return self

    def __exit__(self, *a):
        self.close()


def _val_one(args):
    cfg, i = args
    import dataset

    global _VAL_SAMPLER
    if "_VAL_SAMPLER" not in globals():
        _VAL_SAMPLER = dataset.Sampler(cfg, "val")
    x = _VAL_SAMPLER.safe(np.random.default_rng([cfg.seed, 999, i]))
    T, _, H, W = x["clean"].shape
    x["clean"] = np.clip(x["clean"], -65000, 65000).astype(np.float16)
    x["noisy"] = np.zeros_like(x["clean"]) if x["noisy"] is None else np.clip(x["noisy"], -65000, 65000).astype(np.float16)
    x["flow"] = np.zeros((T, 2, H, W), np.float16) if x["flow"] is None else x["flow"].astype(np.float16)
    return x


def val_batches(cfg, n, batch=16, workers=8):
    """The fixed validation set (sample i is always the same), as NumPy batches of
    clean clips and recipes (the GPU applies the damage, with a fixed seed)."""
    ctx = mp.get_context("spawn")
    with ctx.Pool(workers) as pool:
        xs = pool.map(_val_one, [(cfg, i) for i in range(n)])
    out = []
    for i in range(0, n, batch):
        part = xs[i : i + batch]
        b = {k: np.stack([x[k] for x in part]) for k in ("clean", "noisy", "flow")}
        b["kind"] = [x["kind"] for x in part]
        b["recipe"] = [x["recipe"] for x in part]
        out.append(b)
    return out
