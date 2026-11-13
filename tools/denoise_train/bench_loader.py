"""Throughput of the data pipeline alone (no training): batches per second.

  python bench_loader.py [--workers 14] [--batch 8] [--seconds 40]
"""
import argparse, os, sys, time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from loader import BatchLoader  # sets the BLAS thread limits first
from dataset import DataConfig

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=14)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--seconds", type=float, default=40)
    ap.add_argument("--device", default="cuda", help="where batches go (as in training)")
    a = ap.parse_args()
    with BatchLoader(DataConfig(), "train", a.batch, a.workers) as ld:
        ld.next(a.device)  # wait for start-up
        n, kinds, t = 0, {}, time.time()
        while time.time() - t < a.seconds:
            b = ld.next(a.device)
            n += 1
            for k in b["kind"]:
                kinds[k] = kinds.get(k, 0) + 1
        dt = time.time() - t
    print(f"{a.workers} workers, batch {a.batch}: {n / dt:.2f} batches/s = {n * a.batch / dt:.1f} samples/s; mix {kinds}")
