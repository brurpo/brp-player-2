"""Difference between two sets of flows (e.g. our port against the official code):
mean and 99th percentile of the end-point difference, per pair.
  python flowdiff.py dir_a dir_b
"""
import os, struct, sys
import numpy as np


def read(p):
    b = open(p, "rb").read()
    c, h, w = struct.unpack("<3I", b[:12])
    return np.frombuffer(b[12:], "<f4").reshape(c, h, w)[:2]


a, b = sys.argv[1:3]
for f in sorted(os.listdir(a)):
    if f.endswith(".f32") and os.path.exists(os.path.join(b, f)):
        x, y = read(os.path.join(a, f)), read(os.path.join(b, f))
        d = np.hypot(x[0] - y[0], x[1] - y[1])
        print(f"{f}: mean {d.mean():.4f} px, 99% {np.percentile(d, 99):.3f} px, max {d.max():.2f} px (flow median {np.median(np.hypot(y[0], y[1])):.2f} px)")
