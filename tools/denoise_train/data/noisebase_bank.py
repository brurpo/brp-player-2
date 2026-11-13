"""Noisebase into compact arrays for training (its Zarr chunks are small files: a sample
of 8 frames at 16 samples per pixel opened ~70 of them, ~3 ms each on Windows).

  python noisebase_bank.py <data dir> [--workers 12]

<data>/noisebase/scene*/ ->
<data>/bank/noisebase/<scene>.npz-like pair of raw arrays:
  <scene>.ref.f16.npy    (frames, H, W, 3) float16, the reference
  <scene>.noisy.f16.npy  (frames, K, H, W, 3) float16, the mean of the first k
                         samples per pixel for k in SPP (1, 4, 16)
Resumable: finished sequences are kept.
"""
import os, sys, glob, json
from concurrent.futures import ProcessPoolExecutor

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

SPP = (1, 4, 16)


def job(args):
    d, out = args
    import dataset  # Noisebase chunk readers

    name = os.path.basename(d)
    ref_p, noisy_p = os.path.join(out, f"{name}.ref.f16.npy"), os.path.join(out, f"{name}.noisy.f16.npy")
    if os.path.exists(ref_p) and os.path.exists(noisy_p):
        return name
    nb = dataset.Noisebase.__new__(dataset.Noisebase)
    nb.spp = {}
    have = nb._spp(d)
    F, _, H, W = dataset._nb_meta(d, "reference")["shape"]
    exposure = dataset._nb_chunk(d, "exposure", (0, 0))
    ref = np.lib.format.open_memmap(ref_p + ".part.npy", "w+", np.float16, (F, H, W, 3))
    noisy = np.lib.format.open_memmap(noisy_p + ".part.npy", "w+", np.float16, (F, len(SPP), H, W, 3))
    for f in range(F):
        ref[f] = np.clip(dataset._nb_chunk(d, "reference", (f, 0, 0, 0))[0].transpose(1, 2, 0), 0, 65000)
        for i, k in enumerate(SPP):
            noisy[f, i] = np.clip(nb._frame(d, f, min(k, have), exposure), 0, 65000)
    ref.flush(), noisy.flush()
    del ref, noisy
    os.replace(ref_p + ".part.npy", ref_p)
    os.replace(noisy_p + ".part.npy", noisy_p)
    return name


def main():
    data = sys.argv[1]
    a = sys.argv[2:]
    workers = int(a[a.index("--workers") + 1]) if "--workers" in a else 12
    out = os.path.join(data, "bank", "noisebase")
    os.makedirs(out, exist_ok=True)
    seqs = sorted(d for d in glob.glob(os.path.join(data, "noisebase", "scene*")) if os.path.exists(os.path.join(d, ".done")))
    with ProcessPoolExecutor(workers) as ex:
        for i, n in enumerate(ex.map(job, [(d, out) for d in seqs])):
            if i % 16 == 0:
                print(f"[{i + 1}/{len(seqs)}] {n}", flush=True)
    json.dump({"spp": SPP, "sequences": [os.path.basename(d) for d in seqs]}, open(os.path.join(out, "meta.json"), "w"), indent=1)
    print("done")


if __name__ == "__main__":
    main()
