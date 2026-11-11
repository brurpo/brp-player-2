"""Runs one of the reference optical flow models (official PyTorch code, CUDA) on a
sequence of frames and writes the forward flow of each consecutive pair (frame i to
i+1) as raw f32: c, h, w as u32, then planar floats (dx, dy) in pixels.

  python run_flow.py <method> <out_dir> frame0.png frame1.png ...

Methods (code in third_party/flowsrc, weights in third_party/flowckpt or Hugging Face):
  memfof   MEMFOF-Tartan-T-TSKH (3 frames: the middle one's forward flow; first and last
           pairs use the edge frame twice)
  searaft  SEA-RAFT Tartan-C-T-TSKH-spring540x960-M (runs at half size, 4 iterations)
  waft     WAFT a1 (Depth Anything V2 ViT-S), 5 iterations, full size
  memflow  MemFlowNet (things), with its memory over the sequence, at half size (its
           all-pairs correlation does not fit 16 GB at 1080p)
Prints the time per pair (after a warm-up), fp32.
"""
import json, os, struct, sys, time, types
import numpy as np, torch, torch.nn.functional as F, cv2

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
SRC = os.path.join(ROOT, "third_party", "flowsrc")
CKPT = os.path.join(ROOT, "third_party", "flowckpt")


def load(p):
    return torch.from_numpy(cv2.cvtColor(cv2.imread(p, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)).permute(2, 0, 1).float()[None].cuda()


def save(path, flow):
    f = flow[0].float().cpu().numpy()
    with open(path, "wb") as fh:
        fh.write(struct.pack("<3I", 2, f.shape[1], f.shape[2]))
        fh.write(f.astype("<f4").tobytes())


def ns(cfg_path, **extra):
    with open(cfg_path) as f:
        d = json.load(f)
    d.update(extra)
    return types.SimpleNamespace(**d)


def timed(fn):
    fn()
    torch.cuda.synchronize()
    t = time.time()
    r = fn()
    torch.cuda.synchronize()
    return r, time.time() - t


def main():
    method, out = sys.argv[1], os.path.abspath(sys.argv[2])
    frames = [load(p) for p in sys.argv[3:]]
    os.makedirs(out, exist_ok=True)
    times = []
    with torch.inference_mode():
        if method == "memfof":
            sys.path.insert(0, os.path.join(SRC, "memfof"))
            from memfof.model import MEMFOF
            model = MEMFOF.from_pretrained("egorchistov/optical-flow-MEMFOF-Tartan-T-TSKH").eval().cuda()
            for i in range(len(frames) - 1):
                trio = torch.stack([frames[max(i - 1, 0)][0], frames[i][0], frames[i + 1][0]])[None]
                res, t = timed(lambda: model(trio, iters=8))
                save(f"{out}/{i:03d}.f32", res["flow"][-1][:, 1])
                times.append(t)
        elif method == "searaft":
            sys.path.insert(0, os.path.join(SRC, "SEA-RAFT"))
            sys.path.insert(0, os.path.join(SRC, "SEA-RAFT", "core"))
            from core.raft import RAFT
            args = ns(os.path.join(SRC, "SEA-RAFT", "config", "eval", "spring-M.json"))
            model = RAFT.from_pretrained("MemorySlices/Tartan-C-T-TSKH-spring540x960-M", args=args).eval().cuda()
            s = 2 ** args.scale

            def run(a, b):
                a2 = F.interpolate(a, scale_factor=s, mode="bilinear", align_corners=False)
                b2 = F.interpolate(b, scale_factor=s, mode="bilinear", align_corners=False)
                fl = model(a2, b2, iters=args.iters, test_mode=True)["flow"][-1]
                return F.interpolate(fl, scale_factor=1 / s, mode="bilinear", align_corners=False) / s
            for i in range(len(frames) - 1):
                fl, t = timed(lambda: run(frames[i], frames[i + 1]))
                save(f"{out}/{i:03d}.f32", fl)
                times.append(t)
        elif method == "waft":
            os.chdir(os.path.join(SRC, "WAFT"))
            sys.path.insert(0, os.path.join(SRC, "WAFT"))
            from model import fetch_model
            from inference_tools import InferenceWrapper
            args = ns(os.path.join(SRC, "WAFT", "config", "a1", "tar-c-t-spring-1080p.json"))
            model = fetch_model(args)
            sd = torch.load(os.path.join(CKPT, "waft_a1.pth"), map_location="cpu")
            missing, unexpected = model.load_state_dict(sd, strict=False)
            if missing or unexpected:
                print(f"waft: {len(missing)} missing, {len(unexpected)} unexpected weights: {missing[:4]} {unexpected[:4]}")
            model = model.eval().cuda()
            w = InferenceWrapper(model, scale=args.scale, train_size=args.image_size, pad_to_train_size=False, tiling=False)
            for i in range(len(frames) - 1):
                res, t = timed(lambda: w.calc_flow(frames[i], frames[i + 1]))
                save(f"{out}/{i:03d}.f32", res["flow"][-1])
                times.append(t)
        elif method == "memflow":
            os.chdir(os.path.join(SRC, "MemFlow"))
            sys.path.insert(0, os.path.join(SRC, "MemFlow"))
            sys.path.insert(0, os.path.join(SRC, "MemFlow", "core"))
            from core.Networks import build_network
            from configs.things_memflownet import get_cfg
            from inference import inference_core_skflow as inference_core
            from utils.utils import InputPadder
            cfg = get_cfg()
            model = build_network(cfg).cuda()
            sd = torch.load(os.path.join(CKPT, "MemFlowNet_things.pth"), map_location="cpu")
            sd = sd.get("model", sd)
            sd = {k.replace("module.", "", 1): v for k, v in sd.items()}
            model.load_state_dict(sd, strict=True)
            model.eval()
            # Half size: its all-pairs correlation needs about 8 GB at 1080p (out of memory).
            images = F.interpolate(torch.cat(frames), scale_factor=0.5, mode="bilinear", align_corners=False)[None]
            padder = InputPadder(images.shape)
            images = 2 * (padder.pad(images) / 255.0) - 1.0
            proc = inference_core.InferenceCore(model, config=cfg)
            for i in range(len(frames) - 1):
                torch.cuda.synchronize()
                t0 = time.time()
                _, fl = proc.step(images[:, i:i + 2], end=(i == len(frames) - 2), add_pe=("rope" in cfg and cfg.rope), flow_init=None)
                torch.cuda.synchronize()
                times.append(time.time() - t0)
                full = F.interpolate(padder.unpad(fl[0])[None], scale_factor=2, mode="bilinear", align_corners=False) * 2
                save(f"{out}/{i:03d}.f32", full)
        else:
            sys.exit(f"unknown method {method}")
    h, w = frames[0].shape[2:]
    print(f"{method}: {w}x{h}, {len(times)} pairs, {np.median(times) * 1000:.0f} ms per pair (median)")


if __name__ == "__main__":
    main()
