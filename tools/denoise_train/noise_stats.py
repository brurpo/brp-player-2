"""Statistics of the noise in flat areas of a clip, to calibrate the synthetic damage:
strength in luma and the two chroma channels (8-bit levels), the main chroma axis,
blob size (half width of the autocorrelation) and how much of the pattern stays from
one frame to the next (after aligning the frames).

    stats(frames)  # frames: (T, H, W, 3) display values 0..1 of a flat area
"""
import cv2
import numpy as np

# BT.709 display RGB -> Y, Cb, Cr
M = np.array([[0.2126, 0.7152, 0.0722], [-0.1146, -0.3854, 0.5], [0.5, -0.4542, -0.0458]], np.float32)


def residual(f, s=10.0):
    """Remove the picture's own slow shape (folds, gradients)."""
    return f - cv2.GaussianBlur(f, (0, 0), s)


def acf_halfwidth(x):
    """Distance (px) where the radial autocorrelation falls to half."""
    x = x - x.mean()
    F = np.fft.rfft2(x)
    ac = np.fft.irfft2(np.abs(F) ** 2, s=x.shape)
    ac = np.fft.fftshift(ac) / ac.flat[0]
    cy, cx = np.array(ac.shape) // 2
    yy, xx = np.indices(ac.shape)
    r = np.hypot(yy - cy, xx - cx).astype(int)
    prof = np.bincount(r.ravel(), ac.ravel()) / np.bincount(r.ravel())
    below = np.nonzero(prof < 0.5)[0]
    if len(below) == 0:
        return float("nan")
    i = below[0]
    return float(i - 1 + (prof[i - 1] - 0.5) / (prof[i - 1] - prof[i]))


def stats(frames):
    T = len(frames)
    ycc = [np.einsum("hwc,kc->hwk", residual(f), M) * 255 for f in frames]
    out = {}
    allr = np.stack(ycc)
    out["std_Y"], out["std_Cb"], out["std_Cr"] = [float(allr[..., k].std()) for k in range(3)]
    c = allr[..., 1:].reshape(-1, 2)
    w, v = np.linalg.eigh(np.cov(c.T))
    out["chroma_axis_deg"] = float(np.degrees(np.arctan2(v[1, 1], v[0, 1])))  # angle in the Cb-Cr plane
    out["chroma_axis_ratio"] = float(np.sqrt(w[1] / max(w[0], 1e-12)))
    out["size_Y_px"] = float(np.nanmean([acf_halfwidth(a[..., 0]) for a in ycc]))
    out["size_C_px"] = float(np.nanmean([acf_halfwidth(a[..., 1]) for a in ycc] + [acf_halfwidth(a[..., 2]) for a in ycc]))
    # Persistence: correlation of consecutive residuals after aligning by phase correlation.
    keep = []
    for t in range(T - 1):
        a, b = ycc[t], ycc[t + 1]
        (dx, dy), _ = cv2.phaseCorrelate(cv2.GaussianBlur(frames[t] @ M[0], (0, 0), 3).astype(np.float64), cv2.GaussianBlur(frames[t + 1] @ M[0], (0, 0), 3).astype(np.float64))
        Mx = np.float32([[1, 0, -dx], [0, 1, -dy]])
        bw = cv2.warpAffine(b, Mx, (b.shape[1], b.shape[0]), borderMode=cv2.BORDER_REFLECT)
        m = 16
        for k in (0, 1, 2):
            x, y = a[m:-m, m:-m, k].ravel(), bw[m:-m, m:-m, k].ravel()
            keep.append((k, float(np.corrcoef(x, y)[0, 1])))
    out["frame_to_frame_Y"] = float(np.mean([c for k, c in keep if k == 0]))
    out["frame_to_frame_C"] = float(np.mean([c for k, c in keep if k > 0]))
    out["mean_level"] = float(np.mean(np.stack(frames) @ M[0]))
    return out
