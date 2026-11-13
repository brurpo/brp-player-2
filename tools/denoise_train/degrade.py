"""Random degradation chain for the stable denoiser (PLAN.md §4).

    from degrade import degrade
    out = degrade(clean, rng, kind="camera")    # clean: (T, H, W, 3) float32, linear light
    out.noisy, out.sigma, out.steps

`clean` is a short clip in linear light (SDR footage decoded to display-linear with
white at 1.0; renders as rendered, values above 1 allowed). The degraded clip comes
back in the same space, with a noise map (`sigma`: local RMS of the damage in the
display encoding, per frame, what the player's measured profile approximates) and the
list of steps taken, for the gallery and debugging.

Where each step happens (§3): sensor noise, render noise, fireflies and impulses in
linear light; grain, blotches, aliasing, banding and compression in the display
encoding (sRGB curve, or PQ with 1.0 = 203 nits for HDR), then decoded back.

Settings are drawn once per clip; the noise itself is fresh every frame, except
blotches (partly carried from frame to frame) and hot pixels (fixed). Kinds:
  camera   footage: sensor noise, in-camera NR blotches, grain, compression...
  render   3D renders: path-tracing speckle, fireflies, salt and pepper
  any      either, at random
About 5% of clips come back nearly clean, so the network also learns to leave clean
pictures alone.
"""
from dataclasses import dataclass, field
import io
import os

os.environ.setdefault("SVT_LOG", "1")  # SVT-AV1: errors only

import cv2
import numpy as np

# --- transfer curves -------------------------------------------------------------


# The curves use OpenCV's vectorised pow (about 2x NumPy's, same results to 1e-5):
# they run on every sample, several times.
def _pow(x, e):
    x = np.ascontiguousarray(x, np.float32)
    return cv2.pow(x.reshape(-1, 1), float(e)).reshape(x.shape)


def srgb_oetf(x):
    x = np.maximum(x, 0, dtype=np.float32)
    return np.where(x <= 0.0031308, np.float32(12.92) * x, np.float32(1.055) * _pow(x, 1 / 2.4) - np.float32(0.055))


def srgb_eotf(v):
    v = np.maximum(v, 0, dtype=np.float32)
    return np.where(v <= 0.04045, v / np.float32(12.92), _pow((v + np.float32(0.055)) / np.float32(1.055), 2.4))


_M1, _M2, _C1, _C2, _C3 = 0.1593017578125, 78.84375, 0.8359375, 18.8515625, 18.6875
WHITE_NITS = 203.0


def pq_oetf(x):
    y = _pow(np.clip(np.asarray(x, np.float32) * np.float32(WHITE_NITS / 10000.0), 0, 1), _M1)
    return _pow((np.float32(_C1) + np.float32(_C2) * y) / (1 + np.float32(_C3) * y), _M2)


def pq_eotf(v):
    p = _pow(np.clip(np.asarray(v, np.float32), 0, 1), 1 / _M2)
    return _pow(np.maximum(p - np.float32(_C1), 0) / (np.float32(_C2) - np.float32(_C3) * p), 1 / _M1) * np.float32(10000.0 / WHITE_NITS)


LUMA = np.array([0.2126, 0.7152, 0.0722], np.float32)

# --- helpers ---------------------------------------------------------------------


def blur(x, s):
    """Gaussian blur of (H, W) or (H, W, C) float32."""
    if s <= 0.05:
        return x
    return cv2.GaussianBlur(x, (0, 0), s, borderType=cv2.BORDER_REFLECT)


def shaped_noise(rng, shape, size, laplace=False):
    """Unit-variance noise with grain size `size` px (Gaussian-filtered white noise)."""
    # float32 straight from the generator (twice as fast as float64 then cast).
    n = rng.laplace(0, 1 / np.sqrt(2), shape).astype(np.float32) if laplace else rng.standard_normal(shape, dtype=np.float32)
    if size > 0.6:
        n = blur(n, size * 0.5)
        n /= n.std() + 1e-8
    return n


def colour_mix(rng, h, w, size, colour, laplace=False):
    """(H, W, 3) noise: `colour` 0 = monochrome, 1 = independent per channel."""
    m = shaped_noise(rng, (h, w), size, laplace)[..., None]
    if colour <= 0:
        return np.repeat(m, 3, 2)
    e = np.stack([shaped_noise(rng, (h, w), size, laplace) for _ in range(3)], 2)
    return np.sqrt(1 - colour) * m + np.sqrt(colour) * e


def lu(r, a, b):
    """Log-uniform."""
    return float(np.exp(r.uniform(np.log(a), np.log(b))))


@dataclass
class Out:
    noisy: np.ndarray
    sigma: np.ndarray
    steps: list = field(default_factory=list)


# --- linear-light steps ----------------------------------------------------------


CCM_K = np.array([[0.6, -0.4, -0.2], [-0.2, 0.5, -0.3], [0.0, -0.5, 0.5]], np.float32)


def sensor_noise(r, clip):
    """Camera noise through a camera's processing (see recipes.sensor_settings): shot
    + read noise per raw channel, times the white balance gains, mixed by a colour-
    correction matrix (strong chroma noise along magenta-teal), softened by
    demosaicing; sometimes in-camera chroma NR."""
    gain = lu(r, 2e-4, 1e-2)
    read = lu(r, 1e-4, 8e-3)
    k = CCM_K.copy()
    for i in range(3):
        for j in range(3):
            if i != j:
                k[i, j] += r.uniform(-0.25, 0.25)
        k[i, i] = -sum(k[i, j] for j in range(3) if j != i)
    spread = r.uniform(0.2, 0.9)
    nr = r.random() < 0.3
    nr_s = r.uniform(1.5, 4)
    wb = np.array([r.uniform(1.0, 3.0), 1.0, r.uniform(1.0, 3.0)], np.float32)
    ccm = np.eye(3, dtype=np.float32) + r.uniform(0.3, 1.5) * k
    sat = r.uniform(1.0, 2.2)
    persist = r.uniform(0.0, 0.3)
    T, H, W, _ = clip.shape
    out = np.empty_like(clip)
    prev = None
    for t in range(T):
        x = clip[t]
        raw = np.maximum(x, 0) / wb
        sd = np.sqrt(raw * gain + read * read)
        fresh = r.standard_normal((H, W, 3), dtype=np.float32)
        if prev is not None and persist > 0:
            fresh = persist * prev + np.sqrt(1 - persist * persist) * fresh
        prev = fresh
        n = (fresh * sd * wb) @ ccm.T
        y = (n @ LUMA)[..., None]
        n = y + sat * (n - y)
        if spread > 0.1:
            n = blur(n, spread) * (1 + spread)
        if nr:
            y = (n @ LUMA)[..., None]
            n = y + blur(n - y, nr_s)
        out[t] = x + n
    return out, f"sensor gain {gain:.1e} read {read:.1e} wb {wb[0]:.1f}/{wb[2]:.1f}" + (" +chromaNR" if nr else "")


def render_noise(r, clip):
    """Path-tracing-like noise: each pixel is the mean of N samples (N 1-64, log-
    uniform). A sample misses the light with probability `miss` (dark pixels at low N),
    otherwise carries a heavy-tailed (log-normal) weight; the mean of each sample is
    the clean value, so the estimate is unbiased as in a renderer. Channels share part
    of each sample (colour vs grey speckle). Above 8 samples the mean is drawn from
    its normal approximation, with the same variance."""
    n = int(round(lu(r, 1, 64)))
    miss = r.uniform(0.0, 0.85)
    s = r.uniform(0.3, 1.5)  # log-std of a sample's weight
    colour = r.uniform(0.1, 0.8)
    m2 = np.exp(s * s) / (1 - miss)  # E[w^2] of one sample, E[w] = 1
    T, H, W, _ = clip.shape
    out = np.empty_like(clip)
    for t in range(T):
        if n <= 8:
            acc = np.zeros((H, W, 3), np.float32)
            for _ in range(n):
                hit = (r.random((H, W, 1)) >= miss).astype(np.float32)
                g = colour_mix(r, H, W, 0.0, colour)
                acc += hit * np.exp(s * g - s * s / 2) / (1 - miss)
            w = acc / n
        else:
            w = 1 + np.sqrt((m2 - 1) / n) * colour_mix(r, H, W, 0.0, colour)
            w = np.maximum(w, 0)
        out[t] = clip[t] * w
    return out, f"render {n} spp miss {miss:.2f} spread {s:.2f}"


def fireflies(r, clip):
    """Rare pixels far brighter than the scene (HDR): 10-1000x local light."""
    rate = lu(r, 2e-5, 2e-3)
    T, H, W, _ = clip.shape
    out = clip.copy()
    base = max(float(np.mean(clip)), 0.05)
    for t in range(T):
        m = r.random((H, W)) < rate
        k = m.sum()
        if k:
            v = base * np.exp(r.uniform(np.log(10), np.log(1000), k)).astype(np.float32)
            tint = r.uniform(0.5, 1.5, (k, 3)).astype(np.float32)
            out[t][m] += v[:, None] * tint
    return out, f"fireflies {rate:.0e}"


def salt_pepper(r, clip, hdr):
    """Isolated white/black pixels (1 or 2 px), fresh each frame; optionally coloured
    (single channel stuck) and, for renders, salt brighter than white."""
    rate = lu(r, 1e-4, 1e-2)
    salt_frac = r.uniform(0.2, 0.8)
    coloured = r.random() < 0.3
    cluster = r.random() < 0.3
    hi = lu(r, 1.0, 50.0) if hdr else 1.0
    T, H, W, _ = clip.shape
    out = clip.copy()
    for t in range(T):
        m = r.random((H, W)) < rate
        if cluster:
            m |= np.roll(m & (r.random((H, W)) < 0.5), 1, int(r.integers(2)))
        salt = m & (r.random((H, W)) < salt_frac)
        pep = m & ~salt
        if coloured:
            ch = r.integers(0, 3, (H, W))
            for c in range(3):
                out[t][..., c][salt & (ch == c)] = hi
                out[t][..., c][pep & (ch == c)] = 0
        else:
            out[t][salt] = hi
            out[t][pep] = 0
    return out, f"salt&pepper {rate:.0e}" + (" colour" if coloured else "") + (" 2px" if cluster else "")


def hot_pixels(r, clip, hdr):
    """Fixed-position stuck pixels (same in every frame)."""
    rate = lu(r, 2e-5, 5e-4)
    T, H, W, _ = clip.shape
    m = r.random((H, W)) < rate
    v = r.uniform(0.5, 1.0, (int(m.sum()), 3)).astype(np.float32) * (lu(r, 1, 20) if hdr else 1)
    out = clip.copy()
    for t in range(T):
        out[t][m] = np.maximum(out[t][m], v)
    return out, f"hot pixels {rate:.0e}"


# --- display-encoded steps (values ~0..1) ----------------------------------------


def grain(r, v):
    size = lu(r, 0.5, 4.0)
    amp = lu(r, 0.004, 0.06)
    colour = r.choice([0.0, r.uniform(0.1, 1.0)])
    laplace = r.random() < 0.4
    sparse = r.random() < 0.25
    clump = r.random() < 0.3
    T, H, W, _ = v.shape
    out = np.empty_like(v)
    for t in range(T):
        n = colour_mix(r, H, W, size, colour, laplace)
        if sparse:  # few, stronger specks
            k = np.abs(n[..., :1]) > r.uniform(1.0, 2.0)
            n = n * k * 2.0
        if clump:  # grain stronger in patches
            n *= 0.4 + 1.2 * np.clip(blur(r.random((H, W)).astype(np.float32), size * 6) * 3 - 1, 0, 1)[..., None]
        out[t] = v[t] + amp * n
    return out, f"grain {amp:.3f} size {size:.1f}" + (" colour" if colour else " mono") + (" sparse" if sparse else "") + (" clumped" if clump else "")


SS = 4  # film grain super-sampling (as gpu_degrade.film_grain)
RADIUS_SCALE = (1.0, 0.9, 1.3)  # R, G, B dye layers


def _boolean_grain(r, u, radius, spread):
    """Boolean-model grain layer (see gpu_degrade.py): u (H, W) in [0, 1] -> covered
    fraction per pixel. Discs far smaller than a pixel, dropped by a Poisson process
    whose density follows u, rendered at SS x and averaged down."""
    H, W = u.shape
    up = np.clip(cv2.resize(u, (W * SS, H * SS), interpolation=cv2.INTER_LINEAR), 0, 0.995)
    radii = [radius * SS] if spread <= 0.02 else [radius * SS * (1 - spread), radius * SS * (1 + spread)]
    hits = np.zeros_like(up)
    for rr in radii:
        R = max(1, int(np.ceil(rr)))
        t = np.arange(-R, R + 1)
        k = ((t[:, None] ** 2 + t[None, :] ** 2) <= max(rr, 0.5) ** 2).astype(np.float32)
        lam = -np.log1p(-up) / (k.sum() * len(radii))
        pts = r.poisson(lam).astype(np.float32)
        hits += cv2.filter2D(pts, -1, k, borderType=cv2.BORDER_CONSTANT)
    return cv2.resize((hits > 0).astype(np.float32), (W, H), interpolation=cv2.INTER_AREA)


def film_grain(r, v):
    """Film grain, Boolean model (Newson et al. 2017): grains far smaller than a pixel,
    the visible grain is their clumping, sized by the scan blur. Black and white (one
    luma layer) or colour (each dye layer's grain mixed with a shared luma grain,
    `chroma` setting how colourful). Negative film (70%): modelled in linear light, so
    on screen strongest in the shadows; otherwise on display values (mid-tones)."""
    radius = lu(r, 0.05, 0.25)
    spread = r.uniform(0, 0.4)
    mono = r.random() < 0.3
    sblur = r.uniform(0.5, 1.2)
    strength = r.uniform(0.4, 1.0)
    chroma = r.uniform(0.15, 0.6)
    neg = r.random() < 0.7
    T, H, W, _ = v.shape
    out = np.empty_like(v)
    for t in range(T):
        x = np.clip(v[t], 0, 1)
        if neg:
            x = srgb_eotf(x)
        y = np.ascontiguousarray(x @ LUMA)
        gy = blur(_boolean_grain(r, y, radius, spread), sblur) - blur(y, sblur)
        if mono:
            o = x + strength * gy[..., None]
        else:
            o = np.empty_like(x)
            for c in range(3):
                u = np.ascontiguousarray(x[..., c])
                gc = blur(_boolean_grain(r, u, radius * RADIUS_SCALE[c], spread), sblur) - blur(u, sblur)
                o[..., c] = u + strength * ((1 - chroma) * gy + chroma * gc)
        if neg:
            o = srgb_oetf(o)
        out[t] = o + (v[t] - np.clip(v[t], 0, 1))
    return out, f"film grain r {radius:.2f} px blur {sblur:.1f}" + (" b&w" if mono else "") + (" negative" if neg else "")


def blotches(r, v):
    """Low-frequency colour blobs (heavy in-camera NR + compression), strongest in the
    shadows, partly carried from frame to frame."""
    size = lu(r, 4, 32)  # blur sigma: blobs ~8-64 px
    amp = lu(r, 0.01, 0.06)
    keep = r.uniform(0, 0.95)  # frame-to-frame persistence
    luma_part = r.uniform(0, 0.5)
    shadow_pow = r.uniform(1, 4)
    T, H, W, _ = v.shape
    out = np.empty_like(v)
    prev = None
    for t in range(T):
        n = colour_mix(r, H, W, size * 2, 1.0)
        n = n - (1 - luma_part) * (n @ LUMA)[..., None]
        n /= n.std() + 1e-8
        prev = n if prev is None else keep * prev + np.sqrt(1 - keep * keep) * n
        y = np.clip(v[t] @ LUMA, 0, 1)[..., None]
        out[t] = v[t] + amp * (1 - y) ** shadow_pow * prev
    return out, f"blotches {amp:.3f} size {size:.0f} keep {keep:.2f}"


def _fill_field(f, parity, smooth):
    """Keep the lines of one field, rebuild the others: averaged from the lines above
    and below (`smooth`, a plain bob) or repeated (line doubling)."""
    H = f.shape[0]
    out = f.copy()
    for y in range(1 - parity, H, 2):
        up = y - 1 if y - 1 >= 0 else y + 1
        dn = y + 1 if y + 1 < H else y - 1
        out[y] = 0.5 * (f[up] + f[dn]) if smooth else f[up]
    return out


def _half_way(a, b):
    """Picture half way from a to b (a warped by half the optical flow), and the
    length of that half motion per pixel."""
    g = lambda x: cv2.cvtColor((np.clip(x, 0, 1) * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
    dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_ULTRAFAST)
    flow = dis.calc(g(a), g(b), None)
    H, W = flow.shape[:2]
    gx, gy = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32))
    mid = cv2.remap(a, gx - flow[..., 0] / 2, gy - flow[..., 1] / 2, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    return mid, np.hypot(flow[..., 0], flow[..., 1]) / 2


def aliasing(r, v):
    """Aliasing and field problems as they reach a denoiser: what is left after a
    deinterlacer, not raw interlaced video (heavy combing is the deinterlacer's job).
      decimate    unfiltered downscale, smooth upscale
      nearest     nearest-neighbour rescale
      weave       motion-adaptive deinterlacer leftovers: the other field comes from
                  half a frame later (a field's time), woven in where that motion is
                  small (a few px of combing on edges); where it is larger the lines
                  are rebuilt from one field (jaggies)
      bob         one field per frame, alternating, lines rebuilt: jaggies plus a
                  one-line up/down jitter on fine horizontal detail
      linedouble  one field, lines repeated, same field every frame"""
    T, H, W, _ = v.shape
    mode = str(r.choice(["decimate", "nearest", "weave", "weave", "bob", "linedouble"]))
    out = np.empty_like(v)
    if mode == "decimate":
        f = int(r.integers(2, 5))
        up = cv2.INTER_LINEAR if r.random() < 0.5 else cv2.INTER_CUBIC
        for t in range(T):
            out[t] = cv2.resize(np.ascontiguousarray(v[t][::f, ::f]), (W, H), interpolation=up)
        return out, f"aliasing: decimate x{f}"
    if mode == "nearest":
        s = r.uniform(0.4, 0.8)
        for t in range(T):
            small = cv2.resize(v[t], (max(8, int(W * s)), max(8, int(H * s))), interpolation=cv2.INTER_NEAREST)
            out[t] = cv2.resize(small, (W, H), interpolation=cv2.INTER_NEAREST)
        return out, f"aliasing: nearest x{s:.2f}"
    first = int(r.integers(2))
    if mode == "weave":
        lim = r.uniform(1.0, 4.0)  # half-frame motion (px) still woven in
        smooth = r.random() < 0.7
        for t in range(T):
            nb = v[t + 1] if t + 1 < T else v[t - 1]
            mid, motion = _half_way(v[t], nb)
            bobbed = _fill_field(v[t], first, smooth)
            m = blur((motion <= lim).astype(np.float32), 1.0)[..., None]
            woven = v[t].copy()
            woven[1 - first :: 2] = mid[1 - first :: 2]
            out[t] = m * woven + (1 - m) * bobbed
        return out, f"aliasing: weave (motion-adaptive, <= {lim:.1f} px)"
    if mode == "bob":
        smooth = r.random() < 0.7
        for t in range(T):
            out[t] = _fill_field(v[t], (t + first) % 2, smooth)
        return out, "aliasing: bob" + ("" if smooth else " (repeated lines)")
    for t in range(T):
        out[t] = _fill_field(v[t], first, False)
    return out, "aliasing: line doubling"


def banding(r, v):
    bits = int(r.integers(5, 9))
    q = 2**bits - 1
    return np.round(np.clip(v, 0, 1) * q) / q, f"banding {bits} bit"


# Codec: (pixel format, options from a random quality draw)
CODECS = {
    "libx264": lambda r: ("yuv420p", {"crf": str(int(r.integers(18, 41))), "preset": str(r.choice(["ultrafast", "superfast", "veryfast"]))}),
    "libx265": lambda r: (r.choice(["yuv420p", "yuv420p10le"]), {"crf": str(int(r.integers(20, 41))), "preset": str(r.choice(["ultrafast", "superfast"])), "x265-params": "log-level=none:pools=none:frame-threads=1"}),
    "mpeg2video": lambda r: ("yuv420p", {"qmin": (q := str(int(r.integers(2, 21)))), "qmax": q}),
    "mpeg4": lambda r: ("yuv420p", {"qmin": (q := str(int(r.integers(2, 21)))), "qmax": q}),
    "mjpeg": lambda r: ("yuvj420p", {"qmin": (q := str(int(r.integers(2, 21)))), "qmax": q}),
    "libvpx-vp9": lambda r: ("yuv420p", {"crf": str(int(r.integers(20, 55))), "b": "0", "deadline": "realtime", "cpu-used": "8"}),
    "libsvtav1": lambda r: ("yuv420p", {"crf": str(int(r.integers(25, 55))), "preset": "12", "svtav1-params": "lp=1"}),
    "prores_ks": lambda r: ("yuv422p10le", {"profile": "0", "qscale": str(int(r.integers(4, 32)))}),
    "dvvideo": lambda r: ("yuv420p", {}),
    # DNxHR: broadcast editing codec, 8x8 DCT at a fixed bitrate, 4:2:2. On noisy dark
    # footage it runs out of bits: 8x8 block mosaics (as on the black-curtain clip).
    "dnxhd": lambda r: ("yuv422p", {"profile": str(r.choice(["dnxhr_lb", "dnxhr_sq"]))}),
}
WEIGHTS = {"libx264": 4, "libx265": 2, "mpeg2video": 2, "mpeg4": 1, "mjpeg": 1.5, "libvpx-vp9": 1, "libsvtav1": 1, "prores_ks": 0.5, "dvvideo": 0.5, "dnxhd": 1}
# Double compression: a camera codec, then an editing / broadcast one.
CAMERA_CODECS = ("libx264", "libx265")
EDIT_CODECS = ("dnxhd", "prores_ks", "mpeg2video")


FAILURES = {}  # codec -> count of encodes that failed and fell back to H.264


def compress(r, v, codec=None):
    """Encode and decode the clip with a real encoder (PyAV) at a random quality and
    GOP. Frames are padded to what the codec needs and cropped back. An encoder error
    falls back to H.264 (counted in FAILURES) instead of stopping training."""
    names = list(WEIGHTS)
    p = np.array([WEIGHTS[n] for n in names], float)
    codec = codec or str(r.choice(names, p=p / p.sum()))
    try:
        return _encode(r, v, codec)
    except Exception as e:
        FAILURES[codec] = FAILURES.get(codec, 0) + 1
        if codec == "libx264":
            raise
        out, s = _encode(r, v, "libx264")
        return out, f"{s} (fallback: {codec} {type(e).__name__})"


# RGB <-> Y'CbCr done here, exactly (BT.709, rounded once), instead of by the library:
# its quick conversion lost about 1 level of 255 on every round trip, green most, with
# any codec. The network learned to give that back and so brightened real footage
# (+1.6 levels, greenish). Colour-space conversion is not damage a codec does.
_KR, _KB = 0.2126, 0.0722
_KG = 1.0 - _KR - _KB


def _rgb_to_planes(v, pix):
    """(T, H, W, 3) display values 0..1 -> per frame the three planes of `pix` (planar
    Y'CbCr: 4:2:0 or 4:2:2, 8 or 10 bit, "yuvj" = full range), as integer arrays."""
    bits = 10 if "10" in pix else 8
    full = pix.startswith("yuvj")
    y = _KR * v[..., 0] + _KG * v[..., 1] + _KB * v[..., 2]
    cb = (v[..., 2] - y) / (2 * (1 - _KB))
    cr = (v[..., 0] - y) / (2 * (1 - _KR))
    if "420" in pix:  # chroma: the mean of each 2x2 (4:2:0) or 1x2 (4:2:2) block
        sub = lambda c: c.reshape(c.shape[0], c.shape[1] // 2, 2, c.shape[2] // 2, 2).mean((2, 4))
    else:
        sub = lambda c: c.reshape(c.shape[0], c.shape[1], c.shape[2] // 2, 2).mean(3)
    cb, cr = sub(cb), sub(cr)
    k = 2 ** (bits - 8)
    top = 2**bits - 1
    if full:
        y, cb, cr = y * top, 128 * k + cb * top, 128 * k + cr * top
    else:
        y, cb, cr = (16 + 219 * y) * k, (128 + 224 * cb) * k, (128 + 224 * cr) * k
    dt = np.uint16 if bits > 8 else np.uint8
    return [np.clip(np.round(a), 0, top).astype(dt) for a in (y, cb, cr)]


def _planes_to_rgb(f, H, W):
    """A decoded frame (planar Y'CbCr of any subsampling and depth) -> (H, W, 3) float32
    display values, or None when the frame is something else."""
    name = f.format.name
    if not (name.startswith("yuv") and f.format.is_planar and len(f.planes) >= 3):
        return None
    bits = f.format.components[0].bits
    dt = np.uint16 if bits > 8 else np.uint8
    pl = []
    for p_ in f.planes[:3]:
        a = np.frombuffer(p_, dt).reshape(p_.height, -1)[:, : p_.width].astype(np.float32)
        pl.append(a)
    k = 2 ** (bits - 8)
    top = 2**bits - 1
    if name.startswith("yuvj"):
        y, cb, cr = pl[0] / top, (pl[1] - 128 * k) / top, (pl[2] - 128 * k) / top
    else:
        y, cb, cr = (pl[0] / k - 16) / 219, (pl[1] / k - 128) / 224, (pl[2] / k - 128) / 224
    if cb.shape != y.shape:
        cb = cv2.resize(cb, (y.shape[1], y.shape[0]), interpolation=cv2.INTER_LINEAR)
        cr = cv2.resize(cr, (y.shape[1], y.shape[0]), interpolation=cv2.INTER_LINEAR)
    r_ = y + 2 * (1 - _KR) * cr
    b_ = y + 2 * (1 - _KB) * cb
    g_ = (y - _KR * r_ - _KB * b_) / _KG
    return np.clip(np.stack([r_, g_, b_], -1)[:H, :W], 0, 1)


def _encode(r, v, codec):
    import av
    from fractions import Fraction

    pix, opts = CODECS[codec](r)
    T, H, W, _ = v.shape
    if codec == "dvvideo":
        PW, PH = 720, 576
        if W > PW or H > PH:
            return _encode(r, v, "mpeg2video")
    else:
        m = 16 if codec in ("libsvtav1", "mjpeg", "dnxhd") else 2
        PW, PH = max(64, -(-W // m) * m), max(64, -(-H // m) * m)
    u8 = np.round(np.clip(v, 0, 1) * 255).astype(np.uint8)
    if (PW, PH) != (W, H):
        u8 = np.pad(u8, ((0, 0), (0, PH - H), (0, PW - W), (0, 0)), mode="reflect" if PH - H < H and PW - W < W else "edge")
    buf = io.BytesIO()
    fmt = {"dvvideo": "dv", "dnxhd": "mov"}.get(codec, "matroska")
    with av.open(buf, "w", format=fmt) as c:
        s = c.add_stream(codec, rate=25)
        s.width, s.height, s.pix_fmt = PW, PH, pix
        s.time_base = Fraction(1, 25)
        gop = int(r.choice([1, 4, 12, 50]))
        s.codec_context.gop_size = gop
        # One thread per encoder: the data workers already fill every CPU thread.
        s.codec_context.thread_count = 1
        s.options = opts
        planes = _rgb_to_planes(u8.astype(np.float32) / 255.0, pix)
        for t in range(T):
            fr = av.VideoFrame(PW, PH, pix)
            for p_, a in zip(fr.planes, planes):
                row = np.zeros((p_.height, p_.line_size), np.uint8)
                b_ = a[t].view(np.uint8).reshape(p_.height, -1)
                row[:, : b_.shape[1]] = b_
                p_.update(row.tobytes())
            fr.pts, fr.time_base = t, Fraction(1, 25)
            for pkt in s.encode(fr):
                c.mux(pkt)
        for pkt in s.encode():
            c.mux(pkt)
    buf.seek(0)
    frames = []
    with av.open(buf, "r") as c:
        for f in c.decode(video=0):
            rgb = _planes_to_rgb(f, H, W)
            frames.append(rgb if rgb is not None else f.to_ndarray(format="rgb24")[:H, :W].astype(np.float32) / 255.0)
    if len(frames) != T:
        raise RuntimeError(f"{codec}: {len(frames)} frames back of {T}")
    q = opts.get("crf") or opts.get("qmin") or opts.get("qscale") or opts.get("profile") or "-"
    return np.stack(frames).astype(np.float32), f"{codec} q {q} gop {gop}" + (" 10bit" if "10" in pix else "")


# --- the chain -------------------------------------------------------------------


def degrade(clean, rng, kind="any", hdr=None, strength=1.0):
    """Degrade a clean clip (T, H, W, 3) float32 in linear light. `hdr`: encode with PQ
    (default: when the clip goes above 1.0). `strength` scales the probability of
    every step (0.5 = gentler mixes)."""
    r = rng
    clean = np.ascontiguousarray(clean, np.float32)
    if kind == "any":
        kind = "render" if r.random() < 0.3 else "camera"
    if hdr is None:
        hdr = bool(clean.max() > 1.0)
    enc, dec = (pq_oetf, pq_eotf) if hdr else (srgb_oetf, srgb_eotf)
    P = lambda p: r.random() < min(1.0, p * strength)
    steps = []
    x = clean
    if r.random() < 0.05:  # nearly clean: untouched, or only lightly compressed
        steps.append("clean")
        if r.random() < 0.5:
            v, s = compress(r, enc(x))
            steps.append(s)
            x = dec(v)
        return _finish(clean, x, enc, steps)

    # linear light
    if kind == "camera":
        if P(0.75):
            x, s = sensor_noise(r, x)
            steps.append(s)
    else:
        if P(0.7):
            x, s = render_noise(r, x)
            steps.append(s)
        if P(0.5):
            x, s = fireflies(r, x)
            steps.append(s)
    if P(0.35 if kind == "render" else 0.12):
        x, s = salt_pepper(r, x, hdr)
        steps.append(s)
    if P(0.03):
        x, s = hot_pixels(r, x, hdr)
        steps.append(s)

    # display encoding
    v = enc(x)
    disp = [
        (grain, 0.45 if kind == "camera" else 0.15),
        (film_grain, 0.12 if kind == "camera" else 0.03),
        (blotches, 0.3 if kind == "camera" else 0.05),
        (aliasing, 0.15),
    ]
    order = r.permutation(len(disp))
    for i in order:
        f, p = disp[i]
        if P(p):
            v, s = f(r, v)
            steps.append(s)
    if P(0.08):
        v, s = banding(r, v)
        steps.append(s)
    if P(0.65 if kind == "camera" else 0.3):
        v, s = compress(r, v)
        steps.append(s)
    if len(steps) == 0:  # never return untouched by accident
        v, s = grain(r, v)
        steps.append(s)
    return _finish(clean, dec(v), enc, steps)


def _finish(clean, noisy, enc, steps):
    d = enc(noisy) - enc(clean)
    sig = np.stack([np.sqrt(blur(np.mean(f * f, 2), 4.0)) for f in d])[..., None]
    return Out(noisy.astype(np.float32), sig.astype(np.float32), steps)
