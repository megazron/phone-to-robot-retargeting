"""Mug-centric hand tracks from top-down, handheld phone clips.

The phone drifts and tilts, so pixel coordinates are not a fixed frame. The black mug is: it is
tracked per frame as a dark circle, and its rim radius in pixels (rim diameter ~8 cm) gives the
metric scale of that frame. The pinch point (midpoint of thumb and index tips) is then expressed
in a mug-centred table frame: x right, y away from the camera bottom edge, z up.

Height is the weak axis. With a top-down camera it comes from the apparent hand size:
size is proportional to 1 / distance, so with L_ref the hand size when it touches the table
(median over grasp frames) and D the camera-to-table distance,
    z = D (1 - L_ref / L).
D is estimated from the mug radius and an assumed focal length, so z is approximate (roughly
+-30%). Contact-anchored retargeting only needs z to have the right shape, not the right scale.
"""
import cv2
import numpy as np
from scipy.ndimage import median_filter, uniform_filter1d
from scipy.signal import savgol_filter

from .hand import INDEX_TIP, THUMB_TIP, _fill_nans, focal_px

MUG_RIM_R = 0.04        # m, outer rim radius of the mug
MUG_HEIGHT = 0.095      # m
SOURCE_LONG_SIDE = 1280  # px, long side of the original phone video the crops were cut from
PALM = [(0, 5), (0, 9), (0, 17), (5, 17), (5, 9), (9, 13), (13, 17)]


def find_mug(frame_rgb, prev=None):
    g = cv2.GaussianBlur(cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2GRAY), (9, 9), 0)
    m = (g < 55).astype(np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((7, 7), np.uint8))
    cs, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    best = None
    for c in cs:
        a = cv2.contourArea(c)
        if a < 3000:
            continue
        (x, y), r = cv2.minEnclosingCircle(c)
        circ = a / (np.pi * r * r)
        if circ < 0.55 or r > 160:
            continue
        d = 0 if prev is None else np.hypot(x - prev[0], y - prev[1])
        if prev is not None and d > 80:
            continue
        score = d - 50 * circ
        if best is None or score < best[0]:
            best = (score, x, y, r)
    return None if best is None else np.array(best[1:])


def track_mug(frames):
    out = np.full((len(frames), 3), np.nan)
    prev = None
    for i, f in enumerate(frames):
        m = find_mug(f, prev)
        if m is not None:
            out[i] = m
            prev = m[:2]
    if np.isnan(out[:, 0]).all():
        raise ValueError("mug not found in any frame")
    out = _fill_nans(out)
    for k in range(3):   # the hand occludes the rim during drops; median then box filter
        out[:, k] = uniform_filter1d(median_filter(out[:, k], 15, mode="nearest"), 15, mode="nearest")
    return out


def palm_size(img):
    return np.sum([np.linalg.norm(img[:, a] - img[:, b], axis=-1) for a, b in PALM], axis=0)


def lift_topdown(raw, mug):
    """raw: hand.detect() output for a clip; mug: (T, 3) x, y, r in pixels."""
    img, world = raw["img"], raw["world"]
    T = len(img)
    valid = np.isfinite(img[:, 0, 0])
    if valid.mean() < 0.3:
        raise ValueError(f"hand found in only {valid.mean():.0%} of frames")
    f = focal_px(SOURCE_LONG_SIDE)
    d_rim = f * MUG_RIM_R / mug[:, 2]              # camera to rim plane
    d_table = d_rim + MUG_HEIGHT
    m_per_px = (MUG_RIM_R / mug[:, 2]) * d_table / d_rim   # at table level

    pinch_px = 0.5 * (img[:, THUMB_TIP] + img[:, INDEX_TIP])
    xy = (pinch_px - mug[:, :2]) * m_per_px[:, None]
    xy[:, 1] *= -1                                 # image y down -> table y up the image
    aperture = np.linalg.norm(world[:, THUMB_TIP] - world[:, INDEX_TIP], axis=1)
    size = palm_size(img)

    xy = _fill_nans(xy)
    aperture = _fill_nans(aperture[:, None])[:, 0]
    size = _fill_nans(size[:, None])[:, 0]
    win = max(5, int(raw["fps"] * 0.3) | 1)
    xy = savgol_filter(xy, win, 2, axis=0)
    aperture = savgol_filter(aperture, win, 2)
    size = savgol_filter(size, 2 * win + 1, 2)
    return {"t": np.arange(T) / raw["fps"], "xy": xy, "aperture": aperture, "size": size,
            "valid": valid, "img": _fill_nans(img.reshape(T, -1)).reshape(T, 21, 2),
            "mug": mug, "d_table": d_table, "fps": raw["fps"], "width": raw["width"], "height": raw["height"]}


def split_episodes(tk, near_mug=0.09, min_reach=0.12, pad=1.0):
    """Pick-and-drop episodes from mug proximity plus aperture.

    Release: the closest approach to the mug in each interval where the pinch point is within
    near_mug of the mug axis. Grasp: between the previous release and this one, the frame with
    the smallest thumb-index aperture among frames where the hand is out at the disc (farther
    than 60% of that segment's maximum distance). The aperture alone is too noisy in a
    top-down view (the fingers occlude each other), the distance signal is very clean."""
    fps = tk["fps"]
    dist = np.linalg.norm(tk["xy"], axis=1)
    near = (dist < near_mug) & tk["valid"]
    edges = np.flatnonzero(np.diff(np.r_[0, near.astype(int), 0]))
    releases = [s + int(np.argmin(dist[s:e])) for s, e in zip(edges[::2], edges[1::2]) if e - s >= 0.15 * fps]
    eps, prev = [], 0
    for k, r in enumerate(releases):
        w = np.arange(prev, r)
        if len(w) < fps or dist[w].max() < min_reach:
            prev = r
            continue
        far = w[(dist[w] > 0.6 * dist[w].max()) & tk["valid"][w]]
        if len(far) == 0:
            prev = r
            continue
        g = int(far[np.argmin(tk["aperture"][far])])
        start = max(prev, g - int(2.5 * fps))
        nxt = releases[k + 1] if k + 1 < len(releases) else len(dist)
        end = min(r + int(pad * fps), nxt - 1, len(dist) - 1)
        eps.append(make_episode(tk, start, end, g, r))
        prev = r
    return eps


def make_episode(tk, start, end, grasp, release):
    sl = slice(start, end + 1)
    size = tk["size"][sl]
    L_ref = np.median(tk["size"][max(grasp - 3, 0):grasp + 4])     # hand at the table at grasp
    D = np.median(tk["d_table"][sl])
    z = np.clip(D * (1 - L_ref / size), 0.0, 0.4)
    wrist = np.c_[tk["xy"][sl], z]
    return {"t": tk["t"][sl] - tk["t"][start], "wrist": wrist, "aperture": tk["aperture"][sl],
            "img": tk["img"][sl], "fps": tk["fps"], "width": tk["width"], "height": tk["height"],
            "up": np.array([0.0, 0.0, 1.0]), "frame0": start, "grasp": grasp - start,
            "release": release - start}
