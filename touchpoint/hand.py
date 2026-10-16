"""Phone video -> 3D wrist track + grasp aperture.

detect():  MediaPipe HandLandmarker, per frame 21 image landmarks (px) + 21 world landmarks (m).
lift():    camera-frame wrist position and thumb-index aperture, gap-filled and smoothed.

Depth from a monocular phone clip: MediaPipe's world landmarks give the hand's metric shape
(centred on the hand, camera-aligned axes). With a pinhole camera, each landmark satisfies
    u_i - cx = f (X_i + tx) / (Z_i + tz)
which is linear in the hand translation t = (tx, ty, tz). Solving it over all 21 points is the
"apparent hand size" depth cue, but it uses the whole hand instead of a single palm length, so
it degrades less when the palm is foreshortened. f comes from an assumed phone FOV.
"""
from pathlib import Path

import numpy as np
from scipy.signal import savgol_filter

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "hand_landmarker.task"
PHONE_HFOV_DEG = 66.0   # typical main camera, landscape video; only affects global scale

WRIST, THUMB_TIP, INDEX_TIP, MIDDLE_MCP = 0, 4, 8, 9
HAND_EDGES = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8), (5, 9), (9, 10),
              (10, 11), (11, 12), (9, 13), (13, 14), (14, 15), (15, 16), (13, 17), (0, 17),
              (17, 18), (18, 19), (19, 20)]


def focal_px(width):
    return 0.5 * width / np.tan(np.deg2rad(PHONE_HFOV_DEG) / 2)


def read_video(path, max_side=960):
    import imageio.v3 as iio
    meta = iio.immeta(path)
    fps = float(meta.get("fps", 30.0))
    frames = []
    for fr in iio.imiter(path):
        h, w = fr.shape[:2]
        s = max_side / max(h, w)
        if s < 1:
            import cv2
            fr = cv2.resize(fr, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
        frames.append(fr)
    if not frames:
        raise ValueError(f"no frames decoded from {path}")
    return np.stack(frames), fps


def detect(video_path):
    """Run MediaPipe on every frame. Returns dict of raw arrays (NaN where no hand)."""
    import mediapipe as mp
    from mediapipe.tasks.python import BaseOptions, vision

    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"{MODEL_PATH} missing. Run `make setup` to download it.")
    frames, fps = read_video(video_path)
    T, H, W = frames.shape[:3]
    img = np.full((T, 21, 2), np.nan)
    world = np.full((T, 21, 3), np.nan)
    opts = vision.HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(MODEL_PATH)),
        running_mode=vision.RunningMode.VIDEO, num_hands=1,
        min_hand_detection_confidence=0.3, min_tracking_confidence=0.3)
    with vision.HandLandmarker.create_from_options(opts) as lm:
        for i, fr in enumerate(frames):
            res = lm.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(fr)),
                                      int(i * 1000 / fps))
            if res.hand_landmarks:
                img[i] = [(p.x * W, p.y * H) for p in res.hand_landmarks[0]]
                world[i] = [(p.x, p.y, p.z) for p in res.hand_world_landmarks[0]]
    return {"img": img, "world": world, "fps": fps, "width": W, "height": H}


def solve_translation(img_pts, world_pts, f, cx, cy):
    """Least-squares hand translation in the camera frame (see module docstring)."""
    X, Y, Z = world_pts.T
    du, dv = img_pts[:, 0] - cx, img_pts[:, 1] - cy
    n = len(X)
    A = np.zeros((2 * n, 3))
    b = np.zeros(2 * n)
    A[:n, 0], A[:n, 2], b[:n] = f, -du, du * Z - f * X
    A[n:, 1], A[n:, 2], b[n:] = f, -dv, dv * Z - f * Y
    t, *_ = np.linalg.lstsq(A, b, rcond=None)
    return t


def _fill_nans(x):
    x = x.copy()
    idx = np.arange(len(x))
    for j in range(x.shape[1]):
        ok = np.isfinite(x[:, j])
        if ok.sum() < 2:
            raise ValueError("hand detected in fewer than 2 frames")
        x[:, j] = np.interp(idx, idx[ok], x[ok, j])
    return x


def lift(raw):
    img, world = raw["img"], raw["world"]
    W, H = float(raw["width"]), float(raw["height"])
    f = focal_px(W)
    T = len(img)
    valid = np.isfinite(img[:, 0, 0])
    if valid.mean() < 0.5:
        raise ValueError(f"hand found in only {valid.mean():.0%} of frames; re-record with the hand fully in view")
    wrist = np.full((T, 3), np.nan)
    aperture = np.full(T, np.nan)
    for i in np.flatnonzero(valid):
        t = solve_translation(img[i], world[i], f, W / 2, H / 2)
        wrist[i] = world[i, WRIST] + t
        aperture[i] = np.linalg.norm(world[i, THUMB_TIP] - world[i, INDEX_TIP])
    wrist = _fill_nans(wrist)
    aperture = _fill_nans(aperture[:, None])[:, 0]
    win = max(5, int(raw["fps"] * 0.3) | 1)
    wrist_s = savgol_filter(wrist, win, 2, axis=0)
    aperture_s = savgol_filter(aperture, win, 2)
    img_filled = _fill_nans(img.reshape(T, -1)).reshape(T, 21, 2)
    return {"t": np.arange(T) / raw["fps"], "wrist": wrist_s, "aperture": aperture_s,
            "aperture_raw": aperture, "valid": valid, "img": img_filled, "fps": raw["fps"],
            "width": W, "height": H}


def draw_overlay(frame, img_lm, label=None, color=(60, 230, 120)):
    import cv2
    out = frame.copy()
    p = img_lm.astype(int)
    for a, b in HAND_EDGES:
        cv2.line(out, tuple(p[a]), tuple(p[b]), color, 2, cv2.LINE_AA)
    for q in p:
        cv2.circle(out, tuple(q), 3, (255, 255, 255), -1, cv2.LINE_AA)
    cv2.line(out, tuple(p[THUMB_TIP]), tuple(p[INDEX_TIP]), (240, 80, 60), 2, cv2.LINE_AA)
    if label:
        cv2.rectangle(out, (0, 0), (out.shape[1], 30), (0, 0, 0), -1)
        cv2.putText(out, label, (8, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    return out
