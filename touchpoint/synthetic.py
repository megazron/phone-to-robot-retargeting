"""Synthetic stand-in for MediaPipe output, used only for tests and for the pipeline check
before real clips exist. Produces the same raw dict as hand.detect() plus a rendered
skeleton video, for a scripted pick-and-place seen by a propped-up phone at a 3/4 view.
"""
import numpy as np

from .hand import HAND_EDGES, focal_px

W, H, FPS = 960, 540, 30.0

# hand-local frame: x along the fingers, y towards the thumb, z out of the back of the hand
_BASE = np.array([
    [0, 0, 0],
    [0.025, 0.03, -0.01], [0.045, 0.048, -0.02], [0.062, 0.058, -0.03], [0.075, 0.062, -0.04],
    [0.09, 0.025, 0], [0.12, 0.028, -0.015], [0.135, 0.029, -0.03], [0.145, 0.03, -0.045],
    [0.09, 0.0, 0], [0.125, 0.0, -0.015], [0.145, 0.0, -0.035], [0.155, 0.0, -0.05],
    [0.085, -0.02, 0], [0.115, -0.022, -0.02], [0.13, -0.023, -0.04], [0.135, -0.023, -0.055],
    [0.075, -0.04, 0], [0.098, -0.044, -0.02], [0.11, -0.046, -0.035], [0.115, -0.047, -0.048],
], float)


def hand_points(aperture):
    """21 points with thumb-tip/index-tip distance equal to `aperture`."""
    p = _BASE.copy()
    pinch = 0.5 * (p[4] + p[8])
    d = np.array([0.15, 0.95, 0.25])
    d /= np.linalg.norm(d)
    tips = {4: pinch + d * aperture / 2, 8: pinch - d * aperture / 2}
    for tip, mcp in ((4, 1), (8, 5)):
        for k, j in enumerate(range(mcp, tip + 1)):
            s = k / (tip - mcp)
            p[j] = (1 - s) * p[j] + s * tips[tip] if j == tip else p[j] + s**2 * (tips[tip] - _BASE[tip])
    return p


def _minjerk(s):
    s = np.clip(s, 0, 1)
    return 10 * s**3 - 15 * s**4 + 6 * s**5


def _rot(axis, ang):
    axis = np.asarray(axis, float) / np.linalg.norm(axis)
    K = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    return np.eye(3) + np.sin(ang) * K + (1 - np.cos(ang)) * K @ K


def look_at(eye, target):
    """Camera rotation (rows = camera x right, y down, z forward) in table coordinates."""
    z = target - eye
    z /= np.linalg.norm(z)
    x = np.cross(z, [0, 0, 1.0])
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    return np.stack([x, y, z])


def demo_script(rng):
    """Wrist keyframes in a table frame (x away from the person, y left, z up)."""
    obj = np.array([0.28, -0.06, 0.0]) + rng.normal(0, 0.02, 3) * [1, 1, 0]
    mug = np.array([0.30, 0.20, 0.0]) + rng.normal(0, 0.02, 3) * [1, 1, 0]
    rest = np.array([0.05, -0.02, 0.06])
    lift = rng.uniform(0.10, 0.16)
    hz = 0.075                                   # wrist height when fingertips are at the table
    keys = [  # (duration to reach, position, aperture)
        (0.6, rest, 0.065),
        (1.1, obj + [-0.06, 0, hz + 0.07], 0.10),
        (0.6, obj + [-0.02, 0, hz], 0.10),
        (0.35, obj + [-0.02, 0, hz], 0.045),
        (0.9, obj + [-0.02, 0, hz + lift], 0.045),
        (1.0, mug + [-0.02, 0, hz + 0.06 + lift * 0.4], 0.045),
        (0.5, mug + [-0.02, 0, hz + 0.05], 0.045),
        (0.35, mug + [-0.02, 0, hz + 0.05], 0.10),
        (0.9, mug + [-0.12, -0.05, hz + 0.12], 0.07),
        (0.5, mug + [-0.12, -0.05, hz + 0.12], 0.07),
    ]
    speed = rng.uniform(0.85, 1.2)
    times = np.cumsum([0] + [k[0] * speed for k in keys[1:]])
    T = int(times[-1] * FPS) + 1
    t = np.arange(T) / FPS
    pos = np.zeros((T, 3))
    ap = np.zeros(T)
    for i in range(len(keys) - 1):
        m = (t >= times[i]) & (t <= times[i + 1])
        s = _minjerk((t[m] - times[i]) / (times[i + 1] - times[i]))[:, None]
        pos[m] = (1 - s) * keys[i][1] + s * keys[i + 1][1]
        ap[m] = (1 - s[:, 0]) * keys[i][2] + s[:, 0] * keys[i + 1][2]
    held = (t >= times[3]) & (t <= times[7] - 0.15)
    return t, pos, ap, obj, mug, held


def make_clip(seed=0, render=True):
    rng = np.random.default_rng(seed)
    t, pos, ap, obj, mug, held = demo_script(rng)
    eye = np.array([0.20, -0.62, 0.42]) + rng.normal(0, 0.03, 3)
    Rc = look_at(eye, np.array([0.25, 0.08, 0.05]))
    f = focal_px(W) * rng.uniform(0.9, 1.1)          # true focal differs from the assumed one
    # palm faces down and slightly inwards, fingers point away from the person
    R_hand = _rot([0, 0, 1], 0.25) @ _rot([1, 0, 0], np.pi) @ _rot([0, 1, 0], -0.5)
    T = len(t)
    img = np.full((T, 21, 2), np.nan)
    world = np.full((T, 21, 3), np.nan)
    cube_px = np.zeros((T, 2))

    def project(P):
        c = (P - eye) @ Rc.T
        return np.c_[f * c[:, 0] / c[:, 2] + W / 2, f * c[:, 1] / c[:, 2] + H / 2]

    for i in range(T):
        P = pos[i] + hand_points(ap[i]) @ R_hand.T            # table frame
        centre = P.mean(0)
        world[i] = (P - centre) @ Rc.T + rng.normal(0, 0.002, (21, 3))
        img[i] = project(P) + rng.normal(0, 1.5, (21, 2))
        cpos = pos[i] + R_hand @ (0.5 * (_BASE[4] + _BASE[8])) if held[i] else (obj if t[i] < 2 else mug)
        cube_px[i] = project(cpos[None])[0]
    drop = rng.random(T) < 0.05
    drop[T // 3:T // 3 + 4] = True                            # a short tracking dropout
    img[drop] = np.nan
    world[drop] = np.nan
    raw = {"img": img, "world": world, "fps": FPS, "width": W, "height": H}
    frames = _render(img, cube_px, project(np.array([mug])), drop) if render else None
    truth = {"wrist_table": pos, "aperture": ap, "R_cam": Rc, "eye": eye, "f": f}
    return raw, frames, truth


def _render(img, cube_px, mug_px, drop):
    import cv2
    frames = []
    last = None
    for i in range(len(img)):
        fr = np.full((H, W, 3), (205, 196, 182), np.uint8)
        cv2.ellipse(fr, tuple(mug_px[0].astype(int)), (70, 28), 0, 0, 360, (190, 115, 50), -1, cv2.LINE_AA)
        c = cube_px[i].astype(int)
        cv2.rectangle(fr, tuple(c - 14), tuple(c + 14), (45, 55, 215), -1)
        lm = img[i] if not drop[i] else last
        if lm is not None:
            p = lm.astype(int)
            for a, b in HAND_EDGES:
                cv2.line(fr, tuple(p[a]), tuple(p[b]), (150, 170, 220), 9, cv2.LINE_AA)
            last = lm
        frames.append(fr[..., ::-1].copy())
    return np.stack(frames)
