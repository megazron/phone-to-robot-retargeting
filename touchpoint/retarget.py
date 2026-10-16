"""Contact-anchored retargeting of a human wrist track onto the robot workspace.

The monocular track has an unknown camera pose and only roughly known scale, but two
moments in it are unambiguous: where the hand closed on the object and where it opened
over the bowl. We pin those two contact points to the object and bowl in the sim and keep
everything else about the human motion (approach direction, lift height profile, timing):

    x(t) = g + F S H^T (p(t) - h_g) + w(t) r

    p(t)  wrist in camera frame        h_g, h_r  wrist at grasp / release
    H     human task frame: e1 along the table-projected transport h_g->h_r, e3 = table normal
    F     robot task frame: f1 along cube->bowl, f3 = world z
    S     diag(s_along, s, s): s_along makes the transport length match, s = metric scale
    r, w  residual that makes the release (and the start pose) land exactly, blended in with a
          minimum-jerk weight over the phase, so the anchors are hit and the shape is kept.

The same human clip can be re-anchored to any cube/bowl layout, which is what turns a
handful of phone clips into a few hundred robot demonstrations.
"""
import numpy as np

from .sim import CTRL_DT, CUBE_HALF, MAX_STEP

PHASES = ["approach", "grasp", "transport", "release", "retreat"]
GRASP_Z = CUBE_HALF - 0.002   # TCP height when closing on the cube
RELEASE_Z = 0.085             # TCP height when opening above the bowl
MAX_Z = 0.55
V_MAX = 0.7 * MAX_STEP / CTRL_DT
DWELL_GRASP, DWELL_RELEASE, DWELL_END = 8, 6, 12


def _minjerk(s):
    s = np.clip(s, 0, 1)
    return 10 * s**3 - 15 * s**4 + 6 * s**5


def segment(aperture, fps, close_thr=0.3, open_thr=0.55):
    """Grasp/release frames from the aperture signal (normalised per clip, with hysteresis).
    The grasp is taken as the longest closed interval."""
    lo, hi = np.percentile(aperture, [5, 95])
    a = (aperture - lo) / max(hi - lo, 1e-6)
    closed = np.zeros(len(a), bool)
    state = False
    for i, v in enumerate(a):
        if v < close_thr:
            state = True
        elif v > open_thr:
            state = False
        closed[i] = state
    edges = np.flatnonzero(np.diff(np.r_[0, closed.astype(int), 0]))
    runs = list(zip(edges[::2], edges[1::2]))
    runs = [r for r in runs if r[1] - r[0] >= 0.3 * fps]
    if not runs:
        raise ValueError("no grasp found in the aperture signal (hand never closed for >0.3 s)")
    g, r = max(runs, key=lambda r: r[1] - r[0])
    r = min(r, len(a) - 1)
    labels = np.zeros(len(a), int)
    w = int(0.15 * fps)
    labels[g:r] = 2
    labels[max(g - w, 0):g + w] = 1
    labels[r - w:r + w] = 3
    labels[r + w:] = 4
    return {"grasp": int(g), "release": int(r), "labels": labels, "a_norm": a}


def human_frame(track, seg):
    p = track["wrist"]
    hg, hr, h0 = p[seg["grasp"]], p[seg["release"]], p[0]
    # table normal from three moments the hand is near the table: rest, grasp, release
    n = np.cross(hg - h0, hr - h0)
    cam_up = np.array([0.0, -1.0, 0.0])          # image up, used for sign and as fallback
    if np.linalg.norm(n) < 1e-3 * np.linalg.norm(hg - h0) * np.linalg.norm(hr - h0):
        n = cam_up.copy()
    n /= np.linalg.norm(n)
    if n @ cam_up < 0:
        n = -n
    e1 = (hr - hg) - ((hr - hg) @ n) * n
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(n, e1)
    return np.stack([e1, e2, n]), hg, hr


def retarget(track, cube_xy, bowl_xy, start_tcp, metric_scale=1.0):
    """Returns dict with ref (N,3) TCP targets at CTRL_DT, grip (N,), phase (N,), frame (N,)
    (index into the source video), and the grasp / release control steps."""
    seg = segment(track["aperture"], track["fps"])
    Hm, hg, hr = human_frame(track, seg)
    q = (track["wrist"] - hg) @ Hm.T              # human task coordinates, origin at grasp
    gi, ri = seg["grasp"], seg["release"]

    g = np.array([*cube_xy, GRASP_Z])
    r = np.array([*bowl_xy, RELEASE_Z])
    f1 = np.r_[(r - g)[:2], 0.0]
    f1 /= np.linalg.norm(f1)
    F = np.stack([f1, np.cross([0, 0, 1.0], f1), [0, 0, 1.0]], axis=1)
    s_along = np.linalg.norm((r - g)[:2]) / max(q[ri, 0], 1e-3)
    S = np.diag([s_along, metric_scale, metric_scale])
    x = g + q @ (F @ S).T

    # residuals so the start pose and the release anchor are hit exactly
    t = np.arange(len(x))
    res_start = start_tcp - x[0]
    res_rel = r - x[ri]
    w_app = 1 - _minjerk(t / max(gi, 1))
    w_tr = _minjerk((t - gi) / max(ri - gi, 1))
    x = x + w_app[:, None] * res_start * (t <= gi)[:, None] + w_tr[:, None] * res_rel[None]
    x[:, 2] = np.clip(x[:, 2], GRASP_Z, MAX_Z)

    # uniform time scaling so the robot never exceeds V_MAX; the speed profile shape is kept
    v = np.linalg.norm(np.diff(x, axis=0), axis=1) * track["fps"]
    k = max(1.0, v.max() / V_MAX)
    n_out = int(np.ceil((len(x) - 1) / track["fps"] * k / CTRL_DT)) + 1
    src = np.linspace(0, len(x) - 1, n_out)
    ref = np.stack([np.interp(src, t, x[:, j]) for j in range(3)], 1)
    frame = np.round(src).astype(int)
    g_step = int(np.searchsorted(src, gi))
    r_step = int(np.searchsorted(src, ri))
    ref[g_step] = g
    ref[r_step] = r

    # explicit dwells: the robot gripper needs time to close/open
    def hold(arr, i, n):
        return np.concatenate([arr[:i + 1], np.repeat(arr[i:i + 1], n, 0), arr[i + 1:]])
    # close/open as soon as the TCP has arrived at the anchor, not when the human's fingers
    # crossed the threshold: otherwise the robot idles open at the cube and BC learns to idle
    def arrival(i, p):
        while i > 0 and np.linalg.norm(ref[i - 1] - p) < 0.01:
            i -= 1
        return i
    g_step, r_step = arrival(g_step, g), arrival(r_step, r)
    grip = ((np.arange(n_out) > g_step) & (np.arange(n_out) <= r_step)).astype(float)
    labels = seg["labels"][frame]
    ref, frame, labels = hold(ref, r_step, DWELL_RELEASE), hold(frame, r_step, DWELL_RELEASE), hold(labels, r_step, DWELL_RELEASE)
    grip = np.concatenate([grip[:r_step + 1], np.zeros(DWELL_RELEASE), grip[r_step + 1:]])
    ref, frame, labels = hold(ref, g_step, DWELL_GRASP), hold(frame, g_step, DWELL_GRASP), hold(labels, g_step, DWELL_GRASP)
    grip = np.concatenate([grip[:g_step + 1], np.ones(DWELL_GRASP), grip[g_step + 1:]])
    r_step += DWELL_GRASP
    n = len(ref) - 1
    ref, frame, labels, grip = (hold(a, n, DWELL_END) for a in (ref, frame, labels, grip))
    return {"ref": ref, "grip": grip, "phase": labels, "frame": frame,
            "grasp_step": g_step, "release_step": r_step + DWELL_RELEASE, "time_scale": k, "seg": seg}


def reanchor(plan, step, new_cube_xy, old_cube_xy):
    """Shift the remaining plan after the cube moved: the offset is blended in before the
    grasp and blended back out over the transport, so the release anchor is unchanged."""
    out = dict(plan)
    ref = plan["ref"].copy()
    d = np.r_[np.asarray(new_cube_xy) - np.asarray(old_cube_xy), 0.0]
    gs, rs = plan["grasp_step"], plan["release_step"]
    t = np.arange(len(ref))
    w = np.where(t <= gs, _minjerk((t - step) / max(gs - step, 1)),
                 1 - _minjerk((t - gs) / max(rs - gs, 1)))
    w[t < step] = 0
    ref += w[:, None] * d
    out["ref"] = ref
    return out


def fixed_fit_plan(track, start_tcp, nominal=((0.50, -0.15), (0.50, 0.15))):
    """Baseline: a single global similarity fit on one nominal layout, replayed unchanged."""
    return retarget(track, nominal[0], nominal[1], start_tcp)
