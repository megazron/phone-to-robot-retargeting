"""Render docs/reanchor_nudge.gif: open-loop replay vs online re-anchoring on the same nudged layout.

    python scripts/make_nudge_gif.py [--seed N]

Without --seed it searches evaluation layouts for one where re-anchoring succeeds and the
open-loop replay fails under the same 5 cm nudge.
"""
import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from run_all import load_raw  # noqa: E402
from touchpoint.evaluate import EVAL_SEED  # noqa: E402
from touchpoint.topdown import lift_topdown, split_episodes  # noqa: E402


def load_tracks(track_dir):
    tracks = []
    for f in sorted(Path(track_dir).glob("*.npz")):
        raw = load_raw(f)
        try:
            tracks += split_episodes(lift_topdown(raw, raw["mug"]))
        except ValueError:
            continue
    return tracks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=None, help="evaluation layout index")
    ap.add_argument("--tracks", default=str(ROOT / "data" / "tracks"))
    ap.add_argument("--out", default=str(ROOT / "docs" / "reanchor_nudge.gif"))
    args = ap.parse_args()
    from touchpoint.retarget import retarget
    from touchpoint.sim import PickPlaceEnv, sample_layout
    try:
        env = PickPlaceEnv(render_size=(480, 360))
        env.reset(sample_layout(np.random.default_rng(0)))
        env.render()
    except Exception:
        if os.environ["MUJOCO_GL"] == "osmesa":
            raise
        # the GL backend is fixed at import time, so retry in a fresh process
        os.execve(sys.executable, [sys.executable, *sys.argv], {**os.environ, "MUJOCO_GL": "osmesa"})
    from touchpoint.rollout import run_plan
    from touchpoint.viz import nudge_comparison

    tracks = load_tracks(args.tracks)
    candidates = [args.seed] if args.seed is not None else range(60)
    for i in candidates:
        layout = sample_layout(np.random.default_rng(EVAL_SEED + i))
        plan = retarget(tracks[i % len(tracks)], layout["cube"], layout["mug"], env.home_tcp)
        runs = {}
        for name, track in (("open", False), ("reanchor", True)):
            frames, marks = [], []
            env.nudge_cube = lambda d, f=frames, m=marks, n=type(env).nudge_cube: (m.append(len(f)), n(env, d))
            ok = run_plan(env, layout, plan, nudge_rng=np.random.default_rng(EVAL_SEED * 7 + i),
                          track_cube=track, frames=frames)[0]
            runs[name] = (ok, frames, marks[0] if marks else None)
        print(f"layout {i}: open-loop={runs['open'][0]} re-anchoring={runs['reanchor'][0]}")
        if runs["reanchor"][0] and not runs["open"][0]:
            break
    else:
        sys.exit("no layout with that outcome pair")
    nudge_comparison(runs["open"][1], runs["reanchor"][1], runs["reanchor"][2], Path(args.out),
                     outcomes=(runs["open"][0], runs["reanchor"][0]))
    print(f"seed {i} -> {args.out} ({Path(args.out).stat().st_size / 1e6:.2f} MB)")


if __name__ == "__main__":
    main()
