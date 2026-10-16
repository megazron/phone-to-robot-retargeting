"""Success rates over fixed-seed random layouts for the replay baselines and the BC policy."""
import multiprocessing as mp

import numpy as np
import torch

from .policy import as_controller
from .retarget import fixed_fit_plan, retarget
from .rollout import run_plan, run_policy
from .sim import PickPlaceEnv, sample_layout

EVAL_SEED = 12345
METHODS = [
    ("fixed_fit", False, "Fixed-fit replay (one global fit, no anchoring)"),
    ("anchored", False, "Contact-anchored replay (open loop)"),
    ("bc", False, "BC policy (closed loop)"),
    ("anchored", True, "Contact-anchored replay (open loop)"),
    ("bc", True, "BC policy (closed loop)"),
]


def _eval_worker(args):
    tracks, state_dict, idxs = args
    torch.set_num_threads(1)
    env = PickPlaceEnv()
    policy = None
    if state_dict is not None:
        from .policy import MLP
        m = MLP()
        m.load_state_dict(state_dict)
        policy = as_controller(m)
    rows = []
    for i in idxs:
        layout = sample_layout(np.random.default_rng(EVAL_SEED + i))
        track = tracks[i % len(tracks)]
        env.reset(layout)
        res = {}
        for name, nudged, _ in METHODS:
            nudge = np.random.default_rng(EVAL_SEED * 7 + i) if nudged else None
            key = f"{name}{'_nudged' if nudged else ''}"
            if name == "bc":
                if policy is None:
                    continue
                res[key] = run_policy(env, layout, policy, nudge_rng=nudge)
            else:
                plan = (fixed_fit_plan(track, env.home_tcp) if name == "fixed_fit"
                        else retarget(track, layout["cube"], layout["mug"], env.home_tcp))
                res[key] = run_plan(env, layout, plan, nudge_rng=nudge)[0]
        rows.append(res)
    return rows


def evaluate(tracks, model=None, n=100, workers=None):
    workers = workers or max(1, min(mp.cpu_count() - 2, 16))
    sd = None if model is None else {k: v.clone() for k, v in model.state_dict().items()}
    chunks = [(tracks, sd, list(range(n))[w::workers]) for w in range(workers)]
    with mp.get_context("fork").Pool(workers) as pool:
        rows = [r for part in pool.map(_eval_worker, chunks) for r in part]
    keys = rows[0].keys()
    return {k: float(np.mean([r[k] for r in rows])) for k in keys}, n


def table(metrics, n):
    lines = ["| Method | Block nudged mid-approach | Success |", "|---|---|---|"]
    for name, nudged, label in METHODS:
        key = f"{name}{'_nudged' if nudged else ''}"
        if key in metrics:
            lines.append(f"| {label} | {'yes' if nudged else 'no'} | {metrics[key] * 100:.0f}% |")
    lines.append(f"\n{n} random layouts, fixed seeds.")
    return "\n".join(lines)
