"""Closed-loop behaviour cloning on re-anchored human demos.

Data: every human clip is re-anchored to many random layouts and executed in sim. During
collection the executed command is perturbed with Gaussian noise (DART) while the label stays
the clean plan, and a fraction of episodes get a mid-approach cube nudge with the plan
re-anchored on the fly. Only successful episodes are kept.
"""
import multiprocessing as mp

import numpy as np
import torch
import torch.nn as nn

from .retarget import retarget
from .rollout import run_plan
from .sim import PickPlaceEnv, sample_layout

ACT_SCALE = 0.05
OBS_DIM = 11


def _collect_worker(args):
    tracks, seeds, noise, nudge_frac = args
    torch.set_num_threads(1)
    env = PickPlaceEnv()
    out = []
    for seed in seeds:
        rng = np.random.default_rng(seed)
        layout = sample_layout(rng)
        track = tracks[seed % len(tracks)]
        env.reset(layout)
        try:
            plan = retarget(track, layout["cube"], layout["bowl"], env.home_tcp)
        except ValueError:
            continue
        nudge = np.random.default_rng(seed + 10**6) if rng.random() < nudge_frac else None
        ok, obs, act = run_plan(env, layout, plan, nudge_rng=nudge, replan=True, noise=noise, rng=rng)
        out.append((ok, obs, act))
    return out


def collect(tracks, n_episodes, seed=0, noise=0.008, nudge_frac=0.3, workers=None):
    workers = workers or max(1, min(mp.cpu_count() - 2, 16))
    seeds = [seed * 100000 + i for i in range(n_episodes)]
    chunks = [(tracks, seeds[i::workers], noise, nudge_frac) for i in range(workers)]
    with mp.get_context("fork").Pool(workers) as pool:
        res = [r for part in pool.map(_collect_worker, chunks) for r in part]
    kept = [(o, a) for ok, o, a in res if ok]
    if not kept:
        raise RuntimeError("no successful demonstrations collected; check the hand tracks")
    obs = np.concatenate([o for o, _ in kept])
    act = np.concatenate([a for _, a in kept])
    return obs, act, {"episodes": len(res), "kept": len(kept), "samples": len(obs)}


class MLP(nn.Module):
    def __init__(self, hidden=256):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(OBS_DIM, hidden), nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU(),
                                 nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, 4))
        self.register_buffer("mu", torch.zeros(OBS_DIM))
        self.register_buffer("sd", torch.ones(OBS_DIM))

    def forward(self, x):
        return self.net((x - self.mu) / self.sd)


def train(obs, act, epochs=40, seed=0, log=print):
    torch.manual_seed(seed)
    model = MLP()
    X = torch.tensor(obs)
    Y = torch.tensor(act)
    model.mu.copy_(X.mean(0))
    model.sd.copy_(X.std(0) + 1e-3)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    n = len(X)
    for ep in range(epochs):
        perm = torch.randperm(n)
        tot = 0.0
        for i in range(0, n, 512):
            idx = perm[i:i + 512]
            out = model(X[idx])
            l_pos = ((out[:, :3] - Y[idx, :3] / ACT_SCALE) ** 2).mean()
            l_grip = nn.functional.binary_cross_entropy_with_logits(out[:, 3], Y[idx, 3])
            loss = l_pos + 0.5 * l_grip
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += loss.item() * len(idx)
        sched.step()
        if ep % 10 == 0 or ep == epochs - 1:
            log(f"  epoch {ep:3d}  loss {tot / n:.4f}")
    return model


def as_controller(model):
    model.eval()

    def act(o):
        with torch.no_grad():
            out = model(torch.from_numpy(o)[None])[0].numpy()
        return np.r_[out[:3] * ACT_SCALE, float(out[3] > 0)]
    return act


def save(model, path):
    torch.save(model.state_dict(), path)


def load(path):
    m = MLP()
    m.load_state_dict(torch.load(path, map_location="cpu"))
    return m
