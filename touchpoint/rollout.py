"""Executing plans and policies in the sim, with optional mid-episode cube nudges."""
import numpy as np

from .retarget import reanchor
from .sim import X_RANGE, Y_RANGE

NUDGE = 0.05
LABEL_LEAD = 3
RELEASE_LEAD = 6
LOOKAHEAD = 8


def _maybe_nudge(env, rng, state):
    """Move the cube once, while the gripper is closing in on it."""
    if state["done"]:
        return None
    o = env.obs()
    if o["grip"] < 0.5 and np.linalg.norm(o["tcp"][:2] - o["cube"][:2]) < 0.08 and o["tcp"][2] < 0.12:
        a = rng.uniform(0, 2 * np.pi)
        old = o["cube"][:2].copy()
        new = np.clip(old + NUDGE * np.array([np.cos(a), np.sin(a)]),
                      [X_RANGE[0], Y_RANGE[0]], [X_RANGE[1], Y_RANGE[1]])
        env.nudge_cube(new - old)
        state["done"] = True
        return old, new
    return None


def obs_vec(o):
    # position = the commanded TCP reference, so actions are increments of the command and
    # do not absorb the servo's steady-state tracking error
    tcp, cube, mug = o["ref"], o["cube"], o["mug"]
    # no previous gripper command in the observation: the policy has to infer the phase
    # from geometry and finger width, which avoids the "never closes" copycat failure
    return np.r_[tcp, cube - tcp, mug[:2] - tcp[:2], mug[:2] - cube[:2], o["width"] / 0.08].astype(np.float32)


def run_plan(env, layout, plan, nudge_rng=None, replan=False, noise=0.0, rng=None, frames=None):
    """Open-loop execution of a retargeted plan. With replan=True the plan is re-anchored when
    the cube is nudged (that is the expert used to label recovery data).
    Returns success flag and (obs, action) pairs labelled with the clean plan."""
    env.reset(layout)
    obs, acts = [], []
    nst = {"done": nudge_rng is None}
    i = 0
    while i < len(plan["ref"]):
        if nudge_rng is not None and i < plan["grasp_step"]:
            moved = _maybe_nudge(env, nudge_rng, nst)
            if moved and replan:
                plan = reanchor(plan, i, moved[1], moved[0])
        o = env.obs()
        target, grip = plan["ref"][i], plan["grip"][i]
        obs.append(obs_vec(o))
        # gripper label looks LABEL_LEAD steps ahead so the switch is not a single-frame event
        # (longer lead for opening: dropping the cube from slightly early is harmless)
        g = plan["grip"]
        grip_label = g[min(i + LABEL_LEAD, len(g) - 1)] * g[min(i + RELEASE_LEAD, len(g) - 1)] if grip else g[min(i + LABEL_LEAD, len(g) - 1)]
        # position label: mean velocity of the plan over the next LOOKAHEAD steps, measured from
        # the current (possibly noise-perturbed) command; this both corrects drift and avoids
        # zero labels at the slow start of a movement
        ahead = plan["ref"][min(i + LOOKAHEAD, len(plan["ref"]) - 1)]
        acts.append(np.r_[(ahead - o["ref"]) / LOOKAHEAD, grip_label].astype(np.float32))
        cmd = target + (rng.normal(0, noise, 3) if noise > 0 else 0)
        env.step(cmd, grip)
        if frames is not None:
            frames.append(env.render())
        i += 1
    for _ in range(10):
        env.step(env.ref, 0.0)
    obs, acts = np.array(obs), np.array(acts)
    # drop pure waiting frames (plan not moving, fingers settled): they are identical states
    # with zero-motion labels and teach the policy to stall at the grasp and release points
    still = np.linalg.norm(acts[:, :3], axis=1) < 5e-4
    settled = np.r_[False, np.abs(np.diff(obs[:, -1])) < 0.01]
    keep = ~(still & settled)
    return env.success(), obs[keep], acts[keep]


def run_policy(env, layout, policy, max_steps=450, nudge_rng=None, frames=None):
    env.reset(layout)
    nst = {"done": nudge_rng is None}
    released_steps = 0
    for _ in range(max_steps):
        if nudge_rng is not None:
            _maybe_nudge(env, nudge_rng, nst)
        o = env.obs()
        a = policy(obs_vec(o))
        env.step(o["ref"] + a[:3], a[3])
        if frames is not None:
            frames.append(env.render())
        if env.success():
            released_steps += 1
            if released_steps > 5:
                return True
    return env.success()
