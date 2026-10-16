"""Smoke test on synthetic hand tracks (CPU, well under 2 minutes)."""
import numpy as np
import pytest

from touchpoint import policy as bc
from touchpoint.hand import lift, solve_translation
from touchpoint.retarget import retarget, segment
from touchpoint.rollout import run_plan, run_policy
from touchpoint.sim import PickPlaceEnv, sample_layout
from touchpoint.synthetic import make_clip


@pytest.fixture(scope="module")
def tracks():
    return [lift(make_clip(s, render=False)[0]) for s in range(2)]


def test_translation_solve_recovers_depth():
    rng = np.random.default_rng(0)
    world = rng.normal(0, 0.04, (21, 3))
    t_true = np.array([0.05, -0.02, 0.6])
    P = world + t_true
    f, cx, cy = 700.0, 480.0, 270.0
    img = np.c_[f * P[:, 0] / P[:, 2] + cx, f * P[:, 1] / P[:, 2] + cy]
    assert np.allclose(solve_translation(img, world, f, cx, cy), t_true, atol=1e-6)


def test_lift_handles_dropouts_and_tracks_aperture():
    raw, _, truth = make_clip(1, render=False)
    tk = lift(raw)
    assert np.isfinite(tk["wrist"]).all()
    assert np.abs(tk["aperture"] - truth["aperture"]).mean() < 0.01


def test_segmentation_finds_grasp_then_release(tracks):
    for tk in tracks:
        seg = segment(tk["aperture"], tk["fps"])
        assert 0 < seg["grasp"] < seg["release"] < len(tk["t"])


def test_anchored_plan_hits_anchors_and_succeeds(tracks):
    env = PickPlaceEnv()
    rng = np.random.default_rng(7)
    for tk in tracks:
        layout = sample_layout(rng)
        env.reset(layout)
        plan = retarget(tk, layout["cube"], layout["bowl"], env.home_tcp)
        assert np.allclose(plan["ref"][plan["grasp_step"], :2], layout["cube"], atol=0.01)
        ok, obs, act = run_plan(env, layout, plan)
        assert ok
        assert obs.shape[1] == bc.OBS_DIM and act.shape[1] == 4


def test_bc_trains_and_runs(tracks):
    obs, act, info = bc.collect(tracks, 16, seed=1, workers=4)
    assert info["kept"] > 8
    model = bc.train(obs, act, epochs=2, log=lambda *_: None)
    env = PickPlaceEnv()
    run_policy(env, sample_layout(np.random.default_rng(0)), bc.as_controller(model), max_steps=40)
