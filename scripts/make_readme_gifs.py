"""Render the two pipeline GIFs used in the README.

    python scripts/make_readme_gifs.py [--only layouts|pipeline]

docs/one_demo_many_layouts.gif: one phone episode next to the Panda executing it on four random layouts.
docs/raw_to_robot.gif: raw clip with the pinch trail, the mug-centred track, and the simulated robot.
"""
import argparse
import os
import sys
from pathlib import Path

os.environ["MUJOCO_GL"] = "osmesa"   # EGL is not available on every machine
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from run_all import load_clip, load_raw  # noqa: E402
from touchpoint.hand import draw_overlay  # noqa: E402
from touchpoint.retarget import PHASES, retarget  # noqa: E402
from touchpoint.sim import PickPlaceEnv, sample_layout  # noqa: E402
from touchpoint.topdown import lift_topdown, split_episodes  # noqa: E402

FPS, EVERY = 12, 2
FONT = cv2.FONT_HERSHEY_SIMPLEX


def label(img, text, h=22, scale=0.45):
    cv2.rectangle(img, (0, 0), (img.shape[1], h), (0, 0, 0), -1)
    cv2.putText(img, text, (6, h - 7), FONT, scale, (255, 255, 255), 1, cv2.LINE_AA)
    return img


def fit(img, size):
    return cv2.resize(img, size, interpolation=cv2.INTER_AREA)


def save_gif(frames, path, colors=128):
    from PIL import Image
    pal = Image.fromarray(np.concatenate(frames[::max(1, len(frames) // 8)], axis=0)).quantize(
        colors, method=Image.Quantize.MEDIANCUT)
    ims = [Image.fromarray(f).quantize(palette=pal, dither=Image.Dither.NONE) for f in frames]
    ims[0].save(path, save_all=True, append_images=ims[1:], duration=int(1000 / FPS), loop=0, optimize=True)
    print(f"{path} ({Path(path).stat().st_size / 1e6:.2f} MB, {len(frames)} frames)")


def rollout(env, layout, plan, size):
    env.reset(layout)
    frames = []
    for i in range(len(plan["ref"])):
        env.step(plan["ref"][i], plan["grip"][i])
        if i % EVERY == 0:
            frames.append(fit(env.render(), size))
    frames.append(fit(env.render(), size))
    return frames, env.success()


def load_episode(clip_name, ep_idx):
    raw = load_raw(ROOT / "data" / "tracks" / f"{clip_name}.npz")
    eps = split_episodes(lift_topdown(raw, raw["mug"]))
    tk = eps[ep_idx]
    clip = load_clip(ROOT / "data" / "raw" / f"{clip_name}.mp4")
    return tk, raw["mug"], clip


def phone_frame(clip, tk, k, mug=None, trail=None):
    s = clip.shape[2] / tk["width"]
    f = clip[min(tk["frame0"] + k, len(clip) - 1)].copy()
    f = draw_overlay(f, tk["img"][k] * s)
    if mug is not None:
        x, y, r = mug[min(tk["frame0"] + k, len(mug) - 1)] * s
        cv2.circle(f, (int(x), int(y)), int(r), (255, 200, 60), 6, cv2.LINE_AA)
    if trail is not None:
        pts = (0.5 * (tk["img"][:k + 1, 4] + tk["img"][:k + 1, 8]) * s).astype(np.int32)
        if len(pts) > 1:
            cv2.polylines(f, [pts.reshape(-1, 1, 2)], False, (235, 104, 52), 5, cv2.LINE_AA)
    return f


def one_demo_many_layouts(tk, clip, out, cell=(240, 180)):
    env = PickPlaceEnv(render_size=(320, 240))
    rng = np.random.default_rng(11)
    runs = []
    while len(runs) < 4:
        layout = sample_layout(rng)
        env.reset(layout)
        plan = retarget(tk, layout["cube"], layout["mug"], env.home_tcp)
        frames, ok = rollout(env, layout, plan, cell)
        if ok:
            runs.append((plan, frames))
    plan0 = runs[0][0]
    n = max(len(f) for _, f in runs)
    H = 2 * cell[1] + 4
    out_frames = []
    for j in list(range(n)) + [n - 1] * 10:
        i = min(j * EVERY, len(plan0["frame"]) - 1)
        k = min(int(plan0["frame"][i]), len(tk["img"]) - 1)
        ph = phone_frame(clip, tk, k)
        ph = fit(ph, (int(ph.shape[1] * H / ph.shape[0]) // 2 * 2, H))
        label(ph, "phone (real)")
        tiles = [label(fr[min(j, len(fr) - 1)].copy(), f"layout {m + 1}") for m, (_, fr) in enumerate(runs)]
        v = np.full((cell[1], 4, 3), 255, np.uint8)
        top = np.concatenate([tiles[0], v, tiles[1]], axis=1)
        bot = np.concatenate([tiles[2], v, tiles[3]], axis=1)
        grid = np.concatenate([top, np.full((4, top.shape[1], 3), 255, np.uint8), bot], axis=0)
        out_frames.append(np.concatenate([ph, np.full((H, 4, 3), 255, np.uint8), grid], axis=1))
    save_gif(out_frames, out, colors=48)


def track_panel(tk, upto, size, g, r):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    w = tk["wrist"] * 100
    fig, ax = plt.subplots(figsize=(size[0] / 100, size[1] / 100), dpi=100)
    ax.add_patch(plt.Circle((0, 0), 4, color="#3a3a38", alpha=0.25, lw=0))
    ax.text(0, 0, "mug", ha="center", va="center", fontsize=7)
    pad = 4
    ax.set_xlim(w[:, 0].min() - pad, w[:, 0].max() + pad)
    ax.set_ylim(w[:, 1].min() - pad, w[:, 1].max() + pad)
    ax.set_aspect("equal", adjustable="datalim")
    ax.scatter(w[:upto + 1, 0], w[:upto + 1, 1], c=w[:upto + 1, 2], cmap="viridis", vmin=0, vmax=max(w[:, 2].max(), 1),
               s=6, lw=0)
    if upto >= g:
        ax.plot(w[g, 0], w[g, 1], "o", color="#eb6834", ms=7, mec="white", label="grasp")
    if upto >= r:
        ax.plot(w[r, 0], w[r, 1], "s", color="#1f1f1e", ms=6, mec="white", label="release")
    ax.tick_params(labelsize=6)
    ax.set_xlabel("x (cm, mug frame)   colour = height", fontsize=7)
    fig.tight_layout(pad=0.3)
    fig.canvas.draw()
    img = np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()
    plt.close(fig)
    return fit(img, size)


def raw_to_robot(tk, mug, clip, out, size=(240, 240)):
    env = PickPlaceEnv(render_size=(320, 320))
    layout = {"cube": np.array([0.45, -0.17]), "mug": np.array([0.52, 0.16])}
    env.reset(layout)
    plan = retarget(tk, layout["cube"], layout["mug"], env.home_tcp)
    sim, _ = rollout(env, layout, plan, size)
    out_frames = []
    v = np.full((size[1], 4, 3), 255, np.uint8)
    for j in list(range(len(sim))) + [len(sim) - 1] * 10:
        i = min(j * EVERY, len(plan["frame"]) - 1)
        k = min(int(plan["frame"][i]), len(tk["img"]) - 1)
        phase = PHASES[int(plan["phase"][i])]
        ph = phone_frame(clip, tk, k, mug=mug, trail=True)
        s = size[1] / ph.shape[0]
        ph = fit(ph, (int(ph.shape[1] * s) // 2 * 2, size[1]))
        label(ph, f"raw clip  {phase}")
        tr = label(track_panel(tk, k, size, tk["grasp"], tk["release"]), "mug-centred track")
        sm = label(sim[min(j, len(sim) - 1)].copy(), "sim")
        out_frames.append(np.concatenate([ph, v, tr, v, sm], axis=1))
    save_gif(out_frames, out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clip", default="clip01")
    ap.add_argument("--episode", type=int, default=0)
    ap.add_argument("--only", choices=["layouts", "pipeline"])
    args = ap.parse_args()
    tk, mug, clip = load_episode(args.clip, args.episode)
    if args.only in (None, "layouts"):
        one_demo_many_layouts(tk, clip, ROOT / "docs" / "one_demo_many_layouts.gif")
    if args.only in (None, "pipeline"):
        raw_to_robot(tk, mug, clip, ROOT / "docs" / "raw_to_robot.gif")


if __name__ == "__main__":
    main()
