"""End-to-end: phone clips -> hand tracks -> contact-anchored robot demos -> BC -> evaluation.

    python run_all.py data/raw              # MediaPipe on data/raw/*.mp4, then everything else
    python run_all.py --tracks data/tracks  # reuse committed hand tracks (no MediaPipe)
    python run_all.py --synthetic           # pipeline check on generated hand tracks
    add --quick for a small smoke run
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
VIDEO_EXT = {".mp4", ".mov", ".m4v", ".avi", ".mkv"}


def save_raw(raw, path):
    np.savez_compressed(path, img=raw["img"].astype(np.float32), world=raw["world"].astype(np.float32),
                        fps=raw["fps"], width=raw["width"], height=raw["height"])


def load_raw(path):
    d = np.load(path)
    return {"img": d["img"].astype(float), "world": d["world"].astype(float), "fps": float(d["fps"]),
            "width": float(d["width"]), "height": float(d["height"])}


def save_clip(frames, fps, path, width=480):
    import cv2
    import imageio.v2 as imageio
    s = width / frames.shape[2]
    small = [cv2.resize(f, (width, int(round(f.shape[0] * s / 2)) * 2), interpolation=cv2.INTER_AREA) for f in frames]
    imageio.mimsave(path, small, fps=fps, macro_block_size=1, quality=6)


def extract_tracks(raw_dir, data_dir):
    from touchpoint.hand import detect, read_video
    videos = sorted(p for p in Path(raw_dir).iterdir() if p.suffix.lower() in VIDEO_EXT) if Path(raw_dir).is_dir() else []
    if not videos:
        sys.exit(f"No videos found in {raw_dir}/ (looked for {', '.join(sorted(VIDEO_EXT))}).\n"
                 "Record 3-5 short clips as described in the README, copy them there and re-run,\n"
                 "or run `make synthetic` to check the pipeline without real data.")
    (data_dir / "tracks").mkdir(parents=True, exist_ok=True)
    (data_dir / "clips").mkdir(parents=True, exist_ok=True)
    for v in videos:
        print(f"[hand] {v.name}")
        raw = detect(v)
        hit = np.isfinite(raw["img"][:, 0, 0]).mean()
        print(f"       {len(raw['img'])} frames, hand found in {hit:.0%}")
        save_raw(raw, data_dir / "tracks" / f"{v.stem}.npz")
        frames, fps = read_video(v)
        save_clip(frames, fps, data_dir / "clips" / f"{v.stem}.mp4")


def make_synthetic(data_dir, n=3):
    from touchpoint.synthetic import FPS, make_clip
    (data_dir / "tracks").mkdir(parents=True, exist_ok=True)
    (data_dir / "clips").mkdir(parents=True, exist_ok=True)
    for s in range(n):
        raw, frames, _ = make_clip(s)
        save_raw(raw, data_dir / "tracks" / f"synthetic_{s}.npz")
        save_clip(frames, FPS, data_dir / "clips" / f"synthetic_{s}.mp4")


def load_clip(path):
    import imageio.v2 as imageio
    return np.stack(imageio.mimread(path, memtest=False))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("raw_dir", nargs="?", help="folder of phone clips (runs MediaPipe)")
    ap.add_argument("--tracks", help="folder with *.npz hand tracks (skips MediaPipe)")
    ap.add_argument("--synthetic", action="store_true", help="use generated hand tracks")
    ap.add_argument("--quick", action="store_true", help="small smoke run")
    ap.add_argument("--episodes", type=int, default=600, help="re-anchored demos to collect")
    ap.add_argument("--eval-n", type=int, default=100)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--out", help="output folder (default results/real or results/synthetic)")
    args = ap.parse_args()

    if args.synthetic:
        data_dir, tag = ROOT / "data" / "synthetic", "synthetic"
        if not any((data_dir / "tracks").glob("*.npz")):
            make_synthetic(data_dir)
    elif args.raw_dir:
        data_dir, tag = ROOT / "data", "real"
        extract_tracks(args.raw_dir, data_dir)
    else:
        data_dir, tag = (Path(args.tracks).parent if args.tracks else ROOT / "data"), "real"
    track_dir = Path(args.tracks) if args.tracks else data_dir / "tracks"
    files = sorted(track_dir.glob("*.npz"))
    if not files:
        sys.exit(f"No hand tracks in {track_dir}/. Put clips in data/raw and run `python run_all.py data/raw`,\n"
                 "or run `make synthetic` for the pipeline check.")
    if args.quick:
        args.episodes, args.eval_n, args.epochs = 64, 16, 8
    out = Path(args.out) if args.out else ROOT / "results" / tag
    out.mkdir(parents=True, exist_ok=True)

    from touchpoint import policy as bc
    from touchpoint.evaluate import evaluate, table
    from touchpoint.hand import lift
    from touchpoint.retarget import retarget
    from touchpoint.rollout import run_plan
    from touchpoint.sim import PickPlaceEnv
    from touchpoint.viz import phase_plot, results_chart, side_by_side

    t0 = time.time()
    tracks, names = [], []
    for f in files:
        try:
            tk = lift(load_raw(f))
            retarget(tk, (0.45, -0.15), (0.5, 0.15), np.array([0.55, 0, 0.5]))  # checks segmentation
        except ValueError as e:
            print(f"[skip] {f.name}: {e}")
            continue
        tracks.append(tk)
        names.append(f.stem)
    if not tracks:
        sys.exit("None of the hand tracks could be segmented into a grasp and a release.")
    print(f"[tracks] {len(tracks)} usable: {', '.join(names)}")

    # 1. one demo, re-anchored to a showcase layout, rendered next to the phone clip
    env = PickPlaceEnv(render_size=(400, 300))
    layout = {"cube": np.array([0.45, -0.17]), "bowl": np.array([0.52, 0.16])}
    env.reset(layout)
    plan = retarget(tracks[0], layout["cube"], layout["bowl"], env.home_tcp)
    frames, widths = [], []
    env.reset(layout)
    for i in range(len(plan["ref"])):
        env.step(plan["ref"][i], plan["grip"][i])
        widths.append(env.finger_width())
        frames.append(env.render())
    print(f"[retarget] {names[0]}: {len(plan['ref'])} steps, time scale x{plan['time_scale']:.2f}, "
          f"success={env.success()}")
    clip_path = data_dir / "clips" / f"{names[0]}.mp4"
    if clip_path.exists():
        side_by_side(load_clip(clip_path), tracks[0], plan, frames, out / "side_by_side.gif", every=4, height=240)
        side_by_side(load_clip(clip_path), tracks[0], plan, frames, out / "side_by_side.mp4", every=1)
    phase_plot(tracks[0], plan, widths, out / "phases.png")

    # 2. data multiplier: every clip re-anchored to random layouts, DART noise, keep successes
    obs, act, info = bc.collect(tracks, args.episodes, seed=0)
    print(f"[collect] {info['kept']}/{info['episodes']} successful episodes, {info['samples']} samples")
    model = bc.train(obs, act, epochs=args.epochs, seed=0)
    bc.save(model, out / "bc_policy.pt")

    # 3. evaluation
    metrics, n = evaluate(tracks, model, n=args.eval_n)
    metrics_all = {"data": tag, "clips": names, "eval_layouts": n, "success": metrics, "collection": info,
                   "runtime_s": round(time.time() - t0, 1)}
    (out / "metrics.json").write_text(json.dumps(metrics_all, indent=2))
    title = "Synthetic hand tracks" if tag == "synthetic" else "Phone-recorded demos"
    md = f"### {title}\n\n" + table(metrics, n) + "\n"
    (out / "metrics.md").write_text(md)
    results_chart(metrics, n, out / "results.png", title)

    # a BC rollout with a nudged cube, for the README
    from touchpoint.policy import as_controller
    from touchpoint.rollout import run_policy
    bc_frames = []
    ok = run_policy(env, layout, as_controller(model), nudge_rng=np.random.default_rng(3), frames=bc_frames)
    import imageio.v2 as imageio
    import cv2
    small = [cv2.resize(f, (320, 240), interpolation=cv2.INTER_AREA) for f in bc_frames[::4]]
    imageio.mimsave(out / "bc_nudged.gif", small, duration=200, loop=0)
    print(f"[bc] nudged showcase rollout success={ok}")

    print("\n" + md)
    print(f"done in {time.time() - t0:.0f}s -> {out}")


if __name__ == "__main__":
    main()
