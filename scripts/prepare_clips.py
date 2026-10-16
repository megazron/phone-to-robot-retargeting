"""Preprocessing used to make data/raw/*.mp4 from the original handheld phone videos.

The phone was held by hand above the table, so the view drifts. Each frame is cropped to a
fixed window placed so the mug centre stays at the same pixel (mug tracked as a dark circle,
median + box filtered). This also crops out everything but the table. Audio and metadata are
dropped by re-encoding. The original videos are not part of the repo.

    python scripts/prepare_clips.py original.mp4 data/raw/clip01.mp4
"""
import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
from scipy.ndimage import median_filter, uniform_filter1d

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from touchpoint.topdown import find_mug  # noqa: E402

WINDOW = {True: (560, 720), False: (720, 480)}     # (w, h) for portrait / landscape sources
ANCHOR = {True: (130, 170), False: (170, 330)}     # mug centre inside the window


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("dst")
    args = ap.parse_args()

    cap = cv2.VideoCapture(args.src)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frames = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        frames.append(f)
    if not frames:
        sys.exit(f"could not read {args.src}")
    H, W = frames[0].shape[:2]
    portrait = H > W
    win, anchor = WINDOW[portrait], ANCHOR[portrait]

    track, prev, miss = [], None, 0
    for f in frames:
        m = find_mug(cv2.cvtColor(f, cv2.COLOR_BGR2RGB), prev)
        if m is None:
            miss += 1
            track.append(track[-1] if track else (W / 2, H / 2, 60))
        else:
            prev = m[:2]
            track.append(m)
    track = np.array(track, float)
    for k in range(3):
        track[:, k] = uniform_filter1d(median_filter(track[:, k], 15, mode="nearest"), 15, mode="nearest")

    out = cv2.VideoWriter(args.dst, cv2.VideoWriter_fourcc(*"mp4v"), fps, win)
    for f, (x, y, _) in zip(frames, track):
        x0 = int(np.clip(round(x - anchor[0]), 0, W - win[0]))
        y0 = int(np.clip(round(y - anchor[1]), 0, H - win[1]))
        out.write(f[y0:y0 + win[1], x0:x0 + win[0]])
    out.release()
    print(f"{Path(args.src).name}: {len(frames)} frames, mug missed in {miss}, "
          f"median mug radius {np.median(track[:, 2]):.0f} px")
    print("re-encode to H.264 without audio/metadata, e.g.\n"
          f"  ffmpeg -i {args.dst} -an -map_metadata -1 -c:v libx264 -crf 23 out.mp4")


if __name__ == "__main__":
    main()
