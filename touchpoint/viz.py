"""Side-by-side video, phase plot and results chart."""
import cv2
import imageio.v2 as imageio
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from .hand import draw_overlay  # noqa: E402
from .retarget import PHASES  # noqa: E402

BLUE, ORANGE = "#2a78d6", "#eb6834"
INK, MUTED, GRID = "#1f1f1e", "#6b6a63", "#e4e3dc"
PHASE_COLORS = ["#cde2fb", "#fbd3c0", "#c7ecd9", "#fbe3a8", "#e9e8e2"]


def side_by_side(clip_frames, track, plan, sim_frames, out_path, every=2, height=300):
    """Left: phone frame + landmarks + human phase. Right: Panda executing the plan."""
    out = []
    lm_scale = clip_frames.shape[2] / track["width"]   # clips are stored downsampled
    for i in range(0, len(sim_frames), every):
        k = min(int(plan["frame"][min(i, len(plan["frame"]) - 1)]), len(clip_frames) - 1)
        phase = PHASES[int(plan["phase"][min(i, len(plan["phase"]) - 1)])]
        left = draw_overlay(clip_frames[k], track["img"][k] * lm_scale, f"human  t={track['t'][k]:.2f}s  {phase}")
        s = height / left.shape[0]
        left = cv2.resize(left, (int(left.shape[1] * s) // 2 * 2, height), interpolation=cv2.INTER_AREA)
        right = sim_frames[i]
        right = cv2.resize(right, (int(right.shape[1] * height / right.shape[0]) // 2 * 2, height))
        grip = "closed" if plan["grip"][min(i, len(plan["grip"]) - 1)] > 0.5 else "open"
        cv2.rectangle(right, (0, 0), (right.shape[1], 30), (0, 0, 0), -1)
        cv2.putText(right, f"panda  step {i}  gripper {grip}", (8, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (255, 255, 255), 1, cv2.LINE_AA)
        out.append(np.concatenate([left, right], axis=1))
    fps = 1.0 / (0.05 * every)
    if out_path.suffix == ".gif":
        imageio.mimsave(out_path, out, duration=1000 / fps, loop=0)
    else:
        imageio.mimsave(out_path, out, fps=fps, macro_block_size=1)


def _style(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.set_axisbelow(True)


def phase_plot(track, plan, widths, out_path):
    seg = plan["seg"]
    t = track["t"]
    fig, axes = plt.subplots(2, 1, figsize=(8, 4.6), sharex=True)
    ax = axes[0]
    labels = seg["labels"]
    for p in range(len(PHASES)):
        m = labels == p
        if m.any():
            ax.fill_between(t, 0, 1, where=m, color=PHASE_COLORS[p], step="mid", lw=0,
                            transform=ax.get_xaxis_transform(), label=PHASES[p])
    ax.plot(t, track["aperture"] * 100, color=BLUE, lw=2, label="thumb-index aperture")
    ax.axvline(t[seg["grasp"]], color=INK, lw=1, ls="--")
    ax.axvline(t[seg["release"]], color=INK, lw=1, ls="--")
    ax.set_ylabel("aperture (cm)", color=INK, fontsize=9)
    ax.set_title("Human: aperture and detected phases", loc="left", fontsize=10, color=INK)
    ax.legend(fontsize=7, ncol=6, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.02))
    _style(ax)

    ax = axes[1]
    # robot steps mapped back onto the human clock via the frame index of each step
    tr = t[np.clip(plan["frame"][:len(widths)], 0, len(t) - 1)]
    ax.step(tr, plan["grip"][:len(widths)], where="post", color=ORANGE, lw=2, label="gripper command (1 = close)")
    ax.plot(tr, np.asarray(widths) / 0.08, color=BLUE, lw=2, label="finger width / max")
    ax.set_ylim(-0.1, 1.15)
    ax.set_xlabel("human clip time (s)", color=INK, fontsize=9)
    ax.set_title("Robot: gripper command and finger width (re-timed onto the clip)", loc="left", fontsize=10, color=INK)
    ax.legend(fontsize=8, frameon=False, loc="center right")
    _style(ax)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def results_chart(metrics, n, out_path, title):
    rows = [("Fixed-fit replay", "fixed_fit", None),
            ("Contact-anchored replay", "anchored", "anchored_nudged"),
            ("BC policy (closed loop)", "bc", "bc_nudged")]
    fig, ax = plt.subplots(figsize=(7, 3.2))
    y = np.arange(len(rows))[::-1]
    h = 0.36
    for yi, (label, a, b) in zip(y, rows):
        for off, key, col, name in ((h / 2 + 0.02, a, BLUE, "static layout"), (-h / 2 - 0.02, b, ORANGE, "cube nudged mid-approach")):
            if key and key in metrics:
                v = metrics[key] * 100
                ax.barh(yi + off, v, height=h, color=col, label=name if yi == y[1] else None)
                ax.text(v + 1.5, yi + off, f"{v:.0f}%", va="center", fontsize=9, color=INK)
    ax.set_yticks(y, [r[0] for r in rows], fontsize=9, color=INK)
    ax.set_xlim(0, 112)
    ax.set_xlabel("success rate (%)", fontsize=9, color=INK)
    ax.set_title(f"{title}  ({n} random layouts, fixed seeds)", loc="left", fontsize=10, color=INK, pad=26)
    ax.legend(fontsize=8, frameon=False, loc="lower left", bbox_to_anchor=(0.0, 1.06), ncol=2)
    _style(ax)
    ax.grid(axis="x", color=GRID, lw=0.8)
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
