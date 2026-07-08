"""
ED Fig 2 - Per-model replications: the alpha-beta decomposition for each of
the seven models separately (humans first as reference). Compact version of
fig_alphabeta panels; per-model point caches from fig_alphabeta_all.py.
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

import style

MODELS = [
    ("gpt-4o-mini",               "GPT-4o-mini"),
    ("gpt-4o",                    "GPT-4o"),
    ("claude-haiku-4-5-20251001", "Claude Haiku 4.5"),
    ("gpt-5.5-2026-04-23",        "GPT-5.5 (2 waves)"),
    ("gemma2_9b",                 "Gemma-2 9B"),
    ("mistral_latest",            "Mistral-7B"),
    ("llama3_1_8b_instruct_q4",   "Llama-3.1 8B"),
]

CM_MODEL = LinearSegmentedColormap.from_list(
    "d", ["#ffffff", "#c6dbee", "#3f7fbf", "#08306b"])
CM_HUMAN = LinearSegmentedColormap.from_list(
    "h", ["#ffffff", "#d5e8d0", "#5ba05b", "#0b3d0b"])
LIM = (-0.9, 2.4)


def panel(ax, pts, title, cmap, show_xlab, show_ylab):
    med = np.median(pts, axis=0)
    budget = np.median(pts.sum(axis=1))
    ax.hexbin(pts[:, 0], pts[:, 1], gridsize=42, extent=(*LIM, *LIM),
              cmap=cmap, bins="log", linewidths=0)
    ax.axhline(0, color="0.65", lw=0.5)
    ax.axvline(0, color="0.65", lw=0.5)
    ax.plot([LIM[0], LIM[1]], [1 - LIM[0], 1 - LIM[1]], ls="--", lw=0.9,
            color="#c23b22", alpha=0.75, zorder=4)
    ax.plot(1, 1, marker="*", ms=9, color="#e58606", mec="white", mew=0.6,
            zorder=5)
    ax.plot([1, 0], [0, 1], marker="o", ms=3.5, color="#c23b22",
            mec="white", mew=0.4, lw=0, zorder=5)
    ax.plot(*med, marker="P", ms=6, color="white", mec="black", mew=0.7,
            zorder=6)
    ax.annotate(f"$\\alpha{{+}}\\beta$ = {budget:.2f}",
                xy=(0.04, 0.04), xycoords="axes fraction", fontsize=6.6,
                color="0.15",
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="0.7",
                          lw=0.5))
    ax.set_xlim(LIM); ax.set_ylim(LIM)
    ax.set_aspect("equal")
    ax.set_title(f"{title}\n({len(pts):,} cells)", fontsize=7.3, pad=2.5)
    ax.tick_params(labelsize=6)
    if show_xlab:
        ax.set_xlabel(r"$\alpha$", fontsize=7)
    else:
        ax.set_xticklabels([])
    if show_ylab:
        ax.set_ylabel(r"$\beta$", fontsize=7)
    else:
        ax.set_yticklabels([])


def main():
    hp = np.load(os.path.join(style.HERE, "alphabeta_points.npz"))["hp"]
    fig, axes = plt.subplots(2, 4, figsize=(7.2, 4.3))
    panel(axes[0, 0], hp, "Humans (steering space)", CM_HUMAN,
          show_xlab=False, show_ylab=True)
    for i, (tag, name) in enumerate(MODELS, start=1):
        r, c = divmod(i, 4)
        pts = np.load(os.path.join(style.HERE,
                                   f"alphabeta_pts_{tag}.npz"))["mp"]
        panel(axes[r, c], pts, name, CM_MODEL,
              show_xlab=(r == 1), show_ylab=(c == 0))
    fig.subplots_adjust(left=0.065, right=0.985, bottom=0.09, top=0.92,
                        wspace=0.1, hspace=0.28)
    style.save(fig, "fig_ed_permodel")


if __name__ == "__main__":
    main()
