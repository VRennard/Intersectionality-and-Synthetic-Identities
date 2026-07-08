"""
Fig 2 — Who drives opinion: humans vs LLM, per topic.

Two side-by-side heatmaps (topics x dimensions), shared colorbar:
cell = mean steering magnitude (TV between subgroup and population),
pooled over the dimension's values. Right strip: Spearman rank correlation
between the human and LLM dimension rankings per topic.

Data: paper_figs/exaggeration_data.json (gpt-4o-mini, built by
fig_exaggeration.py). W49 excluded there.
"""

import os, sys, json
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

import style

DIMS  = ["Religion", "Race", "Political Party", "Education", "Age", "Income", "Gender"]
SHORT = {"Religion": "Religion", "Race": "Race", "Political Party": "Party",
         "Education": "Education", "Age": "Age", "Income": "Income", "Gender": "Gender"}


MODEL = sys.argv[1] if len(sys.argv) > 1 else "gpt-4o-mini"


def main():
    fname = ("exaggeration_data.json" if MODEL == "gpt-4o-mini"
             else f"exaggeration_data_{MODEL}.json")
    with open(os.path.join(style.HERE, fname)) as f:
        points = json.load(f)

    waves = sorted({p["wave"] for p in points}, key=int)
    H = np.full((len(waves), len(DIMS)), np.nan)
    L = np.full((len(waves), len(DIMS)), np.nan)
    for i, w in enumerate(waves):
        for j, d in enumerate(DIMS):
            pts = [p for p in points if p["wave"] == w and p["dim"] == d]
            if pts:
                H[i, j] = np.mean([p["tv_h"] for p in pts])
                L[i, j] = np.mean([p["tv_l"] for p in pts])

    rho = [spearmanr(H[i], L[i]).statistic for i in range(len(waves))]

    fig, axes = plt.subplots(1, 3, figsize=(6.5, 3.3),
                             gridspec_kw={"width_ratios": [7, 7, 1.1]})
    vmax = max(np.nanmax(H), np.nanmax(L))

    for ax, M, title in ((axes[0], H, "humans"), (axes[1], L, f"LLM ({MODEL})")):
        im = ax.imshow(M, cmap="YlOrRd", vmin=0, vmax=vmax, aspect="auto")
        ax.set_xticks(range(len(DIMS)))
        ax.set_xticklabels([SHORT[d] for d in DIMS], rotation=45, ha="right", fontsize=7)
        ax.set_title(title, fontsize=8.5)
        ax.spines["top"].set_visible(True); ax.spines["right"].set_visible(True)
        # mark the per-topic argmax with a dot
        for i in range(len(waves)):
            if not np.all(np.isnan(M[i])):
                ax.plot(np.nanargmax(M[i]), i, "o", color="0.1", markersize=2.6)
    axes[0].set_yticks(range(len(waves)))
    axes[0].set_yticklabels([style.TOPIC.get(w, f"W{w}") for w in waves], fontsize=7)
    axes[1].set_yticks([])

    ax = axes[2]
    im2 = ax.imshow(np.array(rho)[:, None], cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    for i, r in enumerate(rho):
        ax.text(0, i, f"{r:+.2f}", ha="center", va="center", fontsize=6.2,
                color="white" if abs(r) > 0.65 else "0.15")
    ax.set_xticks([0]); ax.set_xticklabels([r"rank $\rho$"], fontsize=7)
    ax.set_yticks([])
    ax.spines["top"].set_visible(True); ax.spines["right"].set_visible(True)

    cb = fig.colorbar(im, ax=axes[:2], fraction=0.035, pad=0.21, location="bottom",
                      aspect=45)
    cb.set_label("steering magnitude (mean TV to population)", fontsize=7.5)
    cb.ax.tick_params(labelsize=7)

    fig.subplots_adjust(left=0.135, right=0.985, bottom=0.31, top=0.93, wspace=0.08)
    style.save(fig, "fig_paired_heatmap" if MODEL == "gpt-4o-mini"
               else f"appendix/fig_paired_heatmap_{MODEL}")
    print(f"  mean rank rho = {np.mean(rho):+.3f}  "
          f"(range {min(rho):+.2f} .. {max(rho):+.2f})")
    print("  human top dim per wave:",
          {style.TOPIC.get(w, w): DIMS[int(np.nanargmax(H[i]))] for i, w in enumerate(waves)})
    print("  LLM top dim per wave:  ",
          {style.TOPIC.get(w, w): DIMS[int(np.nanargmax(L[i]))] for i, w in enumerate(waves)})


if __name__ == "__main__":
    main()
