"""
Fig 5b — Topic-blind collapse direction.

Heatmap: waves (topics) x dimensions; cell = LLM keep-rate minus human
dominance rate (percentage points) for that dimension in contested pairs on
that wave. Red = over-kept, blue = under-kept. Pooled over the 3 full-dim
models (all but gemma2). W49 included since the 2026-06-11 repair; its row
pools gpt-4o-mini only (other models lack W49) — footnote in caption.

Data: verification/collapse_direction.json
"""

import os, sys, json
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt

import style

ONLY_MODEL = sys.argv[1] if len(sys.argv) > 1 else None   # appendix variant

DIMS  = ["Race", "Religion", "Education", "Political Party", "Age", "Income", "Gender"]
SHORT = {"Race": "Race", "Religion": "Religion", "Education": "Education",
         "Political Party": "Party", "Age": "Age", "Income": "Income", "Gender": "Gender"}


def main():
    with open(os.path.join(style.VERIF, f"collapse_direction{style.SUFFIX}.json")) as f:
        rows = json.load(f)
    rows = [r for r in rows if "dimA" in r]  # depth-2 rows only (d3 rows use "dims")
    if ONLY_MODEL:
        rows = [r for r in rows if r["wave"] not in style.EXCLUDE_WAVES
                and r["model"] == ONLY_MODEL]
    else:
        rows = [r for r in rows if r["wave"] not in style.EXCLUDE_WAVES
                and r["model"] != "gemma2_9b"]

    waves = sorted({r["wave"] for r in rows}, key=int)
    by_wave = defaultdict(list)
    for r in rows:
        by_wave[r["wave"]].append(r)

    M = np.full((len(waves), len(DIMS)), np.nan)
    for i, w in enumerate(waves):
        rs = by_wave[w]
        for j, d in enumerate(DIMS):
            contested = [r for r in rs if d in (r["dimA"], r["dimB"])]
            if len(contested) < 100:
                continue
            keep   = np.mean([r["winner"] == d for r in contested])
            should = np.mean([r["human_dom"] == d for r in contested])
            M[i, j] = (keep - should) * 100

    fig, ax = plt.subplots(figsize=(3.4, 3.4))
    vmax = 16
    im = ax.imshow(M, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")

    ax.set_xticks(range(len(DIMS)))
    ax.set_xticklabels([SHORT[d] for d in DIMS], rotation=45, ha="right")
    ax.set_yticks(range(len(waves)))
    ax.set_yticklabels([style.TOPIC.get(w, f"W{w}") for w in waves], fontsize=7)
    ax.spines["top"].set_visible(True)
    ax.spines["right"].set_visible(True)

    for i in range(len(waves)):
        for j in range(len(DIMS)):
            if not np.isnan(M[i, j]):
                ax.text(j, i, f"{M[i, j]:+.0f}", ha="center", va="center",
                        fontsize=6.2,
                        color="white" if abs(M[i, j]) > vmax * 0.6 else "0.15")

    cb = fig.colorbar(im, ax=ax, fraction=0.045, pad=0.03)
    cb.set_label("LLM keep rate $-$ human dominance rate (pp)", fontsize=7.5)
    cb.ax.tick_params(labelsize=7)

    title = "over-kept (red)  /  under-kept (blue)"
    if ONLY_MODEL:
        title += f" — {ONLY_MODEL}"
    ax.set_title(title, fontsize=7.5 if ONLY_MODEL else 8.5, loc="left")
    fig.subplots_adjust(left=0.265, right=0.92, bottom=0.155, top=0.94)
    style.save(fig, "fig_direction_heatmap" if not ONLY_MODEL
               else f"appendix/fig_direction_heatmap_{ONLY_MODEL}")


if __name__ == "__main__":
    main()
