"""
Fig 5a — Dominance: which dimension survives the collapse, and does the
hierarchy match what humans would keep?

Left: 7x7 matrix; cell (row A, col B) = share of collapsed (A,B)-pair cells
where A wins. Pooled over the 3 full-dim models, W49 excluded.
Right: slope chart connecting each dimension's HUMAN dominance rank
(how often it is the human-dominant dim in its contests) to its LLM keep rank.

Data: verification/collapse_direction.json
"""

import os, sys, json
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt

import style

ONLY_MODEL = sys.argv[1] if len(sys.argv) > 1 else None   # appendix variant

DIMS  = ["Race", "Religion", "Political Party", "Education", "Age", "Income", "Gender"]
SHORT = {"Race": "Race", "Religion": "Religion", "Political Party": "Party",
         "Education": "Education", "Age": "Age", "Income": "Income", "Gender": "Gender"}
import sys
sys.path.insert(0, os.path.join(style.BASE, "advanced_bias_analysis"))
from utils import DIM_COLORS


def main():
    with open(os.path.join(style.VERIF, f"collapse_direction{style.SUFFIX}.json")) as f:
        rows = json.load(f)
    rows = [r for r in rows if "dimA" in r]  # depth-2 rows only (d3 rows use "dims")
    # main-text panel pools the three full-dimension models on the 14-wave
    # basis (matches caption and pooled text statistics)
    FULL_MODELS = {"gpt-4o-mini", "gpt-4o", "claude-haiku-4-5-20251001", "llama3_1_8b_instruct_q4", "mistral_latest"}
    if ONLY_MODEL:
        rows = [r for r in rows if r["wave"] not in style.EXCLUDE_WAVES
                and r["wave"] != "49" and r["model"] == ONLY_MODEL]
    else:
        rows = [r for r in rows if r["wave"] not in style.EXCLUDE_WAVES
                and r["wave"] != "49" and r["model"] in FULL_MODELS]

    # ---- left: pairwise win matrix ----
    wins = defaultdict(int); tot = defaultdict(int)
    for r in rows:
        a, b = r["dimA"], r["dimB"]
        tot[(a, b)] += 1; tot[(b, a)] += 1
        wins[(r["winner"], b if r["winner"] == a else a)] += 1
    M = np.full((len(DIMS), len(DIMS)), np.nan)
    for i, a in enumerate(DIMS):
        for j, b in enumerate(DIMS):
            if i != j and tot[(a, b)] >= 200:
                M[i, j] = wins[(a, b)] / tot[(a, b)] * 100

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.5, 3.0),
                                   gridspec_kw={"width_ratios": [1.35, 1]})
    im = ax1.imshow(M, cmap="RdBu_r", vmin=20, vmax=80, aspect="auto")
    ax1.set_xticks(range(len(DIMS)))
    ax1.set_xticklabels([SHORT[d] for d in DIMS], rotation=45, ha="right", fontsize=7)
    ax1.set_yticks(range(len(DIMS)))
    ax1.set_yticklabels([SHORT[d] for d in DIMS], fontsize=7)
    for i in range(len(DIMS)):
        for j in range(len(DIMS)):
            if not np.isnan(M[i, j]):
                ax1.text(j, i, f"{M[i, j]:.0f}", ha="center", va="center", fontsize=6.5,
                         color="white" if abs(M[i, j] - 50) > 18 else "0.15")
    ax1.set_title("row dim beats column dim (%)", fontsize=8.5)
    ax1.spines["top"].set_visible(True); ax1.spines["right"].set_visible(True)
    cb = fig.colorbar(im, ax=ax1, fraction=0.045, pad=0.03)
    cb.ax.tick_params(labelsize=7)

    # ---- right: slope chart, human dominance rank -> LLM keep rank ----
    keep = defaultdict(int); should = defaultdict(int); contested = defaultdict(int)
    for r in rows:
        for d in (r["dimA"], r["dimB"]):
            contested[d] += 1
        keep[r["winner"]] += 1
        should[r["human_dom"]] += 1
    keep_rate   = {d: keep[d] / contested[d] for d in DIMS}
    should_rate = {d: should[d] / contested[d] for d in DIMS}
    h_rank = {d: i for i, d in enumerate(sorted(DIMS, key=should_rate.get, reverse=True))}
    l_rank = {d: i for i, d in enumerate(sorted(DIMS, key=keep_rate.get, reverse=True))}

    for d in DIMS:
        c = DIM_COLORS[d]
        ax2.plot([0, 1], [h_rank[d], l_rank[d]], "-o", color=c,
                 linewidth=1.6, markersize=3.5)
        ax2.annotate(f"{SHORT[d]} {should_rate[d]:.0%}", xy=(0, h_rank[d]),
                     xytext=(-0.07, h_rank[d]), ha="right", va="center",
                     fontsize=7.2, color=c)
        ax2.annotate(f"{keep_rate[d]:.0%} {SHORT[d]}", xy=(1, l_rank[d]),
                     xytext=(1.07, l_rank[d]), ha="left", va="center",
                     fontsize=7.2, color=c)
    ax2.set_xlim(-0.65, 1.7)
    ax2.set_ylim(len(DIMS) - 0.5, -0.5)
    ax2.set_xticks([0, 1])
    ax2.set_xticklabels(["humans say\nshould win", "LLM\nkeeps"], fontsize=7.5)
    ax2.set_yticks([])
    for s in ("left", "bottom"):
        ax2.spines[s].set_visible(False)
    ax2.tick_params(length=0)
    ax2.set_title("dominance rank: human vs LLM", fontsize=8.5)

    if ONLY_MODEL:
        fig.suptitle(ONLY_MODEL, fontsize=8, y=0.995)
    fig.subplots_adjust(left=0.105, right=0.97, bottom=0.17, top=0.90, wspace=0.32)
    style.save(fig, "fig_dominance" if not ONLY_MODEL
               else f"appendix/fig_dominance_{ONLY_MODEL}")


if __name__ == "__main__":
    main()
