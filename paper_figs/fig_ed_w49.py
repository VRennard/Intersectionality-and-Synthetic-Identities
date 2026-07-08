"""
ED Fig 3 - Wave W49 forensics. Left: per-dimension pair-cell error rate of
the ORIGINAL (pre-repair) GPT-4o-mini W49 run vs the same model's rate on the
other 14 waves - the failure was infrastructure, dimension-biased toward
Education's long option strings. Right: flagship headline statistics computed
with and without the repaired W49 - inclusion moves nothing.

Data: paper_figs/w49_original_errors.json (timestamp-stratified from
data/results/gpt-4o-mini/W49.jsonl), verification/missingness_by_dim.py
output (14-wave baseline), verification/summary_excl_w49.log.
"""
import json
import os

import matplotlib.pyplot as plt
import numpy as np

import style

DIMS = ["Education", "Income", "Gender", "Race", "Age",
        "Political Party", "Religion"]
BASELINE = {"Education": 42.8, "Income": 10.4, "Gender": 10.0, "Race": 10.9,
            "Age": 9.9, "Political Party": 9.5, "Religion": 11.2}
# with -> without repaired W49 (verification/summary_excl_w49.log, flagship)
HEADLINES = [
    ("depth-1 mean TV",        0.2141, 0.2142),
    ("depth-2 mean TV",        0.2351, 0.2358),
    ("depth-3 mean TV",        0.2300, 0.2311),
    ("d2 best-single (share)", 0.784,  0.783),
    ("d3 additive (share)",    0.083,  0.083),
]
C_W49, C_BASE = "#c23b22", "0.55"


def main():
    with open(os.path.join(style.HERE, "w49_original_errors.json")) as f:
        w49 = json.load(f)

    fig, (ax, ax2) = plt.subplots(
        1, 2, figsize=(7.2, 3.1), gridspec_kw={"width_ratios": [1.25, 1]})

    # -- left: per-dimension error rate, original W49 vs 14-wave baseline --
    y = np.arange(len(DIMS))[::-1]
    ax.barh(y + 0.19, [w49["by_dim"][d] for d in DIMS], height=0.36,
            color=C_W49, label="W49, original run")
    ax.barh(y - 0.19, [BASELINE[d] for d in DIMS], height=0.36,
            color=C_BASE, label="other 14 waves")
    ax.axvline(w49["overall"], color=C_W49, lw=0.9, ls=":", alpha=0.8)
    ax.text(w49["overall"] + 0.8, len(DIMS) - 0.55,
            f"overall {w49['overall']:.0f}%", fontsize=6.8, color=C_W49)
    ax.set_yticks(y)
    ax.set_yticklabels(DIMS, fontsize=7.5)
    ax.set_xlabel("pair-cell error rate (%)", fontsize=8)
    ax.legend(loc="lower right", frameon=False, fontsize=7)
    ax.set_title("a  Where the failed wave lost cells", fontsize=8.5,
                 loc="left")

    # -- right: headline stats with vs without repaired W49 --
    y2 = np.arange(len(HEADLINES))[::-1]
    for i, (lab, with49, without) in zip(y2, HEADLINES):
        ax2.plot([with49, without], [i, i], color="0.75", lw=1.4, zorder=1)
        ax2.plot(with49, i, "o", ms=5, color="#3f7fbf", zorder=3)
        ax2.plot(without, i, "o", ms=5, mfc="white", mec="#3f7fbf",
                 mew=1.2, zorder=3)
        ax2.text(max(with49, without) + 0.028, i,
                 f"$\\Delta$ = {abs(with49-without):.3f}", fontsize=6.4,
                 color="0.35", va="center")
    ax2.set_yticks(y2)
    ax2.set_yticklabels([h[0] for h in HEADLINES], fontsize=7)
    ax2.set_xlim(0, 1.0)
    ax2.set_xlabel("value (TV or share)", fontsize=8)
    ax2.plot([], [], "o", ms=5, color="#3f7fbf", label="W49 included")
    ax2.plot([], [], "o", ms=5, mfc="white", mec="#3f7fbf", mew=1.2,
             label="W49 excluded")
    ax2.legend(loc="lower right", frameon=False, fontsize=7)
    ax2.set_title("b  Headline numbers barely move", fontsize=8.5,
                  loc="left")

    fig.subplots_adjust(left=0.14, right=0.985, bottom=0.16, top=0.9,
                        wspace=0.52)
    style.save(fig, "fig_ed_w49")


if __name__ == "__main__":
    main()
