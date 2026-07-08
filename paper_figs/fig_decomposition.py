"""Decomposition figure (referee major #1): the accuracy gap is single-feature
miscalibration, not the composition rule.

Per model, three TV-to-truth quantities for depth-2 pair cells:
  1. the model's actual pair output,
  2. the additive rule applied to the model's OWN (biased) single outputs,
  3. the same additive rule applied to the TRUE single-subgroup distributions.
(1)~=~(2): composing the model's own ingredients does not help. (3) collapses
to near the human noise floor: with calibrated ingredients the same composition
rule is almost exact. So the lever is single-feature calibration, not composition.

Numbers from verification/collapse_vs_miscalib.py.
"""
import numpy as np
import matplotlib.pyplot as plt
import style

# (model, model-actual, additive-on-model-singles, additive-on-true-singles)
DATA = [
    ("GPT-4o-mini", 0.238, 0.247, 0.062),
    ("Gemma-2-9B",  0.296, 0.322, 0.062),
    ("Mistral-7B",  0.345, 0.358, 0.061),
]
FLOOR = 0.120  # depth-2 split-half human noise floor (Methods)

C_ACT  = "#c23b22"   # model's actual pair output
C_OWN  = "#5d69b1"   # additive on model's own (biased) singles
C_TRUE = "#52bca3"   # additive on TRUE singles (calibrated ingredients)
LABELS = ["model's pair output",
          "additive rule, model's own singles",
          "additive rule, true singles"]


def main():
    fig, ax = plt.subplots(figsize=(5.0, 3.1))
    x = np.arange(len(DATA)); w = 0.26
    act  = [d[1] for d in DATA]
    own  = [d[2] for d in DATA]
    true = [d[3] for d in DATA]

    b1 = ax.bar(x - w, act,  w, color=C_ACT,  label=LABELS[0])
    b2 = ax.bar(x,     own,  w, color=C_OWN,  label=LABELS[1])
    b3 = ax.bar(x + w, true, w, color=C_TRUE, label=LABELS[2])

    ax.axhline(FLOOR, ls="--", lw=1.0, color="0.4")
    ax.text(len(DATA) - 0.5, FLOOR + 0.004, "human noise floor (d2)",
            fontsize=7, color="0.4", ha="right", va="bottom")

    for bars in (b1, b2, b3):
        for r in bars:
            ax.text(r.get_x() + r.get_width() / 2, r.get_height() + 0.006,
                    f"{r.get_height():.2f}", ha="center", va="bottom", fontsize=6.5)

    ax.set_xticks(x); ax.set_xticklabels([d[0] for d in DATA], fontsize=8.5)
    ax.set_ylabel("TV distance to real subgroup")
    ax.set_ylim(0, 0.40)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    ax.legend(loc="upper left", frameon=False, fontsize=7.0,
              handlelength=1.1, labelspacing=0.3)
    fig.subplots_adjust(left=0.115, right=0.98, bottom=0.10, top=0.97)
    style.save(fig, "fig_decomposition")


if __name__ == "__main__":
    main()
