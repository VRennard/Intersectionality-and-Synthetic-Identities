"""
Fig 4 — What explains the LLM's pair bias? (calibrated against the null)

Left panel: ECDFs of per-cell cosine similarity between the pair-profile bias
and three predictors: a random unrelated dimension's bias (null floor), the
additive sum e_A+e_B, and the best single dimension. Right panel: histogram of
the magnitude ratio ||actual pair bias|| / ||additive prediction|| (flattening).

Data: verification/additivity_null_baseline.json (gpt-4o-mini, 5 waves).
"""

import os, sys, json
import numpy as np
import matplotlib.pyplot as plt

import style
from style import C_NULL, C_ADD, C_SINGLE

MODEL = sys.argv[1] if len(sys.argv) > 1 else "gpt-4o-mini"


def ecdf(ax, v, color, label):
    v = np.sort(v)
    y = np.arange(1, len(v) + 1) / len(v)
    ax.plot(v, y, color=color, linewidth=1.5, label=label)


def main():
    suffix = "" if MODEL == "gpt-4o-mini" else f"_{MODEL}"
    with open(os.path.join(style.VERIF, f"additivity_null_baseline{suffix}{style.SUFFIX}.json")) as f:
        d = json.load(f)
    null, add, best = (np.array(d[k]) for k in ("cos_null", "cos_add", "cos_best"))
    mag = np.array(d["mag_ratio"])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.5, 2.4),
                                   gridspec_kw={"width_ratios": [1.5, 1]})

    # ---- left: ECDFs of cosine to each predictor ----
    ecdf(ax1, null, C_NULL,   "random unrelated dim (null)")
    ecdf(ax1, add,  C_ADD,    "additive  $e_A{+}e_B$")
    ecdf(ax1, best, C_SINGLE, "best single dimension")
    for v, c in ((np.median(null), C_NULL), (np.median(add), C_ADD),
                 (np.median(best), C_SINGLE)):
        ax1.plot([v], [0.5], "o", color=c, markersize=3.5, zorder=3)
    ax1.set_xlim(-0.55, 1.02)
    ax1.set_ylim(0, 1)
    ax1.set_xlabel("cosine similarity to pair-profile bias")
    ax1.set_ylabel("fraction of cells")
    ax1.legend(loc="upper left", frameon=False, handlelength=1.4)
    ax1.set_title("(a) collapse beats addition; both ride a high null", fontsize=8.5, loc="left")

    # ---- right: magnitude ratio (flattening) ----
    shown = mag[mag <= 2]            # drops <2% extreme tail
    ax2.hist(shown, bins=60, range=(0, 2), color=C_ADD, alpha=0.85)
    med = np.median(mag)
    ax2.axvline(1.0, color="0.3", linewidth=1.0, linestyle="--")
    ax2.axvline(med, color="#b3320b", linewidth=1.3)
    ax2.annotate("faithful\nmagnitude", xy=(1.0, ax2.get_ylim()[1]*0.97),
                 ha="left", va="top", fontsize=7.5, color="0.3", xytext=(1.06, ax2.get_ylim()[1]*0.97))
    ax2.annotate(f"median {med:.2f}", xy=(med, ax2.get_ylim()[1]*0.72),
                 ha="right", fontsize=7.5, color="#b3320b", xytext=(med-0.05, ax2.get_ylim()[1]*0.72))
    ax2.set_xlabel(r"$\|e_{pair}\|\,/\,\|e_A+e_B\|$")
    ax2.set_yticks([])
    ax2.spines["left"].set_visible(False)
    ax2.set_title("(b) and under-steers: half the\nexpected distinctiveness", fontsize=8.5, loc="left")

    fig.subplots_adjust(left=0.09, right=0.99, bottom=0.21, top=0.82, wspace=0.18)
    style.save(fig, "fig_predictors" if MODEL == "gpt-4o-mini"
               else f"appendix/fig_predictors_{MODEL}")

    print(f"  medians: null={np.median(null):.3f} add={np.median(add):.3f} "
          f"best={np.median(best):.3f}; best>add in {(best>add).mean():.1%}; "
          f"add>null in {(add>null).mean():.1%}")


if __name__ == "__main__":
    main()
