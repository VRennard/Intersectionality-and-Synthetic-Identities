"""
Concept schematic (new Fig 1): the vector objects (steering vs bias) and
the three-predictor contest, for readers who don't want to hold the
machinery from prose. Pure diagram, no data.
"""
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch

C_HUM, C_LLM = "0.25", "#c23b22"
C_ADD, C_SINGLE_A, C_SINGLE_B = "#e58606", "#5d69b1", "#52bca3"


def arrow(ax, p0, p1, color, lw=2.4, ls="-", alpha=1.0, z=3):
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle="-|>", mutation_scale=17,
                                 color=color, lw=lw, linestyle=ls,
                                 alpha=alpha, zorder=z,
                                 shrinkA=4, shrinkB=4))


def panel_objects(ax):
    # parallelogram: human arrow top-left, model arrow parallel below,
    # vertical dashed bias arrow joining the subgroup dots at x=0.60.
    # label zones (all va="center", widths checked): steering ABOVE the
    # human arrow; simulated steering in the empty band BETWEEN arrows;
    # bias block in the right column; dot labels hug their dots.
    Ppop, Pg = np.array([0.10, 0.52]), np.array([0.55, 0.78])
    Qpop, Qg = np.array([0.10, 0.02]), np.array([0.55, 0.28])
    for p, c, lbl, dx, dy, ha in [
            (Ppop, C_HUM, "population $p_{\\mathrm{pop}}$", -0.05, -0.09, "left"),
            (Pg,   C_HUM, "real subgroup $p_g$", 0.06, 0.02, "left"),
            (Qpop, C_LLM, "simulated population $\\hat p_{\\mathrm{pop}}$", -0.05, -0.09, "left"),
            (Qg,   C_LLM, "simulated subgroup $\\hat p_g$", 0.06, -0.04, "left")]:
        ax.plot(*p, "o", ms=12, color=c, mec="white", mew=1.4, zorder=5)
        ax.annotate(lbl, p, p + [dx, dy], ha=ha, va="center", fontsize=10,
                    color=c)
    arrow(ax, Ppop, Pg, C_HUM)
    ax.annotate("steering $s_g = p_g - p_{\\mathrm{pop}}$:\nidentity's pull on opinion",
                (0.01, 0.95), ha="left", va="center", fontsize=9.5,
                color=C_HUM, linespacing=1.35)
    arrow(ax, Qpop, Qg, C_LLM)
    ax.annotate("simulated steering $\\hat s_g$",
                (0.47, 0.05), ha="center", va="center", fontsize=9.5,
                color=C_LLM)
    arrow(ax, Pg, Qg, "0.5", ls=(0, (2, 2)), lw=2.0)
    ax.annotate("bias $e_g = \\hat p_g - p_g$:\nsimulation vs. reality",
                (0.615, 0.53), ha="left", va="center", fontsize=9.5,
                color="0.4", linespacing=1.35)
    ax.set_title("a  Two kinds of difference vector", loc="left",
                 fontsize=12, fontweight="bold")
    ax.set_xlim(0, 1.18); ax.set_ylim(-0.22, 1.04)
    ax.axis("off")
    ax.annotate("all vectors live in one question's response space",
                (0.56, -0.175), ha="center", va="center", fontsize=8.5,
                color="0.55", style="italic")


def panel_contest(ax, collapse, title):
    eA = np.array([0.78, 0.10])
    eB = np.array([0.18, 0.60])
    eSum = eA + eB
    O = np.array([0.0, 0.0])
    arrow(ax, O, eA, C_SINGLE_A, lw=2.2)
    arrow(ax, O, eB, C_SINGLE_B, lw=2.2)
    arrow(ax, O, eSum, C_ADD, lw=2.2)
    ax.plot([eA[0], eSum[0]], [eA[1], eSum[1]], color=C_ADD, lw=0.7,
            ls=":", alpha=0.6)
    ax.plot([eB[0], eSum[0]], [eB[1], eSum[1]], color=C_ADD, lw=0.7,
            ls=":", alpha=0.6)
    ax.annotate("$e_A$", eA + [0.03, -0.02], fontsize=12, color=C_SINGLE_A)
    ax.annotate("$e_B$", eB + [-0.02, 0.05], fontsize=12, color=C_SINGLE_B)
    ax.annotate("$e_A{+}e_B$", eSum + [-0.2, 0.06], fontsize=12, color=C_ADD)
    eAB = 0.93 * eA + 0.16 * eB if collapse else 0.62 * eSum
    arrow(ax, O, eAB, "black", lw=3.2, z=6)
    lbl_off = np.array([0.0, 0.07]) if collapse else np.array([0.03, -0.12])
    ax.annotate("$e_{AB}$", eAB * (0.55 if collapse else 0.62) + lbl_off,
                fontsize=13, color="black", fontweight="bold")
    winner = ("closest to $e_A$: the pair behaves\nas one identity \u2014 collapse"
              if collapse else
              "closest to the sum: both identities\nintegrated \u2014 composition")
    ax.annotate(winner, (0.5, -0.2), ha="center",
                fontsize=10,
                color=C_SINGLE_A if collapse else C_ADD)
    ax.set_title(title, loc="left", fontsize=12, fontweight="bold")
    ax.set_xlim(-0.06, 1.22); ax.set_ylim(-0.32, 0.95)
    ax.set_aspect("equal")
    ax.axis("off")


def main():
    fig, axes = plt.subplots(1, 3, figsize=(8.6, 3.5),
                             gridspec_kw={"width_ratios": [1.25, 1, 1]})
    panel_objects(axes[0])
    panel_contest(axes[1], collapse=False,
                  title="b  If conditioning composed")
    panel_contest(axes[2], collapse=True,
                  title="c  What models do")
    fig.subplots_adjust(left=0.01, right=0.99, top=0.89, bottom=0.05,
                        wspace=0.1)
    import style
    style.save(fig, "fig_schematic")


if __name__ == "__main__":
    main()
