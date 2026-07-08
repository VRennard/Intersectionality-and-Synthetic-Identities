"""'How the budget is spent' — stepped waterfall of per-cell sorted median
coefficients. Each feature is a step; negative steps pull back (hatched)."""
import os
import numpy as np
import matplotlib.pyplot as plt
import style

z2 = np.load(os.path.join(style.HERE, "alphabeta_points.npz"))
z3 = np.load(os.path.join(style.HERE, "abc_points.npz"))

def sorted_meds(p):
    p = p[np.abs(p).max(axis=1) < 3]
    return np.median(np.sort(p, axis=1)[:, ::-1], axis=0)

rows = [
    ("humans, 2 features", sorted_meds(z2["hp"]), "#2b6cb0"),
    ("model, 2 features",  sorted_meds(z2["mp"]), "#e58606"),
    ("humans, 3 features", sorted_meds(z3["hp"]), "#08306b"),
    ("model, 3 features",  sorted_meds(z3["mp"]), "#c23b22"),
]

fig, ax = plt.subplots(figsize=(6.6, 3.6))
sub = 0.30          # step height
gap = 1.35          # group spacing
for i, (name, meds, col) in enumerate(rows):
    ytop = -i * gap
    x = 0.0
    for k, m in enumerate(meds):
        y = ytop - k * sub
        if m >= 0:
            ax.barh(y, m, left=x, height=sub * 0.92, color=col,
                    alpha=1.0 - 0.28 * k, edgecolor="white", lw=0.5)
        else:
            ax.barh(y, m, left=x, height=sub * 0.92, color="white",
                    edgecolor="#c23b22", lw=1.1, hatch="////")
        lx = x + m / 2
        ax.text(lx, y, f"{m:.2f}", ha="center", va="center", fontsize=6.8,
                color="white" if m >= 0 else "#c23b22", fontweight="bold")
        x += m
        if k < len(meds) - 1:            # connector to next step
            ax.plot([x, x], [y - sub * 0.46, y - sub * 1.38], color="0.55",
                    lw=0.8, ls=":")
    ylast = ytop - (len(meds) - 1) * sub
    ax.plot(x, ylast - sub * 0.85, marker="^", ms=6, color="black", zorder=5)
    ax.text(x, ylast - sub * 1.75, f"total {x:.2f}", ha="center", fontsize=7.8,
            fontweight="bold")
    ax.text(-0.07, ytop, name, ha="right", va="center", fontsize=8.2)
for v, lbl in ((1, "one identity's worth"), (2, "two"), (3, "three")):
    ax.axvline(v, color="0.55", ls=":", lw=0.9, zorder=0)
    ax.text(v, 0.62, lbl, ha="center", fontsize=7, color="0.4")
ax.axvline(0, color="0.3", lw=1.0)
ax.set_xlim(-1.15, 3.3); ax.set_ylim(-5.45, 0.9)
ax.set_yticks([])
ax.set_xlabel("identity weight, spent feature by feature (per-cell sorted medians)")
for s in ("left", "right", "top"):
    ax.spines[s].set_visible(False)
ax.text(-1.1, -5.3, "hatched = negative: the weakest feature is actively subtracted",
        fontsize=7.2, color="#c23b22")
fig.subplots_adjust(left=0.185, right=0.985, bottom=0.145, top=0.93)
style.save(fig, "fig_spending")
