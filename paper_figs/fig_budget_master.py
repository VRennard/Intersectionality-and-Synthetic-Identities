"""Master 'identity budget' figure: (a) coefficient-plane densities (humans vs
7 LLMs pooled), (b) total-budget ridgelines by depth, (c) per-cell anatomy."""
import os, glob
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from scipy.stats import gaussian_kde
import style

BLUE, DBLUE, ORANGE, RED = "#3f7fbf", "#08306b", "#e58606", "#c23b22"
z2 = np.load(os.path.join(style.HERE, "alphabeta_points.npz"))
z3 = np.load(os.path.join(style.HERE, "abc_points.npz"))
hp2, hp3 = z2["hp"], z3["hp"]
mp2_pool = np.vstack([np.load(f)["mp"] for f in sorted(glob.glob(
    os.path.join(style.HERE, "alphabeta_pts_*.npz")))])
mp2_flag, mp3 = z2["mp"], z3["mp"]

fig = plt.figure(figsize=(7.2, 6.6))
gs = fig.add_gridspec(2, 2, height_ratios=[1.15, 1], hspace=0.34, wspace=0.26,
                      left=0.075, right=0.975, top=0.95, bottom=0.075)

# ---------- (a) hexbin pair ----------
cmap = LinearSegmentedColormap.from_list("d", ["#ffffff", "#c6dbee", "#3f7fbf", "#08306b"])
LIM = (-0.9, 2.4)
for j, (pts, title) in enumerate(((hp2, f"real subgroups\n({len(hp2):,} pair cells)"),
                                   (mp2_pool, f"7 LLMs pooled\n({len(mp2_pool):,} pair cells)"))):
    ax = fig.add_subplot(gs[0, j])
    ax.hexbin(pts[:, 0], pts[:, 1], gridsize=52, extent=(*LIM, *LIM), cmap=cmap,
              bins="log", linewidths=0)
    ax.axhline(0, color="0.6", lw=0.6); ax.axvline(0, color="0.6", lw=0.6)
    ax.plot([LIM[0], LIM[1]], [1 - LIM[0], 1 - LIM[1]], ls="--", lw=1.1,
            color=RED, alpha=0.8)
    ax.plot(1, 1, marker="*", ms=14, color=ORANGE, mec="white", mew=0.8, zorder=5)
    ax.plot([1, 0], [0, 1], "o", ms=5.5, color=RED, mec="white", zorder=5)
    med = np.median(pts, axis=0)
    ax.plot(*med, marker="P", ms=8, color="white", mec="black", mew=0.9, zorder=6)
    bud = np.median(pts.sum(axis=1))
    ax.annotate(f"median budget {bud:.2f}", xy=(0.04, 0.045), xycoords="axes fraction",
                fontsize=7.8, bbox=dict(boxstyle="round,pad=0.24", fc="white", ec="0.7", lw=0.6))
    if j == 0:
        ax.annotate("perfect\ncomposition", xy=(1.03, 1.06), xytext=(1.4, 1.65),
                    fontsize=7, color="#b26a04",
                    arrowprops=dict(arrowstyle="-", color="#b26a04", lw=0.7))
        ax.annotate("one identity's\nworth of weight", xy=(-0.35, 1.28), fontsize=6.7,
                    color=RED, rotation=-45)
        ax.set_ylabel(r"weight on feature $B$")
    else:
        ax.set_yticklabels([])
    ax.set_xlim(LIM); ax.set_ylim(LIM); ax.set_aspect("equal")
    ax.set_xlabel(r"weight on feature $A$")
    ax.set_title(("a  " if j == 0 else "") + title, fontsize=8.6, loc="left")

# ---------- (b) budget ridgelines ----------
axb = fig.add_subplot(gs[1, 0])
groups = [("humans, 2 features", hp2.sum(1), BLUE),
          ("humans, 3 features", hp3.sum(1), DBLUE),
          ("model, 2 features", mp2_flag.sum(1), ORANGE),
          ("model, 3 features", mp3.sum(1), RED)]
xs = np.linspace(-1.5, 5.5, 400)
for i, (name, a, col) in enumerate(groups):
    med = np.median(a)
    a = a[(a > -1.5) & (a < 5.5)]
    sub = a if len(a) < 150000 else np.random.default_rng(0).choice(a, 150000, replace=False)
    y = gaussian_kde(sub, bw_method=0.12)(xs); y /= y.max()
    base = -i * 1.14
    axb.fill_between(xs, base, base + y, color=col, alpha=0.8, lw=0)
    axb.plot([med, med], [base, base + 1.0], color="white", lw=1.4)
    axb.text(5.4, base + 0.36, f"{name}  ({med:.2f})", ha="right", fontsize=7.2,
             color=col, fontweight="bold")
for v in (1, 2, 3):
    axb.axvline(v, color="0.45", ls=":", lw=0.8, zorder=0)
axb.text(1, 1.24, "1", ha="center", fontsize=7, color="0.35")
axb.text(2, 1.24, "2", ha="center", fontsize=7, color="0.35")
axb.text(3, 1.24, "3", ha="center", fontsize=7, color="0.35")
axb.set_yticks([]); axb.set_xlim(-1.5, 5.5); axb.set_ylim(-3.65, 1.5)
axb.set_xlabel("total identity weight")
for s_ in ("left", "right", "top"): axb.spines[s_].set_visible(False)
axb.set_title("b  the budget by depth", fontsize=8.6, loc="left")

# ---------- (c) anatomy dumbbell ----------
axc = fig.add_subplot(gs[1, 1])
def sorted_meds(p):
    p = p[np.abs(p).max(axis=1) < 3]
    return np.median(np.sort(p, axis=1)[:, ::-1], axis=0)
rows = [("2 features",  sorted_meds(hp2), sorted_meds(mp2_flag), ["larger", "smaller"]),
        ("3 features",  sorted_meds(hp3), sorted_meds(mp3), ["largest", "middle", "smallest"])]
y = 0; yticks, ylabels = [], []
for depth, hmed, mmed, labs in rows:
    for k, lab in enumerate(labs):
        axc.plot([hmed[k], mmed[k]], [y, y], color="0.75", lw=1.4, zorder=1)
        axc.plot(hmed[k], y, "o", ms=7, color=DBLUE, zorder=3)
        axc.plot(mmed[k], y, "o", ms=7, color=RED, zorder=3)
        yticks.append(y); ylabels.append(f"{depth}: {lab}")
        y -= 1
    y -= 0.6
axc.axvline(0, color="0.5", lw=0.8)
axc.axvline(1, color="0.5", ls=":", lw=0.8)
axc.set_yticks(yticks); axc.set_yticklabels(ylabels, fontsize=7.4)
axc.set_xlabel("median coefficient (per-cell, sorted)")
axc.set_xlim(-1.0, 2.0)
axc.plot([], [], "o", color=DBLUE, label="humans"); axc.plot([], [], "o", color=RED, label="model")
axc.legend(frameon=False, fontsize=7.4, loc="lower right", handletextpad=0.3)
for s_ in ("right", "top"): axc.spines[s_].set_visible(False)
axc.set_title("c  per-cell anatomy: one feature takes it all", fontsize=8.6, loc="left")

style.save(fig, "fig_budget_master")
