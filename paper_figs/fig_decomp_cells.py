"""Decomposition, star-space style: per pair cell, error of the model's native
pair output (x) vs error of the additive rule on TRUE singles (y)."""
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import style

p = np.load(os.path.join(style.HERE, "decomp_cells.npz"))["p"]
FLOOR = 0.120
fig, ax = plt.subplots(figsize=(4.7, 4.4))
cmap = LinearSegmentedColormap.from_list("d", ["#ffffff", "#c6dbee", "#3f7fbf", "#08306b"])
LIM = 0.85
ax.hexbin(p[:, 0], p[:, 1], gridsize=52, extent=(0, LIM, 0, LIM), cmap=cmap,
          bins="log", linewidths=0)
ax.plot([0, LIM], [0, LIM], ls="--", color="0.35", lw=1.1)
ax.axhline(FLOOR, color="#c23b22", ls=":", lw=1.1)
med = np.median(p, axis=0)
ax.plot(*med, marker="P", ms=9, color="white", mec="black", mew=0.9, zorder=6)
ax.text(0.63, 0.67, "same error\n(recombination\ndoesn't help)", fontsize=7.3,
        color="0.3", rotation=45, ha="center")
ax.text(LIM - 0.02, FLOOR + 0.015, "human sampling noise floor (depth 2)",
        fontsize=7.3, color="#c23b22", ha="right")
ax.annotate(f"median ({med[0]:.2f}, {med[1]:.2f}):\nwhatever the model's own error,\n"
            "calibrated ingredients land at the floor",
            xy=(med[0], med[1]), xytext=(0.06, 0.47), fontsize=7.6,
            arrowprops=dict(arrowstyle="->", color="0.25", lw=0.8,
                            connectionstyle="arc3,rad=-0.25"))
ax.set_xlim(0, LIM); ax.set_ylim(0, LIM); ax.set_aspect("equal")
ax.set_xlabel("TV error of the model's own pair simulation")
ax.set_ylabel("TV error of additive rule on true singles")
fig.subplots_adjust(left=0.13, right=0.97, bottom=0.12, top=0.97)
style.save(fig, "fig_decomp_cells")
print("medians:", med)
