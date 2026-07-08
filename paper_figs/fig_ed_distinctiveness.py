"""
Extended Data Fig. 1: real subgroups grow more distinctive as identities
intersect. Noise-corrected squared distinctiveness of human cells by
profile depth, with raw TV and expected sampling noise for reference.

Data: verification/human_rigor_2_3.json (rigor item 2).
"""

import os, json

import numpy as np
import matplotlib.pyplot as plt

import style

with open(os.path.join(style.VERIF, "human_rigor_2_3.json")) as f:
    D = json.load(f)["distinctiveness"]

depths  = [1, 2, 3]
l2corr  = [D[str(d)]["l2sq_corr"] for d in depths]
tv      = [D[str(d)]["tv"] for d in depths]
noise   = [D[str(d)]["noise_tv"] for d in depths]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.0, 2.5))

ax1.bar(depths, l2corr, width=0.55, color="#5d69b1")
for d, v in zip(depths, l2corr):
    ax1.annotate(f"{v:.4f}", xy=(d, v), xytext=(0, 2),
                 textcoords="offset points", ha="center", fontsize=7.5)
ax1.set_xticks(depths)
ax1.set_xlabel("Identity features in profile")
ax1.set_ylabel("Unbiased squared distinctiveness\n"
               r"$\|p_g - p_{\mathrm{pop}}\|^2$ (noise-corrected)")
ax1.text(0.05, 0.92, f"{l2corr[2]/l2corr[0]:.1f}$\\times$ from one to three",
         transform=ax1.transAxes, fontsize=8, color="#5d69b1")

ax2.plot(depths, tv, "-o", color="0.25", markersize=3.5, label="raw TV")
ax2.plot(depths, noise, ":o", color="0.55", markersize=3.5,
         label="expected sampling noise")
ax2.set_xticks(depths)
ax2.set_xlabel("Identity features in profile")
ax2.set_ylabel("TV to population")
ax2.legend(frameon=False, fontsize=7.5)

fig.subplots_adjust(left=0.13, right=0.98, bottom=0.19, top=0.95, wspace=0.42)
style.save(fig, "fig_ed_distinctiveness")
