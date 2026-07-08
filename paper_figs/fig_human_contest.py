"""
M1 figure: humans pass the additivity contest that models fail.

Three bars: human additive-win share (cells n>=100 and n>=200) and the
LLM's share on the same cells in the same steering space. Dashed marks:
the additive-truth reference at matched cell size (what a perfectly
additive composer would score through multinomial sampling noise).

Data: verification/human_additivity_calibration{SUFFIX}.json
"""

import os, json

import matplotlib.pyplot as plt

import style

with open(os.path.join(style.VERIF,
          f"human_additivity_calibration{style.SUFFIX}.json")) as f:
    D = json.load(f)

h100 = D["1"]["human_additive_share_n100"]
h200 = D["1"]["human_additive_share_n200"]
llm  = D["1"]["llm_steering_additive_share"]
r100 = D["1"]["additive_truth_ref_n100"]
r200 = D["1"]["additive_truth_ref_n200"]

vals   = [h100, h200, llm]
refs   = [r100, r200, None]
labels = ["humans\n($n{\\geq}100$)", "humans\n($n{\\geq}200$)",
          "LLM\n(same cells)"]
colors = ["#5d69b1", "#5d69b1", "#e58606"]

fig, ax = plt.subplots(figsize=(3.0, 2.5))
bars = ax.bar(range(3), vals, width=0.6, color=colors)
bars[1].set_alpha(0.75)
for i, v in enumerate(vals):
    ax.annotate(f"{v:.1%}", xy=(i, v), xytext=(0, 2),
                textcoords="offset points", ha="center", fontsize=8)

for i, r in enumerate(refs):
    if r is not None:
        ax.plot([i - 0.38, i + 0.38], [r, r], "--", color="0.35",
                linewidth=1.1)
ax.annotate("additive truth at\nmatched cell size", xy=(1.4, r200),
            xytext=(2.4, r200 + 0.045), fontsize=7, color="0.35",
            ha="right")

ax.set_xticks(range(3))
ax.set_xticklabels(labels, fontsize=8)
ax.set_ylabel("Additive prediction wins")
ax.set_ylim(0, 0.62)
fig.subplots_adjust(left=0.17, right=0.97, bottom=0.15, top=0.96)
style.save(fig, "fig_human_contest")
