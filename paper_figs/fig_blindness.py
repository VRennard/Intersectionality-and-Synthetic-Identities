"""Near-blindness, star-space style: each point is a (dimension, partner) pair.
x = share of contested cells humans say the dimension should win;
y = share the model actually keeps it.
Faithful retention = identity line. Blindness = flat line at 50%."""
import os, json
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt
import style
import sys
sys.path.insert(0, os.path.join(style.BASE, "advanced_bias_analysis"))
from utils import DIM_COLORS

with open(os.path.join(style.VERIF, f"collapse_direction{style.SUFFIX}.json")) as f:
    rows = json.load(f)
FULL = {"gpt-4o-mini","gpt-4o","claude-haiku-4-5-20251001","llama3_1_8b_instruct_q4","mistral_latest"}
rows = [r for r in rows if "dimA" in r and r["wave"] not in style.EXCLUDE_WAVES
        and r["wave"] != "49" and r["model"] in FULL]
keep = defaultdict(int); dom = defaultdict(int); n = defaultdict(int)
for r in rows:
    pair = tuple(sorted((r["dimA"], r["dimB"])))
    for d in pair:
        n[(d, pair)] += 1
    keep[(r["winner"], pair)] += 1
    dom[(r["human_dom"], pair)] += 1

fig, ax = plt.subplots(figsize=(4.6, 4.4))
ax.axhspan(43, 53, color="0.93", zorder=0)
ax.plot([15, 85], [15, 85], color="0.35", lw=1.1, ls="--")
ax.axhline(50, color="#c23b22", lw=1.0, ls=":")
SHORT = {"Political Party": "Party"}
seen = set()
for (d, pair), tot in n.items():
    if tot < 2000: continue
    x = 100 * dom[(d, pair)] / tot
    y = 100 * keep[(d, pair)] / tot
    c = DIM_COLORS.get(d, "0.4")
    ax.plot(x, y, "o", ms=6.5, color=c, mec="white", mew=0.6, alpha=0.95,
            zorder=4, label=SHORT.get(d, d) if d not in seen else None)
    seen.add(d)
ax.text(68, 73.5, "faithful retention\n(keeps what matters)", fontsize=7.4,
        color="0.3", rotation=38, ha="center")
ax.text(83.5, 47.2, "blind retention\n(coin flip)", fontsize=7.4, color="#c23b22",
        ha="right")
ax.set_xlim(15, 85); ax.set_ylim(15, 85)
ax.set_aspect("equal")
ax.set_xlabel("humans: share of contests the dimension should win (%)")
ax.set_ylabel("model: share of contests it keeps the dimension (%)")
ax.legend(frameon=False, fontsize=7.2, loc="upper left", handletextpad=0.2,
          borderaxespad=0.1, labelspacing=0.28)
fig.subplots_adjust(left=0.13, right=0.97, bottom=0.12, top=0.97)
style.save(fig, "fig_blindness")
