"""
Fig 7 — The identity budget: what best explains the LLM's bias, by depth.

Stacked horizontal bars: at depth 2 and depth 3, the share of
(profile x question) cells whose bias vector is best predicted by a single
dimension / a sub-pair / the full additive sum.

Data: verification/collapse_results.json
(gpt-4o-mini + gemma2_9b + claude-haiku [15 waves] + gpt-5.5 [W26+W34 subset]).
"""

import os, json
from collections import defaultdict

import matplotlib.pyplot as plt

import style
from style import C_ADD, C_SINGLE, C_PAIR

import sys
ALL = "--all" in sys.argv
MODELS = [("gpt-4o-mini", "GPT-4o-mini"), ("gemma2_9b", "Gemma-2 9B"),
          ("claude-haiku-4-5-20251001", "Claude Haiku"),
          ("gpt-5.5-2026-04-23", "GPT-5.5*"),
          ("claude-sonnet-5", "Sonnet 5*"),
          ("mistral_latest", "Mistral 7B"),
          ("llama3_1_8b_instruct_q4", "Llama-3.1 8B")]
PREDS  = [("best_single", "one dimension", C_SINGLE),
          ("best_pair",   "two dimensions", C_PAIR),
          ("additive",    "all (additive)", C_ADD)]


def main():
    with open(os.path.join(style.VERIF, f"collapse_results{style.SUFFIX}.json")) as f:
        rows = json.load(f)
    rows = [r for r in rows if r["wave"] not in style.EXCLUDE_WAVES]

    bars = []   # (label, {pred: share})
    for tag, name in MODELS:
        for depth in (2, 3):
            agg = defaultdict(int)
            for r in rows:
                if r["model"] == tag and r["depth"] == depth:
                    agg[r["predictor"]] += r["wins"]
            tot = sum(agg.values())
            if tot:
                bars.append((f"{name}\n{depth} features",
                             {p: agg.get(p, 0) / tot for p, _, _ in PREDS}))

    fig, ax = plt.subplots(figsize=(3.6, 0.46 * len(bars) + 0.5))
    ys = range(len(bars))[::-1]
    for y, (label, shares) in zip(ys, bars):
        left = 0.0
        for p, pname, color in PREDS:
            s = shares.get(p, 0.0)
            if s == 0:
                continue
            ax.barh(y, s, left=left, color=color, height=0.56)
            if s > 0.07:
                ax.text(left + s / 2, y, f"{s:.0%}", ha="center", va="center",
                        fontsize=7.5, color="white", fontweight="bold")
            left += s

    ax.set_yticks(list(ys))
    ax.set_yticklabels([b[0] for b in bars], fontsize=7.5,
                       linespacing=0.95)
    ax.set_ylim(-0.7, len(bars) - 0.3)
    ax.set_xlim(0, 1)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_xticklabels(["0%", "25%", "50%", "75%", "100%"])
    ax.set_xlabel("share of cells best explained by ...")

    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for _, _, c in PREDS]
    ax.legend(handles, [n for _, n, _ in PREDS], loc="upper center",
              bbox_to_anchor=(0.42, 1.07), ncol=3, frameon=False, fontsize=7.5,
              handlelength=1.0, columnspacing=0.8, handletextpad=0.5)

    fig.subplots_adjust(left=0.26, right=0.975, bottom=0.09, top=0.93)
    style.save(fig, "appendix/fig_depth_collapse_all" if ALL else "fig_depth_collapse")


if __name__ == "__main__":
    main()
