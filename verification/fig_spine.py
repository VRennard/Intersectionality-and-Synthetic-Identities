"""
Spine figure prototype: mean TV error vs. profile depth, one line per model.

Reads verification/spine_results.json (written by verify_spine_collapse.py).
Run with --mock to render the design with plausible placeholder numbers
before the real data is available.

Per-model uncertainty: 95% CI across waves (wave means treated as replicates).
"""

import os, sys, json
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams.update({
    "pdf.fonttype": 42,
    "font.size": 9,
    "axes.titlesize": 9,
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "axes.spines.top": False,
    "axes.spines.right": False,
})
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))

# Survey-weighted human baselines are the paper's primary analysis
# (2026-06-11 switch); pass --unweighted to reproduce the old figures.
SUFFIX = "" if "--unweighted" in sys.argv else "_weighted"

# W49 repaired 2026-06-11 (missing gpt-4o-mini cells re-run, anomaly gone)
# and included. Still absent for gpt-4o / claude-haiku, and gemma2 /
# mistral-7B lack W49 depth-2 rows; wave means as replicates absorb this.
EXCLUDE_WAVES = set()

MODEL_STYLE = {
    # display name, color, linestyle (solid = API, dashed = open-weights)
    "gpt-4o":                    ("GPT-4o",        "#1f77b4", "-"),
    "gpt-4o-mini":               ("GPT-4o-mini",   "#6baed6", "-"),
    "claude-haiku-4-5-20251001": ("Claude Haiku",  "#d62728", "-"),
    "gemma2_9b":                 ("Gemma-2 9B",    "#2ca02c", "--"),
    "mistral_latest":            ("Mistral 7B",    "#c5b0d5", "--"),
    "llama3_1_8b_instruct_q4":   ("Llama-3.1 8B",  "#ff7f0e", "--"),
}

MOCK = [
    # model, depth, list of per-wave mean TVs (plausible placeholders)
    ("gpt-4o-mini", 1, [.16, .17, .15, .18, .16, .17, .15, .16, .18, .17, .16, .15, .17, .16]),
    ("gpt-4o-mini", 2, [.19, .20, .18, .21, .19, .20, .18, .19, .21, .20, .19, .18, .20, .19]),
    ("gpt-4o-mini", 3, [.22, .23, .21, .24, .22, .23, .21, .22, .24, .23, .22, .21, .23, .22]),
    ("gemma2_9b",   1, [.20, .21, .19, .22, .20, .21, .19, .20, .22, .21, .20, .19, .21, .20]),
    ("gemma2_9b",   2, [.23, .24, .22, .25, .23, .24, .22, .23, .25, .24, .23, .22, .24, .23]),
    ("gemma2_9b",   3, [.27, .28, .26, .29, .27, .28, .26, .27, .29, .28, .27, .26, .28, .27]),
    ("gpt-4o",      1, [.15, .16, .14, .17, .15, .16, .14, .15, .17, .16, .15, .14, .16, .15]),
    ("gpt-4o",      2, [.18, .19, .17, .20, .18, .19, .17, .18, .20, .19, .18, .17, .19, .18]),
    ("claude-haiku-4-5-20251001", 1, [.17, .18, .16, .19, .17, .18, .16, .17, .19, .18, .17, .16, .18, .17]),
    ("claude-haiku-4-5-20251001", 2, [.20, .21, .19, .22, .20, .21, .19, .20, .22, .21, .20, .19, .21, .20]),
]


def load_series(mock=False):
    """-> {model: {depth: (mean, ci_lo, ci_hi, n_waves)}} using wave means as replicates."""
    per = defaultdict(lambda: defaultdict(list))
    if mock:
        for model, depth, vals in MOCK:
            per[model][depth] = list(vals)
    else:
        path = os.path.join(HERE, f"spine_results{SUFFIX}.json")
        with open(path) as f:
            rows = json.load(f)
        for r in rows:
            if r["wave"] in EXCLUDE_WAVES:
                continue
            per[r["model"]][r["depth"]].append(r["mean_tv"])

    series = {}
    for model, depths in per.items():
        series[model] = {}
        for depth, vals in depths.items():
            vals = np.array(vals, dtype=float)
            m = vals.mean()
            se = vals.std(ddof=1) / np.sqrt(len(vals)) if len(vals) > 1 else 0.0
            series[model][depth] = (m, m - 1.96 * se, m + 1.96 * se, len(vals))
    return series


def load_noise_floor():
    """-> {depth: mean split-half TV} pooled over waves, or None if absent."""
    path = os.path.join(HERE, f"noise_floor{SUFFIX}.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        rows = json.load(f)
    floor = {}
    for depth in (1, 2, 3):
        rs = [r for r in rows if r["depth"] == depth and r["wave"] not in EXCLUDE_WAVES]
        if rs:
            n = sum(r["n_obs"] for r in rs)
            floor[depth] = sum(r["mean_tv"] * r["n_obs"] for r in rs) / n
    return floor or None


def main():
    mock = "--mock" in sys.argv
    series = load_series(mock=mock)
    floor = load_noise_floor()

    fig, ax = plt.subplots(figsize=(3.4, 2.6))

    if floor:
        depths = sorted(floor)
        ax.plot(depths, [floor[d] for d in depths], ":", color="0.45",
                marker="o", markersize=2.5, linewidth=1.2, zorder=2)
        ax.annotate("human\nsplit-half", xy=(depths[-1], floor[depths[-1]]),
                    xytext=(depths[-1] + 0.07, floor[depths[-1]]),
                    color="0.45", fontsize=7.5, va="center", ha="left",
                    annotation_clip=False)

    label_slots = []
    for model, by_depth in series.items():
        if model not in MODEL_STYLE:
            continue
        name, color, ls = MODEL_STYLE[model]
        depths = sorted(by_depth)
        means  = [by_depth[d][0] for d in depths]
        los    = [by_depth[d][1] for d in depths]
        his    = [by_depth[d][2] for d in depths]
        ax.plot(depths, means, ls, color=color, marker="o", markersize=3.5,
                linewidth=1.4, clip_on=False, zorder=3)
        ax.fill_between(depths, los, his, color=color, alpha=0.15, linewidth=0)
        label_slots.append((means[-1], depths[-1], name, color))

    # Direct labels at line ends, nudged apart to avoid overlap
    label_slots.sort(key=lambda t: t[0])
    min_gap = 0.011
    ys = [y for y, *_ in label_slots]
    for i in range(1, len(ys)):
        if ys[i] - ys[i - 1] < min_gap:
            ys[i] = ys[i - 1] + min_gap
    for (y0, x, name, color), y in zip(label_slots, ys):
        ax.annotate(name, xy=(x, y0), xytext=(x + 0.07, y),
                    color=color, fontsize=7.5, va="center", ha="left",
                    annotation_clip=False)

    ax.set_xticks([1, 2, 3])
    ax.set_xlim(0.9, 3.1)
    ax.set_xlabel("Identity features in profile")
    ax.set_ylabel("TV distance to human subgroup")
    ax.margins(y=0.08)
    fig.subplots_adjust(left=0.155, right=0.78, bottom=0.17, top=0.97)

    suffix = "_mock" if mock else ""
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(HERE, f"fig_spine{suffix}.{ext}"), dpi=300)
    print(f"wrote fig_spine{suffix}.png/.pdf")
    for model, by_depth in sorted(series.items()):
        for d in sorted(by_depth):
            m, lo, hi, n = by_depth[d]
            print(f"  {model:28s} d{d}: {m:.4f}  [{lo:.4f}, {hi:.4f}]  ({n} waves)")


if __name__ == "__main__":
    main()
