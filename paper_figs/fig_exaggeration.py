"""
Fig 3 — Exaggeration scatter (depth 1).

x = human steering magnitude: mean TV between the real subgroup's distribution
    and the real population distribution, per (dimension-value x wave).
y = LLM steering magnitude: same, using the LLM's subgroup and the LLM's
    population ("Average American", else mean over singles).

Above the diagonal = the LLM exaggerates the group's distinctiveness
(caricature); below = it flattens the group into the average.

Model: gpt-4o-mini. Waves: all except W49. Intermediate data cached to
paper_figs/exaggeration_data.json (delete to recompute).
"""

import os, sys, json
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt

import style

sys.path.insert(0, os.path.join(style.BASE, "advanced_bias_analysis"))
import utils

MODEL = sys.argv[1] if len(sys.argv) > 1 else "gpt-4o-mini"
CACHE = os.path.join(style.HERE, "exaggeration_data.json" if MODEL == "gpt-4o-mini"
                     else f"exaggeration_data_{MODEL}.json")
PKL   = os.path.join(style.VERIF, "cache")


def compute():
    import pickle
    utils.MODEL_TAG = MODEL
    d = os.path.join(style.BASE, "data", "results", MODEL)
    waves = sorted((fn[1:-6] for fn in os.listdir(d)
                    if fn.startswith("W") and fn.endswith(".jsonl")), key=int)
    points = []   # {dim, val, wave, tv_h, tv_l, n_q}
    for wave in waves:
        if wave in style.EXCLUDE_WAVES:
            continue
        with open(os.path.join(PKL, f"human_W{wave}.pkl"), "rb") as f:
            human, _ = pickle.load(f)
        llm, qmeta = utils.build_llm_index(wave, max_level=1)
        h_pop = human.get(utils.AVG_PROFILE, {})

        l_pop = dict(llm.get(utils.AVG_PROFILE, {}))
        if not l_pop:   # population proxy: mean over singles
            acc = defaultdict(list)
            for p, qd in llm.items():
                if len(p) == 1:
                    for qid, dist in qd.items():
                        acc[qid].append(dist)
            l_pop = {qid: np.mean(v, axis=0) for qid, v in acc.items()}

        for profile, qd in llm.items():
            if len(profile) != 1:
                continue
            (dim, val), = profile
            hq = human.get(profile, {})
            tvs_h, tvs_l = [], []
            for qid, ld in qd.items():
                hd, hp, lp = hq.get(qid), h_pop.get(qid), l_pop.get(qid)
                if hd is None or hp is None or lp is None:
                    continue
                if not (len(ld) == len(hd) == len(hp) == len(lp)):
                    continue
                tvs_h.append(utils.tv(hd, hp))
                tvs_l.append(utils.tv(ld, lp))
            if len(tvs_h) >= 10:
                points.append({"dim": dim, "val": val, "wave": wave,
                               "tv_h": float(np.mean(tvs_h)),
                               "tv_l": float(np.mean(tvs_l)),
                               "n_q": len(tvs_h)})
        print(f"W{wave}: {sum(p['wave'] == wave for p in points)} profiles", flush=True)
    with open(CACHE, "w") as f:
        json.dump(points, f)
    return points


def main():
    if os.path.exists(CACHE):
        with open(CACHE) as f:
            points = json.load(f)
    else:
        points = compute()

    fig, ax = plt.subplots(figsize=(3.4, 3.2))
    lim = 0.30
    ax.fill_between([0, lim], [0, lim], lim, color="#d62728", alpha=0.05, linewidth=0)
    ax.fill_between([0, lim], 0, [0, lim], color="#1f77b4", alpha=0.05, linewidth=0)
    ax.plot([0, lim], [0, lim], color="0.4", linewidth=0.9)

    for dim in utils.DIMS:
        pts = [p for p in points if p["dim"] == dim]
        if not pts:
            continue
        ax.scatter([p["tv_h"] for p in pts], [p["tv_l"] for p in pts],
                   s=11, color=utils.DIM_COLORS[dim], alpha=0.75,
                   linewidths=0, label=utils.DIM_SHORT[dim])

    ax.text(0.025, lim - 0.015, "caricature\n(exaggerated)", fontsize=7.5,
            color="#d62728", va="top")
    ax.text(lim - 0.012, 0.018, "flattened\n(erased)", fontsize=7.5,
            color="#1f77b4", ha="right")

    ax.set_xlim(0, lim); ax.set_ylim(0, lim)
    ax.set_aspect("equal")
    ax.set_xlabel("human steering magnitude (TV to population)")
    ax.set_ylabel("LLM steering magnitude")
    ax.legend(loc="upper right", frameon=False, fontsize=6.8,
              handletextpad=0.2, borderaxespad=0.1, labelspacing=0.25)

    fig.subplots_adjust(left=0.15, right=0.97, bottom=0.14, top=0.97)
    style.save(fig, "fig_exaggeration" if MODEL == "gpt-4o-mini"
               else f"appendix/fig_exaggeration_{MODEL}")

    # quick stats
    pts = np.array([(p["tv_h"], p["tv_l"]) for p in points])
    above = (pts[:, 1] > pts[:, 0]).mean()
    print(f"  {len(pts)} points; LLM above diagonal (exaggerates): {above:.1%}")
    for dim in utils.DIMS:
        sel = np.array([(p["tv_h"], p["tv_l"]) for p in points if p["dim"] == dim])
        if len(sel):
            print(f"    {dim:18s} mean human={sel[:,0].mean():.3f} "
                  f"llm={sel[:,1].mean():.3f} above={np.mean(sel[:,1]>sel[:,0]):.0%}")


if __name__ == "__main__":
    main()
