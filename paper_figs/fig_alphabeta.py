"""The collapse, seen directly: decompose each pair cell's realised vector as
alpha*(A-part) + beta*(B-part) and plot the (alpha, beta) density.

Model side (bias space): e_AB = alpha*e_A + beta*e_B  (gpt-4o-mini)
Human side (steering space): s_AB = alpha*s_A + beta*s_B  (cells n>=100)

Perfect additivity = mass at (1,1). Pure collapse = mass on the axes.
Least-squares on the 2-vector basis; cells with |cos(basis)|>0.95 dropped
(coefficients unidentifiable when the two singles are collinear).
"""
import os, sys, json, pickle, itertools
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

import style
sys.path.insert(0, os.path.join(style.BASE, "advanced_bias_analysis"))
import utils
utils.USE_WEIGHTS = True

WAVES = ("26","27","29","32","34","36","41","42","43","45","50","54","82","92")
MODEL = "gpt-4o-mini"


def coeffs(target, u, v):
    """least-squares alpha,beta for target ~ alpha*u + beta*v."""
    G = np.array([[u @ u, u @ v], [u @ v, v @ v]])
    b = np.array([target @ u, target @ v])
    try:
        a, be = np.linalg.solve(G, b)
    except np.linalg.LinAlgError:
        return None
    return a, be


def collinear(u, v):
    d = np.linalg.norm(u) * np.linalg.norm(v)
    return d == 0 or abs(u @ v) / d > 0.95


def model_points():
    utils.MODEL_TAG = MODEL
    pts = []
    for w in WAVES:
        human, _ = pickle.load(open(os.path.join(style.BASE, f"verification/cache/human_W{w}_weighted.pkl"), "rb"))
        llm, _ = utils.build_llm_index(w, max_level=2)
        for prof in llm:
            if len(prof) != 2:
                continue
            (a, va), (b, vb) = sorted(prof)
            pa, pb = frozenset([(a, va)]), frozenset([(b, vb)])
            for q, lab in llm[prof].items():
                hab = human.get(prof, {}).get(q)
                la = llm.get(pa, {}).get(q); ha = human.get(pa, {}).get(q)
                lb = llm.get(pb, {}).get(q); hb = human.get(pb, {}).get(q)
                if any(x is None for x in (hab, la, ha, lb, hb)):
                    continue
                if len({len(lab), len(hab), len(la), len(ha), len(lb), len(hb)}) != 1:
                    continue
                eA, eB, eAB = la - ha, lb - hb, lab - hab
                if collinear(eA, eB):
                    continue
                c = coeffs(eAB, eA, eB)
                if c is not None:
                    pts.append(c)
    return np.array(pts)


def human_points():
    pts = []
    for w in WAVES:
        opts = {q["question_id"]: [r["option"] for r in q["responses"]]
                for q in json.load(open(os.path.join(style.BASE, f"data/responses/survey_responses_W{w}.json")))}
        df = pd.read_csv(os.path.join(style.BASE, f"human_resp/American_Trends_Panel_W{w}/responses.csv"), low_memory=False)
        wcol = utils._weight_col(df, w)
        qids = [q for q in opts if q in df.columns]

        def dist(mask, q, mn):
            sl = (df.loc[mask] if mask is not None else df).dropna(subset=[q])
            if len(sl) < mn:
                return None
            wt = sl[wcol].to_numpy() if wcol else np.ones(len(sl))
            acc = dict.fromkeys(opts[q], 0.0)
            for v_, w_ in zip(sl[q], wt):
                if v_ in acc and np.isfinite(w_):
                    acc[v_] += w_
            c = np.array([acc[o] for o in opts[q]], float)
            return c / c.sum() if c.sum() > 0 else None

        masks = {}
        for dim, col in utils.DIM_TO_COL.items():
            if col not in df.columns:
                continue
            for val in utils.DIM_VALUES[dim]:
                if val in utils.IGNORE_VALUES.get(dim, set()):
                    continue
                m = (df[col] == val).to_numpy()
                if m.sum() >= 100:
                    masks[(dim, val)] = m
        pop = {q: dist(None, q, 100) for q in qids}
        sing = {k: {q: dist(m, q, 100) for q in qids} for k, m in masks.items()}
        for a, b in itertools.combinations(sorted(masks), 2):
            if a[0] == b[0]:
                continue
            m = masks[a] & masks[b]
            if m.sum() < 100:
                continue
            for q in qids:
                pq, sa, sb = pop.get(q), sing[a].get(q), sing[b].get(q)
                sab = dist(m, q, 100)
                if any(x is None for x in (pq, sa, sb, sab)):
                    continue
                sA, sB, sAB = sa - pq, sb - pq, sab - pq
                if collinear(sA, sB):
                    continue
                c = coeffs(sAB, sA, sB)
                if c is not None:
                    pts.append(c)
    return np.array(pts)


def panel(ax, pts, title, n):
    med = np.median(pts, axis=0)
    budget = np.median(pts.sum(axis=1))
    lim = (-0.9, 2.4)
    cmap = LinearSegmentedColormap.from_list("d", ["#ffffff", "#c6dbee", "#3f7fbf", "#08306b"])
    hb = ax.hexbin(pts[:, 0], pts[:, 1], gridsize=55, extent=(*lim, *lim),
                   cmap=cmap, bins="log", linewidths=0)
    ax.axhline(0, color="0.6", lw=0.7); ax.axvline(0, color="0.6", lw=0.7)
    ax.plot(1, 1, marker="*", ms=15, color="#e58606", mec="white", mew=0.8, zorder=5)
    ax.annotate("perfect\ncomposition", xy=(1, 1), xytext=(1.45, 1.55),
                fontsize=7.5, color="#b26a04", ha="left",
                arrowprops=dict(arrowstyle="-", color="#b26a04", lw=0.8))
    ax.plot([1], [0], marker="o", ms=6, color="#c23b22", mec="white", zorder=5)
    ax.plot([0], [1], marker="o", ms=6, color="#c23b22", mec="white", zorder=5)
    ax.annotate("pure collapse", xy=(1.06, -0.02), xytext=(1.35, -0.55),
                fontsize=7.5, color="#c23b22", ha="left",
                arrowprops=dict(arrowstyle="-", color="#c23b22", lw=0.8))
    # the budget line alpha+beta = 1 (one identity's worth of weight)
    ax.plot([lim[0], lim[1]], [1 - lim[0], 1 - lim[1]], ls="--", lw=1.0,
            color="#c23b22", alpha=0.75, zorder=4)
    ax.annotate(r"$\alpha{+}\beta{=}1$", xy=(-0.62, 1.52), fontsize=7.5,
                color="#c23b22", rotation=-45)
    ax.plot(*med, marker="P", ms=9, color="white", mec="black", mew=0.9, zorder=6)
    w = pts[pts.sum(axis=1) > 0.3]
    dom = np.median(np.max(w, axis=1) / w.sum(axis=1))
    ax.annotate(f"median budget $\\alpha{{+}}\\beta$ = {budget:.2f}\n"
                f"median dominant share = {dom:.2f}",
                xy=(0.03, 0.03), xycoords="axes fraction", fontsize=8,
                color="0.15",
                bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="0.7", lw=0.6))
    ax.set_xlim(lim); ax.set_ylim(lim)
    ax.set_xlabel(r"weight on feature $A$  ($\alpha$)")
    ax.set_title(f"{title}\n({n:,} pair cells)", fontsize=9)
    ax.set_aspect("equal")
    return hb


def main():
    cache = os.path.join(style.HERE, "alphabeta_points.npz")
    if os.path.exists(cache) and "--recompute" not in sys.argv:
        z = np.load(cache); mp, hp = z["mp"], z["hp"]
    else:
        mp = model_points()
        hp = human_points()
        np.savez_compressed(cache, mp=mp, hp=hp)
    print(f"model cells: {len(mp):,}   human cells: {len(hp):,}")
    med_m = np.median(mp, axis=0); med_h = np.median(hp, axis=0)
    print(f"medians  model: ({med_m[0]:.2f},{med_m[1]:.2f})   human: ({med_h[0]:.2f},{med_h[1]:.2f})")

    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.6), sharey=True)
    panel(axes[0], hp, "real subgroups (steering space)", len(hp))
    panel(axes[1], mp, f"LLM pair bias ({MODEL})", len(mp))
    axes[0].set_ylabel(r"weight on feature $B$  ($\beta$)")
    fig.subplots_adjust(left=0.09, right=0.985, bottom=0.13, top=0.87, wspace=0.08)
    style.save(fig, "fig_alphabeta")


if __name__ == "__main__":
    main()
