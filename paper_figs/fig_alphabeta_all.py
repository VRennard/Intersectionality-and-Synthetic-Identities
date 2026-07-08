"""Pooled variant of fig_alphabeta: humans (left) vs ALL LLMs pooled (right),
with each model's own median marked. Reuses cached human points; computes and
caches per-model (alpha,beta) points.
"""
import os, sys, pickle
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

import style
sys.path.insert(0, os.path.join(style.BASE, "advanced_bias_analysis"))
import utils
utils.USE_WEIGHTS = True

WAVES = ("26","27","29","32","34","36","41","42","43","45","50","54","82","92")
MODELS = [
    ("gpt-4o-mini",               "GPT-4o-mini"),
    ("gpt-4o",                    "GPT-4o"),
    ("claude-haiku-4-5-20251001", "Haiku 4.5"),
    ("gemma2_9b",                 "Gemma-2"),
    ("mistral_latest",            "Mistral-7B"),
    ("llama3_1_8b_instruct_q4",   "Llama-3.1"),
    ("gpt-5.5-2026-04-23",        "GPT-5.5"),
]


def coeffs(t, u, v):
    G = np.array([[u @ u, u @ v], [u @ v, v @ v]])
    b = np.array([t @ u, t @ v])
    try:
        return np.linalg.solve(G, b)
    except np.linalg.LinAlgError:
        return None


def collinear(u, v):
    d = np.linalg.norm(u) * np.linalg.norm(v)
    return d == 0 or abs(u @ v) / d > 0.95


def model_points(tag):
    cache = os.path.join(style.HERE, f"alphabeta_pts_{tag}.npz")
    if os.path.exists(cache):
        return np.load(cache)["mp"]
    utils.MODEL_TAG = tag
    pts = []
    for w in WAVES:
        try:
            human, _ = pickle.load(open(os.path.join(style.BASE, f"verification/cache/human_W{w}_weighted.pkl"), "rb"))
            llm, _ = utils.build_llm_index(w, max_level=2)
        except Exception:
            continue
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
        print(f"  {tag} W{w} done ({len(pts):,})", flush=True)
    pts = np.array(pts) if pts else np.zeros((0, 2))
    np.savez_compressed(cache, mp=pts)
    return pts


def panel(ax, pts, title, n, med_markers=None):
    med = np.median(pts, axis=0)
    budget = np.median(pts.sum(axis=1))
    lim = (-0.9, 2.4)
    cmap = LinearSegmentedColormap.from_list("d", ["#ffffff", "#c6dbee", "#3f7fbf", "#08306b"])
    ax.hexbin(pts[:, 0], pts[:, 1], gridsize=55, extent=(*lim, *lim),
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
    ax.plot([lim[0], lim[1]], [1 - lim[0], 1 - lim[1]], ls="--", lw=1.0,
            color="#c23b22", alpha=0.75, zorder=4)
    ax.annotate(r"$\alpha{+}\beta{=}1$", xy=(-0.62, 1.52), fontsize=7.5,
                color="#c23b22", rotation=-45)
    if med_markers:
        for (mx, my), lbl in med_markers:
            ax.plot(mx, my, marker="P", ms=7, color="white", mec="black",
                    mew=0.8, zorder=6)
    else:
        ax.plot(*med, marker="P", ms=9, color="white", mec="black", mew=0.9, zorder=6)
    ax.annotate(f"median budget $\\alpha{{+}}\\beta$ = {budget:.2f}",
                xy=(0.03, 0.03), xycoords="axes fraction", fontsize=8,
                color="0.15",
                bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="0.7", lw=0.6))
    ax.set_xlim(lim); ax.set_ylim(lim)
    ax.set_xlabel(r"weight on feature $A$  ($\alpha$)")
    ax.set_title(f"{title}\n({n:,} pair cells)", fontsize=9)
    ax.set_aspect("equal")


def main():
    z = np.load(os.path.join(style.HERE, "alphabeta_points.npz"))
    hp = z["hp"]
    pools, meds = [], []
    for tag, name in MODELS:
        p = model_points(tag)
        if len(p) == 0:
            continue
        pools.append(p)
        meds.append((np.median(p, axis=0), name))
        print(f"{name}: {len(p):,} cells, median ({np.median(p[:,0]):.2f},{np.median(p[:,1]):.2f}), "
              f"budget {np.median(p.sum(axis=1)):.2f}", flush=True)
    mp = np.vstack(pools)
    print(f"pooled: {len(mp):,} cells")

    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.6), sharey=True)
    panel(axes[0], hp, "real subgroups (steering space)", len(hp))
    panel(axes[1], mp, f"{len(pools)} LLMs pooled (pair bias)", len(mp), med_markers=meds)
    axes[0].set_ylabel(r"weight on feature $B$  ($\beta$)")
    fig.subplots_adjust(left=0.09, right=0.985, bottom=0.13, top=0.87, wspace=0.08)
    style.save(fig, "fig_alphabeta_all")


if __name__ == "__main__":
    main()
