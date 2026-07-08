"""The identity budget is a constant: total coefficient weight by depth.

Depth 2 (cached from fig_alphabeta): e_AB ~ a*e_A + b*e_B  -> budget a+b
Depth 3 (computed here):            e_ABC ~ a*e_A + b*e_B + c*e_C -> a+b+c
Model side: gpt-4o-mini bias space (human cells n>=20, cached indices).
Human side: steering space, cells n>=100 (microdata pass).
"""
import os, sys, pickle, itertools
import numpy as np
import matplotlib.pyplot as plt

import style
sys.path.insert(0, os.path.join(style.BASE, "advanced_bias_analysis"))
import utils
utils.USE_WEIGHTS = True

WAVES = ("26","27","29","32","34","36","41","42","43","45","50","54","82","92")
MODEL = "gpt-4o-mini"
AVG = utils.AVG_PROFILE


def solve3(t, us):
    G = np.array([[u @ v for v in us] for u in us])
    b = np.array([t @ u for u in us])
    try:
        return np.linalg.solve(G, b)
    except np.linalg.LinAlgError:
        return None


def basis_ok(us):
    for u, v in itertools.combinations(us, 2):
        d = np.linalg.norm(u) * np.linalg.norm(v)
        if d == 0 or abs(u @ v) / d > 0.95:
            return False
    return True


def model_d3():
    cache = os.path.join(style.HERE, "budget_d3_model.npz")
    if os.path.exists(cache):
        return np.load(cache)["s"]
    utils.MODEL_TAG = MODEL
    sums = []
    for w in WAVES:
        human, _ = pickle.load(open(os.path.join(style.BASE, f"verification/cache/human_W{w}_weighted.pkl"), "rb"))
        llm, _ = utils.build_llm_index(w, max_level=3)
        for prof in llm:
            if len(prof) != 3:
                continue
            feats = sorted(prof)
            singles = [frozenset([f]) for f in feats]
            for q, lab in llm[prof].items():
                hab = human.get(prof, {}).get(q)
                if hab is None or len(hab) != len(lab):
                    continue
                es, hs_ = [], []
                good = True
                for s_ in singles:
                    lp = llm.get(s_, {}).get(q); hp = human.get(s_, {}).get(q)
                    if lp is None or hp is None or len(lp) != len(lab):
                        good = False; break
                    es.append(lp - hp)
                if not good or not basis_ok(es):
                    continue
                c = solve3(lab - hab, es)
                if c is not None:
                    sums.append(c.sum())
        print(f"  model W{w} done ({len(sums):,})", flush=True)
    s = np.array(sums)
    np.savez_compressed(cache, s=s)
    return s


def human_d3():
    cache = os.path.join(style.HERE, "budget_d3_human.npz")
    if os.path.exists(cache):
        return np.load(cache)["s"]
    utils.MODEL_TAG = MODEL
    utils.MIN_HUMAN_N = 100
    sums = []
    for w in WAVES:
        llm, qmeta = utils.build_llm_index(w, max_level=3)
        human = utils.load_human_index(w, list(qmeta.keys()), qmeta, max_level=3)
        pop = human.get(AVG, {})
        for prof in human:
            if not hasattr(prof, "__len__") or len(prof) != 3:
                continue
            feats = sorted(prof)
            singles = [frozenset([f]) for f in feats]
            for q, pab in human[prof].items():
                pq = pop.get(q)
                if pq is None or len(pq) != len(pab):
                    continue
                ss = []
                good = True
                for s_ in singles:
                    hp = human.get(s_, {}).get(q)
                    if hp is None or len(hp) != len(pab):
                        good = False; break
                    ss.append(hp - pq)
                if not good or not basis_ok(ss):
                    continue
                c = solve3(pab - pq, ss)
                if c is not None:
                    sums.append(c.sum())
        print(f"  human W{w} done ({len(sums):,})", flush=True)
    s = np.array(sums)
    np.savez_compressed(cache, s=s)
    return s


def main():
    z = np.load(os.path.join(style.HERE, "alphabeta_points.npz"))
    h2 = z["hp"].sum(axis=1)
    m2 = z["mp"].sum(axis=1)
    m3 = model_d3()
    h3 = human_d3()
    for name, a in (("human d2", h2), ("human d3", h3), ("model d2", m2), ("model d3", m3)):
        print(f"{name}: n={len(a):,} median={np.median(a):.2f}")

    fig, ax = plt.subplots(figsize=(6.2, 3.4))
    groups = [("humans, 2 features", h2, "#3f7fbf"),
              ("humans, 3 features", h3, "#08306b"),
              ("model, 2 features", m2, "#e58606"),
              ("model, 3 features", m3, "#c23b22")]
    lim = (-1.5, 5.5)
    xs = np.linspace(*lim, 400)
    from scipy.stats import gaussian_kde
    for i, (name, a, col) in enumerate(groups):
        med = np.median(a)          # full-sample median (untrimmed)
        a = a[(a > lim[0]) & (a < lim[1])]
        sub = a if len(a) < 200000 else np.random.default_rng(0).choice(a, 200000, replace=False)
        kde = gaussian_kde(sub, bw_method=0.12)
        y = kde(xs); y = y / y.max()
        base = -i * 1.18
        ax.fill_between(xs, base, base + y, color=col, alpha=0.75, lw=0)
        ax.plot([med, med], [base, base + 1.02], color="white", lw=1.6)
        ax.text(lim[1] - 0.05, base + 0.44, f"{name}\nmedian {med:.2f}",
                ha="right", fontsize=8.2, color=col, fontweight="bold")
    for v, lbl in ((1, "one identity's\nworth of weight"), (2, "two"), (3, "three")):
        ax.axvline(v, color="0.45", ls=":", lw=0.9, zorder=0)
        ax.text(v, 1.32, lbl, ha="center", fontsize=7.3, color="0.35")
    ax.set_yticks([])
    ax.set_xlim(lim); ax.set_ylim(-3.85, 1.75)
    ax.set_xlabel("total identity weight (sum of least-squares coefficients)")
    for s_ in ("left", "right", "top"):
        ax.spines[s_].set_visible(False)
    fig.subplots_adjust(left=0.03, right=0.985, bottom=0.16, top=0.99)
    style.save(fig, "fig_budget_depth")


if __name__ == "__main__":
    main()
