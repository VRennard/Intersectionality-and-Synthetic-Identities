"""
Referee Comment 2(b): is human intersectional opinion super-additive?
For human pair cells (n_AB >= 100), compare realised pair steering s_AB to the
additive prediction s_A + s_B:
  - magnitude ratio ||s_AB|| / ||s_A + s_B||  (>1 super-additive, ~1 additive, <1 sub)
  - fraction of cells with ratio > 1
  - excess distinctiveness beyond additive, corrected for the cell's own
    sampling-noise expectation (so the ratio isn't inflated by multinomial noise).
"""
import os, sys, json, itertools
import numpy as np
import pandas as pd
from collections import defaultdict

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(BASE, "advanced_bias_analysis"))
import utils
utils.USE_WEIGHTS = True
EXCL = {"49"}


def waves():
    d = os.path.join(BASE, "human_resp")
    return sorted((fn.rsplit("W", 1)[1] for fn in os.listdir(d)
                   if fn.startswith("American_Trends_Panel_W")
                   and fn.rsplit("W", 1)[1] not in EXCL
                   and os.path.exists(os.path.join(d, fn, "responses.csv"))), key=int)


def load_options(w):
    with open(os.path.join(BASE, "data", "responses", f"survey_responses_W{w}.json")) as f:
        return {q["question_id"]: [r["option"] for r in q["responses"]] for q in json.load(f)}


def dist_n(series, opts, min_n, weights=None):
    vc = series.value_counts()
    n = int(sum(vc.get(o, 0) for o in opts))
    if n < min_n:
        return None, 0
    if weights is None:
        c = np.array([vc.get(o, 0) for o in opts], dtype=float)
    else:
        acc = dict.fromkeys(opts, 0.0)
        for v, w in zip(series, weights):
            if v in acc and np.isfinite(w):
                acc[v] += w
        c = np.array([acc[o] for o in opts], dtype=float)
    s = c.sum()
    return (c / s, n) if s > 0 else (None, 0)


ratios, excess, frac_super = [], [], []
for wave in waves():
    options = load_options(wave)
    df = pd.read_csv(os.path.join(BASE, "human_resp",
                                  f"American_Trends_Panel_W{wave}", "responses.csv"),
                     low_memory=False)
    wcol = utils._weight_col(df, wave)
    qids = [q for q in options if q in df.columns]
    pop = {}
    for qid in qids:
        sl = df.dropna(subset=[qid])
        w = sl[wcol].to_numpy() if wcol else None
        d, _ = dist_n(sl[qid], options[qid], 20, w)
        if d is not None:
            pop[qid] = d
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
    singles = {}
    for k, m in masks.items():
        sub = df.loc[m]; singles[k] = {}
        for qid in qids:
            sl = sub.dropna(subset=[qid])
            w = sl[wcol].to_numpy() if wcol else None
            d, _ = dist_n(sl[qid], options[qid], 100, w)
            if d is not None and qid in pop:
                singles[k][qid] = d
    for a, b in itertools.combinations(sorted(masks), 2):
        if a[0] == b[0]:
            continue
        m = masks[a] & masks[b]
        if m.sum() < 100:
            continue
        sub = df.loc[m]
        for qid in qids:
            if qid not in pop or qid not in singles.get(a, {}) or qid not in singles.get(b, {}):
                continue
            sl = sub.dropna(subset=[qid])
            w = sl[wcol].to_numpy() if wcol else None
            d_ab, n_ab = dist_n(sl[qid], options[qid], 100, w)
            if d_ab is None:
                continue
            s_ab = d_ab - pop[qid]
            s_add = (singles[a][qid] - pop[qid]) + (singles[b][qid] - pop[qid])
            na, nadd = np.linalg.norm(s_ab), np.linalg.norm(s_add)
            if nadd < 1e-9:
                continue
            # sampling-noise expectation of ||s_ab||^2 at n_ab (E||noise||^2)
            noise_var = np.sum(d_ab * (1 - d_ab)) / n_ab
            ex = (na**2 - noise_var) - nadd**2   # excess squared-distinctiveness beyond additive
            ratios.append(na / nadd)
            excess.append(ex)
            frac_super.append(na > nadd)

r = np.array(ratios)
print(f"human pair cells (n_AB>=100): {len(r):,}")
print(f"  median ||s_AB|| / ||s_A+s_B||      : {np.median(r):.3f}")
print(f"  mean                                : {r.mean():.3f}")
print(f"  fraction super-additive (ratio>1)   : {100*np.mean(frac_super):.1f}%")
print(f"  mean noise-corrected excess (sq dist): {np.mean(excess):+.5f}")
print(f"  (>0 = super-additive, ~0 = additive, <0 = sub-additive)")
