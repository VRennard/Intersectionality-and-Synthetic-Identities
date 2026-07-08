"""
Referee point 3: does the depth-2 collapse contest survive a large-cell
restriction (the same cell-size scrutiny applied to the human contest, n>=100/200)?

For each pair cell (A,B,q) we form the model bias e_g = p_hat_g - p_g from
human distributions computed directly from microdata (so we have the human
pair cell's n) and the LLM index. We report the best-single win rate (raw
e_A+e_B predictor AND e_pop-corrected) at n_AB >= 20 / 100 / 200.
"""
import os, sys, json, itertools
import numpy as np, pandas as pd
from collections import defaultdict

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(BASE, "advanced_bias_analysis"))
import utils
utils.USE_WEIGHTS = True
AVG = utils.AVG_PROFILE
EXCL = {"49"}
MODELS = sys.argv[1:] or ["gpt-4o-mini"]
THRESH = [20, 100, 200]


def waves():
    d = os.path.join(BASE, "human_resp")
    return sorted((fn.rsplit("W", 1)[1] for fn in os.listdir(d)
                   if fn.startswith("American_Trends_Panel_W")
                   and fn.rsplit("W", 1)[1] not in EXCL
                   and os.path.exists(os.path.join(d, fn, "responses.csv"))), key=int)

def load_options(w):
    return {q["question_id"]: [r["option"] for r in q["responses"]]
            for q in json.load(open(os.path.join(BASE, "data", "responses", f"survey_responses_W{w}.json")))}

def dist_n(s, opts, mn, wt=None):
    vc = s.value_counts(); n = int(sum(vc.get(o, 0) for o in opts))
    if n < mn: return None, 0
    if wt is None:
        c = np.array([vc.get(o, 0) for o in opts], float)
    else:
        acc = dict.fromkeys(opts, 0.0)
        for v, w in zip(s, wt):
            if v in acc and np.isfinite(w): acc[v] += w
        c = np.array([acc[o] for o in opts], float)
    sm = c.sum(); return (c / sm, n) if sm > 0 else (None, 0)


for model in MODELS:
    utils.MODEL_TAG = model
    win = {t: [0, 0] for t in THRESH}        # best_single, additive  (raw)
    winc = {t: [0, 0] for t in THRESH}       # best_single, additive  (e_pop-corrected)
    for w in waves():
        opts = load_options(w)
        df = pd.read_csv(os.path.join(BASE, "human_resp", f"American_Trends_Panel_W{w}", "responses.csv"), low_memory=False)
        wcol = utils._weight_col(df, w); qids = [q for q in opts if q in df.columns]
        llm, _ = utils.build_llm_index(w, max_level=2)
        lpop = llm.get(AVG, {})
        # human pop + singles from microdata
        hpop = {}
        for q in qids:
            sl = df.dropna(subset=[q]); wt = sl[wcol].to_numpy() if wcol else None
            d, _ = dist_n(sl[q], opts[q], 20, wt)
            if d is not None: hpop[q] = d
        masks = {}
        for dim, col in utils.DIM_TO_COL.items():
            if col not in df.columns: continue
            for val in utils.DIM_VALUES[dim]:
                if val in utils.IGNORE_VALUES.get(dim, set()): continue
                m = (df[col] == val).to_numpy()
                if m.sum() >= 20: masks[(dim, val)] = m
        hsing = {}
        for k, m in masks.items():
            sub = df.loc[m]; hsing[k] = {}
            for q in qids:
                sl = sub.dropna(subset=[q]); wt = sl[wcol].to_numpy() if wcol else None
                d, _ = dist_n(sl[q], opts[q], 20, wt)
                if d is not None: hsing[k][q] = d
        for a, b in itertools.combinations(sorted(masks), 2):
            if a[0] == b[0]: continue
            m = masks[a] & masks[b]
            if m.sum() < 20: continue
            sub = df.loc[m]
            lab = llm.get(frozenset([a, b]), {})
            la = llm.get(frozenset([a]), {}); lb = llm.get(frozenset([b]), {})
            for q in qids:
                if q not in hpop or q not in hsing.get(a, {}) or q not in hsing.get(b, {}): continue
                sl = sub.dropna(subset=[q]); wt = sl[wcol].to_numpy() if wcol else None
                hab, nab = dist_n(sl[q], opts[q], 20, wt)
                if hab is None: continue
                lab_q, la_q, lb_q, lpop_q = lab.get(q), la.get(q), lb.get(q), lpop.get(q)
                if any(x is None for x in (lab_q, la_q, lb_q, lpop_q)): continue
                if len({len(hab), len(hsing[a][q]), len(hsing[b][q]), len(hpop[q]),
                        len(lab_q), len(la_q), len(lb_q), len(lpop_q)}) != 1: continue
                e_ab = lab_q - hab; e_a = la_q - hsing[a][q]; e_b = lb_q - hsing[b][q]
                e_pop = lpop_q - hpop[q]
                cs = max(utils.cosine_sim(e_ab, e_a), utils.cosine_sim(e_ab, e_b))
                cadd = utils.cosine_sim(e_ab, e_a + e_b)
                cadd_c = utils.cosine_sim(e_ab, e_a + e_b - e_pop)
                for t in THRESH:
                    if nab >= t:
                        win[t][0 if cs >= cadd else 1] += 1
                        winc[t][0 if cs >= cadd_c else 1] += 1
    print(f"\n=== {model} ===")
    for t in THRESH:
        tot = sum(win[t]); totc = sum(winc[t])
        if tot:
            print(f"  n>={t:3d} (n={tot:>7,}):  best-single RAW={100*win[t][0]/tot:.1f}%   "
                  f"e_pop-CORRECTED={100*winc[t][0]/totc:.1f}%")
