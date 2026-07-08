"""Referee major #5: is the depth-2 collapse robust to the conditioning prompt?
Score the best-single-vs-additive contest on W26 for baseline gemma2_9b and the
two alternative-framing runs (promptA neutral, promptB first-person), restricted
to the (pair,question) cells common to all three so the comparison is exact.
"""
import os, sys, json, itertools
import numpy as np, pandas as pd
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(BASE, "advanced_bias_analysis"))
import utils
utils.USE_WEIGHTS = True
AVG = utils.AVG_PROFILE
WAVE = "26"
TAGS = ["gemma2_9b", "gemma2_9b_promptA", "gemma2_9b_promptB"]


def load_options(w):
    return {q["question_id"]: [r["option"] for r in q["responses"]]
            for q in json.load(open(os.path.join(BASE, "data", "responses", f"survey_responses_W{w}.json")))}


def dist_n(s, opts, mn, wt=None):
    vc = s.value_counts(); n = int(sum(vc.get(o, 0) for o in opts))
    if n < mn:
        return None
    if wt is None:
        c = np.array([vc.get(o, 0) for o in opts], float)
    else:
        acc = dict.fromkeys(opts, 0.0)
        for v, w in zip(s, wt):
            if v in acc and np.isfinite(w):
                acc[v] += w
        c = np.array([acc[o] for o in opts], float)
    sm = c.sum()
    return c / sm if sm > 0 else None


# ---- human W26 side (built once) ----
opts = load_options(WAVE)
df = pd.read_csv(os.path.join(BASE, "human_resp", f"American_Trends_Panel_W{WAVE}", "responses.csv"), low_memory=False)
wcol = utils._weight_col(df, WAVE)
qids = [q for q in opts if q in df.columns]
hpop = {}
for q in qids:
    sl = df.dropna(subset=[q]); wt = sl[wcol].to_numpy() if wcol else None
    d = dist_n(sl[q], opts[q], 20, wt)
    if d is not None:
        hpop[q] = d
masks = {}
for dim, col in utils.DIM_TO_COL.items():
    if col not in df.columns:
        continue
    for val in utils.DIM_VALUES[dim]:
        if val in utils.IGNORE_VALUES.get(dim, set()):
            continue
        m = (df[col] == val).to_numpy()
        if m.sum() >= 20:
            masks[(dim, val)] = m
hsing = {}
for k, m in masks.items():
    sub = df.loc[m]; hsing[k] = {}
    for q in qids:
        sl = sub.dropna(subset=[q]); wt = sl[wcol].to_numpy() if wcol else None
        d = dist_n(sl[q], opts[q], 20, wt)
        if d is not None:
            hsing[k][q] = d


def score_tag(tag):
    utils.MODEL_TAG = tag
    llm, _ = utils.build_llm_index(WAVE, max_level=2)
    out = {}  # (pairkey, q) -> is_single (bool)
    for a, b in itertools.combinations(sorted(masks), 2):
        if a[0] == b[0]:
            continue
        lab = llm.get(frozenset([a, b]))
        la = llm.get(frozenset([a])); lb = llm.get(frozenset([b]))
        if lab is None or la is None or lb is None:
            continue
        for q in qids:
            if q not in hpop or q not in hsing.get(a, {}) or q not in hsing.get(b, {}):
                continue
            lab_q, la_q, lb_q = lab.get(q), la.get(q), lb.get(q)
            if any(x is None for x in (lab_q, la_q, lb_q)):
                continue
            hab = None
            m = masks[a] & masks[b]
            if m.sum() < 20:
                continue
            sub = df.loc[m].dropna(subset=[q]); wt = sub[wcol].to_numpy() if wcol else None
            hab = dist_n(sub[q], opts[q], 20, wt)
            if hab is None:
                continue
            if len({len(hab), len(hsing[a][q]), len(hsing[b][q]), len(hpop[q]),
                    len(lab_q), len(la_q), len(lb_q)}) != 1:
                continue
            e_ab = lab_q - hab; e_a = la_q - hsing[a][q]; e_b = lb_q - hsing[b][q]
            cs = max(utils.cosine_sim(e_ab, e_a), utils.cosine_sim(e_ab, e_b))
            cadd = utils.cosine_sim(e_ab, e_a + e_b)
            out[(frozenset([a, b]), q)] = cs >= cadd
    return out


res = {t: score_tag(t) for t in TAGS}
for t in TAGS:
    print(f"  {t}: {len(res[t]):,} scored cells")
common = set.intersection(*(set(res[t]) for t in TAGS))
print(f"\ncommon (pair,question) cells across all three: {len(common):,}\n")
print(f"{'framing':30s} {'best-single win (common cells)':>32s}")
names = {"gemma2_9b": "baseline (expert researcher)",
         "gemma2_9b_promptA": "neutral (survey simulator)",
         "gemma2_9b_promptB": "first-person (individuals)"}
for t in TAGS:
    w = sum(res[t][k] for k in common)
    print(f"  {names[t]:28s} {100*w/len(common):>30.1f}%")
