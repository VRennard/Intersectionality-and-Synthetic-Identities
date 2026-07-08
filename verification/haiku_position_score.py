"""Referee major #1 (dimension-order confound): does the retained feature track
the DIMENSION or the SLOT position?

Feature order in the prompt is fixed canonical (Age<Gender<Race<Party<Religion).
promptA = canonical order, promptR = reversed order, same neutral framing, same
Gemma-2-9B / W26 cells. For each matched pair cell we find which single feature
the pair bias aligns with (cosine) under each order. If retention is the SAME
dimension regardless of order, it is position-independent (dimension-driven); if
it follows slot-1, it is position-driven.
"""
import os, sys, json, itertools
import numpy as np, pandas as pd
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(BASE, "advanced_bias_analysis"))
import utils
utils.USE_WEIGHTS = True
WAVE = "26"

DIMS_ORDER = ["Age", "Gender", "Race", "Political Party", "Religion"]  # canonical


def parse_feat(f):
    for d in sorted(DIMS_ORDER, key=len, reverse=True):
        if f.startswith(d + " "):
            return d, f[len(d) + 1:]
    return None, None


def load_pairs(tag):
    """tuple(sorted-canonical (dim,val)) -> {q: dist}, keeping canonical A,B order."""
    out = {}
    f = os.path.join(BASE, "data", "results", f"claude-haiku-4-5-20251001_{tag}", f"W{WAVE}.jsonl")
    for ln in open(f):
        try: r = json.loads(ln)
        except: continue
        if r.get("status") != "success" or len(r.get("demographics", [])) != 2:
            continue
        dv = [parse_feat(x) for x in r["demographics"]]
        if any(d is None for d, _ in dv):
            continue
        # canonical A,B = ordered by DIMS_ORDER
        dv = sorted(dv, key=lambda x: DIMS_ORDER.index(x[0]))
        key = (dv[0], dv[1])
        d = np.array(r["response_distribution"], float)
        if d.sum() > 0:
            out.setdefault(key, {})[r["question_id"]] = d / d.sum()
    return out


def load_singles(tag):
    out = {}
    f = os.path.join(BASE, "data", "results", f"claude-haiku-4-5-20251001_{tag}", f"W{WAVE}.jsonl")
    for ln in open(f):
        try: r = json.loads(ln)
        except: continue
        if r.get("status") != "success" or len(r.get("demographics", [])) != 1:
            continue
        dm, vl = parse_feat(r["demographics"][0])
        if dm is None: continue
        d = np.array(r["response_distribution"], float)
        if d.sum() > 0:
            out.setdefault((dm, vl), {})[r["question_id"]] = d / d.sum()
    return out


# ---- human side from microdata ----
opts = {q["question_id"]: [rr["option"] for rr in q["responses"]]
        for q in json.load(open(os.path.join(BASE, "data", "responses", f"survey_responses_W{WAVE}.json")))}
df = pd.read_csv(os.path.join(BASE, "human_resp", f"American_Trends_Panel_W{WAVE}", "responses.csv"), low_memory=False)
wcol = utils._weight_col(df, WAVE)
qids = [q for q in opts if q in df.columns]

def hdist(mask, q):
    sl = df.loc[mask].dropna(subset=[q]) if mask is not None else df.dropna(subset=[q])
    if len(sl) < 20: return None
    wt = sl[wcol].to_numpy() if wcol else None
    acc = dict.fromkeys(opts[q], 0.0)
    for v, w in zip(sl[q], (wt if wt is not None else np.ones(len(sl)))):
        if v in acc and np.isfinite(w): acc[v] += w
    c = np.array([acc[o] for o in opts[q]], float)
    return c / c.sum() if c.sum() > 0 else None

masks = {}
for dim in DIMS_ORDER:
    col = utils.DIM_TO_COL.get(dim)
    if col is None or col not in df.columns: continue
    for val in utils.DIM_VALUES.get(dim, []):
        if val in utils.IGNORE_VALUES.get(dim, set()): continue
        m = (df[col] == val).to_numpy()
        if m.sum() >= 20: masks[(dim, val)] = m

# ---- score ----
sA = load_singles("promptA")
pA = load_pairs("promptA")   # canonical order
pR = load_pairs("promptR")   # reversed order

agree = flip = 0
slot1_canon = slot1_rev = 0
tot = 0
for key in set(pA) & set(pR):
    (a, b) = key  # a = canonical slot-1, b = canonical slot-2
    if a not in masks or b not in masks or a not in sA or b not in sA:
        continue
    ma, mb = masks[a], masks[b]
    mab = ma & mb
    if mab.sum() < 20: continue
    for q in set(pA[key]) & set(pR[key]):
        ha = hdist(ma, q); hb = hdist(mb, q); hab = hdist(mab, q)
        la = sA[a].get(q); lb = sA[b].get(q)
        if any(x is None for x in (ha, hb, hab, la, lb)): continue
        if q not in pA[key] or q not in pR[key]: continue
        if len({len(ha), len(hb), len(hab), len(la), len(lb),
                len(pA[key][q]), len(pR[key][q])}) != 1: continue
        eA = la - ha; eB = lb - hb
        e_canon = pA[key][q] - hab
        e_rev = pR[key][q] - hab
        # retained = single with higher cosine to the pair bias
        ret_canon = "A" if utils.cosine_sim(e_canon, eA) >= utils.cosine_sim(e_canon, eB) else "B"
        ret_rev = "A" if utils.cosine_sim(e_rev, eA) >= utils.cosine_sim(e_rev, eB) else "B"
        tot += 1
        if ret_canon == ret_rev: agree += 1
        else: flip += 1
        # slot-1 is A in canonical prompt, B in reversed prompt
        if ret_canon == "A": slot1_canon += 1
        if ret_rev == "B": slot1_rev += 1

print(f"matched pair-question cells: {tot:,}\n")
print(f"retained SAME dimension regardless of order (position-independent): {100*agree/tot:.1f}%")
print(f"retained flips with order                 (position-driven)      : {100*flip/tot:.1f}%\n")
print(f"slot-1 retention rate, canonical order: {100*slot1_canon/tot:.1f}%")
print(f"slot-1 retention rate, reversed order : {100*slot1_rev/tot:.1f}%")
print("\n(If retention were position-driven, slot-1 rate would be high in BOTH orders")
print(" and agreement would be low. Position-independent => high agreement, and the")
print(" slot-1 rate flips between orders because it tracks the dimension, not the slot.)")
