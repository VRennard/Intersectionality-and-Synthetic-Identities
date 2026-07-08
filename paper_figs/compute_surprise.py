"""
Compute data for Fig 6 (counter-stereotypical failure).

Per pair profile {A:va, B:vb}, pooled over waves (W49 excluded):
  surprise = P(va & vb) / (P(va) * P(vb))   from pooled human microdata
             (< 1 = counter-stereotypical, rarer than chance)
  mean_tv  = mean TV(LLM pair dist, human pair dist) over questions
  n_cells  = number of (wave x question) cells

Also dumps full distributions (human, LLM, additive prediction p_pair_hat =
p_human_pair + (e_A + e_B)) for three case-study profiles, for every question
where all components exist, with the human cell size.

Model: gpt-4o-mini. Output: paper_figs/surprise_tv.json, case_studies.json
"""

import os, sys, json, pickle
from collections import defaultdict

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(BASE, "advanced_bias_analysis"))
import utils

MODEL = "gpt-4o-mini"
EXCL  = {"49"}
PKL   = os.path.join(BASE, "verification", "cache")

CASES = [
    frozenset([("Race", "Black"), ("Political Party", "Republican")]),
    frozenset([("Political Party", "Republican"), ("Religion", "Atheist")]),
    frozenset([("Gender", "Male"), ("Religion", "Muslim")]),
]


def waves_list():
    d = os.path.join(BASE, "data", "results", MODEL)
    return sorted((fn[1:-6] for fn in os.listdir(d)
                   if fn.startswith("W") and fn.endswith(".jsonl") and fn[1:-6] not in EXCL),
                  key=int)


def main():
    utils.MODEL_TAG = MODEL

    # ---- pooled co-occurrence counts from human microdata ----
    single_n = defaultdict(int)   # (dim,val) -> count
    pair_n   = defaultdict(int)   # frozenset -> count
    total_n  = 0
    for wave in waves_list():
        csv = os.path.join(BASE, "human_resp", f"American_Trends_Panel_W{wave}", "responses.csv")
        df = pd.read_csv(csv, low_memory=False)
        total_n += len(df)
        col_of = {dim: col for dim, col in utils.DIM_TO_COL.items() if col in df.columns}
        vals = {}
        for dim, col in col_of.items():
            for val in utils.DIM_VALUES[dim]:
                if val in utils.IGNORE_VALUES.get(dim, set()):
                    continue
                m = (df[col] == val).to_numpy()
                single_n[(dim, val)] += int(m.sum())
                vals[(dim, val)] = m
        keys = list(vals)
        for i, ka in enumerate(keys):
            for kb in keys[i + 1:]:
                if ka[0] == kb[0]:
                    continue
                pair_n[frozenset([ka, kb])] += int((vals[ka] & vals[kb]).sum())
        print(f"W{wave}: co-occurrence done", flush=True)

    # ---- TV error per pair profile + case-study dumps ----
    tv_acc = defaultdict(list)
    cases_out = {str(sorted(c)): [] for c in CASES}
    for wave in waves_list():
        with open(os.path.join(PKL, f"human_W{wave}.pkl"), "rb") as f:
            human, _ = pickle.load(f)
        llm, qmeta = utils.build_llm_index(wave, max_level=2)
        for profile, qd in llm.items():
            if len(profile) != 2:
                continue
            hp = human.get(profile, {})
            for qid, lp in qd.items():
                hd = hp.get(qid)
                if hd is None or len(hd) != len(lp):
                    continue
                tv_acc[profile].append(utils.tv(lp, hd))
        # case studies: full distributions + additive prediction
        for c in CASES:
            if c not in llm:
                continue
            (da, va), (db, vb) = sorted(c)
            pa, pb = frozenset([(da, va)]), frozenset([(db, vb)])
            for qid, lp in llm[c].items():
                comps = (human.get(c, {}).get(qid), human.get(pa, {}).get(qid),
                         human.get(pb, {}).get(qid), llm.get(pa, {}).get(qid),
                         llm.get(pb, {}).get(qid))
                if any(x is None for x in comps):
                    continue
                h_p, h_a, h_b, l_a, l_b = comps
                if not (len(lp) == len(h_p) == len(h_a) == len(h_b) == len(l_a) == len(l_b)):
                    continue
                additive = np.clip(h_p + (l_a - h_a) + (l_b - h_b), 0, None)
                s = additive.sum()
                if s > 0:
                    additive = additive / s
                cases_out[str(sorted(c))].append({
                    "wave": wave, "qid": qid,
                    "question": qmeta[qid]["text"],
                    "options": qmeta[qid]["options"],
                    "human": [float(x) for x in h_p],
                    "llm": [float(x) for x in lp],
                    "additive": [float(x) for x in additive],
                    "tv_llm": utils.tv(lp, h_p),
                    "tv_additive": utils.tv(additive, h_p),
                })
        print(f"W{wave}: TV + cases done", flush=True)

    out = []
    for profile, tvs in tv_acc.items():
        (da, va), (db, vb) = sorted(profile)
        n_a, n_b = single_n[(da, va)], single_n[(db, vb)]
        n_ab = pair_n[frozenset(profile)]
        if n_a == 0 or n_b == 0:
            continue
        surprise = (n_ab / total_n) / ((n_a / total_n) * (n_b / total_n)) if n_ab else 0.0
        out.append({"dimA": da, "valA": va, "dimB": db, "valB": vb,
                    "surprise": surprise, "human_n": n_ab,
                    "mean_tv": float(np.mean(tvs)), "n_cells": len(tvs)})

    with open(os.path.join(HERE, "surprise_tv.json"), "w") as f:
        json.dump(out, f)
    with open(os.path.join(HERE, "case_studies.json"), "w") as f:
        json.dump(cases_out, f)
    print(f"\nwrote surprise_tv.json ({len(out)} pair profiles) and case_studies.json")
    cs = sorted(out, key=lambda r: r["surprise"])[:10]
    print("most counter-stereotypical pairs:")
    for r in cs:
        print(f"  {r['valA']} x {r['valB']}: surprise={r['surprise']:.2f} "
              f"tv={r['mean_tv']:.3f} n={r['human_n']}")


if __name__ == "__main__":
    main()
