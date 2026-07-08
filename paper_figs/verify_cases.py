"""
Case-study verification (paper plan §7 item 6).

Augments every record in case_studies.json with the human cell size for that
exact (profile x wave x question) and the expected multinomial noise TV at
that n, then audits the cases fig_surprise.py would select:
  - current rule: max(tv_llm - tv_additive), unconstrained
  - robust rule:  same, restricted to human_n >= MIN_N
Prints caption-ready stats per profile (share of cells where the additive
prediction beats the LLM, median gap, n distribution).

Rewrites case_studies.json in place (adds keys human_n, noise_tv).
"""

import os, sys, json
from collections import defaultdict

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(BASE, "advanced_bias_analysis"))
import utils

MIN_N = 50   # featured case must have at least this many human respondents


def expected_noise_tv(p, n):
    p = np.asarray(p, dtype=float)
    return float(0.5 * np.sqrt(2.0 / (np.pi * n)) * np.sum(np.sqrt(p * (1 - p))))


def main():
    path = os.path.join(HERE, "case_studies.json")
    with open(path) as f:
        cases = json.load(f)

    # group (profile, wave) -> records, to load each CSV once
    by_wave = defaultdict(list)
    for key, recs in cases.items():
        profile = [tuple(t) for t in eval(key)]
        for rec in recs:
            by_wave[(key, rec["wave"])].append((profile, rec))

    for (key, wave), items in sorted(by_wave.items(), key=lambda t: t[0][1]):
        csv = os.path.join(BASE, "human_resp",
                           f"American_Trends_Panel_W{wave}", "responses.csv")
        df = pd.read_csv(csv, low_memory=False)
        profile = items[0][0]
        mask = np.ones(len(df), dtype=bool)
        for dim, val in profile:
            mask &= (df[utils.DIM_TO_COL[dim]] == val).values
        for _, rec in items:
            qid, opts = rec["qid"], rec["options"]
            vals = df.loc[mask, qid].dropna() if qid in df.columns else []
            n = int(sum(str(v).strip() in opts for v in vals))
            rec["human_n"] = n
            rec["noise_tv"] = expected_noise_tv(rec["human"], n) if n else None

    with open(path, "w") as f:
        json.dump(cases, f, indent=1)

    print(f"{'profile':46s} cases  add<llm  med_gap   n: min/med/max")
    for key, recs in cases.items():
        gaps = [r["tv_llm"] - r["tv_additive"] for r in recs]
        ns   = [r["human_n"] for r in recs]
        wins = np.mean([g > 0 for g in gaps])
        print(f"{key[:46]:46s} {len(recs):5d}  {wins:6.0%}  {np.median(gaps):+.3f}"
              f"   {min(ns)}/{int(np.median(ns))}/{max(ns)}")

    for key, recs in cases.items():
        cur = max(recs, key=lambda r: r["tv_llm"] - r["tv_additive"])
        ok  = [r for r in recs if r["human_n"] >= MIN_N]
        rob = max(ok, key=lambda r: r["tv_llm"] - r["tv_additive"]) if ok else None
        print(f"\n{key}")
        for tag, r in [("current", cur), (f"robust(n>={MIN_N})", rob)]:
            if r is None:
                print(f"  {tag}: NONE available")
                continue
            print(f"  {tag}: W{r['wave']} {r['qid']}  n={r['human_n']}"
                  f"  noise_tv={r['noise_tv']:.3f}"
                  f"  tv_llm={r['tv_llm']:.3f}  tv_add={r['tv_additive']:.3f}"
                  f"  gap={r['tv_llm']-r['tv_additive']:+.3f}")


if __name__ == "__main__":
    main()
