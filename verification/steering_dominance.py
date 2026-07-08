"""
Steering-space dominance: which dimension does the pair LOOK like, in the
model's own expressive space (no ground truth involved)?

For each pair profile {A, B} x question:
  s_pair = p_llm(pair) - p_llm(avg)
  s_A    = p_llm(A)    - p_llm(avg)     (avg = "Average American", else mean
  s_B    = p_llm(B)    - p_llm(avg)      over singles)
  winner = argmax cos(s_pair, s_dim)

Per-dimension steering win rates complete the reconciliation:
depth-1 steering magnitude (Party loudest) vs bias-direction keep rates
(Party ~coin flip) vs THIS (does the model EXPRESS pairs as Party-flavored?).

Model: gpt-4o-mini, waves excl W49. Output: steering_dominance.json
"""

import os, sys, json
from collections import defaultdict

import numpy as np

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(BASE, "advanced_bias_analysis"))
import utils

MODEL = "gpt-4o-mini"
EXCL  = {"49"}


def main():
    utils.MODEL_TAG = MODEL
    d = os.path.join(BASE, "data", "results", MODEL)
    waves = sorted((fn[1:-6] for fn in os.listdir(d)
                    if fn.startswith("W") and fn.endswith(".jsonl") and fn[1:-6] not in EXCL),
                   key=int)

    wins = defaultdict(int)
    contested = defaultdict(int)
    pairwise = defaultdict(lambda: [0, 0])   # (dimA,dimB) sorted -> [A wins, total]

    for wave in waves:
        llm, qmeta = utils.build_llm_index(wave, max_level=2)
        avg = dict(llm.get(utils.AVG_PROFILE, {}))
        if not avg:
            acc = defaultdict(list)
            for p, qd in llm.items():
                if len(p) == 1:
                    for qid, dist in qd.items():
                        acc[qid].append(dist)
            avg = {qid: np.mean(v, axis=0) for qid, v in acc.items()}

        n = 0
        for profile, qd in llm.items():
            if len(profile) != 2:
                continue
            (da, va), (db, vb) = sorted(profile)
            pa, pb = frozenset([(da, va)]), frozenset([(db, vb)])
            la, lb = llm.get(pa, {}), llm.get(pb, {})
            for qid, lp in qd.items():
                a, b, av = la.get(qid), lb.get(qid), avg.get(qid)
                if a is None or b is None or av is None:
                    continue
                if not (len(lp) == len(a) == len(b) == len(av)):
                    continue
                s_p, s_a, s_b = lp - av, a - av, b - av
                ca, cb = utils.cosine_sim(s_p, s_a), utils.cosine_sim(s_p, s_b)
                winner = da if ca >= cb else db
                wins[winner] += 1
                contested[da] += 1
                contested[db] += 1
                key = tuple(sorted([da, db]))
                pairwise[key][1] += 1
                if winner == key[0]:
                    pairwise[key][0] += 1
                n += 1
        print(f"W{wave}: {n} cells", flush=True)

    out = {
        "keep_rate": {d: wins[d] / contested[d] for d in contested},
        "pairwise": {f"{k[0]} | {k[1]}": [w, t] for k, (w, t) in pairwise.items()},
    }
    with open(os.path.join(BASE, "verification", "steering_dominance.json"), "w") as f:
        json.dump(out, f, indent=1)

    print("\n=== STEERING-SPACE DOMINANCE (model's own expressive space) ===")
    for d, r in sorted(out["keep_rate"].items(), key=lambda kv: -kv[1]):
        print(f"  {d:18s} steering win rate = {r:.1%}")
    print("\n  Party pairwise:")
    for k, (w, t) in sorted(pairwise.items()):
        if "Political Party" in k:
            other = k[0] if k[1] == "Political Party" else k[1]
            pw = w / t if k[0] == "Political Party" else 1 - w / t
            print(f"    Party vs {other:12s}: Party wins {pw:.1%}  (n={t})")


if __name__ == "__main__":
    main()
