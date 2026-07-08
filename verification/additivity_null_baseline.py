"""
Null baseline for additive fidelity cosines.

Question: cos(e_pair, e_A + e_B) ~ 0.89 looks high, but high relative to what?
All bias vectors from one model share a global error component, so unrelated
bias vectors also correlate. Compute, per (pair profile x question):

  cos_add    : cos(e_pair, e_A + e_B)           (the "additivity" score)
  cos_best   : max cos(e_pair, e_A), cos(e_pair, e_B)   (collapse predictor)
  cos_null   : cos(e_pair, e_C) for a random single C whose dimension is
               NOT in the pair (same question)           (null floor)
  mag_ratio  : ||e_pair|| / ||e_A + e_B||       (flattening)

Model: gpt-4o-mini. Waves: subset for speed. Uses verification/cache.
"""

import os, sys, json, pickle, random
from collections import defaultdict

import numpy as np

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(BASE, "advanced_bias_analysis"))
import utils

CACHE_DIR = os.path.join(BASE, "verification", "cache")
# --weighted: survey-weighted human caches (primary basis), _weighted output
WEIGHTED = "--weighted" in sys.argv
WSUF     = "_weighted" if WEIGHTED else ""
_args = [a for a in sys.argv[1:] if a != "--weighted"]
MODEL = _args[0] if _args else "gpt-4o-mini"
WAVES = ["26", "27", "29", "32", "34", "36", "41", "42", "43", "45",
         "50", "54", "82", "92"]   # all waves, W49 excluded
rng = random.Random(0)


def main():
    utils.MODEL_TAG = MODEL
    agg = defaultdict(list)

    for wave in WAVES:
        with open(os.path.join(CACHE_DIR, f"human_W{wave}{WSUF}.pkl"), "rb") as f:
            human, _ = pickle.load(f)
        llm, qmeta = utils.build_llm_index(wave, max_level=2)

        # bias vectors of singles per (qid) for null sampling, grouped by dim
        singles_by_qid = defaultdict(list)   # qid -> [(dim, e_vec)]
        for profile, qd in llm.items():
            if len(profile) != 1:
                continue
            (dim, val), = profile
            hq = human.get(profile, {})
            for qid, ld in qd.items():
                hd = hq.get(qid)
                if hd is not None and len(hd) == len(ld):
                    singles_by_qid[qid].append((dim, ld - hd))

        n = 0
        for profile, qd in llm.items():
            if len(profile) != 2:
                continue
            (da, va), (db, vb) = sorted(profile)
            pa, pb = frozenset([(da, va)]), frozenset([(db, vb)])
            hp, ha, hb = human.get(profile, {}), human.get(pa, {}), human.get(pb, {})
            la, lb = llm.get(pa, {}), llm.get(pb, {})
            for qid, lp in qd.items():
                need = (hp.get(qid), ha.get(qid), hb.get(qid), la.get(qid), lb.get(qid))
                if any(x is None for x in need):
                    continue
                h_p, h_a, h_b, l_a, l_b = need
                if not (len(lp) == len(h_p) == len(h_a) == len(h_b) == len(l_a) == len(l_b)):
                    continue
                e_p = lp - h_p
                e_a = l_a - h_a
                e_b = l_b - h_b
                # null: random single from an unrelated dimension, same question
                pool = [e for d, e in singles_by_qid.get(qid, []) if d not in (da, db)]
                if not pool:
                    continue
                e_c = rng.choice(pool)

                agg["cos_add"].append(utils.cosine_sim(e_p, e_a + e_b))
                agg["cos_best"].append(max(utils.cosine_sim(e_p, e_a),
                                           utils.cosine_sim(e_p, e_b)))
                agg["cos_null"].append(utils.cosine_sim(e_p, e_c))
                denom = np.linalg.norm(e_a + e_b)
                if denom > 1e-10:
                    agg["mag_ratio"].append(float(np.linalg.norm(e_p) / denom))
                n += 1
        print(f"W{wave}: {n} cells", flush=True)

    print(f"\n=== ADDITIVITY VS NULL BASELINE ({MODEL}, waves {','.join(WAVES)}) ===")
    for k in ("cos_null", "cos_add", "cos_best"):
        v = np.array(agg[k])
        print(f"  {k:9s}: mean={v.mean():.4f}  median={np.median(v):.4f}")
    v = np.array(agg["mag_ratio"])
    print(f"  mag_ratio: mean={v.mean():.4f}  median={np.median(v):.4f}  "
          f"(||actual pair bias|| / ||additive prediction||)")
    add, null, best = (np.array(agg[k]) for k in ("cos_add", "cos_null", "cos_best"))
    print(f"\n  additive minus null:  {np.mean(add - null):+.4f}  (unique value of 'adding')")
    print(f"  best-single minus additive: {np.mean(best - add):+.4f}")
    print(f"  share where additive beats the random-unrelated null: {(add > null).mean():.1%}")

    suffix = ("" if MODEL == "gpt-4o-mini" else f"_{MODEL}") + WSUF
    with open(os.path.join(BASE, "verification",
                           f"additivity_null_baseline{suffix}.json"), "w") as f:
        json.dump({k: [float(x) for x in v] for k, v in agg.items()}, f)


if __name__ == "__main__":
    main()
