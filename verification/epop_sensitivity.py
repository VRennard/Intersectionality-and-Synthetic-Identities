"""
Referee Comment 5(ii): does dropping the population-error term e_pop change
the depth-2 collapse contest?

Under JOINT additivity (model and humans both additive in steering),
  e_AB = e_A + e_B - e_pop,   where e_pop = p_hat_pop - p_pop.
The published contest uses the additive predictor e_A + e_B (drops -e_pop).
We re-run the contest with the corrected predictor e_A + e_B - e_pop and
compare the best-single win rate, and report the magnitude of e_pop.
"""
import os, sys
import numpy as np
from collections import defaultdict

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(BASE, "advanced_bias_analysis"))
import utils
utils.USE_WEIGHTS = True  # weighted = paper's primary analysis
AVG = utils.AVG_PROFILE

MODELS = sys.argv[1:] or ["gpt-4o-mini", "gemma2_9b", "claude-haiku-4-5-20251001"]


def model_waves(model):
    d = os.path.join(BASE, "data", "results", model)
    return sorted((fn[1:-6] for fn in os.listdir(d)
                   if fn.startswith("W") and fn.endswith(".jsonl") and fn[1:-6].isdigit()),
                  key=int)


for model in MODELS:
    utils.MODEL_TAG = model
    win_pub = win_cor = total = 0
    epop_norms, eAB_norms = [], []
    flips = 0
    for wave in model_waves(model):
        llm, qmeta = utils.build_llm_index(wave, max_level=2)
        human = utils.load_human_index(wave, list(qmeta.keys()), qmeta, max_level=2)

        def bias(profile, qid):
            ld = llm.get(profile, {}).get(qid); hd = human.get(profile, {}).get(qid)
            if ld is None or hd is None or len(ld) != len(hd):
                return None
            return ld - hd

        for profile, qd in llm.items():
            if len(profile) != 2:
                continue
            a, b = sorted(profile)
            pa, pb = frozenset([a]), frozenset([b])
            for qid in qd:
                e_p = bias(profile, qid); e_a = bias(pa, qid); e_b = bias(pb, qid)
                e_pop = bias(AVG, qid)
                if any(x is None for x in (e_p, e_a, e_b, e_pop)):
                    continue
                cos_single = max(utils.cosine_sim(e_p, e_a), utils.cosine_sim(e_p, e_b))
                cos_add_pub = utils.cosine_sim(e_p, e_a + e_b)
                cos_add_cor = utils.cosine_sim(e_p, e_a + e_b - e_pop)
                bs_pub = cos_single >= cos_add_pub
                bs_cor = cos_single >= cos_add_cor
                win_pub += bs_pub; win_cor += bs_cor; total += 1
                flips += (bs_pub != bs_cor)
                epop_norms.append(float(np.linalg.norm(e_pop)))
                eAB_norms.append(float(np.linalg.norm(e_p)))

    if not total:
        print(f"{model}: no depth-2 cells"); continue
    print(f"\n{model}  (n={total:,} pair cells)")
    print(f"  best-single win  published (e_A+e_B)        : {100*win_pub/total:.1f}%")
    print(f"  best-single win  corrected (e_A+e_B - e_pop): {100*win_cor/total:.1f}%")
    print(f"  delta                                       : {100*(win_cor-win_pub)/total:+.1f} pts")
    print(f"  contests that flip decision                 : {100*flips/total:.1f}%")
    print(f"  median ||e_pop|| / median ||e_AB||          : "
          f"{np.median(epop_norms):.3f} / {np.median(eAB_norms):.3f} "
          f"= {np.median(epop_norms)/np.median(eAB_norms):.2f}")
