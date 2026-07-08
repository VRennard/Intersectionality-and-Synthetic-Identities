"""
Additive-estimator evaluation (NHB gating item C).

For each pair cell the model emits an actual two-feature distribution p_hat_AB.
The ADDITIVE ESTIMATOR instead composes the model's OWN single-feature outputs:
    a_hat = clip(p_hat_A + p_hat_B - p_hat_pop, 0), renormalised
    (depth 3:  p_hat_A + p_hat_B + p_hat_C - 2 p_hat_pop)
We compare each against the REAL subgroup p_AB (human), by TV:
    TV(p_hat_AB, p_AB)   -- the model's actual two-feature error (the spine)
    TV(a_hat,     p_AB)  -- what a simple additive composition would have scored
If a_hat beats p_hat_AB, the model would be MORE accurate by adding its own
single-feature outputs than by conditioning on both features: the collapse costs
accuracy. Needs the model population p_hat_pop (["Average American"]).
"""
import os, sys, json
import numpy as np
from collections import defaultdict

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(BASE, "advanced_bias_analysis"))
import utils
utils.USE_WEIGHTS = True
AVG = utils.AVG_PROFILE

MODELS = sys.argv[1:] or ["gpt-4o-mini"]

def waves(m):
    d = os.path.join(BASE, "data", "results", m)
    return sorted((fn[1:-6] for fn in os.listdir(d)
                   if fn.startswith("W") and fn.endswith(".jsonl") and fn[1:-6].isdigit()), key=int)

def norm_clip(v):
    v = np.clip(v, 0, None); s = v.sum()
    return v / s if s > 0 else None

for m in MODELS:
    utils.MODEL_TAG = m
    # tv_model[depth] = list of TV(model, human); tv_add[depth] = list of TV(additive, human)
    tvm = defaultdict(list); tva = defaultdict(list); win = defaultdict(int); tot = defaultdict(int)
    for w in waves(m):
        llm, qmeta = utils.build_llm_index(w, max_level=3)
        human = utils.load_human_index(w, list(qmeta.keys()), qmeta, max_level=3)
        if not llm.get(AVG):
            continue
        def L(p, q): return llm.get(p, {}).get(q)
        def H(p, q): return human.get(p, {}).get(q)
        for profile, qd in llm.items():
            d = len(profile)
            if d not in (2, 3):
                continue
            feats = sorted(profile); singles = [frozenset([f]) for f in feats]
            for q in qd:
                pab = L(profile, q); hab = H(profile, q); pop = L(AVG, q)
                es = [L(s, q) for s in singles]
                if pab is None or hab is None or pop is None or any(e is None for e in es): continue
                if len({len(pab), len(hab), len(pop), *[len(e) for e in es]}) != 1: continue
                add = norm_clip(sum(es) - (d - 1) * pop)
                if add is None: continue
                t_m = utils.tv(pab, hab); t_a = utils.tv(add, hab)
                tvm[d].append(t_m); tva[d].append(t_a)
                tot[d] += 1; win[d] += (t_a < t_m)
    print(f"\n=== {m} ===")
    for d in (2, 3):
        if not tot[d]: continue
        mm, ma = np.mean(tvm[d]), np.mean(tva[d])
        print(f"  depth {d} (n={tot[d]:,}): TV model-actual={mm:.4f}  TV additive-estimator={ma:.4f}  "
              f"(additive {'better' if ma<mm else 'worse'} by {abs(mm-ma):.4f}); "
              f"additive beats model in {100*win[d]/tot[d]:.1f}% of cells")
