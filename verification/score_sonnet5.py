"""Score claude-sonnet-5 on W26+W34: depth-2 collapse contest (best-single vs
additive) and depth-3 contest (best-single vs best-pair vs additive)."""
import os, sys, pickle, itertools
import numpy as np
sys.path.insert(0, 'advanced_bias_analysis'); import utils
utils.USE_WEIGHTS = True
utils.MODEL_TAG = "claude-sonnet-5"
d2 = [0, 0]           # single, additive
d3 = [0, 0, 0]        # single, pair, additive
for w in ("26", "34"):
    human, _ = pickle.load(open(f'verification/cache/human_W{w}_weighted.pkl', 'rb'))
    llm, _ = utils.build_llm_index(w, max_level=3)
    # ---- depth 2 ----
    for prof in llm:
        if len(prof) != 2: continue
        (a, va), (b, vb) = sorted(prof)
        pa, pb = frozenset([(a, va)]), frozenset([(b, vb)])
        for q, lab in llm[prof].items():
            hab = human.get(prof, {}).get(q)
            la = llm.get(pa, {}).get(q); ha = human.get(pa, {}).get(q)
            lb = llm.get(pb, {}).get(q); hb = human.get(pb, {}).get(q)
            if any(x is None for x in (hab, la, ha, lb, hb)): continue
            if len({len(lab), len(hab), len(la), len(ha), len(lb), len(hb)}) != 1: continue
            eAB = lab - hab; eA = la - ha; eB = lb - hb
            cs = max(utils.cosine_sim(eAB, eA), utils.cosine_sim(eAB, eB))
            cadd = utils.cosine_sim(eAB, eA + eB)
            d2[0 if cs >= cadd else 1] += 1
    # ---- depth 3 ----
    for prof in llm:
        if len(prof) != 3: continue
        feats = sorted(prof)
        singles = [frozenset([f]) for f in feats]
        pairs = [frozenset(p) for p in itertools.combinations(feats, 2)]
        for q, lab in llm[prof].items():
            hab = human.get(prof, {}).get(q)
            if hab is None or len(hab) != len(lab): continue
            es = []
            ok = True
            for s_ in singles:
                lp = llm.get(s_, {}).get(q); hp = human.get(s_, {}).get(q)
                if lp is None or hp is None or len(lp) != len(lab): ok = False; break
                es.append(lp - hp)
            if not ok: continue
            eprs = []
            for p_ in pairs:
                lp = llm.get(p_, {}).get(q); hp = human.get(p_, {}).get(q)
                if lp is not None and hp is not None and len(lp) == len(lab):
                    eprs.append(lp - hp)
            if not eprs: continue
            ep = lab - hab
            cs = max(utils.cosine_sim(ep, x) for x in es)
            cp = max(utils.cosine_sim(ep, x) for x in eprs)
            ca = utils.cosine_sim(ep, sum(es))
            if ca >= cs and ca >= cp: d3[2] += 1
            elif cp >= cs: d3[1] += 1
            else: d3[0] += 1
    print(f"  W{w} done", flush=True)
t2 = sum(d2); t3 = sum(d3)
print(f"\nclaude-sonnet-5 (W26+W34):")
print(f"  depth-2: n={t2:,}  best-single={100*d2[0]/t2:.1f}%")
print(f"  depth-3: n={t3:,}  best-single={100*d3[0]/t3:.1f}%  best-pair={100*d3[1]/t3:.1f}%  additive={100*d3[2]/t3:.1f}%")
