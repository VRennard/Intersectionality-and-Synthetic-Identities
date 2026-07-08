"""Score gpt-4o-mini instructed-composition battery on W26+W34.
Design A (within-regime): variant pairs vs variant singles.
Design B (cross-regime): variant pairs vs baseline singles."""
import os, sys, pickle
import numpy as np
sys.path.insert(0, 'advanced_bias_analysis'); import utils
utils.USE_WEIGHTS = True
def load(tag, wave):
    utils.MODEL_TAG = tag
    try: return utils.build_llm_index(wave, max_level=2)[0]
    except Exception: return {}
res = {}
for wave in ("26","34"):
    human,_ = pickle.load(open(f'verification/cache/human_W{wave}_weighted.pkl','rb'))
    base = load("gpt-4o-mini", wave)
    for var in ("promptI","promptC"):
        vidx = load(f"gpt-4o-mini_{var}", wave)
        for design, singles_src in (("within", vidx), ("cross", base)):
            key=(var,design)
            res.setdefault(key, [0,0])
            for prof in vidx:
                if len(prof)!=2: continue
                (a,va),(b,vb)=sorted(prof)
                pa,pb=frozenset([(a,va)]),frozenset([(b,vb)])
                for q,lab in vidx[prof].items():
                    hab=human.get(prof,{}).get(q)
                    la=singles_src.get(pa,{}).get(q); ha=human.get(pa,{}).get(q)
                    lb=singles_src.get(pb,{}).get(q); hb=human.get(pb,{}).get(q)
                    if any(x is None for x in (hab,la,ha,lb,hb)): continue
                    if len({len(lab),len(hab),len(la),len(ha),len(lb),len(hb)})!=1: continue
                    eAB=lab-hab; eA=la-ha; eB=lb-hb
                    cs=max(utils.cosine_sim(eAB,eA),utils.cosine_sim(eAB,eB))
                    res[key][0 if cs>=utils.cosine_sim(eAB,eA+eB) else 1]+=1
    # baseline on the same pair cells (use promptI's pair set as reference cells)
    vidx = load("gpt-4o-mini_promptI", wave)
    res.setdefault(("baseline","--"), [0,0])
    for prof in vidx:
        if len(prof)!=2: continue
        (a,va),(b,vb)=sorted(prof)
        pa,pb=frozenset([(a,va)]),frozenset([(b,vb)])
        for q in vidx[prof]:
            lab=base.get(prof,{}).get(q)
            hab=human.get(prof,{}).get(q)
            la=base.get(pa,{}).get(q); ha=human.get(pa,{}).get(q)
            lb=base.get(pb,{}).get(q); hb=human.get(pb,{}).get(q)
            if any(x is None for x in (lab,hab,la,ha,lb,hb)): continue
            if len({len(lab),len(hab),len(la),len(ha),len(lb),len(hb)})!=1: continue
            eAB=lab-hab; eA=la-ha; eB=lb-hb
            cs=max(utils.cosine_sim(eAB,eA),utils.cosine_sim(eAB,eB))
            res[("baseline","--")][0 if cs>=utils.cosine_sim(eAB,eA+eB) else 1]+=1
print(f"{'condition':22s} {'design':8s} {'best-single':>12s} {'n':>8s}")
for (var,design),(w,l) in sorted(res.items()):
    t=w+l
    if t: print(f"  {var:20s} {design:8s} {100*w/t:>10.1f}%  {t:>8,}")
