"""Referee W1: memorization-asymmetry test. If collapse were retrieval of
published single-feature toplines, pair cells whose parent singles are BEST
matched to ground truth (plausibly memorized) should collapse MORE. Stratify
the flagship's pair contest by parent-single accuracy."""
import sys, pickle
import numpy as np
sys.path.insert(0,'advanced_bias_analysis'); import utils
utils.USE_WEIGHTS=True; utils.MODEL_TAG='gpt-4o-mini'
WAVES=("26","27","29","32","34","36","41","42","43","45","50","54","82","92")
rows=[]
for w in WAVES:
    human,_=pickle.load(open(f'verification/cache/human_W{w}_weighted.pkl','rb'))
    llm,_=utils.build_llm_index(w, max_level=2)
    for prof in llm:
        if len(prof)!=2: continue
        (a,va),(b,vb)=sorted(prof)
        pa,pb=frozenset([(a,va)]),frozenset([(b,vb)])
        for q,lab in llm[prof].items():
            hab=human.get(prof,{}).get(q); la=llm.get(pa,{}).get(q)
            ha=human.get(pa,{}).get(q); lb=llm.get(pb,{}).get(q); hb=human.get(pb,{}).get(q)
            if any(x is None for x in (hab,la,ha,lb,hb)): continue
            if len({len(lab),len(hab),len(la),len(ha),len(lb),len(hb)})!=1: continue
            eA=la-ha; eB=lb-hb; eAB=lab-hab
            cs=max(utils.cosine_sim(eAB,eA),utils.cosine_sim(eAB,eB))
            col = cs>=utils.cosine_sim(eAB,eA+eB)
            acc = min(0.5*np.abs(eA).sum(), 0.5*np.abs(eB).sum())  # best parent-single TV
            rows.append((acc,col))
    print(f"W{w} done ({len(rows):,})", flush=True)
r=np.array(rows)
qs=np.quantile(r[:,0],[0,1/3,2/3,1])
print("\ncollapse share by parent-single accuracy (best single TV error) terciles:")
for i in range(3):
    m=r[(r[:,0]>=qs[i])&(r[:,0]<qs[i+1]+ (1e-9 if i==2 else 0))]
    print(f"  TV {qs[i]:.3f}-{qs[i+1]:.3f} ({'most' if i==0 else 'least' if i==2 else 'mid'} accurate singles): n={len(m):,}  collapse={100*m[:,1].mean():.1f}%")
