"""Per-cell decomposition, star-space style: for every flagship pair cell,
x = TV(model pair output, truth); y = TV(additive rule on TRUE singles, truth).
If y ~ noise floor regardless of x, calibrated ingredients fix accuracy
everywhere -- the 0.062-vs-0.238 bars as a 200K-cell density."""
import os, sys, pickle
import numpy as np
import style
sys.path.insert(0, os.path.join(style.BASE, "advanced_bias_analysis"))
import utils
utils.USE_WEIGHTS = True
WAVES = ("26","27","29","32","34","36","41","42","43","45","50","54","82","92")
utils.MODEL_TAG = "gpt-4o-mini"
AVG = utils.AVG_PROFILE
def tv(a,b): return 0.5*np.abs(a-b).sum()
pts=[]
for w in WAVES:
    human,_ = pickle.load(open(os.path.join(style.BASE,f"verification/cache/human_W{w}_weighted.pkl"),"rb"))
    llm,_ = utils.build_llm_index(w, max_level=2)
    hpop = human.get(AVG, {})
    for prof in llm:
        if len(prof)!=2: continue
        (a,va),(b,vb)=sorted(prof)
        pa,pb=frozenset([(a,va)]),frozenset([(b,vb)])
        for q,lab in llm[prof].items():
            hab=human.get(prof,{}).get(q); ha=human.get(pa,{}).get(q)
            hb=human.get(pb,{}).get(q); hp=hpop.get(q)
            if any(x is None for x in (hab,ha,hb,hp)): continue
            if len({len(lab),len(hab),len(ha),len(hb),len(hp)})!=1: continue
            add=np.clip(ha+hb-hp,0,None)
            s=add.sum()
            if s<=0: continue
            pts.append((tv(lab,hab), tv(add/s,hab)))
    print(f"  W{w} ({len(pts):,})",flush=True)
np.savez_compressed(os.path.join(style.HERE,"decomp_cells.npz"), p=np.array(pts))
print("saved",len(pts))
