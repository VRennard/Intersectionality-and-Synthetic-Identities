"""Referee #6: which lever dominates the accuracy failure -- collapse or
single-feature miscalibration? For pair cells compute TV-to-human of:
 (a) model actual pair p_hat_AB
 (b) compose model's OWN (biased) singles: clip(p_hat_A+p_hat_B-p_hat_pop)  [fix collapse only]
 (d) compose HUMAN (unbiased) singles:     clip(p_A+p_B-p_pop)              [fix singles -> ideal]
 (s) model single-feature only (mean of p_hat_A, p_hat_B)                    [collapse w/ biased single]
If (b)~(a) but (d)<<(b), single-feature error is the dominant lever."""
import os,sys,numpy as np,itertools
sys.path.insert(0,'advanced_bias_analysis'); import utils
utils.USE_WEIGHTS=True; AVG=utils.AVG_PROFILE
def waves(m):
    d=f'data/results/{m}'
    return sorted((fn[1:-6] for fn in os.listdir(d) if fn.startswith('W') and fn.endswith('.jsonl') and fn[1:-6].isdigit()),key=int)
def nc(v):
    v=np.clip(v,0,None); s=v.sum(); return v/s if s>0 else None
for m in sys.argv[1:] or ['gpt-4o-mini']:
    utils.MODEL_TAG=m; A=B=D=S=tot=0.0
    for w in waves(m):
        llm,qm=utils.build_llm_index(w,max_level=2)
        human=utils.load_human_index(w,list(qm.keys()),qm,max_level=2)
        if not llm.get(AVG): continue
        for prof,qd in llm.items():
            if len(prof)!=2: continue
            a,b=sorted(prof); pa,pb=frozenset([a]),frozenset([b])
            for q in qd:
                lab=llm[prof].get(q); la=llm.get(pa,{}).get(q); lb=llm.get(pb,{}).get(q); lp=llm[AVG].get(q)
                hab=human.get(prof,{}).get(q); ha=human.get(pa,{}).get(q); hb=human.get(pb,{}).get(q); hp=human.get(AVG,{}).get(q)
                if any(x is None for x in (lab,la,lb,lp,hab,ha,hb,hp)): continue
                if len({len(lab),len(la),len(lb),len(lp),len(hab),len(ha),len(hb),len(hp)})!=1: continue
                bm=nc(la+lb-lp); dm=nc(ha+hb-hp)
                if bm is None or dm is None: continue
                A+=utils.tv(lab,hab); B+=utils.tv(bm,hab); D+=utils.tv(dm,hab)
                S+=0.5*(utils.tv(la,hab)+utils.tv(lb,hab)); tot+=1
    if tot:
        print(f"{m} (n={int(tot):,}):  (a)model-actual={A/tot:.4f}  (b)compose-biased-singles={B/tot:.4f}  "
              f"(s)single-only={S/tot:.4f}  (d)compose-HUMAN-singles[ideal]={D/tot:.4f}",flush=True)
