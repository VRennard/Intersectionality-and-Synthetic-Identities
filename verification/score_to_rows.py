"""Score a model tag over 14 waves and dump per-wave predictor rows in the
collapse_results format: {model, wave, depth, predictor, wins, n_total}.
Usage: score_to_rows.py TAG [d2only]"""
import os, sys, pickle, itertools, json
import numpy as np
sys.path.insert(0, 'advanced_bias_analysis'); import utils
utils.USE_WEIGHTS = True
TAG = sys.argv[1]; D2ONLY = "d2only" in sys.argv
utils.MODEL_TAG = TAG
WAVES = ("26","27","29","32","34","36","41","42","43","45","50","54","82","92")
rows=[]
for w in WAVES:
    try:
        human,_ = pickle.load(open(f'verification/cache/human_W{w}_weighted.pkl','rb'))
        llm,_ = utils.build_llm_index(w, max_level=2 if D2ONLY else 3)
    except Exception as e:
        print(f"  W{w} skip: {e}", flush=True); continue
    a2=[0,0]; a3=[0,0,0]
    for prof in llm:
        d=len(prof)
        if d==2:
            (a,va),(b,vb)=sorted(prof)
            pa,pb=frozenset([(a,va)]),frozenset([(b,vb)])
            for q,lab in llm[prof].items():
                hab=human.get(prof,{}).get(q); la=llm.get(pa,{}).get(q)
                ha=human.get(pa,{}).get(q); lb=llm.get(pb,{}).get(q); hb=human.get(pb,{}).get(q)
                if any(x is None for x in (hab,la,ha,lb,hb)): continue
                if len({len(lab),len(hab),len(la),len(ha),len(lb),len(hb)})!=1: continue
                eAB=lab-hab; eA=la-ha; eB=lb-hb
                cs=max(utils.cosine_sim(eAB,eA),utils.cosine_sim(eAB,eB))
                a2[0 if cs>=utils.cosine_sim(eAB,eA+eB) else 1]+=1
        elif d==3 and not D2ONLY:
            feats=sorted(prof)
            singles=[frozenset([f]) for f in feats]
            pairs=[frozenset(p) for p in itertools.combinations(feats,2)]
            for q,lab in llm[prof].items():
                hab=human.get(prof,{}).get(q)
                if hab is None or len(hab)!=len(lab): continue
                es=[]; ok=True
                for s_ in singles:
                    lp=llm.get(s_,{}).get(q); hp=human.get(s_,{}).get(q)
                    if lp is None or hp is None or len(lp)!=len(lab): ok=False; break
                    es.append(lp-hp)
                if not ok: continue
                eprs=[]
                for p_ in pairs:
                    lp=llm.get(p_,{}).get(q); hp=human.get(p_,{}).get(q)
                    if lp is not None and hp is not None and len(lp)==len(lab): eprs.append(lp-hp)
                if not eprs: continue
                ep=lab-hab
                cs=max(utils.cosine_sim(ep,x) for x in es)
                cp=max(utils.cosine_sim(ep,x) for x in eprs)
                ca=utils.cosine_sim(ep,sum(es))
                if ca>=cs and ca>=cp: a3[2]+=1
                elif cp>=cs: a3[1]+=1
                else: a3[0]+=1
    n2=sum(a2)
    if n2:
        rows.append(dict(model=TAG,wave=w,depth=2,predictor="best_single",wins=a2[0],n_total=n2))
        rows.append(dict(model=TAG,wave=w,depth=2,predictor="additive",wins=a2[1],n_total=n2))
    n3=sum(a3)
    if n3:
        for pred,i in (("best_single",0),("best_pair",1),("additive",2)):
            rows.append(dict(model=TAG,wave=w,depth=3,predictor=pred,wins=a3[i],n_total=n3))
    print(f"  W{w}: d2 n={n2:,} single={100*a2[0]/max(n2,1):.1f}%" + (f"  d3 n={n3:,} add={100*a3[2]/max(n3,1):.1f}%" if n3 else ""), flush=True)
out=f"/tmp/rows_{TAG}.json"
json.dump(rows, open(out,"w"))
print(f"WROTE {out} ({len(rows)} rows)")
