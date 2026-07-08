"""Rescore mistral_latest on the merged (full-coverage) data:
depth-2 collapse + depth-3 (single/pair/additive), per wave, with
wave-cluster bootstrap CIs on the 14-wave (W49-excluded) basis."""
import os, sys, pickle, itertools, json
import numpy as np
sys.path.insert(0, 'advanced_bias_analysis'); import utils
utils.USE_WEIGHTS = True
utils.MODEL_TAG = "mistral_latest"
WAVES = ("26","27","29","32","34","36","41","42","43","45","50","54","82","92")
w2 = {}; w3 = {}
for w in WAVES:
    human,_ = pickle.load(open(f'verification/cache/human_W{w}_weighted.pkl','rb'))
    llm,_ = utils.build_llm_index(w, max_level=3)
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
        elif d==3:
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
    w2[w]=a2; w3[w]=a3
    print(f"  W{w}: d2 n={sum(a2):,} single={100*a2[0]/max(sum(a2),1):.1f}%   d3 n={sum(a3):,} add={100*a3[2]/max(sum(a3),1):.1f}%", flush=True)
def boot(waves_dict, idx, tot_idx=None):
    RNG=np.random.default_rng(0); ws=list(waves_dict)
    stats=[]
    for _ in range(10000):
        smp=RNG.choice(ws, len(ws), replace=True)
        num=sum(waves_dict[w][idx] for w in smp)
        den=sum(sum(waves_dict[w]) for w in smp)
        stats.append(100*num/den)
    return np.percentile(stats,[2.5,97.5])
t2=[sum(w2[w][i] for w in WAVES) for i in range(2)]
t3=[sum(w3[w][i] for w in WAVES) for i in range(3)]
n2=sum(t2); n3=sum(t3)
ci2=boot(w2,0); ci3=boot(w3,2)
print(f"\nMISTRAL-7B MERGED (14-wave):")
print(f"  d2: n={n2:,} best-single={100*t2[0]/n2:.1f}%  CI[{ci2[0]:.1f},{ci2[1]:.1f}]")
print(f"  d3: n={n3:,} additive={100*t3[2]/n3:.1f}%  CI[{ci3[0]:.1f},{ci3[1]:.1f}]  pair={100*t3[1]/n3:.1f}% single={100*t3[0]/n3:.1f}%")
