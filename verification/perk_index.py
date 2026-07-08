"""Referee W3: collapse index by option-count band. Observed per-K share
(flagship, 14 waves) + per-K calibration endpoints (same construction as
human_additivity_calibration.calibration, tallied by K). --weighted."""
import os, sys, json, pickle, itertools
from collections import defaultdict
import numpy as np
BASE=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(BASE,"advanced_bias_analysis"))
import utils
utils.USE_WEIGHTS=True; utils.MODEL_TAG="gpt-4o-mini"
HERE=os.path.join(BASE,"verification")
RNG=np.random.default_rng(0)
STOCH=range(1,5)
# noise pool by K
runs={}
for i in STOCH:
    d={}
    for ln in open(os.path.join(BASE,"data/results/stoch_test",f"W26_run{i}.jsonl")):
        ln=ln.strip()
        if not ln: continue
        try: r=json.loads(ln)
        except: continue
        if r.get("status")!="success": continue
        v=np.array(r["response_distribution"],float)
        if v.sum()<=0: continue
        d[(tuple(sorted(r["demographics"])),r["question_id"])]=v/v.sum()
    runs[i]=d
pool=defaultdict(list)
for i,j in itertools.combinations(STOCH,2):
    for k,da in runs[i].items():
        db=runs[j].get(k)
        if db is not None and len(db)==len(da): pool[len(da)].append((da-db)/np.sqrt(2))
pool={K:np.array(v) for K,v in pool.items()}
def band(K): return "2" if K==2 else "3-5" if K<=5 else "6-8" if K<=8 else "9+"
# per-K observed + endpoints over 14 waves
WAVES=("26","27","29","32","34","36","41","42","43","45","50","54","82","92")
obs=defaultdict(lambda:[0,0]); ends=defaultdict(lambda:{"add":[0,0],"col":[0,0]})
for w in WAVES:
    human,_=pickle.load(open(os.path.join(HERE,"cache",f"human_W{w}_weighted.pkl"),"rb"))
    llm,_=utils.build_llm_index(w, max_level=2)
    def bias(p,q):
        lp=llm.get(p,{}).get(q); hp=human.get(p,{}).get(q)
        if lp is None or hp is None or len(lp)!=len(hp): return None
        return lp-hp
    for prof in llm:
        if len(prof)!=2: continue
        (a,va),(b,vb)=sorted(prof)
        pa,pb=frozenset([(a,va)]),frozenset([(b,vb)])
        for q in llm[prof]:
            eA,eB,eAB=bias(pa,q),bias(pb,q),bias(prof,q)
            if eA is None or eB is None or eAB is None: continue
            K=len(eA); bd=band(K)
            cs=max(utils.cosine_sim(eAB,eA),utils.cosine_sim(eAB,eB))
            obs[bd][0]+= cs>=utils.cosine_sim(eAB,eA+eB); obs[bd][1]+=1
            if K in pool and len(pool[K]):
                eta=pool[K][RNG.integers(len(pool[K]))]
                for reg,et in (("add",eA+eB),("col",eA if np.linalg.norm(eA)>=np.linalg.norm(eB) else eB)):
                    es=et+eta
                    c1=max(utils.cosine_sim(es,eA),utils.cosine_sim(es,eB))
                    ends[bd][reg][0]+= c1>=utils.cosine_sim(es,eA+eB); ends[bd][reg][1]+=1
    print(f"W{w} done", flush=True)
print(f"\n{'K band':8s} {'n':>10s} {'observed':>9s} {'add-end':>8s} {'col-end':>8s} {'index':>6s}")
for bd in ("2","3-5","6-8","9+"):
    if obs[bd][1]==0: continue
    o=100*obs[bd][0]/obs[bd][1]
    lo=100*ends[bd]["add"][0]/max(ends[bd]["add"][1],1)
    hi=100*ends[bd]["col"][0]/max(ends[bd]["col"][1],1)
    idx=(o-lo)/(hi-lo) if hi>lo else float('nan')
    print(f"{bd:8s} {obs[bd][1]:>10,} {o:>8.1f}% {lo:>7.1f}% {hi:>7.1f}% {idx:>6.2f}")
