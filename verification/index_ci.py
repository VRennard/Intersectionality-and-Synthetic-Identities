"""Reviewer major 4: CI on the flagship's self-calibrated collapse index.
Endpoint uncertainty via jackknife over the six repeat-pairs (each synthetic
cell's eta assigned to one specific pair, one pass, six sub-tallies);
observed-share uncertainty via wave-cluster bootstrap; delta-method combine."""
import os, sys, json, pickle, itertools
from collections import defaultdict
import numpy as np
BASE=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(BASE,"advanced_bias_analysis"))
import utils
utils.USE_WEIGHTS=True; utils.MODEL_TAG="gpt-4o-mini"
HERE=os.path.join(BASE,"verification")
RNG=np.random.default_rng(0)
STOCH=list(range(1,5))
PAIRS=list(itertools.combinations(STOCH,2))
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
pool=defaultdict(lambda: defaultdict(list))   # pair -> K -> deltas
for pi,(i,j) in enumerate(PAIRS):
    for k,da in runs[i].items():
        db=runs[j].get(k)
        if db is not None and len(db)==len(da): pool[pi][len(da)].append((da-db)/np.sqrt(2))
pool={pi:{K:np.array(v) for K,v in d.items()} for pi,d in pool.items()}
WAVES=("26","27","29","32","34","36","41","42","43","45","50","54","82","92")
# endpoints tallied per repeat-pair; observed tallied per wave
end=defaultdict(lambda:{"add":[0,0],"col":[0,0]})
obs_w={}
for w in WAVES:
    human,_=pickle.load(open(os.path.join(HERE,"cache",f"human_W{w}_weighted.pkl"),"rb"))
    llm,_=utils.build_llm_index(w, max_level=2)
    def bias(p,q):
        lp=llm.get(p,{}).get(q); hp=human.get(p,{}).get(q)
        if lp is None or hp is None or len(lp)!=len(hp): return None
        return lp-hp
    ww=[0,0]
    for prof in llm:
        if len(prof)!=2: continue
        (a,va),(b,vb)=sorted(prof)
        pa,pb=frozenset([(a,va)]),frozenset([(b,vb)])
        for q in llm[prof]:
            eA,eB,eAB=bias(pa,q),bias(pb,q),bias(prof,q)
            if eA is None or eB is None or eAB is None: continue
            K=len(eA)
            cs=max(utils.cosine_sim(eAB,eA),utils.cosine_sim(eAB,eB))
            ww[0]+= cs>=utils.cosine_sim(eAB,eA+eB); ww[1]+=1
            pi=RNG.integers(len(PAIRS))
            if K in pool[pi] and len(pool[pi][K]):
                eta=pool[pi][K][RNG.integers(len(pool[pi][K]))]
                for reg,et in (("add",eA+eB),("col",eA if np.linalg.norm(eA)>=np.linalg.norm(eB) else eB)):
                    es=et+eta
                    c1=max(utils.cosine_sim(es,eA),utils.cosine_sim(es,eB))
                    end[pi][reg][0]+= c1>=utils.cosine_sim(es,eA+eB); end[pi][reg][1]+=1
    obs_w[w]=ww
    print(f"W{w} done", flush=True)
# point estimates
O=100*sum(v[0] for v in obs_w.values())/sum(v[1] for v in obs_w.values())
lo_all=100*sum(end[p]["add"][0] for p in end)/sum(end[p]["add"][1] for p in end)
hi_all=100*sum(end[p]["col"][0] for p in end)/sum(end[p]["col"][1] for p in end)
IDX=(O-lo_all)/(hi_all-lo_all)
# SE_obs: wave bootstrap
ws=list(obs_w); B=10000; st=[]
for _ in range(B):
    smp=RNG.choice(ws,len(ws),replace=True)
    st.append(100*sum(obs_w[w][0] for w in smp)/sum(obs_w[w][1] for w in smp))
se_o=np.std(st)
# SE_endpoints: jackknife over the six pairs
jlo=[]; jhi=[]
for drop in end:
    keep=[p for p in end if p!=drop]
    jlo.append(100*sum(end[p]["add"][0] for p in keep)/sum(end[p]["add"][1] for p in keep))
    jhi.append(100*sum(end[p]["col"][0] for p in keep)/sum(end[p]["col"][1] for p in keep))
n=len(end)
se_lo=np.sqrt((n-1)/n*sum((x-np.mean(jlo))**2 for x in jlo))
se_hi=np.sqrt((n-1)/n*sum((x-np.mean(jhi))**2 for x in jhi))
# delta method on I=(O-lo)/(hi-lo)
d=hi_all-lo_all
gO=1/d; glo=-(hi_all-O)/d**2; ghi=-(O-lo_all)/d**2
se_I=np.sqrt((gO*se_o)**2+(glo*se_lo)**2+(ghi*se_hi)**2)
print(f"\nFLAGSHIP INDEX CI: obs={O:.1f}(SE {se_o:.2f})  endpoints={lo_all:.1f}(SE {se_lo:.2f})/{hi_all:.1f}(SE {se_hi:.2f})")
print(f"  index={IDX:.3f}  SE={se_I:.3f}  95% CI [{IDX-1.96*se_I:.2f}, {IDX+1.96*se_I:.2f}]")
