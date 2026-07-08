"""Referee #4: are the calibration endpoints stable to the noise magnitude?
Recompute additive-truth & collapse-truth best-single endpoints with eta scaled
0.5/1/1.5/2x (eta from gpt-4o-mini run-to-run deltas)."""
import os,sys,json,pickle,itertools,numpy as np
from collections import defaultdict
sys.path.insert(0,'advanced_bias_analysis'); import utils
utils.USE_WEIGHTS=True; MODEL='gpt-4o-mini'; RNG=np.random.default_rng(3)
runs={}
for i in (1,2,3,4):
    d={}
    for l in open(f'data/results/stoch_test/W26_run{i}.jsonl'):
        l=l.strip()
        if not l: continue
        try: r=json.loads(l)
        except: continue
        if r.get('status')!='success': continue
        dist=np.array(r['response_distribution'],float)
        if dist.sum()<=0: continue
        d[(tuple(sorted(r['demographics'])),r['question_id'])]=dist/dist.sum()
    runs[i]=d
npool=defaultdict(list)
for i,j in itertools.combinations((1,2,3,4),2):
    for k,da in runs[i].items():
        db=runs[j].get(k)
        if db is not None and len(db)==len(da): npool[len(da)].append((da-db)/np.sqrt(2))
for K in npool: npool[K]=np.array(npool[K])
utils.MODEL_TAG=MODEL
cells=[]
for wave in ('26','43'):
    human,_=pickle.load(open(f'verification/cache/human_W{wave}_weighted.pkl','rb'))
    llm,_=utils.build_llm_index(wave,max_level=2)
    def bias(p,q):
        lp=llm.get(p,{}).get(q); hp=human.get(p,{}).get(q)
        if lp is None or hp is None or len(lp)!=len(hp): return None
        return lp-hp
    nd=0
    for prof in llm:
        if len(prof)!=2 or nd>=25000: continue
        (a,va),(b,vb)=sorted(prof); pa,pb=frozenset([(a,va)]),frozenset([(b,vb)])
        for q in llm[prof]:
            ea,eb=bias(pa,q),bias(pb,q)
            if ea is None or eb is None: continue
            K=len(ea)
            if K not in npool or not len(npool[K]): continue
            cells.append((ea,eb,K)); nd+=1
print(f"{MODEL}: {len(cells):,} synthetic cells")
for scale in (0.5,1.0,1.5,2.0):
    aw=cw=at=ct=0
    for ea,eb,K in cells:
        eta=npool[K][RNG.integers(0,len(npool[K]))]*scale
        edom=ea if np.linalg.norm(ea)>=np.linalg.norm(eb) else eb
        for reg,etrue in (('add',ea+eb),('col',edom)):
            es=etrue+eta
            csgl=max(utils.cosine_sim(es,ea),utils.cosine_sim(es,eb)); cadd=utils.cosine_sim(es,ea+eb)
            single=csgl>=cadd
            if reg=='add': aw+=single; at+=1
            else: cw+=single; ct+=1
    lo,hi=100*aw/at,100*cw/ct
    obs=78.4; idx=(obs-lo)/(hi-lo)
    print(f"  eta x{scale}:  additive-truth floor={lo:.1f}%  collapse-truth ceil={hi:.1f}%  -> index(obs 78.4)={idx:.2f}",flush=True)
