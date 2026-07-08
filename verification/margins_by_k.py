"""Round-5 M1/M16: collapse share by option count, near-tie fraction, K>=3
robustness, and share where BOTH singles beat additive. gpt-4o-mini, 14 waves."""
import os,sys,json,itertools
import numpy as np,pandas as pd
from collections import defaultdict
sys.path.insert(0,'advanced_bias_analysis'); import utils
utils.USE_WEIGHTS=True; EXCL={"49"}
def waves():
    d='human_resp'
    return sorted((fn.rsplit('W',1)[1] for fn in os.listdir(d) if fn.startswith('American_Trends_Panel_W') and fn.rsplit('W',1)[1] not in EXCL and os.path.exists(os.path.join(d,fn,'responses.csv'))),key=int)
def load_options(w):
    return {q["question_id"]:[r["option"] for r in q["responses"]] for q in json.load(open(f'data/responses/survey_responses_W{w}.json'))}
def dist_n(s,opts,mn,wt=None):
    vc=s.value_counts(); n=int(sum(vc.get(o,0) for o in opts))
    if n<mn: return None
    if wt is None: c=np.array([vc.get(o,0) for o in opts],float)
    else:
        acc=dict.fromkeys(opts,0.0)
        for v,w in zip(s,wt):
            if v in acc and np.isfinite(w): acc[v]+=w
        c=np.array([acc[o] for o in opts],float)
    sm=c.sum(); return c/sm if sm>0 else None
utils.MODEL_TAG='gpt-4o-mini'
byK=defaultdict(lambda:[0,0]); margins=[]; both=0; tot=0; refused_k=defaultdict(int)
for w in waves():
    opts=load_options(w)
    df=pd.read_csv(f'human_resp/American_Trends_Panel_W{w}/responses.csv',low_memory=False)
    wcol=utils._weight_col(df,w); qids=[q for q in opts if q in df.columns]
    llm,_=utils.build_llm_index(w,max_level=2)
    masks={}
    for dim,col in utils.DIM_TO_COL.items():
        if col not in df.columns: continue
        for val in utils.DIM_VALUES[dim]:
            if val in utils.IGNORE_VALUES.get(dim,set()): continue
            m=(df[col]==val).to_numpy()
            if m.sum()>=20: masks[(dim,val)]=m
    hs={}
    for k,m in masks.items():
        sub=df.loc[m]; hs[k]={}
        for q in qids:
            sl=sub.dropna(subset=[q]); wt=sl[wcol].to_numpy() if wcol else None
            d=dist_n(sl[q],opts[q],20,wt)
            if d is not None: hs[k][q]=d
    for a,b in itertools.combinations(sorted(masks),2):
        if a[0]==b[0]: continue
        m=masks[a]&masks[b]
        if m.sum()<20: continue
        sub=df.loc[m]
        lab=llm.get(frozenset([a,b]),{}); la=llm.get(frozenset([a]),{}); lb=llm.get(frozenset([b]),{})
        for q in qids:
            if q not in hs.get(a,{}) or q not in hs.get(b,{}): continue
            lab_q,la_q,lb_q=lab.get(q),la.get(q),lb.get(q)
            if any(x is None for x in (lab_q,la_q,lb_q)): continue
            sl=sub.dropna(subset=[q]); wt=sl[wcol].to_numpy() if wcol else None
            hab=dist_n(sl[q],opts[q],20,wt)
            if hab is None: continue
            if len({len(hab),len(hs[a][q]),len(hs[b][q]),len(lab_q),len(la_q),len(lb_q)})!=1: continue
            K=len(lab_q)
            if 'Refused' in opts[q]: refused_k[K]+=1
            e_ab=lab_q-hab; e_a=la_q-hs[a][q]; e_b=lb_q-hs[b][q]
            ca=utils.cosine_sim(e_ab,e_a); cb=utils.cosine_sim(e_ab,e_b)
            cs=max(ca,cb); cmn=min(ca,cb); cadd=utils.cosine_sim(e_ab,e_a+e_b)
            sw=cs>=cadd
            byK[K][0 if sw else 1]+=1
            margins.append(abs(cs-cadd)); tot+=1
            if cmn>cadd: both+=1
    print(f"  W{w} done",flush=True)
margins=np.array(margins)
print("\n=== collapse share by option count K (gpt-4o-mini, 14 waves) ===")
gtot=g3=0; w3=0
for K in sorted(byK):
    n=sum(byK[K]); share=100*byK[K][0]/n
    ref=100*refused_k[K]/n if n else 0
    print(f"  K={K:2d}: n={n:>8,}  best-single={share:.1f}%  (items w/ Refused option: {ref:.0f}%)")
    gtot+=n
    if K>=3: g3+=n; w3+=byK[K][0]
print(f"\n  overall n={gtot:,}")
print(f"  K>=3 only: n={g3:,}  best-single={100*w3/g3:.1f}%")
print(f"  near-ties |cs-cadd|<0.01: {100*(margins<0.01).mean():.1f}%   <0.001: {100*(margins<0.001).mean():.1f}%   median margin={np.median(margins):.3f}")
print(f"  BOTH singles beat additive: {100*both/tot:.1f}%")
