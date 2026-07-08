"""Round-6 M4: does predictor-target noise sharing (the AB respondents being
inside the A and B cells) drive the human additive win rate? Stratify the
human contest by leakage = max(n_AB/n_A, n_AB/n_B); if the win rate is flat
across strata, shared noise is not the driver."""
import os, sys, json, itertools
import numpy as np, pandas as pd
sys.path.insert(0, 'advanced_bias_analysis'); import utils
utils.USE_WEIGHTS = True
WAVES = ("26","27","29","32","34","36","41","42","43","45","50","54","82","92")
strata = {0:[0,0],1:[0,0],2:[0,0]}   # tertile -> [add_wins, total]
leaks=[]
recs=[]
for w in WAVES:
    opts = {q["question_id"]:[r["option"] for r in q["responses"]]
            for q in json.load(open(f'data/responses/survey_responses_W{w}.json'))}
    df = pd.read_csv(f'human_resp/American_Trends_Panel_W{w}/responses.csv', low_memory=False)
    wcol = utils._weight_col(df, w)
    qids=[q for q in opts if q in df.columns]
    def dist_n(mask,q):
        sl=(df.loc[mask] if mask is not None else df).dropna(subset=[q])
        n=len(sl)
        if n<100: return None,0
        wt=sl[wcol].to_numpy() if wcol else np.ones(n)
        acc=dict.fromkeys(opts[q],0.0)
        for v_,w_ in zip(sl[q],wt):
            if v_ in acc and np.isfinite(w_): acc[v_]+=w_
        c=np.array([acc[o] for o in opts[q]],float)
        return (c/c.sum(), n) if c.sum()>0 else (None,0)
    masks={}
    for dim,col in utils.DIM_TO_COL.items():
        if col not in df.columns: continue
        for val in utils.DIM_VALUES[dim]:
            if val in utils.IGNORE_VALUES.get(dim,set()): continue
            m=(df[col]==val).to_numpy()
            if m.sum()>=100: masks[(dim,val)]=m
    pop={q:dist_n(None,q)[0] for q in qids}
    sing={k:{q:dist_n(m,q) for q in qids} for k,m in masks.items()}
    for a,b in itertools.combinations(sorted(masks),2):
        if a[0]==b[0]: continue
        m=masks[a]&masks[b]
        if m.sum()<100: continue
        for q in qids:
            pq=pop.get(q); sa,na=sing[a][q]; sb,nb=sing[b][q]
            if pq is None or sa is None or sb is None: continue
            sab,nab=dist_n(m,q)
            if sab is None: continue
            sA=sa-pq; sB=sb-pq; sAB=sab-pq
            cs=max(utils.cosine_sim(sAB,sA),utils.cosine_sim(sAB,sB))
            ca=utils.cosine_sim(sAB,sA+sB)
            leak=max(nab/na, nab/nb)
            recs.append((leak, 1 if ca>cs else 0, nab))
    print(f"  W{w} done ({len(recs):,})", flush=True)
leaks=np.array([r[0] for r in recs]); wins=np.array([r[1] for r in recs]); ns=np.array([r[2] for r in recs])
qs=np.quantile(leaks,[1/3,2/3])
print(f"\nleakage tertile cut points: {qs[0]:.3f}, {qs[1]:.3f}  (n={len(recs):,})")
for lo,hi,name in ((0,qs[0],"low"),(qs[0],qs[1],"mid"),(qs[1],1.01,"high")):
    m=(leaks>=lo)&(leaks<hi)
    print(f"  leakage {name:4s} [{lo:.2f},{hi:.2f}): additive wins {100*wins[m].mean():.1f}%  (n={m.sum():,})")
print("\nwithin n_AB bands (controls the target-noise confound):")
for nlo,nhi in ((100,200),(200,400),(400,10**9)):
    bm=(ns>=nlo)&(ns<nhi)
    if bm.sum()<3000: continue
    ql=np.quantile(leaks[bm],[1/3,2/3])
    row=[]
    for lo,hi in ((0,ql[0]),(ql[0],ql[1]),(ql[1],1.01)):
        m=bm&(leaks>=lo)&(leaks<hi)
        row.append(f"{100*wins[m].mean():.1f}% (n={m.sum():,})")
    print(f"  n_AB [{nlo},{'inf' if nhi>10**8 else nhi}): low={row[0]}  mid={row[1]}  high={row[2]}")
