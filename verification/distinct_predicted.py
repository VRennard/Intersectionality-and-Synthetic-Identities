"""Round-6 M5: does the additive prediction quantitatively reproduce the
distinctiveness growth? Per pair/triple cell: observed corrected ||p_g-pop||^2
vs the additive prediction's corrected ||sum singles - k*pop||^2."""
import os, sys, json, itertools
import numpy as np, pandas as pd
sys.path.insert(0, 'advanced_bias_analysis'); import utils
utils.USE_WEIGHTS = True
WAVES = ("26","27","29","32","34","36","41","42","43","45","50","54","82","92")
obs={2:[],3:[]}; pred={2:[],3:[]}
for w in WAVES:
    opts = {q["question_id"]:[r["option"] for r in q["responses"]]
            for q in json.load(open(f'data/responses/survey_responses_W{w}.json'))}
    df = pd.read_csv(f'human_resp/American_Trends_Panel_W{w}/responses.csv', low_memory=False)
    wcol = utils._weight_col(df, w)
    qids=[q for q in opts if q in df.columns]
    def dist_n(mask,q,mn=20):
        sl=(df.loc[mask] if mask is not None else df).dropna(subset=[q])
        n=len(sl)
        if n<mn: return None,0
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
            if m.sum()>=20: masks[(dim,val)]=m
    pop={}; sing={}
    for q in qids:
        d,n=dist_n(None,q); 
        if d is not None: pop[q]=(d,n)
    for k,m in masks.items():
        sing[k]={q:dist_n(m,q) for q in qids}
    def nv(p,n): return float((p*(1-p)).sum()/n)
    for depth in (2,3):
        for combo in itertools.combinations(sorted(masks),depth):
            if len({c[0] for c in combo})!=depth: continue
            m=masks[combo[0]].copy()
            for c in combo[1:]: m&=masks[c]
            if m.sum()<20: continue
            for q in qids:
                if q not in pop: continue
                ppop,npop=pop[q]
                parts=[]
                ok=True
                for c in combo:
                    pc,nc=sing[c][q]
                    if pc is None or len(pc)!=len(ppop): ok=False; break
                    parts.append((pc,nc))
                if not ok: continue
                pg,ng=dist_n(m,q)
                if pg is None or len(pg)!=len(ppop): continue
                # observed corrected
                diff=pg-ppop
                obs[depth].append(float(diff@diff) - nv(pg,ng) - nv(ppop,npop))
                # predicted: sum singles - depth*pop
                s=sum(p for p,_ in parts) - depth*ppop
                noise=sum(nv(p,n) for p,n in parts) + (depth**2)*nv(ppop,npop)
                pred[depth].append(float(s@s) - noise)
    print(f"  W{w} done (d2 {len(obs[2]):,}, d3 {len(obs[3]):,})", flush=True)
print("\n=== predicted vs observed corrected squared distinctiveness ===")
for d in (2,3):
    print(f"  depth {d}: observed={np.mean(obs[d]):.4f}   additive-predicted={np.mean(pred[d]):.4f}   (n={len(obs[d]):,})")
