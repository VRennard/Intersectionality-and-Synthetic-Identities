"""Independent reimplementation of the human steering-space contest + the
additive-truth reference, to adjudicate 44.4/49.3 (June log, in paper) vs
49.3/44.9 (current script rerun). Faithful to the stated method: weighted
distributions, n>=100 cells, synthetic pair = clip(pop + sA + sB) renorm,
observed through one multinomial draw at the cell's raw n."""
import os, sys, json, itertools
import numpy as np, pandas as pd
BASE=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(BASE,"advanced_bias_analysis"))
import utils
RNG=np.random.default_rng(7)
def cos(a,b):
    na,nb=np.linalg.norm(a),np.linalg.norm(b)
    return float(a@b/(na*nb)) if na>0 and nb>0 else 0.0
WAVES=("26","27","29","32","34","36","41","42","43","45","50","54","82","92")
res={100:[0,0],200:[0,0]}; ref={100:[0,0],200:[0,0]}
for w in WAVES:
    qopts={}
    survey=json.load(open(os.path.join(BASE,f"data/responses/survey_responses_W{w}.json")))
    for q in survey: qopts[q['question_id']]=[r['option'] for r in q['responses']]
    df=pd.read_csv(os.path.join(BASE,"human_resp",f"American_Trends_Panel_W{w}","responses.csv"),low_memory=False)
    wcol=utils._weight_col(df,w)
    def dist(sl,qid,minn):
        sl=sl.dropna(subset=[qid])
        n=len(sl)
        if n<minn: return None,0
        wts=sl[wcol].to_numpy() if wcol and wcol in sl else np.ones(n)
        opts=qopts[qid]
        v=np.array([wts[(sl[qid]==o).to_numpy()].sum() for o in opts],float)
        return (v/v.sum() if v.sum()>0 else None), n
    qids=[q for q in qopts if q in df.columns]
    masks={}
    for dim,col in utils.DIM_TO_COL.items():
        if col not in df.columns: continue
        for val in utils.DIM_VALUES[dim]:
            if val in utils.IGNORE_VALUES.get(dim,set()): continue
            m=(df[col]==val).to_numpy()
            if m.sum()>=100: masks[(dim,val)]=m
    pop={};
    for qid in qids:
        d,_=dist(df,qid,20)
        if d is not None: pop[qid]=d
    singles={}
    for k,m in masks.items():
        singles[k]={}
        for qid in qids:
            if qid not in pop: continue
            d,_=dist(df.loc[m],qid,100)
            if d is not None: singles[k][qid]=d
    for a,b in itertools.combinations(sorted(masks),2):
        if a[0]==b[0]: continue
        m=masks[a]&masks[b]
        if m.sum()<100: continue
        sub=df.loc[m]
        for qid in qids:
            if qid not in pop or qid not in singles[a] or qid not in singles[b]: continue
            d_ab,n_ab=dist(sub,qid,100)
            if d_ab is None: continue
            sA=singles[a][qid]-pop[qid]; sB=singles[b][qid]-pop[qid]; sAB=d_ab-pop[qid]
            c_add=cos(sAB,sA+sB); c_sgl=max(cos(sAB,sA),cos(sAB,sB))
            pt=np.clip(pop[qid]+sA+sB,0,None)
            if pt.sum()<=0: continue
            pt/=pt.sum()
            ssyn=RNG.multinomial(n_ab,pt)/n_ab-pop[qid]
            r_add=cos(ssyn,sA+sB); r_sgl=max(cos(ssyn,sA),cos(ssyn,sB))
            for thr in (100,200):
                if n_ab>=thr:
                    res[thr][0 if c_add>c_sgl else 1]+=1
                    ref[thr][0 if r_add>r_sgl else 1]+=1
    print(f"W{w} done", flush=True)
for thr in (100,200):
    t=sum(res[thr]); tr=sum(ref[thr])
    print(f"ADJUDICATION n>={thr}: HUMAN additive={100*res[thr][0]/t:.1f}% ({t:,})  REF ceiling={100*ref[thr][0]/tr:.1f}% ({tr:,})")
