"""Human depth-3 additivity contest, mirroring the model d3 contest in
steering space: for every human TRIPLE cell with n>=100 (population, all
three singles, all three sub-pairs, and the triple observed), the additive
predictor s_A+s_B+s_C competes against the best single and the best
sub-pair-plus-single-free... (predictors mirror the model contest: 3
singles, 3 measured sub-pairs, 1 additive sum) for cosine similarity to
the cell's real steering. Matched additive-truth ceiling via one
multinomial draw at the cell's n. Weighted (--weighted semantics baked in)."""
import os, sys, json, itertools
import numpy as np, pandas as pd
BASE=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(BASE,"advanced_bias_analysis"))
import utils
utils.USE_WEIGHTS=True
RNG=np.random.default_rng(11)
def cos(a,b):
    na,nb=np.linalg.norm(a),np.linalg.norm(b)
    return float(a@b/(na*nb)) if na>0 and nb>0 else 0.0
WAVES=("26","27","29","32","34","36","41","42","43","45","50","54","82","92")
res=[0,0,0]; ref=[0,0,0]   # [additive, sub-pair, single] wins
for w in WAVES:
    survey=json.load(open(os.path.join(BASE,f"data/responses/survey_responses_W{w}.json")))
    qopts={q['question_id']:[r['option'] for r in q['responses']] for q in survey}
    df=pd.read_csv(os.path.join(BASE,"human_resp",f"American_Trends_Panel_W{w}","responses.csv"),low_memory=False)
    wcol=utils._weight_col(df,w)
    def dist(sl,qid,minn):
        sl=sl.dropna(subset=[qid]); n=len(sl)
        if n<minn: return None,0
        wts=sl[wcol].to_numpy() if wcol and wcol in sl else np.ones(n)
        v=np.array([wts[(sl[qid]==o).to_numpy()].sum() for o in qopts[qid]],float)
        return (v/v.sum() if v.sum()>0 else None), n
    qids=[q for q in qopts if q in df.columns]
    CORE=["Age","Gender","Race","Income","Political Party","Religion"]
    masks={}
    for dim,col in utils.DIM_TO_COL.items():
        if dim not in CORE or col not in df.columns: continue
        for val in utils.DIM_VALUES[dim]:
            if val in utils.IGNORE_VALUES.get(dim,set()): continue
            m=(df[col]==val).to_numpy()
            if m.sum()>=100: masks[(dim,val)]=m
    pop={}
    for qid in qids:
        d,_=dist(df,qid,20)
        if d is not None: pop[qid]=d
    def steer(mask,qid,minn=100):
        d,n=dist(df.loc[mask],qid,minn)
        return (d-pop[qid],n) if d is not None and qid in pop else (None,0)
    keys=sorted(masks)
    for a,b,c in itertools.combinations(keys,3):
        if len({a[0],b[0],c[0]})!=3: continue
        m3=masks[a]&masks[b]&masks[c]
        if m3.sum()<100: continue
        for qid in qids:
            if qid not in pop: continue
            sT,n3=steer(m3,qid)
            if sT is None: continue
            singles=[]
            ok=True
            for k in (a,b,c):
                sv,_=steer(masks[k],qid)
                if sv is None: ok=False; break
                singles.append(sv)
            if not ok: continue
            prs=[]
            for k1,k2 in itertools.combinations((a,b,c),2):
                sv,_=steer(masks[k1]&masks[k2],qid)
                if sv is not None: prs.append(sv)
            if not prs: continue
            add=singles[0]+singles[1]+singles[2]
            cA=cos(sT,add)
            cP=max(cos(sT,x) for x in prs)
            cS=max(cos(sT,x) for x in singles)
            if cA>=cP and cA>=cS: res[0]+=1
            elif cP>=cS: res[1]+=1
            else: res[2]+=1
            # ceiling: perfectly additive triple seen through multinomial noise at n3
            pt=np.clip(pop[qid]+add,0,None)
            if pt.sum()>0:
                pt/=pt.sum()
                ssyn=RNG.multinomial(n3,pt)/n3-pop[qid]
                rA=cos(ssyn,add); rP=max(cos(ssyn,x) for x in prs); rS=max(cos(ssyn,x) for x in singles)
                if rA>=rP and rA>=rS: ref[0]+=1
                elif rP>=rS: ref[1]+=1
                else: ref[2]+=1
    print(f"W{w} done (cum n={sum(res):,})", flush=True)
t=sum(res); tr=sum(ref)
print(f"\nHUMAN D3 CONTEST (weighted, n>=100 triples): n={t:,}")
print(f"  additive={100*res[0]/t:.1f}%  best sub-pair={100*res[1]/t:.1f}%  best single={100*res[2]/t:.1f}%")
print(f"  CEILING (additive truth + multinomial noise): additive={100*ref[0]/tr:.1f}%  sub-pair={100*ref[1]/tr:.1f}%  single={100*ref[2]/tr:.1f}%")
