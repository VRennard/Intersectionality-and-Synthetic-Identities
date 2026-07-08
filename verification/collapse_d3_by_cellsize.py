"""Referee #3 gap: depth-3 additive-share under the same human triple-cell
restriction (n>=20/100/200). Additive predictor = sum(singles) - 2*e_pop
(or raw sum); competitors = 3 best-single + 3 best-pair."""
import os,sys,json,itertools
import numpy as np, pandas as pd
sys.path.insert(0,'advanced_bias_analysis'); import utils
utils.USE_WEIGHTS=True; AVG=utils.AVG_PROFILE; EXCL={"49"}; THRESH=[20,100,200]
def waves():
    d='human_resp'
    return sorted((fn.rsplit('W',1)[1] for fn in os.listdir(d) if fn.startswith('American_Trends_Panel_W') and fn.rsplit('W',1)[1] not in EXCL and os.path.exists(os.path.join(d,fn,'responses.csv'))),key=int)
def lop(w): return {q['question_id']:[r['option'] for r in q['responses']] for q in json.load(open(f'data/responses/survey_responses_W{w}.json'))}
def dn(s,opts,mn,wt=None):
    vc=s.value_counts(); n=int(sum(vc.get(o,0) for o in opts))
    if n<mn: return None,0
    c=np.array([vc.get(o,0) for o in opts],float) if wt is None else None
    if wt is not None:
        acc=dict.fromkeys(opts,0.0)
        for v,w in zip(s,wt):
            if v in acc and np.isfinite(w): acc[v]+=w
        c=np.array([acc[o] for o in opts],float)
    sm=c.sum(); return (c/sm,n) if sm>0 else (None,0)
TRIP=getattr(utils,'TRIPLE_FEATURES',None) or ['Age','Gender','Race','Income']
for model in sys.argv[1:] or ['gpt-4o-mini']:
    utils.MODEL_TAG=model
    win={t:[0,0] for t in THRESH}   # additive, other
    for w in waves():
        opts=lop(w); df=pd.read_csv(f'human_resp/American_Trends_Panel_W{w}/responses.csv',low_memory=False)
        wcol=utils._weight_col(df,w); qids=[q for q in opts if q in df.columns]
        llm,_=utils.build_llm_index(w,max_level=3); lpop=llm.get(AVG,{})
        masks={}
        for dim,col in utils.DIM_TO_COL.items():
            if col not in df.columns or dim not in TRIP: continue
            for val in utils.DIM_VALUES[dim]:
                if val in utils.IGNORE_VALUES.get(dim,set()): continue
                m=(df[col]==val).to_numpy()
                if m.sum()>=20: masks[(dim,val)]=m
        def hd(mask,q):
            sl=df.loc[mask].dropna(subset=[q]); wt=sl[wcol].to_numpy() if wcol else None
            return dn(sl[q],opts[q],20,wt)
        hpop={}
        for q in qids:
            d,_=hd(np.ones(len(df),bool),q)
            if d is not None: hpop[q]=d
        for combo in itertools.combinations(sorted(masks),3):
            dims={c[0] for c in combo}
            if len(dims)<3: continue
            a,b,c=combo; m=masks[a]&masks[b]&masks[c]
            if m.sum()<20: continue
            singles=[frozenset([x]) for x in (a,b,c)]; pairs=[frozenset(p) for p in itertools.combinations((a,b,c),2)]
            for q in qids:
                if q not in hpop: continue
                habc,nabc=hd(m,q)
                if habc is None: continue
                def e(prof):
                    l=llm.get(prof,{}).get(q)
                    if prof==AVG: h=hpop.get(q)
                    else:
                        mm=np.ones(len(df),bool)
                        for x in prof: mm=mm&masks.get(x,np.zeros(len(df),bool))
                        h,_=hd(mm,q)
                    if l is None or h is None or len(l)!=len(h): return None
                    return l-h
                eabc=(llm.get(frozenset(combo),{}).get(q)); 
                if eabc is None or len(eabc)!=len(habc): continue
                eabc=eabc-habc
                es=[e(s) for s in singles]; eprs=[e(p) for p in pairs]; epop=e(AVG)
                if any(x is None for x in es) or epop is None: continue
                eprs=[x for x in eprs if x is not None]
                if not eprs: continue
                cs=max(utils.cosine_sim(eabc,x) for x in es); cp=max(utils.cosine_sim(eabc,x) for x in eprs)
                cadd=utils.cosine_sim(eabc,sum(es)-2*epop)
                add_wins=(cadd>=cs and cadd>=cp)
                for t in THRESH:
                    if nabc>=t: win[t][0 if add_wins else 1]+=1
    print(f"\n=== {model} (d3 additive-share, e_pop-corrected) ===")
    for t in THRESH:
        tot=sum(win[t])
        if tot: print(f"  n>={t:3d} (n={tot:>6,}):  additive-share={100*win[t][0]/tot:.1f}%")
