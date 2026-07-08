"""#3 depth-3 gap (fast): additive-share at n>=20/100/200, dists from the human
index (built once), n from vectorized notna() per triple."""
import os,sys,json,itertools
import numpy as np, pandas as pd
sys.path.insert(0,'advanced_bias_analysis'); import utils
utils.USE_WEIGHTS=True; AVG=utils.AVG_PROFILE; EXCL={"49"}; THRESH=[20,100,200]
TRIP=['Age','Gender','Race','Income','Political Party','Religion','Education']
def waves():
    d='human_resp'
    return sorted((fn.rsplit('W',1)[1] for fn in os.listdir(d) if fn.startswith('American_Trends_Panel_W') and fn.rsplit('W',1)[1] not in EXCL and os.path.exists(os.path.join(d,fn,'responses.csv'))),key=int)
for model in sys.argv[1:] or ['gpt-4o-mini']:
    utils.MODEL_TAG=model; win={t:[0,0] for t in THRESH}; winr={t:[0,0] for t in THRESH}
    for w in waves():
        df=pd.read_csv(f'human_resp/American_Trends_Panel_W{w}/responses.csv',low_memory=False)
        llm,qm=utils.build_llm_index(w,max_level=3)
        human=utils.load_human_index(w,list(qm.keys()),qm,max_level=3)
        if not llm.get(AVG): continue
        qids=list(qm.keys())
        masks={}
        for dim,col in utils.DIM_TO_COL.items():
            if col not in df.columns or dim not in TRIP: continue
            for val in utils.DIM_VALUES[dim]:
                if val in utils.IGNORE_VALUES.get(dim,set()): continue
                key=(dim,val); m=(df[col]==val).to_numpy()
                if m.sum()>=20: masks[key]=m
        def bias(prof,q):
            l=llm.get(prof,{}).get(q); h=human.get(prof,{}).get(q)
            if l is None or h is None or len(l)!=len(h): return None
            return l-h
        for prof,qd in llm.items():
            if len(prof)!=3: continue
            if not all(f in masks for f in prof): continue
            m=masks[list(prof)[0]].copy()
            for f in list(prof)[1:]: m=m&masks[f]
            if m.sum()<20: continue
            # per-question valid n for this triple (vectorized)
            nq=df.loc[m,[q for q in qids if q in df.columns]].notna().sum()
            feats=sorted(prof); singles=[frozenset([f]) for f in feats]; pairs=[frozenset(p) for p in itertools.combinations(feats,2)]
            for q in qd:
                n=int(nq.get(q,0))
                if n<20: continue
                ep=bias(prof,q); epop=bias(AVG,q)
                es=[bias(s,q) for s in singles]; eprs=[x for x in (bias(p,q) for p in pairs) if x is not None]
                if ep is None or epop is None or any(x is None for x in es) or not eprs: continue
                cs=max(utils.cosine_sim(ep,x) for x in es); cp=max(utils.cosine_sim(ep,x) for x in eprs)
                cadd=utils.cosine_sim(ep,sum(es)-2*epop); cadd_r=utils.cosine_sim(ep,sum(es))
                aw=(cadd>=cs and cadd>=cp); aw_r=(cadd_r>=cs and cadd_r>=cp)
                for t in THRESH:
                    if n>=t:
                        win[t][0 if aw else 1]+=1; winr[t][0 if aw_r else 1]+=1
        print(f"  W{w} done",flush=True)
    print(f"\n=== {model} (d3 additive-share) ===")
    for t in THRESH:
        tot=sum(win[t]); totr=sum(winr[t])
        if tot: print(f"  n>={t:3d} (n={tot:>7,}):  RAW={100*winr[t][0]/totr:.1f}%   e_pop-corrected={100*win[t][0]/tot:.1f}%",flush=True)
