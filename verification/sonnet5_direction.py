"""Direction stats for claude-sonnet-5 (W26+W34): per-dimension keep-rate minus
human-dominance rate, on collapsed pair cells. Compare with GPT-5.5 and the
prior-gen matched-wave numbers (race -7.4, religion -3.2, gender +2.2)."""
import pickle, sys
from collections import defaultdict
sys.path.insert(0, 'advanced_bias_analysis'); import utils
utils.USE_WEIGHTS = True
def tv(a,b): return 0.5*abs(a-b).sum()
for TAG in ("claude-sonnet-5", "gpt-5.5-2026-04-23"):
    utils.MODEL_TAG = TAG
    keep=defaultdict(int); dom=defaultdict(int); n=defaultdict(int)
    for w in ("26","34"):
        human,_ = pickle.load(open(f'verification/cache/human_W{w}_weighted.pkl','rb'))
        llm,_ = utils.build_llm_index(w, max_level=2)
        for prof in llm:
            if len(prof)!=2: continue
            (a,va),(b,vb)=sorted(prof)
            pa,pb=frozenset([(a,va)]),frozenset([(b,vb)])
            for q,lab in llm[prof].items():
                hab=human.get(prof,{}).get(q); la=llm.get(pa,{}).get(q)
                ha=human.get(pa,{}).get(q); lb=llm.get(pb,{}).get(q); hb=human.get(pb,{}).get(q)
                if any(x is None for x in (hab,la,ha,lb,hb)): continue
                if len({len(lab),len(hab),len(la),len(ha),len(lb),len(hb)})!=1: continue
                eAB=lab-hab; eA=la-ha; eB=lb-hb
                ca=utils.cosine_sim(eAB,eA); cb=utils.cosine_sim(eAB,eB)
                if max(ca,cb) < utils.cosine_sim(eAB,eA+eB): continue   # collapsed cells only
                winner = a if ca>=cb else b
                hdom = a if tv(ha- human.get(utils.AVG_PROFILE,{}).get(q, ha),0*ha) >= 0 else a  # placeholder
                # human-dominant: larger TV(subgroup, pop)
                hp = human.get(utils.AVG_PROFILE,{}).get(q)
                if hp is None: continue
                hdom = a if tv(ha,hp) >= tv(hb,hp) else b
                for d in (a,b): n[d]+=1
                keep[winner]+=1; dom[hdom]+=1
    print(f"\n{TAG} (W26+W34, collapsed pair cells):")
    for d in sorted(n, key=lambda d: -(keep[d]/n[d]-dom[d]/n[d])):
        k=100*keep[d]/n[d]; h=100*dom[d]/n[d]
        print(f"  {d:16s} keep={k:5.1f}%  human-dom={h:5.1f}%  delta={k-h:+5.1f}pp   (n={n[d]:,})")
