"""Referee #8: leave-one-wave-out + wild-cluster bootstrap for the headline
collapse (best-single d2) and additive-share (d3) rates, to check the
wave-cluster CI isn't anticonservative with ~14 clusters."""
import json, numpy as np
from collections import defaultdict
EXCL={"49"}
rows=json.load(open("verification/collapse_results_weighted.json"))
def per_wave(model, depth, pred):
    win=defaultdict(int); tot=defaultdict(int)
    for r in rows:
        if r["model"]!=model or r["depth"]!=depth or r["wave"] in EXCL: continue
        tot[r["wave"]]+=r["wins"]
        if r["predictor"]==pred: win[r["wave"]]+=r["wins"]
    return win,tot
def analyze(model, depth, pred, name):
    win,tot=per_wave(model,depth,pred); ws=[w for w in tot if tot[w]]
    if not ws: return
    pooled=100*sum(win[w] for w in ws)/sum(tot[w] for w in ws)
    # leave-one-wave-out
    lowo=[100*sum(win[w] for w in ws if w!=d)/sum(tot[w] for w in ws if w!=d) for d in ws]
    # per-wave rates
    pw=[100*win[w]/tot[w] for w in ws]
    # wild cluster bootstrap (Rademacher on wave-level deviation of the rate)
    rng=np.random.default_rng(0); n_w=np.array([tot[w] for w in ws]); r_w=np.array([100*win[w]/tot[w] for w in ws])
    base=np.average(r_w,weights=n_w); boots=[]
    for _ in range(10000):
        s=rng.choice([-1,1],len(ws)); rr=base+s*(r_w-base)
        boots.append(np.average(rr,weights=n_w))
    print(f"{name} ({model} d{depth} {pred}):")
    print(f"  pooled={pooled:.1f}%  per-wave [{min(pw):.1f}, {max(pw):.1f}]  ({len(ws)} waves)")
    print(f"  leave-one-wave-out range: [{min(lowo):.2f}, {max(lowo):.2f}]")
    print(f"  wild-cluster 95% CI: [{np.percentile(boots,2.5):.2f}, {np.percentile(boots,97.5):.2f}]")
for m in ["gpt-4o-mini","gemma2_9b","mistral_latest","llama3_1_8b_instruct_q4"]:
    analyze(m,2,"best_single","collapse d2")
analyze("gpt-4o-mini",3,"additive","additive-share d3")
