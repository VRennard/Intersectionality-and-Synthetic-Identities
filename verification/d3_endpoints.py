"""Fable-review M1+M2: (A) depth-3 calibration endpoints — additive-truth and
collapse-truth shares for the d3 contest (additive vs best-single vs best-pair),
mirroring the d2 calibration; (B) d2 additive-truth endpoint under n=10
multinomial histogram noise (the individual-sampling regime).
Flagship (gpt-4o-mini), synthetic cells from measured single biases + run noise.
"""
import os, sys, json, pickle, itertools
import numpy as np
from collections import defaultdict

sys.path.insert(0, 'advanced_bias_analysis'); import utils
utils.USE_WEIGHTS = True
RNG = np.random.default_rng(7)
MODEL = 'gpt-4o-mini'

# ---- eta pool from the 4 repeated runs (as in per_model_endpoints) ----
runs = {}
for i in (1, 2, 3, 4):
    d = {}
    for l in open(f'data/results/stoch_test/W26_run{i}.jsonl'):
        l = l.strip()
        if not l: continue
        try: r = json.loads(l)
        except: continue
        if r.get('status') != 'success': continue
        dist = np.array(r['response_distribution'], float)
        if dist.sum() <= 0: continue
        d[(tuple(sorted(r['demographics'])), r['question_id'])] = dist / dist.sum()
    runs[i] = d
npool = defaultdict(list)
for i, j in itertools.combinations((1, 2, 3, 4), 2):
    for k, da in runs[i].items():
        db = runs[j].get(k)
        if db is not None and len(db) == len(da):
            npool[len(da)].append((da - db) / np.sqrt(2))
for K in npool: npool[K] = np.array(npool[K])

# ---- measured single biases + model single dists on W26/W43 ----
utils.MODEL_TAG = MODEL
singles = []          # list of (dim, e, K, phat)
for wave in ('26', '43'):
    human, _ = pickle.load(open(f'verification/cache/human_W{wave}_weighted.pkl', 'rb'))
    llm, _ = utils.build_llm_index(wave, max_level=1)
    for prof in llm:
        if len(prof) != 1: continue
        (dim, val), = prof
        for q, lp in llm[prof].items():
            hp = human.get(prof, {}).get(q)
            if hp is None or len(hp) != len(lp): continue
            K = len(lp)
            if K not in npool or not len(npool[K]): continue
            singles.append((dim, lp - hp, K, lp))

def eta(K, scale=1.0):
    arr = npool[K]
    return arr[RNG.integers(0, len(arr))] * scale

# =========== (A) depth-3 endpoints ===========
# build synthetic triples: 3 singles, distinct dims, same K
byK = defaultdict(list)
for s in singles: byK[s[2]].append(s)
MAXT = 20000
aw = ct_ = at = cw = 0
made = 0
Ks = [K for K in byK if len(byK[K]) >= 6]
while made < MAXT:
    K = Ks[RNG.integers(0, len(Ks))]
    pool = byK[K]
    idx = RNG.choice(len(pool), size=3, replace=False)
    (dA, eA, _, _), (dB, eB, _, _), (dC, eC, _, _) = (pool[i] for i in idx)
    if len({dA, dB, dC}) != 3: continue
    made += 1
    es = [eA, eB, eC]
    # pair predictors carry their own measurement noise (they are model outputs)
    for regime in ('add', 'col'):
        if regime == 'add':
            target = eA + eB + eC + eta(K)
            pairs = [eA + eB + eta(K), eA + eC + eta(K), eB + eC + eta(K)]
        else:
            dom = max(es, key=np.linalg.norm)
            target = dom + eta(K)
            pairs = []
            for (x, y) in ((eA, eB), (eA, eC), (eB, eC)):
                pdom = x if np.linalg.norm(x) >= np.linalg.norm(y) else y
                pairs.append(pdom + eta(K))
        c_add = utils.cosine_sim(target, eA + eB + eC)
        c_sgl = max(utils.cosine_sim(target, e) for e in es)
        c_pr = max(utils.cosine_sim(target, p) for p in pairs)
        win_add = (c_add >= c_sgl and c_add >= c_pr)
        if regime == 'add': at += 1; aw += win_add
        else: ct_ += 1; cw += win_add

ceil_add = 100 * aw / at      # additive share when truth IS additive
floor_col = 100 * cw / ct_    # additive share when truth is pure collapse
obs = 7.6
idx3 = (ceil_add - obs) / (ceil_add - floor_col)
print(f"[A] depth-3 endpoints ({made:,} synthetic triples):")
print(f"    additive-truth additive-share (ceiling) = {ceil_add:.1f}%")
print(f"    collapse-truth additive-share (floor)   = {floor_col:.1f}%")
print(f"    observed 7.6%  ->  depth-3 collapse index = {idx3:.2f}", flush=True)

# =========== (B) d2 additive-truth endpoint under n=10 histogram noise ===========
MAXP = 25000
aw2 = at2 = cw2 = ct2 = 0
made = 0
while made < MAXP:
    K = Ks[RNG.integers(0, len(Ks))]
    pool = byK[K]
    idx = RNG.choice(len(pool), size=2, replace=False)
    (dA, eA, _, pA), (dB, eB, _, pB) = (pool[i] for i in idx)
    if dA == dB: continue
    made += 1
    # n=10 multinomial histogram noise around the model's own answer
    base = (pA + pB) / 2
    base = np.clip(base, 1e-9, None); base = base / base.sum()
    def hist_noise():
        draw = RNG.multinomial(10, base) / 10.0
        return draw - base
    for regime in ('add', 'col'):
        truth = (eA + eB) if regime == 'add' else (eA if np.linalg.norm(eA) >= np.linalg.norm(eB) else eB)
        target = truth + hist_noise()
        c_sgl = max(utils.cosine_sim(target, eA), utils.cosine_sim(target, eB))
        c_add = utils.cosine_sim(target, eA + eB)
        single = c_sgl >= c_add
        if regime == 'add': at2 += 1; aw2 += single
        else: ct2 += 1; cw2 += single

lo = 100 * aw2 / at2; hi = 100 * cw2 / ct2
print(f"\n[B] d2 endpoints under n=10 histogram noise ({made:,} cells):")
print(f"    additive-truth best-single = {lo:.1f}%   collapse-truth = {hi:.1f}%")
print(f"    observed indiv-sampling ~78%  ->  index = {(78.0 - lo) / (hi - lo):.2f}")
