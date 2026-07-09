"""#11 — per-model depth-3 calibration endpoints and collapse index.
Mirrors verification/d3_endpoints.py (flagship) for every d3 model, using each
model's OWN run-to-run noise (stoch repeats) and OWN single-feature biases."""
import os, json, pickle, itertools
import numpy as np
from collections import defaultdict
from shared import BASE, utils

utils.USE_WEIGHTS = True
RNG = np.random.default_rng(7)
MAXT = 20000

MODELS = [  # (tag, stoch_dir, singles waves, observed d3 additive share %)
    ("gpt-4o-mini",               "stoch_test",                       ("26", "43"), 7.8),
    ("claude-haiku-4-5-20251001", "stoch_claude-haiku-4-5-20251001",  ("26", "43"), 8.6),
    ("gemma2_9b",                 "stoch_gemma2_9b",                  ("26", "43"), 9.2),
    ("mistral_latest",            "stoch_mistral_latest",             ("26", "43"), 8.1),
    ("llama3_1_8b_instruct_q4",   "stoch_llama3_1_8b_instruct_q4",    ("26", "43"), 10.0),
    ("gpt-5.5-2026-04-23",        "stoch_gpt-5.5-2026-04-23",         ("26", "34"), 8.9),
    ("claude-sonnet-5",           "stoch_claude-sonnet-5",            ("26", "34"), 8.6),
]


def load_npool(stoch_dir):
    runs = {}
    for i in (1, 2, 3, 4):
        fn = f"{BASE}/data/results/{stoch_dir}/W26_run{i}.jsonl"
        if not os.path.exists(fn):
            continue
        d = {}
        for l in open(fn):
            l = l.strip()
            if not l:
                continue
            try:
                r = json.loads(l)
            except Exception:
                continue
            if r.get("status") != "success":
                continue
            dist = np.array(r["response_distribution"], float)
            if dist.sum() <= 0:
                continue
            d[(tuple(sorted(r["demographics"])), r["question_id"])] = dist / dist.sum()
        if d:
            runs[i] = d
    npool = defaultdict(list)
    for i, j in itertools.combinations(sorted(runs), 2):
        for k, da in runs[i].items():
            db = runs[j].get(k)
            if db is not None and len(db) == len(da):
                npool[len(da)].append((da - db) / np.sqrt(2))
    return {K: np.array(v) for K, v in npool.items() if v}


print(f"{'model':28s} {'ceil_add':>8s} {'floor_col':>9s} {'obs':>5s} {'index_d3':>8s}  (synthetic triples)")
for tag, stoch, waves, obs in MODELS:
    npool = load_npool(stoch)
    if not npool:
        print(f"{tag:28s}  -- no stoch repeats --")
        continue
    utils.MODEL_TAG = tag
    singles = []
    for wave in waves:
        try:
            human, _ = pickle.load(open(f"{BASE}/verification/cache/human_W{wave}_weighted.pkl", "rb"))
            llm, _ = utils.build_llm_index(wave, max_level=1)
        except Exception as e:
            print(f"  {tag} W{wave} skip: {e}")
            continue
        for prof in llm:
            if len(prof) != 1:
                continue
            (dim, val), = prof
            for q, lp in llm[prof].items():
                hp = human.get(prof, {}).get(q)
                if hp is None or len(hp) != len(lp):
                    continue
                K = len(lp)
                if K not in npool:
                    continue
                singles.append((dim, lp - hp, K))
    byK = defaultdict(list)
    for s in singles:
        byK[s[2]].append(s)
    Ks = [K for K in byK if len(byK[K]) >= 6]
    if not Ks:
        print(f"{tag:28s}  -- insufficient singles --")
        continue

    def eta(K):
        arr = npool[K]
        return arr[RNG.integers(0, len(arr))]

    aw = at = cw = ct = 0
    made = 0
    while made < MAXT:
        K = Ks[RNG.integers(0, len(Ks))]
        pool = byK[K]
        idx = RNG.choice(len(pool), size=3, replace=False)
        (dA, eA, _), (dB, eB, _), (dC, eC, _) = (pool[i] for i in idx)
        if len({dA, dB, dC}) != 3:
            continue
        made += 1
        es = [eA, eB, eC]
        for regime in ("add", "col"):
            if regime == "add":
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
            if regime == "add":
                at += 1; aw += win_add
            else:
                ct += 1; cw += win_add
    ceil_add = 100 * aw / at
    floor_col = 100 * cw / ct
    idx3 = (ceil_add - obs) / (ceil_add - floor_col)
    print(f"{tag:28s} {ceil_add:7.1f}% {floor_col:8.1f}% {obs:4.1f}% {idx3:8.2f}  ({made:,})", flush=True)
