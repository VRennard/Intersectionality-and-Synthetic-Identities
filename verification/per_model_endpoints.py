"""Referee #4: per-model calibration endpoints.

For each model we estimate its OWN run-to-run noise eta (from W26 stochastic
repeats), then run the synthetic additive-truth / collapse-truth contest using
that model's OWN single-feature bias vectors. The collapse index is
    index = (obs - floor) / (ceil - floor)
with floor = best-single win-rate when the truth is additive (e_A+e_B) and
ceil = best-single win-rate when the truth is pure collapse (dominant single),
both evaluated at the model's native noise. obs = published raw best-single rate.
"""
import os, sys, json, pickle, itertools
import numpy as np
from collections import defaultdict

sys.path.insert(0, 'advanced_bias_analysis'); import utils
utils.USE_WEIGHTS = True
RNG = np.random.default_rng(3)

# (MODEL_TAG, stoch_dir, observed raw best-single depth-2 %)
MODELS = [
    ("gpt-4o-mini",                "stoch_test",                        78.4),
    ("gemma2_9b",                  "stoch_gemma2_9b",                   75.3),
    ("mistral_latest",             "stoch_mistral_latest",              81.2),
    ("claude-haiku-4-5-20251001",  "stoch_claude-haiku-4-5-20251001",   77.9),
    ("gpt-4o",                     "stoch_gpt-4o",                      77.6),
    ("llama3_1_8b_instruct_q4",    "stoch_llama3_1_8b_instruct_q4",     80.3),
    ("claude-sonnet-5",            "stoch_claude-sonnet-5",             77.8),
    ("gpt-5.5-2026-04-23",         "stoch_gpt-5.5-2026-04-23",          79.1),
]
WAVES = ('26', '43')
MAXCELLS = 25000


def load_npool(stoch_dir):
    runs = {}
    for i in (1, 2, 3, 4):
        fn = f'data/results/{stoch_dir}/W26_run{i}.jsonl'
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
            if r.get('status') != 'success':
                continue
            dist = np.array(r['response_distribution'], float)
            if dist.size == 0 or dist.sum() <= 0:
                continue
            d[(tuple(sorted(r['demographics'])), r['question_id'])] = dist / dist.sum()
        if d:
            runs[i] = d
    npool = defaultdict(list)
    for i, j in itertools.combinations(sorted(runs), 2):
        for k, da in runs[i].items():
            db = runs[j].get(k)
            if db is not None and len(db) == len(da):
                npool[len(da)].append((da - db) / np.sqrt(2))
    for K in list(npool):
        npool[K] = np.array(npool[K])
    return npool, len(runs)


def build_cells(tag, npool):
    utils.MODEL_TAG = tag
    cells = []
    for wave in WAVES:
        try:
            human, _ = pickle.load(open(f'verification/cache/human_W{wave}_weighted.pkl', 'rb'))
            llm, _ = utils.build_llm_index(wave, max_level=2)
        except Exception as e:
            print(f"    [skip W{wave}: {e}]"); continue

        def bias(p, q):
            lp = llm.get(p, {}).get(q); hp = human.get(p, {}).get(q)
            if lp is None or hp is None or len(lp) != len(hp):
                return None
            return lp - hp
        nd = 0
        for prof in llm:
            if len(prof) != 2 or nd >= MAXCELLS:
                continue
            (a, va), (b, vb) = sorted(prof)
            pa, pb = frozenset([(a, va)]), frozenset([(b, vb)])
            for q in llm[prof]:
                ea, eb = bias(pa, q), bias(pb, q)
                if ea is None or eb is None:
                    continue
                K = len(ea)
                if K not in npool or not len(npool[K]):
                    continue
                cells.append((ea, eb, K)); nd += 1
    return cells


def endpoints(cells, npool, scale=1.0):
    aw = cw = at = ct = 0
    for ea, eb, K in cells:
        eta = npool[K][RNG.integers(0, len(npool[K]))] * scale
        edom = ea if np.linalg.norm(ea) >= np.linalg.norm(eb) else eb
        for reg, etrue in (('add', ea + eb), ('col', edom)):
            es = etrue + eta
            csgl = max(utils.cosine_sim(es, ea), utils.cosine_sim(es, eb))
            cadd = utils.cosine_sim(es, ea + eb)
            single = csgl >= cadd
            if reg == 'add':
                aw += single; at += 1
            else:
                cw += single; ct += 1
    return 100 * aw / at, 100 * cw / ct


print(f"{'model':28s} {'runs':>4s} {'eta(med TV)':>11s} {'floor':>7s} {'ceil':>7s} {'obs':>6s} {'index':>6s}")
for tag, sdir, obs in MODELS:
    npool, nruns = load_npool(sdir)
    if not npool:
        print(f"{tag:28s}  no usable stoch runs"); continue
    # median run-to-run TV as a scalar summary of eta
    tvs = []
    for K, arr in npool.items():
        tvs.extend(0.5 * np.abs(arr).sum(axis=1) * np.sqrt(2))  # back to raw |da-db| TV
    eta_tv = float(np.median(tvs))
    cells = build_cells(tag, npool)
    if not cells:
        print(f"{tag:28s}  no cells"); continue
    floor, ceil = endpoints(cells, npool)
    idx = (obs - floor) / (ceil - floor)
    print(f"{tag:28s} {nruns:>4d} {eta_tv:>11.3f} {floor:>6.1f}% {ceil:>6.1f}% {obs:>5.1f}% {idx:>6.2f}", flush=True)
