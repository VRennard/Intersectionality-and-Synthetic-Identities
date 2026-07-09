"""#2 small-norm strata, #17 retention by pair + value level, #21b residualized
null, #20 entropy robustness. One flagship (gpt-4o-mini) pass over 15 waves.
Requires cellstats_W*.pkl from t2_micro_batch.py."""
import os, pickle, itertools
import numpy as np
from collections import defaultdict
from shared import BASE, WAVES15, utils, bootstrap_ci, load_options

utils.USE_WEIGHTS = True
utils.MODEL_TAG = "gpt-4o-mini"
HERE = os.path.dirname(os.path.abspath(__file__))
rng = np.random.default_rng(9)
AVG = utils.AVG_PROFILE

norms, wins = [], []                       # 2: ||e_AB||, single-won
zero_guard = 0                             # 2: cells hitting the cosine zero-guard
pair_stats = defaultdict(lambda: [0, 0])   # 17: (dimA,dimB) -> [contested, single-wins]
keep_dim = defaultdict(lambda: defaultdict(int))   # 17: dim -> kept count; and dom count
dom_dim = defaultdict(lambda: defaultdict(int))
keep_val = defaultdict(lambda: [0, 0])     # 17: (dim,val) -> [kept, contested-with-it]
dom_val = defaultdict(lambda: [0, 0])
null_raw, null_res = [], []                # 21b
ent = defaultdict(lambda: defaultdict(list))  # 20: variant -> (wave,depth) -> dH

for wave in WAVES15:
    human, _ = pickle.load(open(f"{BASE}/verification/cache/human_W{wave}_weighted.pkl", "rb"))
    qopts = load_options(wave)
    llm, _ = utils.build_llm_index(wave, max_level=3)
    cellstats = pickle.load(open(os.path.join(HERE, f"cellstats_W{wave}.pkl"), "rb"))
    hpop = human.get(AVG, {})
    lpop = llm.get(AVG, {})

    # singles pool per question for the null (dim -> list of (dim, e))
    singles_by_q = defaultdict(list)
    for prof in llm:
        if len(prof) != 1:
            continue
        (dim, val), = prof
        for q, lp in llm[prof].items():
            hp = human.get(prof, {}).get(q)
            if hp is not None and len(hp) == len(lp):
                singles_by_q[q].append((dim, lp - hp))

    for prof in llm:
        d = len(prof)
        # ---------- entropy (#20), depths 1-3 ----------
        for q, lp in llm[prof].items():
            hp = human.get(prof, {}).get(q)
            st = cellstats.get((prof, q))
            if hp is None or st is None or len(hp) != len(lp):
                continue
            n, neff = st
            K = len(lp)
            def H(p):
                p = p[p > 0]
                return float(-(p * np.log(p)).sum())
            dH = H(lp) - (H(hp) + (K - 1) / (2 * n))
            key = (wave, d)
            ent["raw"][key].append(dH)
            ent["norm"][key].append(dH / np.log(K) if K > 1 else np.nan)
            ent["herf"][key].append(float((hp @ hp) - (lp @ lp)))  # impurity diff (llm more dispersed -> positive)
            if n >= 100:
                ent["n100"][key].append(dH)
            opts = qopts.get(q)
            if opts and len(opts) == K:
                keep = [i for i, o in enumerate(opts) if "refus" not in str(o).lower()]
                if 1 < len(keep) < K:
                    hp2, lp2 = hp[keep], lp[keep]
                    if hp2.sum() > 0 and lp2.sum() > 0:
                        hp2, lp2 = hp2 / hp2.sum(), lp2 / lp2.sum()
                        dH2 = H(lp2) - (H(hp2) + (len(keep) - 1) / (2 * n))
                        ent["norefuse"][key].append(dH2)
                elif len(keep) == K:
                    ent["norefuse"][key].append(dH)
        if d != 2:
            continue
        # ---------- pair contest (#2, #17, #21b) ----------
        a, b = sorted(prof)
        pa, pb = frozenset([a]), frozenset([b])
        for q, lab in llm[prof].items():
            hab = human.get(prof, {}).get(q)
            la, ha = llm.get(pa, {}).get(q), human.get(pa, {}).get(q)
            lb, hb = llm.get(pb, {}).get(q), human.get(pb, {}).get(q)
            if any(x is None for x in (hab, la, ha, lb, hb)):
                continue
            if len({len(lab), len(hab), len(la), len(ha), len(lb), len(hb)}) != 1:
                continue
            eAB, eA, eB = lab - hab, la - ha, lb - hb
            nAB = np.linalg.norm(eAB)
            if nAB < 1e-10 or np.linalg.norm(eA) < 1e-10 or np.linalg.norm(eB) < 1e-10:
                zero_guard += 1
            cA, cB = utils.cosine_sim(eAB, eA), utils.cosine_sim(eAB, eB)
            c_sgl = max(cA, cB)
            c_add = utils.cosine_sim(eAB, eA + eB)
            single_won = c_sgl >= c_add
            norms.append(nAB); wins.append(single_won)
            dA, dB = a[0], b[0]
            key = tuple(sorted((dA, dB)))
            pair_stats[key][0] += 1
            pair_stats[key][1] += single_won
            if single_won:
                # winner value/dim; human-dominant by TV to population
                win = a if cA >= cB else b
                hp_pop = hpop.get(q)
                keep_dim[key][win[0]] += 1
                keep_val[win][0] += 1
                keep_val[a][1] += 1; keep_val[b][1] += 1
                if hp_pop is not None and len(hp_pop) == len(ha):
                    tvA = 0.5 * np.abs(ha - hp_pop).sum()
                    tvB = 0.5 * np.abs(hb - hp_pop).sum()
                    domf = a if tvA >= tvB else b
                    dom_dim[key][domf[0]] += 1
                    dom_val[domf][0] += 1
                    dom_val[a][1] += 1; dom_val[b][1] += 1
            # null (#21b): random single from an unrelated dimension, same question
            pool = [e for (dm, e) in singles_by_q.get(q, []) if dm not in (dA, dB) and len(e) == len(eAB)]
            if pool:
                eC = pool[rng.integers(0, len(pool))]
                null_raw.append(utils.cosine_sim(eAB, eC))
                ep_l, ep_h = lpop.get(q), hpop.get(q)
                if ep_l is not None and ep_h is not None and len(ep_l) == len(eAB):
                    epop = ep_l - ep_h
                    null_res.append(utils.cosine_sim(eAB - epop, eC - epop))
    print(f"W{wave} pass done ({len(norms):,} pair contests so far)", flush=True)

norms = np.array(norms); wins = np.array(wins, bool)

print("\n#2 contest by ||e_AB|| strata (flagship, 15 waves):")
print(f"  cosine zero-guard (<1e-10 norm) triggered in {zero_guard:,}/{len(norms):,} contests ({100*zero_guard/len(norms):.3f}%)")
qs = np.percentile(norms, np.arange(0, 101, 10))
for i in range(10):
    m = (norms >= qs[i]) & (norms <= qs[i + 1] if i == 9 else norms < qs[i + 1])
    print(f"  decile {i+1:2d} (||e_AB|| {qs[i]:.3f}-{qs[i+1]:.3f}): best-single wins {100*wins[m].mean():.1f}%  (n={m.sum():,})")
for thr in (0.01, 0.02, 0.05):
    m = norms >= thr
    print(f"  excluding ||e_AB|| < {thr}: {100*wins[m].mean():.1f}% ({100*(1-m.mean()):.1f}% of cells excluded)  [all cells: {100*wins.mean():.1f}%]")

print("\n#17 collapse rate and retention by dimension pair (flagship):")
print(f"  {'pair':28s} {'contested':>10s} {'collapsed':>9s}  keep split (dim: kept% | human-dom%)")
for key in sorted(pair_stats):
    tot, sw = pair_stats[key]
    parts = []
    for dm in key:
        k = keep_dim[key].get(dm, 0)
        dm_dom = dom_dim[key].get(dm, 0)
        dtot = sum(dom_dim[key].values()) or 1
        parts.append(f"{dm} {100*k/max(sw,1):.0f}%|{100*dm_dom/dtot:.0f}%")
    print(f"  {'+'.join(key):28s} {tot:>10,} {100*sw/tot:>8.1f}%  {'  '.join(parts)}")

print("\n#17 value-level keep-rate minus dom-rate (Race and Religion values):")
for (dm, val), (k, c) in sorted(keep_val.items()):
    if dm not in ("Race", "Religion") or c < 2000:
        continue
    dk, dc = dom_val.get((dm, val), (0, 1))
    delta = 100 * k / c - 100 * dk / max(dc, 1)
    print(f"  {dm:9s} {val:28s} kept {100*k/c:5.1f}%  dom {100*dk/max(dc,1):5.1f}%  delta {delta:+.1f} pts  (contested {c:,})")

print("\n#21b random-unrelated null, raw vs population-bias-residualized (flagship):")
print(f"  raw:          median cos {np.median(null_raw):.3f}  (n={len(null_raw):,})")
print(f"  residualized: median cos {np.median(null_res):.3f}  (n={len(null_res):,})")

print("\n#20 entropy robustness, dH = H(model) - H_MM(human) by depth (wave-cluster 95% CI):")
for variant, label in (("raw", "dH (nats)"), ("n100", "dH, cells n>=100"), ("norm", "dH / log K"), ("herf", "Herfindahl impurity diff"), ("norefuse", "dH, refusal-excluded")):
    for d in (1, 2, 3):
        per_wave = [np.mean(ent[variant][(w, d)]) for w in WAVES15 if ent[variant].get((w, d))]
        if not per_wave:
            continue
        lo, hi = bootstrap_ci(per_wave, B=5000)
        print(f"  {label:26s} d{d}: {np.mean(per_wave):+.3f} [{lo:+.3f}, {hi:+.3f}]  ({len(per_wave)} waves)")
