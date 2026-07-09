"""#7 Kish n_eff sensitivity, #8b clip-activation rate, #9 cross-fitted
recombination, #10 within-family depth growth. One pass over microdata.
Also writes cellstats_W{w}.pkl ({(profile,q): (n_raw, n_eff)}) for t3."""
import os, pickle, itertools
import numpy as np
from collections import defaultdict
from shared import BASE, WAVES15, load_options, load_micro, question_index, val_masks, cells, wdist

rng = np.random.default_rng(42)
HERE = os.path.dirname(os.path.abspath(__file__))

deff = defaultdict(list)                                   # depth -> n/n_eff
l2 = {"n": defaultdict(list), "neff": defaultdict(list)}   # variant -> depth -> l2sq_corr
famcells = defaultdict(list)                               # (dims tuple, depth) -> l2sq_corr (raw-n)
clip_hits = clip_tot = 0                                   # 8b
xfit = {"cross": [], "insample": []}                       # 9

for wave in WAVES15:
    options = load_options(wave)
    df, w = load_micro(wave)
    qidx = question_index(df, options)
    vm = val_masks(df)
    all_mask = np.ones(len(df), bool)

    pop = {}
    for q, (idx, K) in qidx.items():
        d, n, ne = wdist(idx, K, all_mask, w)
        if d is not None:
            pop[q] = (d, n, ne)

    half = rng.random(len(df)) < 0.5
    h1, h2 = half, ~half

    singles_full, singles_h1 = {}, {}
    cellstats = {}

    for depth, prof, mask in cells(vm):
        dims = tuple(sorted(dv[0] for dv in prof))
        if depth == 1:
            (dv,) = prof
            singles_full[dv], singles_h1[dv] = {}, {}
        for q, (idx, K) in qidx.items():
            pq = pop.get(q)
            if pq is None:
                continue
            d_full, n, ne = wdist(idx, K, mask, w)
            if d_full is None:
                continue
            cellstats[(prof, q)] = (n, ne)
            deff[depth].append(n / ne if ne > 0 else np.nan)
            ppop, npop, nepop = pq
            diff = d_full - ppop
            l2sq = float(diff @ diff)
            for variant, ng, npp in (("n", n, npop), ("neff", ne, nepop)):
                nv = float((d_full * (1 - d_full)).sum() / max(ng, 1e-9)
                           + (ppop * (1 - ppop)).sum() / max(npp, 1e-9))
                l2[variant][depth].append(l2sq - nv)
            famcells[(dims, depth)].append(l2["n"][depth][-1])

            if depth == 1:
                singles_full[dv][q] = d_full
                dh1, _, _ = wdist(idx, K, mask & h1, w)
                if dh1 is not None:
                    singles_h1[dv][q] = dh1
            elif depth == 2:
                a, b = sorted(prof)
                pA = singles_full.get(a, {}).get(q)
                pB = singles_full.get(b, {}).get(q)
                if pA is None or pB is None:
                    continue
                add = pA + pB - ppop
                if n >= 100:
                    clip_tot += 1
                    clip_hits += bool((add < -1e-12).any())
                rec = np.clip(add, 0, None)
                if rec.sum() > 0:
                    xfit["insample"].append(0.5 * np.abs(rec / rec.sum() - d_full).sum())
                pA1 = singles_h1.get(a, {}).get(q)
                pB1 = singles_h1.get(b, {}).get(q)
                pop1, _, _ = wdist(idx, K, h1, w)
                pAB2, _, _ = wdist(idx, K, mask & h2, w)
                if pA1 is None or pB1 is None or pop1 is None or pAB2 is None:
                    continue
                rec = np.clip(pA1 + pB1 - pop1, 0, None)
                if rec.sum() > 0:
                    xfit["cross"].append(0.5 * np.abs(rec / rec.sum() - pAB2).sum())

    with open(os.path.join(HERE, f"cellstats_W{wave}.pkl"), "wb") as f:
        pickle.dump(cellstats, f)
    print(f"W{wave} done: {len(cellstats):,} (cell,q) stats", flush=True)

# ================= reports =================
print("\n#7 Kish design effect (n / n_eff), by depth:")
for d in (1, 2, 3):
    arr = np.array(deff[d]); arr = arr[np.isfinite(arr)]
    print(f"  depth {d}: mean deff {arr.mean():.3f}  median {np.median(arr):.3f}  p90 {np.percentile(arr,90):.3f}  ({len(arr):,} cell-questions)")
print("  (analytic floors scale by sqrt(deff))")

print("\n#7 noise-corrected squared distinctiveness by depth (raw n vs Kish n_eff):")
for variant in ("n", "neff"):
    m = {d: np.mean(l2[variant][d]) for d in (1, 2, 3)}
    print(f"  {variant:4s}: d1 {m[1]:.4f}  d2 {m[2]:.4f}  d3 {m[3]:.4f}   growth d3/d1 = {m[3]/m[1]:.2f}x")

print("\n#10 within-family depth growth (dimension-triple families with d3 cells):")
dims_all = sorted({dims[0] for (dims, dep) in famcells if dep == 1})
ratios = []
for trip in itertools.combinations(dims_all, 3):
    d3v = famcells.get((tuple(sorted(trip)), 3), [])
    d1v = sum((famcells.get(((d,), 1), []) for d in trip), [])
    d2v = sum((famcells.get((tuple(sorted(p)), 2), []) for p in itertools.combinations(trip, 2)), [])
    if d3v and d1v and d2v:
        r = np.mean(d3v) / np.mean(d1v)
        ratios.append(r)
        print(f"  {'+'.join(sorted(trip)):40s} d1 {np.mean(d1v):.4f} d2 {np.mean(d2v):.4f} d3 {np.mean(d3v):.4f}  d3/d1 {r:.2f}x")
print(f"  -> {len(ratios)} families, median growth {np.median(ratios):.2f}x, range {min(ratios):.2f}-{max(ratios):.2f}x")

print(f"\n#8b clipping activation (additive-truth pair cells, per-question n>=100):")
print(f"  {clip_hits:,}/{clip_tot:,} = {100*clip_hits/max(clip_tot,1):.1f}% of cells have any negative component before clipping")

print("\n#9 additive recombination from true singles (pair cells):")
for k in ("insample", "cross"):
    arr = np.array(xfit[k])
    print(f"  {k:9s}: mean TV {arr.mean():.3f}  median {np.median(arr):.3f}  ({len(arr):,} cells)")
print("  (cross-fitted halves carry ~2x the sampling variance; agreement within that bound")
print("   means shared-sample noise, not composition, is not driving the low error)")
