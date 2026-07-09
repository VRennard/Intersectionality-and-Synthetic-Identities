"""#18b — semantic-compliance diagnostic for the base-vs-tuned contrast:
per-dimension single-feature (i) bias magnitude TV(model, human) and
(ii) within-dimension differentiation (mean pairwise TV between the dimension's
value outputs), under conditions A (base weights, base prompt), B (instruct
weights, base prompt), C (instruct weights, chat template). Shared cells only.
Human differentiation from the weighted caches as reference."""
import pickle, itertools
import numpy as np
from collections import defaultdict
from shared import BASE, WAVES15, utils

utils.USE_WEIGHTS = True
CONDS = [("A base-ckpt/base-prompt", "llama3_1_8b_text"),
         ("B instr-ckpt/base-prompt", "llama3_1_8b_instruct_base"),
         ("C instr-ckpt/chat-tmpl",  "llama3_1_8b_instruct_q4")]

bias = defaultdict(lambda: defaultdict(list))   # cond -> dim -> TV
diff = defaultdict(lambda: defaultdict(list))   # cond+human -> dim -> pairwise TV
waves_used = []

for wave in WAVES15:
    human, _ = pickle.load(open(f"{BASE}/verification/cache/human_W{wave}_weighted.pkl", "rb"))
    idxs = {}
    ok = True
    for label, tag in CONDS:
        utils.MODEL_TAG = tag
        try:
            llm, _ = utils.build_llm_index(wave, max_level=1)
        except Exception:
            ok = False
            break
        if not llm:
            ok = False
            break
        idxs[label] = llm
    if not ok:
        print(f"W{wave}: skipped (missing condition coverage)", flush=True)
        continue
    waves_used.append(wave)

    # shared (prof, q) across all three conditions + human
    shared = None
    for label, _ in CONDS:
        keys = {(p, q) for p, qs in idxs[label].items() if len(p) == 1 for q in qs}
        shared = keys if shared is None else shared & keys
    shared = {(p, q) for (p, q) in shared
              if human.get(p, {}).get(q) is not None
              and len(human[p][q]) == len(idxs[CONDS[0][0]][p][q])}

    for (prof, q) in shared:
        (dim, val), = prof
        hp = human[prof][q]
        for label, _ in CONDS:
            lp = idxs[label][prof][q]
            if len(lp) == len(hp):
                bias[label][dim].append(0.5 * np.abs(lp - hp).sum())

    # within-dimension differentiation on shared value sets per (dim, q)
    byq = defaultdict(list)
    for (prof, q) in shared:
        (dim, val), = prof
        byq[(dim, q)].append(prof)
    for (dim, q), profs in byq.items():
        if len(profs) < 2:
            continue
        for label, _ in CONDS:
            tvs = [0.5 * np.abs(idxs[label][p1][q] - idxs[label][p2][q]).sum()
                   for p1, p2 in itertools.combinations(profs, 2)]
            diff[label][dim].append(np.mean(tvs))
        tvs = [0.5 * np.abs(human[p1][q] - human[p2][q]).sum()
               for p1, p2 in itertools.combinations(profs, 2)]
        diff["human"][dim].append(np.mean(tvs))
    print(f"W{wave}: {len(shared):,} shared single cells", flush=True)

dims = sorted({d for c in bias.values() for d in c})
print(f"\n#18b per-dimension single-feature BIAS magnitude, mean TV(model, human) — {len(waves_used)} waves, shared cells:")
print(f"  {'dimension':16s} " + " ".join(f"{lbl:>26s}" for lbl, _ in CONDS))
for d in dims:
    print(f"  {d:16s} " + " ".join(f"{np.mean(bias[lbl][d]):26.3f}" for lbl, _ in CONDS))

print(f"\n#18b within-dimension DIFFERENTIATION, mean pairwise TV between value outputs:")
print(f"  {'dimension':16s} " + " ".join(f"{lbl:>26s}" for lbl, _ in CONDS) + f" {'human':>8s}")
for d in dims:
    row = " ".join(f"{np.mean(diff[lbl][d]):26.3f}" for lbl, _ in CONDS)
    print(f"  {d:16s} {row} {np.mean(diff['human'][d]):8.3f}")
print("\n(base conditions produce no population cell, so steering is probed via")
print(" differentiation between a dimension's own value outputs; a base checkpoint")
print(" that ignored demographics would show near-zero differentiation)")
