"""#12 — matched-subset depth-3 comparison: every d3 model evaluated on the
same restricted triple set (Age/Gender/Race/Income dims) and the same waves
(W26+W34, the frontier coverage), plus a strict common-cell intersection."""
import pickle, itertools
import numpy as np
from shared import BASE, utils

utils.USE_WEIGHTS = True
DIMS4 = {"Age", "Gender", "Race", "Income"}
WAVES = ("26", "34")
MODELS = ["gpt-4o-mini", "claude-haiku-4-5-20251001", "gemma2_9b",
          "mistral_latest", "llama3_1_8b_instruct_q4",
          "gpt-5.5-2026-04-23", "claude-sonnet-5"]

human_cache = {w: pickle.load(open(f"{BASE}/verification/cache/human_W{w}_weighted.pkl", "rb"))[0]
               for w in WAVES}

results = {}   # model -> {(wave, prof, q): additive_won}
for tag in MODELS:
    utils.MODEL_TAG = tag
    cells = {}
    for w in WAVES:
        human = human_cache[w]
        try:
            llm, _ = utils.build_llm_index(w, max_level=3)
        except Exception as e:
            print(f"{tag} W{w} skip: {e}")
            continue
        for prof in llm:
            if len(prof) != 3:
                continue
            dims = {dv[0] for dv in prof}
            if not dims <= DIMS4:
                continue
            feats = sorted(prof)
            singles = [frozenset([f]) for f in feats]
            pairs = [frozenset(p) for p in itertools.combinations(feats, 2)]
            for q, lab in llm[prof].items():
                hab = human.get(prof, {}).get(q)
                if hab is None or len(hab) != len(lab):
                    continue
                es, ok = [], True
                for s_ in singles:
                    lp, hp = llm.get(s_, {}).get(q), human.get(s_, {}).get(q)
                    if lp is None or hp is None or len(lp) != len(lab):
                        ok = False; break
                    es.append(lp - hp)
                if not ok:
                    continue
                eprs = []
                for p_ in pairs:
                    lp, hp = llm.get(p_, {}).get(q), human.get(p_, {}).get(q)
                    if lp is None or hp is None or len(lp) != len(lab):
                        ok = False; break
                    eprs.append(lp - hp)
                if not ok:
                    continue
                eABC = lab - hab
                c_add = utils.cosine_sim(eABC, es[0] + es[1] + es[2])
                c_sgl = max(utils.cosine_sim(eABC, e) for e in es)
                c_pr = max(utils.cosine_sim(eABC, e) for e in eprs)
                cells[(w, prof, q)] = bool(c_add >= c_sgl and c_add >= c_pr)
    results[tag] = cells
    share = 100 * np.mean(list(cells.values())) if cells else float("nan")
    print(f"{tag:28s} matched-dims share {share:5.1f}%  (n={len(cells):,})", flush=True)

common = None
for tag, cells in results.items():
    keys = set(cells)
    common = keys if common is None else (common & keys)
print(f"\nstrict common-cell set across all {len(MODELS)} models: {len(common):,} cells")
print(f"{'model':28s} {'common-cell additive share':>27s}")
for tag in MODELS:
    cells = results[tag]
    vals = [cells[k] for k in common]
    print(f"{tag:28s} {100*np.mean(vals):26.1f}%")
