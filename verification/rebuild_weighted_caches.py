"""
Build survey-weighted human ground-truth caches (weighted-primary switch).

For every wave: load qmeta from gpt-4o-mini's results (covers the full
question set on all 15 waves), build the human intersectional index up to
triples with utils.USE_WEIGHTS=True, and pickle to
verification/cache/human_W{wave}_weighted.pkl in the same (human, qids)
format as the unweighted caches.

Cell inclusion is unchanged (n >= 20 raw respondents per cell), so weighted
and unweighted indices cover identical profiles; only the distributions move.
"""

import os, sys, pickle, time

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(BASE, "advanced_bias_analysis"))
import utils

CACHE_DIR = os.path.join(BASE, "verification", "cache")
QMETA_MODEL = "gpt-4o-mini"


def wave_list():
    d = os.path.join(BASE, "data", "results", QMETA_MODEL)
    return sorted((fn[1:-6] for fn in os.listdir(d)
                   if fn.startswith("W") and fn.endswith(".jsonl")), key=int)


def main():
    utils.MODEL_TAG = QMETA_MODEL
    utils.USE_WEIGHTS = True
    for wave in wave_list():
        out = os.path.join(CACHE_DIR, f"human_W{wave}_weighted.pkl")
        if os.path.exists(out):
            print(f"W{wave}: cache exists, skipping", flush=True)
            continue
        t0 = time.time()
        _, qmeta = utils.build_llm_index(wave, max_level=3)
        human = utils.load_human_index(wave, list(qmeta.keys()), qmeta, max_level=3)
        with open(out, "wb") as f:
            pickle.dump((human, set(qmeta.keys())), f)
        n_cells = sum(len(qd) for qd in human.values())
        print(f"W{wave}: {len(human)} profiles, {n_cells} cells, "
              f"{time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
