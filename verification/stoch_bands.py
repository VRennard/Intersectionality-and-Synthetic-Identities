"""
Stochasticity error bands for the spine (Fig 8).

Recomputes the spine metric — mean TV(LLM, human) per depth — separately for
each of the 4 repeat gpt-4o-mini runs of W26 (temp 0.7, identical prompts,
data/results/stoch_test/W26_run{1..4}.jsonl). The across-run spread is the
run-level noise band to overlay on the spine figure.

Reuses the cached human index (verification/cache/human_W26.pkl) built by
verify_spine_collapse.py; stoch records carry no question options, so the
cache is required.

Output: verification/stoch_bands.json + console summary.
"""

import os, sys, json, pickle
from collections import defaultdict

import numpy as np

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(BASE, "advanced_bias_analysis"))
import utils

OUT_DIR   = os.path.join(BASE, "verification")
CACHE_DIR = os.path.join(OUT_DIR, "cache")
STOCH_DIR = os.path.join(BASE, "data", "results", "stoch_test")
WAVE      = "26"
RUNS      = [1, 2, 3, 4]


def load_run_index(run):
    """{profile_frozenset: {qid: normalized np.array}} for one stoch run."""
    llm = {}
    path = os.path.join(STOCH_DIR, f"W{WAVE}_run{run}.jsonl")
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            if r.get("status") != "success":
                continue
            dist = np.array(r.get("response_distribution", []), dtype=float)
            if dist.sum() == 0:
                continue
            profile = utils.parse_demo(r.get("demographics", []))
            if profile is None or len(profile) > 3:
                continue
            if any(v in utils.IGNORE_VALUES.get(d, set()) for d, v in profile):
                continue
            llm.setdefault(profile, {})[r["question_id"]] = dist / dist.sum()
    return llm


def main():
    with open(os.path.join(CACHE_DIR, f"human_W{WAVE}.pkl"), "rb") as f:
        human, _ = pickle.load(f)

    rows = []
    for run in RUNS:
        llm = load_run_index(run)
        tv_by_depth = defaultdict(list)
        for profile, qdists in llm.items():
            hq = human.get(profile)
            if not hq:
                continue
            for qid, ld in qdists.items():
                hd = hq.get(qid)
                if hd is None or len(hd) != len(ld):
                    continue
                tv_by_depth[len(profile)].append(utils.tv(ld, hd))
        for depth, vals in sorted(tv_by_depth.items()):
            rows.append({"run": run, "depth": depth,
                         "n_cells": len(vals), "mean_tv": float(np.mean(vals))})
        print(f"run {run}: " + ", ".join(f"d{d}={np.mean(v):.4f}(n={len(v)})"
                                         for d, v in sorted(tv_by_depth.items())),
              flush=True)

    with open(os.path.join(OUT_DIR, "stoch_bands.json"), "w") as f:
        json.dump(rows, f, indent=1)

    print("\n=== ACROSS-RUN SPREAD OF SPINE MEAN (W26, gpt-4o-mini, temp 0.7) ===")
    for depth in (1, 2, 3):
        ms = [r["mean_tv"] for r in rows if r["depth"] == depth]
        print(f"  depth {depth}: mean={np.mean(ms):.4f}  std={np.std(ms, ddof=1):.5f}  "
              f"range=[{min(ms):.4f}, {max(ms):.4f}]  ({len(ms)} runs)")


if __name__ == "__main__":
    main()
