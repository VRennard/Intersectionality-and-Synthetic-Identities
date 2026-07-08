"""
Survey-weight sensitivity check.

All human ground-truth distributions in the pipeline are UNWEIGHTED counts;
Pew's official estimates use survey weights (WEIGHT_W{wave}). This script
quantifies the difference: for each wave, the TV distance between weighted
and unweighted distributions, for the full population and for single/pair
subgroup cells (n >= 20).

If the weighted-unweighted TV is small relative to LLM error (~0.17-0.33),
the unweighted analysis stands with a sensitivity note; otherwise the
pipeline must be re-run weighted.
"""

import os, sys, json, itertools
from collections import defaultdict

import numpy as np
import pandas as pd

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(BASE, "advanced_bias_analysis"))
import utils

EXCL = {"49"}


def wave_list():
    d = os.path.join(BASE, "human_resp")
    out = []
    for fn in sorted(os.listdir(d)):
        if fn.startswith("American_Trends_Panel_W"):
            w = fn.rsplit("W", 1)[1]
            if w not in EXCL and os.path.exists(os.path.join(d, fn, "responses.csv")):
                out.append(w)
    return sorted(out, key=int)


def load_options(wave):
    path = os.path.join(BASE, "data", "responses", f"survey_responses_W{wave}.json")
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return {q["question_id"]: [r["option"] for r in q["responses"]] for q in data}


def dists(sub, wcol, qid, opts):
    """(unweighted, weighted) normalized distributions or (None, None)."""
    s = sub[[qid, wcol]].dropna()
    if len(s) < utils.MIN_HUMAN_N:
        return None, None
    cu = np.zeros(len(opts)); cw = np.zeros(len(opts))
    idx = {o: i for i, o in enumerate(opts)}
    for v, w in zip(s[qid].astype(str).str.strip(), s[wcol]):
        i = idx.get(v)
        if i is not None:
            cu[i] += 1
            cw[i] += w
    if cu.sum() < utils.MIN_HUMAN_N or cw.sum() <= 0:
        return None, None
    return cu / cu.sum(), cw / cw.sum()


def main():
    rows = []
    for wave in wave_list():
        csv = os.path.join(BASE, "human_resp",
                           f"American_Trends_Panel_W{wave}", "responses.csv")
        df = pd.read_csv(csv, low_memory=False)
        wcols = [c for c in df.columns if c.upper() == f"WEIGHT_W{wave}"]
        if not wcols:
            wcols = [c for c in df.columns if c.upper().startswith("WEIGHT")]
        if not wcols:
            print(f"W{wave}: NO WEIGHT COLUMN", flush=True)
            continue
        wcol = wcols[0]
        options = load_options(wave)
        qids = [q for q in options if q in df.columns][:40]   # cap for speed

        # population
        tv_pop = []
        for qid in qids:
            u, w = dists(df, wcol, qid, options[qid])
            if u is not None:
                tv_pop.append(utils.tv(u, w))

        # singles and a sample of pairs
        tv_single, tv_pair = [], []
        val_masks = {}
        for dim, col in utils.DIM_TO_COL.items():
            if col not in df.columns:
                continue
            for val in utils.DIM_VALUES[dim]:
                if val in utils.IGNORE_VALUES.get(dim, set()):
                    continue
                m = df[col] == val
                if m.sum() >= utils.MIN_HUMAN_N:
                    val_masks[(dim, val)] = m
        for (dim, val), m in val_masks.items():
            sub = df.loc[m]
            for qid in qids[:15]:
                u, w = dists(sub, wcol, qid, options[qid])
                if u is not None:
                    tv_single.append(utils.tv(u, w))
        keys = list(val_masks)
        rng = np.random.default_rng(0)
        pair_keys = [(a, b) for a, b in itertools.combinations(keys, 2) if a[0] != b[0]]
        rng.shuffle(pair_keys)
        n_pair_cells = 0
        for a, b in pair_keys:
            if n_pair_cells >= 60:
                break
            m = val_masks[a] & val_masks[b]
            if m.sum() < utils.MIN_HUMAN_N:
                continue
            n_pair_cells += 1
            sub = df.loc[m]
            for qid in qids[:10]:
                u, w = dists(sub, wcol, qid, options[qid])
                if u is not None:
                    tv_pair.append(utils.tv(u, w))

        rows.append({"wave": wave,
                     "tv_pop": float(np.mean(tv_pop)),
                     "tv_single": float(np.mean(tv_single)),
                     "tv_pair": float(np.mean(tv_pair)) if tv_pair else None})
        print(f"W{wave}: pop={np.mean(tv_pop):.4f} single={np.mean(tv_single):.4f} "
              f"pair={np.mean(tv_pair):.4f}", flush=True)

    with open(os.path.join(BASE, "verification", "weight_sensitivity.json"), "w") as f:
        json.dump(rows, f, indent=1)
    print("\n=== WEIGHTED vs UNWEIGHTED TV (pooled over waves) ===")
    for k in ("tv_pop", "tv_single", "tv_pair"):
        v = [r[k] for r in rows if r[k] is not None]
        print(f"  {k}: mean={np.mean(v):.4f}  max={np.max(v):.4f}")
    print("  (reference: LLM simulation error is 0.17-0.33 TV)")


if __name__ == "__main__":
    main()
