"""
Human split-half noise floor for the spine figure.

For each human profile cell (depth 1/2/3, same DIM_VALUES/IGNORE filters as the
spine) with >= 2*MIN_HUMAN_N respondents, randomly split respondents into two
halves and compute the TV distance between the halves' answer distributions,
per question. Averaged per depth, this is the TV you'd expect between two
independent surveys of the same subgroup — a reference floor for LLM error.

Conservative by construction: each half has n/2 respondents, so this is
roughly 2x the sampling error of the full-sample ground truth used in the
spine. Reported as-is and labeled "human split-half".

Question options are read from data/responses/survey_responses_W*.json
(same option lists used in the LLM prompts), so distributions are directly
comparable to the spine computation.

Output: verification/noise_floor.json
"""

import os, sys, json, itertools
from collections import defaultdict

import numpy as np
import pandas as pd

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(BASE, "advanced_bias_analysis"))
import utils

OUT_DIR  = os.path.join(BASE, "verification")
N_SPLITS = 2
MIN_CELL = 2 * utils.MIN_HUMAN_N   # 40: each half keeps the spine's n>=20 rule
RNG      = np.random.default_rng(0)

# --weighted: distributions per half are survey-weighted (WEIGHT_W{wave});
# output goes to noise_floor_weighted.json.
WEIGHTED = "--weighted" in sys.argv
SUFFIX   = "_weighted" if WEIGHTED else ""


def wave_list():
    d = os.path.join(BASE, "human_resp")
    waves = []
    for fn in os.listdir(d):
        if fn.startswith("American_Trends_Panel_W"):
            w = fn.rsplit("W", 1)[1]
            if os.path.exists(os.path.join(d, fn, "responses.csv")):
                waves.append(w)
    return sorted(waves, key=int)


def load_options(wave):
    """qid -> ordered option list, from the survey response summaries."""
    path = os.path.join(BASE, "data", "responses", f"survey_responses_W{wave}.json")
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return {q["question_id"]: [r["option"] for r in q["responses"]] for q in data}


def dist_from_series(series, opts, weights=None):
    """Normalized distribution over opts; None if fewer than MIN_HUMAN_N hits.
    With weights (aligned to series), counts are survey-weighted; the
    MIN_HUMAN_N gate counts raw respondents."""
    if weights is None:
        vc = series.value_counts()
        counts = np.array([vc.get(o, 0) for o in opts], dtype=float)
        if counts.sum() < utils.MIN_HUMAN_N:
            return None
        return counts / counts.sum()
    idx = {o: i for i, o in enumerate(opts)}
    counts = np.zeros(len(opts), dtype=float)
    n_raw = 0
    for v, w in zip(series, weights):
        i = idx.get(v)
        if i is not None and np.isfinite(w) and w > 0:
            counts[i] += w
            n_raw += 1
    if n_raw < utils.MIN_HUMAN_N or counts.sum() <= 0:
        return None
    return counts / counts.sum()


def cell_masks(df):
    """Yield (depth, mask) for all single/pair/triple cells passing filters."""
    val_masks = {}   # (dim, val) -> bool array
    for dim, col in utils.DIM_TO_COL.items():
        if col not in df.columns:
            continue
        for val in utils.DIM_VALUES[dim]:
            if val in utils.IGNORE_VALUES.get(dim, set()):
                continue
            m = (df[col] == val).to_numpy()
            if m.sum() >= MIN_CELL:
                val_masks[(dim, val)] = m

    keys = sorted(val_masks, key=lambda kv: (utils.DIMS.index(kv[0]), kv[1]))
    for k in keys:
        yield 1, val_masks[k]
    for (d1, v1), (d2, v2) in itertools.combinations(keys, 2):
        if d1 == d2:
            continue
        m = val_masks[(d1, v1)] & val_masks[(d2, v2)]
        if m.sum() >= MIN_CELL:
            yield 2, m
    for (d1, v1), (d2, v2), (d3, v3) in itertools.combinations(keys, 3):
        if len({d1, d2, d3}) < 3:
            continue
        m = val_masks[(d1, v1)] & val_masks[(d2, v2)] & val_masks[(d3, v3)]
        if m.sum() >= MIN_CELL:
            yield 3, m


def main():
    rows = []
    for wave in wave_list():
        try:
            options = load_options(wave)
        except FileNotFoundError:
            print(f"W{wave}: no survey_responses json, skipped", flush=True)
            continue
        csv = os.path.join(BASE, "human_resp",
                           f"American_Trends_Panel_W{wave}", "responses.csv")
        df = pd.read_csv(csv, low_memory=False)
        qids = [q for q in options if q in df.columns]
        wvals = None
        if WEIGHTED:
            wcols = [c for c in df.columns if c.upper() == f"WEIGHT_W{wave}"] \
                    or [c for c in df.columns if c.upper().startswith("WEIGHT")]
            if not wcols:
                print(f"W{wave}: no weight column, skipped", flush=True)
                continue
            wvals = df[wcols[0]]

        tv_by_depth = defaultdict(list)
        cells_by_depth = defaultdict(int)
        for depth, mask in cell_masks(df):
            idx = np.flatnonzero(mask)
            cells_by_depth[depth] += 1
            for _ in range(N_SPLITS):
                perm = RNG.permutation(idx)
                half_a, half_b = perm[: len(perm) // 2], perm[len(perm) // 2 :]
                for qid in qids:
                    col = df[qid]
                    if wvals is None:
                        da = dist_from_series(col.iloc[half_a].dropna(), options[qid])
                        db = dist_from_series(col.iloc[half_b].dropna(), options[qid])
                    else:
                        sa, sb = col.iloc[half_a], col.iloc[half_b]
                        ma, mb = sa.notna(), sb.notna()
                        da = dist_from_series(sa[ma], options[qid],
                                              wvals.iloc[half_a][ma])
                        db = dist_from_series(sb[mb], options[qid],
                                              wvals.iloc[half_b][mb])
                    if da is None or db is None:
                        continue
                    tv_by_depth[depth].append(utils.tv(da, db))

        for depth, vals in sorted(tv_by_depth.items()):
            rows.append({
                "wave": wave, "depth": depth,
                "n_cells": cells_by_depth[depth],
                "n_obs": len(vals),
                "mean_tv": float(np.mean(vals)),
                "std_tv": float(np.std(vals)),
            })
        print(f"W{wave}: " +
              ", ".join(f"d{d}={np.mean(v):.4f}(cells={cells_by_depth[d]},obs={len(v)})"
                        for d, v in sorted(tv_by_depth.items())),
              flush=True)

    with open(os.path.join(OUT_DIR, f"noise_floor{SUFFIX}.json"), "w") as f:
        json.dump(rows, f, indent=1)

    print("\n=== NOISE FLOOR SUMMARY (pooled over waves, obs-weighted) ===")
    for depth in (1, 2, 3):
        rs = [r for r in rows if r["depth"] == depth]
        if not rs:
            continue
        n = sum(r["n_obs"] for r in rs)
        m = sum(r["mean_tv"] * r["n_obs"] for r in rs) / n
        print(f"  depth {depth}: split-half TV = {m:.4f}  (obs={n})")


if __name__ == "__main__":
    main()
