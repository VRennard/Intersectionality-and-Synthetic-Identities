"""Shared loaders for the 2026-07-09 technical-review analysis batch.
Reuses the exact conventions of human_rigor_2_3.py / score_to_rows.py."""
import os, sys, json, itertools
import numpy as np
import pandas as pd

BASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(BASE, "advanced_bias_analysis"))
import utils  # noqa: E402

WAVES15 = ("26","27","29","32","34","36","41","42","43","45","49","50","54","82","92")


def load_options(wave):
    path = os.path.join(BASE, "data", "responses", f"survey_responses_W{wave}.json")
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return {q["question_id"]: [r["option"] for r in q["responses"]] for q in data}


def load_micro(wave):
    """responses df + per-respondent weight array (nan -> 0)."""
    csv = os.path.join(BASE, "human_resp", f"American_Trends_Panel_W{wave}", "responses.csv")
    df = pd.read_csv(csv, low_memory=False)
    wcol = f"WEIGHT_W{wave}"
    if wcol not in df.columns:
        cands = [c for c in df.columns if c.startswith("WEIGHT_W") and c.endswith(f"W{wave}")]
        wcol = cands[0] if cands else [c for c in df.columns if c.startswith("WEIGHT")][0]
    w = pd.to_numeric(df[wcol], errors="coerce").fillna(0.0).to_numpy(float)
    return df, w


def question_index(df, options):
    """{qid: (idx array with -1 = invalid, K)} for every question present."""
    out = {}
    for q, opts in options.items():
        if q not in df.columns:
            continue
        mapping = {o: i for i, o in enumerate(opts)}
        idx = df[q].map(mapping)
        out[q] = (idx.fillna(-1).to_numpy(int), len(opts))
    return out


def val_masks(df):
    out = {}
    for dim, col in utils.DIM_TO_COL.items():
        if col not in df.columns:
            continue
        for val in utils.DIM_VALUES[dim]:
            if val in utils.IGNORE_VALUES.get(dim, set()):
                continue
            m = (df[col] == val).to_numpy()
            if m.sum() >= utils.MIN_HUMAN_N:
                out[(dim, val)] = m
    return out


def cells(vm, min_n=None):
    """yield (depth, profile frozenset, mask) for d1/d2/d3, n>=MIN_HUMAN_N."""
    min_n = min_n or utils.MIN_HUMAN_N
    keys = sorted(vm, key=lambda kv: (utils.DIMS.index(kv[0]), kv[1]))
    for k in keys:
        yield 1, frozenset([k]), vm[k]
    for a, b in itertools.combinations(keys, 2):
        if a[0] == b[0]:
            continue
        m = vm[a] & vm[b]
        if m.sum() >= min_n:
            yield 2, frozenset([a, b]), m
    for a, b, c in itertools.combinations(keys, 3):
        if len({a[0], b[0], c[0]}) < 3:
            continue
        m = vm[a] & vm[b] & vm[c]
        if m.sum() >= min_n:
            yield 3, frozenset([a, b, c]), m


def wdist(qidx, K, mask, w):
    """weighted dist, raw n, kish n_eff over valid respondents in mask."""
    sel = mask & (qidx >= 0)
    n_raw = int(sel.sum())
    if n_raw < utils.MIN_HUMAN_N:
        return None, n_raw, 0.0
    ww = w[sel]
    sw = ww.sum()
    if sw <= 0:
        return None, n_raw, 0.0
    counts = np.bincount(qidx[sel], weights=ww, minlength=K).astype(float)
    neff = sw * sw / (ww * ww).sum() if (ww * ww).sum() > 0 else 0.0
    return counts / sw, n_raw, float(neff)


def bootstrap_ci(per_cluster_vals, stat=np.mean, B=10000, seed=11):
    """percentile CI of stat over cluster resamples (list of per-cluster values)."""
    rng = np.random.default_rng(seed)
    vals = np.asarray(per_cluster_vals, float)
    n = len(vals)
    stats = np.array([stat(vals[rng.integers(0, n, n)]) for _ in range(B)])
    return float(np.percentile(stats, 2.5)), float(np.percentile(stats, 97.5))
