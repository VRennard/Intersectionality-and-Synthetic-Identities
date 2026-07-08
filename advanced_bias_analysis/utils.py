"""
Shared utilities for advanced_bias_analysis scripts.
All scripts import from here to avoid code duplication.
"""

import json, os, itertools
from collections import defaultdict

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["pdf.fonttype"] = 42

# ── Paths & config ─────────────────────────────────────────────────────────────
BASE      = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_TAG = "llama3_1_8b_instruct_q4"
OUT_BASE  = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output", MODEL_TAG)

# Dimension name -> CSV column
DIM_TO_COL = {
    "Age":             "AGE",
    "Gender":          "SEX",
    "Race":            "RACE",
    "Income":          "INCOME",
    "Political Party": "POLPARTY",
    "Religion":        "RELIG",
    "Education":       "EDUCATION",
}
# Sorted longest-first for unambiguous prefix matching
DIM_NAMES = sorted(DIM_TO_COL.keys(), key=len, reverse=True)
DIMS      = list(DIM_TO_COL.keys())

DIM_VALUES = {
    "Age":             ["18-29", "30-49", "50-64", "65+"],
    "Gender":          ["Male", "Female"],
    "Race":            ["White", "Black", "Hispanic", "Asian", "Mixed Race", "Other"],
    "Income":          ["Less than $30,000", "$30,000-$50,000", "$50,000-$75,000",
                        "$75,000-$100,000", "$100,000 or more"],
    "Political Party": ["Democrat", "Republican", "Independent"],
    "Religion":        ["Roman Catholic", "Atheist", "Jewish", "Muslim", "Protestant"],
    "Education":       ["Less than high school", "High school graduate",
                        "Some college, no degree", "Associate's degree",
                        "College graduate/some postgrad", "Postgraduate"],
}

IGNORE_VALUES = {
    "Age":             {"Refused"},
    "Gender":          {"Refused"},
    "Race":            {"Refused"},
    "Income":          {"Refused"},
    "Political Party": {"Other", "Refused"},
    "Religion":        {"Nothing in particular", "Agnostic", "Other",
                        "Mormon", "Christian", "Buddhist", "Unitarian", "Hindu",
                        "Orthodox", "Refused"},
    "Education":       set(),
}

DIM_SHORT = {
    "Age": "Age", "Gender": "Gender", "Race": "Race",
    "Income": "Income", "Political Party": "Party",
    "Religion": "Religion", "Education": "Education",
}

DIM_COLORS = {
    "Age":             "#4e79a7",
    "Gender":          "#e15759",
    "Race":            "#76b7b2",
    "Income":          "#59a14f",
    "Political Party": "#edc948",
    "Religion":        "#b07aa1",
    "Education":       "#ff9da7",
}

AVG_PROFILE = frozenset()   # sentinel for "Average American"
MIN_HUMAN_N = 20            # minimum cell size


# ── Wave detection ──────────────────────────────────────────────────────────────

def available_waves():
    d = os.path.join(BASE, "data", "results", MODEL_TAG)
    if not os.path.isdir(d):
        return []
    waves = []
    for fn in os.listdir(d):
        if fn.startswith("W") and fn.endswith(".jsonl"):
            waves.append(fn[1:-6])
    return sorted(waves, key=lambda x: int(x))


# ── Parsing helpers ─────────────────────────────────────────────────────────────

def parse_demo(features):
    """List[str] -> frozenset of (dim, val), or None if unparseable."""
    if not features or features == ["Average American"]:
        return AVG_PROFILE
    result = []
    for f in features:
        matched = False
        for dim in DIM_NAMES:
            if f.startswith(dim + " "):
                val = f[len(dim) + 1:]
                result.append((dim, val))
                matched = True
                break
        if not matched:
            return None
    return frozenset(result)


def profile_label(profile, short=True):
    """frozenset of (dim,val) -> readable label."""
    parts = sorted(profile, key=lambda x: DIM_NAMES.index(x[0]) if x[0] in DIM_NAMES else 99)
    d = DIM_SHORT if short else {k: k for k in DIM_NAMES}
    return " × ".join(f"{d.get(dim, dim)}:{val[:10]}" for dim, val in parts)


def dim_combo_label(profile):
    dims = sorted({d for d, v in profile},
                  key=lambda x: DIM_NAMES.index(x) if x in DIM_NAMES else 99)
    return " × ".join(DIM_SHORT.get(d, d) for d in dims)


# ── JSONL loading ───────────────────────────────────────────────────────────────

def load_jsonl(wave):
    path = os.path.join(BASE, "data", "results", MODEL_TAG, f"W{wave}.jsonl")
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return records


def build_llm_index(wave, max_level=3):
    """
    Returns:
      llm   : {profile_frozenset: {qid: np.array normalized dist}}
      qmeta : {qid: {"text": str, "options": [str]}}
    """
    records = load_jsonl(wave)
    llm   = {}
    qmeta = {}
    for r in records:
        if r.get("status") != "success":
            continue
        dist = np.array(r.get("response_distribution", []), dtype=float)
        if dist.sum() == 0:
            continue
        dist = dist / dist.sum()

        features = r.get("demographics", [])
        qid      = r["question_id"]
        opts     = r.get("options", [])

        if qid not in qmeta:
            qmeta[qid] = {"text": r.get("question_text", qid), "options": opts}

        profile = parse_demo(features)
        if profile is None:
            continue
        if len(profile) > max_level:
            continue
        # Filter ignored values
        if any(val in IGNORE_VALUES.get(dim, set()) for dim, val in profile):
            continue

        llm.setdefault(profile, {})[qid] = dist

    return llm, qmeta


# ── Human data loading ──────────────────────────────────────────────────────────

def _norm_series(series, opts, weights=None):
    """Normalized distribution over opts from a pandas Series of string values.
    If weights is given (aligned with series), counts are survey-weighted;
    the n >= MIN_HUMAN_N gate always counts raw respondents."""
    counts = np.zeros(len(opts), dtype=float)
    n_raw = 0
    idx = {o: i for i, o in enumerate(opts)}
    if weights is None:
        for v in series:
            i = idx.get(str(v).strip())
            if i is not None:
                counts[i] += 1
                n_raw += 1
    else:
        for v, w in zip(series, weights):
            i = idx.get(str(v).strip())
            if i is not None and np.isfinite(w) and w > 0:
                counts[i] += w
                n_raw += 1
    if n_raw < MIN_HUMAN_N or counts.sum() <= 0:
        return None
    return counts / counts.sum()


# When True, load_human_index returns survey-weighted distributions
# (WEIGHT_W{wave} column); cell inclusion still requires MIN_HUMAN_N raw
# respondents. Set before calling load_human_index.
USE_WEIGHTS = False


def _weight_col(df, wave):
    if not USE_WEIGHTS:
        return None
    for c in df.columns:
        if c.upper() == f"WEIGHT_W{wave}":
            return c
    for c in df.columns:
        if c.upper().startswith("WEIGHT"):
            return c
    raise ValueError(f"USE_WEIGHTS=True but no weight column for W{wave}")


def _cell_dist(df, mask, qid, opts, wcol):
    """Distribution for one cell, honouring USE_WEIGHTS via wcol."""
    if wcol is None:
        return _norm_series(df.loc[mask, qid].dropna(), opts)
    sub = df.loc[mask, [qid, wcol]].dropna(subset=[qid])
    return _norm_series(sub[qid], opts, sub[wcol])


def load_human_index(wave, qids, qmeta, max_level=2):
    """
    Returns {profile_frozenset: {qid: np.array}} for avg + singles + pairs (+ triples if max_level=3).
    Only computes profiles for valid DIM_VALUES (not all CSV values).
    """
    csv = os.path.join(BASE, "human_resp", f"American_Trends_Panel_W{wave}", "responses.csv")
    df  = pd.read_csv(csv, low_memory=False)
    wcol = _weight_col(df, wave)

    human = {}

    # Average American
    human[AVG_PROFILE] = {}
    all_mask = np.ones(len(df), dtype=bool)
    for qid in qids:
        if qid not in df.columns:
            continue
        d = _cell_dist(df, all_mask, qid, qmeta[qid]["options"], wcol)
        if d is not None:
            human[AVG_PROFILE][qid] = d

    # Singles
    for dim, col in DIM_TO_COL.items():
        if col not in df.columns:
            continue
        for val in DIM_VALUES[dim]:
            if val in IGNORE_VALUES.get(dim, set()):
                continue
            mask    = df[col] == val
            if mask.sum() < MIN_HUMAN_N:
                continue
            profile = frozenset([(dim, val)])
            human.setdefault(profile, {})
            for qid in qids:
                if qid not in df.columns:
                    continue
                d = _cell_dist(df, mask, qid, qmeta[qid]["options"], wcol)
                if d is not None:
                    human[profile][qid] = d

    if max_level < 2:
        return human

    # Pairs
    for dim_a, dim_b in itertools.combinations(DIMS, 2):
        col_a, col_b = DIM_TO_COL[dim_a], DIM_TO_COL[dim_b]
        if col_a not in df.columns or col_b not in df.columns:
            continue
        for val_a in DIM_VALUES[dim_a]:
            if val_a in IGNORE_VALUES.get(dim_a, set()):
                continue
            for val_b in DIM_VALUES[dim_b]:
                if val_b in IGNORE_VALUES.get(dim_b, set()):
                    continue
                mask    = (df[col_a] == val_a) & (df[col_b] == val_b)
                if mask.sum() < MIN_HUMAN_N:
                    continue
                profile = frozenset([(dim_a, val_a), (dim_b, val_b)])
                human.setdefault(profile, {})
                for qid in qids:
                    if qid not in df.columns:
                        continue
                    d = _cell_dist(df, mask, qid, qmeta[qid]["options"], wcol)
                    if d is not None:
                        human[profile][qid] = d

    if max_level < 3:
        return human

    # Triples
    for dim_a, dim_b, dim_c in itertools.combinations(DIMS, 3):
        col_a, col_b, col_c = DIM_TO_COL[dim_a], DIM_TO_COL[dim_b], DIM_TO_COL[dim_c]
        if not all(c in df.columns for c in [col_a, col_b, col_c]):
            continue
        for val_a in DIM_VALUES[dim_a]:
            if val_a in IGNORE_VALUES.get(dim_a, set()):
                continue
            for val_b in DIM_VALUES[dim_b]:
                if val_b in IGNORE_VALUES.get(dim_b, set()):
                    continue
                for val_c in DIM_VALUES[dim_c]:
                    if val_c in IGNORE_VALUES.get(dim_c, set()):
                        continue
                    mask = (df[col_a] == val_a) & (df[col_b] == val_b) & (df[col_c] == val_c)
                    if mask.sum() < MIN_HUMAN_N:
                        continue
                    profile = frozenset([(dim_a, val_a), (dim_b, val_b), (dim_c, val_c)])
                    human.setdefault(profile, {})
                    for qid in qids:
                        if qid not in df.columns:
                            continue
                        d = _cell_dist(df, mask, qid, qmeta[qid]["options"], wcol)
                        if d is not None:
                            human[profile][qid] = d

    return human


# ── Metrics ─────────────────────────────────────────────────────────────────────

def tv(p, q):
    """Total Variation distance."""
    return float(0.5 * np.abs(p - q).sum())


def cosine_sim(a, b):
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-10 or nb < 1e-10:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def error_vec(llm_dist, human_dist):
    return llm_dist - human_dist


# ── Question selection ──────────────────────────────────────────────────────────

def select_questions(human, llm, qmeta, n=40):
    """Top-N questions by mean absolute LLM-Human difference."""
    avg_h = human.get(AVG_PROFILE, {})
    avg_l = llm.get(AVG_PROFILE, {})

    # If LLM has no Average American profile, compute it as mean over singles
    if not avg_l:
        singles = [p for p in llm if len(p) == 1]
        for qid in qmeta:
            dists = [llm[p][qid] for p in singles if qid in llm.get(p, {})]
            if dists:
                avg_l[qid] = np.mean(dists, axis=0)

    scores = {}
    for qid, meta in qmeta.items():
        if qid not in avg_h or qid not in avg_l:
            continue
        opts = meta["options"]
        if len([o for o in opts if "Refused" not in o]) < 3:
            continue
        gap, cnt = 0.0, 0
        for profile in human:
            hd = human[profile].get(qid)
            ld = llm.get(profile, {}).get(qid)
            if hd is not None and ld is not None:
                gap += np.abs(hd - ld).sum()
                cnt += 1
        if cnt >= 5:
            scores[qid] = gap / cnt
    return sorted(scores, key=scores.get, reverse=True)[:n]


def out_dir(name):
    d = os.path.join(OUT_BASE, name)
    os.makedirs(d, exist_ok=True)
    return d
