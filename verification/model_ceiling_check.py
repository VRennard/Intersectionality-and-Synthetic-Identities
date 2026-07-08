"""
Priority items 1 + 2.

(1) HUMAN ADDITIVITY CONTEST — do real subgroups compose?
    For human pair cells, contest on steering vectors:
        cos(s_AB, s_A + s_B)   vs   max(cos(s_AB, s_A), cos(s_AB, s_B))
    s_g = p_g - p_pop, all from microdata. Restricted to n_AB >= 100
    (and >= 200 robustness) so ground-truth noise is small.
    The SAME steering contest is run for the LLM (gpt-4o-mini) on the
    same cells — apples-to-apples: human vs model composition rate.

(2) CONTEST CALIBRATION — what would the bias-vector contest show if
    additivity were TRUE?
    Synthetic cells: e*_AB = e_A + e_B + eta   (additive truth)
                     e*_AB = e_dom + eta       (collapse truth)
    where e_A, e_B are the model's real single bias vectors and eta is
    sampled from the model's own run-to-run differences (3 stochastic
    W26 runs; eta = delta/sqrt(2) matched on option count). The contest
    win rates under both regimes bracket the observed 78%.

Output: verification/human_additivity_calibration.json
"""

import os, sys, json, pickle, itertools
from collections import defaultdict

import numpy as np
import pandas as pd

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(BASE, "advanced_bias_analysis"))
import utils

HERE  = os.path.join(BASE, "verification")
MODEL = "gpt-4o-mini"
EXCL  = {"49"}
RNG   = np.random.default_rng(3)

# --weighted: survey-weighted human distributions (primary basis), weighted
# human caches, output human_additivity_calibration_weighted.json.
WEIGHTED = "--weighted" in sys.argv
SUFFIX   = "_weighted" if WEIGHTED else ""
utils.USE_WEIGHTS = WEIGHTED

# all stochastic reruns present on disk (4 as of 2026-06-11)
STOCH_RUNS = (1, 2, 3, 4)

out = {}


def wave_list():
    d = os.path.join(BASE, "human_resp")
    res = []
    for fn in sorted(os.listdir(d)):
        if fn.startswith("American_Trends_Panel_W"):
            w = fn.rsplit("W", 1)[1]
            if w not in EXCL and os.path.exists(os.path.join(d, fn, "responses.csv")):
                res.append(w)
    return sorted(res, key=int)


def load_options(wave):
    with open(os.path.join(BASE, "data", "responses",
                           f"survey_responses_W{wave}.json"), encoding="utf-8") as f:
        return {q["question_id"]: [r["option"] for r in q["responses"]]
                for q in json.load(f)}


def dist_n(series, opts, min_n=20, weights=None):
    """Distribution + raw n; weighted probabilities when weights given,
    but the min_n gate always counts raw respondents."""
    vc = series.value_counts()
    n = int(sum(vc.get(o, 0) for o in opts))
    if n < min_n:
        return None, 0
    if weights is None:
        c = np.array([vc.get(o, 0) for o in opts], dtype=float)
    else:
        acc = dict.fromkeys(opts, 0.0)
        for v, w in zip(series, weights):
            if v in acc and np.isfinite(w):
                acc[v] += w
        c = np.array([acc[o] for o in opts], dtype=float)
    s = c.sum()
    return (c / s, n) if s > 0 else (None, 0)


def col_weights(df_slice, wcol):
    return df_slice[wcol].to_numpy() if wcol else None


# ---------------------------------------------------------------- (1)
def build_noise_pool():
    runs = {}
    for i in STOCH_RUNS:
        d = {}
        with open(os.path.join(BASE, "data", "results", "stoch_test",
                               f"W26_run{i}.jsonl"), encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if r.get("status") != "success":
                    continue
                dist = np.array(r["response_distribution"], dtype=float)
                if dist.sum() <= 0:
                    continue
                d[(tuple(sorted(r["demographics"])), r["question_id"])] = dist / dist.sum()
        runs[i] = d
    pool = defaultdict(list)
    for i, j in itertools.combinations(STOCH_RUNS, 2):
        for key, da in runs[i].items():
            db = runs[j].get(key)
            if db is not None and len(db) == len(da):
                pool[len(da)].append((da - db) / np.sqrt(2))
    return {K: np.array(v) for K, v in pool.items()}


def human_and_llm_steering_contest():
    print("=== (1) STEERING-SPACE ADDITIVITY CONTEST, human vs LLM ===", flush=True)
    npool = build_noise_pool()
    llm_ceil = {"add": 0, "single": 0}
    res = {thr: {"h_add": 0, "h_single": 0} for thr in (100, 200)}
    # (1b) additive-truth reference at matched human n: synthetic pair cell
    # p_pop + s_a + s_b observed through multinomial sampling at n_ab
    ref = {thr: {"h_add": 0, "h_single": 0} for thr in (100, 200)}
    llm_res = {"add": 0, "single": 0}

    utils.MODEL_TAG = MODEL
    for wave in wave_list():
        options = load_options(wave)
        df = pd.read_csv(os.path.join(BASE, "human_resp",
                                      f"American_Trends_Panel_W{wave}",
                                      "responses.csv"), low_memory=False)
        wcol = utils._weight_col(df, wave)
        qids = [q for q in options if q in df.columns]
        llm, qmeta = utils.build_llm_index(wave, max_level=2)
        l_avg = dict(llm.get(utils.AVG_PROFILE, {}))
        if not l_avg:
            acc = defaultdict(list)
            for p, qd in llm.items():
                if len(p) == 1:
                    for qid, dd in qd.items():
                        acc[qid].append(dd)
            l_avg = {qid: np.mean(v, axis=0) for qid, v in acc.items()}

        # population dists
        pop = {}
        for qid in qids:
            sl = df.dropna(subset=[qid])
            d, n = dist_n(sl[qid], options[qid], weights=col_weights(sl, wcol))
            if d is not None:
                pop[qid] = d

        # single masks/dists
        val_masks = {}
        for dim, col in utils.DIM_TO_COL.items():
            if col not in df.columns:
                continue
            for val in utils.DIM_VALUES[dim]:
                if val in utils.IGNORE_VALUES.get(dim, set()):
                    continue
                m = (df[col] == val).to_numpy()
                if m.sum() >= 100:
                    val_masks[(dim, val)] = m
        singles = {}
        for k, m in val_masks.items():
            sub = df.loc[m]
            singles[k] = {}
            for qid in qids:
                sl = sub.dropna(subset=[qid])
                d, n = dist_n(sl[qid], options[qid], min_n=100,
                              weights=col_weights(sl, wcol))
                if d is not None and qid in pop:
                    singles[k][qid] = d

        n_cells = 0
        for a, b in itertools.combinations(sorted(val_masks), 2):
            if a[0] == b[0]:
                continue
            m = val_masks[a] & val_masks[b]
            if m.sum() < 100:
                continue
            sub = df.loc[m]
            profile = frozenset([a, b])
            lq = llm.get(profile, {})
            la = llm.get(frozenset([a]), {})
            lb = llm.get(frozenset([b]), {})
            for qid in qids:
                if qid not in pop or qid not in singles.get(a, {}) \
                        or qid not in singles.get(b, {}):
                    continue
                sl = sub.dropna(subset=[qid])
                d_ab, n_ab = dist_n(sl[qid], options[qid], min_n=100,
                                    weights=col_weights(sl, wcol))
                if d_ab is None:
                    continue
                s_ab = d_ab - pop[qid]
                s_a  = singles[a][qid] - pop[qid]
                s_b  = singles[b][qid] - pop[qid]
                c_add = utils.cosine_sim(s_ab, s_a + s_b)
                c_sgl = max(utils.cosine_sim(s_ab, s_a),
                            utils.cosine_sim(s_ab, s_b))
                p_true = np.clip(pop[qid] + s_a + s_b, 0, None)
                if p_true.sum() > 0:
                    p_true = p_true / p_true.sum()
                    s_syn = RNG.multinomial(n_ab, p_true) / n_ab - pop[qid]
                    r_add = utils.cosine_sim(s_syn, s_a + s_b)
                    r_sgl = max(utils.cosine_sim(s_syn, s_a),
                                utils.cosine_sim(s_syn, s_b))
                else:
                    r_add = r_sgl = None
                for thr in (100, 200):
                    if n_ab >= thr:
                        res[thr]["h_add" if c_add > c_sgl else "h_single"] += 1
                        if r_add is not None:
                            ref[thr]["h_add" if r_add > r_sgl else "h_single"] += 1
                # LLM same cells, steering space
                lp, lA, lB, av = lq.get(qid), la.get(qid), lb.get(qid), l_avg.get(qid)
                if lp is not None and lA is not None and lB is not None \
                        and av is not None and len(lp) == len(av) == len(lA) == len(lB):
                    sh_ab, sh_a, sh_b = lp - av, lA - av, lB - av
                    cl_add = utils.cosine_sim(sh_ab, sh_a + sh_b)
                    cl_sgl = max(utils.cosine_sim(sh_ab, sh_a),
                                 utils.cosine_sim(sh_ab, sh_b))
                    llm_res["add" if cl_add > cl_sgl else "single"] += 1
                    K = len(sh_a)
                    if K in npool and len(npool[K]):
                        eta = npool[K][RNG.integers(len(npool[K]))]
                        s_syn2 = sh_a + sh_b + eta
                        rc_add = utils.cosine_sim(s_syn2, sh_a + sh_b)
                        rc_sgl = max(utils.cosine_sim(s_syn2, sh_a),
                                     utils.cosine_sim(s_syn2, sh_b))
                        llm_ceil["add" if rc_add > rc_sgl else "single"] += 1
                n_cells += 1
        print(f"  W{wave}: {n_cells} matched cells", flush=True)

    out["1"] = {}
    for thr in (100, 200):
        t = res[thr]["h_add"] + res[thr]["h_single"]
        share = res[thr]["h_add"] / t if t else float("nan")
        out["1"][f"human_additive_share_n{thr}"] = share
        out["1"][f"n_cells_n{thr}"] = t
        print(f"  HUMAN additive wins (n>={thr}): {share:.1%}  ({t} cells)", flush=True)
        tr = ref[thr]["h_add"] + ref[thr]["h_single"]
        rshare = ref[thr]["h_add"] / tr if tr else float("nan")
        out["1"][f"additive_truth_ref_n{thr}"] = rshare
        print(f"  REF additive truth + multinomial noise (n>={thr}): "
              f"{rshare:.1%}  ({tr} cells)", flush=True)
    t = llm_res["add"] + llm_res["single"]
    out["1"]["llm_steering_additive_share"] = llm_res["add"] / t
    out["1"]["llm_n_cells"] = t
    print(f"  LLM   additive wins (same cells, steering space): "
          f"{llm_res['add']/t:.1%}  ({t} cells)", flush=True)
    tc = llm_ceil["add"] + llm_ceil["single"]
    if tc:
        print(f"  LLM CEILING: additive-truth + model eta (same cells): "
              f"{llm_ceil['add']/tc:.1%}  ({tc} cells)", flush=True)


# ---------------------------------------------------------------- (2)
def calibration():
    print("\n=== (2) CONTEST CALIBRATION under synthetic truth ===", flush=True)
    # noise pool from stochastic runs: per option-count K, deltas between
    # all pairs of repeated runs
    runs = {}
    for i in STOCH_RUNS:
        d = {}
        with open(os.path.join(BASE, "data", "results", "stoch_test",
                               f"W26_run{i}.jsonl"), encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if r.get("status") != "success":
                    continue
                dist = np.array(r["response_distribution"], dtype=float)
                if dist.sum() <= 0:
                    continue
                d[(tuple(sorted(r["demographics"])), r["question_id"])] = dist / dist.sum()
        runs[i] = d
    noise_pool = defaultdict(list)
    for i, j in itertools.combinations(STOCH_RUNS, 2):
        for key, da in runs[i].items():
            db = runs[j].get(key)
            if db is not None and len(db) == len(da):
                noise_pool[len(da)].append((da - db) / np.sqrt(2))
    for K in noise_pool:
        noise_pool[K] = np.array(noise_pool[K])
    print(f"  noise pool: {sum(len(v) for v in noise_pool.values())} deltas",
          flush=True)

    # real single bias vectors from W26 + W43
    utils.MODEL_TAG = MODEL
    stats = {"additive_truth": {"add": 0, "single": 0},
             "collapse_truth": {"add": 0, "single": 0}}
    for wave in ("26", "43"):
        with open(os.path.join(HERE, "cache",
                               f"human_W{wave}{SUFFIX}.pkl"), "rb") as f:
            human, _ = pickle.load(f)
        llm, qmeta = utils.build_llm_index(wave, max_level=2)

        def bias(profile, qid):
            lp = llm.get(profile, {}).get(qid)
            hp = human.get(profile, {}).get(qid)
            if lp is None or hp is None or len(lp) != len(hp):
                return None
            return lp - hp

        n_done = 0
        for profile in llm:
            if len(profile) != 2 or n_done >= 25000:
                continue
            (a, va), (b, vb) = sorted(profile)
            pa, pb = frozenset([(a, va)]), frozenset([(b, vb)])
            for qid in llm[profile]:
                e_a, e_b = bias(pa, qid), bias(pb, qid)
                if e_a is None or e_b is None:
                    continue
                K = len(e_a)
                if K not in noise_pool or not len(noise_pool[K]):
                    continue
                eta = noise_pool[K][RNG.integers(0, len(noise_pool[K]))]
                # additive truth (cosine is scale-invariant, so the
                # observed 0.57x magnitude shrinkage does not matter here)
                for regime, e_true in (("additive_truth", e_a + e_b),
                                       ("collapse_truth",
                                        e_a if np.linalg.norm(e_a) >= np.linalg.norm(e_b) else e_b)):
                    e_syn = e_true + eta
                    c_add = utils.cosine_sim(e_syn, e_a + e_b)
                    c_sgl = max(utils.cosine_sim(e_syn, e_a),
                                utils.cosine_sim(e_syn, e_b))
                    stats[regime]["single" if c_sgl >= c_add else "add"] += 1
                n_done += 1
        print(f"  W{wave}: {n_done} synthetic cells", flush=True)

    out["2"] = {}
    for regime, s in stats.items():
        t = s["add"] + s["single"]
        share = s["single"] / t
        out["2"][f"best_single_wins_under_{regime}"] = share
        print(f"  best-single win rate under {regime:15s}: {share:.1%}",
              flush=True)


# ---------------------------------------------------------------- (3)
def collapse_index():
    """Place every model's observed depth-2 share between the two
    synthetic endpoints: 0 = additive truth, 1 = collapse truth."""
    print("\n=== (3) PER-MODEL COLLAPSE INDEX ===", flush=True)
    lo = out["2"]["best_single_wins_under_additive_truth"]
    hi = out["2"]["best_single_wins_under_collapse_truth"]
    with open(os.path.join(HERE, f"collapse_results{SUFFIX}.json")) as f:
        rows = [r for r in json.load(f) if r["wave"] not in EXCL]
    out["3"] = {"endpoints": [lo, hi]}
    for model in sorted({r["model"] for r in rows}):
        rs = [r for r in rows if r["model"] == model and r["depth"] == 2]
        tot = sum(r["wins"] for r in rs)
        win = sum(r["wins"] for r in rs if r["predictor"] == "best_single")
        if not tot:
            continue
        obs = win / tot
        idx = (obs - lo) / (hi - lo)
        out["3"][model] = {"observed": obs, "index": idx}
        print(f"  {model:28s} obs={obs:.1%}  index={idx:.2f}", flush=True)


if __name__ == "__main__":
    human_and_llm_steering_contest()
