"""
Verification computations for paper plan:
(a) SPINE: mean TV(LLM, human) by profile depth (1/2/3) per model.
(b) COLLAPSE SHARE: at depth 2, per (profile, question), which predictor of the
    LLM bias vector wins: best-single vs additive-singles.
    At depth 3: best-single vs best-pair vs additive-singles.

Reuses advanced_bias_analysis/utils.py loaders. Human indices are cached per
wave (model-independent).
"""

import os, sys, json, pickle, itertools
from collections import defaultdict

import numpy as np

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(BASE, "advanced_bias_analysis"))
import utils

OUT_DIR   = os.path.join(BASE, "verification")
CACHE_DIR = os.path.join(OUT_DIR, "cache")
os.makedirs(CACHE_DIR, exist_ok=True)

SPINE_MODELS    = ["gpt-4o-mini", "gemma2_9b", "gpt-4o", "claude-haiku-4-5-20251001"]
COLLAPSE_MODELS = ["gpt-4o-mini", "gemma2_9b"]   # models with triples

# CLI: verify_spine_collapse.py [--weighted] model1 model2 ...  (models also
# used for collapse if they have triples). Results merge into the existing
# JSONs; --weighted uses survey-weighted human caches and writes *_weighted
# output files.
WEIGHTED = "--weighted" in sys.argv
_args = [a for a in sys.argv[1:] if a != "--weighted"]
if WEIGHTED:
    utils.USE_WEIGHTS = True
if _args:
    SPINE_MODELS    = _args
    COLLAPSE_MODELS = _args
SUFFIX = "_weighted" if WEIGHTED else ""


def model_waves(model):
    d = os.path.join(BASE, "data", "results", model)
    return sorted(
        (fn[1:-6] for fn in os.listdir(d)
         if fn.startswith("W") and fn.endswith(".jsonl") and fn[1:-6].isdigit()),
        key=int,
    )


def get_human_index(wave, qmeta):
    """Cached human index up to triples. Cache key includes qid set hash."""
    cache = os.path.join(CACHE_DIR, f"human_W{wave}{SUFFIX}.pkl")
    if os.path.exists(cache):
        with open(cache, "rb") as f:
            human, cached_qids = pickle.load(f)
        if set(qmeta.keys()) <= cached_qids:
            return human
    human = utils.load_human_index(wave, list(qmeta.keys()), qmeta, max_level=3)
    with open(cache, "wb") as f:
        pickle.dump((human, set(qmeta.keys())), f)
    return human


def main():
    # ----- pass 1: union of qmeta per wave across models (they share questions,
    # but be safe), then build/cache human indices -----
    spine_rows    = []   # model, wave, depth, n_cells, mean_tv
    collapse_rows = []   # model, wave, depth, predictor, wins, mean_cos

    for model in SPINE_MODELS:
        utils.MODEL_TAG = model
        for wave in model_waves(model):
            llm, qmeta = utils.build_llm_index(wave, max_level=3)
            human = get_human_index(wave, qmeta)

            # ---------- (a) spine ----------
            tv_by_depth = defaultdict(list)
            for profile, qdists in llm.items():
                depth = len(profile)
                if depth < 1 or depth > 3:
                    continue
                hq = human.get(profile)
                if not hq:
                    continue
                for qid, ld in qdists.items():
                    hd = hq.get(qid)
                    if hd is None or len(hd) != len(ld):
                        continue
                    tv_by_depth[depth].append(utils.tv(ld, hd))
            for depth, vals in sorted(tv_by_depth.items()):
                spine_rows.append({
                    "model": model, "wave": wave, "depth": depth,
                    "n_cells": len(vals), "mean_tv": float(np.mean(vals)),
                    "std_tv": float(np.std(vals)),
                })
            print(f"[spine] {model} W{wave}: " +
                  ", ".join(f"d{d}={np.mean(v):.4f}(n={len(v)})"
                            for d, v in sorted(tv_by_depth.items())),
                  flush=True)

            # ---------- (b) collapse share ----------
            if model not in COLLAPSE_MODELS:
                continue

            def bias(profile, qid):
                ld = llm.get(profile, {}).get(qid)
                hd = human.get(profile, {}).get(qid)
                if ld is None or hd is None or len(ld) != len(hd):
                    return None
                return ld - hd

            stats = {2: defaultdict(lambda: [0, []]),   # predictor -> [wins, cos list]
                     3: defaultdict(lambda: [0, []])}

            for profile, qdists in llm.items():
                depth = len(profile)
                if depth == 2:
                    a, b = sorted(profile)
                    pa, pb = frozenset([a]), frozenset([b])
                    for qid in qdists:
                        e_p = bias(profile, qid)
                        e_a = bias(pa, qid)
                        e_b = bias(pb, qid)
                        if e_p is None or e_a is None or e_b is None:
                            continue
                        cos_single = max(utils.cosine_sim(e_p, e_a),
                                         utils.cosine_sim(e_p, e_b))
                        cos_add    = utils.cosine_sim(e_p, e_a + e_b)
                        win = "best_single" if cos_single >= cos_add else "additive"
                        stats[2][win][0] += 1
                        stats[2]["best_single"][1].append(cos_single)
                        stats[2]["additive"][1].append(cos_add)
                elif depth == 3:
                    feats = sorted(profile)
                    singles = [frozenset([f]) for f in feats]
                    pairs   = [frozenset(c) for c in itertools.combinations(feats, 2)]
                    for qid in qdists:
                        e_p = bias(profile, qid)
                        if e_p is None:
                            continue
                        e_singles = [bias(s, qid) for s in singles]
                        if any(e is None for e in e_singles):
                            continue
                        e_pairs = [bias(p, qid) for p in pairs]
                        e_pairs = [e for e in e_pairs if e is not None]
                        if not e_pairs:
                            continue
                        cos_single = max(utils.cosine_sim(e_p, e) for e in e_singles)
                        cos_pair   = max(utils.cosine_sim(e_p, e) for e in e_pairs)
                        cos_add    = utils.cosine_sim(e_p, sum(e_singles))
                        cands = {"best_single": cos_single,
                                 "best_pair":   cos_pair,
                                 "additive":    cos_add}
                        win = max(cands, key=cands.get)
                        stats[3][win][0] += 1
                        for k, v in cands.items():
                            stats[3][k][1].append(v)

            for depth in (2, 3):
                total = sum(w for w, _ in stats[depth].values())
                for pred, (wins, coss) in sorted(stats[depth].items()):
                    collapse_rows.append({
                        "model": model, "wave": wave, "depth": depth,
                        "predictor": pred, "wins": wins, "n_total": total,
                        "mean_cos": float(np.mean(coss)) if coss else None,
                    })
                if total:
                    print(f"[collapse] {model} W{wave} d{depth}: " +
                          ", ".join(f"{p}={w/total:.1%}"
                                    for p, (w, _) in sorted(stats[depth].items())),
                          flush=True)

    def merge_save(name, new_rows):
        path = os.path.join(OUT_DIR, name)
        old = []
        if os.path.exists(path):
            with open(path) as f:
                old = json.load(f)
        new_models = {r["model"] for r in new_rows}
        merged = [r for r in old if r["model"] not in new_models] + new_rows
        with open(path, "w") as f:
            json.dump(merged, f, indent=1)

    merge_save(f"spine_results{SUFFIX}.json", spine_rows)
    merge_save(f"collapse_results{SUFFIX}.json", collapse_rows)

    # ----- pooled summaries -----
    print("\n=== SPINE SUMMARY (pooled over waves, cell-weighted) ===")
    for model in SPINE_MODELS:
        for depth in (1, 2, 3):
            rows = [r for r in spine_rows if r["model"] == model and r["depth"] == depth]
            if not rows:
                continue
            n = sum(r["n_cells"] for r in rows)
            m = sum(r["mean_tv"] * r["n_cells"] for r in rows) / n
            print(f"  {model:28s} depth {depth}: mean TV = {m:.4f}  (n={n})")

    print("\n=== COLLAPSE SHARE SUMMARY (pooled over waves) ===")
    for model in COLLAPSE_MODELS:
        for depth in (2, 3):
            rows = [r for r in collapse_rows if r["model"] == model and r["depth"] == depth]
            total = sum(r["wins"] for r in rows)
            if not total:
                continue
            agg_w = defaultdict(int)
            agg_c = defaultdict(list)
            for r in rows:
                agg_w[r["predictor"]] += r["wins"]
                if r["mean_cos"] is not None:
                    agg_c[r["predictor"]].append((r["mean_cos"], r["n_total"]))
            for pred in sorted(agg_w):
                cs = agg_c[pred]
                mc = (sum(c * n for c, n in cs) / sum(n for _, n in cs)) if cs else float("nan")
                print(f"  {model:12s} depth {depth}: {pred:12s} wins {agg_w[pred]/total:6.1%}  mean_cos={mc:.4f}")


if __name__ == "__main__":
    main()
