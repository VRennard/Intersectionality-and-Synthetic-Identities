"""
Collapse-direction metric.

Depth 2 — for each pair profile {A, B} x question where best-single wins:
  - winner   : feature whose single bias best matches the pair bias (max cosine)
  - human_dom: feature whose single subgroup deviates most from population (TV)
  - agree    : winner == human_dom

Depth 3 — for each triple profile {A, B, C} x question:
  - When best-single wins over best-pair and additive:
      winner / human_dom are the best individual feature (same logic as d2)
  - When best-pair wins over best-single and additive:
      winner_pair / human_dom_pair are the best feature-pair

CLI: collapse_direction.py [--weighted] model1 model2 ...
Output: verification/collapse_direction{_weighted}.json + console summary.
"""

import itertools
import os, sys, json, pickle
from collections import defaultdict

import numpy as np

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(BASE, "advanced_bias_analysis"))
import utils

OUT_DIR   = os.path.join(BASE, "verification")
CACHE_DIR = os.path.join(OUT_DIR, "cache")

MODELS = ["gpt-4o-mini", "gemma2_9b", "gpt-4o", "claude-haiku-4-5-20251001"]
WEIGHTED = "--weighted" in sys.argv
_args = [a for a in sys.argv[1:] if a != "--weighted"]
if _args:
    MODELS = _args
if WEIGHTED:
    utils.USE_WEIGHTS = True
SUFFIX = "_weighted" if WEIGHTED else ""


def model_waves(model):
    d = os.path.join(BASE, "data", "results", model)
    return sorted(
        (fn[1:-6] for fn in os.listdir(d) if fn.startswith("W") and fn.endswith(".jsonl")),
        key=int,
    )


def load_human(wave):
    with open(os.path.join(CACHE_DIR, f"human_W{wave}{SUFFIX}.pkl"), "rb") as f:
        human, _ = pickle.load(f)
    return human


def _dim(feat_tuple):
    """Return the dimension name from a (dim, val) tuple."""
    return feat_tuple[0]


def main():
    # rows carry a 'depth' field; depth-2 rows also have dimA/dimB;
    # depth-3 rows have dims (list of 3), collapse_type ('single'|'pair'),
    # winner, human_dom, agree.
    rows = []
    for model in MODELS:
        utils.MODEL_TAG = model
        for wave in model_waves(model):
            cache = os.path.join(CACHE_DIR, f"human_W{wave}{SUFFIX}.pkl")
            if not os.path.exists(cache):
                print(f"{model} W{wave}: no human cache, skipped", flush=True)
                continue
            human = load_human(wave)
            llm, qmeta = utils.build_llm_index(wave, max_level=3)
            h_avg = human.get(utils.AVG_PROFILE, {})

            counts = defaultdict(lambda: {"n": 0, "collapsed": 0, "agree": 0})

            for profile, qdists in llm.items():
                depth = len(profile)
                if depth not in (2, 3):
                    continue

                feats   = sorted(profile)          # list of (dim, val) tuples
                singles = [frozenset([f]) for f in feats]
                dims    = [_dim(f) for f in feats]

                if depth == 3:
                    pair_profiles = [frozenset(c) for c in itertools.combinations(feats, 2)]
                    pair_dims     = [tuple(sorted(_dim(f) for f in p)) for p in pair_profiles]

                hp   = human.get(profile, {})
                h_singles = [human.get(s, {}) for s in singles]
                l_singles = [llm.get(s, {})   for s in singles]

                if depth == 3:
                    h_pairs = [human.get(p, {}) for p in pair_profiles]
                    l_pairs = [llm.get(p, {})   for p in pair_profiles]

                key2 = f"d{depth}"
                counts[key2]["n"] += len(qdists)   # rough tally

                for qid, lp in qdists.items():
                    h_pop = h_avg.get(qid)
                    if h_pop is None:
                        continue
                    h_p = hp.get(qid)
                    h_s = [hs.get(qid) for hs in h_singles]
                    l_s = [ls.get(qid) for ls in l_singles]
                    if h_p is None or any(x is None for x in h_s + l_s):
                        continue
                    n = len(lp)
                    if not all(len(x) == n for x in [h_p, h_pop] + h_s + l_s):
                        continue

                    e_p      = lp   - h_p
                    e_singles = [l_s[i] - h_s[i] for i in range(len(feats))]
                    cos_singles = [utils.cosine_sim(e_p, e) for e in e_singles]
                    cos_add     = utils.cosine_sim(e_p, sum(e_singles))
                    best_single_cos = max(cos_singles)

                    if depth == 2:
                        if best_single_cos < cos_add:
                            continue   # additive wins
                        winner_dim = dims[int(cos_singles[1] > cos_singles[0])]
                        tv_s = [utils.tv(h_s[i], h_pop) for i in range(2)]
                        human_dom = dims[int(tv_s[1] > tv_s[0])]
                        agree = winner_dim == human_dom
                        counts[key2]["collapsed"] += 1
                        counts[key2]["agree"]     += agree
                        rows.append({
                            "depth": 2, "model": model, "wave": wave,
                            "dimA": dims[0], "dimB": dims[1],
                            "collapse_type": "single",
                            "winner": winner_dim, "human_dom": human_dom,
                            "agree": agree,
                        })

                    else:  # depth == 3
                        # gather valid pair biases
                        h_p2 = [h_pairs[i].get(qid) for i in range(3)]
                        l_p2 = [l_pairs[i].get(qid) for i in range(3)]
                        valid_pairs = [(i, l_p2[i] - h_p2[i])
                                       for i in range(3)
                                       if h_p2[i] is not None and l_p2[i] is not None
                                       and len(l_p2[i]) == n]
                        if not valid_pairs:
                            continue
                        pair_cos  = [(i, utils.cosine_sim(e_p, ep2)) for i, ep2 in valid_pairs]
                        best_pair_idx, best_pair_cos = max(pair_cos, key=lambda x: x[1])

                        winner_type = max(
                            [("single", best_single_cos),
                             ("pair",   best_pair_cos),
                             ("add",    cos_add)],
                            key=lambda x: x[1]
                        )[0]
                        if winner_type == "add":
                            continue   # additive wins: skip

                        counts[key2]["collapsed"] += 1

                        if winner_type == "single":
                            best_s_idx  = int(np.argmax(cos_singles))
                            winner_dim  = dims[best_s_idx]
                            tv_s        = [utils.tv(h_s[i], h_pop) for i in range(3)]
                            human_dom   = dims[int(np.argmax(tv_s))]
                            agree       = winner_dim == human_dom
                            counts[key2]["agree"] += agree
                            rows.append({
                                "depth": 3, "model": model, "wave": wave,
                                "dims": dims,
                                "collapse_type": "single",
                                "winner": winner_dim, "human_dom": human_dom,
                                "agree": agree,
                            })
                        else:  # best_pair wins
                            winner_pair = pair_dims[best_pair_idx]
                            # human-dominant pair: highest TV of the actual pair human dist
                            tv_pairs = []
                            for i, ep2 in valid_pairs:
                                h_pair_dist = h_p2[i]
                                tv_pairs.append((i, utils.tv(h_pair_dist, h_pop)))
                            human_dom_pair = pair_dims[max(tv_pairs, key=lambda x: x[1])[0]]
                            agree = winner_pair == human_dom_pair
                            counts[key2]["agree"] += agree
                            rows.append({
                                "depth": 3, "model": model, "wave": wave,
                                "dims": dims,
                                "collapse_type": "pair",
                                "winner": str(winner_pair), "human_dom": str(human_dom_pair),
                                "agree": agree,
                            })

            for key, c in sorted(counts.items()):
                if c["collapsed"]:
                    print(f"{model} W{wave} {key}: collapsed={c['collapsed']} "
                          f"agree={c['agree']/c['collapsed']:.1%}", flush=True)

    path = os.path.join(OUT_DIR, f"collapse_direction{SUFFIX}.json")
    old = []
    if os.path.exists(path):
        with open(path) as f:
            old = json.load(f)
    new_models = {r["model"] for r in rows}
    rows = [r for r in old if r["model"] not in new_models] + rows
    with open(path, "w") as f:
        json.dump(rows, f)

    # ── summaries ──────────────────────────────────────────────────────────
    print("\n=== COLLAPSE DIRECTION SUMMARY ===")
    for model in MODELS:
        for depth in (2, 3):
            rs = [r for r in rows if r["model"] == model and r["depth"] == depth]
            if not rs:
                continue
            agree_all = np.mean([r["agree"] for r in rs])
            n_chance  = 2 if depth == 2 else 3
            print(f"\n{model}  depth={depth}  (collapsed n={len(rs)}, "
                  f"chance~{100/n_chance:.0f}%)")
            print(f"  overall agreement with human-dominant: {agree_all:.1%}")

            # per collapse_type breakdown
            for ctype in ("single", "pair"):
                sub = [r for r in rs if r["collapse_type"] == ctype]
                if not sub:
                    continue
                ag = np.mean([r["agree"] for r in sub])
                print(f"  [{ctype}]  n={len(sub)}  agree={ag:.1%}")

                # per-dimension keep vs should (single collapse only)
                if ctype == "single":
                    keep = defaultdict(int)
                    should = defaultdict(int)
                    contested = defaultdict(int)
                    for r in sub:
                        all_dims = ([r["dimA"], r["dimB"]] if depth == 2
                                    else r["dims"])
                        for d in all_dims:
                            contested[d] += 1
                        keep[r["winner"]] += 1
                        should[r["human_dom"]] += 1
                    print(f"  {'dimension':20s} {'LLM keeps':>10s} "
                          f"{'human says':>11s} {'delta':>7s}")
                    for d in sorted(contested,
                                    key=lambda d: keep[d] / contested[d],
                                    reverse=True):
                        k = keep[d] / contested[d]
                        s = should[d] / contested[d]
                        print(f"  {d:20s} {k:10.1%} {s:11.1%} {k - s:+7.1%}")

                    party = [r for r in sub
                             if "Political Party" in (
                                 [r.get("dimA"), r.get("dimB")] if depth == 2
                                 else r.get("dims", []))]
                    party_not_dom = [r for r in party
                                     if r["human_dom"] != "Political Party"]
                    if party_not_dom:
                        kept = np.mean([r["winner"] == "Political Party"
                                        for r in party_not_dom])
                        print(f"  Party kept when humans say other dim matters: "
                              f"{kept:.1%}  (n={len(party_not_dom)})")


if __name__ == "__main__":
    main()
