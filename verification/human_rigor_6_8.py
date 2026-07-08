"""
Human-rigor items 6, 7, 8.

(6) Noise-estimator validation.
    (a) Analytic relationship: E[split-half TV] = 2 x E[TV(p_hat_n, p)].
        Verify on real cells (3 waves, n>=40): observed split-half TV vs
        analytic prediction sqrt(2/(pi*n)) * sum_i sqrt(p_i(1-p_i)).
    (b) Simulation: accuracy of the one-sample analytic E[TV] at small n
        for representative distributions (n in {20,40,100,400}, several p).

(7) Split-sample confirmation: every headline number recomputed on odd vs
    even waves (by chronological index). Pure re-aggregation of stored JSONs.

(8) Reliability table: model run-to-run TV (3 stochastic W26 runs, pairwise,
    matched profile x question, by depth) vs human split-half TV on W26.

Output: verification/human_rigor_6_8.json
"""

import os, sys, json, itertools
from collections import defaultdict

import numpy as np
import pandas as pd

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(BASE, "advanced_bias_analysis"))
import utils

HERE  = os.path.join(BASE, "verification")
EXCL  = {"49"}
RNG   = np.random.default_rng(11)
WAVES_6A = ["27", "43", "82"]

out = {}


# ---------------------------------------------------------------- (6a)
def item_6a():
    print("=== (6a) split-half observed vs analytic (n>=40 cells) ===", flush=True)
    res = defaultdict(lambda: {"obs": [], "pred": []})
    for wave in WAVES_6A:
        with open(os.path.join(BASE, "data", "responses",
                               f"survey_responses_W{wave}.json"), encoding="utf-8") as f:
            options = {q["question_id"]: [r["option"] for r in q["responses"]]
                       for q in json.load(f)}
        df = pd.read_csv(os.path.join(BASE, "human_resp",
                                      f"American_Trends_Panel_W{wave}",
                                      "responses.csv"), low_memory=False)
        qids = [q for q in options if q in df.columns]

        val_masks = {}
        for dim, col in utils.DIM_TO_COL.items():
            if col not in df.columns:
                continue
            for val in utils.DIM_VALUES[dim]:
                if val in utils.IGNORE_VALUES.get(dim, set()):
                    continue
                m = (df[col] == val).to_numpy()
                if m.sum() >= 40:
                    val_masks[(dim, val)] = m

        def cells():
            for k, m in val_masks.items():
                yield 1, m
            ks = sorted(val_masks)
            for a, b in itertools.combinations(ks, 2):
                if a[0] == b[0]:
                    continue
                m = val_masks[a] & val_masks[b]
                if m.sum() >= 40:
                    yield 2, m

        for depth, mask in cells():
            idx = np.flatnonzero(mask)
            perm = RNG.permutation(idx)
            ha, hb = perm[: len(perm) // 2], perm[len(perm) // 2:]
            for qid in qids[:30]:
                col = df[qid]
                opts = options[qid]

                def dn(rows):
                    vc = col.iloc[rows].dropna().value_counts()
                    c = np.array([vc.get(o, 0) for o in opts], dtype=float)
                    n = c.sum()
                    return (c / n, n) if n >= 20 else (None, 0)

                da, na = dn(ha)
                db, nb = dn(hb)
                dfull, nfull = dn(idx)
                if da is None or db is None or dfull is None:
                    continue
                obs = utils.tv(da, db)
                pred = float(np.sqrt(2 / (np.pi * nfull))
                             * np.sqrt(dfull * (1 - dfull)).sum())
                res[depth]["obs"].append(obs)
                res[depth]["pred"].append(pred)
        print(f"  W{wave} done", flush=True)

    out["6a"] = {}
    for d in sorted(res):
        o = np.mean(res[d]["obs"]); p = np.mean(res[d]["pred"])
        out["6a"][d] = {"obs_splithalf": float(o), "analytic_pred": float(p),
                        "ratio": float(o / p), "n": len(res[d]["obs"])}
        print(f"  depth {d}: observed={o:.4f} analytic={p:.4f} "
              f"ratio={o/p:.3f} (n={len(res[d]['obs'])})", flush=True)


# ---------------------------------------------------------------- (6b)
def item_6b():
    print("\n=== (6b) one-sample analytic E[TV] vs simulation ===", flush=True)
    dists = {
        "uniform4":  np.array([.25, .25, .25, .25]),
        "skewed4":   np.array([.6, .25, .1, .05]),
        "typical5":  np.array([.28, .45, .15, .10, .02]),
        "peaked5":   np.array([.8, .1, .05, .04, .01]),
    }
    out["6b"] = {}
    for name, p in dists.items():
        for n in (20, 40, 100, 400):
            draws = RNG.multinomial(n, p, size=20000) / n
            sim = float(0.5 * np.abs(draws - p).sum(axis=1).mean())
            ana = float(0.5 * np.sqrt(2 / (np.pi * n)) * np.sqrt(p * (1 - p)).sum())
            out["6b"][f"{name}_n{n}"] = {"sim": sim, "analytic": ana,
                                         "ratio": ana / sim}
            print(f"  {name:9s} n={n:3d}: sim={sim:.4f} analytic={ana:.4f} "
                  f"ratio={ana/sim:.3f}", flush=True)


# ---------------------------------------------------------------- (7)
def item_7():
    print("\n=== (7) split-sample: odd vs even waves ===", flush=True)
    all_waves = ["26", "27", "29", "32", "34", "36", "41",
                 "42", "43", "45", "50", "54", "82", "92"]
    odd  = set(all_waves[0::2])
    even = set(all_waves[1::2])
    out["7"] = {"odd_waves": sorted(odd, key=int), "even_waves": sorted(even, key=int)}

    with open(os.path.join(HERE, "collapse_results.json")) as f:
        crows = [r for r in json.load(f) if r["wave"] not in EXCL]
    for model in ("gpt-4o-mini", "gemma2_9b"):
        for half, wset in (("odd", odd), ("even", even)):
            for depth, pred in ((2, "best_single"), (3, "additive")):
                w = sum(r["wins"] for r in crows
                        if r["model"] == model and r["depth"] == depth
                        and r["wave"] in wset and r["predictor"] == pred)
                t = sum(r["wins"] for r in crows
                        if r["model"] == model and r["depth"] == depth
                        and r["wave"] in wset)
                if t:
                    key = f"{model}_d{depth}_{pred}_{half}"
                    out["7"][key] = w / t
                    print(f"  {model:12s} d{depth} {pred:12s} {half:4s}: {w/t:.3f}",
                          flush=True)

    with open(os.path.join(HERE, "collapse_direction.json")) as f:
        drows = [r for r in json.load(f) if r["wave"] not in EXCL]
    for model in ("gpt-4o-mini", "gpt-4o", "claude-haiku-4-5-20251001"):
        rs = [r for r in drows if r["model"] == model]
        for half, wset in (("odd", odd), ("even", even)):
            sub = [r for r in rs if r["wave"] in wset]
            agree = float(np.mean([r["agree"] for r in sub]))
            race = [r for r in sub if "Race" in (r["dimA"], r["dimB"])]
            d_race = (np.mean([r["winner"] == "Race" for r in race])
                      - np.mean([r["human_dom"] == "Race" for r in race]))
            out["7"][f"{model}_agree_{half}"] = agree
            out["7"][f"{model}_raceDelta_{half}"] = float(d_race)
            print(f"  {model:26s} {half:4s}: agree={agree:.4f} "
                  f"raceDelta={d_race:+.4f}", flush=True)

    with open(os.path.join(HERE, "spine_results.json")) as f:
        srows = [r for r in json.load(f) if r["wave"] not in EXCL]
    for half, wset in (("odd", odd), ("even", even)):
        for depth in (1, 2, 3):
            rs = [r for r in srows if r["model"] == "gpt-4o-mini"
                  and r["depth"] == depth and r["wave"] in wset]
            n = sum(r["n_cells"] for r in rs)
            m = sum(r["mean_tv"] * r["n_cells"] for r in rs) / n
            out["7"][f"spine_mini_d{depth}_{half}"] = float(m)
            print(f"  spine gpt-4o-mini d{depth} {half:4s}: {m:.4f}", flush=True)


# ---------------------------------------------------------------- (8)
def item_8():
    print("\n=== (8) model run-to-run reliability (stoch W26) vs human split-half ===",
          flush=True)
    runs = {}
    for i in (1, 2, 3):
        path = os.path.join(BASE, "data", "results", "stoch_test",
                            f"W26_run{i}.jsonl")
        d = {}
        with open(path, encoding="utf-8") as f:
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
                dist = np.array(r.get("response_distribution", []), dtype=float)
                if dist.sum() <= 0:
                    continue
                key = (tuple(sorted(r["demographics"])), r["question_id"])
                d[key] = (len(r["demographics"]), dist / dist.sum())
        runs[i] = d
        print(f"  run{i}: {len(d)} cells", flush=True)

    tvs = defaultdict(list)
    for a, b in itertools.combinations((1, 2, 3), 2):
        for key, (depth, da) in runs[a].items():
            v = runs[b].get(key)
            if v is None or len(v[1]) != len(da):
                continue
            tvs[depth].append(utils.tv(da, v[1]))

    with open(os.path.join(HERE, "noise_floor.json")) as f:
        nf = json.load(f)
    out["8"] = {}
    for d in sorted(tvs):
        model_rr = float(np.mean(tvs[d]))
        hw = [r for r in nf if r["wave"] == "26" and r["depth"] == d]
        human_sh = float(hw[0]["mean_tv"]) if hw else None
        out["8"][d] = {"model_run_to_run": model_rr,
                       "human_splithalf_W26": human_sh,
                       "n_pairs": len(tvs[d])}
        print(f"  depth {d}: model run-to-run TV={model_rr:.4f}  "
              f"human split-half (W26)={human_sh:.4f}  (n={len(tvs[d])})",
              flush=True)


if __name__ == "__main__":
    item_6a()
    item_6b()
    item_7()
    item_8()
    with open(os.path.join(HERE, "human_rigor_6_8.json"), "w") as f:
        json.dump(out, f, indent=1)
    print("\nwrote human_rigor_6_8.json")
