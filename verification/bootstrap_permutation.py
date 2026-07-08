"""
Human-rigor items 4 + 5: cluster bootstrap CIs and permutation null.

(4) Cluster bootstrap CIs:
    - depth-2 best-single share and depth-3 additive share per model:
      bootstrap over WAVES (the most conservative independent unit),
      using per-wave win counts from collapse_results.json. B=10000.
    - collapse-direction agreement and per-dimension keep-rate deltas:
      bootstrap over (wave x dimension-pair) clusters from
      collapse_direction.json. B=2000.

(5) Permutation null for direction agreement:
    chance is not exactly 50% because dominance rates differ by dimension.
    Shuffle human_dom labels within (wave, dimA, dimB) strata, B=500,
    -> null distribution of agreement, its mean (true chance level), and
    a permutation p-value for the observed agreement.

W49 excluded throughout. Output: verification/bootstrap_permutation.json
"""

import os, sys, json
from collections import defaultdict

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
# --weighted: read the survey-weighted collapse/direction JSONs and write
# bootstrap_permutation_weighted.json (weighted is the paper's primary).
SUFFIX = "_weighted" if "--weighted" in sys.argv else ""
EXCL = {"49"}
RNG  = np.random.default_rng(7)

DIMS = ["Age", "Gender", "Race", "Income", "Political Party", "Religion", "Education"]


def pct(v, lo=2.5, hi=97.5):
    return float(np.percentile(v, lo)), float(np.percentile(v, hi))


def collapse_share_cis():
    with open(os.path.join(HERE, f"collapse_results{SUFFIX}.json")) as f:
        rows = [r for r in json.load(f) if r["wave"] not in EXCL]
    out = {}
    models = sorted({r["model"] for r in rows})
    for model in models:
        for depth, pred in ((2, "best_single"), (3, "additive")):
            per_wave = defaultdict(lambda: [0, 0])   # wave -> [pred wins, total]
            for r in rows:
                if r["model"] != model or r["depth"] != depth:
                    continue
                per_wave[r["wave"]][1] += r["wins"]
                if r["predictor"] == pred:
                    per_wave[r["wave"]][0] += r["wins"]
            waves = [w for w, (a, t) in per_wave.items() if t > 0]
            if len(waves) < 5:
                continue
            arr = np.array([per_wave[w] for w in waves], dtype=float)
            obs = arr[:, 0].sum() / arr[:, 1].sum()
            boots = []
            for _ in range(10000):
                idx = RNG.integers(0, len(waves), len(waves))
                s = arr[idx]
                boots.append(s[:, 0].sum() / s[:, 1].sum())
            lo, hi = pct(boots)
            key = f"{model}_d{depth}_{pred}"
            out[key] = {"obs": float(obs), "ci": [lo, hi], "n_waves": len(waves)}
            print(f"  {model:18s} d{depth} {pred:12s}: {obs:.3f} "
                  f"[{lo:.3f}, {hi:.3f}]  ({len(waves)} waves)", flush=True)
    return out


def direction_stats():
    with open(os.path.join(HERE, f"collapse_direction{SUFFIX}.json")) as f:
        rows = [r for r in json.load(f) if r["wave"] not in EXCL]
    out = {}
    models = sorted({r["model"] for r in rows})
    for model in models:
        rs = [r for r in rows if r["model"] == model]
        if len(rs) < 1000:
            continue
        winner = np.array([r["winner"] for r in rs])
        dom    = np.array([r["human_dom"] for r in rs])
        agree  = (winner == dom).astype(float)
        strata = np.array([f"{r['wave']}|{r['dimA']}|{r['dimB']}" for r in rs])
        uniq, stratum_idx = np.unique(strata, return_inverse=True)
        obs = float(agree.mean())

        # ---- (4a) cluster bootstrap over strata: agreement ----
        cluster_rows = [np.flatnonzero(stratum_idx == i) for i in range(len(uniq))]
        boots = []
        for _ in range(2000):
            picked = RNG.integers(0, len(uniq), len(uniq))
            idx = np.concatenate([cluster_rows[i] for i in picked])
            boots.append(agree[idx].mean())
        lo, hi = pct(boots)

        # ---- (5) permutation null within strata ----
        order = np.argsort(stratum_idx, kind="stable")
        sorted_dom = dom[order]
        sorted_win = winner[order]
        bounds = np.searchsorted(stratum_idx[order],
                                 np.arange(len(uniq) + 1))
        null = []
        for _ in range(500):
            perm_dom = sorted_dom.copy()
            for s in range(len(uniq)):
                a, b = bounds[s], bounds[s + 1]
                if b - a > 1:
                    perm_dom[a:b] = perm_dom[a + RNG.permutation(b - a)]
            null.append(float((sorted_win == perm_dom).mean()))
        null = np.array(null)
        pval = float((null >= obs).mean())
        out[model] = {
            "agreement": obs, "agreement_ci": [lo, hi],
            "null_mean": float(null.mean()),
            "null_95": pct(null),
            "perm_p": pval if pval > 0 else 1.0 / 501,
            "n_cells": len(rs), "n_strata": len(uniq),
        }
        print(f"  {model:18s} agree={obs:.4f} CI=[{lo:.4f},{hi:.4f}]  "
              f"null={null.mean():.4f} [{pct(null)[0]:.4f},{pct(null)[1]:.4f}]  "
              f"perm p={'<0.002' if pval == 0 else f'{pval:.3f}'}", flush=True)

        # ---- (4b) keep-rate delta CIs per dimension (full-dim models) ----
        if model == "gemma2_9b":
            continue
        dimA = np.array([r["dimA"] for r in rs])
        dimB = np.array([r["dimB"] for r in rs])
        deltas = {}
        for d in DIMS:
            contested = (dimA == d) | (dimB == d)
            if contested.sum() < 1000:
                continue
            keep   = (winner == d)[contested].mean()
            should = (dom == d)[contested].mean()
            boots_d = []
            sub_clusters = [c for c in cluster_rows
                            if contested[c[0]]]
            for _ in range(2000):
                picked = RNG.integers(0, len(sub_clusters), len(sub_clusters))
                idx = np.concatenate([sub_clusters[i] for i in picked])
                boots_d.append((winner[idx] == d).mean() - (dom[idx] == d).mean())
            lo_d, hi_d = pct(boots_d)
            deltas[d] = {"delta": float(keep - should), "ci": [lo_d, hi_d]}
            sig = "" if lo_d <= 0 <= hi_d else " *"
            print(f"      {d:18s} delta={keep-should:+.4f} "
                  f"[{lo_d:+.4f},{hi_d:+.4f}]{sig}", flush=True)
        out[model]["keep_deltas"] = deltas
    return out


def main():
    print("=== (4) COLLAPSE SHARE CIs (wave-cluster bootstrap) ===")
    shares = collapse_share_cis()
    print("\n=== (4b/5) DIRECTION: agreement CI + permutation null + deltas ===")
    direction = direction_stats()
    with open(os.path.join(HERE, f"bootstrap_permutation{SUFFIX}.json"), "w") as f:
        json.dump({"collapse_shares": shares, "direction": direction}, f, indent=1)
    print(f"\nwrote bootstrap_permutation{SUFFIX}.json")


if __name__ == "__main__":
    main()
