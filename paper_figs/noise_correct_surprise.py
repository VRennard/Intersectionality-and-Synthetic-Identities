"""
Noise-corrected version of the Fig 6 analysis.

Observed TV(LLM, human) is inflated by ground-truth sampling noise, which is
larger for small human cells — and counter-stereotypical pairs have small
cells, confounding the surprise->error relationship.

Correction: for each (pair profile x question) cell with human sample size n
and empirical distribution p, the expected TV between the empirical estimate
and the truth under multinomial sampling is approximately
    E[TV] ~= 0.5 * sqrt(2/(pi*n)) * sum_i sqrt(p_i * (1 - p_i))
(normal approximation to each |p_hat_i - p_i|).

excess_tv = observed_tv - E[TV]  per cell, aggregated per pair profile.

Outputs surprise_tv_corrected.json and prints raw vs corrected correlations,
plus a restriction robustness check (cells with n >= 100 only).

Model: gpt-4o-mini, waves excl W49.
"""

import os, sys, json, pickle
from collections import defaultdict

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(BASE, "advanced_bias_analysis"))
import utils

MODEL = "gpt-4o-mini"
EXCL  = {"49"}
PKL   = os.path.join(BASE, "verification", "cache")


def waves_list():
    d = os.path.join(BASE, "data", "results", MODEL)
    return sorted((fn[1:-6] for fn in os.listdir(d)
                   if fn.startswith("W") and fn.endswith(".jsonl") and fn[1:-6] not in EXCL),
                  key=int)


def expected_noise_tv(p, n):
    return float(0.5 * np.sqrt(2.0 / (np.pi * n)) * np.sum(np.sqrt(p * (1 - p))))


def main():
    utils.MODEL_TAG = MODEL
    acc = defaultdict(lambda: {"obs": [], "exp": [], "ns": []})

    for wave in waves_list():
        with open(os.path.join(PKL, f"human_W{wave}.pkl"), "rb") as f:
            human, _ = pickle.load(f)
        llm, qmeta = utils.build_llm_index(wave, max_level=2)

        csv = os.path.join(BASE, "human_resp",
                           f"American_Trends_Panel_W{wave}", "responses.csv")
        df = pd.read_csv(csv, low_memory=False)

        # masks for the pair profiles present in the LLM data
        pair_profiles = [p for p in llm if len(p) == 2]
        masks = {}
        for profile in pair_profiles:
            ok = True
            m = np.ones(len(df), dtype=bool)
            for dim, val in profile:
                col = utils.DIM_TO_COL.get(dim)
                if col not in df.columns:
                    ok = False
                    break
                m &= (df[col] == val).to_numpy()
            if ok and m.sum() >= utils.MIN_HUMAN_N:
                masks[profile] = m

        for profile, m in masks.items():
            hq = human.get(profile, {})
            sub = df.loc[m]
            for qid, lp in llm[profile].items():
                hd = hq.get(qid)
                if hd is None or len(hd) != len(lp) or qid not in df.columns:
                    continue
                opts = qmeta[qid]["options"]
                vc = sub[qid].dropna().value_counts()
                n = int(sum(vc.get(o, 0) for o in opts))
                if n < utils.MIN_HUMAN_N:
                    continue
                a = acc[profile]
                a["obs"].append(utils.tv(lp, hd))
                a["exp"].append(expected_noise_tv(hd, n))
                a["ns"].append(n)
        print(f"W{wave}: {len(masks)} pair cells", flush=True)

    # merge with surprise ratios
    with open(os.path.join(HERE, "surprise_tv.json")) as f:
        sur = {frozenset([(r["dimA"], r["valA"]), (r["dimB"], r["valB"])]): r
               for r in json.load(f)}

    out = []
    for profile, a in acc.items():
        s = sur.get(profile)
        if s is None or not a["obs"]:
            continue
        out.append({
            **{k: s[k] for k in ("dimA", "valA", "dimB", "valB", "surprise", "human_n")},
            "mean_tv": float(np.mean(a["obs"])),
            "mean_noise": float(np.mean(a["exp"])),
            "excess_tv": float(np.mean(a["obs"]) - np.mean(a["exp"])),
            "mean_cell_n": float(np.mean(a["ns"])),
            "n_cells": len(a["obs"]),
        })
    with open(os.path.join(HERE, "surprise_tv_corrected.json"), "w") as f:
        json.dump(out, f)

    # ---- analysis ----
    from scipy.stats import spearmanr
    x   = np.array([r["surprise"] for r in out])
    raw = np.array([r["mean_tv"] for r in out])
    exc = np.array([r["excess_tv"] for r in out])
    cn  = np.array([r["mean_cell_n"] for r in out])
    ok  = x > 0
    print(f"\n=== NOISE-CORRECTED SURPRISE ANALYSIS ({len(out)} pair profiles) ===")
    print(f"  spearman(surprise, raw TV):    rho={spearmanr(x[ok], raw[ok]).statistic:+.3f} "
          f"p={spearmanr(x[ok], raw[ok]).pvalue:.3f}")
    print(f"  spearman(surprise, excess TV): rho={spearmanr(x[ok], exc[ok]).statistic:+.3f} "
          f"p={spearmanr(x[ok], exc[ok]).pvalue:.3f}")
    print(f"  spearman(cell n, raw TV):      rho={spearmanr(cn[ok], raw[ok]).statistic:+.3f}  "
          f"(noise confound size)")
    for lab, lo, hi in (("surprise<0.5", 0, 0.5), ("0.5-2", 0.5, 2), (">=2", 2, 99)):
        sel = ok & (x >= lo) & (x < hi)
        print(f"  {lab:14s} n={sel.sum():3d}  raw={raw[sel].mean():.3f}  "
              f"excess={exc[sel].mean():.3f}  cell_n={cn[sel].mean():.0f}")
    big = ok & (cn >= 100)
    print(f"  [robustness n>=100 only, {big.sum()} profiles] "
          f"spearman(surprise, raw)={spearmanr(x[big], raw[big]).statistic:+.3f} "
          f"p={spearmanr(x[big], raw[big]).pvalue:.3f}")
    tail = sorted([r for r in out if r["surprise"] < 0.3 and r["surprise"] > 0],
                  key=lambda r: r["surprise"])
    print("  tail profiles (surprise<0.3):")
    for r in tail:
        print(f"    {r['valA']} x {r['valB']}: surprise={r['surprise']:.2f} "
              f"raw={r['mean_tv']:.3f} noise={r['mean_noise']:.3f} "
              f"excess={r['excess_tv']:.3f} cell_n={r['mean_cell_n']:.0f}")


if __name__ == "__main__":
    main()
