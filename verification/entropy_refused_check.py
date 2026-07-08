"""
Spot check for experiment 3: is the LLM's higher entropy an artifact of
allocating mass to "Refused" (which real respondents rarely choose)?

Recompute dH = H(p_llm) - H_MM(p_human) on W26 + W43, singles + pairs,
(a) as before, (b) excluding Refused-type options and renormalizing.
Also report mean Refused mass, human vs LLM.
"""

import os, sys, json, itertools
from collections import defaultdict

import numpy as np
import pandas as pd

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(BASE, "advanced_bias_analysis"))
import utils

MODEL = "gpt-4o-mini"
WAVES = ["26", "43"]


def load_options(wave):
    with open(os.path.join(BASE, "data", "responses",
                           f"survey_responses_W{wave}.json"), encoding="utf-8") as f:
        return {q["question_id"]: [r["option"] for r in q["responses"]]
                for q in json.load(f)}


def dist_n(series, opts):
    vc = series.value_counts()
    counts = np.array([vc.get(o, 0) for o in opts], dtype=float)
    n = counts.sum()
    return (counts / n, int(n)) if n >= utils.MIN_HUMAN_N else (None, 0)


def entropy(p):
    p = p[p > 0]
    return float(-(p * np.log(p)).sum())


def main():
    utils.MODEL_TAG = MODEL
    dh_all, dh_norefuse = [], []
    mass_h, mass_l = [], []
    for wave in WAVES:
        options = load_options(wave)
        df = pd.read_csv(os.path.join(BASE, "human_resp",
                                      f"American_Trends_Panel_W{wave}",
                                      "responses.csv"), low_memory=False)
        llm, qmeta = utils.build_llm_index(wave, max_level=2)
        qids = [q for q in options if q in df.columns]

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

        def profiles():
            for k, m in val_masks.items():
                yield frozenset([k]), m
            for a, b in itertools.combinations(sorted(val_masks), 2):
                if a[0] == b[0]:
                    continue
                m = val_masks[a] & val_masks[b]
                if m.sum() >= utils.MIN_HUMAN_N:
                    yield frozenset([a, b]), m

        for profile, mask in profiles():
            lq = llm.get(profile, {})
            if not lq:
                continue
            sub = df.loc[mask]
            for qid, lp in lq.items():
                if qid not in options:
                    continue
                opts = options[qid]
                pg, ng = dist_n(sub[qid].dropna(), opts)
                if pg is None or len(pg) != len(lp):
                    continue
                K = (pg > 0).sum()
                dh_all.append(entropy(lp) - (entropy(pg) + (K - 1) / (2 * ng)))
                keep = [i for i, o in enumerate(opts) if "refus" not in o.lower()]
                if len(keep) < 2 or len(keep) == len(opts):
                    continue
                refuse = [i for i in range(len(opts)) if i not in keep]
                mass_h.append(float(sum(pg[i] for i in refuse)))
                mass_l.append(float(sum(lp[i] for i in refuse)))
                pg2 = pg[keep]; lp2 = np.array(lp)[keep]
                if pg2.sum() <= 0 or lp2.sum() <= 0:
                    continue
                pg2, lp2 = pg2 / pg2.sum(), lp2 / lp2.sum()
                K2 = (pg2 > 0).sum()
                dh_norefuse.append(entropy(lp2) - (entropy(pg2) + (K2 - 1) / (2 * ng)))
        print(f"W{wave} done", flush=True)

    print(f"\ndH with Refused:    {np.mean(dh_all):+.4f}  (n={len(dh_all)})")
    print(f"dH without Refused: {np.mean(dh_norefuse):+.4f}  (n={len(dh_norefuse)})")
    print(f"mean Refused mass: human={np.mean(mass_h):.4f}  llm={np.mean(mass_l):.4f}")


if __name__ == "__main__":
    main()
