"""
Human-variation rigor items 2 and 3 (NHB plan).

(2) Noise-corrected human distinctiveness by depth.
    Does the real distinctiveness of subgroups grow as identities
    intersect, after removing sampling noise? Per (cell, question):
      l2sq_obs  = ||p_g - p_pop||^2
      noisevar  = sum_i p_g_i(1-p_g_i)/n_g + sum_i p_pop_i(1-p_pop_i)/n_pop
      l2sq_corr = l2sq_obs - noisevar          (unbiased for ||true diff||^2)
    plus raw TV for reference. Aggregated by depth (1/2/3).

(3) Within-group heterogeneity (flattening in Wang et al.'s sense).
    Per (cell, question) where the LLM simulated the same profile:
      dH = H(p_llm) - H_MM(p_human)
    with Miller-Madow correction H_MM = H_plugin + (K-1)/(2n) on the human
    side. dH < 0 = simulated answer distribution too peaked (flattened
    within-group variation). Aggregated by depth and dimension.

Model for (3): gpt-4o-mini. Waves: all except W49.
Outputs: verification/human_rigor_2_3.json
"""

import os, sys, json, itertools
from collections import defaultdict

import numpy as np
import pandas as pd

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(BASE, "advanced_bias_analysis"))
import utils

MODEL = "gpt-4o-mini"
EXCL  = {"49"}


def wave_list():
    d = os.path.join(BASE, "human_resp")
    out = []
    for fn in sorted(os.listdir(d)):
        if fn.startswith("American_Trends_Panel_W"):
            w = fn.rsplit("W", 1)[1]
            if w not in EXCL and os.path.exists(os.path.join(d, fn, "responses.csv")):
                out.append(w)
    return sorted(out, key=int)


def load_options(wave):
    path = os.path.join(BASE, "data", "responses", f"survey_responses_W{wave}.json")
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return {q["question_id"]: [r["option"] for r in q["responses"]] for q in data}


def dist_n(series, opts):
    vc = series.value_counts()
    counts = np.array([vc.get(o, 0) for o in opts], dtype=float)
    n = counts.sum()
    if n < utils.MIN_HUMAN_N:
        return None, 0
    return counts / n, int(n)


def entropy(p):
    p = p[p > 0]
    return float(-(p * np.log(p)).sum())


def cell_masks(df):
    """(depth, profile, mask) for singles/pairs/triples passing filters."""
    val_masks = {}
    for dim, col in utils.DIM_TO_COL.items():
        if col not in df.columns:
            continue
        for val in utils.DIM_VALUES[dim]:
            if val in utils.IGNORE_VALUES.get(dim, set()):
                continue
            m = (df[col] == val).to_numpy()
            if m.sum() >= utils.MIN_HUMAN_N:
                val_masks[(dim, val)] = m
    keys = sorted(val_masks, key=lambda kv: (utils.DIMS.index(kv[0]), kv[1]))
    for k in keys:
        yield 1, frozenset([k]), val_masks[k]
    for a, b in itertools.combinations(keys, 2):
        if a[0] == b[0]:
            continue
        m = val_masks[a] & val_masks[b]
        if m.sum() >= utils.MIN_HUMAN_N:
            yield 2, frozenset([a, b]), m
    for a, b, c in itertools.combinations(keys, 3):
        if len({a[0], b[0], c[0]}) < 3:
            continue
        m = val_masks[a] & val_masks[b] & val_masks[c]
        if m.sum() >= utils.MIN_HUMAN_N:
            yield 3, frozenset([a, b, c]), m


def main():
    utils.MODEL_TAG = MODEL
    # experiment 2 accumulators
    e2 = defaultdict(lambda: {"tv": [], "noise_tv": [], "l2sq": [], "l2sq_corr": []})
    # experiment 3 accumulators
    e3_depth = defaultdict(list)               # depth -> dH list
    e3_dim   = defaultdict(list)               # dim (singles only) -> dH list
    e3_hh    = defaultdict(list)               # depth -> human entropy
    e3_lh    = defaultdict(list)               # depth -> llm entropy

    for wave in wave_list():
        options = load_options(wave)
        csv = os.path.join(BASE, "human_resp",
                           f"American_Trends_Panel_W{wave}", "responses.csv")
        df = pd.read_csv(csv, low_memory=False)
        qids = [q for q in options if q in df.columns]

        llm, qmeta = utils.build_llm_index(wave, max_level=3)

        # population distributions + n per qid
        pop = {}
        for qid in qids:
            d, n = dist_n(df[qid].dropna(), options[qid])
            if d is not None:
                pop[qid] = (d, n)

        n_cells = 0
        for depth, profile, mask in cell_masks(df):
            sub = df.loc[mask]
            lq = llm.get(profile, {})
            n_cells += 1
            for qid in qids:
                if qid not in pop:
                    continue
                pg, ng = dist_n(sub[qid].dropna(), options[qid])
                if pg is None:
                    continue
                ppop, npop = pop[qid]
                if len(pg) != len(ppop):
                    continue
                # ---- (2) distinctiveness ----
                diff = pg - ppop
                l2sq = float(diff @ diff)
                noisevar = float((pg * (1 - pg)).sum() / ng
                                 + (ppop * (1 - ppop)).sum() / npop)
                noise_tv = float(0.5 * np.sqrt(2 / (np.pi * ng))
                                 * np.sqrt(pg * (1 - pg)).sum())
                a = e2[depth]
                a["tv"].append(utils.tv(pg, ppop))
                a["noise_tv"].append(noise_tv)
                a["l2sq"].append(l2sq)
                a["l2sq_corr"].append(l2sq - noisevar)
                # ---- (3) entropy flattening ----
                lp = lq.get(qid)
                if lp is not None and len(lp) == len(pg):
                    K = (pg > 0).sum()
                    h_h = entropy(pg) + (K - 1) / (2 * ng)   # Miller-Madow
                    h_l = entropy(lp)
                    e3_depth[depth].append(h_l - h_h)
                    e3_hh[depth].append(h_h)
                    e3_lh[depth].append(h_l)
                    if depth == 1:
                        (dim, val), = profile
                        e3_dim[dim].append(h_l - h_h)
        print(f"W{wave}: {n_cells} cells", flush=True)

    out = {"distinctiveness": {}, "entropy": {}}
    print("\n=== (2) HUMAN DISTINCTIVENESS BY DEPTH (noise-corrected) ===")
    print(f"{'depth':>6} {'raw TV':>8} {'noiseTV':>8} {'L2^2 obs':>9} "
          f"{'L2^2 corr':>10} {'RMS corr':>9} {'n':>9}")
    for d in sorted(e2):
        a = e2[d]
        corr = float(np.mean(a["l2sq_corr"]))
        row = {
            "tv": float(np.mean(a["tv"])),
            "noise_tv": float(np.mean(a["noise_tv"])),
            "l2sq": float(np.mean(a["l2sq"])),
            "l2sq_corr": corr,
            "rms_corr": float(np.sqrt(max(corr, 0))),
            "n": len(a["tv"]),
        }
        out["distinctiveness"][d] = row
        print(f"{d:>6} {row['tv']:>8.4f} {row['noise_tv']:>8.4f} "
              f"{row['l2sq']:>9.5f} {row['l2sq_corr']:>10.5f} "
              f"{row['rms_corr']:>9.4f} {row['n']:>9}")

    print("\n=== (3) ENTROPY FLATTENING: H(LLM) - H_MM(human) ===")
    for d in sorted(e3_depth):
        v = np.array(e3_depth[d])
        row = {"mean_dH": float(v.mean()),
               "share_llm_peakier": float((v < 0).mean()),
               "mean_H_human": float(np.mean(e3_hh[d])),
               "mean_H_llm": float(np.mean(e3_lh[d])),
               "n": len(v)}
        out["entropy"][d] = row
        print(f"  depth {d}: dH={row['mean_dH']:+.4f} nats  "
              f"LLM peakier in {row['share_llm_peakier']:.1%}  "
              f"(H_human={row['mean_H_human']:.3f}, H_llm={row['mean_H_llm']:.3f}, "
              f"n={row['n']})")
    print("  by dimension (singles):")
    for dim in sorted(e3_dim, key=lambda d: np.mean(e3_dim[d])):
        v = np.array(e3_dim[dim])
        print(f"    {dim:18s} dH={v.mean():+.4f}  peakier {np.mean(v<0):.0%}")

    with open(os.path.join(BASE, "verification", "human_rigor_2_3.json"), "w") as f:
        json.dump(out, f, indent=1)


if __name__ == "__main__":
    main()
