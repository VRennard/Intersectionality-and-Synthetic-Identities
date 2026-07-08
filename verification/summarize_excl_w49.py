"""
Re-pool all verification results with W49 excluded (un-retried batch failures,
dimension-biased missingness — see PAPER_PLAN_v2.md section 7 item 2).

Reads the per-wave/per-cell JSONs written by verify_spine_collapse.py,
noise_floor.py, and collapse_direction.py. No recomputation.
Prints OLD (all waves) vs NEW (excl. W49) side by side.
"""

import os, json
from collections import defaultdict

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
EXCL = {"49"}


def jload(name):
    with open(os.path.join(HERE, name)) as f:
        return json.load(f)


def spine():
    rows = jload("spine_results.json")
    print("=== SPINE (mean TV, cell-weighted): all waves -> excl W49 ===")
    models = sorted({r["model"] for r in rows})
    for m in models:
        for d in (1, 2, 3):
            def pool(rs):
                n = sum(r["n_cells"] for r in rs)
                return (sum(r["mean_tv"] * r["n_cells"] for r in rs) / n, n) if n else (None, 0)
            all_rs  = [r for r in rows if r["model"] == m and r["depth"] == d]
            kept_rs = [r for r in all_rs if r["wave"] not in EXCL]
            if not all_rs:
                continue
            (m_all, n_all), (m_new, n_new) = pool(all_rs), pool(kept_rs)
            delta = m_new - m_all
            print(f"  {m:28s} d{d}: {m_all:.4f} -> {m_new:.4f}  ({delta:+.4f})  n={n_new}")


def collapse():
    rows = jload("collapse_results.json")
    print("\n=== COLLAPSE SHARES: all waves -> excl W49 ===")
    models = sorted({r["model"] for r in rows})
    for m in models:
        for d in (2, 3):
            def shares(rs):
                tot = sum(r["wins"] for r in rs)
                if not tot:
                    return None
                agg = defaultdict(int)
                for r in rs:
                    agg[r["predictor"]] += r["wins"]
                return {p: w / tot for p, w in agg.items()}
            all_rs  = [r for r in rows if r["model"] == m and r["depth"] == d]
            kept_rs = [r for r in all_rs if r["wave"] not in EXCL]
            s_all, s_new = shares(all_rs), shares(kept_rs)
            if not s_all:
                continue
            for p in sorted(s_all):
                print(f"  {m:12s} d{d} {p:12s}: {s_all[p]:6.1%} -> {s_new[p]:6.1%}")


def direction():
    rows = jload("collapse_direction.json")
    kept = [r for r in rows if r["wave"] not in EXCL]
    print("\n=== COLLAPSE DIRECTION: all waves -> excl W49 ===")
    for m in sorted({r["model"] for r in rows}):
        a_all = np.mean([r["agree"] for r in rows if r["model"] == m])
        a_new = np.mean([r["agree"] for r in kept if r["model"] == m])
        print(f"  {m:28s} agreement: {a_all:.1%} -> {a_new:.1%}")

    full = [r for r in kept if r["model"] != "gemma2_9b"]
    print("  per-dim delta (LLM keeps - human says), pooled 3 full models, excl W49:")
    dims = sorted({d for r in full for d in (r["dimA"], r["dimB"])})
    out = []
    for d in dims:
        contested = [r for r in full if d in (r["dimA"], r["dimB"])]
        keep   = np.mean([r["winner"] == d for r in contested])
        should = np.mean([r["human_dom"] == d for r in contested])
        out.append((keep - should, d, keep, should))
    for delta, d, keep, should in sorted(out, reverse=True):
        print(f"    {d:18s} keeps={keep:5.1%}  says={should:5.1%}  delta={delta*100:+5.1f}")

    pnd = [r for r in full if "Political Party" in (r["dimA"], r["dimB"])
           and r["human_dom"] != "Political Party"]
    print(f"  Party kept when other dim human-dominant: "
          f"{np.mean([r['winner'] == 'Political Party' for r in pnd]):.1%} (n={len(pnd)})")


def floor():
    rows = jload("noise_floor.json")
    print("\n=== NOISE FLOOR: all waves -> excl W49 ===")
    for d in (1, 2, 3):
        def pool(rs):
            n = sum(r["n_obs"] for r in rs)
            return sum(r["mean_tv"] * r["n_obs"] for r in rs) / n
        all_rs  = [r for r in rows if r["depth"] == d]
        kept_rs = [r for r in all_rs if r["wave"] not in EXCL]
        print(f"  depth {d}: {pool(all_rs):.4f} -> {pool(kept_rs):.4f}")


if __name__ == "__main__":
    spine()
    collapse()
    direction()
    floor()
