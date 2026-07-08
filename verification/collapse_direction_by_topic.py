"""
Per-topic (per-wave) breakdown of the collapse-direction results.
Reads verification/collapse_direction.json (written by collapse_direction.py).

For each model x wave:
  - agreement with human-dominant dimension
  - Party kept when humans say the other dim matters more
For the flagship model (and pooled), a wave x dimension matrix of
  delta = (LLM keep rate) - (human dominance rate).
"""

import os, sys, json
from collections import defaultdict

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
# --weighted: read the survey-weighted direction JSON (paper primary).
SUFFIX = "_weighted" if "--weighted" in sys.argv else ""

TOPIC = {
    "26": "Crime/Safety", "27": "Technology", "29": "Science/Health",
    "32": "Media Trust", "34": "Religion", "36": "Economy",
    "41": "Covid/Politics", "42": "Covid", "43": "Politics",
    "45": "Race/Society", "49": "(W49)", "50": "Politics",
    "54": "Biden/Politics", "82": "Economy/Jobs", "92": "Tech/AI",
}
DIMS = ["Age", "Gender", "Race", "Income", "Political Party", "Religion", "Education"]
SHORT = {"Age": "Age", "Gender": "Gen", "Race": "Race", "Income": "Inc",
         "Political Party": "Party", "Religion": "Relig", "Education": "Edu"}


def main():
    with open(os.path.join(HERE, f"collapse_direction{SUFFIX}.json")) as f:
        rows = json.load(f)

    models = sorted({r["model"] for r in rows})
    waves  = sorted({r["wave"] for r in rows}, key=int)

    # ---- agreement per model x wave ----
    print("=== AGREEMENT WITH HUMAN-DOMINANT DIM, per wave (chance ~50%) ===")
    header = f"{'wave':5s}{'topic':16s}" + "".join(f"{m[:12]:>14s}" for m in models)
    print(header)
    for w in waves:
        cells = []
        for m in models:
            rs = [r for r in rows if r["model"] == m and r["wave"] == w]
            cells.append(f"{np.mean([r['agree'] for r in rs])*100:13.1f}%" if rs else f"{'—':>14s}")
        print(f"W{w:4s}{TOPIC.get(w, ''):16s}" + "".join(cells))

    # ---- Party retention per wave (pooled over models that contest Party) ----
    print("\n=== PARTY KEPT WHEN HUMANS SAY OTHER DIM MATTERS MORE, per wave (pooled) ===")
    for w in waves:
        rs = [r for r in rows if r["wave"] == w
              and "Political Party" in (r["dimA"], r["dimB"])
              and r["human_dom"] != "Political Party"]
        if not rs:
            continue
        kept = np.mean([r["winner"] == "Political Party" for r in rs])
        # and the reverse: Party dropped when humans say Party SHOULD win
        rs2 = [r for r in rows if r["wave"] == w
               and "Political Party" in (r["dimA"], r["dimB"])
               and r["human_dom"] == "Political Party"]
        kept2 = np.mean([r["winner"] == "Political Party" for r in rs2]) if rs2 else float("nan")
        print(f"W{w:4s}{TOPIC.get(w, ''):16s} kept-when-shouldn't={kept:6.1%} (n={len(rs):6d})   "
              f"kept-when-should={kept2:6.1%} (n={len(rs2):6d})")

    # ---- delta matrix per wave x dim, pooled over the 3 full-dim models ----
    full_models = [m for m in models if m != "gemma2_9b"]
    print("\n=== DELTA = LLM-KEEP% - HUMAN-DOM% per wave x dim "
          f"(pooled over {', '.join(full_models)}) ===")
    print(f"{'wave':5s}{'topic':16s}" + "".join(f"{SHORT[d]:>8s}" for d in DIMS))
    for w in waves:
        rs = [r for r in rows if r["wave"] == w and r["model"] in full_models]
        line = f"W{w:4s}{TOPIC.get(w, ''):16s}"
        for d in DIMS:
            contested = [r for r in rs if d in (r["dimA"], r["dimB"])]
            if len(contested) < 100:
                line += f"{'—':>8s}"
                continue
            keep   = np.mean([r["winner"] == d for r in contested])
            should = np.mean([r["human_dom"] == d for r in contested])
            line += f"{(keep-should)*100:+8.1f}"
        print(line)

    # column summary: mean |delta| and sign consistency
    print("\n=== DIM SUMMARY ACROSS WAVES (pooled models) ===")
    for d in DIMS:
        deltas = []
        for w in waves:
            rs = [r for r in rows if r["wave"] == w and r["model"] in full_models]
            contested = [r for r in rs if d in (r["dimA"], r["dimB"])]
            if len(contested) < 100:
                continue
            keep   = np.mean([r["winner"] == d for r in contested])
            should = np.mean([r["human_dom"] == d for r in contested])
            deltas.append(keep - should)
        deltas = np.array(deltas)
        same_sign = max((deltas > 0).mean(), (deltas < 0).mean())
        print(f"  {d:18s} mean delta={deltas.mean()*100:+6.1f}  "
              f"range=[{deltas.min()*100:+6.1f},{deltas.max()*100:+6.1f}]  "
              f"sign-consistent across {same_sign:.0%} of waves")


if __name__ == "__main__":
    main()
