"""
One-off helper for the weighted-primary draft update: numbers quoted in the
papers' direction section that the standard logs don't print.

Reads collapse_direction_weighted.json and spine_results_weighted.json.
W49 excluded to match the drafts' 14-wave pooled basis.

  1. Per-dimension keep-rate deltas pooled over the three full-coverage
     models (gpt-4o-mini, gpt-4o, claude-haiku).
  2. Party kept-when-human-dominant vs kept-when-not, per model.
  3. GPT-5.5 vs GPT-4o-mini per-depth TV on the matched waves (W26+W34).
"""

import os, json
from collections import defaultdict

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
EXCL = {"49"}
FULL3 = ["gpt-4o-mini", "gpt-4o", "claude-haiku-4-5-20251001"]
DIMS = ["Age", "Gender", "Race", "Income", "Political Party", "Religion", "Education"]

with open(os.path.join(HERE, "collapse_direction_weighted.json")) as f:
    rows = [r for r in json.load(f) if r["wave"] not in EXCL]

print("=== (1) POOLED KEEP-RATE DELTAS, 3 full-coverage models, excl W49 ===")
sub = [r for r in rows if r["model"] in FULL3]
for d in DIMS:
    cont = [r for r in sub if r["dimA"] == d or r["dimB"] == d]
    if len(cont) < 1000:
        continue
    keep   = np.mean([r["winner"] == d for r in cont])
    should = np.mean([r["human_dom"] == d for r in cont])
    print(f"  {d:18s} keep={keep:.3f} human={should:.3f} delta={100*(keep-should):+.1f}pp (n={len(cont)})")

print("\n=== (2) PARTY KEPT WHEN DOMINANT vs NOT, per model, excl W49 ===")
for m in FULL3 + ["gpt-5.5-2026-04-23"]:
    cont = [r for r in rows if r["model"] == m
            and (r["dimA"] == "Political Party" or r["dimB"] == "Political Party")]
    if not cont:
        continue
    dom    = [r for r in cont if r["human_dom"] == "Political Party"]
    notdom = [r for r in cont if r["human_dom"] != "Political Party"]
    kd = np.mean([r["winner"] == "Political Party" for r in dom])
    kn = np.mean([r["winner"] == "Political Party" for r in notdom])
    print(f"  {m:28s} kept-when-dom={100*kd:.1f}% (n={len(dom)})  "
          f"kept-when-not={100*kn:.1f}% (n={len(notdom)})")

print("\n=== (3) GPT-5.5 vs GPT-4o-mini, matched waves W26+W34, weighted ===")
with open(os.path.join(HERE, "spine_results_weighted.json")) as f:
    spine = json.load(f)
for m in ["gpt-5.5-2026-04-23", "gpt-4o-mini"]:
    for d in (1, 2, 3):
        rs = [r for r in spine if r["model"] == m and r["depth"] == d
              and r["wave"] in ("26", "34")]
        if rs:
            n = sum(r["n_cells"] for r in rs)
            tv = sum(r["mean_tv"] * r["n_cells"] for r in rs) / n
            print(f"  {m:22s} d{d}: TV={tv:.4f} (waves={sorted(r['wave'] for r in rs)})")
