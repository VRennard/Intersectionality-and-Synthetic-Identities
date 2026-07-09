"""#16b — wave-cluster bootstrap for the hierarchy rank correlation (rho)
and per-dimension mean steering TVs. Data: paper_figs/exaggeration_data*.json."""
import os, json
import numpy as np
from scipy.stats import spearmanr
from shared import BASE, bootstrap_ci

FILES = {
    "gpt-4o-mini": "exaggeration_data.json",
    "claude-haiku-4-5-20251001": "exaggeration_data_claude-haiku-4-5-20251001.json",
    "gemma2_9b": "exaggeration_data_gemma2_9b.json",
    "gpt-4o": "exaggeration_data_gpt-4o.json",
}
rng = np.random.default_rng(5)

for model, fn in FILES.items():
    path = os.path.join(BASE, "paper_figs", fn)
    if not os.path.exists(path):
        continue
    pts = json.load(open(path))
    waves = sorted({p["wave"] for p in pts}, key=int)
    dims = sorted({p["dim"] for p in pts})
    rhos = []
    for w in waves:
        h, l = [], []
        for d in dims:
            sub = [p for p in pts if p["wave"] == w and p["dim"] == d]
            if not sub:
                h.append(np.nan); l.append(np.nan); continue
            h.append(np.mean([p["tv_h"] for p in sub]))
            l.append(np.mean([p["tv_l"] for p in sub]))
        ok = ~(np.isnan(h) | np.isnan(l))
        if ok.sum() >= 3:
            rhos.append(spearmanr(np.array(h)[ok], np.array(l)[ok]).statistic)
    rhos = np.array(rhos)
    lo, hi = bootstrap_ci(rhos, B=10000)
    print(f"{model:28s} mean rho = {rhos.mean():+.3f}  95% wave-bootstrap CI [{lo:+.3f}, {hi:+.3f}]  ({len(rhos)} waves)")

# per-dimension mean TVs with wave CIs (flagship)
pts = json.load(open(os.path.join(BASE, "paper_figs", FILES["gpt-4o-mini"])))
waves = sorted({p["wave"] for p in pts}, key=int)
print("\nflagship per-dimension mean steering TV (wave-bootstrap 95% CI):")
for d in sorted({p["dim"] for p in pts}):
    hw = [np.mean([p["tv_h"] for p in pts if p["wave"] == w and p["dim"] == d] or [np.nan]) for w in waves]
    lw = [np.mean([p["tv_l"] for p in pts if p["wave"] == w and p["dim"] == d] or [np.nan]) for w in waves]
    hw = [x for x in hw if not np.isnan(x)]; lw = [x for x in lw if not np.isnan(x)]
    hlo, hhi = bootstrap_ci(hw, B=5000); llo, lhi = bootstrap_ci(lw, B=5000)
    print(f"  {d:16s} human {np.mean(hw):.3f} [{hlo:.3f},{hhi:.3f}]   model {np.mean(lw):.3f} [{llo:.3f},{lhi:.3f}]")
