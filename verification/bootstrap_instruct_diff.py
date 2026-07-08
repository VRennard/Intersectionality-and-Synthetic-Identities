"""
Direct bootstrap test: is the Race suppression delta significantly
more negative for mistral_latest (instruct) vs mistral_7b_text (base)?

Approach: cluster bootstrap over waves, sampling each model independently,
compute diff = delta_instruct - delta_base per bootstrap draw.
B = 10000.
"""

import os, json
import numpy as np
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
RNG  = np.random.default_rng(42)
B    = 10_000
EXCL = {"49"}
DIM  = "Race"

MODELS = {
    "base":    "mistral_7b_text",
    "instruct": "mistral_latest",
}

def load_rows(path, model, excl=EXCL):
    with open(path) as f:
        rows = json.load(f)
    return [r for r in rows if r["model"] == model and r["wave"] not in excl]


def wave_delta(rows, dim):
    """
    Returns dict: wave -> (keep_sum, should_sum, n) for the given dimension.
    keep_sum  = # collapsed cells where LLM kept dim
    should_sum = # collapsed cells where human dom = dim
    """
    per_wave = defaultdict(lambda: [0, 0, 0])
    for r in rows:
        if r["dimA"] != dim and r["dimB"] != dim:
            continue
        w = r["wave"]
        per_wave[w][2] += 1
        if r["winner"] == dim:
            per_wave[w][0] += 1
        if r["human_dom"] == dim:
            per_wave[w][1] += 1
    return per_wave


def obs_delta(per_wave):
    keep  = sum(v[0] for v in per_wave.values())
    should = sum(v[1] for v in per_wave.values())
    n     = sum(v[2] for v in per_wave.values())
    return keep / n - should / n


def bootstrap_delta(per_wave, B, rng):
    waves = list(per_wave.keys())
    arr = np.array([[per_wave[w][0], per_wave[w][1], per_wave[w][2]]
                    for w in waves], dtype=float)
    boots = []
    for _ in range(B):
        idx = rng.integers(0, len(waves), len(waves))
        s = arr[idx]
        boots.append(s[:, 0].sum() / s[:, 2].sum() - s[:, 1].sum() / s[:, 2].sum())
    return np.array(boots)


def main():
    path = os.path.join(HERE, "collapse_direction.json")

    rows_base     = load_rows(path, MODELS["base"])
    rows_instruct = load_rows(path, MODELS["instruct"])

    pw_base     = wave_delta(rows_base, DIM)
    pw_instruct = wave_delta(rows_instruct, DIM)

    delta_base     = obs_delta(pw_base)
    delta_instruct = obs_delta(pw_instruct)
    obs_diff       = delta_instruct - delta_base

    print(f"Dimension: {DIM}")
    print(f"  {MODELS['base']:20s}  delta = {delta_base:+.4f}  (n_waves={len(pw_base)})")
    print(f"  {MODELS['instruct']:20s}  delta = {delta_instruct:+.4f}  (n_waves={len(pw_instruct)})")
    print(f"  Observed diff (instruct - base) = {obs_diff:+.4f}")

    # Bootstrap each model independently
    boots_base     = bootstrap_delta(pw_base,     B, RNG)
    boots_instruct = bootstrap_delta(pw_instruct, B, RNG)
    boots_diff     = boots_instruct - boots_base

    lo, hi = float(np.percentile(boots_diff, 2.5)), float(np.percentile(boots_diff, 97.5))
    pval   = float((boots_diff >= 0).mean())   # one-sided: instruct ≥ base (null: no extra suppression)

    print(f"\n  Bootstrap diff 95% CI: [{lo:+.4f}, {hi:+.4f}]")
    print(f"  One-sided p (diff >= 0, no extra suppression): {pval:.4f}")
    print(f"  Two-sided p:                                   {2*min(pval, 1-pval):.4f}")

    if hi < 0:
        print(f"\n  -> CI entirely negative: instruct suppresses Race SIGNIFICANTLY more (p_1s={pval:.4f})")
    elif lo < 0 < hi:
        print(f"\n  -> CI crosses zero: difference NOT significant (p_2s={2*min(pval,1-pval):.4f})")
    else:
        print(f"\n  -> CI entirely positive: instruct suppresses Race LESS (unexpected)")


if __name__ == "__main__":
    main()
