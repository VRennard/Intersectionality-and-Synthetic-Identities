"""
3-way study: does instruction tuning cause race/religion suppression?

Conditions:
  A. llama3_1_8b_text          — base model,    base prompt
  B. llama3_1_8b_instruct_base — instruct model, base prompt
  C. llama3_1_8b_instruct_q4  — instruct model, instruct prompt

Comparing A vs B isolates model-weight effect (same prompt).
Comparing B vs C isolates prompt-format effect (same model).

For each dimension: delta = keep_rate - should_rate (pp).
Bootstrap CIs cluster over waves (B=10,000).
Significance: bootstrap diff CI.
"""

import os, json
import numpy as np
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
RNG  = np.random.default_rng(42)
B    = 10_000

MODELS = {
    "A_base_model":      "llama3_1_8b_text",
    "B_instruct_base_prompt": "llama3_1_8b_instruct_base",
    "C_instruct_instruct_prompt": "llama3_1_8b_instruct_q4",
}
LABELS = {
    "A_base_model":               "Base model\n(base prompt)",
    "B_instruct_base_prompt":     "Instruct model\n(base prompt)",
    "C_instruct_instruct_prompt": "Instruct model\n(instruct prompt)",
}

# Common waves: exclude W49 (per paper convention) and W92 (incomplete for instruct_base)
EXCL = {"49", "92"}
DIMS = ["Race", "Religion", "Political Party", "Education", "Age", "Income", "Gender"]

path = os.path.join(HERE, "collapse_direction_weighted.json")
with open(path) as f:
    all_rows = json.load(f)

# ── per-model, per-wave, per-dimension stats ──────────────────────────────
def extract(rows, model_tag):
    rows = [r for r in rows if r["model"] == model_tag and r["wave"] not in EXCL]
    per_wave_dim = defaultdict(lambda: defaultdict(lambda: [0, 0, 0]))
    # [keep, should, n_contested]
    for r in rows:
        for d in (r["dimA"], r["dimB"]):
            pw = per_wave_dim[r["wave"]][d]
            pw[2] += 1
            if r["winner"]    == d: pw[0] += 1
            if r["human_dom"] == d: pw[1] += 1
    return per_wave_dim

def obs_delta(per_wave_dim, dim):
    keep = should = n = 0
    for wd in per_wave_dim.values():
        k, s, c = wd.get(dim, [0, 0, 0])
        keep += k; should += s; n += c
    if n == 0:
        return float("nan"), 0
    return (keep - should) / n, n

def bootstrap_delta(per_wave_dim, dim, rng, B=B):
    waves = list(per_wave_dim.keys())
    arr = np.array([[per_wave_dim[w].get(dim, [0, 0, 0])] for w in waves],
                   dtype=float).squeeze(1)  # shape (W, 3)
    boots = []
    for _ in range(B):
        idx = rng.integers(0, len(waves), len(waves))
        s = arr[idx]
        total_n = s[:, 2].sum()
        if total_n == 0:
            boots.append(float("nan"))
        else:
            boots.append((s[:, 0].sum() - s[:, 1].sum()) / total_n)
    return np.array(boots)

def bootstrap_diff_ci(boots_a, boots_b, rng, B=B):
    diff = boots_b - boots_a
    lo = float(np.percentile(diff, 2.5))
    hi = float(np.percentile(diff, 97.5))
    pval = float((diff >= 0).mean())
    return lo, hi, pval

# ── extract data ──────────────────────────────────────────────────────────
data = {k: extract(all_rows, v) for k, v in MODELS.items()}

print("Waves used per model:")
for k, v in MODELS.items():
    waves = sorted(data[k].keys(), key=int)
    print(f"  {k:35s}: {len(waves)} waves  {waves}")

print()
print(f"{'Dimension':20s}  {'A delta':>9}  {'B delta':>9}  {'C delta':>9}  "
      f"{'B-A (95%CI)':>24}  {'C-B (95%CI)':>24}")
print("-" * 110)

results = {}
for cond, model_tag in MODELS.items():
    pw = data[cond]
    results[cond] = {}
    for dim in DIMS:
        d, n = obs_delta(pw, dim)
        boots = bootstrap_delta(pw, dim, RNG)
        lo = float(np.percentile(boots, 2.5))
        hi = float(np.percentile(boots, 97.5))
        results[cond][dim] = {"delta": d, "boots": boots, "lo": lo, "hi": hi, "n": n}

for dim in DIMS:
    rA = results["A_base_model"][dim]
    rB = results["B_instruct_base_prompt"][dim]
    rC = results["C_instruct_instruct_prompt"][dim]

    lo_BA, hi_BA, p_BA = bootstrap_diff_ci(rA["boots"], rB["boots"], RNG)
    lo_CB, hi_CB, p_CB = bootstrap_diff_ci(rB["boots"], rC["boots"], RNG)

    sig_BA = "*" if (lo_BA > 0 or hi_BA < 0) else ""
    sig_CB = "*" if (lo_CB > 0 or hi_CB < 0) else ""

    print(f"{dim:20s}  "
          f"{rA['delta']*100:+7.1f}pp  "
          f"{rB['delta']*100:+7.1f}pp  "
          f"{rC['delta']*100:+7.1f}pp  "
          f"[{lo_BA*100:+.1f}, {hi_BA*100:+.1f}]{sig_BA:1s}  "
          f"[{lo_CB*100:+.1f}, {hi_CB*100:+.1f}]{sig_CB:1s}")

print()
print("A = base model / base prompt")
print("B = instruct model / base prompt   (B-A: model-weight effect)")
print("C = instruct model / instruct prompt  (C-B: prompt-format effect)")
print("* = 95% CI excludes zero")

# ── overall agreement ─────────────────────────────────────────────────────
print("\n\nOverall agreement rates:")
for cond, model_tag in MODELS.items():
    rows = [r for r in all_rows if r["model"] == model_tag and r["wave"] not in EXCL]
    if not rows:
        continue
    agree = sum(r["agree"] for r in rows) / len(rows)
    print(f"  {cond:40s}: {agree:.1%}  (n={len(rows):,})")

# ── figure ────────────────────────────────────────────────────────────────
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import sys, os
sys.path.insert(0, os.path.join(HERE, "..", "paper_figs"))
import style

COLORS = {"A_base_model": "#888888", "B_instruct_base_prompt": "#f4a261",
          "C_instruct_instruct_prompt": "#e63946"}
PLOT_DIMS = ["Race", "Religion", "Gender", "Income", "Age", "Education", "Political Party"]

fig, ax = plt.subplots(figsize=(7.5, 4.0))

n_dims = len(PLOT_DIMS)
n_conds = 3
width = 0.22
cond_keys = list(MODELS.keys())

for ci, cond in enumerate(cond_keys):
    offsets = np.arange(n_dims) + (ci - 1) * width
    deltas = [results[cond][d]["delta"] * 100 for d in PLOT_DIMS]
    los    = [results[cond][d]["lo"] * 100    for d in PLOT_DIMS]
    his    = [results[cond][d]["hi"] * 100    for d in PLOT_DIMS]
    yerr_lo = [d - lo for d, lo in zip(deltas, los)]
    yerr_hi = [hi - d for d, hi in zip(deltas, his)]
    ax.bar(offsets, deltas, width * 0.9, color=COLORS[cond], alpha=0.85,
           label=LABELS[cond].replace("\n", " "))
    ax.errorbar(offsets, deltas, yerr=[yerr_lo, yerr_hi],
                fmt="none", color="0.2", linewidth=0.9, capsize=2.5)

ax.axhline(0, color="0.3", linewidth=0.7)
ax.set_xticks(np.arange(n_dims))
ax.set_xticklabels([d.replace(" ", "\n") if d == "Political Party" else d
                    for d in PLOT_DIMS], fontsize=8.5)
ax.set_ylabel("Keep-rate minus human-dominance rate (pp)", fontsize=8.5)
ax.set_title("Race/religion suppression: base model vs instruction tuning\n"
             "delta = LLM keep-rate − human-dominance rate", fontsize=9)

patches = [mpatches.Patch(color=COLORS[k], alpha=0.85, label=LABELS[k].replace("\n", " "))
           for k in cond_keys]
ax.legend(handles=patches, fontsize=7.5, loc="upper right")
ax.tick_params(axis="y", labelsize=8)

for sp in ("top", "right"):
    ax.spines[sp].set_visible(False)

fig.tight_layout()
out = os.path.join(HERE, "fig_instruct_vs_base.pdf")
fig.savefig(out, bbox_inches="tight")
print(f"\nFigure saved: {out}")
