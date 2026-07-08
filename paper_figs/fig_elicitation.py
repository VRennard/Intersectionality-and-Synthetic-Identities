"""
Fig (robustness) — Collapse is invariant to elicitation format.

For each model family, the depth-2 best-single win rate (the collapse statistic)
under three elicitation paradigms:
  - Aggregate:   "distribute 1,000 respondents across the options" (main pipeline)
  - Individual:  one call = one sampled persona answering the whole wave, x N
                 (the per-respondent paradigm of the silicon-sampling literature)
  - Log-prob:    softmax over option-letter logits, no sampling (OpinionQA style)

All bars land in a narrow band (~75-83%), so the collapse is a property of
demographic conditioning, not of the aggregate elicitation prompt.

Data: verification/collapse_results_weighted.json
"""

import os, json
from collections import defaultdict

import matplotlib.pyplot as plt

import style

# (aggregate tag, display name, individual-sampling tag, log-prob tag)
FAMS = [
    ("gpt-4o-mini",                "GPT-4o-mini",      "gpt-4o-mini_indiv",                None),
    ("claude-haiku-4-5-20251001",  "Claude\nHaiku 4.5","claude-haiku-4-5-20251001_indiv",  None),
    ("gemma2_9b",                  "Gemma-2\n9B",      "gemma2_9b_indiv",                  "gemma2_9b_logprob"),
    ("mistral_latest",             "Mistral\n7B",      "mistral_latest_indiv_q10",         "mistral_latest_logprob"),
    ("llama3_1_8b_instruct_q4",    "Llama-3.1\n8B",    "llama3_1_8b_instruct_q4_indiv",    "llama3_1_8b_instruct_q4_logprob"),
]

# paradigm -> (label, color)
PARADIGMS = [
    ("agg",  "Aggregate",          "#5d69b1"),
    ("ind",  "Individual sampling","#52bca3"),
    ("lp",   "Log-prob readout",   "#e58606"),
]


def pooled_best_single(rows, tag):
    a = defaultdict(int); waves = set()
    for r in rows:
        if r["model"] == tag and r["depth"] == 2:
            a[r["predictor"]] += r["wins"]; waves.add(r["wave"])
    tot = a["best_single"] + a["additive"]
    return (100.0 * a["best_single"] / tot, len(waves)) if tot else (None, 0)


def main():
    with open(os.path.join(style.VERIF, f"collapse_results{style.SUFFIX}.json")) as f:
        rows = json.load(f)
    rows = [r for r in rows if r["wave"] not in style.EXCLUDE_WAVES]

    # collect values: vals[family_index][paradigm] = (pct, nwaves)
    data = []
    allvals = []
    for agg_tag, name, ind_tag, lp_tag in FAMS:
        d = {}
        d["agg"] = pooled_best_single(rows, agg_tag)
        d["ind"] = pooled_best_single(rows, ind_tag) if ind_tag else (None, 0)
        d["lp"]  = pooled_best_single(rows, lp_tag)  if lp_tag  else (None, 0)
        data.append((name, d))
        allvals += [v[0] for v in d.values() if v[0] is not None]

    lo, hi = min(allvals), max(allvals)

    fig, ax = plt.subplots(figsize=(7.0, 3.1))

    # shaded "collapse band" spanning observed range
    ax.axhspan(lo, hi, color="0.85", alpha=0.55, zorder=0, lw=0)
    ax.axhline(50, color="0.6", lw=0.8, ls=(0, (4, 3)), zorder=1)
    ax.text(len(FAMS) - 0.5, 50.8, "chance (additive ties single)",
            ha="right", va="bottom", fontsize=6.5, color="0.45")

    n_par = len(PARADIGMS)
    bw = 0.78 / n_par
    for fi, (name, d) in enumerate(data):
        for pj, (key, plabel, color) in enumerate(PARADIGMS):
            pct, nw = d[key]
            x = fi + (pj - (n_par - 1) / 2) * bw
            if pct is None:
                ax.text(x, lo + 1.0, "n/a", ha="center", va="bottom",
                        fontsize=6, color="0.6", rotation=90)
                continue
            ax.bar(x, pct, width=bw * 0.92, color=color, zorder=3)
            ax.text(x, pct + 0.6, f"{pct:.0f}", ha="center", va="bottom",
                    fontsize=6.8, color="0.2")
            # mark partial-coverage runs
            if nw < 15 and nw > 0:
                ax.text(x, pct - 2.2, f"{nw}w", ha="center", va="top",
                        fontsize=5.5, color="white", fontweight="bold")

    ax.set_xticks(range(len(FAMS)))
    ax.set_xticklabels([d[0] for d in data], fontsize=8)
    ax.set_ylim(60, 90)
    ax.set_yticks([60, 65, 70, 75, 80, 85, 90])
    ax.set_ylabel("collapse: cells best explained\nby one dimension (%)", fontsize=8.5)
    ax.set_axisbelow(True)
    ax.grid(axis="y", lw=0.3, alpha=0.5, color="#ccc", zorder=0)

    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for _, _, c in PARADIGMS]
    ax.legend(handles, [lab for _, lab, _ in PARADIGMS],
              loc="upper center", bbox_to_anchor=(0.5, 1.16), ncol=3,
              frameon=False, fontsize=8, handlelength=1.1,
              columnspacing=1.4, handletextpad=0.5)
    ax.annotate(f"all paradigms within {lo:.0f}–{hi:.0f}%",
                xy=(0.012, 0.04), xycoords="axes fraction", fontsize=7,
                style="italic", color="0.35")

    fig.subplots_adjust(left=0.13, right=0.985, bottom=0.17, top=0.86)
    style.save(fig, "fig_elicitation")


if __name__ == "__main__":
    main()
