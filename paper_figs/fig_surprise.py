"""
Fig 6 — Failure is uniform across profile typicality.

Left: per pair profile, x = co-occurrence surprise P(A&B)/(P(A)P(B)) from
pooled human microdata (log scale; <1 = counter-stereotypical). Two binned-
mean lines: raw TV error (rises for rare profiles) and noise-corrected excess
TV (flat) — what looked like counter-stereotypical failure is ground-truth
sampling noise from small human cells.

Right: three case studies (one question each, picked mechanically — see
selection rule at the pick site / verify_cases.py): human vs LLM vs additive
distributions. The additive bars show the model had the ingredients and
didn't use them — illustrations of collapse, not "worst failures".

Data: paper_figs/surprise_tv_corrected.json (noise_correct_surprise.py),
case_studies.json (compute_surprise.py).
"""

import os, json, textwrap
import numpy as np
import matplotlib.pyplot as plt

import style

C_HUM, C_LLM, C_ADD = "0.35", "#c23b22", "#5d69b1"

CASE_LABELS = {
    "[('Political Party', 'Republican'), ('Religion', 'Atheist')]": "Republican Atheist",
    "[('Political Party', 'Republican'), ('Race', 'Black')]":       "Black Republican",
    "[('Gender', 'Male'), ('Religion', 'Muslim')]":                 "Muslim man",
}


def binned(x, y):
    # fixed geometric edges spanning the observed range, so the lines reach
    # the counter-stereotypical tail; small bins tolerated at the extremes
    edges = np.array([x.min() * 0.999, 0.45, 0.7, 0.9, 1.1, 1.5, 2.5,
                      x.max() * 1.001])
    idx = np.digitize(x, edges)
    bx, by = [], []
    for b in range(1, len(edges)):
        sel = idx == b
        if sel.sum() >= 3:
            bx.append(np.exp(np.log(x[sel]).mean()))
            by.append(y[sel].mean())
    return bx, by


def main():
    with open(os.path.join(style.HERE, "surprise_tv_corrected.json")) as f:
        pairs = json.load(f)
    with open(os.path.join(style.HERE, "case_studies.json")) as f:
        cases = json.load(f)

    fig = plt.figure(figsize=(7.2, 5.2))
    gs = fig.add_gridspec(3, 2, width_ratios=[1.3, 1], hspace=1.55,
                          left=0.095, right=0.965, bottom=0.09, top=0.84)
    ax = fig.add_subplot(gs[:, 0])

    # ---- left: raw vs noise-corrected error across surprise ----
    x   = np.array([p["surprise"] for p in pairs])
    raw = np.array([p["mean_tv"] for p in pairs])
    exc = np.array([p["excess_tv"] for p in pairs])
    ok = x > 0
    x, raw, exc = x[ok], raw[ok], exc[ok]

    ax.scatter(x, exc, s=7, color="0.6", alpha=0.45, linewidths=0)
    ax.set_xscale("log")

    bx, by = binned(x, raw)
    ax.plot(bx, by, "--o", color="0.45", linewidth=1.4, markersize=3.2,
            label="raw error (incl. ground-truth noise)")
    bx, by = binned(x, exc)
    ax.plot(bx, by, "-o", color="#c23b22", linewidth=1.7, markersize=3.5,
            label="noise-corrected excess error")

    ax.axvline(1.0, color="0.3", linewidth=0.9, linestyle="--")
    ax.set_xlim(0.04, 15)
    top = max(raw) * 1.0
    ax.text(1.08, top, "chance\nco-occurrence", fontsize=7, color="0.3", va="top")
    ax.text(0.047, top, "counter-\nstereotypical", fontsize=7.5,
            color="0.45", va="top")

    ax.set_xlabel("co-occurrence surprise  $P(A{\\cap}B)\\,/\\,P(A)P(B)$")
    ax.set_ylabel("TV error of pair simulation")
    ax.legend(loc="lower left", frameon=False, fontsize=6.8,
              handlelength=1.6, labelspacing=0.3)

    # ---- right: case studies ----
    for row, (key, label) in enumerate(CASE_LABELS.items()):
        recs = cases.get(key, [])
        axc = fig.add_subplot(gs[row, 1])
        if not recs:
            axc.axis("off")
            continue
        # Mechanical pick (no hand-curation, audited in verify_cases.py):
        # additive must be within 2x expected sampling noise of the human
        # truth ("ingredients suffice"); among those, biggest LLM-vs-additive
        # gap in cells at or above the profile's median human cell size.
        ok = [r for r in recs if r.get("noise_tv")
              and r["tv_additive"] <= 2 * r["noise_tv"]]
        med_n = float(np.median([r["human_n"] for r in ok])) if ok else 0
        ok = [r for r in ok if r["human_n"] >= med_n] or recs
        rec = max(ok, key=lambda r: r["tv_llm"] - r["tv_additive"])
        opts  = rec["options"]
        keep  = [i for i, o in enumerate(opts) if o != "Refused"]
        labels = [textwrap.fill(opts[i], 16) for i in keep]
        xpos = np.arange(len(keep))
        w = 0.27
        b_h = axc.bar(xpos - w, [rec["human"][i] for i in keep], w, color=C_HUM, label="humans")
        b_l = axc.bar(xpos,      [rec["llm"][i] for i in keep], w, color=C_LLM, label="LLM")
        b_a = axc.bar(xpos + w,  [rec["additive"][i] for i in keep], w, color=C_ADD,
                      label="additive pred.")
        axc.set_xticks(xpos)
        axc.set_xticklabels(labels, fontsize=5.2)
        axc.set_yticks([])
        axc.spines["left"].set_visible(False)
        qtext = rec["question"].replace(
            "about equally in when seeking medical treatment situations?",
            "about equally, when seeking medical treatment?")
        q = textwrap.fill(qtext, 52)
        axc.set_title(f"{label} (n={rec['human_n']})\n" + q,
                      fontsize=5.6, loc="left", linespacing=1.05)

    # single horizontal legend across the top of the case-study column
    fig.legend([b_h, b_l, b_a], ["humans", "LLM", "additive pred."],
               loc="upper center", bbox_to_anchor=(0.79, 0.99), ncol=3,
               frameon=False, fontsize=6.2, handlelength=0.9, columnspacing=1.0,
               handletextpad=0.4)

    style.save(fig, "fig_surprise")


if __name__ == "__main__":
    main()
