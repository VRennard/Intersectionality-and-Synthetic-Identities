"""
Find questions with bimodal or evenly-distributed responses across all waves.
Outputs: interesting_questions/ folder with data CSV and histograms.
"""

import json, glob, os, re
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from collections import defaultdict

# ── Config ──────────────────────────────────────────────────────────────────
RESP_DIR   = "data/responses"
OUT_DIR    = "interesting_questions"
SKIP_OPTS  = {"refused", "don't know", "dk/refused", "don't know/refused",
              "don't know / refused", "volunteered: don't know", "volunteered: refused",
              "volunteered: neither", "(vol.) don't know", "(vol.) refused"}

# ── Bimodal thresholds ────────────────────────────────────────────────────────
# 2-option split: both options must be within 40/60 territory
SPLIT2_BALANCE       = 0.65   # min/max >= 0.65 (roughly 39/61 or closer)

# 3-option: positions 0 and 2 are peaks, position 1 is valley
# 4/5-option: FIRST and LAST positions are peaks (extremes), middle options are valley
BIMODAL_PEAK_MIN     = 25.0   # each peak >= 25%
BIMODAL_BALANCE      = 0.55   # min_peak / max_peak >= 0.55
BIMODAL_VALLEY_MAX   = 20.0   # each valley option < 20%

EVEN_CV_THRESHOLD    = 0.20   # coefficient of variation < 20%
EVEN_MIN_OPTIONS     = 3      # at least 3 options to be "even"
MIN_RESPONDENTS      = 500    # skip tiny sub-questions

# ── Helpers ──────────────────────────────────────────────────────────────────

def clean_responses(responses):
    """Return list of (option_label, percentage) excluding skip options."""
    out = []
    for r in responses:
        lbl = r["option"].strip()
        if lbl.lower() in SKIP_OPTS:
            continue
        out.append((lbl, r["percentage"]))
    return out


def rescale(pairs):
    """Re-normalise percentages after dropping Refused/DK."""
    total = sum(p for _, p in pairs)
    if total == 0:
        return pairs
    return [(lbl, p / total * 100) for lbl, p in pairs]


def bimodal_score(pcts):
    """
    For an ordered sequence of percentages return a bimodality score.
    Score = (top2_share - 1/n) * valley_ratio
    High → strong bimodality.
    """
    n = len(pcts)
    if n < 3:
        return 0.0
    arr = np.array(pcts)
    top2_idx = np.argsort(arr)[-2:]
    top2_share = arr[top2_idx].sum() / arr.sum()

    # valley: mean of middle options (those NOT in top 2)
    middle = np.delete(arr, top2_idx)
    valley_mean = middle.mean() if len(middle) else 1e-9

    # penalise if top-2 are adjacent (not truly bimodal for ordered scale)
    gap = abs(top2_idx[1] - top2_idx[0])
    gap_bonus = 1.0 if gap == 1 else 1.3   # reward non-adjacent peaks

    if valley_mean < 1e-9:
        return 0.0
    valley_ratio = (arr[top2_idx].min() / valley_mean) * gap_bonus
    score = top2_share * valley_ratio
    return float(score)


def evenness_score(pcts):
    """1 - CV  (higher = more even). Returns (score, cv)."""
    arr = np.array(pcts)
    if arr.mean() == 0:
        return 0.0, 999
    cv = arr.std() / arr.mean()
    return float(1 - cv), float(cv)


def classify(pairs):
    """
    Return ('bimodal', subtype) | ('even', '') | None.
    Subtypes:
      'split2'  — 2-option near 50/50
      'polar3'  — 3-option: outer options high, middle low
      'ushape'  — 4/5-option: first and last (extremes) both high, middle low
      'dom2'    — 3+ options: any two non-adjacent dominant peaks (fallback)
    """
    if len(pairs) < 2:
        return None
    pcts = [p for _, p in pairs]
    n = len(pcts)

    # ── 2-option split ────────────────────────────────────────────────────────
    if n == 2:
        lo, hi = sorted(pcts)
        if hi > 0 and (lo / hi) >= SPLIT2_BALANCE:
            return ("bimodal", "split2")
        return None

    # ── 4 or 5-option U-shape: extremes (pos 0 and pos -1) both peak ─────────
    if n in (4, 5):
        p_first, p_last = pcts[0], pcts[-1]
        valley = pcts[1:-1]
        both_strong  = min(p_first, p_last) >= BIMODAL_PEAK_MIN
        balanced     = (min(p_first, p_last) / max(p_first, p_last)) >= BIMODAL_BALANCE
        clear_valley = max(valley) < BIMODAL_VALLEY_MAX
        if both_strong and balanced and clear_valley:
            return ("bimodal", "ushape")

    # ── 3-option: positions 0 and 2 peak, position 1 is valley ───────────────
    if n == 3:
        p0, p1, p2 = pcts
        both_strong  = min(p0, p2) >= BIMODAL_PEAK_MIN
        balanced     = (min(p0, p2) / max(p0, p2)) >= BIMODAL_BALANCE if max(p0, p2) > 0 else False
        clear_valley = p1 < BIMODAL_VALLEY_MAX
        if both_strong and balanced and clear_valley:
            return ("bimodal", "polar3")

    # ── General: any two non-adjacent dominant peaks (6+ options) ────────────
    if n >= 6:
        top2_idx = sorted(range(n), key=lambda i: pcts[i])[-2:]
        top2     = [pcts[i] for i in top2_idx]
        middle   = [pcts[i] for i in range(n) if i not in top2_idx]
        pos_gap  = abs(top2_idx[1] - top2_idx[0])
        both_strong  = min(top2) >= BIMODAL_PEAK_MIN
        balanced     = (min(top2) / max(top2)) >= BIMODAL_BALANCE if max(top2) > 0 else False
        clear_valley = (max(middle) < BIMODAL_VALLEY_MAX) if middle else True
        if both_strong and balanced and clear_valley and pos_gap >= 2:
            return ("bimodal", "dom2")

    # ── Evenly distributed ────────────────────────────────────────────────────
    _, cv = evenness_score(pcts)
    if cv < EVEN_CV_THRESHOLD and n >= EVEN_MIN_OPTIONS:
        return ("even", "")

    return None


# ── Main ──────────────────────────────────────────────────────────────────────

def load_all_questions():
    rows = []
    for fpath in sorted(glob.glob(f"{RESP_DIR}/survey_responses_W*.json")):
        with open(fpath) as f:
            data = json.load(f)
        for q in data:
            if q["total_respondents"] < MIN_RESPONDENTS:
                continue
            pairs = clean_responses(q["responses"])
            pairs = rescale(pairs)
            if len(pairs) < 2:
                continue
            result = classify(pairs)
            if result is None:
                continue
            kind, subtype = result
            pcts = [p for _, p in pairs]

            # bimodal score: balance × valley-depth
            if kind == "bimodal":
                if subtype == "split2":
                    lo, hi = sorted(pcts)
                    bscore = lo / hi if hi > 0 else 0.0
                elif subtype in ("polar3", "ushape"):
                    peaks  = [pcts[0], pcts[-1]]
                    valley = pcts[1:-1]
                    vm     = np.mean(valley) if valley else 1e-9
                    bal    = min(peaks) / max(peaks) if max(peaks) > 0 else 0
                    depth  = min(peaks) / (vm + 1e-9)
                    bscore = float(bal * depth)
                else:
                    top2   = sorted(pcts)[-2:]
                    middle = sorted(pcts)[:-2]
                    vm     = np.mean(middle) if middle else 1e-9
                    bal    = min(top2) / max(top2) if max(top2) > 0 else 0
                    bscore = float(bal * min(top2) / (vm + 1e-9))
            else:
                bscore = 0.0

            rows.append({
                "wave":        q["wave"],
                "question_id": q["question_id"],
                "question_text": q["question_text"],
                "n_respondents": q["total_respondents"],
                "n_options":   len(pairs),
                "kind":        kind,
                "subtype":     subtype,
                "options":     [lbl for lbl, _ in pairs],
                "pcts":        pcts,
                "bscore":      float(bscore),
                "cv":          evenness_score(pcts)[1],
            })
    return rows


def make_histogram(row, out_path):
    opts  = row["options"]
    pcts  = row["pcts"]
    kind  = row["kind"]
    wave  = row["wave"]
    qid   = row["question_id"]
    qtxt  = row["question_text"]
    n     = row["n_respondents"]

    subtype = row.get("subtype", "")
    color_map = {"split2": "#c0392b", "polar3": "#e67e22",
                 "ushape": "#8e44ad", "dom2": "#d64e4e"}
    color = color_map.get(subtype, "#4e8ad6") if kind == "bimodal" else "#4e8ad6"

    fig, ax = plt.subplots(figsize=(max(7, len(opts) * 1.4), 4.5))
    bars = ax.bar(range(len(opts)), pcts, color=color, edgecolor="white",
                  linewidth=0.8, alpha=0.85)

    for bar, pct in zip(bars, pcts):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                f"{pct:.1f}%", ha="center", va="bottom", fontsize=8, fontweight="bold")

    ax.set_xticks(range(len(opts)))
    ax.set_xticklabels(opts, rotation=25, ha="right", fontsize=8)
    ax.set_ylabel("% of respondents", fontsize=9)
    ax.set_ylim(0, max(pcts) * 1.25)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.0f}%"))

    # Wrap question text
    import textwrap
    wrapped = "\n".join(textwrap.wrap(qtxt, width=90))
    subtype_labels = {"split2": "50/50 SPLIT", "polar3": "BIMODAL (outer peaks)",
                      "ushape": "U-SHAPE (extremes dominate)", "dom2": "BIMODAL (two peaks)"}
    label_str = subtype_labels.get(subtype, "BIMODAL") if kind == "bimodal" else "EVENLY DISTRIBUTED"
    ax.set_title(f"[{wave}] {qid}  •  {label_str}\n{wrapped}\n(n={n:,})",
                 fontsize=9, loc="left", pad=8)

    patch = mpatches.Patch(color=color, alpha=0.85,
                           label=f"{label_str}: {len(opts)} options")
    ax.legend(handles=[patch], fontsize=8, loc="upper right")

    plt.tight_layout()
    plt.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close()


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    bimodal_dir  = os.path.join(OUT_DIR, "bimodal")
    split2_dir   = os.path.join(OUT_DIR, "bimodal", "split2_50_50")
    polar3_dir   = os.path.join(OUT_DIR, "bimodal", "polar3_outer_peaks")
    ushape_dir   = os.path.join(OUT_DIR, "bimodal", "ushape_extremes")
    even_dir     = os.path.join(OUT_DIR, "even")
    for d in [bimodal_dir, split2_dir, polar3_dir, ushape_dir, even_dir]:
        os.makedirs(d, exist_ok=True)
    subtype_dirs = {"split2": split2_dir, "polar3": polar3_dir,
                    "ushape": ushape_dir, "dom2": bimodal_dir}

    print("Loading all waves…")
    rows = load_all_questions()

    bimodal = sorted([r for r in rows if r["kind"] == "bimodal"],
                     key=lambda r: -r["bscore"])
    even    = sorted([r for r in rows if r["kind"] == "even"],
                     key=lambda r: r["cv"])

    print(f"Found {len(bimodal)} bimodal questions, {len(even)} evenly-distributed questions.")

    # ── Save summary CSV ──────────────────────────────────────────────────────
    import csv

    def write_csv(rows_list, path, kind):
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["wave","question_id","question_text","n_respondents",
                        "n_options","kind","subtype","options","percentages",
                        "bimodal_score","cv"])
            for r in rows_list:
                w.writerow([
                    r["wave"], r["question_id"], r["question_text"],
                    r["n_respondents"], r["n_options"], r["kind"],
                    r.get("subtype",""),
                    " | ".join(r["options"]),
                    " | ".join(f"{p:.1f}" for p in r["pcts"]),
                    f"{r['bscore']:.3f}", f"{r['cv']:.3f}",
                ])

    write_csv(bimodal, os.path.join(OUT_DIR, "bimodal_questions.csv"), "bimodal")
    write_csv(even,    os.path.join(OUT_DIR, "even_questions.csv"),    "even")

    # ── Generate histograms ───────────────────────────────────────────────────
    print("Generating bimodal histograms…")
    for i, row in enumerate(bimodal):
        fname = f"{row['wave']}_{row['question_id'].replace('/', '_')}.png"
        dest  = subtype_dirs.get(row.get("subtype",""), bimodal_dir)
        make_histogram(row, os.path.join(dest, fname))
        if (i + 1) % 20 == 0:
            print(f"  {i+1}/{len(bimodal)}")

    print("Generating even histograms…")
    for i, row in enumerate(even):
        fname = f"{row['wave']}_{row['question_id'].replace('/', '_')}.png"
        make_histogram(row, os.path.join(even_dir, fname))
        if (i + 1) % 20 == 0:
            print(f"  {i+1}/{len(even)}")

    # ── Print top examples ────────────────────────────────────────────────────
    print("\n-- TOP 10 BIMODAL --")
    for r in bimodal[:10]:
        opts_str = "  |  ".join(f"{o}: {p:.1f}%" for o, p in zip(r["options"], r["pcts"]))
        print(f"  [{r['wave']}] {r['question_id']}  [{r.get('subtype','')}]  score={r['bscore']:.2f}")
        print(f"    {r['question_text'][:80]}")
        print(f"    {opts_str}\n")

    print("-- TOP 10 EVEN --")
    for r in even[:10]:
        opts_str = "  |  ".join(f"{o}: {p:.1f}%" for o, p in zip(r["options"], r["pcts"]))
        print(f"  [{r['wave']}] {r['question_id']}  cv={r['cv']:.3f}")
        print(f"    {r['question_text'][:80]}")
        print(f"    {opts_str}\n")

    from collections import Counter
    subtypes = Counter(r.get("subtype","") for r in bimodal)
    print(f"\nDone. Output in: {OUT_DIR}")
    print(f"  bimodal_questions.csv ({len(bimodal)} rows)")
    for st, cnt in sorted(subtypes.items()):
        print(f"    {st}: {cnt}")
    print(f"  even_questions.csv    ({len(even)} rows)")
    print(f"  bimodal/  ({len(bimodal)} histograms, split into subfolders)")
    print(f"  even/     ({len(even)} histograms)")


if __name__ == "__main__":
    main()
