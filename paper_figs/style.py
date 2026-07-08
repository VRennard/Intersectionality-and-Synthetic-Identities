"""Shared matplotlib style for all paper figures (matches fig_spine.py)."""

import os

import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams.update({
    "pdf.fonttype": 42,
    "font.size": 9,
    "axes.titlesize": 9,
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

HERE          = os.path.dirname(os.path.abspath(__file__))
BASE          = os.path.dirname(HERE)
VERIF         = os.path.join(BASE, "verification")
EXCLUDE_WAVES = set()    # W49 repaired 2026-06-11, see PAPER_PLAN_v2.md §7
# Survey-weighted human baselines are the paper's primary analysis
# (2026-06-11 switch); set SUFFIX="" to reproduce unweighted figures.
SUFFIX        = "_weighted"

TOPIC = {
    "26": "Crime/Safety", "27": "Technology", "29": "Science/Health",
    "32": "Media Trust", "34": "Religion", "36": "Economy",
    "41": "Covid/Politics", "42": "Covid", "43": "Politics (W43)",
    "45": "Race/Society", "49": "Privacy", "50": "Politics (W50)",
    "54": "Biden/Politics", "82": "Economy/Jobs", "92": "Tech/AI",
}

# predictor palette (used across Figs 4 and 7)
C_NULL   = "0.55"
C_ADD    = "#e58606"
C_SINGLE = "#5d69b1"
C_PAIR   = "#52bca3"


def save(fig, name):
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(HERE, f"{name}.{ext}"), dpi=300)
    print(f"wrote paper_figs/{name}.png/.pdf")
