"""Two redesigns:
(1) fig_simplex   — the explanation simplex: each model-depth is a point in the
    (one-dim / two-dim / additive) share triangle; arrows d2 -> d3.
(2) fig_squeeze   — compression to a coin flip: per dimension, arrow from human
    dominance rate to LLM keep rate; all arrows converge on ~50%.
"""
import os, json
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt
import style

# ============ (1) explanation simplex ============
with open(os.path.join(style.VERIF, f"collapse_results{style.SUFFIX}.json")) as f:
    rows = json.load(f)
rows = [r for r in rows if r["wave"] not in style.EXCLUDE_WAVES]
MODELS = [("gpt-4o-mini","GPT-4o-mini"),("gemma2_9b","Gemma-2 9B"),
          ("claude-haiku-4-5-20251001","Claude Haiku"),("gpt-5.5-2026-04-23","GPT-5.5"),
          ("llama3_1_8b_instruct_q4","Llama-3.1 8B"),("mistral-small_24b","Mistral 24B"),
          ("mistral_latest","Mistral 7B")]
def shares(tag, depth):
    agg = defaultdict(int)
    for r in rows:
        if r["model"] == tag and r["depth"] == depth:
            agg[r["predictor"]] += r["wins"]
    t = sum(agg.values())
    if not t: return None
    return (agg.get("best_single",0)/t, agg.get("best_pair",0)/t, agg.get("additive",0)/t)

A = np.array([0,0]); B = np.array([1,0]); C = np.array([0.5, np.sqrt(3)/2])
def bary(w): return w[0]*A + w[1]*B + w[2]*C

fig, ax = plt.subplots(figsize=(5.6, 4.6))
tri = plt.Polygon([A,B,C], fill=False, ec="0.3", lw=1.2)
ax.add_patch(tri)
# grid lines at 25/50/75%
for f in (0.25,0.5,0.75):
    for (p,q,r_) in ((A,B,C),(B,C,A),(C,A,B)):
        ax.plot(*zip(f*p+(1-f)*q, f*p+(1-f)*r_), color="0.88", lw=0.6, zorder=0)
ax.annotate("100% one dimension", xy=A, xytext=(-0.02,-0.05), ha="left", fontsize=8.5, fontweight="bold", color="#5d69b1")
ax.annotate("100% two dimensions", xy=B, xytext=(1.02,-0.05), ha="right", fontsize=8.5, fontweight="bold", color="#52bca3")
ax.annotate("100% additive\n(composition)", xy=C, xytext=(0.5,0.92), ha="center", fontsize=8.5, fontweight="bold", color="#e58606")
for tag,name in MODELS:
    s2, s3 = shares(tag,2), shares(tag,3)
    if not s2 or not s3: continue
    p2, p3 = bary(s2), bary(s3)
    ax.annotate("", xy=p3, xytext=p2,
                arrowprops=dict(arrowstyle="-|>", color="0.45", lw=1.1,
                                shrinkA=4, shrinkB=4))
    ax.plot(*p2, "o", ms=7, color="#5d69b1", mec="white", mew=0.8, zorder=5)
    ax.plot(*p3, "o", ms=7, color="#c23b22", mec="white", mew=0.8, zorder=5)
ax.plot([],[], "o", color="#5d69b1", label="2 features")
ax.plot([],[], "o", color="#c23b22", label="3 features")
# annotate clusters instead of each point
ax.annotate("all 7 models,\n2 features", xy=(0.06,0.30), fontsize=8, color="#5d69b1",
            ha="center")
ax.annotate("all 7 models,\n3 features", xy=(0.80,0.17), fontsize=8, color="#c23b22",
            ha="center")
ax.annotate("perfect composition\nwould be here", xy=C, xytext=(0.86,0.72),
            fontsize=7.5, color="#b26a04",
            arrowprops=dict(arrowstyle="->", color="#b26a04", lw=0.8))
ax.legend(frameon=False, fontsize=8, loc="upper left")
ax.set_xlim(-0.06,1.06); ax.set_ylim(-0.1,1.0)
ax.set_aspect("equal"); ax.axis("off")
style.save(fig, "fig_simplex")

# ============ (2) compression to a coin flip ============
with open(os.path.join(style.VERIF, f"collapse_direction{style.SUFFIX}.json")) as f:
    drows = json.load(f)
FULL = {"gpt-4o-mini","gpt-4o","claude-haiku-4-5-20251001","llama3_1_8b_instruct_q4"}
drows = [r for r in drows if "dimA" in r and r["wave"] not in style.EXCLUDE_WAVES
         and r["wave"] != "49" and r["model"] in FULL]
keep=defaultdict(int); should=defaultdict(int); contested=defaultdict(int)
for r in drows:
    for d in (r["dimA"], r["dimB"]): contested[d]+=1
    keep[r["winner"]]+=1; should[r["human_dom"]]+=1
DIMS = sorted(contested, key=lambda d: should[d]/contested[d], reverse=True)
fig2, ax2 = plt.subplots(figsize=(6.4, 3.6))
ax2.axvspan(43, 53, color="0.92", zorder=0)
ax2.text(48, len(DIMS)+0.1, "where the model lives (43\u201353%)",
         ha="center", fontsize=7.6, color="0.35")
ax2.axvline(50, color="0.55", ls=":", lw=0.9)
labels=[]
for i, d in enumerate(DIMS):
    y = len(DIMS)-1-i
    h = 100*should[d]/contested[d]; l = 100*keep[d]/contested[d]
    col = "#c23b22" if h > l else "#2b6cb0"
    ax2.annotate("", xy=(l, y), xytext=(h, y),
                 arrowprops=dict(arrowstyle="-|>", color=col, lw=2.2,
                                 shrinkA=0, shrinkB=0))
    ax2.plot(h, y, "o", ms=6.5, color="0.25", zorder=5)
    ax2.text(h, y+0.30, f"{h:.0f}%", ha="center", fontsize=7.4, color="0.25")
    ax2.text(l, y-0.42, f"{l:.0f}%", ha="center", fontsize=7.4, color=col,
             fontweight="bold")
    labels.append((y, d.replace("Political Party","Party")))
ax2.set_yticks([y for y,_ in labels])
ax2.set_yticklabels([n for _,n in labels], fontsize=8.4)
ax2.plot([],[],"o",color="0.25",label="humans: share of contests it should win")
ax2.annotate("", xy=(0.70,0.06), xytext=(0.63,0.06), xycoords="axes fraction",
             arrowprops=dict(arrowstyle="-|>", color="0.4", lw=1.6))
ax2.text(0.715, 0.06, "what the model keeps", transform=ax2.transAxes,
         fontsize=7.6, va="center", color="0.25")
ax2.set_xlim(24, 70); ax2.set_ylim(-0.9, len(DIMS)+0.35)
ax2.set_xlabel("share of contested pair cells the dimension wins (%)")
for sp in ("right","top"): ax2.spines[sp].set_visible(False)
ax2.legend(frameon=False, fontsize=7.4, loc="lower left", handletextpad=0.3)
style.save(fig2, "fig_squeeze")
