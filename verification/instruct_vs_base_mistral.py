"""
Mistral-7B replication of the base-vs-tuned contrast (referee Comment 3).
A: mistral_7b_text (base model, base prompt)
B: mistral_7b_instruct_baseprompt (instruct model, base prompt)
C: mistral_latest (instruct model, instruct prompt)
Restricted to the waves common to all three (B only has W26+W34).
Same delta = keep_rate - human_dom_rate, wave-cluster bootstrap, MDE@80%.
"""
import os, json
import numpy as np
from collections import defaultdict
from scipy.stats import norm

HERE = os.path.dirname(os.path.abspath(__file__))
RNG = np.random.default_rng(42); B = 10_000
MODELS = {"A": "mistral_7b_text", "B": "mistral_7b_instruct_baseprompt", "C": "mistral_latest"}
DIMS = ["Race", "Religion", "Political Party", "Education", "Age", "Income", "Gender"]
Z = norm.ppf(0.975) + norm.ppf(0.80)  # 80% power, two-sided 0.05

rows = json.load(open(os.path.join(HERE, "collapse_direction_weighted.json")))

def waves_of(tag):
    return {r["wave"] for r in rows if r["model"] == tag}
common = waves_of(MODELS["A"]) & waves_of(MODELS["B"]) & waves_of(MODELS["C"])
print("common waves:", sorted(common, key=int))

def extract(tag):
    pwd = defaultdict(lambda: defaultdict(lambda: [0, 0, 0]))
    for r in rows:
        if r["model"] != tag or r["wave"] not in common:
            continue
        for d in (r["dimA"], r["dimB"]):
            pw = pwd[r["wave"]][d]; pw[2] += 1
            if r["winner"] == d: pw[0] += 1
            if r["human_dom"] == d: pw[1] += 1
    return pwd

def boot(pwd, dim):
    ws = list(pwd.keys())
    arr = np.array([pwd[w].get(dim, [0, 0, 0]) for w in ws], dtype=float)
    out = []
    for _ in range(B):
        s = arr[RNG.integers(0, len(ws), len(ws))]; tn = s[:, 2].sum()
        out.append((s[:, 0].sum() - s[:, 1].sum()) / tn if tn else np.nan)
    return np.array(out)

data = {k: extract(v) for k, v in MODELS.items()}
print(f"\n{'Dim':16s} {'A':>7} {'B':>7} {'C':>7} | {'B-A':>6} {'95%CI':>16} {'MDE80':>6} | {'C-B':>6} {'95%CI':>16} {'MDE80':>6}")
for dim in DIMS:
    bA, bB, bC = boot(data["A"], dim), boot(data["B"], dim), boot(data["C"], dim)
    A, Bd, C = bA.mean()*100, bB.mean()*100, bC.mean()*100
    dBA, dCB = (bB-bA), (bC-bB)
    ciBA = (np.percentile(dBA, 2.5)*100, np.percentile(dBA, 97.5)*100)
    ciCB = (np.percentile(dCB, 2.5)*100, np.percentile(dCB, 97.5)*100)
    mdeBA, mdeCB = Z*dBA.std()*100, Z*dCB.std()*100
    sBA = "*" if (ciBA[0] > 0 or ciBA[1] < 0) else ""
    sCB = "*" if (ciCB[0] > 0 or ciCB[1] < 0) else ""
    print(f"{dim:16s} {A:+7.1f} {Bd:+7.1f} {C:+7.1f} | {dBA.mean()*100:+6.1f} "
          f"[{ciBA[0]:+.1f},{ciBA[1]:+.1f}]{sBA:1s} {mdeBA:6.1f} | {dCB.mean()*100:+6.1f} "
          f"[{ciCB[0]:+.1f},{ciCB[1]:+.1f}]{sCB:1s} {mdeCB:6.1f}")
print("\n* = 95% CI excludes 0.  B-A = weight/tuning effect, C-B = prompt effect.")
