"""
De-confound: does the BASE few-shot prompt (vs the instruct prompt) change the
instruct model's Race/Religion keep-rate deltas? Compares three Mistral-7B runs
on W26+W34, on INTERSECTED pair-cells (only cells all relevant models have, so
the new model's question-biased dropout can't skew the comparison).

  base            = mistral_7b_text             (base model    + base prompt)
  instruct_base   = mistral_7b_instruct_baseprompt (instruct model + base prompt)  [NEW]
  instruct_chat   = mistral_latest              (instruct model + instruct prompt)

Logic:
  instruct_base vs instruct_chat -> same model, prompt differs  => PROMPT effect
  instruct_base vs base          -> same prompt, model differs   => MODEL effect
Religion: instruct_chat excluded (no Religion pairs at d2).
"""
import os, sys, pickle
from collections import defaultdict
import numpy as np

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(BASE, "advanced_bias_analysis"))
import utils

CACHE = os.path.join(BASE, "verification", "cache")
SUFFIX = "_weighted"
WAVES = ["26", "34"]
MODELS = {
    "base":          "mistral_7b_text",
    "instruct_base": "mistral_7b_instruct_baseprompt",
    "instruct_chat": "mistral_latest",
}


def load_human(wave):
    with open(os.path.join(CACHE, f"human_W{wave}{SUFFIX}.pkl"), "rb") as f:
        return pickle.load(f)[0]


def main():
    # llm[model][wave] = (index, qmeta)
    llm = {}
    for tag, model in MODELS.items():
        llm[tag] = {}
        utils.MODEL_TAG = model
        for w in WAVES:
            try:
                llm[tag][w] = utils.build_llm_index(w, max_level=2)[0]
            except Exception as e:
                print(f"{model} W{w}: index error {e}")
                llm[tag][w] = {}

    # accumulators[tag][dim] = [keep, should, contested]
    acc = {t: defaultdict(lambda: [0, 0, 0]) for t in MODELS}
    n_cells = defaultdict(int)   # per tag, collapsed cells used

    for w in WAVES:
        human = load_human(w)
        h_avg = human.get(utils.AVG_PROFILE, {})
        # iterate pairs from the base model index (superset of cells)
        base_idx = llm["base"][w]
        for profile, qdists in base_idx.items():
            if len(profile) != 2:
                continue
            (da, va), (db, vb) = sorted(profile)
            pa, pb = frozenset([(da, va)]), frozenset([(db, vb)])
            hp, ha, hb = human.get(profile, {}), human.get(pa, {}), human.get(pb, {})
            for qid in qdists:
                # require ALL models to have this pair-cell + its singles
                ok = True
                per = {}
                for t in MODELS:
                    idx = llm[t][w]
                    lp = idx.get(profile, {}).get(qid)
                    la = idx.get(pa, {}).get(qid)
                    lb = idx.get(pb, {}).get(qid)
                    if lp is None or la is None or lb is None:
                        ok = False; break
                    per[t] = (lp, la, lb)
                if not ok:
                    continue
                h_p, h_a, h_b, h_pop = (hp.get(qid), ha.get(qid), hb.get(qid), h_avg.get(qid))
                if any(x is None for x in (h_p, h_a, h_b, h_pop)):
                    continue
                L = len(h_p)
                if any(len(x) != L for x in (h_a, h_b, h_pop)):
                    continue
                if any(len(v) != L for t in per for v in per[t]):
                    continue

                tv_a, tv_b = utils.tv(h_a, h_pop), utils.tv(h_b, h_pop)
                human_dom = da if tv_a >= tv_b else db

                for t in MODELS:
                    lp, la, lb = per[t]
                    e_p, e_a, e_b = lp - h_p, la - h_a, lb - h_b
                    cos_a, cos_b = utils.cosine_sim(e_p, e_a), utils.cosine_sim(e_p, e_b)
                    if max(cos_a, cos_b) < utils.cosine_sim(e_p, e_a + e_b):
                        continue   # additive wins, no collapse
                    winner = da if cos_a >= cos_b else db
                    n_cells[t] += 1
                    for d in (da, db):
                        acc[t][d][2] += 1
                    acc[t][winner][0] += 1
                    acc[t][human_dom][1] += 1

    # report
    print(f"\nDe-confound keep-rate deltas (W26+W34, weighted, intersected cells)\n")
    print(f"{'':16s}{'base':>22s}{'instruct_base(NEW)':>22s}{'instruct_chat':>22s}")
    print(f"{'(model/prompt)':16s}{'base/base':>22s}{'instr/base':>22s}{'instr/instr':>22s}")
    print(f"collapsed cells " + "".join(f"{n_cells[t]:>22,}" for t in MODELS))
    for dim in ["Race", "Religion", "Gender", "Income", "Age"]:
        cells = []
        for t in MODELS:
            a = acc[t]
            if dim in a and a[dim][2] > 0:
                keep = a[dim][0] / a[dim][2]
                should = a[dim][1] / a[dim][2]
                cells.append(f"{keep-should:+.1%} (n={a[dim][2]:,})")
            else:
                cells.append("- (no pairs)")
        print(f"{dim:16s}" + "".join(f"{c:>22s}" for c in cells))


if __name__ == "__main__":
    main()
