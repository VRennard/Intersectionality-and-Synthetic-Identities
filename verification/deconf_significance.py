"""
Is the Race/etc. suppression ('muting') significantly DIFFERENT between
mistral_latest (instruct) and mistral_7b_text (base)?

Paired test on MATCHED cells, all 15 waves: for each pair-cell present in both
models where dim D is contested and BOTH models collapse, record whether each
model keeps D. On matched cells the human-dominant label is identical, so
(delta_instruct - delta_base) == (keeprate_instruct - keeprate_base). We
bootstrap that difference clustered by WAVE (the independent unit), B=10000,
and report the per-dimension difference, 95% CI, and two-sided p.

Only Age/Gender/Race/Income are comparable (mistral_latest lacks the other pairs).
"""
import os, sys, pickle
from collections import defaultdict
import numpy as np

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(BASE, "advanced_bias_analysis"))
import utils

CACHE = os.path.join(BASE, "verification", "cache")
SUF = "_weighted"
RNG = np.random.default_rng(7)
B = 10000
ALL = ["26","27","29","32","34","36","41","42","43","45","49","50","54","82","92"]
# argv: BASE_M INST_M [wave wave ...]
BASE_M = sys.argv[1] if len(sys.argv) > 1 else "mistral_7b_text"
INST_M = sys.argv[2] if len(sys.argv) > 2 else "mistral_latest"
_waves = [a for a in sys.argv[3:]]
WAVES = _waves if _waves else ALL
DIMS = ["Race", "Religion", "Education", "Income", "Gender", "Age", "Political Party"]


def load_human(w):
    return pickle.load(open(os.path.join(CACHE, f"human_W{w}{SUF}.pkl"), "rb"))[0]


def collapse_winner(lp, la, lb, h_p, h_a, h_b, da, db):
    e_p, e_a, e_b = lp - h_p, la - h_a, lb - h_b
    ca, cb = utils.cosine_sim(e_p, e_a), utils.cosine_sim(e_p, e_b)
    if max(ca, cb) < utils.cosine_sim(e_p, e_a + e_b):
        return None  # additive, no collapse
    return da if ca >= cb else db


def main():
    # per wave, per dim: list of per-cell (instruct_keep - base_keep)
    perwave = {d: defaultdict(list) for d in DIMS}
    for w in WAVES:
        human = load_human(w); h_avg = human.get(utils.AVG_PROFILE, {})
        utils.MODEL_TAG = BASE_M;  bidx = utils.build_llm_index(w, max_level=2)[0]
        utils.MODEL_TAG = INST_M;  iidx = utils.build_llm_index(w, max_level=2)[0]
        for profile, qd in bidx.items():
            if len(profile) != 2:
                continue
            (da, va), (db, vb) = sorted(profile)
            pa, pb = frozenset([(da, va)]), frozenset([(db, vb)])
            hp, ha, hb = human.get(profile, {}), human.get(pa, {}), human.get(pb, {})
            for qid in qd:
                bl = (bidx.get(profile,{}).get(qid), bidx.get(pa,{}).get(qid), bidx.get(pb,{}).get(qid))
                il = (iidx.get(profile,{}).get(qid), iidx.get(pa,{}).get(qid), iidx.get(pb,{}).get(qid))
                if any(x is None for x in bl) or any(x is None for x in il):
                    continue
                h_p, h_a, h_b, h_pop = hp.get(qid), ha.get(qid), hb.get(qid), h_avg.get(qid)
                if any(x is None for x in (h_p, h_a, h_b, h_pop)):
                    continue
                L = len(h_p)
                if any(len(x) != L for x in (h_a, h_b, h_pop, *bl, *il)):
                    continue
                wb = collapse_winner(*bl, h_p, h_a, h_b, da, db)
                wi = collapse_winner(*il, h_p, h_a, h_b, da, db)
                if wb is None or wi is None:   # require BOTH to collapse
                    continue
                for d in (da, db):
                    if d in DIMS:
                        perwave[d][w].append(int(wi == d) - int(wb == d))

    print(f"Paired matched-cell test: instruct ({INST_M}) - base ({BASE_M})")
    print(f"15-wave, weighted, wave-clustered bootstrap B={B}\n")
    print(f"{'dim':8s}{'diff(keep)':>12s}{'95% CI':>22s}{'p(2-sided)':>12s}{'cells':>9s}")
    for d in DIMS:
        waves = [w for w in WAVES if perwave[d][w]]
        if not waves:
            print(f"{d:8s}{'- (no shared pairs)':>57s}")
            continue
        arr = [np.array(perwave[d][w]) for w in waves]
        n = sum(len(a) for a in arr)
        obs = np.concatenate(arr).mean()
        boots = []
        for _ in range(B):
            idx = RNG.integers(0, len(waves), len(waves))
            s = np.concatenate([arr[i] for i in idx])
            boots.append(s.mean())
        boots = np.array(boots)
        lo, hi = np.percentile(boots, 2.5), np.percentile(boots, 97.5)
        p = 2 * min((boots >= 0).mean(), (boots <= 0).mean())
        sig = " *" if (lo > 0 or hi < 0) else ""
        print(f"{d:8s}{obs:+11.1%}{f'[{lo:+.1%}, {hi:+.1%}]':>22s}{p:>12.3f}{n:>9,}{sig}")
    print("\nInterpretation: diff>0 = instruct keeps the dim MORE than base; "
          "negative delta = suppression. A CI crossing 0 = no significant "
          "base/instruct difference in muting on matched cells.")


if __name__ == "__main__":
    main()
