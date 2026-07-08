"""Store full (alpha,beta,gamma) coefficient vectors for depth-3 cells
(model bias space + human steering space), for the 3D/ternary figures."""
import os, sys, pickle, itertools
import numpy as np
import style
sys.path.insert(0, os.path.join(style.BASE, "advanced_bias_analysis"))
import utils
utils.USE_WEIGHTS = True
WAVES = ("26","27","29","32","34","36","41","42","43","45","50","54","82","92")
MODEL = "gpt-4o-mini"; AVG = utils.AVG_PROFILE

def solve3(t, us):
    G = np.array([[u @ v for v in us] for u in us]); b = np.array([t @ u for u in us])
    try: return np.linalg.solve(G, b)
    except np.linalg.LinAlgError: return None

def basis_ok(us):
    for u, v in itertools.combinations(us, 2):
        d = np.linalg.norm(u) * np.linalg.norm(v)
        if d == 0 or abs(u @ v) / d > 0.95: return False
    return True

def model_pts():
    utils.MODEL_TAG = MODEL; pts = []
    for w in WAVES:
        human, _ = pickle.load(open(os.path.join(style.BASE, f"verification/cache/human_W{w}_weighted.pkl"), "rb"))
        llm, _ = utils.build_llm_index(w, max_level=3)
        for prof in llm:
            if len(prof) != 3: continue
            singles = [frozenset([f]) for f in sorted(prof)]
            for q, lab in llm[prof].items():
                hab = human.get(prof, {}).get(q)
                if hab is None or len(hab) != len(lab): continue
                es = []
                for s_ in singles:
                    lp = llm.get(s_, {}).get(q); hp = human.get(s_, {}).get(q)
                    if lp is None or hp is None or len(lp) != len(lab): es = None; break
                    es.append(lp - hp)
                if es is None or not basis_ok(es): continue
                c = solve3(lab - hab, es)
                if c is not None: pts.append(c)
        print(f"  model W{w} ({len(pts):,})", flush=True)
    return np.array(pts)

def human_pts():
    utils.MODEL_TAG = MODEL; utils.MIN_HUMAN_N = 100; pts = []
    for w in WAVES:
        llm, qmeta = utils.build_llm_index(w, max_level=3)
        human = utils.load_human_index(w, list(qmeta.keys()), qmeta, max_level=3)
        pop = human.get(AVG, {})
        for prof in human:
            if not hasattr(prof, "__len__") or len(prof) != 3: continue
            singles = [frozenset([f]) for f in sorted(prof)]
            for q, pab in human[prof].items():
                pq = pop.get(q)
                if pq is None or len(pq) != len(pab): continue
                ss = []
                for s_ in singles:
                    hp = human.get(s_, {}).get(q)
                    if hp is None or len(hp) != len(pab): ss = None; break
                    ss.append(hp - pq)
                if ss is None or not basis_ok(ss): continue
                c = solve3(pab - pq, ss)
                if c is not None: pts.append(c)
        print(f"  human W{w} ({len(pts):,})", flush=True)
    return np.array(pts)

mp = model_pts(); hp = human_pts()
np.savez_compressed(os.path.join(style.HERE, "abc_points.npz"), mp=mp, hp=hp)
print(f"saved: model {len(mp):,}, human {len(hp):,}")
