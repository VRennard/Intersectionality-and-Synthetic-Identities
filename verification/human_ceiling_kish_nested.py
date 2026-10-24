#!/usr/bin/env python
"""Human additivity ceiling under Kish effective n and nested singles (see HUMAN_CEILING_KISH_NESTED.md).
Human additivity contest: published reference vs three variants.
Scratch reimplementation of the human side of human_and_llm_steering_contest()
in code/verification/human_additivity_calibration.py (weighted basis).
Variants of the additive-truth reference draw:
  base   : multinomial at the cell's RAW n (published construction)
  kish   : multinomial at the Kish effective n of the pair cell on that question
  nested : synthetic singles inherit the synthetic pair draw in proportion to the
           pair's weight share inside each single (real singles inherit the real pair)
  both   : kish + nested
Real human shares are recomputed for verification against the published 44.4% / 47.8%.
"""
import sys, os, json, itertools, time
import numpy as np, pandas as pd
PKG = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # repository root
os.chdir(PKG + '/verification')
sys.path.insert(0, PKG); sys.path.insert(0, PKG + '/advanced_bias_analysis'); sys.path.insert(0, PKG + '/verification')
import utils
from human_additivity_calibration import wave_list, load_options
utils.USE_WEIGHTS = True
RNG = np.random.default_rng(2026)
OUT = os.path.join(PKG, 'verification', 'human_ceiling_kish_nested.json')
LOG = os.path.join(PKG, 'verification', 'human_ceiling_kish_nested.log')

def cos(a, b):
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return float(a @ b / (na * nb)) if na > 0 and nb > 0 else 0.0

def contest(s_ab, s_a, s_b):
    return cos(s_ab, s_a + s_b) > max(cos(s_ab, s_a), cos(s_ab, s_b))

THR = (100, 200)
keys = ['human', 'base', 'kish', 'nested', 'both']
res = {t: {k: 0 for k in keys} | {'n': 0} for t in THR}
neff_ratio = []
t0 = time.time()
for wave in wave_list():
    options = load_options(wave)
    df = pd.read_csv(f'{PKG}/human_resp/American_Trends_Panel_W{wave}/responses.csv', low_memory=False)
    wcol = utils._weight_col(df, wave)
    w_all = pd.to_numeric(df[wcol], errors='coerce').to_numpy()
    qids = [q for q in options if q in df.columns]
    # encode each question as option index (-1 = not an option / missing)
    enc = {}
    for q in qids:
        opts = list(options[q]); lut = {o: i for i, o in enumerate(opts)}
        enc[q] = (np.array([lut.get(v, -1) for v in df[q].tolist()], dtype=int), len(opts))
    # single masks (raw n >= 100 over all respondents, as in the script)
    val_masks = {}
    for dim, col in utils.DIM_TO_COL.items():
        if col not in df.columns: continue
        for val in utils.DIM_VALUES[dim]:
            if val in utils.IGNORE_VALUES.get(dim, set()): continue
            m = (df[col] == val).to_numpy()
            if m.sum() >= 100: val_masks[(dim, val)] = m

    def wdist(mask, q, min_n):
        idx, K = enc[q]; ok = mask & (idx >= 0) & np.isfinite(w_all)
        n = int(ok.sum())
        if n < min_n: return None, 0, 0.0, 0.0
        ww = w_all[ok]; c = np.bincount(idx[ok], weights=ww, minlength=K)
        s = c.sum()
        if s <= 0: return None, 0, 0.0, 0.0
        neff = ww.sum() ** 2 / (ww ** 2).sum()
        return c / s, n, float(s), float(neff)

    pop = {}
    allm = np.ones(len(df), bool)
    for q in qids:
        d, n, _, _ = wdist(allm, q, 20)
        if d is not None: pop[q] = d
    singles = {k: {} for k in val_masks}
    for k, m in val_masks.items():
        for q in qids:
            d, n, W, _ = wdist(m, q, 100)
            if d is not None and q in pop: singles[k][q] = (d, W)
    ncell = 0
    for a, b in itertools.combinations(sorted(val_masks), 2):
        if a[0] == b[0]: continue
        m = val_masks[a] & val_masks[b]
        if m.sum() < 100: continue
        m_aonly = val_masks[a] & ~val_masks[b]
        m_bonly = val_masks[b] & ~val_masks[a]
        for q in qids:
            if q not in pop or q not in singles[a] or q not in singles[b]: continue
            d_ab, n_ab, W_ab, neff_ab = wdist(m, q, 100)
            if d_ab is None: continue
            p = pop[q]
            d_a, W_a = singles[a][q]; d_b, W_b = singles[b][q]
            s_ab, s_a, s_b = d_ab - p, d_a - p, d_b - p
            # A-only / B-only parts (for nesting); weight sums must satisfy W_a = W_aonly + W_ab
            d_ao, _, W_ao, _ = wdist(m_aonly, q, 1)
            d_bo, _, W_bo, _ = wdist(m_bonly, q, 1)
            p_true = np.clip(p + s_a + s_b, 0, None)
            if p_true.sum() <= 0: continue
            p_true = p_true / p_true.sum()
            neff_ratio.append(neff_ab / n_ab)
            # draws
            draw_raw = RNG.multinomial(n_ab, p_true) / n_ab
            n_k = max(1, int(round(neff_ab)))
            draw_kish = RNG.multinomial(n_k, p_true) / n_k
            def nested_singles(p_ab_syn):
                if d_ao is None or d_bo is None: return None
                pa = (W_ao * d_ao + W_ab * p_ab_syn) / (W_ao + W_ab)
                pb = (W_bo * d_bo + W_ab * p_ab_syn) / (W_bo + W_ab)
                return pa - p, pb - p
            outcomes = {
                'human': contest(s_ab, s_a, s_b),
                'base': contest(draw_raw - p, s_a, s_b),
                'kish': contest(draw_kish - p, s_a, s_b),
            }
            ns = nested_singles(draw_raw)
            outcomes['nested'] = contest(draw_raw - p, *ns) if ns else None
            ns2 = nested_singles(draw_kish)
            outcomes['both'] = contest(draw_kish - p, *ns2) if ns2 else None
            for t in THR:
                if n_ab >= t:
                    res[t]['n'] += 1
                    for k, v in outcomes.items():
                        if v: res[t][k] += 1
            ncell += 1
    with open(LOG, 'a') as f:
        f.write(f'W{wave}: {ncell} cells, cumulative n100={res[100]["n"]}, {time.time()-t0:.0f}s\n')

summary = {}
for t in THR:
    n = res[t]['n']
    summary[f'n{t}'] = {k: res[t][k] / n for k in keys} | {'cells': n}
    summary[f'n{t}']['human_over_base'] = summary[f'n{t}']['human'] / summary[f'n{t}']['base']
    for k in ('kish', 'nested', 'both'):
        summary[f'n{t}'][f'human_over_{k}'] = summary[f'n{t}']['human'] / summary[f'n{t}'][k]
summary['neff_over_n_median'] = float(np.median(neff_ratio))
summary['note'] = __doc__
json.dump(summary, open(OUT, 'w'), indent=1)
with open(LOG, 'a') as f:
    f.write('DONE\n' + json.dumps(summary, indent=1) + '\n')
print(json.dumps(summary, indent=1))
