# Technical-review analysis batch — results summary (2026-07-09)

Scripts + full output: `verification/techreview/` (`RESULTS.log`). All analyses
use the paper's conventions (survey-weighted caches, n>=20 gates, the exact
contest of `score_to_rows.py`). Fidelity checks: the batch reproduces the
published 78.4% flagship collapse share, 0.062 recombination TV, 0.84 null
cosine, rho=0.385, and d3 index 0.98 exactly.

## Results by report item

**#2 — small-norm robustness (CLOSED).** Best-single win rate is flat across
all ten deciles of ||e_AB|| (77.3–81.0%); excluding ||e_AB||<0.05 moves the
pooled 78.4% by 0.1 pt (1.5% of cells). The cosine zero-guard (<1e-10) fired
in 0 of 363,714 contests. Collapse is not a low-SNR artifact.

**#7 — Kish n_eff sensitivity (CLOSED, with caveat).** Mean design effect
n/n_eff ≈ 1.8–2.0 by depth. Noise-corrected distinctiveness growth d3/d1 is
2.87x under raw n and 2.86x under n_eff — the depth-growth conclusion is
weight-design-invariant. (Estimator here: weighted dists, 15 waves; ED Fig 1's
published 2.5x uses its own basis — the sensitivity conclusion, not the level,
is the point.) Note the paper's floors are *empirical* split-half quantities
that already embed design effects; only analytic-formula uses would scale by
sqrt(deff)≈1.35–1.43.

**#8b — clipping activation (CLOSED).** The additive-truth distribution has at
least one negative component (clipping active) in 15.1% of pair cells at
n>=100 (38,047/252,099).

**#9 — leakage-free cross-fit of Fig 3 (CLOSED).** In-sample recombination
mean TV 0.062 (exactly the published value); cross-fitted (singles+pop from
one respondent half, pair target from the other) 0.106 — still *below* the
half-sample pure-noise benchmark (weighted split-half d2 TV 0.134). The low
recombination error is not shared-sampling-noise optimism.

**#10 — within-family depth growth (CLOSED).** Growth persists in every one of
the 35 dimension-triple families: median 2.90x, range 2.07–4.56x. Not a
cell-composition/selection artifact.

**#11 — per-model depth-3 index (CLOSED).** Endpoints from each model's own
noise: index 0.92–0.98 for all seven d3 models (gpt-4o-mini 0.98, mistral
0.97, haiku/gemma/llama/gpt-5.5 0.94, sonnet-5 0.92). Ready for Table 1.

**#12 — matched triple subset (CLOSED).** On the strict common-cell set
(13,587 identical Age/Gender/Race/Income triples, W26+W34): frontier models
8.6% (Sonnet 5) and 8.9% (GPT-5.5) additive share vs 9.5–11.6% for the five
prior-generation models. The two-generations claim survives exact matching;
frontier models are, if anything, marginally more collapsed.

**#16b — hierarchy rho uncertainty (CLOSED).** Wave-cluster bootstrap:
flagship mean rho +0.385 [95% CI +0.19, +0.57]; haiku +0.459 [+0.29, +0.62];
gemma +0.551 [+0.39, +0.71]; gpt-4o +0.480 [+0.30, +0.65]. All CIs exclude
zero and sit far below 1; per-dimension mean-TV CIs also computed.

**#17 — retention conditioning + value level (CLOSED; NEW FINDING).**
Collapse rates are tightly banded across all 21 dimension pairs (75.5–83.1%),
so retention deltas are not a denominator-selection artifact. Value-level:
the race/religion suppression is *carried by minority values* —
Black −19.0, Hispanic −22.3, Asian −19.0, Mixed −19.3, Other −22.5 pts
(kept-minus-dominant), against White +16.3; Atheist −20.7 and Jewish −23.0
against Protestant +6.4 and Catholic +3.9. The model under-retains precisely
the minority identities within the suppressed dimensions.

**#18b — base-condition semantic compliance (CLOSED).** Per-dimension bias
magnitudes are near-identical for base-ckpt vs instruct-ckpt under the base
prompt (0.24–0.27 TV); within-dimension differentiation under the base
checkpoint (0.19–0.23 pairwise TV) exceeds the human reference (0.07–0.15).
The base checkpoint demonstrably conditions on demographics; suppression in
base is not "base ignores demographics."

**#20 — entropy robustness (CLOSED).** Over-dispersion is positive with
wave-cluster CIs excluding zero in every variant and monotone in depth:
raw dH +0.180/+0.206/+0.234 (d1/2/3); n>=100 subset same; normalized H/logK
+0.13/+0.15/+0.17; Herfindahl impurity +0.075/+0.092/+0.111;
refusal-excluded +0.088/+0.107/+0.126. The exploratory flag can be softened.

**#21b — residualized null (CLOSED).** Removing the population-bias component
e_pop from both sides collapses the random-unrelated null from median cosine
0.839 to 0.280 — the high null floor is indeed (almost entirely) the shared
population bias, as the paper's mechanistic interpretation claims.

## Suggested manuscript uses (subject to the 5,000-word budget)
1. Table 1: add Index_d3 column (0.92–0.98) — item #11.
2. Methods one-liners: #2 (flat across norm deciles; zero-guard 0 cases),
   #8b (15.1% clip rate), #9 (cross-fitted 0.106 < 0.134 half-sample noise),
   #12 (matched-subset shares), #16b (rho CI), #21b (residualized null 0.28).
3. Consider promoting the #17 value-level result — minority-value suppression
   is a substantive sharpening of the fairness claim, likely worth main text
   or an Extended Data panel.
4. #20 allows strengthening the "exploratory" hedge on the entropy section.
