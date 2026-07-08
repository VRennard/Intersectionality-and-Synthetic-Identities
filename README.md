# Code and data for: "Large language models simulate intersectional identities with a budget of one to two dimensions"

> **Code-only mirror.** This repository contains the code and documentation of the
> replication package. The data (21.1M raw model-output records, canonical analysis
> artifacts, weighted ground-truth caches, figure intermediates) lives in the Zenodo
> deposit: https://doi.org/10.5281/zenodo.21267989 — download and unpack
> `surveysgpt_deposit_full.tar.gz` to run anything that touches data.

Replication package for the Nature Human Behaviour submission. It contains
the full pipeline — survey-response generation with LLMs, scoring against
Pew American Trends Panel (ATP) ground truth, all canonical analysis
artifacts, and the scripts that produce every figure and statistic in the
paper — plus the raw model outputs (21.1M records; see `DATA_MANIFEST.md`
for the per-tag audit).

## Quick start: reproduce every figure (minutes)

```
pip install numpy pandas matplotlib     # or: pip install -r requirements.txt
bash run_all_figures.sh
```

Figures are written next to the scripts (`paper_figs/*.png|pdf`,
`verification/fig_spine.*`). `verification/fig_spine.py` also prints the
paper's TV-distance spine table.

## Layout

```
DATA_MANIFEST.md          authoritative audit of the 21.1M raw records
requirements.txt          python dependencies (numpy/pandas/matplotlib for analysis;
                          openai/anthropic/aiohttp only for regenerating model outputs)
*.py, *.sh                generation pipeline: llm_prompt_survey.py (core prompting),
                          batch_api_runner*.py (OpenAI/Anthropic batch APIs),
                          gen_* / augment_* / launch_* / run_* (campaign runners),
                          indiv_sampling_* (individual-persona paradigm, INDIV_SAMPLING.md),
                          logprob_readout.py (log-probability paradigm)
data/results/{tag}/       RAW MODEL OUTPUTS — one JSONL per wave per model/condition;
                          tags and record counts documented in DATA_MANIFEST.md
data/responses/           aggregated human marginals per question (15 ATP waves)
data/demographics/        aggregated human demographic marginals
human_resp/               EMPTY by design — Pew microdata cannot be redistributed;
                          human_resp/README.md explains how to obtain and place it
verification/             analysis scripts + the canonical artifacts they produce
                          (collapse/direction/spine/noise-floor/additivity/rigor JSONs);
                          cache/ holds the derived weighted human ground truth
                          (aggregate distributions only — no respondent-level data)
paper_figs/               figure scripts (fig_*.py) + computed intermediates (*.npz, *.json)
advanced_bias_analysis/   utils.py — shared index builder over raw results
indiv_cells*/, interesting_questions/   sampling manifests
```

## Reproducing the full chain from raw data

1. *(Optional)* regenerate model outputs with the generation scripts.
   Requires `OPENAI_API_KEY` / `ANTHROPIC_API_KEY` (all keys in this deposit
   are placeholders) and, for the open-weight models, a GPU box running
   ollama. Skippable — `data/results/` already contains all records.
2. Obtain Pew microdata (see `human_resp/README.md`), then
   `verification/rebuild_weighted_caches.py` rebuilds the weighted
   ground-truth caches (already included).
3. `verification/` scripts recompute the canonical artifacts from
   `data/results/` + caches: `score_to_rows.py`, `collapse_direction.py`,
   `verify_spine_collapse.py`, `noise_floor.py`,
   `additivity_null_baseline.py`, `human_additivity_calibration.py`,
   `human_rigor_*.py`, `stoch_bands.py`, `bootstrap_permutation.py`,
   `weight_sensitivity.py`, `steering_dominance.py`. (Hours of CPU; all
   outputs already included, so this is verification, not a prerequisite.)
4. `paper_figs/compute_*.py` (and the `--recompute` flags of
   `fig_alphabeta*.py`) rebuild the figure intermediates.
5. `bash run_all_figures.sh` regenerates all main and extended-data figures.

Survey-weighted human baselines are the paper's primary analysis; unweighted
variants of the artifacts (`*_weighted` absent) are included for the
robustness comparisons and can be reproduced with the scripts' `--unweighted`
/ `SUFFIX=""` paths.

## Notes

- Raw JSONL files may contain occasional failed-request records alongside
  successful ones; `DATA_MANIFEST.md` counts successful records only, and the
  scoring scripts skip failures.
