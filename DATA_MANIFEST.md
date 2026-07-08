# Data manifest — final audit 2026-07-06

Successful records per tag in `data/results/` (personas files excluded).

## PRIMARY PIPELINE (Table 1; 15.7M)
- `claude-haiku-4-5-20251001`: 3,702,725
- `claude-sonnet-5`: 184,694
- `gemma2_9b`: 2,534,060
- `gpt-4o`: 856,779
- `gpt-4o-mini`: 2,760,395
- `gpt-5.5-2026-04-23`: 185,302
- `llama3_1_8b_instruct_q4`: 2,737,801
- `mistral_latest`: 2,737,869
- **subtotal: 15,699,625**

## BASE-VERSUS-TUNED (2.1M)
- `llama3_1_8b_instruct_base`: 687,914
- `llama3_1_8b_text`: 688,524
- `mistral_7b_instruct_baseprompt`: 55,803
- `mistral_7b_text`: 688,555
- **subtotal: 2,120,796**

## LOG-PROBABILITY READOUT (1.3M)
- `gemma2_9b_logprob`: 423,744
- `llama3_1_8b_instruct_q4_logprob`: 423,744
- `mistral_latest_logprob`: 423,744
- **subtotal: 1,271,232**

## INDIVIDUAL SAMPLING (1.8M)
- `claude-haiku-4-5-20251001_indiv`: 48,720
- `gemma2_9b_indiv`: 420,608
- `gemma2_9b_indiv_profilesample10`: 51,390
- `gpt-4o-mini_indiv`: 40,596
- `llama3_1_8b_instruct_q4_indiv`: 420,610
- `mistral_latest_indiv`: 418,090
- `mistral_latest_indiv_q10`: 420,603
- **subtotal: 1,820,617**

## RELIABILITY / CONTROLS (0.2M)
- `claude-haiku-4-5-20251001_promptA`: 1,700
- `claude-haiku-4-5-20251001_promptB`: 1,700
- `claude-haiku-4-5-20251001_promptC`: 1,000
- `claude-haiku-4-5-20251001_promptI`: 1,000
- `claude-haiku-4-5-20251001_promptR`: 1,000
- `gemma2_9b_party`: 5,272
- `gemma2_9b_promptA`: 4,227
- `gemma2_9b_promptB`: 1,563
- `gemma2_9b_promptR`: 999
- `gpt-4o-mini_promptC`: 2,720
- `gpt-4o-mini_promptI`: 2,720
- `stoch_claude-haiku-4-5-20251001`: 2,487
- `stoch_claude-sonnet-5`: 14,641
- `stoch_gemma2_9b`: 953
- `stoch_gpt-4o`: 2,496
- `stoch_gpt-5.5-2026-04-23`: 6,384
- `stoch_llama3_1_8b_instruct_q4`: 1,588
- `stoch_mistral_latest`: 1,142
- `stoch_test`: 168,022
- **subtotal: 221,614**

## TOTAL: 21,133,884 (21.1M exact)

## Canonical analysis artifacts (verification/)
- `collapse_results_weighted.json` — per-model/wave/depth contest rows; 8 paper models
  (gemma religion-complete rows + claude-sonnet-5 spliced 2026-07-06; mistral-small_24b removed;
  backup: `.preaudit_bak`)
- `collapse_direction_weighted.json` — per-cell direction rows (paper direction numbers
  recomputed from this file 2026-07-06)
- `spine_results.json` — TV by model/wave/depth (recomputed 2026-07-06 with gemma religion triples)
- `cache/human_W{w}_weighted.pkl` — weighted human ground truth incl. triples
- `cache/rows_mistral_latest_indiv_final.json` — q20 dose-response rows (74.4%)

## Known gaps (deliberate)
- Mistral-Small-24B: raw records never on this machine — model REMOVED from paper 2026-07-04.
- Religion doubles W43/45/49/50/54/82/92 (~300K cells): pending final generation campaign
  (runner: `augment_religion_doubles.py`, auto-resumes). Affects no current paper number;
  religion-pair predictors for those waves' d3 cells pool available pairs.
- ED Fig 2 (per-model replications) and ED Fig 3 (W49 forensics): legends in paper,
  figure PDFs not yet assembled (marked \TODO in main.tex).
