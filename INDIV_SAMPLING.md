# Individual-Sampling Elicitation Check (referee Comment 1)

The make-or-break robustness check: does the **collapse** statistic replicate
when we elicit responses the way the criticised silicon-sampling literature
does — sampling individual personas — rather than asking the model for an
aggregate distribution?

All 12.2M existing distributions come from one elicitation: *"act as a
researcher, distribute 1,000 hypothetical respondents across the options."*
A Bisbee/Argyle-corner referee will say collapse might be a property of that
aggregate, meta-cognitive task, not of demographic conditioning. This run
answers that with the field's own paradigm.

## Design (everything fixed except elicitation mode)

- **Elicitation:** one call = one simulated persona answering the **entire wave**
  (a survey, not a question), choosing exactly one option per question.
  `N` personas per cell; the per-(cell, question) distribution is the histogram
  of chosen options across the `N` personas.
- **Scope:** W26 + W34 (reuses the GPT-5.5 subset → free frontier cross-check),
  singles + pairs over the 6 protagonist dimensions (Age, Gender, Race, Income,
  Political Party, Religion) → 280 cells/wave. Fully covers race×party and
  religion×party, the cells the contest cares about most.
- **Output:** `data/results/{model}_indiv/W{wave}.jsonl` in the **exact schema**
  of the aggregate pipeline, so the existing collapse contest runs unchanged.
- **Cost:** ~$20/wave at 160 personas (gpt-4o-mini batch) → ~$40 total.

## Commands

```bash
# 0. sanity-check cost/call counts (no API calls; works anywhere)
python indiv_sampling_runner.py estimate --wave 26 --model gpt-4o-mini

# 1. submit (needs OPENAI_API_KEY; run on the Windows box that has openai)
python indiv_sampling_runner.py submit --wave 26 --model gpt-4o-mini --personas 160
python indiv_sampling_runner.py submit --wave 34 --model gpt-4o-mini --personas 160

# 2. fetch when the batch completes (resumable; re-run to top up personas)
python indiv_sampling_runner.py fetch --wave 26 --model gpt-4o-mini
python indiv_sampling_runner.py fetch --wave 34 --model gpt-4o-mini

# 3. run the EXISTING collapse contest on the new tag (no code change)
cd verification && python verify_spine_collapse.py gpt-4o-mini_indiv
```

`verify_spine_collapse.py` prints the depth-2 collapse share
(`best_single` vs `additive` win rate) for `gpt-4o-mini_indiv`. Compare it to
the aggregate baseline (`gpt-4o-mini`: best-single wins **78.3%** at depth 2).

## RESULT (gpt-4o-mini, W26+W34, 2026-06-23)

**PASSED — collapse is elicitation-invariant.** Individual sampling
(89,600 personas, 160/cell, temp 1.0, 0 failures): depth-2 best-single wins
**77.2%** (cos 0.934) vs additive 22.8% (cos 0.853). The aggregate baseline is
**78.3%** — a **1.1-point** difference, well inside the ±5 band. Absolute TV is
higher under individual sampling (d1 0.343 / d2 0.347 vs 0.21/0.23 aggregate),
as expected from one-hard-pick-per-persona at n=160 (multinomial noise, peaked
histograms) — irrelevant to the collapse contest, which compares predictors of
the *same* bias vector and is noise-floor-immune. Outputs in
`data/results/gpt-4o-mini_indiv/`.

## RESULT (gemma2_9b log-prob, W26+W34, 2026-06-24)

**Collapse confirmed under the third paradigm.** Log-prob readout
(`logprob_readout.py`, Ollama top_logprobs, temp 0, softmax over option
letters): depth-2 best-single wins **82.3%** (cos 0.951) vs additive 17.7%
(cos 0.793). Gemma's own aggregate baseline on the same two waves is **76.2%**
— log-prob moves +6 pts in the *reinforcing* direction (temp-0 readout gives
peakier distributions → single feature wins even more). Depth-2 TV is high
(0.479) for the same peaking reason; irrelevant to the (noise-floor-immune)
contest. Outputs in `data/results/gemma2_9b_logprob/`.

## Cross-paradigm summary (depth-2 best-single win rate)

| Model | Aggregate | Individual sampling | Log-prob (full 15w) |
|---|---|---|---|
| gpt-4o-mini | 78.3% | 77.2% (W26+34) | — |
| gemma2_9b | 74.8% | 10%×15w @100p (running) | **83.1%** |
| mistral_latest | 81.9% | 10%×15w @100p (queued) | **82.3%** |
| llama3_1_8b_instruct_q4 | 80.4% | 10%×15w @100p (queued) | **74.4%** |
| claude-haiku-4-5 | (have agg) | W26 @100p cached (running) | — |

All paradigms land in **74–83%** → **collapse is a property of demographic
conditioning, not the elicitation format.** Per-model log-prob deltas go both
ways but never escape the band: mistral 81.9→82.3 (+0.4, tightest), gemma
74.8→83.1 (+8.3), llama 80.4→74.4 (−6.0). Full 15-wave log-prob run on pod
`yearling_jade_trout`, 2026-06-24/25.

**Open-weights individual sampling (3rd paradigm) launched 2026-06-25:**
gemma/mistral/llama via Ollama, all singles + 10% of pairs per wave
(`indiv_cells_10pct`, seed 0 — identical cells across models), 100 personas,
all 15 waves. ~768 cells × 100 × 15 ≈ 77K calls/model, ~55h/model on the 4090
(`run_indiv_all.sh`, sequential). Tags `{model}_indiv`.

Throughput on one RTX 4090 (Ollama): individual sampling **0.39 calls/sec**
(whole-wave generation) vs log-prob **~20–60 calls/sec** (1 forward pass/q,
model-dependent: mistral-7B ~58, gemma-9B ~21) — the gap is why open-weights
coverage goes through log-prob, not sampling.

## Operational lesson — local log-prob needs a context cap (load-bearing)

Ollama sizes the KV cache as `num_ctx × OLLAMA_NUM_PARALLEL`. The log-prob
prompt is one question (~300 tokens), but if you don't cap context, Ollama
loads at the model's **default** context — gemma 8K (fine), but **mistral 32K**
and **llama-3.1 128K** × 8 parallel slots overflow 24 GB VRAM → the model
silently offloads to CPU (5–15× slower) and can wedge the runner (a zombie
`llama-server` holding all VRAM that survives `pkill ollama`).
**Per-request `num_ctx` in /api/chat options is IGNORED on load.** The fix that
works: bake it into a custom model —
`printf 'FROM mistral:latest\nPARAMETER num_ctx 2048\n' | ollama create mistral-lp -f -`
— then run against `mistral-lp` (verify `ollama ps` shows CONTEXT 2048 /
100% GPU). Also start serve with `setsid ollama serve </dev/null` or it dies on
SSH session close. With these, mistral-7B ran all 15 waves at ~58 cells/sec.

## Decision rule (write it down BEFORE looking — the whole point)

- **Within ~5 points of 78% (i.e. ≳73%):** collapse is invariant to elicitation
  paradigm. Strongest sentence in the paper — *"collapse holds under the
  per-respondent sampling used in prior work, not just aggregate elicitation."*
  NHB stays live. Add one figure + one paragraph.
- **Materially below (≲70%):** collapse is partly an aggregate-elicitation
  artifact. Still publishable, but the framing narrows to *"collapse is specific
  to aggregate elicitation"* and the honest venue shifts (PNAS/NMI). Better to
  learn this now than from a hostile referee.

## Methods caveats (state these in the paper)

1. **Within-respondent correlation:** a persona conditions on its own earlier
   answers in the same call. Faithful to how ATP respondents answer a wave, but
   the N personas are not independent across questions within a respondent.
2. **Sampling noise:** per-cell distributions at n=personas carry multinomial
   noise that biases the contest toward the single-feature predictor (same
   effect documented on the human side). The existing correction applies
   verbatim with n=personas (`verification/noise_floor.py`,
   `paper_figs/noise_correct_surprise.py`) — recalibrate the collapse-index
   endpoints at n≈160 before reporting the *index* (the raw win-rate needs no
   recalibration).
3. **Temperature:** default 1.0 (standard for respondent diversity); aggregate
   pipeline used 0.7. Pass `--temperature 0.7` to make elicitation mode the only
   varying factor.

---

# PLAN: complete the elicitation check on the open-weights (non-API) models

The gpt-4o-mini run answers the referee's individual-sampling mode on one API
model. To make the invariance claim airtight we extend to the open-weights
models, where we control inference and compute is ~free, so we can **generate
full coverage** (all 15 waves) rather than a subset. This converts the result
from "one model, one alternative mode, two waves" into "collapse is invariant
across **3 elicitation paradigms × 3 model families × 15 waves**" — which closes
Comment 1 completely instead of merely deflecting it.

## The three elicitation paradigms (per model)

| Mode | What the model does | Backend | We have it? |
|---|---|---|---|
| **Aggregate** | researcher distributes 1,000 respondents | Ollama (done) | ✅ existing data |
| **Individual sampling** | be one persona, answer whole wave, ×N | Ollama | ⬜ generate |
| **Log-prob readout** | softmax over option tokens, no sampling (Santurkar) | transformers/vLLM | ⬜ generate |

The log-prob mode is the one we could never do on the API models (no logits) and
is exactly the OpinionQA paradigm whose data we already use — so generating it on
the open-weights models is high-value and cheap.

## Models (each already has an AGGREGATE baseline to compare against)

- **gemma2:9b** (instruct) → `gemma2_9b_indiv`, `gemma2_9b_logprob`
- **mistral-7b-instruct** (and/or mistral-small-24b) → `..._indiv`, `..._logprob`
- **llama3.1:8b-instruct** → `..._indiv`, `..._logprob`

(Base checkpoints `llama3_1_8b_text` / mistral-base already exist — reuse them for
the base-vs-tuned power follow-up, a separate item, not this one.)

## What to generate

### Mode 2 — Individual sampling (Ollama)
- **Reuse** `build_persona_prompt` + the aggregation/idempotency logic from
  `indiv_sampling_runner.py`; swap the OpenAI batch backend for an Ollama
  `/api/chat` call loop (model multi-worker, like `ollama_fill/run_ollama_fill.py`).
- One call = one persona answering the **whole wave**; temp 1.0; **N personas/cell**.
- New script: `indiv_sampling_local.py` (submit/run + same JSONL writer).

### Mode 3 — Log-prob readout (transformers or vLLM)
- Present options as **labelled letters** (A. … B. … C. …) so each option maps to
  a single token; one forward pass per (cell, question); read logits at the answer
  position; **softmax over the option-letter tokens** → distribution. No sampling.
- Ollama does **not** reliably expose per-token logits — use HF `transformers`
  (load model, forward, index logits) or **vLLM** (`logprobs`/`prompt_logprobs`
  via its OpenAI-compatible server). RunPod GPU box already set up for local
  inference (see `RUNPOD_STATUS.md`).
- New script: `logprob_readout.py`.

## Scope — "all of it"

- **Waves:** all 15 (W26–W92).
- **Cells:** singles + pairs over the 6 protagonist dims (Age, Gender, Race,
  Income, Political Party, Religion) = 280 cells/wave. **Depth 1+2 is mandatory**
  (the collapse claim lives at depth 2). **Depth 3 (triples) optional** — cheap
  for log-prob, expensive for individual sampling; add it for the depth-3
  identity-budget cross-check if budget allows.

## Cost / time reality (decide the knobs before launching)

- **Log-prob:** ~280 cells × ~90 q × 15 waves ≈ **380K single forward passes/model**
  — hours on one GPU. Cheapest path; run full coverage incl. triples if wanted.
- **Individual sampling:** 280 cells × N personas × 15 waves. At N=160 that's
  **~670K whole-wave generations/model** — the heavy item. Options:
  (a) full 15 waves at N=160 (~roughly a day/model on a good GPU);
  (b) drop to N=80–100 (win-rate is robust anywhere in 100–200);
  (c) full coverage on gemma+mistral, W26/W34 parity-only on llama.
  **Recommend:** start log-prob (cheap, all waves, all models) first; launch
  individual sampling in parallel at N=120 and full 15 waves, scaling personas
  down only if GPU time is tight.

## Analysis (drops into the existing pipeline, zero code change)

For every generated tag:
```bash
cd verification && python verify_spine_collapse.py <tag>
```
Then build one summary table: **model × {aggregate, individual, logprob} →
depth-2 best-single %**. Success criterion: win rates cluster in the ~74–80%
band across all (model × elicitation) cells, matching gpt-4o-mini's
78.3% / 77.2%. That single table is the figure/paragraph that closes Comment 1.

## Deliverables checklist
- [ ] `indiv_sampling_local.py` (Ollama individual sampling)
- [ ] `logprob_readout.py` (transformers/vLLM log-prob elicitation)
- [ ] generated `*_indiv` + `*_logprob` results for gemma2_9b / mistral / llama
- [ ] contest run per tag + the model × elicitation summary table
- [ ] one paragraph + table folded into §5 (and Methods caveat for log-prob
      letter-mapping + n-persona noise)
