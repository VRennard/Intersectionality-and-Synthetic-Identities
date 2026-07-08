#!/usr/bin/env python3
"""
Individual-sampling elicitation runner (referee Comment 1, the "make-or-break"
robustness check).

The main pipeline (batch_api_runner.py) elicits an *aggregate distribution*:
one call asks the model, as a researcher, to distribute 1,000 hypothetical
respondents across the options. The criticised silicon-sampling literature
instead samples *individual personas* (Argyle et al.) and aggregates them post
hoc. This runner replicates the headline collapse statistic under that
paradigm, holding everything else fixed:

  * one call = one simulated persona answering the ENTIRE wave (a survey, not a
    question), choosing exactly one option per question;
  * N personas per (profile) cell; the per-(cell, question) distribution is the
    histogram of chosen options across the N personas;
  * output is written to data/results/{model}_indiv/W{wave}.jsonl in the EXACT
    schema of the aggregate pipeline, so the existing collapse contest runs
    unchanged:  python verification/verify_spine_collapse.py {model}_indiv

Scope default: singles + pairs over the 6 protagonist dimensions (Age, Gender,
Race, Income, Political Party, Religion) for the waves you pass, which fully
covers race×party / religion×party plus every other pair among those dims.

Workflow mirrors batch_api_runner.py:
    python indiv_sampling_runner.py estimate --wave 26 --model gpt-4o-mini
    python indiv_sampling_runner.py submit   --wave 26 --model gpt-4o-mini --personas 160
    python indiv_sampling_runner.py fetch     --wave 26 --model gpt-4o-mini

Caveats (state in Methods):
  * Multi-question-per-call introduces within-respondent answer correlation
    (the persona conditions on its own earlier answers). This is faithful to how
    real ATP respondents answer a wave, but the N personas are not independent
    across questions within a respondent.
  * Per-cell distributions at n=personas carry multinomial sampling noise; the
    existing noise-correction / calibrated-endpoint machinery
    (verification/noise_floor.py, paper_figs/noise_correct_surprise.py) applies
    verbatim with n=personas.
  * Sampling temperature defaults to 1.0 (standard for eliciting respondent
    diversity); the aggregate pipeline used 0.7. Pass --temperature 0.7 to match
    it exactly if you want elicitation mode to be the only varying factor.
"""

import argparse
import json
import os
import re
import sys
import time
from collections import defaultdict
from datetime import datetime
from itertools import combinations, product
from pathlib import Path

sys.path.insert(0, os.path.dirname(__file__))
from llm_prompt_survey import DemographicProfile, SurveyQuestion
import simulation_config as _sim_cfg

DATA_DIR = Path("data")
BATCH_DIR = Path("data/batches")

# Protagonist dimensions for the collapse analysis. Pairs are built across these
# only, matching utils.DIM_NAMES / the paper's depth-2 contest.
CORE_DIMS = ["Age", "Gender", "Race", "Income", "Political Party", "Religion"]


# ── prompts (individual elicitation; matched to the aggregate where possible) ──

INDIV_SYSTEM = (
    "You are a single survey respondent. You will be given a demographic profile "
    "describing exactly who you are, followed by a series of survey questions. "
    "Answer every question honestly and realistically as this specific person "
    "would, choosing exactly one option per question. There are no right answers."
)


def build_persona_prompt(profile: DemographicProfile, questions: list) -> str:
    demo_str = "\n".join(f"    {f}" for f in profile.features)
    qs_block = ""
    for i, q in enumerate(questions, 1):
        opts = "\n".join(f"    {j+1}. {o}" for j, o in enumerate(q.options))
        qs_block += f"Question {i}: {q.question_text}\nOptions:\n{opts}\n\n"
    n = len(questions)
    return f"""You are the following person:

{demo_str}

Answer each of the {n} survey questions below as this person would, choosing the
single option that best reflects your own view.

{qs_block}Output ONLY a raw Python list of {n} integers, one per question in order, where
each integer is the number (as shown above) of the option you choose for that
question. Example for 3 questions: [2, 1, 4]
Do NOT wrap the output in markdown. Do NOT add any text or explanation."""


def parse_persona_response(text: str, n_questions: int) -> list:
    """Return a list of length n_questions of chosen 1-based option indices, or
    None per item that is missing/uninterpretable. Forgiving of markdown fences,
    trailing prose, and a truncated final list (keeps the answers before the
    cutoff)."""
    cleaned = re.sub(r"```[a-z]*\n?", "", text)
    m = re.search(r"\[[\s\d,]*\]", cleaned)
    nums = re.findall(r"\d+", m.group(0)) if m else re.findall(r"\d+", cleaned)
    vals = [int(x) for x in nums]
    out = []
    for i in range(n_questions):
        out.append(vals[i] if i < len(vals) else None)
    return out


# ── data loading ──────────────────────────────────────────────────────────────

def load_questions(wave: str):
    path = DATA_DIR / "responses" / f"survey_responses_W{wave}.json"
    with open(path) as f:
        raw = json.load(f)
    return [
        SurveyQuestion(
            question_id=q["question_id"],
            question_text=q["question_text"],
            options=[r["option"] for r in q["responses"]],
            wave=q.get("wave", f"W{wave}"),
        )
        for q in raw
    ]


def get_feature_vals(feature: str, wave: str):
    import pandas as pd
    path = Path("human_resp") / f"American_Trends_Panel_W{wave}" / "responses.csv"
    col = _sim_cfg.FEATURES.get(feature)
    if not col or not path.exists():
        return []
    df = pd.read_csv(path, low_memory=False)
    if col not in df.columns:
        return []
    ignored = _sim_cfg.IGNORE_VALUES.get(feature, [])
    return [str(v) for v in df[col].dropna().unique() if str(v) not in ignored]


def load_cells(wave: str, dims=CORE_DIMS):
    """Singles + pairs over the given dimensions (the depth-1/2 cells the
    collapse contest needs)."""
    cats = {}
    for feature in dims:
        vals = get_feature_vals(feature, wave)
        if vals:
            cats[feature] = vals
    keys = [k for k in dims if k in cats]
    cells = []
    for k in keys:
        for cat in cats[k]:
            cells.append(DemographicProfile(features=[f"{k} {cat}"]))
    for a, b in combinations(keys, 2):
        for ca, cb in product(cats[a], cats[b]):
            cells.append(DemographicProfile(features=[f"{a} {ca}", f"{b} {cb}"]))
    return cells


# ── request body (model-aware, single user message) ─────────────────────────────

def request_body(model: str, system_msg: str, user_msg: str,
                 n_questions: int, temperature: float):
    body = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg},
        ],
    }
    max_out = n_questions * 6 + 100
    if model.startswith("gpt-5"):
        body["max_completion_tokens"] = max_out
        body["reasoning_effort"] = "none"
    else:
        body["temperature"] = temperature
        body["max_tokens"] = max_out
    return body


def make_custom_id(c_idx: int, p_idx: int) -> str:
    return f"c{c_idx}-p{p_idx}"


# ── estimate ────────────────────────────────────────────────────────────────────

def cmd_estimate(args):
    questions = load_questions(args.wave)
    cells = load_cells(args.wave)
    n_calls = len(cells) * args.personas
    # rough token estimate: persona prompt scales with question/option text
    sample_prompt = build_persona_prompt(cells[-1], questions)
    approx_in = len(sample_prompt) / 4 + len(INDIV_SYSTEM) / 4
    approx_out = len(questions) * 4
    tot_in = n_calls * approx_in
    tot_out = n_calls * approx_out
    # gpt-4o-mini batch pricing (per 1M): in $0.075, out $0.30
    cost = tot_in / 1e6 * 0.075 + tot_out / 1e6 * 0.30
    print(f"W{args.wave}: {len(questions)} questions, {len(cells)} cells "
          f"({sum(len(c.features)==1 for c in cells)} singles, "
          f"{sum(len(c.features)==2 for c in cells)} pairs)")
    print(f"  personas/cell: {args.personas}")
    print(f"  total API calls: {n_calls:,}")
    print(f"  approx tokens: {tot_in/1e6:.1f}M in, {tot_out/1e6:.1f}M out")
    print(f"  approx gpt-4o-mini batch cost: ${cost:.2f}")


# ── submit ────────────────────────────────────────────────────────────────────

def _done_personas(out_path: Path):
    """Set of (cell_key, persona_idx) already aggregated. We track completed
    personas in a sidecar so resume is exact even though the JSONL stores only
    aggregates."""
    side = out_path.with_suffix(".personas.json")
    if side.exists():
        with open(side) as f:
            return {tuple(x) for x in json.load(f)}
    return set()


def _validate_batch_files(chunk_paths, expected_total, model):
    """Re-read every written record and assert OpenAI Batch structural rules:
    unique custom_ids, required keys, correct endpoint, well-formed body."""
    seen_ids, total, errors = set(), 0, []
    for cp in chunk_paths:
        with open(cp, encoding="utf-8") as f:
            for ln_no, ln in enumerate(f, 1):
                ln = ln.strip()
                if not ln:
                    continue
                total += 1
                try:
                    rec = json.loads(ln)
                except json.JSONDecodeError as e:
                    errors.append(f"{cp.name}:{ln_no} bad JSON: {e}")
                    continue
                if rec.get("method") != "POST":
                    errors.append(f"{cp.name}:{ln_no} method != POST")
                if rec.get("url") != "/v1/chat/completions":
                    errors.append(f"{cp.name}:{ln_no} wrong url {rec.get('url')}")
                cid = rec.get("custom_id")
                if not cid:
                    errors.append(f"{cp.name}:{ln_no} missing custom_id")
                elif cid in seen_ids:
                    errors.append(f"{cp.name}:{ln_no} DUPLICATE custom_id {cid}")
                else:
                    seen_ids.add(cid)
                body = rec.get("body", {})
                if body.get("model") != model:
                    errors.append(f"{cp.name}:{ln_no} model mismatch {body.get('model')}")
                msgs = body.get("messages", [])
                if len(msgs) != 2 or msgs[0]["role"] != "system" or msgs[1]["role"] != "user":
                    errors.append(f"{cp.name}:{ln_no} malformed messages")
                if model.startswith("gpt-5"):
                    if "max_completion_tokens" not in body:
                        errors.append(f"{cp.name}:{ln_no} gpt-5 missing max_completion_tokens")
                else:
                    if "max_tokens" not in body or "temperature" not in body:
                        errors.append(f"{cp.name}:{ln_no} missing max_tokens/temperature")
    print(f"\n[validate] {total:,} records, {len(seen_ids):,} unique custom_ids")
    if total != expected_total:
        errors.append(f"record count {total} != expected {expected_total}")
    if errors:
        print(f"[validate] ❌ {len(errors)} problem(s):")
        for e in errors[:20]:
            print("   ", e)
        sys.exit(1)
    print("[validate] ✓ all structural checks passed")


def cmd_submit(args):
    model_tag = f"{args.model}_indiv"

    print(f"Loading wave W{args.wave}...")
    questions = load_questions(args.wave)
    cells = load_cells(args.wave)
    print(f"  {len(questions)} questions, {len(cells)} cells, "
          f"{args.personas} personas/cell")

    out_path = DATA_DIR / "results" / model_tag / f"W{args.wave}.jsonl"
    done = _done_personas(out_path)
    if done:
        print(f"  resuming: {len(done)} personas already aggregated")

    BATCH_DIR.mkdir(parents=True, exist_ok=True)
    batch_input_path = BATCH_DIR / f"W{args.wave}_{model_tag}_input.jsonl"

    written = 0
    with open(batch_input_path, "w") as f:
        for c_idx, cell in enumerate(cells):
            user_msg = build_persona_prompt(cell, questions)
            body = request_body(args.model, INDIV_SYSTEM, user_msg,
                                len(questions), args.temperature)
            for p_idx in range(args.personas):
                key = (tuple(sorted(cell.features)), p_idx)
                if key in done:
                    continue
                record = {
                    "custom_id": make_custom_id(c_idx, p_idx),
                    "method": "POST",
                    "url": "/v1/chat/completions",
                    "body": body,
                }
                f.write(json.dumps(record) + "\n")
                written += 1

    print(f"  {written} persona calls to submit")
    if written == 0:
        print("  nothing to do — all personas already aggregated.")
        return

    # OpenAI Batch limits: 50,000 requests AND 200 MB per input file. Each
    # persona record embeds the full questionnaire, so the file is large —
    # chunk by BOTH count and cumulative bytes.
    MAX_REQUESTS = 50_000
    MAX_BYTES = 180 * 1024 * 1024   # 180 MB, safe margin under the 200 MB cap
    with open(batch_input_path, "rb") as f:
        all_lines = f.readlines()

    chunks, cur, cur_bytes = [], [], 0
    for ln in all_lines:
        if cur and (len(cur) >= MAX_REQUESTS or cur_bytes + len(ln) > MAX_BYTES):
            chunks.append(cur)
            cur, cur_bytes = [], 0
        cur.append(ln)
        cur_bytes += len(ln)
    if cur:
        chunks.append(cur)

    total_mb = sum(len(l) for l in all_lines) / 1024 / 1024
    print(f"Batch input: {written:,} requests, {total_mb:.1f} MB total")
    print(f"Splitting into {len(chunks)} batch(es) "
          f"(<= {MAX_REQUESTS:,} requests / {MAX_BYTES/1024/1024:.0f} MB each)...")

    chunk_paths = []
    for i, chunk in enumerate(chunks):
        chunk_path = BATCH_DIR / f"W{args.wave}_{model_tag}_input_part{i+1}.jsonl"
        with open(chunk_path, "wb") as f:
            f.writelines(chunk)
        chunk_paths.append(chunk_path)
        print(f"  part {i+1}/{len(chunks)}: {len(chunk):,} requests, "
              f"{chunk_path.stat().st_size/1024/1024:.1f} MB")

    if args.dry_run:
        print("\n[DRY RUN] batch files written and validated; nothing uploaded.")
        _validate_batch_files(chunk_paths, written, args.model)
        return

    import openai
    client = openai.OpenAI(api_key=args.api_key)

    batch_ids = []
    for i, chunk_path in enumerate(chunk_paths):
        print(f"  Uploading part {i+1}/{len(chunk_paths)} "
              f"({chunk_path.stat().st_size/1024/1024:.1f} MB)...")
        with open(chunk_path, "rb") as f:
            uploaded = client.files.create(file=f, purpose="batch")
        batch = client.batches.create(
            input_file_id=uploaded.id,
            endpoint="/v1/chat/completions",
            completion_window="24h",
        )
        print(f"    Batch ID: {batch.id}  Status: {batch.status}")
        batch_ids.append(batch.id)

    state_path = BATCH_DIR / f"W{args.wave}_{model_tag}_state.json"
    with open(state_path, "w") as f:
        json.dump({"batch_ids": batch_ids, "wave": args.wave, "model": args.model,
                   "model_tag": model_tag, "personas": args.personas,
                   "temperature": args.temperature, "api_key": args.api_key,
                   "submitted_at": datetime.now().isoformat()}, f, indent=2)
    print(f"\n  {len(batch_ids)} batch(es) submitted. State -> {state_path}")
    print(f"\nFetch when ready:\n  python indiv_sampling_runner.py fetch "
          f"--wave {args.wave} --model {args.model}")


# ── fetch ─────────────────────────────────────────────────────────────────────

def cmd_fetch(args):
    model_tag = f"{args.model}_indiv"
    state_path = BATCH_DIR / f"W{args.wave}_{model_tag}_state.json"
    if not state_path.exists():
        print(f"Error: no state file at {state_path}")
        sys.exit(1)
    with open(state_path) as f:
        state = json.load(f)
    if not args.api_key:
        args.api_key = state.get("api_key")

    import openai
    client = openai.OpenAI(api_key=args.api_key)
    batch_ids = state.get("batch_ids") or [state.get("batch_id")]
    print(f"Found {len(batch_ids)} batch(es) to fetch.")

    result_lines = []
    for bid in batch_ids:
        print(f"\nChecking batch {bid}...")
        while True:
            batch = client.batches.retrieve(bid)
            c = batch.request_counts
            print(f"  Status: {batch.status} | completed={c.completed} "
                  f"failed={c.failed} total={c.total}")
            if batch.status == "completed":
                break
            if batch.status in ("failed", "expired", "cancelled"):
                print(f"  Batch {bid} did not complete; aborting.")
                sys.exit(1)
            print("  Waiting 30s...")
            time.sleep(30)
        result_lines.extend(
            client.files.content(batch.output_file_id).text.strip().splitlines())
    print(f"\nTotal result lines: {len(result_lines):,}")

    questions = load_questions(args.wave)
    cells = load_cells(args.wave)
    out_path = DATA_DIR / "results" / model_tag / f"W{args.wave}.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Aggregate persona answers into per-(cell, question) option counts.
    # counts[c_idx][q_idx] = np.array of option counts
    import numpy as np
    counts = {c: [np.zeros(len(q.options), dtype=int) for q in questions]
              for c in range(len(cells))}
    new_personas = set()

    # Idempotency: personas already aggregated into the prior JSONL must not be
    # counted again (re-running fetch re-downloads the SAME batch output). The
    # sidecar is the source of truth for "already merged".
    already = _done_personas(out_path)

    parsed_calls = skipped_dupe = 0
    for line in result_lines:
        obj = json.loads(line)
        cid = obj["custom_id"]
        m = re.match(r"c(\d+)-p(\d+)", cid)
        if not m:
            continue
        c_idx, p_idx = int(m.group(1)), int(m.group(2))
        if (tuple(sorted(cells[c_idx].features)), p_idx) in already:
            skipped_dupe += 1
            continue
        resp = obj.get("response")
        if not resp or resp.get("status_code") != 200:
            continue
        text = resp["body"]["choices"][0]["message"]["content"].strip()
        picks = parse_persona_response(text, len(questions))
        any_valid = False
        for q_idx, pick in enumerate(picks):
            nopt = len(questions[q_idx].options)
            if pick is not None and 1 <= pick <= nopt:
                counts[c_idx][q_idx][pick - 1] += 1
                any_valid = True
        if any_valid:
            new_personas.add((tuple(sorted(cells[c_idx].features)), p_idx))
            parsed_calls += 1
    print(f"Parsed {parsed_calls:,} valid persona responses "
          f"({skipped_dupe:,} already-merged personas skipped).")

    # Merge with any prior aggregates (resume): re-read existing JSONL counts.
    prior = {}
    if out_path.exists():
        with open(out_path, encoding="utf-8") as f:
            for ln in f:
                ln = ln.strip()
                if not ln:
                    continue
                try:
                    r = json.loads(ln)
                    prior[(tuple(sorted(r["demographics"])), r["question_id"])] = \
                        np.array(r["response_distribution"], dtype=int)
                except Exception:
                    pass

    # Rewrite JSONL with merged counts.
    written = 0
    tmp = out_path.with_suffix(".jsonl.tmp")
    with open(tmp, "w") as f:
        for c_idx, cell in enumerate(cells):
            ckey = tuple(sorted(cell.features))
            for q_idx, q in enumerate(questions):
                add = counts[c_idx][q_idx]
                base = prior.get((ckey, q.question_id))
                total = add if base is None else base + add
                if total.sum() == 0:
                    continue
                rec = {
                    "timestamp": datetime.now().isoformat(),
                    "demographics": cell.features,
                    "question_id": q.question_id,
                    "question_text": q.question_text,
                    "options": q.options,
                    "response_distribution": [int(x) for x in total],
                    "status": "success",
                    "elicitation": "individual_sampling",
                    "n_personas": int(total.sum()),
                }
                f.write(json.dumps(rec) + "\n")
                written += 1
    tmp.replace(out_path)

    # Update persona sidecar.
    side = out_path.with_suffix(".personas.json")
    done = _done_personas(out_path)
    done |= new_personas
    with open(side, "w") as f:
        json.dump(sorted([list(x) for x in done]), f)

    print(f"\nWrote {written} (cell, question) aggregate rows -> {out_path}")
    print(f"Personas aggregated so far: {len(done)}")
    print(f"\nRun the collapse contest with:\n"
          f"  cd verification && python verify_spine_collapse.py {model_tag}")


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description="Individual-sampling elicitation runner")
    sub = p.add_subparsers(dest="cmd", required=True)

    pe = sub.add_parser("estimate")
    pe.add_argument("--wave", required=True)
    pe.add_argument("--model", default="gpt-4o-mini")
    pe.add_argument("--personas", type=int, default=160)

    ps = sub.add_parser("submit")
    ps.add_argument("--wave", required=True)
    ps.add_argument("--model", default="gpt-4o-mini")
    ps.add_argument("--personas", type=int, default=160)
    ps.add_argument("--temperature", type=float, default=1.0)
    ps.add_argument("--api-key", default=os.environ.get("OPENAI_API_KEY"))
    ps.add_argument("--dry-run", action="store_true",
                    help="build + validate batch files without uploading")

    pf = sub.add_parser("fetch")
    pf.add_argument("--wave", required=True)
    pf.add_argument("--model", default="gpt-4o-mini")
    pf.add_argument("--api-key", default=os.environ.get("OPENAI_API_KEY"))

    args = p.parse_args()
    os.chdir(Path(__file__).parent)

    if args.cmd == "estimate":
        cmd_estimate(args)
    elif args.cmd == "submit":
        if not args.api_key and not args.dry_run:
            print("Error: provide --api-key or set OPENAI_API_KEY")
            sys.exit(1)
        cmd_submit(args)
    elif args.cmd == "fetch":
        cmd_fetch(args)


if __name__ == "__main__":
    main()
