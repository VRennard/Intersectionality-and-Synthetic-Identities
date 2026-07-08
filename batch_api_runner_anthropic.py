#!/usr/bin/env python3
"""
Anthropic Message Batches API runner for survey simulation.

Mirrors batch_api_runner.py but targets Anthropic models (e.g. claude-haiku-4-5-20251001).
50% cost discount vs. synchronous calls.

Workflow:
  1. submit  – generate all prompts, create batch job(s)
  2. fetch   – poll status; download + parse results when complete
  3. status  – quick check on a batch

Usage:
    python batch_api_runner_anthropic.py submit --wave 26 --api-key sk-ant-...
    python batch_api_runner_anthropic.py fetch  --wave 26
    python batch_api_runner_anthropic.py status --wave 26

Output:
    data/results/{model}/W{wave}.jsonl  (same format as existing pipeline)
"""

import argparse
import json
import os
import re
import sys
import time
from datetime import datetime
from itertools import combinations, product
from pathlib import Path

import anthropic

sys.path.insert(0, os.path.dirname(__file__))
from llm_prompt_survey import DemographicProfile, PromptBuilder, ResponseParser, SurveyQuestion
import simulation_config as _sim_cfg

DATA_DIR  = Path("data")
BATCH_DIR = Path("data/batches")

DEFAULT_MODEL = "claude-haiku-4-5-20251001"
CHUNK_SIZE    = 50_000   # Anthropic allows up to 100k; 50k keeps payloads manageable


# ── helpers ───────────────────────────────────────────────────────────────────

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


TRIPLE_FEATURES = ["Age", "Gender", "Race", "Income", "Political Party", "Religion"]


def load_profiles(wave: str, include_triples: bool = False):
    cats = {}
    for feature in _sim_cfg.FEATURES:
        vals = get_feature_vals(feature, wave)
        if vals:
            cats[feature] = vals

    keys = [k for k in _sim_cfg.FEATURES if k in cats]
    profiles = []
    for k in keys:
        for cat in cats[k]:
            profiles.append(DemographicProfile(features=[f"{k} {cat}"]))
    for a, b in combinations(keys, 2):
        for ca, cb in product(cats[a], cats[b]):
            profiles.append(DemographicProfile(features=[f"{a} {ca}", f"{b} {cb}"]))
    if include_triples:
        tkeys = [k for k in TRIPLE_FEATURES if k in cats]
        for a, b, c in combinations(tkeys, 3):
            for ca, cb, cc in product(cats[a], cats[b], cats[c]):
                profiles.append(DemographicProfile(
                    features=[f"{a} {ca}", f"{b} {cb}", f"{c} {cc}"]))
    return profiles


def build_multi_question_prompt(profile: DemographicProfile, questions: list) -> str:
    demo_str = ", ".join(profile.features)
    qs_block = ""
    for i, q in enumerate(questions):
        opts = "\n".join(f"    {j+1}. {o}" for j, o in enumerate(q.options))
        qs_block += f"Question {i+1}: {q.question_text}\nOptions:\n{opts}\n\n"

    return (
        f"Demographic Profile: {demo_str}\n\n"
        f"You are surveying exactly 1,000 individuals who fit this profile.\n"
        f"For EACH of the {len(questions)} questions below, distribute 1,000 individuals across the options.\n\n"
        f"{qs_block}"
        f"Output a Python list of {len(questions)} lists, one per question, in order.\n"
        f"Each inner list contains integers summing to 1000, matching the number of options.\n"
        f"Output ONLY the raw Python structure. No markdown, no explanation.\n"
        f"Example for 2 questions with 3 and 4 options: [[300,500,200],[100,400,300,200]]"
    )


def parse_multi_question_response(text: str, questions: list) -> list:
    """Forgiving list-of-lists parse: regex-extracts each inner integer list in
    order (tolerates markdown, trailing prose, and a truncated final list),
    rescales near-1000 sums to exactly 1000, matches positionally. An item fails
    only if missing or its integer count != that question's option count."""
    cleaned = re.sub(r"```[a-z]*\n?", "", text)
    inner_lists = re.findall(r"\[\s*\d[\d,\s]*\]", cleaned)
    parsed = [[int(x) for x in re.findall(r"\d+", s)] for s in inner_lists]
    out = []
    for i, q in enumerate(questions):
        v = parsed[i] if i < len(parsed) else None
        if v and len(v) == len(q.options) and sum(v) > 0:
            t = sum(v)
            if t != 1000:
                v = [int(round(x * 1000 / t)) for x in v]
                v[-1] += 1000 - sum(v)
            out.append(v)
        else:
            out.append(None)
    return out


def make_custom_id(d_idx: int, qb_idx: int) -> str:
    return f"d{d_idx}-qb{qb_idx}"


def load_state(wave: str, model: str, run_tag: str = "") -> dict:
    tag = f"_{run_tag}" if run_tag else ""
    state_path = BATCH_DIR / f"W{wave}_{model}{tag}_anthropic_state.json"
    if not state_path.exists():
        return {}
    with open(state_path) as f:
        return json.load(f)


def save_state(wave: str, model: str, state: dict, run_tag: str = ""):
    BATCH_DIR.mkdir(parents=True, exist_ok=True)
    tag = f"_{run_tag}" if run_tag else ""
    state_path = BATCH_DIR / f"W{wave}_{model}{tag}_anthropic_state.json"
    with open(state_path, "w") as f:
        json.dump(state, f, indent=2)


# ── submit ────────────────────────────────────────────────────────────────────

def cmd_submit(args):
    client = anthropic.Anthropic(api_key=args.api_key)

    print(f"Loading wave W{args.wave}...")
    questions = load_questions(args.wave)
    profiles  = load_profiles(args.wave, include_triples=args.include_triples)
    print(f"  {len(questions)} questions, {len(profiles)} demographic profiles")

    # Find already-completed (profile, question) pairs — skip errors so they get retried
    done = set()
    out_path = DATA_DIR / "results" / args.model / f"W{args.wave}.jsonl"
    if out_path.exists():
        with open(out_path) as f:
            for line in f:
                r = json.loads(line)
                if r.get("status") == "success":
                    done.add((tuple(r["demographics"]), r["question_id"]))

    builder      = PromptBuilder()
    q_batch_size = args.q_batch_size
    requests     = []

    for d_idx, profile in enumerate(profiles):
        pending_qs = [q for q in questions
                      if (tuple(profile.features), q.question_id) not in done]

        for qb_idx, i in enumerate(range(0, len(pending_qs), q_batch_size)):
            q_batch = pending_qs[i:i + q_batch_size]
            if not q_batch:
                continue

            if q_batch_size == 1:
                user_msg = builder.build_prompt(profile, q_batch[0])
                max_tok  = 100
            else:
                user_msg = build_multi_question_prompt(profile, q_batch)
                max_tok  = q_batch_size * 30 + 50

            requests.append({
                "custom_id": make_custom_id(d_idx, qb_idx),
                "params": {
                    "model":      args.model,
                    "max_tokens": max_tok,
                    "system":     builder.SYSTEM_MESSAGE,
                    "messages":   [{"role": "user", "content": user_msg}],
                },
            })

    skipped = (len(profiles) * len(questions)) - sum(
        len([q for q in questions if (tuple(p.features), q.question_id) not in done])
        for p in profiles
    )
    print(f"  {len(requests)} API calls to submit ({skipped} questions already done — skipped)")

    # Split into chunks
    chunks    = [requests[i:i + CHUNK_SIZE] for i in range(0, len(requests), CHUNK_SIZE)]
    batch_ids = []

    print(f"Submitting {len(chunks)} batch(es)...")
    for i, chunk in enumerate(chunks):
        print(f"  Submitting part {i+1}/{len(chunks)} ({len(chunk):,} requests)...")
        batch = client.messages.batches.create(requests=chunk)
        print(f"    Batch ID: {batch.id}  Status: {batch.processing_status}")
        batch_ids.append(batch.id)

    save_state(args.wave, args.model, {
        "batch_ids":       batch_ids,
        "wave":            args.wave,
        "model":           args.model,
        "q_batch_size":    args.q_batch_size,
        "include_triples": args.include_triples,
        "submitted_at":    datetime.now().isoformat(),
    }, run_tag=args.run_tag)

    print(f"\n{len(batch_ids)} batch(es) submitted: {batch_ids}")
    print(f"Run this to fetch results when ready:")
    print(f"  python batch_api_runner_anthropic.py fetch --wave {args.wave} --model {args.model}")


# ── fetch ─────────────────────────────────────────────────────────────────────

def cmd_fetch(args):
    state = load_state(args.wave, args.model, run_tag=args.run_tag)
    if not state:
        print(f"Error: no state file found. Run submit first.")
        sys.exit(1)

    if not args.api_key:
        args.api_key = state.get("api_key") or os.environ.get("ANTHROPIC_API_KEY")
    q_batch_size    = state.get("q_batch_size", args.q_batch_size)
    include_triples = state.get("include_triples", False)
    batch_ids       = state.get("batch_ids", [])

    client = anthropic.Anthropic(api_key=args.api_key)

    # Wait for all batches
    all_results = {}
    for bid in batch_ids:
        print(f"\nChecking batch {bid}...")
        while True:
            batch = client.messages.batches.retrieve(bid)
            rc    = batch.request_counts
            print(f"  Status: {batch.processing_status}  |  "
                  f"processing={rc.processing}  succeeded={rc.succeeded}  "
                  f"errored={rc.errored}  canceled={rc.canceled}  expired={rc.expired}")
            if batch.processing_status == "ended":
                break
            print("  Not done yet — waiting 30s...")
            time.sleep(30)

        print(f"  Downloading results for {bid}...")
        for result in client.messages.batches.results(bid):
            all_results[result.custom_id] = result

    print(f"\nTotal results fetched: {len(all_results):,}")

    questions = load_questions(args.wave)
    profiles  = load_profiles(args.wave, include_triples=include_triples)

    out_path = DATA_DIR / "results" / args.model / f"W{args.wave}.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    existing = set()
    if out_path.exists():
        with open(out_path) as f:
            for line in f:
                r = json.loads(line)
                if r.get("status") == "success":
                    existing.add((tuple(r["demographics"]), r["question_id"]))

    success = error = skipped = 0

    with open(out_path, "a") as out_f:
        for d_idx, profile in enumerate(profiles):
            pending_qs = [q for q in questions
                          if (tuple(profile.features), q.question_id) not in existing]
            skipped += len(questions) - len(pending_qs)

            for qb_idx, i in enumerate(range(0, len(pending_qs), q_batch_size)):
                q_batch = pending_qs[i:i + q_batch_size]
                if not q_batch:
                    continue

                cid    = make_custom_id(d_idx, qb_idx)
                result = all_results.get(cid)

                if result and result.result.type == "succeeded":
                    text = result.result.message.content[0].text.strip()
                    if q_batch_size == 1:
                        distributions = [ResponseParser.parse_response(text, len(q_batch[0].options))]
                    else:
                        distributions = parse_multi_question_response(text, q_batch)
                else:
                    distributions = [None] * len(q_batch)

                for question, dist in zip(q_batch, distributions):
                    rec = {
                        "timestamp":             datetime.now().isoformat(),
                        "demographics":          profile.features,
                        "question_id":           question.question_id,
                        "question_text":         question.question_text,
                        "options":               question.options,
                        "response_distribution": dist or [],
                        "status":                "success" if dist else "error",
                    }
                    out_f.write(json.dumps(rec) + "\n")
                    if dist:
                        success += 1
                    else:
                        error += 1

    print(f"\nNew: {success} successful, {error} errors, {skipped} already existed")
    print(f"Results written to {out_path}")


# ── status ────────────────────────────────────────────────────────────────────

def cmd_status(args):
    state = load_state(args.wave, args.model, run_tag=args.run_tag)
    if not state:
        print("No state file found.")
        sys.exit(1)

    client    = anthropic.Anthropic(api_key=args.api_key)
    batch_ids = state.get("batch_ids", [])

    for bid in batch_ids:
        batch = client.messages.batches.retrieve(bid)
        rc    = batch.request_counts
        print(f"Batch   : {bid}")
        print(f"Status  : {batch.processing_status}")
        print(f"Progress: succeeded={rc.succeeded}  processing={rc.processing}  "
              f"errored={rc.errored}  expired={rc.expired}")
        print()


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Anthropic Batch API runner for survey simulation")
    parser.add_argument("--api-key", default=os.environ.get("ANTHROPIC_API_KEY"))

    sub = parser.add_subparsers(dest="cmd", required=True)

    p_submit = sub.add_parser("submit")
    p_submit.add_argument("--wave",             required=True)
    p_submit.add_argument("--model",            default=DEFAULT_MODEL)
    p_submit.add_argument("--api-key",          default=os.environ.get("ANTHROPIC_API_KEY"))
    p_submit.add_argument("--q-batch-size",     type=int, default=20)
    p_submit.add_argument("--include-triples",  action="store_true")
    p_submit.add_argument("--run-tag",          default="",
                          help="suffix for state file, e.g. 'triples' to avoid collisions")

    p_fetch = sub.add_parser("fetch")
    p_fetch.add_argument("--wave",          required=True)
    p_fetch.add_argument("--model",         default=DEFAULT_MODEL)
    p_fetch.add_argument("--api-key",       default=os.environ.get("ANTHROPIC_API_KEY"))
    p_fetch.add_argument("--q-batch-size",  type=int, default=20)
    p_fetch.add_argument("--run-tag",       default="")

    p_status = sub.add_parser("status")
    p_status.add_argument("--wave",    required=True)
    p_status.add_argument("--model",   default=DEFAULT_MODEL)
    p_status.add_argument("--api-key", default=os.environ.get("ANTHROPIC_API_KEY"))
    p_status.add_argument("--run-tag", default="")

    args = parser.parse_args()
    os.chdir(Path(__file__).parent)

    if not args.api_key:
        print("Error: provide --api-key or set ANTHROPIC_API_KEY")
        sys.exit(1)

    if args.cmd == "submit":
        cmd_submit(args)
    elif args.cmd == "fetch":
        cmd_fetch(args)
    elif args.cmd == "status":
        cmd_status(args)


if __name__ == "__main__":
    main()
