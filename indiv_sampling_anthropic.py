#!/usr/bin/env python3
"""
Individual-sampling elicitation on Anthropic models (Claude Haiku) via the
Message Batches API, WITH prompt caching.

Same paradigm as indiv_sampling_runner.py (one call = one persona answering the
whole wave, N personas/cell, histogram → distribution), but two differences:

  1. Anthropic Batches API (50% off) instead of OpenAI.
  2. PROMPT CACHING. The questionnaire (~99% of each prompt) is identical across
     every persona and every cell in a wave, so it goes in a CACHED `system`
     block (cache_control, 1h TTL); only the short demographic profile varies in
     the user turn. The first request in the batch writes the cache; the rest
     read it at ~0.1x input cost. This is the prefix-first restructure the
     persona-first prompt in indiv_sampling_runner.py can't cache.

Output: data/results/{model}_indiv/W{wave}.jsonl — canonical schema, drops into
the collapse contest unchanged (verify_spine_collapse.py {model}_indiv).

Usage:
    python indiv_sampling_anthropic.py estimate --wave 26 --personas 100
    python indiv_sampling_anthropic.py submit   --wave 26 --personas 100 [--dry-run]
    python indiv_sampling_anthropic.py fetch     --wave 26
"""

import argparse
import json
import os
import re
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from indiv_sampling_runner import parse_persona_response
from indiv_sampling_local import load_questions, build_cells
from llm_prompt_survey import DemographicProfile, SurveyQuestion

DATA_DIR = Path("data")
BATCH_DIR = Path("data/batches")
DEFAULT_MODEL = "claude-haiku-4-5-20251001"

# Anthropic Batch limits: 100,000 requests AND 256 MB per batch. Each request
# embeds the full questionnaire in `system`, so the file is large — chunk by
# bytes too.
MAX_REQUESTS = 90_000
MAX_BYTES = 40 * 1024 * 1024


# ── prompt (questionnaire-first, cached; persona-last, variable) ────────────────

SYSTEM_BASE = (
    "You are a single survey respondent. You will be shown a fixed set of survey "
    "questions, then told which specific person you are. Answer every question as "
    "that person would, choosing exactly one option per question. There are no "
    "right answers."
)


def build_questionnaire_block(questions):
    """The big, stable, CACHED text: all questions + the output instruction.
    Identical for every persona and every cell in the wave."""
    n = len(questions)
    qs = ""
    for i, q in enumerate(questions, 1):
        opts = "\n".join(f"    {j+1}. {o}" for j, o in enumerate(q.options))
        qs += f"Question {i}: {q.question_text}\nOptions:\n{opts}\n\n"
    return (
        f"Here are the {n} survey questions:\n\n{qs}"
        f"When you are told which person you are, answer each of the {n} questions "
        f"as that person would, choosing the single option that best reflects their "
        f"own view. Output ONLY a raw Python list of {n} integers, one per question "
        f"in order, where each integer is the number of the option you choose "
        f"(as shown above). Example for 3 questions: [2, 1, 4]. "
        f"Do NOT use markdown. Do NOT add any text or explanation."
    )


def build_system(questions, ttl):
    """System as two blocks; the questionnaire block carries cache_control."""
    return [
        {"type": "text", "text": SYSTEM_BASE},
        {"type": "text", "text": build_questionnaire_block(questions),
         "cache_control": {"type": "ephemeral", "ttl": ttl}},
    ]


def build_user(profile: DemographicProfile) -> str:
    demo = "\n".join(f"    {f}" for f in profile.features)
    return f"You are the following person:\n\n{demo}"


def make_custom_id(c_idx, p_idx):
    return f"c{c_idx}-p{p_idx}"


# ── estimate ────────────────────────────────────────────────────────────────────

def cmd_estimate(args):
    questions = load_questions("data/responses", args.wave)
    cells = build_cells(args.wave, human_dir="human_resp")
    n_calls = len(cells) * args.personas
    qblock = build_questionnaire_block(questions)
    q_tok = len(qblock) // 4 + len(SYSTEM_BASE) // 4   # cached prefix tokens
    var_tok = 40                                        # per-call variable user turn
    out_tok = len(questions) * 4
    # Haiku batch rates ($/1M): input 0.50, output 2.50, cache write(1h) 1.00, read 0.05
    no_cache_in = n_calls * (q_tok + var_tok) / 1e6 * 0.50
    # cached: prefix written ~once per chunk, read by the rest at 0.1x
    cache_in = (q_tok * 1.0 / 1e6                       # ~1 write (1h ≈ 2x→ batch 1.0)
                + n_calls * q_tok / 1e6 * 0.05          # reads at 0.05
                + n_calls * var_tok / 1e6 * 0.50)
    out_cost = n_calls * out_tok / 1e6 * 2.50
    print(f"W{args.wave}: {len(questions)} q, {len(cells)} cells, {args.personas} personas")
    print(f"  calls: {n_calls:,} | cached prefix ~{q_tok} tok | output ~{out_tok} tok/call")
    print(f"  NO cache : input ${no_cache_in:.0f} + output ${out_cost:.0f} = ${no_cache_in+out_cost:.0f}")
    print(f"  W/ cache : input ${cache_in:.0f} + output ${out_cost:.0f} = ${cache_in+out_cost:.0f} "
          f"(best case; batch TTL may lower hit-rate)")


# ── submit ────────────────────────────────────────────────────────────────────

def _done_personas(out_path):
    side = out_path.with_suffix(".personas.json")
    if side.exists():
        with open(side) as f:
            return {tuple(x) for x in json.load(f)}
    return set()


def cmd_submit(args):
    model_tag = f"{args.model}_indiv"
    questions = load_questions("data/responses", args.wave)
    cells = build_cells(args.wave, human_dir="human_resp")
    print(f"W{args.wave}: {len(questions)} questions, {len(cells)} cells, "
          f"{args.personas} personas/cell")

    out_path = DATA_DIR / "results" / model_tag / f"W{args.wave}.jsonl"
    done = _done_personas(out_path)
    if done:
        print(f"  resuming: {len(done)} personas already aggregated")

    system = build_system(questions, args.ttl)
    max_tok = len(questions) * 6 + 100
    requests = []
    for c_idx, cell in enumerate(cells):
        user = build_user(cell)
        for p_idx in range(args.personas):
            if (tuple(sorted(cell.features)), p_idx) in done:
                continue
            requests.append({
                "custom_id": make_custom_id(c_idx, p_idx),
                "params": {
                    "model": args.model,
                    "max_tokens": max_tok,
                    "system": system,
                    "messages": [{"role": "user", "content": user}],
                },
            })
    print(f"  {len(requests):,} persona calls to submit "
          f"(cached prefix ~{len(build_questionnaire_block(questions))//4} tok, ttl={args.ttl})")
    if not requests:
        print("  nothing to do.")
        return

    # chunk by count AND bytes
    chunks, cur, cur_bytes = [], [], 0
    for r in requests:
        rb = len(json.dumps(r))
        if cur and (len(cur) >= MAX_REQUESTS or cur_bytes + rb > MAX_BYTES):
            chunks.append(cur); cur, cur_bytes = [], 0
        cur.append(r); cur_bytes += rb
    if cur:
        chunks.append(cur)
    total_mb = sum(len(json.dumps(r)) for r in requests) / 1024 / 1024
    print(f"  {total_mb:.0f} MB total → {len(chunks)} batch(es)")

    if args.dry_run:
        ids = {r["custom_id"] for r in requests}
        assert len(ids) == len(requests), "duplicate custom_ids!"
        # verify cache_control present + structure
        s = requests[0]["params"]["system"]
        assert s[1].get("cache_control"), "no cache_control on questionnaire block"
        print(f"  [DRY RUN] {len(requests):,} reqs, all custom_ids unique, "
              f"cache_control present (ttl={s[1]['cache_control']['ttl']}). Not submitted.")
        return

    import anthropic
    client = anthropic.Anthropic(api_key=args.api_key)

    BATCH_DIR.mkdir(parents=True, exist_ok=True)
    state_path = BATCH_DIR / f"W{args.wave}_{model_tag}_anthropic_state.json"

    # load any already-submitted batch IDs (resume after mid-run failure)
    if state_path.exists():
        existing = json.load(open(state_path))
        batch_ids = existing.get("batch_ids", [])
        start_chunk = len(batch_ids)
        if start_chunk:
            print(f"  resuming: {start_chunk} batch(es) already submitted")
    else:
        batch_ids = []
        start_chunk = 0

    for i, chunk in enumerate(chunks):
        if i < start_chunk:
            continue
        print(f"  Submitting part {i+1}/{len(chunks)} ({len(chunk):,} requests)...")
        for attempt in range(4):
            try:
                batch = client.messages.batches.create(requests=chunk)
                break
            except Exception as e:
                if attempt == 3:
                    raise
                wait = 10 * (2 ** attempt)
                print(f"    Error (attempt {attempt+1}): {e}. Retrying in {wait}s...")
                time.sleep(wait)
        print(f"    Batch ID: {batch.id}  Status: {batch.processing_status}")
        batch_ids.append(batch.id)
        # save state after every batch so a mid-run failure is recoverable
        with open(state_path, "w") as f:
            json.dump({"batch_ids": batch_ids, "wave": args.wave, "model": args.model,
                       "model_tag": model_tag, "personas": args.personas,
                       "submitted_at": datetime.now().isoformat()}, f, indent=2)

    print(f"\n  {len(batch_ids)} batch(es) submitted. State -> {state_path}")
    print(f"  fetch: python indiv_sampling_anthropic.py fetch --wave {args.wave}")


# ── fetch ─────────────────────────────────────────────────────────────────────

def cmd_fetch(args):
    import anthropic
    model_tag = f"{args.model}_indiv"
    state_path = BATCH_DIR / f"W{args.wave}_{model_tag}_anthropic_state.json"
    if not state_path.exists():
        print(f"Error: no state at {state_path}"); sys.exit(1)
    state = json.load(open(state_path))
    client = anthropic.Anthropic(api_key=args.api_key)

    questions = load_questions("data/responses", args.wave)
    cells = build_cells(args.wave, human_dir="human_resp")
    qmeta = {q.question_id: (q.question_text, q.options) for q in questions}
    out_path = DATA_DIR / "results" / model_tag / f"W{args.wave}.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    import numpy as np
    counts = {c: [np.zeros(len(q.options), dtype=int) for q in questions]
              for c in range(len(cells))}
    already = _done_personas(out_path)
    new_personas = set()
    parsed = dupe = 0
    cache_read = cache_write = uncached_in = 0

    for bid in state["batch_ids"]:
        print(f"Checking batch {bid}...")
        while True:
            b = client.messages.batches.retrieve(bid)
            rc = b.request_counts
            print(f"  {b.processing_status} | processing={rc.processing} "
                  f"succeeded={rc.succeeded} errored={rc.errored}")
            if b.processing_status == "ended":
                break
            time.sleep(30)
        for res in client.messages.batches.results(bid):
            m = re.match(r"c(\d+)-p(\d+)", res.custom_id)
            if not m:
                continue
            c_idx, p_idx = int(m.group(1)), int(m.group(2))
            if (tuple(sorted(cells[c_idx].features)), p_idx) in already:
                dupe += 1; continue
            if res.result.type != "succeeded":
                continue
            msg = res.result.message
            u = msg.usage
            cache_read += getattr(u, "cache_read_input_tokens", 0) or 0
            cache_write += getattr(u, "cache_creation_input_tokens", 0) or 0
            uncached_in += u.input_tokens
            text = "".join(b.text for b in msg.content if b.type == "text").strip()
            picks = parse_persona_response(text, len(questions))
            any_valid = False
            for qi, pick in enumerate(picks):
                if pick is not None and 1 <= pick <= len(questions[qi].options):
                    counts[c_idx][qi][pick - 1] += 1
                    any_valid = True
            if any_valid:
                new_personas.add((tuple(sorted(cells[c_idx].features)), p_idx))
                parsed += 1

    tot_cache = cache_read + cache_write + uncached_in
    print(f"\nParsed {parsed:,} personas ({dupe:,} already-merged skipped).")
    if tot_cache:
        print(f"Cache: read {cache_read/1e6:.1f}M / write {cache_write/1e6:.1f}M / "
              f"uncached {uncached_in/1e6:.1f}M input tokens "
              f"({cache_read/tot_cache:.0%} served from cache)")

    # merge prior + write canonical
    prior = {}
    if out_path.exists():
        for ln in open(out_path):
            try:
                r = json.loads(ln)
                prior[(tuple(sorted(r["demographics"])), r["question_id"])] = \
                    np.array(r["response_distribution"], dtype=int)
            except Exception:
                pass
    written = 0
    tmp = out_path.with_suffix(".jsonl.tmp")
    with open(tmp, "w") as f:
        for c_idx, cell in enumerate(cells):
            ck = tuple(sorted(cell.features))
            for qi, q in enumerate(questions):
                add = counts[c_idx][qi]
                base = prior.get((ck, q.question_id))
                total = add if base is None else base + add
                if total.sum() == 0:
                    continue
                f.write(json.dumps({
                    "timestamp": datetime.now().isoformat(),
                    "demographics": cell.features,
                    "question_id": q.question_id,
                    "question_text": q.question_text,
                    "options": q.options,
                    "response_distribution": [int(x) for x in total],
                    "status": "success",
                    "elicitation": "individual_sampling_cached",
                    "n_personas": int(total.sum()),
                }) + "\n")
                written += 1
    tmp.replace(out_path)
    done = already | new_personas
    with open(out_path.with_suffix(".personas.json"), "w") as f:
        json.dump(sorted([list(x) for x in done]), f)
    print(f"Wrote {written} rows -> {out_path} ({len(done)} personas total)")
    print(f"Contest: cd verification && python verify_spine_collapse.py {model_tag}")


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    for name in ("estimate", "submit", "fetch"):
        sp = sub.add_parser(name)
        sp.add_argument("--wave", required=True)
        sp.add_argument("--model", default=DEFAULT_MODEL)
        sp.add_argument("--personas", type=int, default=100)
        sp.add_argument("--api-key", default=os.environ.get("ANTHROPIC_API_KEY"))
        if name == "submit":
            sp.add_argument("--ttl", default="1h", help="cache TTL: 5m or 1h")
            sp.add_argument("--dry-run", action="store_true")
    args = p.parse_args()
    os.chdir(Path(__file__).parent)

    if args.cmd == "estimate":
        cmd_estimate(args)
    elif args.cmd == "submit":
        if not args.api_key and not args.dry_run:
            print("Error: set ANTHROPIC_API_KEY or pass --api-key"); sys.exit(1)
        cmd_submit(args)
    elif args.cmd == "fetch":
        cmd_fetch(args)


if __name__ == "__main__":
    main()
