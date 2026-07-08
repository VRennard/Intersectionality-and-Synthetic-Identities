#!/usr/bin/env python3
"""
Re-parse the already-completed gpt-4o-mini P/R-triple batches with the robust
parser (no new generation cost). Downloads each wave's batch OUTPUT file from
OpenAI, maps custom_ids back to the SUBMIT-TIME question batches (all questions,
since triples were new at submit), robust-parses, and appends newly-recovered
success cells. Skips cells already present as success.
"""
import json, os, sys
from datetime import datetime
from pathlib import Path

import openai

sys.path.insert(0, str(Path(__file__).parent))
from batch_api_runner import (load_questions, load_profiles,
                              make_batch_custom_id,
                              parse_multi_question_response)  # now robust

MODEL = "gpt-4o-mini"
QBS   = 20
WAVES = sys.argv[1:] or ["26","29","32","36","42","43","45","49","50","54","82","92"]
BD    = Path("data/batches")
OUTD  = Path("data/results") / MODEL


def existing_success(path):
    s = set()
    if path.exists():
        for line in open(path, encoding="utf-8", errors="ignore"):
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except Exception:
                continue
            if r.get("status") == "success":
                s.add((tuple(r["demographics"]), r["question_id"]))
    return s


def reparse_wave(wave, client):
    sf = BD / f"W{wave}_{MODEL}_state.json"
    state = json.load(open(sf))
    if not state.get("triples_only"):
        print(f"W{wave}: state not triples_only — skip"); return 0
    # download all output files for this wave's batches
    results = {}
    for bid in state["batch_ids"]:
        b = client.batches.retrieve(bid)
        ofid = getattr(b, "output_file_id", None)
        if not ofid:
            print(f"W{wave}: batch {bid} has no output_file (status={b.status}) — skip")
            continue
        for line in client.files.content(ofid).text.strip().splitlines():
            o = json.loads(line)
            results[o["custom_id"]] = o
    if not results:
        print(f"W{wave}: no downloadable output — skip"); return 0

    questions = load_questions(wave)
    profiles  = load_profiles(wave, include_triples=True, triples_only=True)
    out_path  = OUTD / f"W{wave}.jsonl"
    have = existing_success(out_path)

    recovered = 0
    new_lines = []
    for d_idx, prof in enumerate(profiles):
        # submit-time batching: ALL questions (triples were new at submit)
        for qb_idx, i in enumerate(range(0, len(questions), QBS)):
            q_batch = questions[i:i + QBS]
            o = results.get(make_batch_custom_id(d_idx, qb_idx))
            if not o or not o.get("response") or o["response"]["status_code"] != 200:
                continue
            text = o["response"]["body"]["choices"][0]["message"]["content"].strip()
            dists = parse_multi_question_response(text, q_batch)
            for q, d in zip(q_batch, dists):
                if not d:
                    continue
                key = (tuple(prof.features), q.question_id)
                if key in have:
                    continue
                have.add(key)
                new_lines.append(json.dumps({
                    "timestamp": datetime.now().isoformat(),
                    "demographics": prof.features, "question_id": q.question_id,
                    "question_text": q.question_text, "options": q.options,
                    "response_distribution": d, "status": "success"}))
                recovered += 1
    if new_lines:
        with open(out_path, "a", encoding="utf-8") as f:
            f.write("\n".join(new_lines) + "\n")
    print(f"W{wave}: recovered {recovered:,} new P/R cells", flush=True)
    return recovered


def main():
    key = json.load(open(BD / f"W{WAVES[0]}_{MODEL}_state.json"))["api_key"]
    client = openai.OpenAI(api_key=key)
    total = 0
    for w in WAVES:
        try:
            total += reparse_wave(w, client)
        except Exception as e:
            print(f"W{w}: ERROR {e}", flush=True)
    print(f"\nTOTAL recovered: {total:,}", flush=True)


if __name__ == "__main__":
    main()
