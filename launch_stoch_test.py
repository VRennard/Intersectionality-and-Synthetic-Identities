#!/usr/bin/env python3
"""
launch_stoch_test.py

Runs the same wave N times independently to measure stochasticity.
Uses all profiles from the existing gpt-4o-mini results for that wave.
Results saved to data/results/stoch_test/W{wave}_run{N}.jsonl

Usage:
    python launch_stoch_test.py submit --wave 26 --runs 4
    python launch_stoch_test.py fetch  --wave 26 --runs 4
"""

import argparse
import json
import os
import re
import sys
import time
from datetime import datetime
from pathlib import Path

import openai

BASE     = Path(__file__).parent
LOGS     = BASE / "logs"
OUT_DIR  = BASE / "data" / "results" / "stoch_test"
BATCH_DIR= BASE / "data" / "batches"
DATA_DIR = BASE / "data"
MODEL    = "gpt-4o-mini"
Q_BATCH  = 20

sys.path.insert(0, str(BASE))
from llm_prompt_survey import DemographicProfile, PromptBuilder, SurveyQuestion
import simulation_config as _sim_cfg


def get_api_key():
    key = os.getenv("OPENAI_API_KEY")
    if not key:
        keys_path = BATCH_DIR / "api_keys.json"
        with open(keys_path) as f:
            key = json.load(f)["key_1"]
    return key


def load_questions(wave):
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


def load_profiles_from_existing(wave):
    """Load all unique demographic profiles from existing gpt-4o-mini results."""
    path = BASE / "data" / "results" / "gpt-4o-mini" / f"W{wave}.jsonl"
    seen = set()
    profiles = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
                key = tuple(d["demographics"])
                if key not in seen:
                    seen.add(key)
                    profiles.append(DemographicProfile(features=list(key)))
            except Exception:
                pass
    return profiles


def build_multi_question_prompt(profile, questions):
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


def parse_multi(text, questions):
    cleaned = re.sub(r"```[a-z]*\n?", "", text).strip()
    try:
        result = eval(cleaned)
        if not isinstance(result, list):
            return [None] * len(questions)
        out = []
        for i, q in enumerate(questions):
            if i < len(result):
                inner = result[i]
                if isinstance(inner, list) and len(inner) == len(q.options) and sum(inner) == 1000:
                    out.append(inner)
                else:
                    out.append(None)
            else:
                out.append(None)
        return out
    except Exception:
        return [None] * len(questions)


def state_path(wave, run):
    return BATCH_DIR / f"W{wave}_stoch_run{run}_state.json"


def out_path(wave, run):
    return OUT_DIR / f"W{wave}_run{run}.jsonl"


# ── submit ─────────────────────────────────────────────────────────────────────

def cmd_submit(wave, runs):
    client = openai.OpenAI(api_key=get_api_key())
    questions = load_questions(wave)
    profiles  = load_profiles_from_existing(wave)
    print(f"W{wave}: {len(profiles)} profiles × {len(questions)} questions = {len(profiles)*len(questions):,} calls per run")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    BATCH_DIR.mkdir(parents=True, exist_ok=True)
    builder = PromptBuilder()

    for run in range(1, runs + 1):
        sp = state_path(wave, run)
        if sp.exists():
            with open(sp) as f:
                state = json.load(f)
            if state.get("status") == "submitted":
                print(f"  Run {run}: already submitted ({state['batch_ids']}) — skipping")
                continue

        input_path = BATCH_DIR / f"W{wave}_stoch_run{run}_input.jsonl"
        written = 0
        with open(input_path, "w") as f:
            for d_idx, profile in enumerate(profiles):
                for qb_idx, i in enumerate(range(0, len(questions), Q_BATCH)):
                    q_batch = questions[i:i + Q_BATCH]
                    user_msg = build_multi_question_prompt(profile, q_batch)
                    max_tok = Q_BATCH * 30 + 50
                    cid = f"d{d_idx}-qb{qb_idx}"
                    record = {
                        "custom_id": cid,
                        "method": "POST",
                        "url": "/v1/chat/completions",
                        "body": {
                            "model": MODEL,
                            "messages": [
                                {"role": "system", "content": builder.SYSTEM_MESSAGE},
                                {"role": "user",   "content": user_msg},
                            ],
                            "temperature": 0.7,
                            "max_tokens": max_tok,
                        },
                    }
                    f.write(json.dumps(record) + "\n")
                    written += 1

        # Upload + submit
        print(f"  Run {run}: uploading {written:,} requests...", end=" ", flush=True)
        with open(input_path, "rb") as f:
            uploaded = client.files.create(file=f, purpose="batch")
        batch = client.batches.create(
            input_file_id=uploaded.id,
            endpoint="/v1/chat/completions",
            completion_window="24h",
        )
        print(f"batch_id={batch.id}  status={batch.status}")

        with open(sp, "w") as f:
            json.dump({
                "wave": wave, "run": run, "model": MODEL,
                "batch_ids": [batch.id],
                "submitted_at": datetime.now().isoformat(),
                "status": "submitted",
                "q_batch_size": Q_BATCH,
            }, f, indent=2)

    print(f"\nAll {runs} runs submitted. Fetch when complete:")
    print(f"  python launch_stoch_test.py fetch --wave {wave} --runs {runs}")


# ── fetch ──────────────────────────────────────────────────────────────────────

def cmd_fetch(wave, runs):
    client = openai.OpenAI(api_key=get_api_key())
    questions = load_questions(wave)
    profiles  = load_profiles_from_existing(wave)
    builder   = PromptBuilder()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    for run in range(1, runs + 1):
        sp = state_path(wave, run)
        if not sp.exists():
            print(f"  Run {run}: no state file — skipping")
            continue
        with open(sp) as f:
            state = json.load(f)

        op = out_path(wave, run)
        if op.exists() and state.get("status") == "fetched":
            print(f"  Run {run}: already fetched ({op.stat().st_size//1024} KB) — skipping")
            continue

        bid = state["batch_ids"][0]
        batch = client.batches.retrieve(bid)
        c = batch.request_counts
        print(f"  Run {run}: status={batch.status} completed={c.completed}/{c.total}", end="")
        if batch.status != "completed":
            print(" — not ready yet")
            continue

        print(" — downloading...", end=" ", flush=True)
        raw_lines = client.files.content(batch.output_file_id).text.strip().splitlines()
        raw = {json.loads(l)["custom_id"]: json.loads(l) for l in raw_lines}

        success = error = 0
        with open(op, "w", encoding="utf-8") as out_f:
            for d_idx, profile in enumerate(profiles):
                for qb_idx, i in enumerate(range(0, len(questions), Q_BATCH)):
                    q_batch = questions[i:i + Q_BATCH]
                    cid = f"d{d_idx}-qb{qb_idx}"
                    obj = raw.get(cid)
                    if obj and obj.get("response") and obj["response"]["status_code"] == 200:
                        text = obj["response"]["body"]["choices"][0]["message"]["content"].strip()
                        dists = parse_multi(text, q_batch)
                    else:
                        dists = [None] * len(q_batch)
                    for question, dist in zip(q_batch, dists):
                        rec = {
                            "run": run,
                            "demographics": profile.features,
                            "question_id": question.question_id,
                            "response_distribution": dist or [],
                            "status": "success" if dist else "error",
                        }
                        out_f.write(json.dumps(rec) + "\n")
                        if dist: success += 1
                        else: error += 1

        state["status"] = "fetched"
        with open(sp, "w") as f:
            json.dump(state, f, indent=2)
        print(f"done - {success} ok, {error} errors -> {op.name}")


# ── main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("cmd", choices=["submit", "fetch"])
    parser.add_argument("--wave", default="26")
    parser.add_argument("--runs", type=int, default=4)
    args = parser.parse_args()

    os.chdir(BASE)
    if args.cmd == "submit":
        cmd_submit(args.wave, args.runs)
    else:
        cmd_fetch(args.wave, args.runs)


if __name__ == "__main__":
    main()
