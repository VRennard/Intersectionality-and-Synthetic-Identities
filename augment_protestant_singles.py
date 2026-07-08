#!/usr/bin/env python3
"""
augment_protestant_singles.py

Adds "Religion Protestant" single-dimension profiles across all waves.
Already-completed (profile, question) pairs are skipped automatically.

Usage:
    python augment_protestant_singles.py --api-key sk-... --waves all
    python augment_protestant_singles.py --api-key sk-... --waves 26 27 29
"""

import os
import sys
import json
import argparse
from datetime import datetime
from pathlib import Path

sys.path.insert(0, os.path.dirname(__file__))

from llm_prompt_survey import BatchPromptSimulator, DemographicProfile, ResponseParser

PROTESTANT_PROFILE = DemographicProfile(features=["Religion Protestant"])


def load_done(jsonl_path):
    done = set()
    if not jsonl_path.exists():
        return done
    with open(jsonl_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                done.add((tuple(rec["demographics"]), rec["question_id"]))
            except Exception:
                pass
    return done


def available_waves():
    results_dir = Path("data") / "results" / "gpt-4o-mini"
    return sorted(
        f.stem.replace("W", "")
        for f in results_dir.glob("W*.jsonl")
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-type", default="openai", choices=["openai", "ollama"])
    parser.add_argument("--model", default="gpt-4o-mini")
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--waves", nargs="+", required=True,
                        help="Wave numbers (e.g. 26 27 50) or 'all'")
    args = parser.parse_args()

    api_key = args.api_key or os.getenv("OPENAI_API_KEY")
    model_tag = args.model.replace("/", "-")

    if args.model_type == "openai" and not api_key:
        parser.error("--api-key or OPENAI_API_KEY required")

    waves = available_waves() if args.waves == ["all"] else args.waves

    print(f"Protestant singles augment | waves: {waves} | model: {args.model}")

    for wave in waves:
        print(f"\n-- Wave W{wave} {'-'*40}")

        survey_path = Path("data") / "responses" / f"survey_responses_W{wave}.json"
        if not survey_path.exists():
            print(f"  [skip] no survey file: {survey_path}")
            continue

        simulator = BatchPromptSimulator(
            model_type=args.model_type,
            model_name=args.model,
            data_file=str(survey_path),
            openai_api_key=api_key,
        )
        questions = simulator.data_loader.questions

        jsonl_path = Path("data") / "results" / model_tag / f"W{wave}.jsonl"
        done = load_done(jsonl_path)

        to_do = [
            q for q in questions
            if (tuple(PROTESTANT_PROFILE.features), q.question_id) not in done
        ]
        print(f"  {len(questions)} questions total | Already done: {len(questions) - len(to_do)} | Remaining: {len(to_do)}")

        if not to_do:
            print(f"  W{wave}: nothing to do, skipping.")
            continue

        added = errors = 0
        for q in to_do:
            try:
                msg = simulator.prompt_builder.build_prompt(PROTESTANT_PROFILE, q)
                resp = simulator.llm.call_model(
                    system_message=simulator.prompt_builder.SYSTEM_MESSAGE,
                    user_message=msg,
                )
                parsed = ResponseParser.parse_response(resp, len(q.options))
                rec = {
                    "timestamp": datetime.now().isoformat(),
                    "demographics": PROTESTANT_PROFILE.features,
                    "question_id": q.question_id,
                    "question_text": q.question_text,
                    "options": q.options,
                    "response_distribution": parsed or [],
                    "status": "success" if parsed else "failed",
                }
            except Exception as e:
                rec = {
                    "timestamp": datetime.now().isoformat(),
                    "demographics": PROTESTANT_PROFILE.features,
                    "question_id": q.question_id,
                    "question_text": q.question_text,
                    "options": q.options,
                    "response_distribution": [],
                    "status": "error",
                    "error": str(e),
                }
                errors += 1

            with open(jsonl_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec) + "\n")
            added += 1

        print(f"  W{wave} done. Added {added} records ({errors} errors).")

    print("\nAll done.")


if __name__ == "__main__":
    main()
