#!/usr/bin/env python3
"""
run_average_american.py

Runs a single "Average American" profile (no demographic features) for all
questions across all specified waves, appending to existing JSONL files.
Already-completed pairs are skipped automatically.

Usage:
    python run_average_american.py --api-key sk-...
    python run_average_american.py --waves 26 27 29
"""

import os
import sys
import json
import argparse
from datetime import datetime
from pathlib import Path

sys.path.insert(0, os.path.dirname(__file__))

from llm_prompt_survey import BatchPromptSimulator, DemographicProfile, ResponseParser
import simulation_config as _cfg

PROFILE = DemographicProfile(features=["Average American"])
ALL_WAVES = [26, 27, 29, 32, 34, 36, 41, 42, 43, 45, 49, 50, 54, 82, 92]


def load_done(jsonl_path):
    done = set()
    if not jsonl_path.exists():
        return done
    with open(jsonl_path) as f:
        for line in f:
            try:
                rec = json.loads(line)
                if tuple(rec["demographics"]) == ("Average American",):
                    done.add(rec["question_id"])
            except Exception:
                pass
    return done


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-type", default="openai", choices=["openai", "ollama"])
    parser.add_argument("--model", default="gpt-4o-mini")
    parser.add_argument("--api-key", default=os.environ.get("OPENAI_API_KEY"))
    parser.add_argument("--waves", nargs="+", type=int, default=ALL_WAVES)
    args = parser.parse_args()

    api_key = args.api_key
    model_tag = args.model.replace("/", "_")

    if args.model_type == "openai" and not api_key:
        parser.error("--api-key or OPENAI_API_KEY required")

    print(f"Average American run | waves: {args.waves} | model: {args.model}")

    for wave in args.waves:
        print(f"\n-- Wave W{wave} {'-'*40}")

        survey_path = Path("data") / "responses" / f"survey_responses_W{wave}.json"
        if not survey_path.exists():
            print(f"  [skip] no survey file")
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
        print(f"  {len(questions)} questions | already done: {len(done)}")

        added = skipped = 0

        for q in questions:
            if q.question_id in done:
                skipped += 1
                continue

            try:
                msg = simulator.prompt_builder.build_prompt(PROFILE, q)
                resp = simulator.llm.call_model(
                    system_message=simulator.prompt_builder.SYSTEM_MESSAGE,
                    user_message=msg,
                )
                parsed = ResponseParser.parse_response(resp, len(q.options))
                rec = {
                    "timestamp": datetime.now().isoformat(),
                    "demographics": PROFILE.features,
                    "question_id": q.question_id,
                    "question_text": q.question_text,
                    "options": q.options,
                    "response_distribution": parsed or [],
                    "status": "success" if parsed else "failed",
                }
            except Exception as e:
                rec = {
                    "timestamp": datetime.now().isoformat(),
                    "demographics": PROFILE.features,
                    "question_id": q.question_id,
                    "question_text": q.question_text,
                    "options": q.options,
                    "response_distribution": [],
                    "status": "error",
                    "error": str(e),
                }

            with open(jsonl_path, "a") as f:
                f.write(json.dumps(rec) + "\n")

            done.add(q.question_id)
            added += 1

        print(f"  W{wave} complete. Added {added}, skipped {skipped}.")

    print("\nDone.")


if __name__ == "__main__":
    main()
