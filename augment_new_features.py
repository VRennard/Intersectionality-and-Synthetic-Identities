#!/usr/bin/env python3
"""
augment_new_features.py

For each wave, runs ONE representative question through the new demographic
features (Political Party, Religion, Education) and appends the results to
the existing JSONL files in data/results/{model_tag}/.

Already-completed (demographics, question_id) pairs are skipped automatically.

Usage:
    python augment_new_features.py --api-key sk-...
    python augment_new_features.py   # reads OPENAI_API_KEY from env
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

# ── One representative question per wave ──────────────────────────────────────
WAVE_QUESTION_IDX = {
    "26": 0,   # How safe is your community from crime?
    "27": 6,   # Driverless vehicles awareness
    "29": 3,   # Men vs women: hobbies & interests
    "32": 7,   # Can you trust most people?
    "34": 0,   # Has science made life easier or harder?
    "36": 1,   # Should high-office holders be honest and ethical?
    "41": 4,   # Race relations in 30 years?
    "42": 7,   # Confidence in scientists
    "43": 4,   # Race/ethnicity in college admissions
    "45": 2,   # How do you prefer to get news?
    "49": 4,   # Social media & 2020 election
    "50": 5,   # Importance of marriage for a man
    "54": 0,   # Household financial situation
    "82": 0,   # Current economic situation in the US
    "92": 0,   # Biden approval
}

# ── New features to simulate ───────────────────────────────────────────────────
NEW_FEATURES = ["Political Party", "Religion", "Education"]

# Values for each feature (from simulation_config, minus ignored)
def get_feature_values(feature: str, wave: str) -> list:
    import pandas as pd
    resp_path = Path("human_resp") / f"American_Trends_Panel_W{wave}" / "responses.csv"
    df = pd.read_csv(resp_path, low_memory=False)
    col = _cfg.FEATURES[feature]
    if col not in df.columns:
        return []
    values = df[col].dropna().unique().tolist()
    ignored = _cfg.IGNORE_VALUES.get(feature, [])
    return [str(v) for v in values if str(v) not in ignored]


def load_done(jsonl_path: Path) -> set:
    """Return set of (demographics_tuple, question_id) already in the JSONL."""
    done = set()
    if not jsonl_path.exists():
        return done
    with open(jsonl_path) as f:
        for line in f:
            try:
                rec = json.loads(line)
                key = (tuple(rec["demographics"]), rec["question_id"])
                done.add(key)
            except Exception:
                pass
    return done


def main():
    parser = argparse.ArgumentParser(description="Augment results with new demographic features.")
    parser.add_argument("--model-type", default="openai", choices=["openai", "ollama"])
    parser.add_argument("--model", default="gpt-4o-mini")
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--waves", nargs="+", default=None,
                        help="Subset of waves, e.g. --waves 26 42 92")
    args = parser.parse_args()

    api_key   = args.api_key or os.getenv("OPENAI_API_KEY")
    model_tag = args.model.replace("/", "-")

    if args.model_type == "openai" and not api_key:
        parser.error("--api-key or OPENAI_API_KEY required for openai backend")

    waves = args.waves or list(WAVE_QUESTION_IDX.keys())

    print("=" * 64)
    print("AUGMENT: Political Party / Religion / Education")
    print(f"Model : {args.model_type} / {args.model}")
    print(f"Waves : {waves}")
    print("=" * 64)

    for wave in waves:
        if wave not in WAVE_QUESTION_IDX:
            print(f"\n[W{wave}] No question index defined, skipping.")
            continue

        print(f"\n-- Wave W{wave} {'-'*50}")

        # Load all questions for this wave
        survey_path = Path("data") / "responses" / f"survey_responses_W{wave}.json"
        if not survey_path.exists():
            print(f"  [skip] {survey_path} not found.")
            continue

        # Build simulator (just for the LLM interface)
        simulator = BatchPromptSimulator(
            model_type=args.model_type,
            model_name=args.model,
            data_file=str(survey_path),
            openai_api_key=api_key,
        )
        llm_questions = simulator.data_loader.questions
        print(f"  Questions: {len(llm_questions)}")

        # Output JSONL path
        jsonl_path = Path("data") / "results" / model_tag / f"W{wave}.jsonl"
        jsonl_path.parent.mkdir(parents=True, exist_ok=True)

        done = load_done(jsonl_path)
        print(f"  Already done: {len(done)} records in file.")

        added = 0
        skipped = 0

        for feature in NEW_FEATURES:
            values = get_feature_values(feature, wave)
            if not values:
                print(f"  [W{wave}] No values found for {feature}, skipping.")
                continue

            for val in values:
                profile = DemographicProfile(features=[f"{feature} {val}"])

                for llm_question in llm_questions:
                    key = (tuple(profile.features), llm_question.question_id)

                    if key in done:
                        skipped += 1
                        continue

                    try:
                        msg = simulator.prompt_builder.build_prompt(profile, llm_question)
                        resp = simulator.llm.call_model(
                            system_message=simulator.prompt_builder.SYSTEM_MESSAGE,
                            user_message=msg,
                        )
                        parsed = ResponseParser.parse_response(resp, len(llm_question.options))
                        rec = {
                            "timestamp": datetime.now().isoformat(),
                            "demographics": profile.features,
                            "question_id": llm_question.question_id,
                            "question_text": llm_question.question_text,
                            "options": llm_question.options,
                            "response_distribution": parsed or [],
                            "status": "success" if parsed else "failed",
                        }
                        status = "ok" if parsed else "parse_failed"
                    except Exception as e:
                        rec = {
                            "timestamp": datetime.now().isoformat(),
                            "demographics": profile.features,
                            "question_id": llm_question.question_id,
                            "question_text": llm_question.question_text,
                            "options": llm_question.options,
                            "response_distribution": [],
                            "status": "error",
                            "error": str(e),
                        }
                        status = f"ERROR: {e}"

                    with open(jsonl_path, "a") as f:
                        f.write(json.dumps(rec) + "\n")

                    done.add(key)
                    added += 1

                print(f"  [{feature} {val}] {len(llm_questions)} questions -> done")

        print(f"  Wave W{wave} done. Added {added}, skipped {skipped} existing.")

    print("\n" + "=" * 64)
    print("Done.")


if __name__ == "__main__":
    main()
