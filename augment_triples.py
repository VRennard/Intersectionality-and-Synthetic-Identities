#!/usr/bin/env python3
"""
augment_triples.py

Runs all triple-feature profiles (Age x Gender x Race, etc.) for the original
4 features (Age, Gender, Race, Income) across specified waves, appending to
existing JSONL files. Already-completed pairs are skipped automatically.

Usage:
    python augment_triples.py --api-key sk-... --waves 26 27 29
"""

import os
import sys
import json
import argparse
from datetime import datetime
from itertools import combinations, product
from pathlib import Path

sys.path.insert(0, os.path.dirname(__file__))

from llm_prompt_survey import BatchPromptSimulator, DemographicProfile, ResponseParser
import simulation_config as _cfg

OLD_FEATURES = ["Age", "Gender", "Race", "Income"]


def get_vals(feature, wave):
    import pandas as pd
    df = pd.read_csv(
        Path("human_resp") / f"American_Trends_Panel_W{wave}" / "responses.csv",
        low_memory=False
    )
    col = _cfg.FEATURES[feature]
    if col not in df.columns:
        return []
    vals = df[col].dropna().unique().tolist()
    ignored = _cfg.IGNORE_VALUES.get(feature, [])
    return [str(v) for v in vals if str(v) not in ignored]


def build_triple_profiles(wave):
    profiles = []
    for a, b, c in combinations(OLD_FEATURES, 3):
        va, vb, vc = get_vals(a, wave), get_vals(b, wave), get_vals(c, wave)
        for xa, xb, xc in product(va, vb, vc):
            profiles.append(DemographicProfile(features=[f"{a} {xa}", f"{b} {xb}", f"{c} {xc}"]))
    return profiles


def load_done(jsonl_path):
    done = set()
    if not jsonl_path.exists():
        return done
    with open(jsonl_path) as f:
        for line in f:
            try:
                rec = json.loads(line)
                done.add((tuple(rec["demographics"]), rec["question_id"]))
            except Exception:
                pass
    return done


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-type", default="openai", choices=["openai", "ollama"])
    parser.add_argument("--model", default="gpt-4o-mini")
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--waves", nargs="+", required=True)
    args = parser.parse_args()

    api_key = args.api_key or os.getenv("OPENAI_API_KEY")
    model_tag = args.model.replace("/", "-")

    if args.model_type == "openai" and not api_key:
        parser.error("--api-key or OPENAI_API_KEY required")

    print(f"Triples worker | waves: {args.waves} | model: {args.model}")

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

        profiles = build_triple_profiles(wave)
        print(f"  {len(profiles)} triple profiles x {len(questions)} questions = {len(profiles)*len(questions)} calls")

        jsonl_path = Path("data") / "results" / model_tag / f"W{wave}.jsonl"
        done = load_done(jsonl_path)
        print(f"  Already done: {len(done)} records in file")

        added = skipped = 0

        for profile in profiles:
            for q in questions:
                key = (tuple(profile.features), q.question_id)
                if key in done:
                    skipped += 1
                    continue

                try:
                    msg = simulator.prompt_builder.build_prompt(profile, q)
                    resp = simulator.llm.call_model(
                        system_message=simulator.prompt_builder.SYSTEM_MESSAGE,
                        user_message=msg,
                    )
                    parsed = ResponseParser.parse_response(resp, len(q.options))
                    rec = {
                        "timestamp": datetime.now().isoformat(),
                        "demographics": profile.features,
                        "question_id": q.question_id,
                        "question_text": q.question_text,
                        "options": q.options,
                        "response_distribution": parsed or [],
                        "status": "success" if parsed else "failed",
                    }
                except Exception as e:
                    rec = {
                        "timestamp": datetime.now().isoformat(),
                        "demographics": profile.features,
                        "question_id": q.question_id,
                        "question_text": q.question_text,
                        "options": q.options,
                        "response_distribution": [],
                        "status": "error",
                        "error": str(e),
                    }

                with open(jsonl_path, "a") as f:
                    f.write(json.dumps(rec) + "\n")

                done.add(key)
                added += 1

            print(f"  profile {profile.features} done")

        print(f"  W{wave} complete. Added {added}, skipped {skipped}.")

    print("\nWorker done.")


if __name__ == "__main__":
    main()
