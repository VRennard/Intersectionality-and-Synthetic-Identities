#!/usr/bin/env python3
"""
launch_edu_doubles_batch.py

Submits (or fetches) Education-doubles batch jobs for gpt-4o-mini.
The batch_api_runner already skips existing (profile, question) pairs,
so this will only submit the 6 missing Education×* double combos.

Usage:
    python launch_edu_doubles_batch.py submit
    python launch_edu_doubles_batch.py fetch
"""

import os
import sys
import subprocess
import argparse
import json
from pathlib import Path

BASE   = Path(__file__).parent
RUNNER = BASE / "batch_api_runner.py"
LOGS   = BASE / "logs"
MODEL  = "gpt-4o-mini"
WAVES  = ["26","27","29","32","34","36","41","42","43","45","49","50","54","82","92"]

# Load API key: env var > api_keys.json (key_1) > launch_w41_w42.py fallback
API_KEY = os.getenv("OPENAI_API_KEY")
if not API_KEY:
    keys_path = BASE / "data" / "batches" / "api_keys.json"
    if keys_path.exists():
        with open(keys_path) as f:
            keys = json.load(f)
        API_KEY = keys.get("key_1")


def run_wave(cmd: str, wave: str):
    log_path = LOGS / f"edu_doubles_{cmd}_W{wave}.log"
    args = [
        sys.executable, "-u", str(RUNNER),
        cmd,
        "--wave", wave,
        "--model", MODEL,
        "--api-key", API_KEY,
    ]
    if cmd == "submit":
        args += ["--q-batch-size", "20"]

    print(f"  W{wave}: running {cmd}...", end=" ", flush=True)
    with open(log_path, "w") as log:
        result = subprocess.run(args, cwd=BASE, stdout=log, stderr=subprocess.STDOUT, text=True)

    if result.returncode == 0:
        # Extract batch IDs from log for submit
        if cmd == "submit":
            with open(log_path) as f:
                for line in f:
                    if "batch_ids" in line.lower() or "batch_" in line.lower():
                        print(line.strip())
                        break
            print(f"  -> log: {log_path.name}")
        else:
            print("done")
    else:
        print(f"ERROR (rc={result.returncode}) — check {log_path.name}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("cmd", choices=["submit", "fetch"])
    parser.add_argument("--waves", nargs="+", default=WAVES,
                        help="Subset of waves, e.g. --waves 26 27")
    args = parser.parse_args()

    if not API_KEY:
        print("Error: OPENAI_API_KEY not found. Set it in the .env file or environment.")
        sys.exit(1)

    LOGS.mkdir(exist_ok=True)

    print(f"\nEducation doubles batch {args.cmd} | model={MODEL} | {len(args.waves)} waves\n")
    for wave in args.waves:
        run_wave(args.cmd, wave)

    if args.cmd == "submit":
        print("\nAll batches submitted.")
        print("Run again with 'fetch' when OpenAI marks them complete (usually < 1h):")
        print("  python launch_edu_doubles_batch.py fetch")
    else:
        print("\nAll waves fetched. Results appended to data/results/gpt-4o-mini/W*.jsonl")


if __name__ == "__main__":
    main()
