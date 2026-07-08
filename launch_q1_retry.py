#!/usr/bin/env python3
"""
Retry stuck cells with q_batch_size=1 for claude-haiku and gpt-4o-mini.
Only cells without a success record are submitted; everything else is skipped.

Usage:
    python launch_q1_retry.py
    python launch_q1_retry.py --dry-run
    python launch_q1_retry.py --model haiku
    python launch_q1_retry.py --model mini
"""

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

BASE   = Path(__file__).parent
PYTHON = sys.executable

def load_env():
    keys = {}
    env_path = BASE / ".env"
    if env_path.exists():
        for line in open(env_path).read().strip().replace("\r", "").split("\n"):
            if "=" in line and not line.startswith("#"):
                k, v = line.split("=", 1)
                keys[k.strip()] = v.strip()
    return keys

_env = load_env()
ANTHROPIC_KEY = _env.get("ANTHROPIC_API_KEY", os.environ.get("ANTHROPIC_API_KEY", ""))
OPENAI_KEY    = _env.get("OPENAI_API_KEY",    os.environ.get("OPENAI_API_KEY",
    "SCRUBBED-SET-YOUR-OWN-KEY"))

ALL_WAVES = ["26","27","29","32","34","36","41","42","43","45","49","50","54","82","92"]

# (label, runner, model, key, extra_submit_flags, extra_fetch_flags)
JOBS = [
    (
        "claude-haiku  q_batch_size=1 retry",
        "batch_api_runner_anthropic.py",
        "claude-haiku-4-5-20251001",
        ANTHROPIC_KEY,
        ["--run-tag", "q1retry"],
        ["--run-tag", "q1retry"],
    ),
    (
        "gpt-4o-mini   q_batch_size=1 retry",
        "batch_api_runner.py",
        "gpt-4o-mini",
        OPENAI_KEY,
        [],
        [],
    ),
]

MAX_RETRIES  = 5
RETRY_SLEEP  = 60


def run_cmd(cmd, label, dry_run):
    print(f"\n{'-'*60}")
    print(f"  {label}")
    print(f"  $ {' '.join(cmd)}")
    if dry_run:
        return 0
    for attempt in range(1, MAX_RETRIES + 1):
        result = subprocess.run(cmd, cwd=str(BASE), capture_output=False)
        if result.returncode == 0:
            return 0
        if attempt < MAX_RETRIES:
            print(f"  Exited {result.returncode} - retry {attempt}/{MAX_RETRIES-1} in {RETRY_SLEEP}s...")
            time.sleep(RETRY_SLEEP)
        else:
            print(f"  WARNING: failed after {MAX_RETRIES} attempts, continuing.")
    return result.returncode


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--model", choices=["haiku", "mini"], default=None,
                    help="run only one model (haiku or mini)")
    args = ap.parse_args()

    jobs = JOBS
    if args.model == "haiku":
        jobs = [j for j in JOBS if "haiku" in j[2]]
    elif args.model == "mini":
        jobs = [j for j in JOBS if "mini" in j[2]]

    for label, runner, model, key, extra_sub, extra_fetch in jobs:
        print(f"\n{'='*60}")
        print(f"  {label}")
        print(f"  Waves: {' '.join('W'+w for w in ALL_WAVES)}")
        print(f"{'='*60}")

        for wave in ALL_WAVES:
            submit_cmd = [PYTHON, runner, "submit",
                          "--wave", wave, "--model", model,
                          "--api-key", key, "--q-batch-size", "1"] + extra_sub
            run_cmd(submit_cmd, f"W{wave}  submit", args.dry_run)

            fetch_cmd = [PYTHON, runner, "fetch",
                         "--wave", wave, "--model", model,
                         "--api-key", key] + extra_fetch
            run_cmd(fetch_cmd, f"W{wave}  fetch", args.dry_run)

    print("\n\nAll q1 retries done.")


if __name__ == "__main__":
    main()
