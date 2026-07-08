#!/usr/bin/env python3
"""
Expand depth-3 triples to include Political Party and Religion for
claude-haiku and gpt-4o-mini. Existing 4-feature triples are skipped
automatically (submit only queues cells without a success record).
"""

import os, subprocess, sys, time
from pathlib import Path

BASE   = Path(__file__).parent
PYTHON = sys.executable

def load_env():
    keys = {}
    for line in open(BASE / ".env").read().strip().replace("\r","").split("\n"):
        if "=" in line and not line.startswith("#"):
            k, v = line.split("=", 1)
            keys[k.strip()] = v.strip()
    return keys

_env = load_env()
ANTHROPIC_KEY = _env.get("ANTHROPIC_API_KEY", os.environ.get("ANTHROPIC_API_KEY", ""))
OPENAI_KEY    = _env.get("OPENAI_API_KEY",    os.environ.get("OPENAI_API_KEY",
    "SCRUBBED-SET-YOUR-OWN-KEY"))

ALL_WAVES = ["26","27","29","32","34","36","41","42","43","45","49","50","54","82","92"]

JOBS = [
    (
        "claude-haiku  triples expand (Party+Religion)",
        "batch_api_runner_anthropic.py",
        "claude-haiku-4-5-20251001",
        ANTHROPIC_KEY,
        ["--include-triples", "--run-tag", "triples_pr"],
        ["--run-tag", "triples_pr"],
    ),
    (
        "gpt-4o-mini   triples expand (Party+Religion)",
        "batch_api_runner.py",
        "gpt-4o-mini",
        OPENAI_KEY,
        ["--include-triples", "--triples-only"],
        [],
    ),
]

MAX_RETRIES = 5
RETRY_SLEEP = 60


def run_cmd(cmd, label, dry_run=False):
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
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--model", choices=["haiku", "mini"], default=None)
    args = ap.parse_args()

    jobs = JOBS
    if args.model == "haiku":
        jobs = [j for j in JOBS if "haiku" in j[2]]
    elif args.model == "mini":
        jobs = [j for j in JOBS if "mini" in j[2]]

    for label, runner, model, key, extra_sub, extra_fetch in jobs:
        print(f"\n{'='*60}")
        print(f"  {label}")
        print(f"{'='*60}")
        for wave in ALL_WAVES:
            run_cmd([PYTHON, runner, "submit",
                     "--wave", wave, "--model", model, "--api-key", key] + extra_sub,
                    f"W{wave}  submit", args.dry_run)
            run_cmd([PYTHON, runner, "fetch",
                     "--wave", wave, "--model", model, "--api-key", key] + extra_fetch,
                    f"W{wave}  fetch", args.dry_run)

    print("\n\nAll triple expansions done.")


if __name__ == "__main__":
    main()
