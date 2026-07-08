#!/usr/bin/env python3
"""
Fill launcher — submits then immediately fetches each wave before moving on,
so state files never collide between jobs.

Jobs (run in order):
  1. claude-haiku  depth-1+2 fill  : W36, W49
  2. claude-haiku  depth-3 triples : all 15 waves
  3. gpt-4o        depth-1+2 fill  : W36, W49

Usage:
    python launch_fills.py            # run everything
    python launch_fills.py --dry-run  # print commands only, don't execute
    python launch_fills.py --job 1    # run only job 1 (1-indexed)
    python launch_fills.py --job 2
    python launch_fills.py --job 3
"""

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

BASE   = Path(__file__).parent
PYTHON = sys.executable

# ── API keys ──────────────────────────────────────────────────────────────────

def load_env():
    keys = {}
    env_path = BASE / ".env"
    if env_path.exists():
        with open(env_path) as f:
            for line in f:
                line = line.strip().replace("\r", "")
                if "=" in line and not line.startswith("#"):
                    k, v = line.split("=", 1)
                    keys[k.strip()] = v.strip()
    return keys

_env = load_env()
ANTHROPIC_KEY = _env.get("ANTHROPIC_API_KEY", os.environ.get("ANTHROPIC_API_KEY", ""))
OPENAI_KEY    = _env.get("OPENAI_API_KEY",    os.environ.get("OPENAI_API_KEY",
    "SCRUBBED-SET-YOUR-OWN-KEY"))

ALL_WAVES = ["26","27","29","32","34","36","41","42","43","45","49","50","54","82","92"]

# ── job definitions ───────────────────────────────────────────────────────────
# Each entry: (label, runner_script, model, waves, extra_submit_flags)
# The launcher does submit → fetch per wave before starting the next wave,
# so state files never collide between jobs that share the same model/wave.

JOBS = [
    (
        "claude-haiku  depth-1+2 fill",
        "batch_api_runner_anthropic.py",
        "claude-haiku-4-5-20251001",
        ["36", "49"],
        [],                          # no extra flags → depth 1+2 only
    ),
    (
        "claude-haiku  depth-3 triples",
        "batch_api_runner_anthropic.py",
        "claude-haiku-4-5-20251001",
        ALL_WAVES,
        ["--include-triples"],
    ),
    (
        "gpt-4o        depth-1+2 fill",
        "batch_api_runner.py",
        "gpt-4o",
        ["36", "49"],
        [],
    ),
]

# ── helpers ───────────────────────────────────────────────────────────────────

def api_key_for(runner):
    return ANTHROPIC_KEY if "anthropic" in runner else OPENAI_KEY


MAX_RETRIES = 8
RETRY_SLEEP  = 60   # seconds between retries on connection error


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
        # Check if it's worth retrying (connection errors exit non-zero)
        if attempt < MAX_RETRIES:
            print(f"  Exited {result.returncode} — retry {attempt}/{MAX_RETRIES-1} in {RETRY_SLEEP}s...")
            time.sleep(RETRY_SLEEP)
        else:
            print(f"  WARNING: failed after {MAX_RETRIES} attempts, continuing.")
    return result.returncode


def run_job(job_idx, label, runner, model, waves, extra_flags, dry_run):
    key = api_key_for(runner)
    print(f"\n{'='*60}")
    print(f"  JOB {job_idx}: {label}")
    print(f"  Waves: {' '.join('W'+w for w in waves)}")
    print(f"{'='*60}")

    for wave in waves:
        wave_label = f"W{wave}  submit"
        submit_cmd = [PYTHON, runner, "submit",
                      "--wave", wave,
                      "--model", model,
                      "--api-key", key] + extra_flags
        run_cmd(submit_cmd, wave_label, dry_run)

        wave_label = f"W{wave}  fetch  (polls until batch ends)"
        fetch_cmd  = [PYTHON, runner, "fetch",
                      "--wave", wave,
                      "--model", model,
                      "--api-key", key]
        run_cmd(fetch_cmd, wave_label, dry_run)


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dry-run", action="store_true",
                    help="print commands without executing")
    ap.add_argument("--job", type=int, default=None,
                    help="run only this job number (1-indexed)")
    args = ap.parse_args()

    if not ANTHROPIC_KEY:
        print("ERROR: ANTHROPIC_API_KEY not in .env or environment"); sys.exit(1)
    if not OPENAI_KEY:
        print("ERROR: OPENAI_API_KEY not in .env or environment"); sys.exit(1)

    print("Keys loaded:")
    print(f"  Anthropic  ...{ANTHROPIC_KEY[-8:]}")
    print(f"  OpenAI     ...{OPENAI_KEY[-8:]}")
    if args.dry_run:
        print("  DRY RUN — commands printed only\n")

    jobs_to_run = [(i+1, *j) for i, j in enumerate(JOBS)]
    if args.job is not None:
        jobs_to_run = [j for j in jobs_to_run if j[0] == args.job]
        if not jobs_to_run:
            print(f"No job {args.job}"); sys.exit(1)

    for job_idx, label, runner, model, waves, extra in jobs_to_run:
        run_job(job_idx, label, runner, model, waves, extra, args.dry_run)

    print("\n\nAll done.")


if __name__ == "__main__":
    main()
