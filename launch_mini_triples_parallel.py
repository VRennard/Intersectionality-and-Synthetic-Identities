#!/usr/bin/env python3
"""
gpt-4o-mini Party/Religion triples — PARALLEL mode.

Phase 1: submit ALL 15 waves up front (each submit just uploads + creates the
         batch; OpenAI then processes all 15 concurrently).
Phase 2: fetch ALL 15 waves (each fetch polls its own batch until done).

Skips a wave's submit if it already has a triples_only batch in its state file
(so the already-submitted W26 is not re-submitted / duplicated).
"""
import os, sys, json, subprocess, time
from pathlib import Path

BASE   = Path(__file__).parent
PYTHON = sys.executable
RUNNER = "batch_api_runner.py"
MODEL  = "gpt-4o-mini"
KEY    = os.environ["OPENAI_API_KEY"]
WAVES  = ["26","27","29","32","34","36","41","42","43","45","49","50","54","82","92"]
BATCH_DIR = BASE / "data" / "batches"

MAX_RETRIES = 5
RETRY_SLEEP = 60


def run(cmd, label):
    print(f"\n--- {label}\n$ {' '.join(cmd[:6])} ...", flush=True)
    for attempt in range(1, MAX_RETRIES + 1):
        rc = subprocess.run(cmd, cwd=str(BASE)).returncode
        if rc == 0:
            return 0
        if attempt < MAX_RETRIES:
            print(f"  exit {rc} — retry {attempt}/{MAX_RETRIES-1} in {RETRY_SLEEP}s", flush=True)
            time.sleep(RETRY_SLEEP)
    print(f"  FAILED after {MAX_RETRIES} attempts — continuing", flush=True)
    return rc


def already_submitted(wave):
    sf = BATCH_DIR / f"W{wave}_{MODEL}_state.json"
    if not sf.exists():
        return False
    try:
        d = json.load(open(sf))
        return bool(d.get("triples_only")) and bool(d.get("batch_ids"))
    except Exception:
        return False


def main():
    print("=" * 60)
    print("PHASE 1 — submit all waves")
    print("=" * 60, flush=True)
    for w in WAVES:
        if already_submitted(w):
            print(f"W{w}: already submitted — skip", flush=True)
            continue
        run([PYTHON, RUNNER, "submit", "--wave", w, "--model", MODEL,
             "--api-key", KEY, "--include-triples", "--triples-only"],
            f"W{w} submit")

    print("\n" + "=" * 60)
    print("PHASE 2 — fetch all waves")
    print("=" * 60, flush=True)
    for w in WAVES:
        run([PYTHON, RUNNER, "fetch", "--wave", w, "--model", MODEL,
             "--api-key", KEY],
            f"W{w} fetch")

    print("\nAll done.", flush=True)


if __name__ == "__main__":
    main()
