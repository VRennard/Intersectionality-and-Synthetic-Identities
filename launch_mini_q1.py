#!/usr/bin/env python3
"""
q_batch_size=1 correction pass for gpt-4o-mini P/R triples: recovers the cells
the multi-question (q20) prompts couldn't yield even with the robust parser
(model returned the wrong integer count). One question per call -> clean parse.

Phase 1: submit all 15 waves with --q-batch-size 1 (skips success cells, so only
the ~347k missing cells are queued). Phase 2: fetch all. Batch API (50% off).
"""
import os, sys, json, subprocess, time
from pathlib import Path

BASE   = Path(__file__).parent
PYTHON = sys.executable
RUNNER = "batch_api_runner.py"
MODEL  = "gpt-4o-mini"
KEY    = os.environ["OPENAI_API_KEY"]
WAVES  = ["26","27","29","32","34","36","41","42","43","45","49","50","54","82","92"]
BD     = BASE / "data" / "batches"
MAXR, SLEEP = 5, 60


def run(cmd, label):
    print(f"\n--- {label}", flush=True)
    for a in range(1, MAXR + 1):
        if subprocess.run(cmd, cwd=str(BASE)).returncode == 0:
            return 0
        if a < MAXR:
            print(f"  retry {a}/{MAXR-1} in {SLEEP}s", flush=True); time.sleep(SLEEP)
    print("  FAILED, continuing", flush=True)


def main():
    print("=== PHASE 1: submit q1 (all waves) ===", flush=True)
    for w in WAVES:
        run([PYTHON, RUNNER, "submit", "--wave", w, "--model", MODEL, "--api-key", KEY,
             "--include-triples", "--triples-only", "--q-batch-size", "1"], f"W{w} submit")
    print("\n=== PHASE 2: fetch q1 (all waves) ===", flush=True)
    for w in WAVES:
        run([PYTHON, RUNNER, "fetch", "--wave", w, "--model", MODEL, "--api-key", KEY,
             "--q-batch-size", "1"], f"W{w} fetch")
    print("\nALL Q1 DONE", flush=True)


if __name__ == "__main__":
    main()
