#!/usr/bin/env python3
"""
Repeatedly runs launch_fills.py until all waves report nothing left to submit.
Safe to interrupt — re-running picks up where it left off.
"""
import subprocess, sys, time, json
from pathlib import Path

BASE    = Path(__file__).parent
PYTHON  = sys.executable
MAX_RUNS = 20

def pending_count():
    """Count waves that still have cells to submit (success < expected)."""
    results = BASE / "data" / "results"
    # quick proxy: any model dir missing a wave file
    all_waves = ["W26","W27","W29","W32","W34","W36","W41","W42","W43",
                 "W45","W49","W50","W54","W82","W92"]
    missing = 0
    for model in ["claude-haiku-4-5-20251001"]:
        mdir = results / model
        for w in all_waves:
            if not (mdir / f"{w}.jsonl").exists():
                missing += 1
    return missing

for run in range(1, MAX_RUNS + 1):
    print(f"\n{'='*60}")
    print(f"  LOOP RUN {run}/{MAX_RUNS}")
    print(f"{'='*60}\n")

    result = subprocess.run(
        [PYTHON, "launch_fills.py"],
        cwd=str(BASE)
    )
    print(f"\nRun {run} finished (exit {result.returncode})")

    still_missing = pending_count()
    if still_missing == 0:
        print("All waves present — done!")
        break

    print(f"{still_missing} wave file(s) still missing. Sleeping 60s then retrying...")
    time.sleep(60)

print("\nLoop complete.")
