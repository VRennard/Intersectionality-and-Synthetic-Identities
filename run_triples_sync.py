#!/usr/bin/env python3
"""
Synchronous gpt-4o-mini Party/Religion triples for the waves whose batches
stalled (W27, W34, W41). Same prompts/cells as the batch path (triples_only,
q_batch_size=20), just called live instead of via the batch queue.

Writes to data/results/gpt-4o-mini/W{wave}.jsonl (resumes: skips success cells).
"""
import json, os, sys, threading, time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

import openai

sys.path.insert(0, str(Path(__file__).parent))
from batch_api_runner import (load_questions, load_profiles,
                              build_multi_question_prompt,
                              parse_multi_question_response)

KEY    = os.environ["OPENAI_API_KEY"]
MODEL  = "gpt-4o-mini"
WAVES  = sys.argv[1:] or ["27", "34", "41"]
QBS    = 20
WORKERS = 16
OUT    = Path("data/results") / MODEL


def load_done(path):
    done = set()
    if path.exists():
        for line in open(path, encoding="utf-8", errors="ignore"):
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except Exception:
                continue
            if r.get("status") == "success":
                done.add((tuple(r["demographics"]), r["question_id"]))
    return done


_local = threading.local()
def client():
    if not hasattr(_local, "c"):
        _local.c = openai.OpenAI(api_key=KEY)
    return _local.c


def run_wave(wave):
    questions = load_questions(wave)
    profiles  = load_profiles(wave, include_triples=True, triples_only=True)
    out_path  = OUT / f"W{wave}.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    done = load_done(out_path)

    # build list of (profile, q_batch) units
    units = []
    for prof in profiles:
        pend = [q for q in questions if (tuple(prof.features), q.question_id) not in done]
        for i in range(0, len(pend), QBS):
            units.append((prof, pend[i:i + QBS]))
    print(f"W{wave}: {len(units)} prompts ({sum(len(u[1]) for u in units):,} cells)", flush=True)

    lock = threading.Lock()
    counts = {"success": 0, "error": 0}

    def work(unit):
        prof, qbatch = unit
        msg = build_multi_question_prompt(prof, qbatch)
        dists = [None] * len(qbatch)
        for attempt in range(5):
            try:
                resp = client().chat.completions.create(
                    model=MODEL, max_tokens=QBS * 30 + 50, temperature=0.7,
                    messages=[{"role": "user", "content": msg}])
                dists = parse_multi_question_response(
                    resp.choices[0].message.content.strip(), qbatch)
                break
            except Exception:
                time.sleep(min(2 ** attempt, 20))
        recs = []
        for q, d in zip(qbatch, dists):
            recs.append(json.dumps({
                "timestamp": datetime.now().isoformat(),
                "demographics": prof.features, "question_id": q.question_id,
                "question_text": q.question_text, "options": q.options,
                "response_distribution": d or [],
                "status": "success" if d else "error"}))
        with lock:
            with open(out_path, "a", encoding="utf-8") as f:
                f.write("\n".join(recs) + "\n")
            for d in dists:
                counts["success" if d else "error"] += 1
            n = counts["success"] + counts["error"]
            if n % 5000 < QBS:
                print(f"  W{wave} ~{n}: {counts}", flush=True)

    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        futs = [pool.submit(work, u) for u in units]
        for f in as_completed(futs):
            f.result()
    print(f"W{wave} DONE: {counts}", flush=True)
    return counts


if __name__ == "__main__":
    print(f"SYNC {MODEL} P/R triples waves={WAVES}", flush=True)
    tot = {"success": 0, "error": 0}
    for w in WAVES:
        for k, v in run_wave(w).items():
            tot[k] += v
    print(f"\nALL DONE: {tot}", flush=True)
