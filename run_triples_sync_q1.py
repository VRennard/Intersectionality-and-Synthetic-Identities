#!/usr/bin/env python3
"""
Single-question (q1) synchronous fill of remaining gpt-4o-mini P/R triple cells
for waves whose q1 batches stalled in OpenAI's queue. One question per call ->
clean parse, fills to ~100%. Resumes (skips success cells).

Usage: OPENAI_API_KEY=... python run_triples_sync_q1.py 32 45 54 82
"""
import json, os, sys, threading, time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

import openai

sys.path.insert(0, str(Path(__file__).parent))
from batch_api_runner import load_questions, load_profiles
from llm_prompt_survey import PromptBuilder, ResponseParser

KEY = os.environ["OPENAI_API_KEY"]
MODEL = "gpt-4o-mini"
WAVES = sys.argv[1:] or ["32", "45", "54", "82"]
WORKERS = 16
OUT = Path("data/results") / MODEL
BUILDER = PromptBuilder()


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
    profiles = load_profiles(wave, include_triples=True, triples_only=True)
    out_path = OUT / f"W{wave}.jsonl"
    done = load_done(out_path)
    tasks = [(p, q) for p in profiles for q in questions
             if (tuple(p.features), q.question_id) not in done]
    print(f"W{wave}: {len(tasks):,} cells to fill ({len(done):,} done)", flush=True)
    lock = threading.Lock()
    counts = {"success": 0, "error": 0}

    def work(t):
        prof, q = t
        dist = None
        for attempt in range(5):
            try:
                resp = client().chat.completions.create(
                    model=MODEL, max_tokens=100, temperature=0.7,
                    messages=[{"role": "user", "content": BUILDER.build_prompt(prof, q)}])
                dist = ResponseParser.parse_response(
                    resp.choices[0].message.content.strip(), len(q.options))
                break
            except Exception:
                time.sleep(min(2 ** attempt, 20))
        rec = json.dumps({
            "timestamp": datetime.now().isoformat(), "demographics": prof.features,
            "question_id": q.question_id, "question_text": q.question_text,
            "options": q.options, "response_distribution": dist or [],
            "status": "success" if dist else "error"})
        with lock:
            with open(out_path, "a", encoding="utf-8") as f:
                f.write(rec + "\n")
            counts["success" if dist else "error"] += 1
            n = counts["success"] + counts["error"]
            if n % 5000 == 0:
                print(f"  W{wave} {n}: {counts}", flush=True)

    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        for f in as_completed([pool.submit(work, t) for t in tasks]):
            f.result()
    print(f"W{wave} DONE: {counts}", flush=True)


if __name__ == "__main__":
    print(f"q1 SYNC fill waves={WAVES}", flush=True)
    for w in WAVES:
        run_wave(w)
    print("\nALL DONE", flush=True)
