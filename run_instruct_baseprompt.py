#!/usr/bin/env python3
"""
De-confound test: run Mistral 7B INSTRUCT (open-mistral-7b, Mistral API) with the
BASE few-shot prompt (verbatim from ollama_fill/run_ollama_fill.py).

Fills the missing 2x2 cell: model=instruct x prompt=base. Compared to:
  - mistral_7b_text   (base model    + base prompt) -> isolates MODEL effect
  - mistral_latest    (instruct model + instruct prompt) -> isolates PROMPT effect

Runs the SAME (demographics, question) cells that exist in mistral_7b_text for the
chosen waves, so the comparison is cell-for-cell. Synchronous (batch API rejects
open-mistral-7b). Resumable: skips success cells already in the output.

Output: data/results/mistral_7b_instruct_baseprompt/W{wave}.jsonl
"""
import json, re, sys, time, threading, os
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).parent / "ollama_fill"))
from run_ollama_fill import build_base_prompt, parse_response   # exact base prompt + parser

KEY    = os.environ["MISTRAL_API_KEY"]
MODEL  = "open-mistral-7b"
WAVES  = sys.argv[1:] or ["26", "34"]
OUT    = Path("data/results/mistral_7b_instruct_baseprompt")
SRC    = Path("data/results/mistral_7b_text")   # cell list to mirror
SURVEY = Path("data/responses")
WORKERS = 12
URL = "https://api.mistral.ai/v1/chat/completions"


def load_questions(wave):
    data = json.load(open(SURVEY / f"survey_responses_W{wave}.json", encoding="utf-8"))
    return {q["question_id"]: (q["question_text"], [r["option"] for r in q["responses"]])
            for q in data}


def load_cells(wave):
    """(demographics, qid) cells that succeeded in mistral_7b_text for this wave."""
    cells, seen = [], set()
    p = SRC / f"W{wave}.jsonl"
    for line in open(p, encoding="utf-8", errors="ignore"):
        line = line.strip()
        if not line:
            continue
        try:
            r = json.loads(line)
        except Exception:
            continue
        if r.get("status") != "success":
            continue
        k = (tuple(r["demographics"]), r["question_id"])
        if k in seen:
            continue
        seen.add(k)
        cells.append((r["demographics"], r["question_id"]))
    return cells


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
def session():
    if not hasattr(_local, "s"):
        _local.s = requests.Session()
    return _local.s


def extract_list(text, n):
    """open-mistral-7b answers conversationally; pull the integer list from prose."""
    for c in reversed(re.findall(r"\[[\s\d,]+\]", text)):
        try:
            v = json.loads(c)
        except Exception:
            continue
        if isinstance(v, list) and all(isinstance(x, int) for x in v) and len(v) >= 2:
            return parse_response(c, n)
    return None


def call(prompt):
    payload = {"model": MODEL, "max_tokens": 450, "temperature": 0.7, "top_p": 0.9,
               "messages": [{"role": "user", "content": prompt}]}
    last = None
    for attempt in range(6):
        try:
            r = session().post(URL, headers={"Authorization": f"Bearer {KEY}",
                               "Content-Type": "application/json"}, json=payload, timeout=120)
            if r.status_code == 429:
                time.sleep(min(2 ** attempt, 30)); continue
            r.raise_for_status()
            return r.json()["choices"][0]["message"]["content"]
        except Exception as e:
            last = e; time.sleep(min(2 ** attempt, 30))
    raise RuntimeError(f"failed: {last}")


def run_wave(wave):
    questions = load_questions(wave)
    out_path = OUT / f"W{wave}.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    done = load_done(out_path)
    cells = [(d, q) for (d, q) in load_cells(wave)
             if (tuple(d), q) not in done and q in questions]
    print(f"W{wave}: {len(cells)} to run ({len(done)} done)", flush=True)
    lock = threading.Lock()
    counts = {"success": 0, "failed": 0, "error": 0}

    def work(cell):
        feats, qid = cell
        qtext, options = questions[qid]
        rec = {"timestamp": datetime.now().isoformat(), "demographics": feats,
               "question_id": qid, "question_text": qtext, "options": options,
               "response_distribution": [], "status": "error",
               "prompt_style": "base_completion_instruct", "model": MODEL}
        try:
            prompt = build_base_prompt(feats, qtext, options).rstrip()
            if prompt.endswith("["):            # drop the raw-completion bracket for chat
                prompt = prompt[:-1].rstrip()
            raw = call(prompt)
            parsed = extract_list(raw, len(options))
            if parsed:
                rec["response_distribution"] = parsed; rec["status"] = "success"
            else:
                rec["status"] = "failed"; rec["raw_response"] = raw[:200]
        except Exception as e:
            rec["error"] = str(e)
        with lock:
            with open(out_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec) + "\n")
            counts[rec["status"]] += 1
            n = sum(counts.values())
            if n % 2000 == 0:
                print(f"  W{wave} {n}: {counts}", flush=True)
        return rec["status"]

    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        futs = [pool.submit(work, c) for c in cells]
        for f in as_completed(futs):
            f.result()
    print(f"W{wave} DONE: {counts}", flush=True)
    return counts


if __name__ == "__main__":
    print(f"Model {MODEL} + BASE prompt -> {OUT}  waves={WAVES}", flush=True)
    tot = {"success": 0, "failed": 0, "error": 0}
    for w in WAVES:
        for k, v in run_wave(w).items():
            tot[k] += v
    print(f"\nALL DONE: {tot}", flush=True)
