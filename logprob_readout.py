#!/usr/bin/env python3
"""
Log-probability elicitation (Santurkar / OpinionQA style) on local models via
Ollama — the third elicitation paradigm, alongside aggregate
(batch_api_runner / run_ollama_fill) and individual sampling
(indiv_sampling_*).

The "distribution over respondents" is read directly from the model's
next-token distribution: present the question with the answer options labelled
A, B, C, … ; do ONE forward pass; read the logprob the model assigns to each
option *letter* at the first generated position; softmax over the option
letters -> the simulated distribution. No sampling, one token generated per
call, so this is far cheaper than individual sampling.

Requires an Ollama that returns logprobs (top-level `logprobs` with
`top_logprobs` per position; verified on this pod). Output is byte-compatible
with the aggregate pipeline, so the collapse contest runs unchanged:

    cd verification && python verify_spine_collapse.py <tag>

Scope: depth 1 + 2 (singles + pairs) over the 6 protagonist dimensions.

Example
-------
  python logprob_readout.py --model gemma2:9b --tag gemma2_9b_logprob \\
      --workers 16 --cells-dir indiv_cells

Methods caveats (state in paper):
  * Options are letter-mapped (A,B,…) so each maps to one token; this is the
    standard OpinionQA readout. Option order matches the survey JSON.
  * An option letter absent from the returned top_logprobs is assigned
    probability ~0 (its mass is below the top-k cutoff).
"""

import argparse
import json
import math
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

import requests

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from indiv_sampling_local import load_questions, build_cells, ALL_WAVES

LETTERS = [chr(65 + i) for i in range(26)]   # A..Z

SYSTEM = (
    "You are a single survey respondent. You answer as the specific person "
    "described, choosing the one option that best reflects your own view."
)


def build_logprob_prompt(features, question_text, options):
    demo = "\n".join(f"    {f}" for f in features)
    opts = "\n".join(f"{LETTERS[i]}. {o}" for i, o in enumerate(options))
    return (f"You are the following person:\n\n{demo}\n\n"
            f"Question: {question_text}\n\nOptions:\n{opts}\n\n"
            f"Answer with the single letter of the option you choose, "
            f"and nothing else.")


# ── ollama logprob client ───────────────────────────────────────────────────────

class OllamaLogprob:
    def __init__(self, url, model, top_logprobs, retries, num_ctx=2048):
        self.url = url.rstrip("/")
        self.model = model
        self.top_logprobs = top_logprobs
        self.retries = retries
        self.num_ctx = num_ctx
        self._local = threading.local()

    def _session(self):
        if not hasattr(self._local, "s"):
            self._local.s = requests.Session()
        return self._local.s

    def first_token_logprobs(self, user_msg):
        """Return {token_text: logprob} for the first generated position."""
        payload = {
            "model": self.model, "stream": False,
            "logprobs": True, "top_logprobs": self.top_logprobs,
            "messages": [{"role": "system", "content": SYSTEM},
                         {"role": "user", "content": user_msg}],
            # num_ctx capped: the prompt is one question (~300 tokens). Without
            # this, Ollama reserves KV cache for the model's DEFAULT context
            # (mistral 32K, llama3.1 128K) x num_parallel, overflowing VRAM and
            # offloading to CPU (5x slowdown). 2048 is ample and keeps it on GPU.
            "options": {"num_predict": 1, "temperature": 0,
                        "num_ctx": self.num_ctx},
        }
        last_err = None
        for attempt in range(self.retries):
            try:
                r = self._session().post(self.url + "/api/chat", json=payload,
                                         timeout=120)
                r.raise_for_status()
                d = r.json()
                lp = d.get("logprobs")
                if not lp:
                    return {}
                out = {}
                for e in lp[0].get("top_logprobs", []):
                    tok = e["token"].strip()
                    if tok and tok not in out:   # keep highest (first) per token
                        out[tok] = e["logprob"]
                return out
            except (requests.exceptions.RequestException, KeyError,
                    IndexError, json.JSONDecodeError) as e:
                last_err = e
                time.sleep(min(2 ** attempt, 30))
        raise RuntimeError(f"ollama logprob call failed after {self.retries}: {last_err}")


def softmax_over_options(tok_lp, n_options):
    """Distribution over the n option letters from a {token: logprob} map.
    Missing letters get prob ~0. Returns integer counts summing to 1000."""
    lps = [tok_lp.get(LETTERS[i]) for i in range(n_options)]
    present = [x for x in lps if x is not None]
    if not present:
        return None
    m = max(present)
    exps = [math.exp(x - m) if x is not None else 0.0 for x in lps]
    s = sum(exps)
    if s == 0:
        return None
    probs = [e / s for e in exps]
    counts = [int(round(p * 1000)) for p in probs]
    counts[counts.index(max(counts))] += 1000 - sum(counts)   # fix rounding
    return counts


# ── per-wave driver ──────────────────────────────────────────────────────────────

def load_done(out_path):
    done = set()
    if not out_path.exists():
        return done
    with open(out_path, encoding="utf-8") as f:
        for line in f:
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if r.get("status") == "success":
                done.add((tuple(sorted(r["demographics"])), r["question_id"]))
    return done


def run_wave(wave, args, client):
    questions = load_questions(args.survey_dir, wave)
    cells = build_cells(wave, human_dir=args.human_dir, cells_dir=args.cells_dir)
    out_path = Path(args.out_dir) / args.tag / f"W{wave}.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    done = load_done(out_path)

    tasks = []
    for cell in cells:
        ckey = tuple(sorted(cell.features))
        for q in questions:
            if (ckey, q.question_id) not in done:
                tasks.append((cell, q))
    print(f"  W{wave}: {len(cells)} cells x {len(questions)} q; "
          f"{len(tasks)} to run ({len(done)} done)")
    if not tasks:
        return 0, 0

    write_lock = threading.Lock()
    ok = bad = 0

    def work(task):
        cell, q = task
        rec = {"timestamp": datetime.now().isoformat(),
               "demographics": cell.features,
               "question_id": q.question_id,
               "question_text": q.question_text,
               "options": q.options,
               "response_distribution": [],
               "status": "error",
               "elicitation": "logprob_readout"}
        try:
            tok_lp = client.first_token_logprobs(
                build_logprob_prompt(cell.features, q.question_text, q.options))
            counts = softmax_over_options(tok_lp, len(q.options))
            if counts:
                rec["response_distribution"] = counts
                rec["status"] = "success"
            else:
                rec["status"] = "failed"
        except Exception as e:
            rec["error"] = str(e)
        with write_lock:
            with open(out_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec) + "\n")
        return rec["status"] == "success"

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(work, t) for t in tasks]
        it = as_completed(futures)
        if tqdm:
            it = tqdm(it, total=len(futures), desc=f"W{wave}", unit="cell-q")
        for fut in it:
            if fut.result():
                ok += 1
            else:
                bad += 1
    print(f"  W{wave} done: {ok} success, {bad} failed")
    return ok, bad


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, help="ollama model, e.g. gemma2:9b")
    ap.add_argument("--tag", required=True, help="output dir, e.g. gemma2_9b_logprob")
    ap.add_argument("--waves", nargs="+", default=None, help="default: all 15")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--url", default="http://localhost:11434")
    ap.add_argument("--top-logprobs", type=int, default=20)
    ap.add_argument("--num-ctx", type=int, default=2048,
                    help="context cap; keep small so KV cache fits GPU "
                         "(prompt is one question)")
    ap.add_argument("--retries", type=int, default=6)
    ap.add_argument("--survey-dir", default="data/responses")
    ap.add_argument("--out-dir", default="data/results")
    ap.add_argument("--human-dir", default="human_resp")
    ap.add_argument("--cells-dir", default=None)
    args = ap.parse_args()
    os.chdir(Path(__file__).parent)

    waves = args.waves or ALL_WAVES
    client = OllamaLogprob(args.url, args.model, args.top_logprobs, args.retries,
                           num_ctx=args.num_ctx)
    print(f"Model {args.model} (logprob readout) -> {args.out_dir}/{args.tag}/  "
          f"workers={args.workers}")
    totals = [0, 0]
    for wave in waves:
        ok, bad = run_wave(wave, args, client)
        totals[0] += ok
        totals[1] += bad
    print(f"\nALL DONE: {totals[0]} success, {totals[1]} failed")
    print(f"  cd verification && python verify_spine_collapse.py {args.tag}")


if __name__ == "__main__":
    main()
