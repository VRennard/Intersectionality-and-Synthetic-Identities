#!/usr/bin/env python3
"""
Individual-sampling elicitation on LOCAL (open-weights) instruct models via
Ollama — the open-weights counterpart of indiv_sampling_runner.py (which does
the OpenAI API models).

Same paradigm: one call = one simulated persona answering the ENTIRE wave,
choosing exactly one option per question; N personas per (profile) cell; the
per-(cell, question) distribution is the histogram of chosen options across the
N personas. Output is byte-compatible with the aggregate pipeline, so the
existing collapse contest runs unchanged:

    cd verification && python verify_spine_collapse.py <tag>

Scope: depth 1 + 2 (singles + pairs) over the 6 protagonist dimensions — the
cells the collapse contest needs. (Depth 3 is a separate, heavier run.)

Two stages, run automatically per wave (or separately):
  1. GENERATE: workers call Ollama /api/chat and append one raw line per persona
     to data/results/<tag>/W{wave}.personas.jsonl  (resumable: done personas are
     skipped on restart).
  2. AGGREGATE: histogram the raw picks into the canonical
     data/results/<tag>/W{wave}.jsonl.

Examples
--------
  # gemma2:9b, all 15 waves, 120 personas/cell, 8 workers
  python indiv_sampling_local.py --model gemma2:9b --tag gemma2_9b_indiv \\
      --personas 120 --workers 8

  # mistral / llama
  python indiv_sampling_local.py --model mistral:7b-instruct --tag mistral_7b_indiv ...
  python indiv_sampling_local.py --model llama3.1:8b-instruct-q4_K_M --tag llama3_1_8b_indiv ...

  # just (re)build the canonical JSONLs from already-generated personas
  python indiv_sampling_local.py --tag gemma2_9b_indiv --aggregate-only

Pod notes
---------
  * Needs only: this file, indiv_sampling_runner.py, llm_prompt_survey.py,
    simulation_config.py, the survey JSONs (--survey-dir) and either human_resp
    CSVs (--human-dir, default) or a pre-baked cells dir (--cells-dir). If the
    pod has no human_resp, run `--dump-cells` locally first and ship the
    indiv_cells/ dir, then pass --cells-dir indiv_cells.
  * Set OLLAMA_NUM_PARALLEL on the server to match --workers.
  * num_ctx is auto-sized per wave to fit the whole-wave prompt — do NOT let it
    fall back to a small default or prompts get silently truncated.
"""

import argparse
import json
import os
import random
import sys
import threading
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from itertools import combinations, product
from pathlib import Path

import requests

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from indiv_sampling_runner import (
    build_persona_prompt, parse_persona_response, INDIV_SYSTEM, CORE_DIMS,
)
from llm_prompt_survey import DemographicProfile, SurveyQuestion
import simulation_config as cfg

ALL_WAVES = ["26", "27", "29", "32", "34", "36", "41", "42", "43",
             "45", "49", "50", "54", "82", "92"]


# ── data loading ────────────────────────────────────────────────────────────────

def load_questions(survey_dir, wave):
    path = Path(survey_dir) / f"survey_responses_W{wave}.json"
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)
    return [
        SurveyQuestion(
            question_id=q["question_id"],
            question_text=q["question_text"],
            options=[r["option"] for r in q["responses"]],
            wave=q.get("wave", f"W{wave}"),
        )
        for q in raw
    ]


def _feature_vals(human_dir, wave, feature):
    import pandas as pd
    path = Path(human_dir) / f"American_Trends_Panel_W{wave}" / "responses.csv"
    col = cfg.FEATURES.get(feature)
    if not col or not path.exists():
        return []
    df = pd.read_csv(path, low_memory=False)
    if col not in df.columns:
        return []
    ignored = cfg.IGNORE_VALUES.get(feature, [])
    return [str(v) for v in df[col].dropna().unique() if str(v) not in ignored]


def build_cells(wave, human_dir=None, cells_dir=None, dims=CORE_DIMS,
                pair_frac=None, seed=0):
    """Singles + pairs over `dims`. Either enumerated from human_resp CSVs
    (human_dir) or read from a pre-baked cells file (cells_dir/W{wave}.json,
    a list of feature-lists).

    pair_frac: if set (e.g. 0.1), keep ALL singles but only a seeded random
    `pair_frac` of the pairs. All singles are kept so every sampled pair stays
    scorable in the collapse contest (it needs its two component singles). The
    seed is offset by wave so the sample differs per wave but is identical
    across models/runs."""
    if cells_dir:
        with open(Path(cells_dir) / f"W{wave}.json") as f:
            return [DemographicProfile(features=feats) for feats in json.load(f)]
    cats = {}
    for feature in dims:
        vals = _feature_vals(human_dir, wave, feature)
        if vals:
            cats[feature] = vals
    keys = [k for k in dims if k in cats]
    singles, pairs = [], []
    for k in keys:
        for cat in cats[k]:
            singles.append(DemographicProfile(features=[f"{k} {cat}"]))
    for a, b in combinations(keys, 2):
        for ca, cb in product(cats[a], cats[b]):
            pairs.append(DemographicProfile(features=[f"{a} {ca}", f"{b} {cb}"]))
    if pair_frac is not None:
        import random
        rng = random.Random(seed + int(wave))
        n = max(1, round(len(pairs) * pair_frac))
        pairs = sorted(rng.sample(pairs, n),
                       key=lambda p: tuple(sorted(p.features)))
    return singles + pairs


# ── ollama (chat, individual-sampling system prompt, auto num_ctx) ───────────────

class OllamaClient:
    def __init__(self, url, model, temperature, num_predict, num_ctx, retries):
        self.url = url.rstrip("/")
        self.model = model
        self.options = {"temperature": temperature, "top_p": 0.9, "top_k": 40,
                        "num_predict": num_predict, "num_ctx": num_ctx}
        self.retries = retries
        self._local = threading.local()

    def _session(self):
        if not hasattr(self._local, "s"):
            self._local.s = requests.Session()
        return self._local.s

    def check(self):
        r = self._session().get(f"{self.url}/api/tags", timeout=10)
        r.raise_for_status()
        names = [m["name"] for m in r.json().get("models", [])]
        if not any(n == self.model or n.split(":")[0] == self.model.split(":")[0]
                   for n in names):
            print(f"WARNING: model '{self.model}' not in `ollama list` ({names}); "
                  f"ollama may pull it on first call.")

    def chat(self, user_msg):
        payload = {"model": self.model, "stream": False,
                   "messages": [
                       {"role": "system", "content": INDIV_SYSTEM},
                       {"role": "user", "content": user_msg}],
                   "options": self.options}
        last_err = None
        for attempt in range(self.retries):
            try:
                r = self._session().post(self.url + "/api/chat", json=payload,
                                         timeout=600)
                r.raise_for_status()
                return r.json()["message"]["content"]
            except (requests.exceptions.RequestException, KeyError,
                    json.JSONDecodeError) as e:
                last_err = e
                time.sleep(min(2 ** attempt, 30))
        raise RuntimeError(f"ollama call failed after {self.retries} attempts: {last_err}")


# ── generate ────────────────────────────────────────────────────────────────────

def _done_personas(personas_path):
    """Set of (sorted-features-tuple, persona_idx) already generated."""
    done = set()
    if not personas_path.exists():
        return done
    with open(personas_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
                done.add((tuple(sorted(r["demographics"])), r["persona"]))
            except Exception:
                pass
    return done


def auto_num_ctx(prompt, num_predict):
    """Round est. (prompt + output) tokens up to a 1024 multiple, min 4096."""
    est = len(prompt) // 4 + num_predict + 256
    ctx = max(4096, ((est + 1023) // 1024) * 1024)
    return ctx


def generate_wave(wave, args, questions, cells):
    personas_path = Path(args.out_dir) / args.tag / f"W{wave}.personas.jsonl"
    personas_path.parent.mkdir(parents=True, exist_ok=True)
    done = _done_personas(personas_path)

    # q_frac: each persona answers a seeded random subset of questions.
    # The subset varies per persona (seed = wave*10^6 + cell_hash + persona_idx)
    # so question coverage is spread across the full set at the cell level.
    q_frac = getattr(args, "q_frac", None)
    n_q = max(1, round(len(questions) * q_frac)) if q_frac else len(questions)

    # size prompt/context on the FULL question list (worst case for ctx),
    # but only on a sample-sized list for num_predict
    sample_prompt = build_persona_prompt(cells[-1], questions[:n_q])
    num_predict = n_q * 6 + 100
    num_ctx = args.num_ctx or auto_num_ctx(sample_prompt, num_predict)

    tasks = []
    for cell in cells:
        ckey = tuple(sorted(cell.features))
        for p in range(args.personas):
            if (ckey, p) not in done:
                tasks.append((cell, p))
    print(f"  W{wave}: {len(cells)} cells x {args.personas} personas; "
          f"{len(tasks)} to generate ({len(done)} done) | "
          f"q_per_persona={n_q}/{len(questions)} "
          f"num_predict={num_predict} num_ctx={num_ctx}")
    if not tasks:
        return 0, 0

    client = OllamaClient(args.url, args.model, args.temperature,
                          num_predict, num_ctx, args.retries)
    write_lock = threading.Lock()
    ok = bad = 0

    def work(task):
        cell, p_idx = task
        if q_frac:
            # seeded per (wave, cell, persona) — reproducible, varies per persona
            rng = random.Random(int(wave) * 10**6
                                + hash(tuple(sorted(cell.features))) % 10**5
                                + p_idx)
            q_subset = sorted(rng.sample(range(len(questions)), n_q))
            q_for_call = [questions[i] for i in q_subset]
        else:
            q_subset = list(range(len(questions)))
            q_for_call = questions
        prompt = build_persona_prompt(cell, q_for_call)
        answers = {}
        try:
            text = client.chat(prompt)
            picks = parse_persona_response(text, len(q_for_call))
            for local_i, pick in enumerate(picks):
                qi = q_subset[local_i]
                if pick is not None and 1 <= pick <= len(questions[qi].options):
                    answers[questions[qi].question_id] = pick
        except Exception:
            pass
        rec = {"demographics": cell.features, "persona": p_idx, "answers": answers}
        with write_lock:
            with open(personas_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec) + "\n")
        return len(answers) > 0

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(work, t) for t in tasks]
        it = as_completed(futures)
        if tqdm:
            it = tqdm(it, total=len(futures), desc=f"W{wave}", unit="persona")
        for fut in it:
            if fut.result():
                ok += 1
            else:
                bad += 1
    print(f"  W{wave} generated: {ok} valid, {bad} empty/failed")
    return ok, bad


# ── aggregate ───────────────────────────────────────────────────────────────────

def aggregate_wave(wave, args, questions):
    import numpy as np
    qmeta = {q.question_id: (q.question_text, q.options) for q in questions}
    personas_path = Path(args.out_dir) / args.tag / f"W{wave}.personas.jsonl"
    out_path = Path(args.out_dir) / args.tag / f"W{wave}.jsonl"
    if not personas_path.exists():
        print(f"  W{wave}: no personas file, skipping aggregation")
        return 0

    hist = defaultdict(dict)     # ckey -> {qid: np.array(counts)}
    feats_of = {}                # ckey -> original feature list (for output)
    n_personas = 0
    with open(personas_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            ckey = tuple(sorted(r["demographics"]))
            feats_of.setdefault(ckey, r["demographics"])
            n_personas += 1
            for qid, opt in r["answers"].items():
                if qid not in qmeta:
                    continue
                nopt = len(qmeta[qid][1])
                if not (1 <= opt <= nopt):
                    continue
                arr = hist[ckey].setdefault(qid, np.zeros(nopt, dtype=int))
                arr[opt - 1] += 1

    written = 0
    tmp = out_path.with_suffix(".jsonl.tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        for ckey, qd in hist.items():
            for qid, counts in qd.items():
                if counts.sum() == 0:
                    continue
                text, options = qmeta[qid]
                f.write(json.dumps({
                    "demographics": feats_of[ckey],
                    "question_id": qid,
                    "question_text": text,
                    "options": options,
                    "response_distribution": [int(x) for x in counts],
                    "status": "success",
                    "elicitation": "individual_sampling_local",
                    "n_personas": int(counts.sum()),
                }) + "\n")
                written += 1
    tmp.replace(out_path)
    print(f"  W{wave}: aggregated {n_personas:,} personas -> {written} rows ({out_path})")
    return written


# ── dump-cells helper (run locally where human_resp lives) ───────────────────────

def cmd_dump_cells(args):
    waves = args.waves or ALL_WAVES
    outdir = Path(args.cells_dir or "indiv_cells")
    outdir.mkdir(parents=True, exist_ok=True)
    for wave in waves:
        cells = build_cells(wave, human_dir=args.human_dir,
                            pair_frac=args.pair_frac, seed=args.seed)
        ns = sum(len(c.features) == 1 for c in cells)
        with open(outdir / f"W{wave}.json", "w") as f:
            json.dump([c.features for c in cells], f)
        print(f"  W{wave}: {len(cells)} cells ({ns} singles, {len(cells)-ns} pairs) "
              f"-> {outdir}/W{wave}.json")


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", help="ollama model tag, e.g. gemma2:9b")
    ap.add_argument("--tag", help="output dir under --out-dir, e.g. gemma2_9b_indiv")
    ap.add_argument("--personas", type=int, default=120,
                    help="personas per cell (win-rate robust in 100-200)")
    ap.add_argument("--waves", nargs="+", default=None,
                    help="subset of waves, e.g. --waves 26 34 (default: all 15)")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--url", default="http://localhost:11434")
    ap.add_argument("--temperature", type=float, default=1.0,
                    help="1.0 = standard for respondent diversity")
    ap.add_argument("--num-ctx", type=int, default=None,
                    help="override auto context size (must fit whole-wave prompt)")
    ap.add_argument("--retries", type=int, default=6)
    ap.add_argument("--survey-dir", default="data/responses")
    ap.add_argument("--out-dir", default="data/results")
    ap.add_argument("--human-dir", default="human_resp",
                    help="dir with American_Trends_Panel_W*/responses.csv (cell source)")
    ap.add_argument("--cells-dir", default=None,
                    help="pre-baked cells dir (W{wave}.json); skips needing human_resp")
    ap.add_argument("--aggregate-only", action="store_true",
                    help="rebuild canonical JSONLs from existing personas files")
    ap.add_argument("--dump-cells", action="store_true",
                    help="write cells/W{wave}.json from human_resp and exit "
                         "(run locally, ship to a pod that lacks human_resp)")
    ap.add_argument("--pair-frac", type=float, default=None,
                    help="with --dump-cells: keep all singles + this fraction of "
                         "pairs (seeded per wave), e.g. 0.1 for a 10%% sample")
    ap.add_argument("--q-frac", type=float, default=None,
                    help="fraction of questions each persona answers (e.g. 0.1); "
                         "subset is seeded per (wave, cell, persona) so coverage "
                         "spreads across all questions at the cell level")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    os.chdir(Path(__file__).parent)

    if args.dump_cells:
        cmd_dump_cells(args)
        return

    if not args.tag:
        ap.error("--tag is required")
    waves = args.waves or ALL_WAVES

    if args.aggregate_only:
        total = 0
        for wave in waves:
            questions = load_questions(args.survey_dir, wave)
            total += aggregate_wave(wave, args, questions)
        print(f"\nAGGREGATE DONE: {total} rows across {len(waves)} waves")
        return

    if not args.model:
        ap.error("--model is required for generation")
    print(f"Model {args.model} (individual sampling) -> {args.out_dir}/{args.tag}/  "
          f"personas={args.personas} workers={args.workers} temp={args.temperature}")
    OllamaClient(args.url, args.model, args.temperature, 1, 4096, args.retries).check()

    for wave in waves:
        questions = load_questions(args.survey_dir, wave)
        cells = build_cells(wave, human_dir=args.human_dir, cells_dir=args.cells_dir)
        generate_wave(wave, args, questions, cells)
        aggregate_wave(wave, args, questions)
    print("\nALL DONE. Run the contest per tag:\n"
          f"  cd verification && python verify_spine_collapse.py {args.tag}")


if __name__ == "__main__":
    main()
