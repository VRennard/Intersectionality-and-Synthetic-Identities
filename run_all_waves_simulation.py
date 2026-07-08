#!/usr/bin/env python3
"""
Run the full LLM survey simulation for every available wave.

For each wave (W26, W27, …, W92):
  1. Export data/responses/survey_responses_W{wave}.json
          +  data/demographics/survey_demographics_W{wave}.csv
     (skipped if already present)
  2. Simulate with the chosen model, writing to
     data/results/{model_tag}/W{wave}.jsonl  (resumes via per-wave checkpoint)
  3. Produce compare_demographics PDFs  -> comparisons_demographics/{model_tag}/W{wave}/
  4. Produce compare_llm_vs_human PDFs  -> llm_vs_human/{model_tag}/W{wave}/

Usage:
    # Mistral via Ollama (default)
    nohup python -u run_all_waves_simulation.py > logs/run_mistral.log 2>&1 &

    # ChatGPT
    nohup python -u run_all_waves_simulation.py \\
        --model-type openai --model gpt-4o-mini \\
        --api-key sk-... \\
        > logs/run_gpt4o-mini.log 2>&1 &
"""

import os
import sys
import json
import time
import argparse
from datetime import datetime
from itertools import combinations, product
from pathlib import Path
from typing import Dict, List

from tqdm import tqdm

# ── ensure we can import local modules ────────────────────────────────────────
sys.path.insert(0, os.path.dirname(__file__))

from llm_prompt_survey import BatchPromptSimulator, DemographicProfile, ResponseParser
from compare_demographics import compare_demographics
from compare_llm_vs_human import main as compare_llm_vs_human

# ── constants ─────────────────────────────────────────────────────────────────
HUMAN_RESP_DIR = Path("human_resp")
DATA_DIR       = Path("data")

import simulation_config as _sim_cfg
_DIM_COLS     = _sim_cfg.FEATURES       # feature label -> CSV column
_IGNORE_VALUES = _sim_cfg.IGNORE_VALUES  # feature label -> [values to skip]


# ── helpers ───────────────────────────────────────────────────────────────────

def available_waves() -> List[str]:
    """Return sorted list of wave IDs (e.g. ['W26','W27',...]) that have data."""
    waves = []
    for d in HUMAN_RESP_DIR.iterdir():
        if d.is_dir() and d.name.startswith("American_Trends_Panel_W"):
            waves.append(d.name.split("_W")[1])
    return sorted(waves)


def jsonl_path(wave: str, model_tag: str = "mistral") -> Path:
    # W50 with mistral: re-use file already moved to data/results/mistral/
    if wave == "50" and model_tag == "mistral":
        legacy = DATA_DIR / "results" / "mistral" / "W50.jsonl"
        if legacy.exists():
            return legacy
    return DATA_DIR / "results" / model_tag / f"W{wave}.jsonl"


def checkpoint_path(wave: str, model_tag: str = "mistral") -> Path:
    return DATA_DIR / "results" / model_tag / f"checkpoint_W{wave}.json"


def survey_json_path(wave: str) -> Path:
    return DATA_DIR / "responses" / f"survey_responses_W{wave}.json"


def demographics_csv_path(wave: str) -> Path:
    return DATA_DIR / "demographics" / f"survey_demographics_W{wave}.csv"


def raw_csv_path(wave: str) -> Path:
    return HUMAN_RESP_DIR / f"American_Trends_Panel_W{wave}" / "responses.csv"


# ── export ────────────────────────────────────────────────────────────────────

def _build_wave_json(wave: str, out_path: Path):
    """Build survey_responses_W{wave}.json directly from info.csv + responses.csv."""
    import ast
    import pandas as pd

    info_path = HUMAN_RESP_DIR / f"American_Trends_Panel_W{wave}" / "info.csv"
    resp_path = HUMAN_RESP_DIR / f"American_Trends_Panel_W{wave}" / "responses.csv"
    info = pd.read_csv(info_path)
    resp = pd.read_csv(resp_path, low_memory=False)

    questions_data = []
    for _, row in info.iterrows():
        key = str(row['key'])
        if key not in resp.columns:
            continue
        mapping  = ast.literal_eval(row['option_mapping'])   # {1.0: 'Text', 99.0: 'Refused'}
        # ordered: non-refused keys first (sorted), then 99 (Refused) last
        ordered_keys = sorted(k for k in mapping if k != 99.0) + \
                       ([99.0] if 99.0 in mapping else [])
        options = [mapping[k] for k in ordered_keys]

        counts = resp[key].value_counts(dropna=True)
        total  = int(counts.sum())
        responses = []
        for opt in options:
            c = int(counts.get(opt, 0))
            responses.append({
                'option': opt,
                'count': c,
                'percentage': round(c / total * 100, 2) if total else 0.0
            })

        questions_data.append({
            'wave': f'W{wave}',
            'question_id': key,
            'question_text': str(row['question']),
            'total_respondents': total,
            'missing_values': int(resp[key].isna().sum()),
            'responses': responses
        })

    with open(out_path, 'w') as f:
        json.dump(questions_data, f, indent=2)
    print(f"  [export] {out_path}  ({len(questions_data)} questions)")


def _build_wave_demographics(wave: str, out_path: Path):
    """Build survey_demographics_W{wave}.csv from responses.csv."""
    import pandas as pd

    resp_path = HUMAN_RESP_DIR / f"American_Trends_Panel_W{wave}" / "responses.csv"
    df = pd.read_csv(resp_path, low_memory=False)

    rows = []
    for demo_name, col in _DIM_COLS.items():
        if col not in df.columns:
            continue
        vc    = df[col].value_counts(dropna=True)
        total = int(vc.sum())
        for cat, cnt in vc.items():
            rows.append({
                'wave': f'W{wave}',
                'demographic': demo_name,
                'category': str(cat),
                'count': int(cnt),
                'percentage': round(cnt / total * 100, 2),
                'total_respondents': total
            })

    pd.DataFrame(rows).to_csv(out_path, index=False)
    print(f"  [export] {out_path}  ({len(rows)} rows)")


def ensure_exports(wave: str):
    """Generate survey JSON and demographics CSV if not already present."""
    jpath = survey_json_path(wave)
    dpath = demographics_csv_path(wave)
    jpath.parent.mkdir(parents=True, exist_ok=True)
    dpath.parent.mkdir(parents=True, exist_ok=True)

    if not jpath.exists():
        _build_wave_json(wave, jpath)
    else:
        print(f"  [export] {jpath} already exists, skipping.")

    if not dpath.exists():
        _build_wave_demographics(wave, dpath)
    else:
        print(f"  [export] {dpath} already exists, skipping.")


# ── demographic profiles ──────────────────────────────────────────────────────

def load_demographic_options(wave: str) -> Dict[str, List[str]]:
    import csv
    path = demographics_csv_path(wave)
    cats: Dict[str, List[str]] = {}
    with open(path, newline='') as f:
        for row in csv.DictReader(f):
            dem = row.get('demographic', '').strip()
            cat = row.get('category', '').strip()
            if dem and cat:
                cats.setdefault(dem, []).append(cat)
    # Remove ignored values
    for dem, ignored in _IGNORE_VALUES.items():
        if dem in cats:
            cats[dem] = [c for c in cats[dem] if c not in ignored]
            if not cats[dem]:
                del cats[dem]
    return cats


def build_profiles(options: Dict[str, List[str]]) -> List[DemographicProfile]:
    keys = [k for k in _sim_cfg.FEATURES if k in options]
    profiles: List[DemographicProfile] = []
    for k in keys:
        for cat in options[k]:
            profiles.append(DemographicProfile(features=[f"{k} {cat}"]))
    for a, b in combinations(keys, 2):
        for ca, cb in product(options[a], options[b]):
            profiles.append(DemographicProfile(features=[f"{a} {ca}", f"{b} {cb}"]))
    return profiles


# ── simulation ────────────────────────────────────────────────────────────────

def is_simulation_complete(wave: str, model_tag: str) -> bool:
    """True if JSONL exists and no checkpoint file (= clean finish)."""
    return jsonl_path(wave, model_tag).exists() and not checkpoint_path(wave, model_tag).exists()


def run_simulation(wave: str, model_type: str, model_name: str,
                   model_tag: str, api_key: str = None):
    """Run or resume the simulation for a single wave."""
    out  = jsonl_path(wave, model_tag)
    ckpt = checkpoint_path(wave, model_tag)
    out.parent.mkdir(parents=True, exist_ok=True)

    print(f"  [sim] Loading questions from {survey_json_path(wave)} …")
    simulator = BatchPromptSimulator(
        model_type=model_type,
        model_name=model_name,
        data_file=str(survey_json_path(wave)),
        openai_api_key=api_key
    )
    questions = simulator.data_loader.questions
    print(f"  [sim] {len(questions)} questions loaded.")

    options  = load_demographic_options(wave)
    profiles = build_profiles(options)
    print(f"  [sim] {len(profiles)} demographic profiles.")

    # Load checkpoint
    if ckpt.exists():
        with open(ckpt) as f:
            cp = json.load(f)
        start_d = cp.get("next_demo_idx", 0)
        start_q = cp.get("next_q_idx", 0)
        processed = cp.get("processed", 0)
        print(f"  [sim] Resuming from profile {start_d}, question {start_q} …")
    else:
        start_d = start_q = processed = 0

    if not out.exists():
        out.touch()

    t0 = time.time()
    profile_bar = tqdm(range(start_d, len(profiles)), desc=f"W{wave} profiles",
                       unit="profile", initial=0, total=len(profiles) - start_d)
    for d_idx in profile_bar:
        profile = profiles[d_idx]
        q_start = start_q if d_idx == start_d else 0
        profile_bar.set_postfix(profile=", ".join(profile.features))

        q_range = range(q_start, len(questions))
        for q_idx in tqdm(q_range, desc="  questions", unit="q",
                          leave=False, total=len(questions) - q_start):
            q = questions[q_idx]
            try:
                msg = simulator.prompt_builder.build_prompt(profile, q)
                resp = simulator.llm.call_model(
                    system_message=simulator.prompt_builder.SYSTEM_MESSAGE,
                    user_message=msg
                )
                parsed = ResponseParser.parse_response(resp, len(q.options))
                rec = {
                    "timestamp": datetime.now().isoformat(),
                    "demographics": profile.features,
                    "question_id": q.question_id,
                    "question_text": q.question_text,
                    "options": q.options,
                    "response_distribution": parsed or [],
                    "status": "success" if parsed else "failed"
                }
            except Exception as e:
                rec = {
                    "timestamp": datetime.now().isoformat(),
                    "demographics": profile.features,
                    "question_id": q.question_id,
                    "question_text": q.question_text,
                    "options": q.options,
                    "response_distribution": [],
                    "status": "error",
                    "error": str(e)
                }

            with open(out, 'a') as f:
                f.write(json.dumps(rec) + "\n")
            processed += 1

            # checkpoint after every question
            with open(ckpt, 'w') as f:
                json.dump({
                    "next_demo_idx": d_idx,
                    "next_q_idx": q_idx + 1,
                    "processed": processed,
                    "timestamp": datetime.now().isoformat()
                }, f)

        # profile done
        with open(ckpt, 'w') as f:
            json.dump({
                "next_demo_idx": d_idx + 1,
                "next_q_idx": 0,
                "processed": processed,
                "timestamp": datetime.now().isoformat()
            }, f)

        elapsed = time.time() - t0
        profile_bar.set_postfix(processed=processed, elapsed=f"{elapsed:.0f}s")

    # clean up checkpoint on success
    ckpt.unlink(missing_ok=True)
    elapsed = time.time() - t0
    print(f"  [sim] Wave W{wave} complete. {processed} records in {elapsed:.0f}s.")


# ── visualisation ─────────────────────────────────────────────────────────────

def run_visualisations(wave: str, model_tag: str = "mistral"):
    jpath = str(jsonl_path(wave, model_tag))
    raw   = str(raw_csv_path(wave))

    demo_out = str(Path("comparisons_demographics") / model_tag / f"W{wave}")
    print(f"  [viz] compare_demographics -> {demo_out}")
    compare_demographics(jpath, demo_out)

    if raw_csv_path(wave).exists():
        hum_out = str(Path("llm_vs_human") / model_tag / f"W{wave}")
        print(f"  [viz] compare_llm_vs_human -> {hum_out}")
        compare_llm_vs_human(jpath, raw, hum_out)
    else:
        print(f"  [viz] No raw CSV for W{wave}, skipping human comparison.")


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Run LLM survey simulation for all Pew waves.")
    parser.add_argument('--model-type', default='ollama',
                        choices=['ollama', 'openai'],
                        help='LLM backend (default: ollama)')
    parser.add_argument('--model',      default='mistral',
                        help='Model name: mistral, gpt-4o-mini, gpt-4o, etc. (default: mistral)')
    parser.add_argument('--api-key',    default=None,
                        help='OpenAI API key (or set OPENAI_API_KEY env var)')
    parser.add_argument('--waves',      default=None, nargs='+',
                        help='Subset of waves to run, e.g. --waves 26 50 82 (default: all)')
    args = parser.parse_args()

    model_type = args.model_type
    model_name = args.model
    api_key    = args.api_key or os.getenv('OPENAI_API_KEY')
    model_tag  = model_name.replace('/', '-')   # safe dirname (e.g. gpt-4o-mini)

    if model_type == 'openai' and not api_key:
        parser.error('--api-key or OPENAI_API_KEY env var required for --model-type openai')

    print("=" * 72)
    print("ALL-WAVES LLM SIMULATION + VISUALISATION")
    print(f"Model  : {model_type} / {model_name}  (tag: {model_tag})")
    print(f"Started: {datetime.now():%Y-%m-%d %H:%M:%S}")
    print("=" * 72)

    waves = args.waves if args.waves else available_waves()
    print(f"\nWaves to process ({len(waves)}): {waves}\n")

    # Put W50 first so its (already-done) simulation is skipped quickly
    ordered = (["50"] if "50" in waves else []) + [w for w in waves if w != "50"]

    summary = {}
    t_global = time.time()

    for wave in ordered:
        print(f"\n{'─'*72}")
        print(f"  WAVE W{wave}")
        print(f"{'─'*72}")

        try:
            ensure_exports(wave)
        except Exception as e:
            print(f"  [export] ERROR for W{wave}: {e}")
            summary[wave] = "export_failed"
            continue

        if is_simulation_complete(wave, model_tag):
            print(f"  [sim] JSONL already complete, skipping simulation.")
        else:
            try:
                run_simulation(wave, model_type, model_name, model_tag, api_key)
            except Exception as e:
                print(f"  [sim] ERROR for W{wave}: {e}")
                summary[wave] = "simulation_failed"
                continue

        try:
            run_visualisations(wave, model_tag)
            summary[wave] = "done"
        except Exception as e:
            print(f"  [viz] ERROR for W{wave}: {e}")
            summary[wave] = "viz_failed"

    elapsed = time.time() - t_global
    print(f"\n{'='*72}")
    print(f"ALL DONE in {elapsed/3600:.1f}h")
    print(f"{'='*72}")
    for w, st in summary.items():
        print(f"  W{w:3s}  {st}")

    out_summary = Path("logs") / f"simulation_summary_{model_tag}.json"
    out_summary.parent.mkdir(exist_ok=True)
    with open(out_summary, "w") as f:
        json.dump({"timestamp": datetime.now().isoformat(),
                   "model_type": model_type, "model_name": model_name,
                   "waves": summary, "elapsed_seconds": elapsed}, f, indent=2)
    print(f"\nSummary -> {out_summary}")


if __name__ == "__main__":
    main()
