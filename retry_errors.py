#!/usr/bin/env python3
"""
Retry only the 'Connection error' records in a JSONL results file.
Patches them in-place with fresh API calls.

Usage:
    python retry_errors.py --wave 45 --api-key sk-...
"""
import os, sys, json, argparse
from datetime import datetime
from pathlib import Path
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(__file__))
from llm_prompt_survey import DemographicProfile, SurveyQuestion, PromptBuilder, LLMInterface, ResponseParser

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--wave', required=True)
    parser.add_argument('--api-key', default=None)
    args = parser.parse_args()

    api_key = args.api_key or os.getenv('OPENAI_API_KEY')
    jsonl = Path(f'data/results/gpt-4o-mini/W{args.wave}.jsonl')

    llm = LLMInterface(model_type='openai', model_name='gpt-4o-mini', openai_api_key=api_key)

    # Load all records
    records = []
    with open(jsonl) as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))

    error_indices = [i for i, r in enumerate(records) if r.get('status') == 'error']
    print(f'W{args.wave}: {len(error_indices)} error records to retry out of {len(records)} total')

    for i in tqdm(error_indices, desc=f'W{args.wave}'):
        r = records[i]
        profile = DemographicProfile(features=r['demographics'])
        question = SurveyQuestion(
            question_id=r['question_id'],
            question_text=r['question_text'],
            options=r['options']
        )
        try:
            msg = PromptBuilder.build_prompt(profile, question)
            resp = llm.call_model(system_message=PromptBuilder.SYSTEM_MESSAGE, user_message=msg)
            parsed = ResponseParser.parse_response(resp, len(question.options))
            records[i] = {
                'timestamp': datetime.now().isoformat(),
                'demographics': r['demographics'],
                'question_id': r['question_id'],
                'question_text': r['question_text'],
                'options': r['options'],
                'response_distribution': parsed or [],
                'status': 'success' if parsed else 'failed',
            }
        except Exception as e:
            records[i]['error'] = str(e)
            records[i]['timestamp'] = datetime.now().isoformat()

    # Write back
    with open(jsonl, 'w') as f:
        for r in records:
            f.write(json.dumps(r) + '\n')

    success = sum(1 for r in records if r.get('response_distribution'))
    print(f'Done. {success}/{len(records)} records now have valid distributions.')

if __name__ == '__main__':
    main()
