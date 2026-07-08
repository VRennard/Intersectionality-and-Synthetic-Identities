import json, os

result_dir = 'data/results/gpt-4o-mini'
response_dir = 'data/responses'

all_waves = sorted([f.replace('survey_responses_W','').replace('.json','')
                    for f in os.listdir(response_dir) if f.endswith('.json')])

print(f'{"Wave":<6} | Status')
print(f'{"------":<6}|-------')
for w in all_waves:
    path = f'{result_dir}/W{w}.jsonl'
    if not os.path.exists(path):
        print(f'W{w:<5} | NOT STARTED')
    else:
        total, errors = 0, 0
        with open(path) as f:
            for line in f:
                r = json.loads(line)
                total += 1
                if r.get('status') != 'success' or not r.get('response_distribution'):
                    errors += 1
        status = 'CLEAN' if errors == 0 else f'{errors} ERRORS'
        print(f'W{w:<5} | {total} records, {status}')
