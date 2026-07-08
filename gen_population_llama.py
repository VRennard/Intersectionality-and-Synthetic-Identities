"""Direct p_hat_pop (["Average American"]) for Llama via local Ollama.
Same prompt + parser as the runner; writes into data/results/llama3_1_8b_instruct_q4/."""
import os, sys, json, requests
from batch_api_runner import build_multi_question_prompt, parse_multi_question_response
from llm_prompt_survey import PromptBuilder, DemographicProfile

SYS = PromptBuilder().SYSTEM_MESSAGE
POP = DemographicProfile(features=["Average American"])
QBATCH = 1
MODEL = "llama3.1:8b-instruct-q4_K_M"
URL = "http://localhost:11434/api/chat"
DIR = "data/results/llama3_1_8b_instruct_q4"

class Q:
    def __init__(s, qid, text, opts): s.question_id, s.question_text, s.options = qid, text, opts

def load_q(wave):
    d = json.load(open(f"data/responses/survey_responses_W{wave}.json"))
    return [Q(q["question_id"], q["question_text"], [r["option"] for r in q["responses"]]) for q in d]

def done_qs(wave):
    f = f"{DIR}/W{wave}.jsonl"; s = set()
    if os.path.exists(f):
        for l in open(f):
            try:
                r = json.loads(l)
                if r.get("demographics") == ["Average American"] and r.get("status") == "success":
                    s.add(r["question_id"])
            except: pass
    return s

def call(user):
    r = requests.post(URL, json={"model": MODEL, "stream": False,
        "messages": [{"role": "system", "content": SYS}, {"role": "user", "content": user}],
        "options": {"temperature": 0.7, "num_ctx": 8192}}, timeout=180)
    return r.json()["message"]["content"]

def waves():
    import re
    return sorted((fn[1:-6] for fn in os.listdir(DIR)
        if fn.startswith("W") and fn.endswith(".jsonl") and fn[1:-6].isdigit()), key=int)

for wave in waves():
    qs = load_q(wave); done = done_qs(wave)
    todo = [q for q in qs if q.question_id not in done]
    if not todo:
        print(f"llama W{wave}: pop done", flush=True); continue
    with open(f"{DIR}/W{wave}.jsonl", "a") as out:
        ok = 0
        for i in range(0, len(todo), QBATCH):
            batch = todo[i:i+QBATCH]
            try:
                dists = parse_multi_question_response(call(build_multi_question_prompt(POP, batch)), batch)
            except Exception as e:
                print(f"llama W{wave} b{i}: ERR {e}", flush=True); continue
            for q, dist in zip(batch, dists):
                out.write(json.dumps({"demographics":["Average American"],"question_id":q.question_id,
                    "question_text":q.question_text,"options":q.options,
                    "response_distribution":dist if dist else [],
                    "status":"success" if dist else "parse_fail"})+"\n")
                ok += 1 if dist else 0
            out.flush()
        print(f"llama W{wave}: pop {ok}/{len(todo)} ok", flush=True)
print("LLAMA POP DONE", flush=True)
