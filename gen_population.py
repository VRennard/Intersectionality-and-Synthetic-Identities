"""Generate the direct p_hat_pop (["Average American"]) cell for the models that
lack it, using the EXACT runner prompt + parser. Synchronous, idempotent."""
import os, sys, json, time
from openai import OpenAI
from anthropic import Anthropic
from batch_api_runner import build_multi_question_prompt, parse_multi_question_response, request_body
from llm_prompt_survey import PromptBuilder, DemographicProfile

SYS = PromptBuilder().SYSTEM_MESSAGE
POP = DemographicProfile(features=["Average American"])
QBATCH = 20

class Q:
    def __init__(s, qid, text, opts): s.question_id, s.question_text, s.options = qid, text, opts

def load_q(wave):
    d = json.load(open(f"data/responses/survey_responses_W{wave}.json"))
    return [Q(q["question_id"], q["question_text"], [r["option"] for r in q["responses"]]) for q in d]

def done_qs(d, wave):
    f = f"data/results/{d}/W{wave}.jsonl"
    s = set()
    if os.path.exists(f):
        for l in open(f):
            try:
                r = json.loads(l)
                if r.get("demographics") == ["Average American"] and r.get("status") == "success":
                    s.add(r["question_id"])
            except: pass
    return s

def gen(model_dir, api_model, provider, client, waves):
    for wave in waves:
        qs = load_q(wave); done = done_qs(model_dir, wave)
        todo = [q for q in qs if q.question_id not in done]
        if not todo:
            print(f"{model_dir} W{wave}: pop already done", flush=True); continue
        with open(f"data/results/{model_dir}/W{wave}.jsonl", "a") as out:
            ok = 0
            for i in range(0, len(todo), QBATCH):
                batch = todo[i:i+QBATCH]
                user = build_multi_question_prompt(POP, batch)
                try:
                    if provider == "openai":
                        body = request_body(api_model, SYS, user, len(batch))
                        txt = client.chat.completions.create(**body).choices[0].message.content
                    else:
                        txt = client.messages.create(model=api_model, max_tokens=len(batch)*40+200,
                            system=SYS, messages=[{"role":"user","content":user}]).content[0].text
                    dists = parse_multi_question_response(txt, batch)
                except Exception as e:
                    print(f"{model_dir} W{wave} b{i}: ERR {e}", flush=True); time.sleep(3); continue
                for q, dist in zip(batch, dists):
                    out.write(json.dumps({"demographics":["Average American"],"question_id":q.question_id,
                        "question_text":q.question_text,"options":q.options,
                        "response_distribution":dist if dist else [],
                        "status":"success" if dist else "parse_fail"})+"\n")
                    ok += 1 if dist else 0
                out.flush()
            print(f"{model_dir} W{wave}: pop {ok}/{len(todo)} ok", flush=True)

def waves_of(d):
    import re
    return sorted((fn[1:-6] for fn in os.listdir(f"data/results/{d}")
        if fn.startswith("W") and fn.endswith(".jsonl") and fn[1:-6].isdigit()), key=int)

if __name__ == "__main__":
    oai = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    ant = Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"].strip())
    targets = sys.argv[1:] or ["gpt-5.5", "gpt-4o", "haiku"]
    if "gpt-5.5" in targets:  gen("gpt-5.5-2026-04-23", "gpt-5.5", "openai", oai, ["26","34"])
    if "gpt-4o" in targets:   gen("gpt-4o", "gpt-4o", "openai", oai, waves_of("gpt-4o"))
    if "haiku" in targets:    gen("claude-haiku-4-5-20251001", "claude-haiku-4-5-20251001", "anthropic", ant, waves_of("claude-haiku-4-5-20251001"))
    print("DONE", flush=True)
