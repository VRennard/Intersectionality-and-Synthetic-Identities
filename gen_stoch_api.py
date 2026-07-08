"""#4: repeated W26 single-cell runs for the API models (per-model noise)."""
import os, sys, json
from openai import OpenAI
from anthropic import Anthropic
from batch_api_runner import build_multi_question_prompt, parse_multi_question_response, request_body
from llm_prompt_survey import PromptBuilder, DemographicProfile
SYS = PromptBuilder().SYSTEM_MESSAGE; NRUNS = 4
SINGLES = [["Age 30-49"],["Gender Female"],["Race Black"],["Income $30-50k"],
           ["Political Party Democrat"],["Religion Catholic"],["Race White"],["Gender Male"]]
class Q:
    def __init__(s,i,t,o): s.question_id,s.question_text,s.options=i,t,o
def load_q(w):
    d=json.load(open(f"data/responses/survey_responses_W{w}.json"))
    return [Q(q["question_id"],q["question_text"],[r["option"] for r in q["responses"]]) for q in d]
def gen(tag, api_model, provider, client):
    qs=load_q("26")
    for run in range(1,NRUNS+1):
        od=f"data/results/stoch_{tag}"; os.makedirs(od,exist_ok=True); path=f"{od}/W26_run{run}.jsonl"
        if os.path.exists(path) and sum(1 for _ in open(path))>=len(SINGLES)*len(qs)*0.9:
            print(f"{tag} run{run}: done",flush=True); continue
        with open(path,"w") as out:
            ok=0
            for feats in SINGLES:
                prof=DemographicProfile(features=feats)
                for q in qs:
                    try:
                        u=build_multi_question_prompt(prof,[q])
                        if provider=="openai":
                            txt=client.chat.completions.create(**request_body(api_model,SYS,u,1)).choices[0].message.content
                        else:
                            txt=client.messages.create(model=api_model,max_tokens=120,system=SYS,messages=[{"role":"user","content":u}]).content[0].text
                        dist=parse_multi_question_response(txt,[q])[0]
                    except Exception: dist=None
                    out.write(json.dumps({"demographics":feats,"question_id":q.question_id,
                        "response_distribution":dist if dist else [],"status":"success" if dist else "fail"})+"\n")
                    ok+=1 if dist else 0
                out.flush()
            print(f"{tag} run{run}: {ok} ok",flush=True)
oai=OpenAI(api_key=os.environ["OPENAI_API_KEY"]); ant=Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"].strip())
gen("claude-haiku-4-5-20251001","claude-haiku-4-5-20251001","anthropic",ant)
gen("gpt-4o","gpt-4o","openai",oai)
print("API STOCH DONE",flush=True)
