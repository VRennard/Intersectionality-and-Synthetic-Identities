"""#4: repeated runs of a sample of W26 single cells to measure each model's
run-to-run noise (eta) -> per-model calibration endpoints. Ollama (open-weights)."""
import os, sys, json, requests, random
from batch_api_runner import build_multi_question_prompt, parse_multi_question_response
from llm_prompt_survey import PromptBuilder, DemographicProfile
SYS = PromptBuilder().SYSTEM_MESSAGE
URL = "http://localhost:11434/api/chat"
NRUNS = 4
random.seed(0)

class Q:
    def __init__(s, qid, t, o): s.question_id, s.question_text, s.options = qid, t, o
def load_q(w):
    d = json.load(open(f"data/responses/survey_responses_W{w}.json"))
    return [Q(q["question_id"], q["question_text"], [r["option"] for r in q["responses"]]) for q in d]

# single profiles: one value per dimension (covers all option-counts via the 78 questions)
SINGLES = [["Age 30-49"],["Gender Female"],["Race Black"],["Income $30-50k"],
           ["Political Party Democrat"],["Religion Catholic"],["Race White"],["Gender Male"]]

def gen(model_tag, ollama_model):
    qs = load_q("26")
    for run in range(1, NRUNS+1):
        outdir = f"data/results/stoch_{model_tag}"; os.makedirs(outdir, exist_ok=True)
        path = f"{outdir}/W26_run{run}.jsonl"
        if os.path.exists(path) and sum(1 for _ in open(path)) >= len(SINGLES)*len(qs)*0.9:
            print(f"{model_tag} run{run}: done", flush=True); continue
        with open(path, "w") as out:
            ok = 0
            for feats in SINGLES:
                prof = DemographicProfile(features=feats)
                for q in qs:
                    try:
                        r = requests.post(URL, json={"model": ollama_model, "stream": False,
                            "messages": [{"role":"system","content":SYS},
                                         {"role":"user","content":build_multi_question_prompt(prof,[q])}],
                            "options": {"temperature": 0.7, "num_ctx": 2048}}, timeout=120)
                        dist = parse_multi_question_response(r.json()["message"]["content"], [q])[0]
                    except Exception:
                        dist = None
                    out.write(json.dumps({"demographics": feats, "question_id": q.question_id,
                        "response_distribution": dist if dist else [],
                        "status": "success" if dist else "fail"})+"\n")
                    ok += 1 if dist else 0
                out.flush()
            print(f"{model_tag} run{run}: {ok} ok", flush=True)

for tag, mdl in [("gemma2_9b","gemma2:9b"), ("mistral_latest","mistral:latest"),
                 ("llama3_1_8b_instruct_q4","llama3.1:8b-instruct-q4_K_M")]:
    if tag in sys.argv[1:] or len(sys.argv) == 1:
        gen(tag, mdl)
print("STOCH REPEATS DONE", flush=True)
