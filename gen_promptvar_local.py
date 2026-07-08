"""Referee major #5: prompt-sensitivity of the depth-2 collapse contest.

Regenerate W26 depth-1 (singles) and depth-2 (pairs) aggregate distributions
for gemma2:9b under TWO alternative prompt framings, to test whether the
collapse is an artefact of the main pipeline's "expert demographic researcher"
system prompt. Baseline data already exist in data/results/gemma2_9b/.

Output: data/results/gemma2_9b_promptA|B/W26.jsonl, same schema as the main
pipeline (demographics, question_id, response_distribution, options, status),
so the existing collapse contest runs on it by swapping MODEL_TAG.
"""
import os, sys, json, re, time, itertools
import requests

OLLAMA = "http://localhost:11434/api/chat"
MODEL = "gemma2:9b"
WAVE = "26"
SURVEY = f"data/responses/survey_responses_W{WAVE}.json"
OUTDIR = "data/results"

# 5 headline dimensions for the sensitivity probe (collapse-direction relevant)
DIMS = {
    "Age": ["18-29", "30-49", "50-64", "65+"],
    "Gender": ["Male", "Female"],
    "Race": ["White", "Black", "Hispanic", "Asian", "Mixed Race"],
    "Political Party": ["Democrat", "Republican", "Independent"],
    "Religion": ["Roman Catholic", "Atheist", "Jewish", "Muslim"],
}

OUTPUT_CONSTRAINTS = """\
Output Constraints:

    You must output ONLY a raw Python list of integers. Example: [150, 250, 500, 100]

    Do NOT wrap the output in Markdown code blocks (do not use ```).

    Do NOT include any variable declarations, text, or explanations.

    The list must contain exactly as many integers as there are Options. \
The order of the integers must exactly match the order of the Options provided above.

    The sum of the integers in the list MUST equal exactly 1000."""

# (system prompt, narrative paragraph) — the Output Constraints block is fixed.
VARIANTS = {
    "promptA": (  # neutral, strips "expert / community consensus / representative"
        "You are a survey response simulator.",
        "Estimate how 1,000 people who fit the profile above would answer the "
        "question below. Distribute exactly 1,000 individuals across the options, "
        "reflecting how this group would realistically respond.",
    ),
    "promptB": (  # first-person / individual framing
        "You are simulating the responses of individual people.",
        "Picture 1,000 individual people who each match the profile above. "
        "Report how many of them would choose each option, as a distribution over "
        "exactly 1,000 individuals.",
    ),
    "promptR": (  # position control: SAME as promptA (neutral), run with REVERSE=1
        "You are a survey response simulator.",
        "Estimate how 1,000 people who fit the profile above would answer the "
        "question below. Distribute exactly 1,000 individuals across the options, "
        "reflecting how this group would realistically respond.",
    ),
}


def load_questions():
    qs = json.load(open(SURVEY))
    out = []
    for q in qs:
        opts = [r["option"] for r in q["responses"]]
        out.append((q["question_id"], q.get("question_text", q["question_id"]), opts))
    return out


PAIRCAP = int(os.environ.get("PAIRCAP", "5"))  # value-combos per dim-pair
QCAP = int(os.environ.get("QCAP", "25"))        # questions used


def cells():
    singles, pairs = [], []
    for d, vals in DIMS.items():
        for v in vals:
            singles.append([f"{d} {v}"])
    keys = list(DIMS)
    for a, b in itertools.combinations(keys, 2):
        combos = [(va, vb) for va in DIMS[a] for vb in DIMS[b]]
        # deterministic spread of up to PAIRCAP combos across the value grid
        step = max(1, len(combos) // PAIRCAP)
        for va, vb in combos[::step][:PAIRCAP]:
            pairs.append([f"{a} {va}", f"{b} {vb}"])
    if os.environ.get("PAIRS_ONLY", "") == "1":
        return pairs
    return singles + pairs


REVERSE = os.environ.get("REVERSE", "") == "1"


def build_messages(system, narrative, feats, qtext, opts):
    if REVERSE:
        feats = list(reversed(feats))
    demo = "\n".join(f"Feature {i+1}: {f}" for i, f in enumerate(feats))
    optstr = "\n".join(f"{i+1}. {o}" for i, o in enumerate(opts))
    user = (f"Demographic Profile:\n\n{demo}\n\nThe Task:\n{narrative}\n\n"
            f"The Question:\n{qtext}\n\nOptions:\n{optstr}\n\n{OUTPUT_CONSTRAINTS}")
    return [{"role": "system", "content": system},
            {"role": "user", "content": user}]


def parse(text, n):
    m = re.search(r"\[[\d,\s]+\]", text)
    if not m:
        return None
    try:
        vals = [int(x) for x in re.findall(r"\d+", m.group(0))]
    except Exception:
        return None
    if len(vals) != n:
        return None
    s = sum(vals)
    if s <= 0:
        return None
    if s != 1000:
        vals = [int(round(v * 1000 / s)) for v in vals]
        vals[-1] += 1000 - sum(vals)
    return vals


def call(messages, retries=3):
    payload = {"model": MODEL, "messages": messages, "stream": False,
               "options": {"temperature": 0.7, "top_p": 0.9, "num_ctx": 2048,
                           "num_predict": 256}}
    for k in range(retries):
        try:
            r = requests.post(OLLAMA, json=payload, timeout=120)
            return r.json()["message"]["content"]
        except Exception:
            time.sleep(2 * (k + 1))
    return None


def done_keys(path):
    seen = set()
    if os.path.exists(path):
        for ln in open(path):
            try:
                r = json.loads(ln)
                if r.get("status") == "success":
                    seen.add((tuple(r["demographics"]), r["question_id"]))
            except Exception:
                pass
    return seen


def main():
    which = sys.argv[1:] or list(VARIANTS)
    qs = load_questions()[:QCAP]
    cs = cells()
    print(f"W{WAVE}: {len(cs)} cells x {len(qs)} questions, variants={which}", flush=True)
    for var in which:
        system, narrative = VARIANTS[var]
        outd = os.path.join(OUTDIR, f"gemma2_9b_{var}")
        os.makedirs(outd, exist_ok=True)
        path = os.path.join(outd, f"W{WAVE}.jsonl")
        seen = done_keys(path)
        ok = skip = fail = 0
        with open(path, "a") as fh:
            for feats in cs:
                for qid, qtext, opts in qs:
                    if (tuple(feats), qid) in seen:
                        skip += 1
                        continue
                    txt = call(build_messages(system, narrative, feats, qtext, opts))
                    dist = parse(txt, len(opts)) if txt else None
                    rec = {"demographics": feats, "question_id": qid,
                           "options": opts,
                           "response_distribution": dist if dist else [],
                           "status": "success" if dist else "error"}
                    fh.write(json.dumps(rec) + "\n"); fh.flush()
                    if dist:
                        ok += 1
                    else:
                        fail += 1
                print(f"  [{var}] {feats}: ok={ok} fail={fail} skip={skip}", flush=True)
        print(f"=== {var} done: ok={ok} fail={fail} skip={skip} ===", flush=True)


if __name__ == "__main__":
    main()
