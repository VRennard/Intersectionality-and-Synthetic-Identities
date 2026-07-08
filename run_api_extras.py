"""B5+B6: (a) GPT-5.5 repeat runs (W26 singles x4, OpenAI Batch) for Index_self;
(b) Haiku framing (promptA/promptB) + reversed-order (promptR) controls on W26
(Anthropic Batch). Modes: submit_gpt | collect_gpt | submit_haiku | collect_haiku
"""
import os, sys, json, re, time, pickle, itertools, urllib.request
BASE = os.path.dirname(os.path.abspath(__file__))
OPENAI_KEY = "SCRUBBED-SET-YOUR-OWN-KEY"
GPT_MODEL = "gpt-5.5-2026-04-23"
HK_MODEL = "claude-haiku-4-5-20251001"
STATE = os.path.join(BASE, "data", "results", "_api_extras_state.pkl")

SYSTEM = ("You are an expert demographic researcher and data simulator. "
          "Your task is to accurately model how specific, intersecting demographic "
          "groups would respond to survey questions based on sociological trends, "
          "polling data, and community consensus.")
CONSTRAINTS = """Output Constraints:

    You must output ONLY a raw Python list of integers. Example: [150, 250, 500, 100]

    Do NOT wrap the output in Markdown code blocks (do not use ```).

    Do NOT include any variable declarations, text, or explanations.

    The list must contain exactly as many integers as there are Options. \
The order of the integers must exactly match the order of the Options provided above.

    The sum of the integers in the list MUST equal exactly 1000."""

def pipeline_prompt(feats, qtext, opts):
    demo = "".join(f"        Feature {i}: {f}\n" for i, f in enumerate(feats, 1))
    optstr = "\n".join(f"    {i}. {o}" for i, o in enumerate(opts, 1))
    return (f"Demographic Profile:\n\n{demo}\nThe Task:\nAct as a representative "
            f"modeling this exact community. You are surveying exactly 1,000 "
            f"individuals who fit this combined profile. Distribute these 1,000 "
            f"individuals across the multiple-choice options provided below, "
            f"reflecting how this specific demographic would realistically vote or "
            f"respond.\n\nThe Question:\n{qtext}\n\nOptions:\n{optstr}\n\n{CONSTRAINTS}")

def neutral_prompt(feats, qtext, opts, narrative):
    demo = "\n".join(f"Feature {i+1}: {f}" for i, f in enumerate(feats))
    optstr = "\n".join(f"{i+1}. {o}" for i, o in enumerate(opts))
    return (f"Demographic Profile:\n\n{demo}\n\nThe Task:\n{narrative}\n\n"
            f"The Question:\n{qtext}\n\nOptions:\n{optstr}\n\n{CONSTRAINTS}")

VARIANTS = {
    "promptI": ("You are an expert demographic researcher and data simulator. "
        "Your task is to accurately model how specific, intersecting demographic "
        "groups would respond to survey questions based on sociological trends, "
        "polling data, and community consensus.",
        "Act as a representative modeling this exact community. Consider "
        "carefully how holding BOTH identities together---the combination, not "
        "either one alone---jointly shapes this person's views. You are "
        "surveying exactly 1,000 individuals who fit this combined profile. "
        "Distribute these 1,000 individuals across the multiple-choice options "
        "provided below, reflecting how this specific joint demographic would "
        "realistically respond."),
    "promptC": ("You are an expert demographic researcher and data simulator.",
        "First, reason step by step (briefly) about how the combination of "
        "these identities---jointly, not separately---shapes views on this "
        "question. Then distribute exactly 1,000 individuals who fit this "
        "combined profile across the options. End your answer with ONLY the "
        "raw Python list of integers."),
    "promptA": ("You are a survey response simulator.",
        "Estimate how 1,000 people who fit the profile above would answer the "
        "question below. Distribute exactly 1,000 individuals across the options, "
        "reflecting how this group would realistically respond."),
    "promptB": ("You are simulating the responses of individual people.",
        "Picture 1,000 individual people who each match the profile above. "
        "Report how many of them would choose each option, as a distribution over "
        "exactly 1,000 individuals."),
}
DIMS = {"Age": ["18-29","30-49","50-64","65+"], "Gender": ["Male","Female"],
        "Race": ["White","Black","Hispanic","Asian","Mixed Race"],
        "Political Party": ["Democrat","Republican","Independent"],
        "Religion": ["Roman Catholic","Atheist","Jewish","Muslim"]}

def load_q(w="26", cap=None):
    qs = json.load(open(os.path.join(BASE, f"data/responses/survey_responses_W{w}.json")))
    out = [(q["question_id"], q.get("question_text", q["question_id"]),
            [r["option"] for r in q["responses"]]) for q in qs]
    return out[:cap] if cap else out

def parse_dist(text, n):
    m = re.search(r"\[[\d,\s]+\]", text or "")
    if not m: return None
    vals = [int(x) for x in re.findall(r"\d+", m.group(0))]
    if len(vals) != n or sum(vals) <= 0: return None
    s = sum(vals)
    if s != 1000:
        vals = [int(round(v*1000/s)) for v in vals]
        vals[-1] += 1000 - sum(vals)
    return vals

def state():
    return pickle.load(open(STATE,"rb")) if os.path.exists(STATE) else {}

def save_state(d):
    pickle.dump(d, open(STATE,"wb"))

# ---------------- GPT-5.5 repeats ----------------
def gpt_manifest():
    # W26 singles from the existing gpt-5.5 data
    seen = {}
    for ln in open(os.path.join(BASE, "data/results/gpt-5.5-2026-04-23/W26.jsonl")):
        try: r = json.loads(ln)
        except: continue
        if r.get("status") != "success" or len(r.get("demographics", [])) != 1: continue
        seen[(tuple(r["demographics"]), r["question_id"])] = (r.get("question_text", r["question_id"]), r.get("options", []))
    man = []
    for run in (1,2,3,4):
        for (demo, qid), (txt, opts) in seen.items():
            man.append(dict(run=run, demographics=list(demo), qid=qid, qtext=txt, options=opts))
    return man

def openai_api(path, data=None, headers=None, method=None, raw=False):
    url = f"https://api.openai.com{path}"
    h = {"Authorization": f"Bearer {OPENAI_KEY}"}
    if headers: h.update(headers)
    req = urllib.request.Request(url, data=data, headers=h, method=method)
    resp = urllib.request.urlopen(req, timeout=300).read()
    return resp if raw else json.loads(resp)

def submit_gpt():
    man = gpt_manifest()
    print(f"gpt-5.5 repeat manifest: {len(man):,} requests")
    lines = []
    for i, m in enumerate(man):
        body = {"model": GPT_MODEL,
                "messages": [{"role":"system","content":SYSTEM},
                             {"role":"user","content":pipeline_prompt(m["demographics"], m["qtext"], m["options"])}]}
        lines.append(json.dumps({"custom_id": f"g{i}", "method": "POST",
                                 "url": "/v1/chat/completions", "body": body}))
    payload = "\n".join(lines).encode()
    # multipart upload
    boundary = "XBOUNDARYX"
    part = (f"--{boundary}\r\nContent-Disposition: form-data; name=\"purpose\"\r\n\r\nbatch\r\n"
            f"--{boundary}\r\nContent-Disposition: form-data; name=\"file\"; filename=\"reps.jsonl\"\r\n"
            f"Content-Type: application/jsonl\r\n\r\n").encode() + payload + f"\r\n--{boundary}--\r\n".encode()
    up = openai_api("/v1/files", data=part, headers={"Content-Type": f"multipart/form-data; boundary={boundary}"})
    print("uploaded file:", up["id"])
    b = openai_api("/v1/batches", data=json.dumps({"input_file_id": up["id"],
                   "endpoint": "/v1/chat/completions", "completion_window": "24h"}).encode(),
                   headers={"Content-Type": "application/json"})
    print("batch:", b["id"], b["status"])
    st = state(); st["gpt_batch"] = b["id"]; st["gpt_manifest"] = man; save_state(st)

def collect_gpt():
    st = state(); bid = st["gpt_batch"]; man = st["gpt_manifest"]
    while True:
        b = openai_api(f"/v1/batches/{bid}")
        print("status:", b["status"], b.get("request_counts"))
        if b["status"] in ("completed","failed","expired","cancelled"): break
        time.sleep(120)
    if b["status"] != "completed" and not b.get("output_file_id"):
        print("no output"); return
    raw = openai_api(f"/v1/files/{b['output_file_id']}/content", raw=True).decode()
    outdir = os.path.join(BASE, "data/results/stoch_gpt-5.5-2026-04-23")
    os.makedirs(outdir, exist_ok=True)
    fhs = {}
    ok = fail = 0
    for ln in raw.splitlines():
        r = json.loads(ln)
        i = int(r["custom_id"][1:]); m = man[i]
        dist = None
        try:
            txt = r["response"]["body"]["choices"][0]["message"]["content"]
            dist = parse_dist(txt, len(m["options"]))
        except Exception: pass
        rec = {"demographics": m["demographics"], "question_id": m["qid"],
               "options": m["options"], "response_distribution": dist or [],
               "status": "success" if dist else "error"}
        p = os.path.join(outdir, f"W26_run{m['run']}.jsonl")
        if p not in fhs: fhs[p] = open(p, "a")
        fhs[p].write(json.dumps(rec)+"\n")
        ok += bool(dist); fail += (not dist)
    for f in fhs.values(): f.close()
    print(f"gpt-5.5 repeats written: ok={ok:,} fail={fail:,}")

# ---------------- Haiku controls ----------------
def haiku_manifest():
    qs20 = load_q(cap=20); qs25 = load_q(cap=25)
    singles, pairs = [], []
    for d, vals in DIMS.items():
        for v in vals: singles.append([f"{d} {v}"])
    keys = list(DIMS)
    for a, b in itertools.combinations(keys, 2):
        combos = [(va, vb) for va in DIMS[a] for vb in DIMS[b]]
        step = max(1, len(combos)//5)
        for va, vb in combos[::step][:5]:
            pairs.append([f"{a} {va}", f"{b} {vb}"])
    man = []
    for var in ("promptA","promptB"):
        for feats in singles + pairs:
            for qid, txt, opts in qs25:
                man.append(dict(var=var, demographics=feats, qid=qid, qtext=txt, options=opts, rev=False))
    for feats in pairs:                                  # promptR: reversed order, pairs only
        for qid, txt, opts in qs20:
            man.append(dict(var="promptR", demographics=feats, qid=qid, qtext=txt, options=opts, rev=True))
    return man

def submit_haiku():
    import anthropic
    c = anthropic.Anthropic()
    man = haiku_manifest()
    print(f"haiku controls manifest: {len(man):,} requests")
    reqs = []
    for i, m in enumerate(man):
        feats = list(reversed(m["demographics"])) if m["rev"] else m["demographics"]
        sysm, narr = VARIANTS[m["var"]] if m["var"] in VARIANTS else VARIANTS["promptA"]
        reqs.append({"custom_id": f"h{i}", "params": {
            "model": HK_MODEL, "max_tokens": 150, "temperature": 0.7,
            "system": sysm,
            "messages": [{"role":"user","content":neutral_prompt(feats, m["qtext"], m["options"], narr)}]}})
    b = c.messages.batches.create(requests=reqs)
    print("batch:", b.id, b.processing_status)
    st = state(); st["haiku_batch"] = b.id; st["haiku_manifest"] = man; save_state(st)

def collect_haiku():
    import anthropic
    c = anthropic.Anthropic()
    st = state(); bid = st["haiku_batch"]; man = st["haiku_manifest"]
    while True:
        b = c.messages.batches.retrieve(bid)
        print("status:", b.processing_status, b.request_counts)
        if b.processing_status == "ended": break
        time.sleep(60)
    fhs = {}; ok = fail = 0
    for res in c.messages.batches.results(bid):
        i = int(res.custom_id[1:]); m = man[i]
        dist = None
        if res.result.type == "succeeded":
            txt = "".join(bl.text for bl in res.result.message.content if getattr(bl,"type","")=="text")
            dist = parse_dist(txt, len(m["options"]))
        rec = {"demographics": m["demographics"], "question_id": m["qid"],
               "options": m["options"], "response_distribution": dist or [],
               "status": "success" if dist else "error"}
        outdir = os.path.join(BASE, f"data/results/{HK_MODEL}_{m['var']}")
        os.makedirs(outdir, exist_ok=True)
        p = os.path.join(outdir, "W26.jsonl")
        if p not in fhs: fhs[p] = open(p, "a")
        fhs[p].write(json.dumps(rec)+"\n")
        ok += bool(dist); fail += (not dist)
    for f in fhs.values(): f.close()
    print(f"haiku controls written: ok={ok:,} fail={fail:,}")

def instructed_manifest():
    qs20 = load_q(cap=20)
    pairs = []
    keys = list(DIMS)
    for a, b in itertools.combinations(keys, 2):
        combos = [(va, vb) for va in DIMS[a] for vb in DIMS[b]]
        step = max(1, len(combos)//5)
        for va, vb in combos[::step][:5]:
            pairs.append([f"{a} {va}", f"{b} {vb}"])
    man = []
    for var in ("promptI","promptC"):
        for feats in pairs:
            for qid, txt, opts in qs20:
                man.append(dict(var=var, demographics=feats, qid=qid, qtext=txt, options=opts, rev=False))
    return man

def submit_instr():
    import anthropic
    c = anthropic.Anthropic()
    man = instructed_manifest()
    print(f"instructed-composition manifest: {len(man):,} requests")
    reqs = []
    for i, m in enumerate(man):
        sysm, narr = VARIANTS[m["var"]]
        mt = 600 if m["var"]=="promptC" else 150
        reqs.append({"custom_id": f"n{i}", "params": {
            "model": HK_MODEL, "max_tokens": mt, "temperature": 0.7,
            "system": sysm,
            "messages": [{"role":"user","content":neutral_prompt(m["demographics"], m["qtext"], m["options"], narr)}]}})
    b = c.messages.batches.create(requests=reqs)
    print("batch:", b.id, b.processing_status)
    st = state(); st["instr_batch"] = b.id; st["instr_manifest"] = man; save_state(st)

def collect_instr():
    import anthropic
    c = anthropic.Anthropic()
    st = state(); bid = st["instr_batch"]; man = st["instr_manifest"]
    while True:
        b = c.messages.batches.retrieve(bid)
        print("status:", b.processing_status, b.request_counts)
        if b.processing_status == "ended": break
        time.sleep(60)
    fhs = {}; ok = fail = 0
    for res in c.messages.batches.results(bid):
        i = int(res.custom_id[1:]); m = man[i]
        dist = None
        if res.result.type == "succeeded":
            txt = "".join(bl.text for bl in res.result.message.content if getattr(bl,"type","")=="text")
            # for CoT take the LAST list in the text
            import re as _re
            lists = _re.findall(r"\[[\d,\s]+\]", txt or "")
            if lists:
                dist = parse_dist(lists[-1], len(m["options"]))
        rec = {"demographics": m["demographics"], "question_id": m["qid"],
               "options": m["options"], "response_distribution": dist or [],
               "status": "success" if dist else "error"}
        outdir = os.path.join(BASE, f"data/results/{HK_MODEL}_{m['var']}")
        os.makedirs(outdir, exist_ok=True)
        p = os.path.join(outdir, "W26.jsonl")
        if p not in fhs: fhs[p] = open(p, "a")
        fhs[p].write(json.dumps(rec)+"\n")
        ok += bool(dist); fail += (not dist)
    for f in fhs.values(): f.close()
    print(f"instructed-composition written: ok={ok:,} fail={fail:,}")

def instr4o_manifest():
    pairs = []
    keys = list(DIMS)
    for a, b in itertools.combinations(keys, 2):
        combos = [(va, vb) for va in DIMS[a] for vb in DIMS[b]]
        step = max(1, len(combos)//5)
        for va, vb in combos[::step][:5]:
            pairs.append([f"{a} {va}", f"{b} {vb}"])
    man = []
    for w in ("26", "34"):
        qs20 = load_q(w=w, cap=20)
        for var in ("promptI","promptC"):
            for feats in pairs:
                for qid, txt, opts in qs20:
                    man.append(dict(var=var, wave=w, demographics=feats, qid=qid,
                                    qtext=txt, options=opts))
    return man

def submit_instr4o():
    man = instr4o_manifest()
    print(f"gpt-4o-mini instructed manifest: {len(man):,} requests")
    lines = []
    for i, m in enumerate(man):
        sysm, narr = VARIANTS[m["var"]]
        mt = 600 if m["var"]=="promptC" else 150
        body = {"model": "gpt-4o-mini-2024-07-18", "temperature": 0.7,
                "max_tokens": mt,
                "messages": [{"role":"system","content":sysm},
                             {"role":"user","content":neutral_prompt(m["demographics"], m["qtext"], m["options"], narr)}]}
        lines.append(json.dumps({"custom_id": f"m{i}", "method": "POST",
                                 "url": "/v1/chat/completions", "body": body}))
    payload = "\n".join(lines).encode()
    boundary = "XBOUNDARYX"
    part = (f"--{boundary}\r\nContent-Disposition: form-data; name=\"purpose\"\r\n\r\nbatch\r\n"
            f"--{boundary}\r\nContent-Disposition: form-data; name=\"file\"; filename=\"instr4o.jsonl\"\r\n"
            f"Content-Type: application/jsonl\r\n\r\n").encode() + payload + f"\r\n--{boundary}--\r\n".encode()
    up = openai_api("/v1/files", data=part, headers={"Content-Type": f"multipart/form-data; boundary={boundary}"})
    print("uploaded:", up["id"])
    b = openai_api("/v1/batches", data=json.dumps({"input_file_id": up["id"],
                   "endpoint": "/v1/chat/completions", "completion_window": "24h"}).encode(),
                   headers={"Content-Type": "application/json"})
    print("batch:", b["id"], b["status"])
    st = state(); st["instr4o_batch"] = b["id"]; st["instr4o_manifest"] = man; save_state(st)

def collect_instr4o():
    st = state(); bid = st["instr4o_batch"]; man = st["instr4o_manifest"]
    while True:
        b = openai_api(f"/v1/batches/{bid}")
        print("status:", b["status"], b.get("request_counts"))
        if b["status"] in ("completed","failed","expired","cancelled"): break
        time.sleep(120)
    if not b.get("output_file_id"):
        print("no output"); return
    raw = openai_api(f"/v1/files/{b['output_file_id']}/content", raw=True).decode()
    import re as _re
    fhs = {}; ok = fail = 0
    for ln in raw.splitlines():
        r = json.loads(ln)
        i = int(r["custom_id"][1:]); m = man[i]
        dist = None
        try:
            txt = r["response"]["body"]["choices"][0]["message"]["content"]
            lists = _re.findall(r"\[[\d,\s]+\]", txt or "")
            if lists: dist = parse_dist(lists[-1], len(m["options"]))
        except Exception: pass
        rec = {"demographics": m["demographics"], "question_id": m["qid"],
               "options": m["options"], "response_distribution": dist or [],
               "status": "success" if dist else "error"}
        outdir = os.path.join(BASE, f"data/results/gpt-4o-mini_{m['var']}")
        os.makedirs(outdir, exist_ok=True)
        p = os.path.join(outdir, f"W{m['wave']}.jsonl")
        if p not in fhs: fhs[p] = open(p, "a")
        fhs[p].write(json.dumps(rec)+"\n")
        ok += bool(dist); fail += (not dist)
    for f in fhs.values(): f.close()
    print(f"gpt-4o-mini instructed written: ok={ok:,} fail={fail:,}")

VARIANTS_S = {   # singular wording for depth-1 under instruction
    "promptI": ("You are an expert demographic researcher and data simulator. "
        "Your task is to accurately model how specific demographic "
        "groups would respond to survey questions based on sociological trends, "
        "polling data, and community consensus.",
        "Act as a representative modeling this exact community. Consider "
        "carefully how holding this identity shapes this person's views. You "
        "are surveying exactly 1,000 individuals who fit this profile. "
        "Distribute these 1,000 individuals across the multiple-choice options "
        "provided below, reflecting how this demographic would realistically "
        "respond."),
    "promptC": ("You are an expert demographic researcher and data simulator.",
        "First, reason step by step (briefly) about how this identity shapes "
        "views on this question. Then distribute exactly 1,000 individuals who "
        "fit this profile across the options. End your answer with ONLY the "
        "raw Python list of integers."),
}

def instr4o_singles_manifest():
    singles = []
    for d, vals in DIMS.items():
        for v in vals: singles.append([f"{d} {v}"])
    man = []
    for w in ("26", "34"):
        qs20 = load_q(w=w, cap=20)
        for var in ("promptI","promptC"):
            for feats in singles:
                for qid, txt, opts in qs20:
                    man.append(dict(var=var, wave=w, demographics=feats, qid=qid,
                                    qtext=txt, options=opts))
    return man

def submit_instr4o_s():
    man = instr4o_singles_manifest()
    print(f"gpt-4o-mini instructed SINGLES manifest: {len(man):,} requests")
    lines = []
    for i, m in enumerate(man):
        sysm, narr = VARIANTS_S[m["var"]]
        mt = 600 if m["var"]=="promptC" else 150
        body = {"model": "gpt-4o-mini-2024-07-18", "temperature": 0.7,
                "max_tokens": mt,
                "messages": [{"role":"system","content":sysm},
                             {"role":"user","content":neutral_prompt(m["demographics"], m["qtext"], m["options"], narr)}]}
        lines.append(json.dumps({"custom_id": f"s{i}", "method": "POST",
                                 "url": "/v1/chat/completions", "body": body}))
    payload = "\n".join(lines).encode()
    boundary = "XBOUNDARYX"
    part = (f"--{boundary}\r\nContent-Disposition: form-data; name=\"purpose\"\r\n\r\nbatch\r\n"
            f"--{boundary}\r\nContent-Disposition: form-data; name=\"file\"; filename=\"instr4os.jsonl\"\r\n"
            f"Content-Type: application/jsonl\r\n\r\n").encode() + payload + f"\r\n--{boundary}--\r\n".encode()
    up = openai_api("/v1/files", data=part, headers={"Content-Type": f"multipart/form-data; boundary={boundary}"})
    b = openai_api("/v1/batches", data=json.dumps({"input_file_id": up["id"],
                   "endpoint": "/v1/chat/completions", "completion_window": "24h"}).encode(),
                   headers={"Content-Type": "application/json"})
    print("batch:", b["id"], b["status"])
    st = state(); st["instr4os_batch"] = b["id"]; st["instr4os_manifest"] = man; save_state(st)

def collect_instr4o_s():
    st = state(); bid = st["instr4os_batch"]; man = st["instr4os_manifest"]
    while True:
        b = openai_api(f"/v1/batches/{bid}")
        print("status:", b["status"], b.get("request_counts"))
        if b["status"] in ("completed","failed","expired","cancelled"): break
        time.sleep(120)
    if not b.get("output_file_id"):
        print("no output"); return
    raw = openai_api(f"/v1/files/{b['output_file_id']}/content", raw=True).decode()
    import re as _re
    fhs = {}; ok = fail = 0
    for ln in raw.splitlines():
        r = json.loads(ln)
        i = int(r["custom_id"][1:]); m = man[i]
        dist = None
        try:
            txt = r["response"]["body"]["choices"][0]["message"]["content"]
            lists = _re.findall(r"\[[\d,\s]+\]", txt or "")
            if lists: dist = parse_dist(lists[-1], len(m["options"]))
        except Exception: pass
        rec = {"demographics": m["demographics"], "question_id": m["qid"],
               "options": m["options"], "response_distribution": dist or [],
               "status": "success" if dist else "error"}
        outdir = os.path.join(BASE, f"data/results/gpt-4o-mini_{m['var']}")
        os.makedirs(outdir, exist_ok=True)
        p = os.path.join(outdir, f"W{m['wave']}.jsonl")
        if p not in fhs: fhs[p] = open(p, "a")
        fhs[p].write(json.dumps(rec)+"\n")
        ok += bool(dist); fail += (not dist)
    for f in fhs.values(): f.close()
    print(f"instructed singles written: ok={ok:,} fail={fail:,}")

if __name__ == "__main__":
    {"submit_gpt": submit_gpt, "collect_gpt": collect_gpt,
     "submit_haiku": submit_haiku, "collect_haiku": collect_haiku,
     "submit_instr": submit_instr, "collect_instr": collect_instr,
     "submit_instr4o": submit_instr4o, "collect_instr4o": collect_instr4o,
     "submit_instr4o_s": submit_instr4o_s, "collect_instr4o_s": collect_instr4o_s}[sys.argv[1]]()
