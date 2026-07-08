"""Referee major #5: add Claude Sonnet 5 as a second current-generation frontier
model on the GPT-5.5 two-wave subset (W26, W34), all depths, via the Anthropic
Message Batches API. Prompt is byte-identical to the pipeline GPT-5.5 used
(SYSTEM_PROMPT + build_instruct_prompt), single question per call, temp 0.7.

Modes:
  python3 run_sonnet5_batch.py plan      # count cells, no spend
  python3 run_sonnet5_batch.py test      # submit a 3-cell batch, poll, verify
  python3 run_sonnet5_batch.py submit    # submit full batches, save state
  python3 run_sonnet5_batch.py collect   # poll + retrieve + write results
"""
import os, sys, json, re, time, pickle, itertools
import anthropic

BASE = os.path.dirname(os.path.abspath(__file__))
MODEL = "claude-sonnet-5"
GPT55 = os.path.join(BASE, "data", "results", "gpt-5.5-2026-04-23")
OUTDIR = os.path.join(BASE, "data", "results", "claude-sonnet-5")
STOCHDIR = os.path.join(BASE, "data", "results", "stoch_claude-sonnet-5")
STATE = os.path.join(BASE, "data", "results", "_sonnet5_batch_state.pkl")
WAVES = ["26", "34"]
N_REPEAT = 4               # W26 singles repeats for Index_self
BATCH_MAX = 50_000
TEMP = 0.7
MAXTOK = 150

# ---- verbatim pipeline prompt (from ollama_fill/run_ollama_fill.py) ----
SYSTEM_PROMPT = (
    "You are an expert demographic researcher and data simulator. "
    "Your task is to accurately model how specific, intersecting demographic "
    "groups would respond to survey questions based on sociological trends, "
    "polling data, and community consensus."
)
USER_PROMPT_TEMPLATE = """\
Demographic Profile:

{demographic_features}

The Task:
Act as a representative modeling this exact community. You are surveying \
exactly 1,000 individuals who fit this combined profile. Distribute these \
1,000 individuals across the multiple-choice options provided below, \
reflecting how this specific demographic would realistically vote or respond.

The Question:
{question_text}

Options:
{options}

Output Constraints:

    You must output ONLY a raw Python list of integers. Example: [150, 250, 500, 100]

    Do NOT wrap the output in Markdown code blocks (do not use ```).

    Do NOT include any variable declarations, text, or explanations.

    The list must contain exactly as many integers as there are Options. \
The order of the integers must exactly match the order of the Options provided above.

    The sum of the integers in the list MUST equal exactly 1000."""


def fmt_features(feats):
    return "".join(f"        Feature {i}: {f}\n" for i, f in enumerate(feats, 1))


def fmt_options(opts):
    return "\n".join(f"    {i}. {o}" for i, o in enumerate(opts, 1))


def build_prompt(feats, qtext, opts):
    return USER_PROMPT_TEMPLATE.format(
        demographic_features=fmt_features(feats),
        question_text=qtext, options=fmt_options(opts))


# ---- build the manifest: every request routed by index ----
def build_manifest():
    manifest = []              # index -> dict(out, wave, demographics, qid, options)
    qmeta = {w: {} for w in WAVES}   # wave -> qid -> (text, options)
    for w in WAVES:
        f = os.path.join(GPT55, f"W{w}.jsonl")
        seen = set()
        for ln in open(f):
            try: r = json.loads(ln)
            except: continue
            if r.get("status") != "success": continue
            demo = r["demographics"]; qid = r["question_id"]
            qmeta[w].setdefault(qid, (r.get("question_text", qid), r.get("options", [])))
            key = (tuple(demo), qid)
            if key in seen: continue
            seen.add(key)
            manifest.append(dict(out=f"main:W{w}", wave=w, demographics=demo,
                                 qid=qid, options=r.get("options", [])))
    # population cells (real e_pop): Average American x every question
    for w in WAVES:
        for qid, (txt, opts) in qmeta[w].items():
            manifest.append(dict(out=f"pop:W{w}", wave=w,
                                 demographics=["Average American"], qid=qid, options=opts))
    # Index_self repeats: W26 singles x N_REPEAT runs
    w = "26"
    singles = [(tuple(m["demographics"]), m["qid"], m["options"])
               for m in manifest if m["out"] == "main:W26" and len(m["demographics"]) == 1]
    for run in range(1, N_REPEAT + 1):
        for demo, qid, opts in singles:
            manifest.append(dict(out=f"rep:W26:run{run}", wave=w,
                                 demographics=list(demo), qid=qid, options=opts))
    return manifest, qmeta


def make_request(idx, m, qmeta):
    txt, opts = qmeta[m["wave"]][m["qid"]]
    prompt = build_prompt(m["demographics"], txt, m["options"] or opts)
    return {
        "custom_id": f"i{idx}",
        "params": {
            "model": MODEL, "max_tokens": MAXTOK,
            "system": SYSTEM_PROMPT,
            "messages": [{"role": "user", "content": prompt}],
        },
    }


def parse_dist(text, n):
    m = re.search(r"\[[\d,\s]+\]", text or "")
    if not m: return None
    vals = [int(x) for x in re.findall(r"\d+", m.group(0))]
    if len(vals) != n or sum(vals) <= 0: return None
    s = sum(vals)
    if s != 1000:
        vals = [int(round(v * 1000 / s)) for v in vals]
        vals[-1] += 1000 - sum(vals)
    return vals


def out_path(tag):
    kind = tag.split(":")
    if kind[0] == "main": return os.path.join(OUTDIR, f"W{kind[1][1:]}.jsonl")
    if kind[0] == "pop":  return os.path.join(OUTDIR, f"W{kind[1][1:]}.jsonl")
    if kind[0] == "rep":  return os.path.join(STOCHDIR, f"W26_{kind[2]}.jsonl")


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "plan"
    client = anthropic.Anthropic()
    manifest, qmeta = build_manifest()

    if mode == "plan":
        from collections import Counter
        c = Counter(m["out"].split(":")[0] for m in manifest)
        print(f"total requests: {len(manifest):,}  ({dict(c)})")
        nb = (len(manifest) + BATCH_MAX - 1) // BATCH_MAX
        print(f"batches of {BATCH_MAX:,}: {nb}")
        print("sample prompt:\n", make_request(0, manifest[0], qmeta)["params"]["messages"][0]["content"][:400])
        return

    if mode == "test":
        reqs = [make_request(i, manifest[i], qmeta) for i in (0, 1, 2)]
        b = client.messages.batches.create(requests=reqs)
        print("test batch:", b.id, b.processing_status)
        while True:
            b = client.messages.batches.retrieve(b.id)
            print("  status:", b.processing_status, b.request_counts)
            if b.processing_status == "ended": break
            time.sleep(15)
        for res in client.messages.batches.results(b.id):
            idx = int(res.custom_id[1:]); m = manifest[idx]
            if res.result.type == "succeeded":
                txt = res.result.message.content[0].text
                print(f"  {res.custom_id} {m['demographics']} -> {txt!r} parsed={parse_dist(txt, len(m['options']))}")
            else:
                print(f"  {res.custom_id} FAILED: {res.result.type}")
        return

    if mode == "submit":
        os.makedirs(OUTDIR, exist_ok=True); os.makedirs(STOCHDIR, exist_ok=True)
        batch_ids = []
        for start in range(0, len(manifest), BATCH_MAX):
            chunk = range(start, min(start + BATCH_MAX, len(manifest)))
            reqs = [make_request(i, manifest[i], qmeta) for i in chunk]
            b = client.messages.batches.create(requests=reqs)
            batch_ids.append(b.id)
            print(f"submitted batch {b.id}: {len(reqs):,} requests [{start}:{start+len(reqs)}]")
        pickle.dump({"batch_ids": batch_ids, "manifest": manifest}, open(STATE, "wb"))
        print(f"saved state ({len(batch_ids)} batches) -> {STATE}")
        return

    if mode == "collect":
        st = pickle.load(open(STATE, "rb"))
        manifest = st["manifest"]; files = {}
        def fh(p):
            if p not in files:
                os.makedirs(os.path.dirname(p), exist_ok=True); files[p] = open(p, "a")
            return files[p]
        ok = fail = 0
        for bid in st["batch_ids"]:
            while True:
                b = client.messages.batches.retrieve(bid)
                if b.processing_status == "ended": break
                print(f"  {bid}: {b.processing_status} {b.request_counts}; waiting..."); time.sleep(60)
            for res in client.messages.batches.results(bid):
                idx = int(res.custom_id[1:]); m = manifest[idx]
                dist = None
                if res.result.type == "succeeded":
                    txt = "".join(b.text for b in res.result.message.content
                                  if getattr(b, "type", "") == "text")
                    dist = parse_dist(txt, len(m["options"]))
                rec = {"demographics": m["demographics"], "question_id": m["qid"],
                       "options": m["options"],
                       "response_distribution": dist or [],
                       "status": "success" if dist else "error"}
                fh(out_path(m["out"])).write(json.dumps(rec) + "\n")
                ok += bool(dist); fail += (not dist)
            print(f"  {bid} written (ok={ok} fail={fail})")
        for f in files.values(): f.close()
        print(f"DONE: ok={ok:,} fail={fail:,}")
        return


if __name__ == "__main__":
    main()
