
import json,subprocess,sys
from pathlib import Path
bd=Path("data/batches")
WAVES=["29","32","36","42","43","45","49","50","54","82","92","26","27","34","41"]
key=None
for w in WAVES:
    sf=bd/f"W{w}_gpt-4o-mini_state.json"
    if sf.exists():
        d=json.load(open(sf))
        if d.get("triples_only") and d.get("api_key"): key=d["api_key"]; break
for w in WAVES:
    print(f"=== fetch W{w} ===",flush=True)
    subprocess.run([sys.executable,"batch_api_runner.py","fetch","--wave",w,"--model","gpt-4o-mini","--api-key",key])
print("FETCH SWEEP DONE",flush=True)
