
import json,subprocess,sys
key='SCRUBBED-SET-YOUR-OWN-KEY'
for w in ["42","43","92"]:
    print(f"=== fetch W{w} ===",flush=True)
    subprocess.run([sys.executable,"batch_api_runner.py","fetch","--wave",w,"--model","gpt-4o-mini","--api-key",key,"--q-batch-size","1"])
print("READY FETCH DONE",flush=True)
