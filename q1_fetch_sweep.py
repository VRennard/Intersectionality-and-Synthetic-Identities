
import json,subprocess,sys
key='SCRUBBED-SET-YOUR-OWN-KEY'
order=['26', '27', '29', '34', '36', '41', '49', '50', '32', '42', '43', '45', '54', '82', '92']
for w in order:
    print(f"=== fetch W{w} ===",flush=True)
    subprocess.run([sys.executable,"batch_api_runner.py","fetch","--wave",w,"--model","gpt-4o-mini","--api-key",key,"--q-batch-size","1"])
print("Q1 FETCH SWEEP DONE",flush=True)
