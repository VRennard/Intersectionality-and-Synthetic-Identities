import subprocess, os, sys

import os; base = os.path.dirname(os.path.abspath(__file__))
key  = "SCRUBBED-SET-YOUR-OWN-KEY"

env = os.environ.copy()
env["PYTHONIOENCODING"] = "utf-8"
env["OPENAI_API_KEY"]   = key

for wave in ["41", "42"]:
    log_out = open(os.path.join(base, "logs", f"run_gpt-4o-mini_W{wave}_resume.log"), "w")
    log_err = open(os.path.join(base, "logs", f"run_gpt-4o-mini_W{wave}_resume_err.log"), "w")
    p = subprocess.Popen(
        [sys.executable, "-u", "run_all_waves_simulation.py",
         "--model-type", "openai", "--model", "gpt-4o-mini",
         "--api-key", key, "--waves", wave],
        cwd=base, env=env, stdout=log_out, stderr=log_err
    )
    print(f"W{wave} PID: {p.pid}")
