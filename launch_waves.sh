#!/bin/bash
KEY="SCRUBBED-SET-YOUR-OWN-KEY"
export PYTHONIOENCODING=utf-8

nohup python -u run_all_waves_simulation.py --model-type openai --model gpt-4o-mini --api-key "$KEY" --waves 32 > logs/run_gpt-4o-mini_W32.log 2>&1 &
echo "W32 PID: $!"

nohup python -u run_all_waves_simulation.py --model-type openai --model gpt-4o-mini --api-key "$KEY" --waves 34 > logs/run_gpt-4o-mini_W34.log 2>&1 &
echo "W34 PID: $!"
