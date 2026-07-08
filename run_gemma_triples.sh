#!/bin/zsh
cd "$(dirname "$0")"
python3 ollama_fill/run_ollama_fill.py --manifest-dir ollama_fill/manifests/gemma2_9b_party --model gemma2:9b --tag gemma2_9b --workers 16
