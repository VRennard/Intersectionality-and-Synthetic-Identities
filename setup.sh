#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# setup.sh  –  Set your options in config.sh, then run:  bash setup.sh
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# ── load config ───────────────────────────────────────────────────────────────
if [[ ! -f config.sh ]]; then
    echo "❌  config.sh not found. Did you unpack the archive correctly?"
    exit 1
fi
source config.sh

# ── python check ──────────────────────────────────────────────────────────────
PYTHON=$(command -v python3 || command -v python || true)
if [[ -z "$PYTHON" ]]; then
    echo "❌  Python 3 not found. Install Python 3.10+ and re-run."
    exit 1
fi

# ── virtualenv ────────────────────────────────────────────────────────────────
if [[ ! -d ".venv" ]]; then
    echo "▶  Creating virtual environment …"
    "$PYTHON" -m venv .venv
fi
ACTIVATE=".venv/Scripts/activate"
[[ ! -f "$ACTIVATE" ]] && ACTIVATE=".venv/bin/activate"
source "$ACTIVATE"

# ── dependencies ──────────────────────────────────────────────────────────────
if [[ "${SKIP_INSTALL:-false}" == false ]]; then
    echo "▶  Installing packages …"
    pip install --upgrade pip -q
    pip install -r requirements.txt -q
    echo "✅  Packages installed."
fi

# ── directories ───────────────────────────────────────────────────────────────
mkdir -p logs data/results data/responses data/demographics

# ── API key ───────────────────────────────────────────────────────────────────
if [[ -z "${OPENAI_API_KEY:-}" ]]; then
    echo "❌  OPENAI_API_KEY is not set. Add it to config.sh and re-run."
    exit 1
fi
export OPENAI_API_KEY

# ── wave expansion ────────────────────────────────────────────────────────────
ALL_WAVES=$(ls -d human_resp/American_Trends_Panel_W* 2>/dev/null \
    | sed 's|.*_W||' | sort -n | tr '\n' ' ')

expand_waves() {
    local input="$1"
    local result=()
    for token in $input; do
        if [[ "$token" =~ ^([0-9]+)-([0-9]+)$ ]]; then
            local lo="${BASH_REMATCH[1]}" hi="${BASH_REMATCH[2]}"
            for w in $(seq "$lo" "$hi"); do
                if echo "$ALL_WAVES" | grep -qw "$w"; then
                    result+=("$w")
                fi
            done
        else
            result+=("$token")
        fi
    done
    echo "${result[@]}"
}

WAVE_ARG=""
if [[ "${WAVES:-all}" != "all" ]]; then
    WAVE_LIST=$(expand_waves "$WAVES")
    if [[ -z "$WAVE_LIST" ]]; then
        echo "❌  No matching waves found for: $WAVES"
        echo "    Available: $ALL_WAVES"
        exit 1
    fi
    WAVE_ARG="--waves $WAVE_LIST"
fi

# ── launch ────────────────────────────────────────────────────────────────────
LOG_FILE="logs/run_${MODEL}.log"

echo ""
echo "════════════════════════════════════════════════════════════"
echo "  Model  : $MODEL"
echo "  Waves  : ${WAVES:-all}"
[[ -n "$WAVE_ARG" ]] && echo "  → expanded: $WAVE_LIST"
echo "  Log    : $LOG_FILE"
echo "════════════════════════════════════════════════════════════"
echo ""

VENV_PYTHON="$SCRIPT_DIR/.venv/Scripts/python"
[[ ! -f "$VENV_PYTHON" ]] && VENV_PYTHON="$SCRIPT_DIR/.venv/bin/python"

# shellcheck disable=SC2086
nohup env PYTHONIOENCODING=utf-8 PYTHONUTF8=1 "$VENV_PYTHON" -u run_all_waves_simulation.py \
    --model-type openai \
    --model "$MODEL" \
    $WAVE_ARG \
    > "$LOG_FILE" 2>&1 &

SIM_PID=$!
echo "✅  Running as PID $SIM_PID"
echo "   Monitor:  tail -f $LOG_FILE"
echo "   Stop:     kill $SIM_PID"
echo ""
echo "   Results  ->  data/results/$MODEL/"
echo "   Figures  ->  comparisons_demographics/$MODEL/  and  llm_vs_human/$MODEL/"
echo "   Summary  ->  logs/simulation_summary_$MODEL.json"
