# ─────────────────────────────────────────────────────────────────────────────
# config.sh  –  Edit this file, then run:  bash setup.sh
# ─────────────────────────────────────────────────────────────────────────────


# OpenAI API key.
# Get yours at https://platform.openai.com/api-keys
# Leave blank to be prompted at runtime, or set the OPENAI_API_KEY env var.
OPENAI_API_KEY=""  # scrubbed for replication package — set your own


# Model to use.
# Options: gpt-4o-mini  gpt-4o  gpt-4o-mini-2024-07-18  (or any valid OpenAI chat model)
MODEL="gpt-4o-mini"


# Which waves to simulate.
# Options:
#   all        – every available wave (26 27 29 32 34 36 41 42 43 45 49 50 54 82 92)
#   26-50      – inclusive range (only waves that actually exist are used)
#   26 50 82   – explicit list
#   26-45 82   – mix of range and list
WAVES="45 50 54 82 92"


# Skip pip install if packages are already installed.
# Set to true after the first successful run to start faster.
SKIP_INSTALL=true
