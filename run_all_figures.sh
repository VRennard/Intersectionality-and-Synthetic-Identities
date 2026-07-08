#!/bin/bash
# Regenerate every paper figure from the packaged canonical artifacts.
# Outputs (.png/.pdf) land next to the scripts: paper_figs/ and verification/.
# Requires: python3 with numpy, pandas, matplotlib (see requirements.txt).
set -uo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
PY=${PY:-python3}

FAIL=0
for f in "$HERE"/paper_figs/fig_*.py; do
  case "$(basename "$f")" in fig_redesigns.py) continue;; esac  # exploratory, not in paper
  echo "== $(basename "$f")"
  if ! "$PY" "$f"; then echo "   FAILED: $(basename "$f")"; FAIL=1; fi
done
echo "== fig_spine.py (verification/)"
if ! "$PY" "$HERE"/verification/fig_spine.py; then echo "   FAILED: fig_spine.py"; FAIL=1; fi

exit $FAIL
