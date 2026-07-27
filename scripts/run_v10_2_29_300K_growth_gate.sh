#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$ROOT"

PYTHON_BIN=${PYTHON_BIN:-python}
OUTROOT=${OUTROOT:-runs/v10_2_29_300K_validation_v1}
R_RATIO=${R_RATIO:-0.1}

"$PYTHON_BIN" -m pytest -q tests/test_v10_2_29_fatigue_growth_extractor.py

OUTROOT="$OUTROOT" R_RATIO="$R_RATIO" \
  bash scripts/run_v10_2_29_300K_validation.sh

CYCLIC="$OUTROOT/cyclic_v10229_weakt_300K"
"$PYTHON_BIN" scripts/extract_v10_2_29_fatigue_growth.py \
  "$CYCLIC" \
  --temperature-K 300 \
  --R "$R_RATIO" \
  --require-event \
  > "$CYCLIC/fatigue_growth_extraction.log"

test -s "$CYCLIC/fatigue_event_growth_0300K.csv"
test -s "$CYCLIC/fatigue_event_growth_0300K.json"

echo "GROWTH_GATE_COMPLETE: $CYCLIC/fatigue_event_growth_0300K.csv"
