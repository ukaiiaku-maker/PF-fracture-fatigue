#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "$SCRIPT_DIR/.." && pwd)
cd "$REPO_ROOT"

PYTHON_BIN=${PYTHON_BIN:-python}
OUT=${OUT:-runs/v10_preflight}

# Direct execution of scripts/audit_matched_stress_v10.py otherwise places only
# scripts/ on sys.path.  Put this checkout first so the preflight is independent
# of editable-install state and cannot fall through to an older installation
# sharing the arrhenius_fracture package name.
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"

"$PYTHON_BIN" -m pytest -q
"$PYTHON_BIN" "$SCRIPT_DIR/audit_matched_stress_v10.py" \
  --T-K "${T_K:-700}" --hold-s "${HOLD_S:-100}" \
  --out "$OUT/matched_stress"
"$PYTHON_BIN" -m arrhenius_fracture.sharp_front_v10 \
  --mode 1d --material-class ceramic --temperatures "${T_K:-700}" \
  --Kdot 5 --Kmax 15 --dt 0.01 --n-advances 1 --out "$OUT/ceramic_1d"
