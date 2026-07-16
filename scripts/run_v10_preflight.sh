#!/usr/bin/env bash
set -euo pipefail
PYTHON_BIN=${PYTHON_BIN:-python}
OUT=${OUT:-runs/v10_preflight}
"$PYTHON_BIN" -m pytest -q
"$PYTHON_BIN" scripts/audit_matched_stress_v10.py --T-K "${T_K:-700}" --hold-s "${HOLD_S:-100}" --out "$OUT/matched_stress"
"$PYTHON_BIN" -m arrhenius_fracture.sharp_front_v10 \
  --mode 1d --material-class ceramic --temperatures "${T_K:-700}" \
  --Kdot 5 --Kmax 15 --dt 0.01 --n-advances 1 --out "$OUT/ceramic_1d"
