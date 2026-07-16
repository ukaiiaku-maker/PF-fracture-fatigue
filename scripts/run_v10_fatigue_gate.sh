#!/usr/bin/env bash
set -euo pipefail
PYTHON_BIN=${PYTHON_BIN:-python}
CLASS=${CLASS:-weakT}
T_K=${T_K:-700}
OUT=${OUT:-runs/v10_fatigue_gate_${CLASS}_${T_K}K}
"$PYTHON_BIN" -m arrhenius_fracture.fatigue_sharp_front \
  --material-class "$CLASS" --temperatures "$T_K" \
  --Kmax-MPa-sqrt-m "${KMAX:-17}" --R "${R:-0.1}" --frequency-Hz "${FREQ:-1000}" \
  --cycles-max "${CYCLES_MAX:-1e6}" --cycle-block-mode hazard_limited \
  --block-cycles "${BLOCK_CYCLES:-1e3}" --max-block-cycles "${MAX_BLOCK_CYCLES:-1e5}" \
  --target-dB "${TARGET_DB:-0.05}" --target-dN-emit "${TARGET_DN_EMIT:-0.25}" \
  --target-dN-mobile "${TARGET_DN_MOBILE:-0.25}" --n-advances "${N_ADVANCES:-1}" \
  --mpz-length-um "${MPZ_LENGTH_UM:-100}" --mpz-n-bins "${MPZ_N_BINS:-200}" \
  --wake-length-um "${WAKE_LENGTH_UM:-100}" --wake-shielding \
  --out "$OUT"
