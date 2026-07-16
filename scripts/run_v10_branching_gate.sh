#!/usr/bin/env bash
set -euo pipefail
PYTHON_BIN=${PYTHON_BIN:-python}
CLASS=${CLASS:-DBTT}
T_K=${T_K:-700}
OUT=${OUT:-runs/v10_0_1_branching_gate_${CLASS}_${T_K}K}
"$PYTHON_BIN" -m arrhenius_fracture.sharp_front_v10_1 \
  --mode 2d --material-class "$CLASS" --temperatures "$T_K" \
  --bulk-plasticity-mode tip_only --directional-j-mode root_signed \
  --steps "${STEPS:-12000}" --nx "${NX:-36}" --ny "${NY:-72}" \
  --dU "${DU:-2e-7}" --dt "${DT:-8.4}" --n-stagger "${N_STAGGER:-2}" \
  --tip-h-fine "${TIP_H_FINE:-1e-6}" --tip-ratio "${TIP_RATIO:-1.2}" \
  --da-phys "${DA_PHYS_M:-5e-6}" --target-crack-extension-um "${TARGET_EXT_UM:-20}" \
  --mpz-length-um "${MPZ_LENGTH_UM:-100}" --mpz-n-bins "${MPZ_N_BINS:-200}" \
  --wake-length-um "${WAKE_LENGTH_UM:-100}" --wake-shielding \
  --crystal-aniso --crystal-compete --crystal-branch --crystal-theta-deg "${THETA:-45}" \
  --crystal-material branchy --max-fronts "${MAX_FRONTS:-4}" \
  --j-decomposition cluster --adaptive-events --adaptive-event-target "${EVENT_TARGET:-0.15}" \
  --print-every "${PRINT_EVERY:-25}" --save-snapshots "${SAVE_SNAPSHOTS:-4}" \
  --out "$OUT"
