#!/usr/bin/env bash
set -euo pipefail
PYTHON_BIN=${PYTHON_BIN:-python}
OUTROOT=${OUTROOT:-runs/v10_three_class_700K_10um_v1}
T_K=${T_K:-700}
TARGET_EXT_UM=${TARGET_EXT_UM:-10}
STEPS=${STEPS:-12000}
for CLASS in ceramic weakT DBTT; do
  "$PYTHON_BIN" -m arrhenius_fracture.sharp_front_v10 \
    --mode 2d --material-class "$CLASS" --temperatures "$T_K" \
    --steps "$STEPS" --nx "${NX:-36}" --ny "${NY:-72}" \
    --dU "${DU:-2e-7}" --dt "${DT:-8.4}" --n-stagger "${N_STAGGER:-2}" \
    --tip-h-fine "${TIP_H_FINE:-1e-6}" --tip-ratio "${TIP_RATIO:-1.2}" \
    --da-phys "${DA_PHYS_M:-5e-6}" --target-crack-extension-um "$TARGET_EXT_UM" \
    --mpz-length-um "${MPZ_LENGTH_UM:-100}" --mpz-n-bins "${MPZ_N_BINS:-200}" \
    --wake-length-um "${WAKE_LENGTH_UM:-100}" --wake-n-bins "${WAKE_N_BINS:-0}" \
    --wake-shielding --wake-shield-projection "${WAKE_SHIELD_PROJECTION:-1}" \
    --crystal-aniso --crystal-compete --crystal-theta-deg "${THETA:-45}" \
    --crystal-material w --j-decomposition cluster \
    --max-fronts 1 --adaptive-events --adaptive-event-target "${EVENT_TARGET:-0.15}" \
    --print-every "${PRINT_EVERY:-25}" --save-snapshots "${SAVE_SNAPSHOTS:-3}" \
    --snapshot-by-crack-extension-um "${SNAPSHOT_BY_EXT_UM:-5}" \
    --out "$OUTROOT/$CLASS/T${T_K}_th${THETA:-45}"
done
