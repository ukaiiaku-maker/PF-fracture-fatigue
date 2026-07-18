#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "$SCRIPT_DIR/.." && pwd)
cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1

PYTHON_BIN=${PYTHON_BIN:-python}
OUTROOT=${OUTROOT:-runs/v10_2_3_temperature_fracture_uncapped_smoke}
CLASS=${CLASS:-DBTT}
TEMPERATURES=${TEMPERATURES:-"300 700 1200"}
THETA=${THETA:-45}
TARGET_EXT_UM=${TARGET_EXT_UM:-25}
STEPS=${STEPS:-4000}
NX=${NX:-48}
NY=${NY:-96}
TIP_H_FINE=${TIP_H_FINE:-5e-7}
TIP_RATIO=${TIP_RATIO:-1.2}
DU=${DU:-2e-7}
DT=${DT:-8.4}
DA_PHYS_M=${DA_PHYS_M:-5e-6}
MPZ_LENGTH_UM=${MPZ_LENGTH_UM:-100}
MPZ_N_BINS=${MPZ_N_BINS:-200}
WAKE_LENGTH_UM=${WAKE_LENGTH_UM:-100}
WAKE_N_BINS=${WAKE_N_BINS:-0}
FORCE=${FORCE:-1}

if [[ "$FORCE" == 1 ]]; then rm -rf "$OUTROOT"; fi
mkdir -p "$OUTROOT"

echo "v10.2.3 monotonic temperature-fracture smoke"
echo "  class=$CLASS temperatures=$TEMPERATURES theta=$THETA target_ext_um=$TARGET_EXT_UM"
echo "  constitutive K-shield cap: OFF in shared CampaignCalibratedTipEngine"

for T_K in $TEMPERATURES; do
  OUT="$OUTROOT/T${T_K}K"
  "$PYTHON_BIN" -u -m arrhenius_fracture.sharp_front_v10_2_3 \
    --mode 2d \
    --material-class "$CLASS" \
    --temperatures "$T_K" \
    --bulk-plasticity-mode tip_only \
    --directional-j-mode root_signed \
    --tip-kinetics-mode moving_velocity \
    --tip-source-model continuum \
    --tip-plasticity \
    --active-shielding \
    --signed-active-shielding \
    --mobile-shield-fraction 1 \
    --no-wake-shielding \
    --crack-backend sharp_wake \
    --crystal-aniso \
    --crystal-compete \
    --crystal-theta-deg "$THETA" \
    --crystal-material w \
    --j-decomposition cluster \
    --max-fronts 1 \
    --steps "$STEPS" \
    --nx "$NX" --ny "$NY" \
    --tip-h-fine "$TIP_H_FINE" \
    --tip-ratio "$TIP_RATIO" \
    --dU "$DU" --dt "$DT" --n-stagger 2 \
    --da-phys "$DA_PHYS_M" \
    --target-crack-extension-um "$TARGET_EXT_UM" \
    --mpz-length-um "$MPZ_LENGTH_UM" \
    --mpz-n-bins "$MPZ_N_BINS" \
    --wake-length-um "$WAKE_LENGTH_UM" \
    --wake-n-bins "$WAKE_N_BINS" \
    --adaptive-events \
    --adaptive-event-target 0.2 \
    --print-every 100 \
    --save-snapshots 0 \
    --no-plots \
    --out "$OUT" \
    "$@"

  "$PYTHON_BIN" scripts/audit_v10_2_3_shared_uncapped_outputs.py "$OUT"
done

echo "v10.2.3 monotonic temperature-fracture smoke complete: $OUTROOT"
