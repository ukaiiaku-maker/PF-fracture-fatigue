#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python}"
OUTROOT="${OUTROOT:-runs/v10_2_3_temperature_fracture_uncapped_smoke}"
MATERIAL="${MATERIAL:-DBTT}"
TEMPERATURES="${TEMPERATURES:-300 700 1200}"
THETA_DEG="${THETA_DEG:-45}"
TARGET_EXT_UM="${TARGET_EXT_UM:-25}"

mkdir -p "$OUTROOT"

echo "v10.2.3 monotonic temperature-fracture smoke"
echo "  material=$MATERIAL temperatures=$TEMPERATURES theta=$THETA_DEG target_ext_um=$TARGET_EXT_UM"
echo "  constitutive K-shield cap: OFF in shared CampaignCalibratedTipEngine"

for T_K in $TEMPERATURES; do
  OUT="$OUTROOT/T${T_K}K"
  "$PYTHON_BIN" -m arrhenius_fracture.sharp_front_v10_2_3 \
    --material-class "$MATERIAL" \
    --temperatures "$T_K" \
    --theta-deg "$THETA_DEG" \
    --target-ext-um "$TARGET_EXT_UM" \
    --out "$OUT" \
    "$@"

  "$PYTHON_BIN" scripts/audit_v10_2_3_shared_uncapped_outputs.py "$OUT"
done

echo "v10.2.3 monotonic temperature-fracture smoke complete: $OUTROOT"
