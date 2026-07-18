#!/usr/bin/env bash
set -euo pipefail

: "${KERNEL_FAMILY:?Set KERNEL_FAMILY to an authorized v10.2.6 family JSON}"
: "${DRIVE_FAMILY:?Set DRIVE_FAMILY to an authorized v10.2.7 drive-family JSON}"
: "${ENGINE_TEMPLATE:?Set ENGINE_TEMPLATE to a complete v10.2.6 engine-config JSON}"
: "${WEAKT_ANCHOR:?Set WEAKT_ANCHOR to the current FCC-like weakT manifest CSV}"
: "${CERAMIC_REFERENCE:?Set CERAMIC_REFERENCE to the frozen ceramic-like manifest CSV}"
: "${OUTROOT:?Set OUTROOT to a new versioned output directory}"

DBTT_ANCHOR_1=${DBTT_ANCHOR_1:-arrhenius_fracture/data/materials/fallback_dbtt/DBTT_A0002333.csv}
DBTT_ANCHOR_2=${DBTT_ANCHOR_2:-arrhenius_fracture/data/materials/fallback_dbtt/DBTT_A0003837.csv}
DBTT_ANCHOR_3=${DBTT_ANCHOR_3:-arrhenius_fracture/data/materials/fallback_dbtt/DBTT_A0002277.csv}
SAMPLES_DBTT=${SAMPLES_DBTT:-128}
SAMPLES_WEAKT=${SAMPLES_WEAKT:-128}
SEED=${SEED:-1201}
WORKERS=${WORKERS:-2}
PROMOTE_PER_CLASS=${PROMOTE_PER_CLASS:-12}
KDOT=${KDOT:-0.005}
KMAX=${KMAX:-80}
DK=${DK:-0.05}
TARGET_EXT_UM=${TARGET_EXT_UM:-50}

for path in \
  "$KERNEL_FAMILY" \
  "$DRIVE_FAMILY" \
  "$ENGINE_TEMPLATE" \
  "$DBTT_ANCHOR_1" \
  "$DBTT_ANCHOR_2" \
  "$DBTT_ANCHOR_3" \
  "$WEAKT_ANCHOR" \
  "$CERAMIC_REFERENCE"; do
  test -f "$path" || {
    echo "ERROR: required input is missing: $path" >&2
    exit 1
  }
done

test ! -e "$OUTROOT" || {
  echo "ERROR: output already exists: $OUTROOT" >&2
  exit 1
}

PARAMETER_CAMPAIGN=1 \
python scripts/run_v10_2_7_two_class_parameterization.py \
  --kernel-family "$KERNEL_FAMILY" \
  --drive-family "$DRIVE_FAMILY" \
  --engine-template "$ENGINE_TEMPLATE" \
  --dbtt-anchor "$DBTT_ANCHOR_1" \
  --dbtt-anchor "$DBTT_ANCHOR_2" \
  --dbtt-anchor "$DBTT_ANCHOR_3" \
  --weakt-anchor "$WEAKT_ANCHOR" \
  --ceramic-reference "$CERAMIC_REFERENCE" \
  --out "$OUTROOT" \
  --samples-dbtt "$SAMPLES_DBTT" \
  --samples-weakt "$SAMPLES_WEAKT" \
  --seed "$SEED" \
  --workers "$WORKERS" \
  --promote-per-class "$PROMOTE_PER_CLASS" \
  --Kdot "$KDOT" \
  --Kmax "$KMAX" \
  --dK "$DK" \
  --target-extension-um "$TARGET_EXT_UM"
