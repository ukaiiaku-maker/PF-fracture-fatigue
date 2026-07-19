#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "$SCRIPT_DIR/.." && pwd)
cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1

STAGE=${STAGE:?Set STAGE to analytical, first-passage, rcurve, or 2d-plan}
OUTROOT=${OUTROOT:?Set OUTROOT to a new versioned stage directory}
WORKERS=${WORKERS:-2}
KDOT=${KDOT:-0.005}
KMAX=${KMAX:-80}
DK=${DK:-0.05}
SEED=${SEED:-1201}

export QD_QUALITY_RESERVE_FRACTION=${QD_QUALITY_RESERVE_FRACTION:-0.25}
export QD_QUALITY_WEIGHT=${QD_QUALITY_WEIGHT:-0.35}
export QD_PARAMETER_WEIGHT=${QD_PARAMETER_WEIGHT:-0.45}
export QD_RESPONSE_WEIGHT=${QD_RESPONSE_WEIGHT:-0.55}
export QD_POOL_FACTOR=${QD_POOL_FACTOR:-12}
export QD_PRESERVE_ANCHOR_LINEAGES=${QD_PRESERVE_ANCHOR_LINEAGES:-1}

case "$STAGE" in
  analytical)
    : "${KERNEL_FAMILY:?Set KERNEL_FAMILY to the authorized v10.2.6 family JSON}"
    : "${DRIVE_FAMILY:?Set DRIVE_FAMILY to the authorized v10.2.7 drive-family JSON}"
    : "${ENGINE_TEMPLATE:?Set ENGINE_TEMPLATE to the complete engine config JSON}"
    : "${WEAKT_ANCHOR:?Set WEAKT_ANCHOR to the current FCC-like weakT manifest CSV}"
    DBTT_ANCHOR_1=${DBTT_ANCHOR_1:-arrhenius_fracture/data/materials/fallback_dbtt/DBTT_A0002333.csv}
    DBTT_ANCHOR_2=${DBTT_ANCHOR_2:-arrhenius_fracture/data/materials/fallback_dbtt/DBTT_A0003837.csv}
    DBTT_ANCHOR_3=${DBTT_ANCHOR_3:-arrhenius_fracture/data/materials/fallback_dbtt/DBTT_A0002277.csv}
    SAMPLES_DBTT=${SAMPLES_DBTT:-4096}
    SAMPLES_WEAKT=${SAMPLES_WEAKT:-2048}
    PROMOTE_DBTT=${PROMOTE_DBTT:-256}
    PROMOTE_WEAKT=${PROMOTE_WEAKT:-256}
    ANALYTICAL_DK=${ANALYTICAL_DK:-0.20}
    required=("$KERNEL_FAMILY" "$DRIVE_FAMILY" "$ENGINE_TEMPLATE" "$WEAKT_ANCHOR" "$DBTT_ANCHOR_1" "$DBTT_ANCHOR_2" "$DBTT_ANCHOR_3")
    for path in "${required[@]}"; do test -f "$path" || { echo "ERROR: missing $path" >&2; exit 1; }; done
    args=(
      --stage analytical --kernel-family "$KERNEL_FAMILY" --drive-family "$DRIVE_FAMILY"
      --engine-template "$ENGINE_TEMPLATE" --dbtt-anchor "$DBTT_ANCHOR_1"
      --dbtt-anchor "$DBTT_ANCHOR_2" --dbtt-anchor "$DBTT_ANCHOR_3"
      --weakt-anchor "$WEAKT_ANCHOR" --samples-dbtt "$SAMPLES_DBTT"
      --samples-weakt "$SAMPLES_WEAKT" --promote-dbtt "$PROMOTE_DBTT"
      --promote-weakt "$PROMOTE_WEAKT" --analytical-dK "$ANALYTICAL_DK"
    )
    ;;
  first-passage)
    : "${KERNEL_FAMILY:?Set KERNEL_FAMILY}"
    : "${DRIVE_FAMILY:?Set DRIVE_FAMILY}"
    : "${ENGINE_TEMPLATE:?Set ENGINE_TEMPLATE}"
    : "${CANDIDATES:?Set CANDIDATES to analytical/promoted_to_first_passage.csv}"
    PROMOTE_DBTT=${PROMOTE_DBTT:-48}
    PROMOTE_WEAKT=${PROMOTE_WEAKT:-48}
    required=("$KERNEL_FAMILY" "$DRIVE_FAMILY" "$ENGINE_TEMPLATE" "$CANDIDATES")
    if [[ -n "${CERAMIC_REFERENCE:-}" ]]; then required+=("$CERAMIC_REFERENCE"); fi
    for path in "${required[@]}"; do test -f "$path" || { echo "ERROR: missing $path" >&2; exit 1; }; done
    args=(
      --stage first-passage --kernel-family "$KERNEL_FAMILY" --drive-family "$DRIVE_FAMILY"
      --engine-template "$ENGINE_TEMPLATE" --candidates "$CANDIDATES"
      --promote-dbtt "$PROMOTE_DBTT" --promote-weakt "$PROMOTE_WEAKT"
    )
    if [[ -n "${CERAMIC_REFERENCE:-}" ]]; then args+=(--ceramic-reference "$CERAMIC_REFERENCE"); fi
    ;;
  rcurve)
    : "${KERNEL_FAMILY:?Set KERNEL_FAMILY}"
    : "${DRIVE_FAMILY:?Set DRIVE_FAMILY}"
    : "${ENGINE_TEMPLATE:?Set ENGINE_TEMPLATE}"
    : "${CANDIDATES:?Set CANDIDATES to first_passage/promoted_to_rcurve.csv}"
    : "${CERAMIC_REFERENCE:?Set CERAMIC_REFERENCE to the frozen ceramic manifest}"
    PROMOTE_DBTT=${PROMOTE_DBTT:-4}
    PROMOTE_WEAKT=${PROMOTE_WEAKT:-4}
    TARGET_EXT_UM=${TARGET_EXT_UM:-50}
    TWO_D_EXT_UM=${TWO_D_EXT_UM:-100}
    required=("$KERNEL_FAMILY" "$DRIVE_FAMILY" "$ENGINE_TEMPLATE" "$CANDIDATES" "$CERAMIC_REFERENCE")
    for path in "${required[@]}"; do test -f "$path" || { echo "ERROR: missing $path" >&2; exit 1; }; done
    args=(
      --stage rcurve --kernel-family "$KERNEL_FAMILY" --drive-family "$DRIVE_FAMILY"
      --engine-template "$ENGINE_TEMPLATE" --candidates "$CANDIDATES"
      --ceramic-reference "$CERAMIC_REFERENCE" --promote-dbtt "$PROMOTE_DBTT"
      --promote-weakt "$PROMOTE_WEAKT" --target-extension-um "$TARGET_EXT_UM"
      --two-d-extension-um "$TWO_D_EXT_UM"
    )
    ;;
  2d-plan)
    : "${CANDIDATES:?Set CANDIDATES to rcurve/promoted_to_2d.csv}"
    TWO_D_EXT_UM=${TWO_D_EXT_UM:-100}
    test -f "$CANDIDATES" || { echo "ERROR: missing $CANDIDATES" >&2; exit 1; }
    args=(--stage 2d-plan --candidates "$CANDIDATES" --two-d-extension-um "$TWO_D_EXT_UM")
    ;;
  *)
    echo "ERROR: invalid STAGE=$STAGE" >&2
    exit 1
    ;;
esac

test ! -e "$OUTROOT" || {
  echo "ERROR: output already exists: $OUTROOT" >&2
  exit 1
}

PARAMETER_CAMPAIGN=1 \
python scripts/run_v10_2_10_staged_parameterization.py \
  "${args[@]}" \
  --out "$OUTROOT" --workers "$WORKERS" --seed "$SEED" \
  --Kdot "$KDOT" --Kmax "$KMAX" --dK "$DK"
