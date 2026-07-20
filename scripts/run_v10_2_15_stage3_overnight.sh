#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$ROOT"

: "${LOAD_INVARIANCE_ROOT:?set LOAD_INVARIANCE_ROOT to the extracted E000/E200/E500/E800 directory}"
: "${ENGINE_CONFIG:?set ENGINE_CONFIG to the selected serialized 2-D engine configuration}"

EXPECTED_ENV=${EXPECTED_ENV:-arrhenius-sharp-front-v10}
if [[ "${CONDA_DEFAULT_ENV:-}" != "$EXPECTED_ENV" ]]; then
  echo "ERROR: activate conda environment $EXPECTED_ENV" >&2
  exit 2
fi

OUTROOT=${OUTROOT:-runs/v10_2_15_stage3_four_option_monotonic_500um_theta45_1x_v1}
MAX_JOBS=${MAX_JOBS:-2}
STEPS=${STEPS:-300000}
TARGET_EXT_UM=${TARGET_EXT_UM:-500}
THETA=${THETA:-45}
SKIP_FINISHED=${SKIP_FINISHED:-1}
MECHANICS_ROOT=${MECHANICS_ROOT:-$OUTROOT/mechanics}
FAMILY=${SIGNED_KERNEL_FAMILY_JSON:-$MECHANICS_ROOT/v10_2_14_active_only_campaign_family.json}

mkdir -p "$MECHANICS_ROOT" "$OUTROOT"

if [[ ! -f "$FAMILY" ]]; then
  echo "[stage3] assembling active-only kernel from completed E-state mechanics"
  python scripts/build_v10_2_14_campaign_ready_active_only_atlas.py \
    --load-invariance-root "$LOAD_INVARIANCE_ROOT" \
    --engine-config "$ENGINE_CONFIG" \
    --out "$FAMILY"
else
  echo "[stage3] reusing kernel family: $FAMILY"
fi

echo "[stage3] starting 40-case full 2-D campaign"
echo "[stage3] outroot=$OUTROOT max_jobs=$MAX_JOBS family=$FAMILY"

exec env \
  MODE=full \
  SIGNED_KERNEL_FAMILY_JSON="$FAMILY" \
  OUTROOT="$OUTROOT" \
  MAX_JOBS="$MAX_JOBS" \
  STEPS="$STEPS" \
  TARGET_EXT_UM="$TARGET_EXT_UM" \
  THETA="$THETA" \
  SKIP_FINISHED="$SKIP_FINISHED" \
  bash scripts/run_v10_2_15_stage3_monotonic_temperature_sweep.sh
