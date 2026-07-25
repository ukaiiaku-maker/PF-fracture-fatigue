#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$ROOT"

PYTHON_BIN=${PYTHON_BIN:-python}
OUTROOT=${OUTROOT:-runs/v10_2_27_paper_four_class_1000um_theta30_varseed_base3621_v1}
TARGET_EXT_UM=${TARGET_EXT_UM:-1000}
THETA=${THETA:-30}
DA_PHYS_UM=${DA_PHYS_UM:-5}
CLEAVAGE_EVENT_MIN_FACTOR=${CLEAVAGE_EVENT_MIN_FACTOR:-0.5}
CLEAVAGE_EVENT_MAX_FACTOR=${CLEAVAGE_EVENT_MAX_FACTOR:-4.0}
KERNEL_MARGIN_EVENTS=${KERNEL_MARGIN_EVENTS:-1}
BRANCHING_MODE=${BRANCHING_MODE:-single_front}
MAX_FRONTS=${MAX_FRONTS:-1}

mkdir -p "$OUTROOT"

FAMILY_JSON=$(
  PYTHON_BIN="$PYTHON_BIN" \
  FAMILY_JSON="${FAMILY_JSON:-}" \
  MECHANICAL_CONFIG="${MECHANICAL_CONFIG:-}" \
  MECHANICAL_PROFILE="${MECHANICAL_PROFILE:-}" \
  KERNEL_RESOLUTION_MODE="${KERNEL_RESOLUTION_MODE:-auto}" \
  KERNEL_BUILD_COMMAND="${KERNEL_BUILD_COMMAND:-}" \
  KERNEL_SNAPSHOT_ARCHIVE="${KERNEL_SNAPSHOT_ARCHIVE:-}" \
  KERNEL_LOAD_INVARIANCE_ARCHIVE="${KERNEL_LOAD_INVARIANCE_ARCHIVE:-}" \
  INITIAL_CRACK_LENGTH_UM="${INITIAL_CRACK_LENGTH_UM:-}" \
  KERNEL_INTERACTION_LENGTH_UM="${KERNEL_INTERACTION_LENGTH_UM:-}" \
  TEMPERATURE_DEPENDENT_MECHANICS="${TEMPERATURE_DEPENDENT_MECHANICS:-0}" \
  KERNEL_TEMPERATURE_K="${KERNEL_TEMPERATURE_K:-}" \
  TARGET_EXT_UM="$TARGET_EXT_UM" \
  THETA="$THETA" \
  BRANCHING_MODE="$BRANCHING_MODE" \
  MAX_FRONTS="$MAX_FRONTS" \
  DA_PHYS_UM="$DA_PHYS_UM" \
  CLEAVAGE_EVENT_MIN_FACTOR="$CLEAVAGE_EVENT_MIN_FACTOR" \
  CLEAVAGE_EVENT_MAX_FACTOR="$CLEAVAGE_EVENT_MAX_FACTOR" \
  KERNEL_MARGIN_EVENTS="$KERNEL_MARGIN_EVENTS" \
  bash scripts/resolve_v10_2_27_kernel_for_runner.sh
)
export FAMILY_JSON

echo "Resolved signed-kernel family: $FAMILY_JSON"

"$PYTHON_BIN" scripts/check_v10_2_27_signed_kernel_coverage.py \
  --family "$FAMILY_JSON" \
  --target-extension-um "$TARGET_EXT_UM" \
  --theta-deg "$THETA" \
  --da-phys-um "$DA_PHYS_UM" \
  --event-minimum-factor "$CLEAVAGE_EVENT_MIN_FACTOR" \
  --event-maximum-factor "$CLEAVAGE_EVENT_MAX_FACTOR" \
  --margin-events "$KERNEL_MARGIN_EVENTS" \
  --output "$OUTROOT/v10_2_27_signed_kernel_coverage_audit.json"

bash scripts/run_v10_2_27_paper_four_class_30deg_long_rcurves.sh "$@"

"$PYTHON_BIN" scripts/plot_v10_2_27_paper_four_class_K_vs_temperature.py \
  --outroot "$OUTROOT" \
  --target-extension-um "$TARGET_EXT_UM"

"$PYTHON_BIN" scripts/plot_v10_2_27_paper_four_class_J_energy_vs_temperature.py \
  --outroot "$OUTROOT" \
  --target-extension-um "$TARGET_EXT_UM"
