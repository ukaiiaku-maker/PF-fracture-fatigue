#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$ROOT"

PYTHON_BIN=${PYTHON_BIN:-python}
CONDA_ENV=${CONDA_ENV:-arrhenius-sharp-front-v10}
if [[ "${CONDA_DEFAULT_ENV:-}" != "$CONDA_ENV" ]]; then
  echo "ERROR: activate conda environment '$CONDA_ENV'" >&2
  exit 2
fi

: "${DBTT_REFERENCE_ROOT:?Set DBTT_REFERENCE_ROOT to a monotonic DBTT campaign containing steps_TTTTK.csv}"
: "${PEAK_REFERENCE_ROOT:?Set PEAK_REFERENCE_ROOT to a monotonic peak campaign containing steps_TTTTK.csv}"

OUTROOT=${OUTROOT:-runs/v10_2_29_dbtt_peak_coupled_transient_v1}
TEMPERATURES=${TEMPERATURES:-"300 700 900 1000 1100"}
DELTA_K_FRACTIONS=${DELTA_K_FRACTIONS:-"0.20 0.35 0.50 0.65"}
HORIZONS=${HORIZONS:-1e10}
HAZARD_SEEDS=${HAZARD_SEEDS:-1720}
R_RATIO=${R_RATIO:-0.1}
FREQUENCY_HZ=${FREQUENCY_HZ:-1000}
TARGET_INCREMENT=${TARGET_INCREMENT:-10}
TARGET_DB=${TARGET_DB:-0.1}
STEPS=${STEPS:-10000}
FORCE=${FORCE:-0}
TRANSIENT_MIN_LOG_SPAN=${TRANSIENT_MIN_LOG_SPAN:-0.30}
TRANSIENT_MIN_STATE_RATIO=${TRANSIENT_MIN_STATE_RATIO:-0.05}
REQUIRE_CANDIDATE=${REQUIRE_CANDIDATE:-0}

mkdir -p "$OUTROOT"

run_family() {
  local label=$1
  local option=$2
  local reference=$3
  local temperature
  for temperature in $TEMPERATURES; do
    local case_root="$OUTROOT/$label/$(printf '%04d' "$temperature")K"
    mkdir -p "$case_root"
    MODE=transient \
    OUTROOT="$case_root" \
    REFERENCE_ROOT="$reference" \
    PARAMETER_OPTION="$option" \
    TEMPERATURE_K="$temperature" \
    R_RATIO="$R_RATIO" \
    FREQUENCY_HZ="$FREQUENCY_HZ" \
    DELTA_K_FRACTIONS="$DELTA_K_FRACTIONS" \
    HORIZONS="$HORIZONS" \
    HAZARD_SEEDS="$HAZARD_SEEDS" \
    TARGET_INCREMENT="$TARGET_INCREMENT" \
    TARGET_DB="$TARGET_DB" \
    TRANSIENT_MIN_LOG_SPAN="$TRANSIENT_MIN_LOG_SPAN" \
    TRANSIENT_MIN_STATE_RATIO="$TRANSIENT_MIN_STATE_RATIO" \
    STEPS="$STEPS" \
    FORCE="$FORCE" \
    bash scripts/run_v10_2_29_high_cycle_fatigue.sh \
      2>&1 | tee "$case_root/console.log"
  done
}

run_family \
  dbtt \
  v913_paper_dbtt01_0202500_persistent_sites \
  "$DBTT_REFERENCE_ROOT"

run_family \
  peak \
  v913_paper_peak01_0242980_persistent_sites \
  "$PEAK_REFERENCE_ROOT"

analysis_args=(
  "$OUTROOT"
  --minimum-log-lambda-span-decades "$TRANSIENT_MIN_LOG_SPAN"
  --minimum-state-target-ratio "$TRANSIENT_MIN_STATE_RATIO"
)
if [[ "$REQUIRE_CANDIDATE" == 1 ]]; then
  analysis_args+=(--require-candidate)
fi
"$PYTHON_BIN" scripts/analyze_v10_2_29_coupled_transient_screen.py "${analysis_args[@]}" \
  | tee "$OUTROOT/coupled_transient_screen.log"

echo "DBTT_PEAK_COUPLED_TRANSIENT_COMPLETE: $OUTROOT"
