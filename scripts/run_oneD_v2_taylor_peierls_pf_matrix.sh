#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
OUTROOT=${OUTROOT:-/private/tmp/oneD-v2-taylor-peierls-pf-runs}
TRANSFER_ROOT=${TRANSFER_ROOT:-/private/tmp/oneD-v2-taylor-peierls-rcurve-search/analysis_outputs/oneD_v2_taylor_peierls_rcurve_search}
TARGET_EXTENSION_UM=${TARGET_EXTENSION_UM:-100}
mkdir -p "$OUTROOT"

run_pair() {
  local material=$1
  local temperature=$2
  local seed=$3
  local control_option variant_option
  if [[ "$material" == "Peak" ]]; then
    control_option=oneD_v2_Peak_control
    variant_option=oneD_v2_Peak_TP
  else
    control_option=oneD_v2_DBTT_control
    variant_option=oneD_v2_DBTT_TP
  fi

  local pids=()
  for option in "$control_option" "$variant_option"; do
    local case_out="$OUTROOT/$option/T${temperature}K_seed${seed}"
    mkdir -p "$case_out"
    env \
      TEMPERATURE_K="$temperature" \
      PARAMETER_OPTION="$option" \
      HAZARD_SEED="$seed" \
      CASE_OUT="$case_out" \
      TRANSFER_ROOT="$TRANSFER_ROOT" \
      TARGET_EXTENSION_UM="$TARGET_EXTENSION_UM" \
      bash "$ROOT/scripts/run_oneD_v2_terminal_pf_case.sh" \
      >"$case_out/run.log" 2>&1 &
    pids+=("$!")
  done
  local failure=0
  for pid in "${pids[@]}"; do
    wait "$pid" || failure=1
  done
  if [[ "$failure" -ne 0 ]]; then
    echo "PF_PAIR_FAILED material=$material temperature=$temperature" >&2
    return 1
  fi
  echo "PF_PAIR_COMPLETE material=$material temperature=$temperature target_um=$TARGET_EXTENSION_UM"
}

for temperature in 600 1000 1200; do
  run_pair Peak "$temperature" 8666
done
for temperature in 600 1100 1200; do
  run_pair DBTT "$temperature" 1008666
done

echo "PF_MATRIX_COMPLETE cases=12 maximum_concurrent_workers=2 target_um=$TARGET_EXTENSION_UM"
