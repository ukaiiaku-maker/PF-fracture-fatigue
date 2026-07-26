#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$ROOT"

PYTHON_BIN=${PYTHON_BIN:-python}
OUTROOT=${OUTROOT:-${1:-runs/v10_2_28_paper_four_class_1000um_theta30_varseed_base3621_v1}}
PLOT_ROOT=${PLOT_ROOT:-$OUTROOT/temperature_response}
TAIL_LENGTH_UM=${TAIL_LENGTH_UM:-200}
TAIL_FRACTION=${TAIL_FRACTION:-0.20}
POLICIES=${POLICIES:-"maximum fraction"}

case_dir() {
  case "$1" in
    maximum)
      printf '%s\n' "$PLOT_ROOT/maximum_${TAIL_LENGTH_UM}um_or_${TAIL_FRACTION}fraction"
      ;;
    length)
      printf '%s\n' "$PLOT_ROOT/length_${TAIL_LENGTH_UM}um"
      ;;
    fraction)
      printf '%s\n' "$PLOT_ROOT/fraction_${TAIL_FRACTION}"
      ;;
    *)
      echo "ERROR: unsupported tail policy: $1" >&2
      return 2
      ;;
  esac
}

for policy in $POLICIES; do
  plot_dir=$(case_dir "$policy")
  echo "PLOT_START: policy=$policy output=$plot_dir"
  "$PYTHON_BIN" scripts/plot_v10_2_28_four_class_KJ_vs_temperature.py \
    --outroot "$OUTROOT" \
    --plot-dir "$plot_dir" \
    --tail-length-um "$TAIL_LENGTH_UM" \
    --tail-fraction "$TAIL_FRACTION" \
    --tail-policy "$policy"
  echo "PLOT_COMPLETE: policy=$policy output=$plot_dir"
done

echo "ALL_PLOTS_COMPLETE: $PLOT_ROOT"
