#!/usr/bin/env bash
set -euo pipefail

MODE=${MODE:-load-invariance}
ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$ROOT_DIR"

EXPECTED_CONDA_ENV=${CONDA_ENV:-arrhenius-sharp-front-v10}
ACTIVE_CONDA_ENV=${CONDA_DEFAULT_ENV:-}
if [[ "$ACTIVE_CONDA_ENV" != "$EXPECTED_CONDA_ENV" ]]; then
  echo "ERROR: activate conda environment '$EXPECTED_CONDA_ENV' before running." >&2
  echo "Current environment: '${ACTIVE_CONDA_ENV:-none}'" >&2
  exit 2
fi

# Keep direct script execution independent of a previous editable installation.
export PYTHONPATH="$ROOT_DIR${PYTHONPATH:+:$PYTHONPATH}"

case "$MODE" in
  load-invariance)
    : "${SNAPSHOT:?SNAPSHOT is required}"
    : "${OUTROOT:?OUTROOT is required}"
    LOAD_SCALES=${LOAD_SCALES:-"0.5 1.0 1.5"}
    MAGNITUDES=${MAGNITUDES:-"0.25 0.50"}
    LINEARITY_TOL=${LINEARITY_TOL:-0.03}
    LOAD_INVARIANCE_TOL=${LOAD_INVARIANCE_TOL:-0.05}
    MIN_RESIDUAL_STIFFNESS=${MIN_RESIDUAL_STIFFNESS:-1e-3}
    python scripts/evaluate_v10_2_14_active_load_invariance.py \
      --snapshot "$SNAPSHOT" \
      --outroot "$OUTROOT" \
      --load-scales $LOAD_SCALES \
      --magnitudes $MAGNITUDES \
      --linearity-tolerance "$LINEARITY_TOL" \
      --load-invariance-tolerance "$LOAD_INVARIANCE_TOL" \
      --minimum-residual-stiffness-fraction "$MIN_RESIDUAL_STIFFNESS"
    ;;
  *)
    echo "ERROR: unsupported MODE=$MODE" >&2
    echo "Supported modes: load-invariance" >&2
    exit 2
    ;;
esac
