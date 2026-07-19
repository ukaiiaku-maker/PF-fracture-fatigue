#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "$SCRIPT_DIR/.." && pwd)
cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1

MODE=${MODE:?Set MODE to plan, preflight, build-review, authorize, or readiness}
PYTHON_BIN=${PYTHON_BIN:-python}

case "$MODE" in
  plan)
    : "${STATE_TABLE:?Set STATE_TABLE to the physical state request CSV}"
    : "${OUTROOT:?Set OUTROOT to a new versioned directory}"
    ACTIVE_BINS=${ACTIVE_BINS:?Set ACTIVE_BINS}
    WAKE_BINS=${WAKE_BINS:?Set WAKE_BINS}
    N_SYSTEMS=${N_SYSTEMS:-2}
    PERTURBATION_MAGNITUDES=${PERTURBATION_MAGNITUDES:-"0.25 0.50"}
    exec "$PYTHON_BIN" scripts/run_v10_2_10_physical_artifact_workflow.py plan \
      --states "$STATE_TABLE" --outroot "$OUTROOT" \
      --n-systems "$N_SYSTEMS" --active-bins "$ACTIVE_BINS" \
      --wake-bins "$WAKE_BINS" \
      --perturbation-magnitudes "$PERTURBATION_MAGNITUDES"
    ;;
  preflight)
    : "${SIGNED_RESPONSES:?Set SIGNED_RESPONSES}"
    : "${TENSOR_RESPONSES:?Set TENSOR_RESPONSES}"
    : "${NORMALIZATION:?Set NORMALIZATION}"
    : "${OUT:?Set OUT}"
    exec "$PYTHON_BIN" scripts/run_v10_2_10_physical_artifact_workflow.py preflight \
      --signed-responses "$SIGNED_RESPONSES" \
      --tensor-responses "$TENSOR_RESPONSES" \
      --normalization "$NORMALIZATION" --out "$OUT"
    ;;
  build-review|authorize)
    : "${SIGNED_RESPONSES:?Set SIGNED_RESPONSES}"
    : "${TENSOR_RESPONSES:?Set TENSOR_RESPONSES}"
    : "${NORMALIZATION:?Set NORMALIZATION}"
    : "${OUTROOT:?Set OUTROOT to a new versioned directory}"
    args=(
      "$MODE" --signed-responses "$SIGNED_RESPONSES"
      --tensor-responses "$TENSOR_RESPONSES"
      --normalization "$NORMALIZATION" --outroot "$OUTROOT"
    )
    if [[ "$MODE" == "authorize" ]]; then
      : "${INDEPENDENT_REVIEW:?Set INDEPENDENT_REVIEW to the signed review JSON}"
      args+=(--independent-review "$INDEPENDENT_REVIEW")
    fi
    exec "$PYTHON_BIN" scripts/run_v10_2_10_physical_artifact_workflow.py "${args[@]}"
    ;;
  readiness)
    : "${KERNEL_FAMILY:?Set KERNEL_FAMILY}"
    : "${DRIVE_FAMILY:?Set DRIVE_FAMILY}"
    : "${ENGINE_TEMPLATE:?Set ENGINE_TEMPLATE}"
    : "${OUT:?Set OUT}"
    exec "$PYTHON_BIN" scripts/run_v10_2_10_physical_artifact_workflow.py readiness \
      --kernel-family "$KERNEL_FAMILY" --drive-family "$DRIVE_FAMILY" \
      --engine-template "$ENGINE_TEMPLATE" --out "$OUT" --require-ready
    ;;
  *)
    echo "ERROR: invalid MODE=$MODE" >&2
    exit 2
    ;;
esac
