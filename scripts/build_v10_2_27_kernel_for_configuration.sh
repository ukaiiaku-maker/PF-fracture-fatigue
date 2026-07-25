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

: "${V10227_KERNEL_CONFIGURATION:?Resolver must set V10227_KERNEL_CONFIGURATION}"
: "${V10227_KERNEL_CACHE_DIR:?Resolver must set V10227_KERNEL_CACHE_DIR}"
: "${V10227_KERNEL_FAMILY_OUT:?Resolver must set V10227_KERNEL_FAMILY_OUT}"
: "${V10227_KERNEL_TARGET_EXTENSION_UM:?Resolver must set V10227_KERNEL_TARGET_EXTENSION_UM}"

CONFIG="$V10227_KERNEL_CONFIGURATION"
CACHE_DIR="$V10227_KERNEL_CACHE_DIR"
FAMILY_OUT="$V10227_KERNEL_FAMILY_OUT"
TARGET_EXT_UM="$V10227_KERNEL_TARGET_EXTENSION_UM"
REQUIRED_EXT_UM=${V10227_KERNEL_REQUIRED_MAX_EXTENSION_UM:-$TARGET_EXT_UM}
DA_PHYS_UM=${V10227_KERNEL_DA_PHYS_UM:-${DA_PHYS_UM:-5}}
EVENT_MINIMUM_FACTOR=${V10227_KERNEL_EVENT_MINIMUM_FACTOR:-${CLEAVAGE_EVENT_MIN_FACTOR:-0.5}}
EVENT_MAXIMUM_FACTOR=${V10227_KERNEL_EVENT_MAXIMUM_FACTOR:-${CLEAVAGE_EVENT_MAX_FACTOR:-4.0}}
MARGIN_EVENTS=${V10227_KERNEL_MARGIN_EVENTS:-${KERNEL_MARGIN_EVENTS:-1}}

read -r THETA BRANCHING_MODE MAX_FRONTS CAPTURE_TEMPERATURE_K < <(
  "$PYTHON_BIN" - "$CONFIG" <<'PY'
import json
import sys
payload = json.loads(open(sys.argv[1]).read())
temperature = payload.get("temperature_K") if payload.get("temperature_dependent_mechanics") else 700.0
print(payload["theta_deg"], payload["branching_mode"], payload["maximum_fronts"], temperature)
PY
)

if [[ "$BRANCHING_MODE" != single_front || "$MAX_FRONTS" != 1 ]]; then
  echo "ERROR: fixed-extension atlas builder cannot serve branching topology." >&2
  echo "Register a topology_cached or direct_fem builder for this configuration." >&2
  exit 4
fi

SNAPSHOT_ROOT=${KERNEL_SNAPSHOT_ROOT:-}
SNAPSHOT_ARCHIVE=${KERNEL_SNAPSHOT_ARCHIVE:-}
LOAD_ROOT=${KERNEL_LOAD_INVARIANCE_ROOT:-}
LOAD_ARCHIVE=${KERNEL_LOAD_INVARIANCE_ARCHIVE:-}

if [[ -n "$SNAPSHOT_ARCHIVE" || -n "$LOAD_ARCHIVE" ]]; then
  if [[ -z "$SNAPSHOT_ARCHIVE" || -z "$LOAD_ARCHIVE" ]]; then
    echo "ERROR: snapshot and load-invariance archives must be supplied together" >&2
    exit 2
  fi
  echo "OPTIONAL CACHE: using explicitly supplied portable mechanics archives" >&2
elif [[ -z "$SNAPSHOT_ROOT" ]]; then
  SNAPSHOT_ROOT="$CACHE_DIR/snapshots"
  if [[ -n "${KERNEL_CAPTURE_COMMAND:-}" ]]; then
    mkdir -p "$SNAPSHOT_ROOT"
    export V10227_KERNEL_CAPTURE_OUTROOT="$SNAPSHOT_ROOT"
    export V10227_KERNEL_CAPTURE_THETA_DEG="$THETA"
    export V10227_KERNEL_CAPTURE_TARGET_EXTENSION_UM="$TARGET_EXT_UM"
    export V10227_KERNEL_CAPTURE_REQUIRED_MAX_EXTENSION_UM="$REQUIRED_EXT_UM"
    echo "CAPTURE kernel states with registered override: $KERNEL_CAPTURE_COMMAND" >&2
    bash -lc "$KERNEL_CAPTURE_COMMAND"
  else
    echo "RECALCULATE frozen FEM states from the current mechanical configuration" >&2
    "$PYTHON_BIN" scripts/capture_v10_2_27_kernel_states_for_configuration.py \
      --mechanical-config "$CONFIG" \
      --snapshot-out "$SNAPSHOT_ROOT" \
      --run-out "$CACHE_DIR/capture" \
      --required-max-extension-um "$REQUIRED_EXT_UM" \
      --target-extension-um "$TARGET_EXT_UM" \
      --theta-deg "$THETA" \
      --capture-temperature-K "$CAPTURE_TEMPERATURE_K" \
      --force
  fi
fi

if [[ -z "$LOAD_ARCHIVE" && -z "$LOAD_ROOT" ]]; then
  if [[ -z "$SNAPSHOT_ROOT" ]]; then
    echo "ERROR: load-invariance recalculation requires captured snapshot states" >&2
    exit 2
  fi
  LOAD_ROOT="$CACHE_DIR/load_invariance"
  rm -rf "$LOAD_ROOT"
  mkdir -p "$LOAD_ROOT"
  found=0
  while IFS= read -r snapshot_json; do
    state_root=$(dirname "$snapshot_json")
    state=$(basename "$state_root")
    destination="$LOAD_ROOT/$state"
    found=$((found + 1))
    echo "RECALCULATE load invariance: $state" >&2
    "$PYTHON_BIN" scripts/evaluate_v10_2_14_active_load_invariance.py \
      --snapshot "$state_root" \
      --outroot "$destination" \
      --load-scales 0.5 1.0 1.5 \
      --magnitudes 0.25 0.50 \
      --linearity-tolerance 0.03 \
      --load-invariance-tolerance 0.05 \
      --minimum-residual-stiffness-fraction 0.001
  done < <(find "$SNAPSHOT_ROOT" -mindepth 2 -maxdepth 2 -name snapshot.json -type f | sort)
  if [[ "$found" -lt 2 ]]; then
    echo "ERROR: current-configuration capture produced fewer than two frozen states" >&2
    exit 2
  fi
fi

ARGS=(
  --mechanical-config "$CONFIG"
  --outroot "$CACHE_DIR"
  --family-out "$FAMILY_OUT"
  --target-extension-um "$TARGET_EXT_UM"
  --theta-deg "$THETA"
  --da-phys-um "$DA_PHYS_UM"
  --event-minimum-factor "$EVENT_MINIMUM_FACTOR"
  --event-maximum-factor "$EVENT_MAXIMUM_FACTOR"
  --margin-events "$MARGIN_EVENTS"
)
if [[ -n "$SNAPSHOT_ARCHIVE" ]]; then
  ARGS+=(--snapshot-archive "$SNAPSHOT_ARCHIVE")
else
  ARGS+=(--snapshot-root "$SNAPSHOT_ROOT")
fi
if [[ -n "$LOAD_ARCHIVE" ]]; then
  ARGS+=(--load-invariance-archive "$LOAD_ARCHIVE")
else
  ARGS+=(--load-invariance-root "$LOAD_ROOT")
fi

exec "$PYTHON_BIN" scripts/build_v10_2_27_kernel_from_mechanics_artifacts.py "${ARGS[@]}"
