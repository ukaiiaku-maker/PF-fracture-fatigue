#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$ROOT"

PYTHON_BIN=${PYTHON_BIN:-python}
THETA=${THETA:-30}
TARGET_EXT_UM=${TARGET_EXT_UM:-1000}
BRANCHING_MODE=${BRANCHING_MODE:-single_front}
MAX_FRONTS=${MAX_FRONTS:-1}
MECHANICAL_PROFILE=${MECHANICAL_PROFILE:-}
KERNEL_RESOLUTION_MODE=${KERNEL_RESOLUTION_MODE:-auto}
DA_PHYS_UM=${DA_PHYS_UM:-5}
CLEAVAGE_EVENT_MIN_FACTOR=${CLEAVAGE_EVENT_MIN_FACTOR:-0.5}
CLEAVAGE_EVENT_MAX_FACTOR=${CLEAVAGE_EVENT_MAX_FACTOR:-4.0}
KERNEL_MARGIN_EVENTS=${KERNEL_MARGIN_EVENTS:-1}

ARGS=(
  --theta-deg "$THETA"
  --target-extension-um "$TARGET_EXT_UM"
  --branching-mode "$BRANCHING_MODE"
  --maximum-fronts "$MAX_FRONTS"
  --mode "$KERNEL_RESOLUTION_MODE"
  --da-phys-um "$DA_PHYS_UM"
  --event-minimum-factor "$CLEAVAGE_EVENT_MIN_FACTOR"
  --event-maximum-factor "$CLEAVAGE_EVENT_MAX_FACTOR"
  --margin-events "$KERNEL_MARGIN_EVENTS"
)

if [[ -n "${MECHANICAL_CONFIG:-}" && "${MECHANICAL_PROFILE_OVERRIDE:-0}" != "1" ]]; then
  : # Preserve the profile embedded in the explicit mechanical configuration.
elif [[ -n "$MECHANICAL_PROFILE" ]]; then
  ARGS+=(--mechanical-profile "$MECHANICAL_PROFILE")
else
  ARGS+=(--mechanical-profile v10_2_27_default_single_front_frontfix)
fi

if [[ -n "${FAMILY_JSON:-}" ]]; then
  if [[ -f "$FAMILY_JSON" ]]; then
    ARGS+=(--family-override "$FAMILY_JSON")
  elif [[ "${KERNEL_STRICT_FAMILY_OVERRIDE:-0}" == "1" ]]; then
    echo "ERROR: explicit FAMILY_JSON does not exist: $FAMILY_JSON" >&2
    exit 2
  else
    echo "Ignoring stale FAMILY_JSON and resolving from the mechanical configuration: $FAMILY_JSON" >&2
  fi
fi
if [[ -n "${MECHANICAL_CONFIG:-}" ]]; then
  ARGS+=(--mechanical-config "$MECHANICAL_CONFIG")
fi
if [[ -n "${KERNEL_BUILD_COMMAND:-}" ]]; then
  ARGS+=(--builder-command "$KERNEL_BUILD_COMMAND")
fi
if [[ -n "${KERNEL_SNAPSHOT_ARCHIVE:-}" ]]; then
  ARGS+=(--snapshot-archive "$KERNEL_SNAPSHOT_ARCHIVE")
fi
if [[ -n "${KERNEL_LOAD_INVARIANCE_ARCHIVE:-}" ]]; then
  ARGS+=(--load-invariance-archive "$KERNEL_LOAD_INVARIANCE_ARCHIVE")
fi
if [[ -n "${INITIAL_CRACK_LENGTH_UM:-}" ]]; then
  ARGS+=(--initial-crack-length-um "$INITIAL_CRACK_LENGTH_UM")
fi
if [[ -n "${KERNEL_INTERACTION_LENGTH_UM:-}" ]]; then
  ARGS+=(--interaction-length-um "$KERNEL_INTERACTION_LENGTH_UM")
fi
if [[ "${TEMPERATURE_DEPENDENT_MECHANICS:-0}" == "1" ]]; then
  : "${KERNEL_TEMPERATURE_K:?Set KERNEL_TEMPERATURE_K when TEMPERATURE_DEPENDENT_MECHANICS=1}"
  ARGS+=(--temperature-dependent-mechanics --temperature-K "$KERNEL_TEMPERATURE_K")
fi

exec "$PYTHON_BIN" scripts/ensure_v10_2_27_signed_kernel.py "${ARGS[@]}"
