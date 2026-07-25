#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$ROOT"

PYTHON_BIN=${PYTHON_BIN:-python}
THETA=${THETA:-30}
TARGET_EXT_UM=${TARGET_EXT_UM:-1000}
BRANCHING_MODE=${BRANCHING_MODE:-single_front}
MAX_FRONTS=${MAX_FRONTS:-1}
MECHANICAL_PROFILE=${MECHANICAL_PROFILE:-v10_2_27_current_single_front_frontfix}
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
  --event-minimum-factor "$CLEAVAGE_EVENT_MIN_FACTOR"
  --event-maximum-factor "$CLEAVAGE_EVENT_MAX_FACTOR"
  --margin-events "$KERNEL_MARGIN_EVENTS"
)

if [[ -n "${MECHANICAL_CONFIG:-}" ]]; then
  ARGS+=(--mechanical-config "$MECHANICAL_CONFIG")
  if [[ "${MECHANICAL_PROFILE_OVERRIDE:-0}" == 1 ]]; then
    ARGS+=(--mechanical-profile "$MECHANICAL_PROFILE")
  fi
  # Explicit override variables remain available for deliberate one-off changes,
  # but no registry/default value silently replaces a field from the JSON file.
  [[ -n "${KERNEL_PROCESS_ZONE_LENGTH_UM:-}" ]] && \
    ARGS+=(--process-zone-length-um "$KERNEL_PROCESS_ZONE_LENGTH_UM")
  [[ -n "${KERNEL_PROCESS_ZONE_BINS:-}" ]] && \
    ARGS+=(--process-zone-bins "$KERNEL_PROCESS_ZONE_BINS")
  [[ -n "${KERNEL_MESH_NX:-}" ]] && ARGS+=(--mesh-nx "$KERNEL_MESH_NX")
  [[ -n "${KERNEL_MESH_NY:-}" ]] && ARGS+=(--mesh-ny "$KERNEL_MESH_NY")
  [[ -n "${KERNEL_TIP_H_FINE_UM:-}" ]] && \
    ARGS+=(--tip-h-fine-um "$KERNEL_TIP_H_FINE_UM")
  [[ -n "${KERNEL_TIP_RATIO:-}" ]] && ARGS+=(--tip-ratio "$KERNEL_TIP_RATIO")
  [[ -n "${KERNEL_ATLAS_ANCHOR_SPACING_UM:-}" ]] && \
    ARGS+=(--atlas-anchor-spacing-um "$KERNEL_ATLAS_ANCHOR_SPACING_UM")
  [[ -n "${KERNEL_MIN_ELEMENTS_PER_PZ:-}" ]] && \
    ARGS+=(--minimum-elements-per-process-zone "$KERNEL_MIN_ELEMENTS_PER_PZ")
  [[ -n "${KERNEL_INTERACTION_LENGTH_UM:-}" ]] && \
    ARGS+=(--interaction-length-um "$KERNEL_INTERACTION_LENGTH_UM")
  [[ -n "${DA_PHYS_UM:-}" ]] && ARGS+=(--da-phys-um "$DA_PHYS_UM")
else
  SOURCE_REGISTRY=${KERNEL_PARAMETER_REGISTRY:-$ROOT/arrhenius_fracture/data/materials/v10_2_27_v913_four_class_paper_registry.csv}
  read -r REGISTRY_PZ_UM REGISTRY_PZ_BINS < <(
    "$PYTHON_BIN" - "$SOURCE_REGISTRY" <<'PY'
import csv
import sys
from pathlib import Path
path = Path(sys.argv[1])
with path.open(newline="") as stream:
    rows = list(csv.DictReader(stream))
lengths = {float(row["L_pz_um_recommended"]) for row in rows}
bins = {int(round(float(row["n_bins_recommended"]))) for row in rows}
if len(lengths) != 1 or len(bins) != 1:
    raise SystemExit(
        "kernel resolution requires one common mechanical process-zone geometry; "
        f"found lengths={sorted(lengths)}, bins={sorted(bins)}"
    )
print(next(iter(lengths)), next(iter(bins)))
PY
  )
  ARGS+=(
    --mechanical-profile "$MECHANICAL_PROFILE"
    --da-phys-um "$DA_PHYS_UM"
    --process-zone-length-um "${KERNEL_PROCESS_ZONE_LENGTH_UM:-$REGISTRY_PZ_UM}"
    --process-zone-bins "${KERNEL_PROCESS_ZONE_BINS:-$REGISTRY_PZ_BINS}"
    --mesh-nx "${KERNEL_MESH_NX:-36}"
    --mesh-ny "${KERNEL_MESH_NY:-72}"
    --tip-h-fine-um "${KERNEL_TIP_H_FINE_UM:-1}"
    --tip-ratio "${KERNEL_TIP_RATIO:-1.20}"
    --atlas-anchor-spacing-um "${KERNEL_ATLAS_ANCHOR_SPACING_UM:-200}"
    --minimum-elements-per-process-zone "${KERNEL_MIN_ELEMENTS_PER_PZ:-3}"
    --interaction-length-um "${KERNEL_INTERACTION_LENGTH_UM:-2}"
  )
fi

# FAMILY_JSON is commonly inherited from an older campaign shell. It is not an
# input to normal automatic resolution. Use an override only when explicitly
# requested; otherwise the current configuration is resolved/recalculated.
if [[ -n "${FAMILY_JSON:-}" ]]; then
  if [[ "${KERNEL_USE_FAMILY_OVERRIDE:-0}" == 1 || "${KERNEL_STRICT_FAMILY_OVERRIDE:-0}" == 1 ]]; then
    if [[ ! -f "$FAMILY_JSON" ]]; then
      echo "ERROR: requested family override does not exist: $FAMILY_JSON" >&2
      exit 2
    fi
    ARGS+=(--family-override "$FAMILY_JSON")
  else
    echo "Ignoring inherited FAMILY_JSON; resolving from the current mechanical configuration" >&2
  fi
fi
if [[ -n "${KERNEL_BUILD_COMMAND:-}" ]]; then
  ARGS+=(--builder-command "$KERNEL_BUILD_COMMAND")
fi
# Archives are optional accelerators only; normal use leaves both unset.
if [[ -n "${KERNEL_SNAPSHOT_ARCHIVE:-}" ]]; then
  ARGS+=(--snapshot-archive "$KERNEL_SNAPSHOT_ARCHIVE")
fi
if [[ -n "${KERNEL_LOAD_INVARIANCE_ARCHIVE:-}" ]]; then
  ARGS+=(--load-invariance-archive "$KERNEL_LOAD_INVARIANCE_ARCHIVE")
fi
if [[ -n "${SPECIMEN_LENGTH_X_UM:-}" ]]; then
  ARGS+=(--specimen-length-x-um "$SPECIMEN_LENGTH_X_UM")
fi
if [[ -n "${SPECIMEN_LENGTH_Y_UM:-}" ]]; then
  ARGS+=(--specimen-length-y-um "$SPECIMEN_LENGTH_Y_UM")
fi
if [[ -n "${INITIAL_CRACK_LENGTH_UM:-}" ]]; then
  ARGS+=(--initial-crack-length-um "$INITIAL_CRACK_LENGTH_UM")
fi
if [[ -n "${NOTCH_HALF_THICKNESS_UM:-}" ]]; then
  ARGS+=(--notch-half-thickness-um "$NOTCH_HALF_THICKNESS_UM")
fi
if [[ "${TEMPERATURE_DEPENDENT_MECHANICS:-0}" == 1 ]]; then
  : "${KERNEL_TEMPERATURE_K:?Set KERNEL_TEMPERATURE_K when TEMPERATURE_DEPENDENT_MECHANICS=1}"
  ARGS+=(--temperature-dependent-mechanics --temperature-K "$KERNEL_TEMPERATURE_K")
fi

exec "$PYTHON_BIN" scripts/ensure_v10_2_27_signed_kernel.py "${ARGS[@]}"
