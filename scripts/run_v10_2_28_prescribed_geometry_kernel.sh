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

: "${V10227_KERNEL_CONFIGURATION:?resolver must set V10227_KERNEL_CONFIGURATION}"
: "${V10227_KERNEL_CACHE_DIR:?resolver must set V10227_KERNEL_CACHE_DIR}"
: "${V10227_KERNEL_FAMILY_OUT:?resolver must set V10227_KERNEL_FAMILY_OUT}"
: "${V10227_KERNEL_TARGET_EXTENSION_UM:?resolver must set V10227_KERNEL_TARGET_EXTENSION_UM}"
: "${V10227_KERNEL_REQUIRED_MAX_EXTENSION_UM:?resolver must set V10227_KERNEL_REQUIRED_MAX_EXTENSION_UM}"

CONFIG="$V10227_KERNEL_CONFIGURATION"
CACHE_DIR="$V10227_KERNEL_CACHE_DIR"
FAMILY_OUT="$V10227_KERNEL_FAMILY_OUT"
TARGET_EXT_UM="$V10227_KERNEL_TARGET_EXTENSION_UM"
REQUIRED_EXT_UM="$V10227_KERNEL_REQUIRED_MAX_EXTENSION_UM"
DA_PHYS_UM=${V10227_KERNEL_DA_PHYS_UM:-${DA_PHYS_UM:-5}}
EVENT_MINIMUM_FACTOR=${V10227_KERNEL_EVENT_MINIMUM_FACTOR:-${CLEAVAGE_EVENT_MIN_FACTOR:-0.5}}
EVENT_MAXIMUM_FACTOR=${V10227_KERNEL_EVENT_MAXIMUM_FACTOR:-${CLEAVAGE_EVENT_MAX_FACTOR:-4.0}}
MARGIN_EVENTS=${V10227_KERNEL_MARGIN_EVENTS:-${KERNEL_MARGIN_EVENTS:-1}}
REFERENCE_OPENING_STRAIN=${KERNEL_REFERENCE_OPENING_STRAIN:-1e-5}

read -r THETA BRANCHING_MODE MAX_FRONTS < <(
  "$PYTHON_BIN" - "$CONFIG" <<'PY'
import json
import sys
p = json.loads(open(sys.argv[1]).read())
print(p["theta_deg"], p["branching_mode"], p["maximum_fronts"])
PY
)

if [[ "$BRANCHING_MODE" != single_front || "$MAX_FRONTS" != 1 ]]; then
  echo "ERROR: the v10.2.28 prescribed-geometry provider is single-front only." >&2
  echo "A branch-topology-specific direct FEM provider must be registered separately." >&2
  exit 4
fi

for forbidden in \
  KERNEL_CAPTURE_SEED_FAMILY \
  KERNEL_CAPTURE_PARAMETER_OPTION \
  KERNEL_CAPTURE_HAZARD_SEED \
  KERNEL_SNAPSHOT_ROOT \
  KERNEL_SNAPSHOT_ARCHIVE \
  KERNEL_LOAD_INVARIANCE_ROOT \
  KERNEL_LOAD_INVARIANCE_ARCHIVE; do
  if [[ -n "${!forbidden:-}" ]]; then
    echo "ERROR: $forbidden is forbidden for the direct prescribed-geometry provider." >&2
    exit 5
  fi
done

exec "$PYTHON_BIN" scripts/build_v10_2_28_prescribed_geometry_kernel.py \
  --mechanical-config "$CONFIG" \
  --outroot "$CACHE_DIR" \
  --family-out "$FAMILY_OUT" \
  --required-max-extension-um "$REQUIRED_EXT_UM" \
  --target-extension-um "$TARGET_EXT_UM" \
  --theta-deg "$THETA" \
  --da-phys-um "$DA_PHYS_UM" \
  --event-minimum-factor "$EVENT_MINIMUM_FACTOR" \
  --event-maximum-factor "$EVENT_MAXIMUM_FACTOR" \
  --margin-events "$MARGIN_EVENTS" \
  --reference-opening-strain "$REFERENCE_OPENING_STRAIN"
