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

: "${LEGACY_LOAD_ROOT:?Set LEGACY_LOAD_ROOT to the passed E000/E200/E500/E800 load-invariance root}"
: "${NEW_SNAPSHOT_ROOT:?Set NEW_SNAPSHOT_ROOT to the captured E1000/E1200 snapshot root}"
: "${NEW_LOAD_ROOT:?Set NEW_LOAD_ROOT to a new E1000/E1200 load-invariance root}"
: "${NORMALIZATION:?Set NORMALIZATION to the existing mechanically derived normalization JSON}"

OUTROOT=${OUTROOT:-runs/v10_2_27_extended_signed_kernel_atlas_E1200_v1}
FAMILY_OUT=${FAMILY_OUT:-$OUTROOT/v10_2_27_active_only_campaign_family_E1200.json}
TARGET_EXT_UM=${TARGET_EXT_UM:-1000}
THETA=${THETA:-30}
DA_PHYS_UM=${DA_PHYS_UM:-5}
CLEAVAGE_EVENT_MIN_FACTOR=${CLEAVAGE_EVENT_MIN_FACTOR:-0.5}
CLEAVAGE_EVENT_MAX_FACTOR=${CLEAVAGE_EVENT_MAX_FACTOR:-4.0}
KERNEL_MARGIN_EVENTS=${KERNEL_MARGIN_EVENTS:-1}

LEGACY_LOAD_ROOT=$(cd "$LEGACY_LOAD_ROOT" && pwd)
NEW_SNAPSHOT_ROOT=$(cd "$NEW_SNAPSHOT_ROOT" && pwd)
mkdir -p "$NEW_LOAD_ROOT" "$OUTROOT" "$(dirname "$FAMILY_OUT")"
NEW_LOAD_ROOT=$(cd "$NEW_LOAD_ROOT" && pwd)
NORMALIZATION=$(cd "$(dirname "$NORMALIZATION")" && pwd)/$(basename "$NORMALIZATION")
FAMILY_OUT=$(cd "$(dirname "$FAMILY_OUT")" && pwd)/$(basename "$FAMILY_OUT")

[[ -f "$NORMALIZATION" ]] || {
  echo "ERROR: missing normalization: $NORMALIZATION" >&2
  exit 2
}
[[ ! -e "$FAMILY_OUT" ]] || {
  echo "ERROR: refusing to overwrite family: $FAMILY_OUT" >&2
  exit 2
}

for state in E000 E200 E500 E800; do
  report="$LEGACY_LOAD_ROOT/$state/frozen_geometry_load_invariance.json"
  response="$LEGACY_LOAD_ROOT/$state/active_station_responses_load_1.csv"
  audit="$LEGACY_LOAD_ROOT/$state/active_station_responses_load_1.audit.json"
  for required in "$report" "$response" "$audit"; do
    [[ -f "$required" ]] || {
      echo "ERROR: missing legacy mechanics artifact: $required" >&2
      exit 2
    }
  done
done

for state in E1000 E1200; do
  snapshot="$NEW_SNAPSHOT_ROOT/$state"
  [[ -f "$snapshot/snapshot.json" ]] || {
    echo "ERROR: missing new snapshot: $snapshot/snapshot.json" >&2
    exit 2
  }
  [[ -f "$snapshot/state_arrays.npz" ]] || {
    echo "ERROR: missing new snapshot arrays: $snapshot/state_arrays.npz" >&2
    exit 2
  }
  destination="$NEW_LOAD_ROOT/$state"
  if [[ -f "$destination/frozen_geometry_load_invariance.json" ]]; then
    echo "REUSE: $destination/frozen_geometry_load_invariance.json"
  else
    "$PYTHON_BIN" scripts/evaluate_v10_2_14_active_load_invariance.py \
      --snapshot "$snapshot" \
      --outroot "$destination" \
      --load-scales 0.5 1.0 1.5 \
      --magnitudes 0.25 0.50 \
      --linearity-tolerance 0.03 \
      --load-invariance-tolerance 0.05 \
      --minimum-residual-stiffness-fraction 0.001
  fi
done

RESPONSES=()
REPORTS=()
for state in E000 E200 E500 E800; do
  RESPONSES+=("$LEGACY_LOAD_ROOT/$state/active_station_responses_load_1.csv")
  REPORTS+=("$LEGACY_LOAD_ROOT/$state/frozen_geometry_load_invariance.json")
done
for state in E1000 E1200; do
  RESPONSES+=("$NEW_LOAD_ROOT/$state/active_station_responses_load_1.csv")
  REPORTS+=("$NEW_LOAD_ROOT/$state/frozen_geometry_load_invariance.json")
done

BUILD_ARGS=(
  --normalization "$NORMALIZATION"
  --out "$FAMILY_OUT"
  --minimum-max-extension-um 1200
)
for path in "${RESPONSES[@]}"; do
  BUILD_ARGS+=(--responses "$path")
done
for path in "${REPORTS[@]}"; do
  BUILD_ARGS+=(--load-invariance "$path")
done

"$PYTHON_BIN" scripts/build_v10_2_27_extended_active_only_atlas.py \
  "${BUILD_ARGS[@]}"

"$PYTHON_BIN" scripts/check_v10_2_27_signed_kernel_coverage.py \
  --family "$FAMILY_OUT" \
  --target-extension-um "$TARGET_EXT_UM" \
  --theta-deg "$THETA" \
  --da-phys-um "$DA_PHYS_UM" \
  --event-minimum-factor "$CLEAVAGE_EVENT_MIN_FACTOR" \
  --event-maximum-factor "$CLEAVAGE_EVENT_MAX_FACTOR" \
  --margin-events "$KERNEL_MARGIN_EVENTS" \
  --output "$OUTROOT/v10_2_27_signed_kernel_coverage_audit.json"

FAMILY_OUT="$FAMILY_OUT" OUTROOT="$OUTROOT" "$PYTHON_BIN" - <<'PY'
import hashlib
import json
import os
from pathlib import Path

family = Path(os.environ["FAMILY_OUT"]).resolve()
root = Path(os.environ["OUTROOT"]).resolve()

digest = hashlib.sha256(family.read_bytes()).hexdigest()
coverage = json.loads(
    (root / "v10_2_27_signed_kernel_coverage_audit.json").read_text()
)
payload = {
    "schema": "v10.2.27_extended_signed_kernel_workflow_v1",
    "family": str(family),
    "family_sha256": digest,
    "coverage_audit": str(
        (root / "v10_2_27_signed_kernel_coverage_audit.json").resolve()
    ),
    "coverage_passed": coverage["pass"],
    "atlas_max_crack_path_extension_um": (
        coverage["atlas_max_crack_path_extension_um"]
    ),
    "required_atlas_max_crack_path_extension_um": (
        coverage["required_atlas_max_crack_path_extension_um"]
    ),
}
(root / "v10_2_27_extended_signed_kernel_workflow.json").write_text(
    json.dumps(payload, indent=2, sort_keys=True) + "\n"
)
print(json.dumps(payload, indent=2, sort_keys=True))
PY

echo "Extended signed-kernel family accepted: $FAMILY_OUT"
