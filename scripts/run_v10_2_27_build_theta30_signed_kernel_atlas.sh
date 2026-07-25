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

: "${SNAPSHOT_ROOT:?Set SNAPSHOT_ROOT to the six-state theta30 capture root}"
: "${LOAD_ROOT:?Set LOAD_ROOT to a new six-state load-invariance root}"
: "${NORMALIZATION:?Set NORMALIZATION to the mechanically derived normalization JSON}"

EXPECTED_THETA_DEG=${EXPECTED_THETA_DEG:-30}
EXPECTED_TEMPERATURE_K=${EXPECTED_TEMPERATURE_K:-700}
STATES=${STATES:-"E000 E200 E500 E800 E1000 E1200"}
OUTROOT=${OUTROOT:-runs/v10_2_27_theta30_signed_kernel_atlas_E1200_v1}
FAMILY_OUT=${FAMILY_OUT:-$OUTROOT/v10_2_27_theta30_active_only_campaign_family_E1200.json}
TARGET_EXT_UM=${TARGET_EXT_UM:-1000}
DA_PHYS_UM=${DA_PHYS_UM:-5}
CLEAVAGE_EVENT_MIN_FACTOR=${CLEAVAGE_EVENT_MIN_FACTOR:-0.5}
CLEAVAGE_EVENT_MAX_FACTOR=${CLEAVAGE_EVENT_MAX_FACTOR:-4.0}
KERNEL_MARGIN_EVENTS=${KERNEL_MARGIN_EVENTS:-1}

SNAPSHOT_ROOT=$(cd "$SNAPSHOT_ROOT" && pwd)
mkdir -p "$LOAD_ROOT" "$OUTROOT" "$(dirname "$FAMILY_OUT")"
LOAD_ROOT=$(cd "$LOAD_ROOT" && pwd)
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

SNAPSHOT_ROOT="$SNAPSHOT_ROOT" STATES="$STATES" \
EXPECTED_THETA_DEG="$EXPECTED_THETA_DEG" \
EXPECTED_TEMPERATURE_K="$EXPECTED_TEMPERATURE_K" \
"$PYTHON_BIN" - <<'PY'
import json
import math
import os
from pathlib import Path

root = Path(os.environ["SNAPSHOT_ROOT"]).resolve()
states = os.environ["STATES"].split()
expected_theta = float(os.environ["EXPECTED_THETA_DEG"])
expected_temperature = float(os.environ["EXPECTED_TEMPERATURE_K"])
expected_extensions = {
    "E000": 0.0,
    "E200": 2.0e-4,
    "E500": 5.0e-4,
    "E800": 8.0e-4,
    "E1000": 1.0e-3,
    "E1200": 1.2e-3,
}

capture_path = root / "capture_complete.json"
if not capture_path.is_file():
    raise SystemExit(f"missing capture finalization record: {capture_path}")
capture = json.loads(capture_path.read_text())
if int(capture.get("requested_states", -1)) != len(states):
    raise SystemExit("capture requested-state count does not match workflow state list")
if int(capture.get("captured_states", -1)) != len(states):
    raise SystemExit("capture did not complete all requested states")
if capture.get("pending_state_ids") not in ([], None):
    raise SystemExit(f"capture still has pending states: {capture.get('pending_state_ids')}")

rows = []
for state in states:
    if state not in expected_extensions:
        raise SystemExit(f"unsupported state label in theta30 workflow: {state}")
    state_root = root / state
    metadata_path = state_root / "snapshot.json"
    arrays_path = state_root / "state_arrays.npz"
    if not metadata_path.is_file() or not arrays_path.is_file():
        raise SystemExit(f"incomplete snapshot artifact for {state}: {state_root}")
    metadata = json.loads(metadata_path.read_text())
    if str(metadata.get("state_id")) != state:
        raise SystemExit(f"snapshot state-id mismatch for {state}")
    temperature = float(metadata.get("temperature_K", float("nan")))
    if not math.isclose(temperature, expected_temperature, rel_tol=0.0, abs_tol=1.0e-8):
        raise SystemExit(
            f"snapshot temperature mismatch for {state}: {temperature} != {expected_temperature}"
        )
    engine = metadata.get("engine_config", {})
    anisotropic = engine.get("anisotropic_config", {})
    theta = float(anisotropic.get("crystal_theta_deg", float("nan")))
    if not math.isclose(theta, expected_theta, rel_tol=0.0, abs_tol=1.0e-10):
        raise SystemExit(
            f"snapshot crystal orientation mismatch for {state}: {theta} != {expected_theta}"
        )
    extension = float(metadata.get("crack_extension_m", float("nan")))
    target = expected_extensions[state]
    if not math.isclose(extension, target, rel_tol=0.0, abs_tol=2.5e-6):
        raise SystemExit(
            f"snapshot extension mismatch for {state}: {extension} != {target}"
        )
    rows.append(
        {
            "state_id": state,
            "temperature_K": temperature,
            "crystal_theta_deg": theta,
            "crack_extension_um": 1.0e6 * extension,
            "snapshot": str(state_root),
        }
    )
print(json.dumps({
    "capture_root": str(root),
    "state_count": len(rows),
    "states": rows,
    "orientation_gate_passed": True,
}, indent=2, sort_keys=True))
PY

for state in $STATES; do
  snapshot="$SNAPSHOT_ROOT/$state"
  destination="$LOAD_ROOT/$state"
  report="$destination/frozen_geometry_load_invariance.json"
  if [[ -f "$report" ]]; then
    REPORT="$report" STATE="$state" "$PYTHON_BIN" - <<'PY'
import json
import os
from pathlib import Path

report = Path(os.environ["REPORT"])
state = os.environ["STATE"]
payload = json.loads(report.read_text())
if payload.get("parent_state_id") != state:
    raise SystemExit(f"existing report state mismatch: {report}")
if payload.get("load_invariance_passed") is not True:
    raise SystemExit(f"existing report did not pass: {report}")
if payload.get("active_kernel_mechanically_measured") is not True:
    raise SystemExit(f"existing report is not mechanically measured: {report}")
if payload.get("wake_shielding_supported") is not False:
    raise SystemExit(f"existing report unexpectedly enables wake shielding: {report}")
print(f"REUSE accepted report: {report}")
PY
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
for state in $STATES; do
  response="$LOAD_ROOT/$state/active_station_responses_load_1.csv"
  audit="$LOAD_ROOT/$state/active_station_responses_load_1.audit.json"
  report="$LOAD_ROOT/$state/frozen_geometry_load_invariance.json"
  for required in "$response" "$audit" "$report"; do
    [[ -f "$required" ]] || {
      echo "ERROR: missing evaluated mechanics artifact: $required" >&2
      exit 2
    }
  done
  RESPONSES+=("$response")
  REPORTS+=("$report")
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
  --theta-deg "$EXPECTED_THETA_DEG" \
  --da-phys-um "$DA_PHYS_UM" \
  --event-minimum-factor "$CLEAVAGE_EVENT_MIN_FACTOR" \
  --event-maximum-factor "$CLEAVAGE_EVENT_MAX_FACTOR" \
  --margin-events "$KERNEL_MARGIN_EVENTS" \
  --output "$OUTROOT/v10_2_27_signed_kernel_coverage_audit.json"

FAMILY_OUT="$FAMILY_OUT" OUTROOT="$OUTROOT" \
SNAPSHOT_ROOT="$SNAPSHOT_ROOT" LOAD_ROOT="$LOAD_ROOT" \
EXPECTED_THETA_DEG="$EXPECTED_THETA_DEG" \
"$PYTHON_BIN" - <<'PY'
import hashlib
import json
import os
from pathlib import Path

family = Path(os.environ["FAMILY_OUT"]).resolve()
root = Path(os.environ["OUTROOT"]).resolve()
coverage_path = root / "v10_2_27_signed_kernel_coverage_audit.json"
coverage = json.loads(coverage_path.read_text())
payload = {
    "schema": "v10.2.27_theta30_six_state_signed_kernel_workflow_v1",
    "crystal_theta_deg": float(os.environ["EXPECTED_THETA_DEG"]),
    "snapshot_root": str(Path(os.environ["SNAPSHOT_ROOT"]).resolve()),
    "load_invariance_root": str(Path(os.environ["LOAD_ROOT"]).resolve()),
    "family": str(family),
    "family_sha256": hashlib.sha256(family.read_bytes()).hexdigest(),
    "coverage_audit": str(coverage_path.resolve()),
    "coverage_passed": coverage["pass"],
    "atlas_max_crack_path_extension_um": coverage[
        "atlas_max_crack_path_extension_um"
    ],
    "required_atlas_max_crack_path_extension_um": coverage[
        "required_atlas_max_crack_path_extension_um"
    ],
}
(root / "v10_2_27_theta30_signed_kernel_workflow.json").write_text(
    json.dumps(payload, indent=2, sort_keys=True) + "\n"
)
print(json.dumps(payload, indent=2, sort_keys=True))
PY

echo "Theta30 six-state signed-kernel family accepted: $FAMILY_OUT"
