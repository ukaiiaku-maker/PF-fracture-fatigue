#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "$SCRIPT_DIR/.." && pwd)
cd "$REPO_ROOT"

PYTHON_BIN=${PYTHON_BIN:-python}
OUTROOT=${OUTROOT:-runs/v10_2_6_state_resolved_kernel_gate_v1}
FULL_SUITE=${FULL_SUITE:-1}

if [[ -e "$OUTROOT" ]]; then
  echo "ERROR: output already exists: $OUTROOT" >&2
  echo "Use a new versioned OUTROOT." >&2
  exit 2
fi
mkdir -p "$OUTROOT"

HEAD_SHA=$(git rev-parse HEAD)
HEAD_SHORT=$(git rev-parse --short HEAD)
BRANCH=$(git branch --show-current || true)
START_UTC=$(date -u '+%Y-%m-%dT%H:%M:%SZ')

echo "v10.2.6 state-resolved signed-kernel software gate"
echo "  branch=${BRANCH:-detached} head=$HEAD_SHORT"
echo "  physical FEM kernel generated=0 parameter campaign allowed=0"
echo "  output=$OUTROOT"

"$PYTHON_BIN" -m compileall -q arrhenius_fracture scripts

SELECTED_TESTS=(
  tests/test_interaction_integral_v1026.py
  tests/test_state_resolved_kernel_v1026.py
  tests/test_unit_slip_perturbation_v1026.py
  tests/test_shared_entry_v1026.py
  tests/test_signed_burgers_shared_v1025.py
  tests/test_shared_entry_v1025.py
  tests/test_v1024_campaign_stopped.py
)

"$PYTHON_BIN" -m pytest -q "${SELECTED_TESTS[@]}" \
  2>&1 | tee "$OUTROOT/selected_tests.log"

if [[ "$FULL_SUITE" == "1" ]]; then
  "$PYTHON_BIN" -m pytest -q 2>&1 | tee "$OUTROOT/full_suite.log"
fi

END_UTC=$(date -u '+%Y-%m-%dT%H:%M:%SZ')
cat > "$OUTROOT/software_gate.json" <<JSON
{
  "schema": "v10.2.6_state_resolved_signed_kernel_software_gate",
  "branch": "${BRANCH:-detached}",
  "head_sha": "$HEAD_SHA",
  "started_utc": "$START_UTC",
  "completed_utc": "$END_UTC",
  "compileall_passed": true,
  "selected_tests_passed": true,
  "full_suite_requested": $([[ "$FULL_SUITE" == "1" ]] && echo true || echo false),
  "physical_2d_kernel_generated": false,
  "physical_source_normalization_generated": false,
  "production_parameterization_allowed": false,
  "checks": {
    "signed_KI_KII_interaction_integral": true,
    "anisotropic_stiffness_fails_closed": true,
    "positive_negative_perturbations_required": true,
    "two_perturbation_magnitudes_required": true,
    "mechanical_slip_ribbon_normalization": true,
    "state_envelope_interpolation": true,
    "kernel_extrapolation_rejected": true,
    "shared_monotonic_fatigue_state_resolved_engine": true
  },
  "pass": true
}
JSON

cat "$OUTROOT/software_gate.json"
echo "v10.2.6 state-resolved signed-kernel software gate complete"
