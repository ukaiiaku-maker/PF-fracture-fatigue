#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "$SCRIPT_DIR/.." && pwd)
cd "$REPO_ROOT"

PYTHON_BIN=${PYTHON_BIN:-python}
OUTROOT=${OUTROOT:-runs/v10_2_5_signed_software_gate_v1}
FULL_SUITE=${FULL_SUITE:-0}

if [[ -e "$OUTROOT" ]]; then
  echo "ERROR: output already exists: $OUTROOT" >&2
  echo "Use a new versioned OUTROOT or remove the old software-gate output." >&2
  exit 2
fi
mkdir -p "$OUTROOT"

HEAD_SHA=$(git rev-parse HEAD)
HEAD_SHORT=$(git rev-parse --short HEAD)
BRANCH=$(git branch --show-current || true)
START_UTC=$(date -u '+%Y-%m-%dT%H:%M:%SZ')

echo "v10.2.5 signed-dislocation software gate"
echo "  branch=${BRANCH:-detached} head=$HEAD_SHORT"
echo "  physical_kernel_used=0 production_physics_validated=0"
echo "  output=$OUTROOT"

"$PYTHON_BIN" -m compileall -q arrhenius_fracture scripts

SELECTED_TESTS=(
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
  "schema": "v10.2.5_signed_dislocation_software_gate",
  "branch": "${BRANCH:-detached}",
  "head_sha": "$HEAD_SHA",
  "started_utc": "$START_UTC",
  "completed_utc": "$END_UTC",
  "compileall_passed": true,
  "selected_tests_passed": true,
  "full_suite_requested": $([[ "$FULL_SUITE" == "1" ]] && echo true || echo false),
  "physical_kernel_used": false,
  "production_fracture_physics_validated": false,
  "production_fatigue_physics_validated": false,
  "checks": {
    "equal_opposite_signed_content_cancels_shielding": true,
    "burgers_sign_reversal_reverses_interaction": true,
    "channel_kernel_can_produce_antishielding": true,
    "moving_frame_preserves_sign_species": true,
    "old_thousands_site_anchor_rejected": true,
    "local_30_GPa_strength_limit_preserved_in_replay": true,
    "monotonic_and_fatigue_install_identical_engine": true,
    "old_v10_2_4_campaign_remains_stopped": true
  },
  "pass": true
}
JSON

cat "$OUTROOT/software_gate.json"
echo "v10.2.5 signed-dislocation software gate complete"
