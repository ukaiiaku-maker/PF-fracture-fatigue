#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$ROOT"

PYTHON_BIN=${PYTHON_BIN:-python}
EXPECTED_BRANCH=${EXPECTED_BRANCH:-v10.2.30-hazard-energy-gated-fatigue-events}
EXPECTED_HEAD=${EXPECTED_HEAD:-}
TOTAL_CYCLES=${TOTAL_CYCLES:-1e6}
PARTITION_CYCLES=${PARTITION_CYCLES:-1e5}
REFERENCE_HORIZON_CYCLES=${REFERENCE_HORIZON_CYCLES:-1e12}
MAX_WALL_SECONDS=${MAX_WALL_SECONDS:-900}
STEPS=${STEPS:-128}
TAG=${TAG:-$(date +%Y%m%d_%H%M%S)}
GATE_ROOT=${GATE_ROOT:-$ROOT/runs/v10_2_30_weakt_partition_robust_gate_$TAG}
REFERENCE_OUT="$GATE_ROOT/one_block"
PARTITIONED_OUT="$GATE_ROOT/partitioned"

if [[ "$(git branch --show-current)" != "$EXPECTED_BRANCH" ]]; then
  echo "ERROR: wrong branch" >&2
  exit 2
fi
ACTUAL_HEAD=$(git rev-parse HEAD)
if [[ -n "$EXPECTED_HEAD" && "$ACTUAL_HEAD" != "$EXPECTED_HEAD" ]]; then
  echo "ERROR: expected HEAD=$EXPECTED_HEAD; observed $ACTUAL_HEAD" >&2
  exit 2
fi
if [[ -n "$(git status --porcelain)" ]]; then
  echo "ERROR: working tree is not clean" >&2
  git status --short >&2
  exit 2
fi
if [[ -e "$GATE_ROOT" ]]; then
  echo "ERROR: gate output already exists: $GATE_ROOT" >&2
  exit 2
fi
mkdir -p "$GATE_ROOT"

COMMON=(
  EXPECTED_HEAD="$ACTUAL_HEAD"
  RUN_CYCLES_MAX="$TOTAL_CYCLES"
  REFERENCE_HORIZON_CYCLES="$REFERENCE_HORIZON_CYCLES"
  STEPS="$STEPS"
  MAX_WALL_SECONDS="$MAX_WALL_SECONDS"
  V10230_FORWARD_INITIAL_CYCLES="${V10230_FORWARD_INITIAL_CYCLES:-1e-3}"
  V10230_FORWARD_MAX_SEGMENT_CYCLES="${V10230_FORWARD_MAX_SEGMENT_CYCLES:-1e6}"
  V10230_FORWARD_STATE_PROFILE_REL_TOL="${V10230_FORWARD_STATE_PROFILE_REL_TOL:-1e-4}"
  V10230_FORWARD_MOBILE_REL_TOL="${V10230_FORWARD_MOBILE_REL_TOL:-1e-4}"
  V10230_FORWARD_RETAINED_REL_TOL="${V10230_FORWARD_RETAINED_REL_TOL:-1e-4}"
  V10230_FORWARD_BACKSTRESS_REL_TOL="${V10230_FORWARD_BACKSTRESS_REL_TOL:-1e-4}"
  V10230_FORWARD_EMISSION_LOG_RATE_TOL_DECADES="${V10230_FORWARD_EMISSION_LOG_RATE_TOL_DECADES:-0.01}"
  V10230_FORWARD_MAX_ACCEPTED_SEGMENTS="${V10230_FORWARD_MAX_ACCEPTED_SEGMENTS:-64}"
  V10230_FORWARD_MAX_TRIAL_INTEGRATIONS="${V10230_FORWARD_MAX_TRIAL_INTEGRATIONS:-256}"
  V10230_FORWARD_HEARTBEAT_SEGMENTS="${V10230_FORWARD_HEARTBEAT_SEGMENTS:-8}"
)

echo "Running one-block reference"
env "${COMMON[@]}" \
  OUTROOT="$REFERENCE_OUT" \
  OUTER_PROPOSAL_CYCLES="$TOTAL_CYCLES" \
  bash scripts/run_v10_2_30_weakt_0p55_transient_diagnostic.sh

echo "Running partitioned comparison"
env "${COMMON[@]}" \
  OUTROOT="$PARTITIONED_OUT" \
  OUTER_PROPOSAL_CYCLES="$PARTITION_CYCLES" \
  bash scripts/run_v10_2_30_weakt_0p55_transient_diagnostic.sh

set +e
"$PYTHON_BIN" scripts/compare_v10_2_30_weakt_partition_equivalence.py \
  "$REFERENCE_OUT" \
  "$PARTITIONED_OUT"
COMPARE_RC=$?
set -e

"$PYTHON_BIN" - "$GATE_ROOT" "$ACTUAL_HEAD" "$REFERENCE_OUT" \
  "$PARTITIONED_OUT" "$COMPARE_RC" "$TOTAL_CYCLES" "$PARTITION_CYCLES" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
payload = {
    "schema": "v10.2.30_weakt_partition_robust_gate_v1",
    "git_head": sys.argv[2],
    "reference_root": sys.argv[3],
    "partitioned_root": sys.argv[4],
    "comparison_exit_code": int(sys.argv[5]),
    "passed": int(sys.argv[5]) == 0,
    "total_cycles": float(sys.argv[6]),
    "partition_cycles": float(sys.argv[7]),
    "stationary_tail_propagation_validated": False,
    "safe_to_run_1e12": False,
}
(root / "partition_robust_gate.json").write_text(
    json.dumps(payload, indent=2, sort_keys=True) + "\n"
)
PY

echo "gate_root=$GATE_ROOT"
echo "reference=$REFERENCE_OUT"
echo "partitioned=$PARTITIONED_OUT"
echo "comparison_exit_code=$COMPARE_RC"
exit "$COMPARE_RC"
