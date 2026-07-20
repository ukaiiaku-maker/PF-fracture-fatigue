#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$ROOT"

EXPECTED_ENV=${EXPECTED_ENV:-arrhenius-sharp-front-v10}
if [[ "${CONDA_DEFAULT_ENV:-}" != "$EXPECTED_ENV" ]]; then
  echo "ERROR: activate conda environment $EXPECTED_ENV" >&2
  exit 2
fi

OUTROOT=${OUTROOT:-runs/v10_2_15_stage3_existing_2d_parameter_overlay_500um_theta45_1x_v1}
MAX_JOBS=${MAX_JOBS:-2}
STEPS=${STEPS:-300000}
TARGET_EXT_UM=${TARGET_EXT_UM:-500}
THETA=${THETA:-45}
SKIP_FINISHED=${SKIP_FINISHED:-1}
RUNTIME_PREFLIGHT=${RUNTIME_PREFLIGHT:-0}
PREFLIGHT_ROOT=${PREFLIGHT_ROOT:-${OUTROOT}_runtime_preflight_v1}
PREFLIGHT_MARKER="$PREFLIGHT_ROOT/RUNTIME_PREFLIGHT_PASSED"
STATUS_FILE="$OUTROOT/overnight_status.json"
PID_FILE="$OUTROOT/overnight_launcher.pid"
PHASE=starting

mkdir -p "$OUTROOT"
echo "$$" > "$PID_FILE"

write_status() {
  local state=$1
  local message=$2
  local exit_code=${3:-}
  STATUS_FILE="$STATUS_FILE" STATE="$state" MESSAGE="$message" EXIT_CODE="$exit_code" \
  OUTROOT_VALUE="$OUTROOT" MAX_JOBS_VALUE="$MAX_JOBS" LAUNCHER_PID="$$" python - <<'PY'
import json
import os
from datetime import datetime, timezone
from pathlib import Path

path = Path(os.environ["STATUS_FILE"])
payload = {
    "schema": "v10.2.15_stage3_existing_2d_overnight_status",
    "state": os.environ["STATE"],
    "message": os.environ["MESSAGE"],
    "updated_utc": datetime.now(timezone.utc).isoformat(),
    "launcher_pid": int(os.environ["LAUNCHER_PID"]),
    "outroot": os.environ["OUTROOT_VALUE"],
    "max_jobs": int(os.environ["MAX_JOBS_VALUE"]),
    "final_2d_entry": "arrhenius_fracture.sharp_front_v10_1_7_5",
    "parameter_overlay_only": True,
    "signed_atlas_used": False,
    "tip_engine_replaced": False,
}
if os.environ.get("EXIT_CODE", ""):
    payload["exit_code"] = int(os.environ["EXIT_CODE"])
path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
PY
}

on_exit() {
  local rc=$?
  if [[ "$rc" -ne 0 ]]; then
    write_status failed "Stage 3 launcher failed during $PHASE" "$rc" || true
  fi
}
trap on_exit EXIT

if [[ "$RUNTIME_PREFLIGHT" == "1" && ! -f "$PREFLIGHT_MARKER" ]]; then
  PHASE=runtime_preflight
  write_status preflighting "Optional one-step initialization of the unchanged final 2-D model"
  echo "[stage3] optional runtime preflight: final v10.1.7.5 2-D model, four parameter rows"
  rm -rf "$PREFLIGHT_ROOT"
  set +e
  env \
    MODE=smoke \
    OUTROOT="$PREFLIGHT_ROOT" \
    MAX_JOBS=1 \
    TEMPS_SMOKE=700 \
    STEPS_SMOKE=1 \
    TARGET_EXT_UM_SMOKE=5 \
    SAVE_SNAPSHOTS_SMOKE=0 \
    SNAPSHOT_BY_EXT_UM_SMOKE=5 \
    THETA="$THETA" \
    SKIP_FINISHED=0 \
    bash scripts/run_v10_2_15_stage3_monotonic_temperature_sweep.sh
  preflight_rc=$?
  set -e
  if [[ "$preflight_rc" -ne 0 ]]; then
    echo "[stage3] optional runtime preflight failed; full campaign was not started" >&2
    exit "$preflight_rc"
  fi
  touch "$PREFLIGHT_MARKER"
  echo "[stage3] optional runtime preflight passed"
fi

PHASE=campaign
write_status running "Running the unchanged final 2-D model with four parameter-manifest overlays"
echo "[stage3] starting 40-case final-2D parameter-overlay campaign"
echo "[stage3] entry=arrhenius_fracture.sharp_front_v10_1_7_5"
echo "[stage3] physics unchanged; signed atlas not used; outroot=$OUTROOT max_jobs=$MAX_JOBS"

set +e
env \
  MODE=full \
  OUTROOT="$OUTROOT" \
  MAX_JOBS="$MAX_JOBS" \
  STEPS="$STEPS" \
  TARGET_EXT_UM="$TARGET_EXT_UM" \
  THETA="$THETA" \
  SKIP_FINISHED="$SKIP_FINISHED" \
  bash scripts/run_v10_2_15_stage3_monotonic_temperature_sweep.sh
rc=$?
set -e

if [[ "$rc" -eq 0 ]]; then
  PHASE=complete
  write_status complete "All final-2D parameter-overlay cases finished and the campaign summary was written" 0
  trap - EXIT
  exit 0
fi

PHASE=campaign
write_status failed "Final-2D parameter-overlay campaign exited with failures" "$rc"
trap - EXIT
exit "$rc"
