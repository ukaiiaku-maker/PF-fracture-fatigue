#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$ROOT"

: "${LOAD_INVARIANCE_ROOT:?set LOAD_INVARIANCE_ROOT to the extracted E000/E200/E500/E800 directory}"
: "${ENGINE_CONFIG:?set ENGINE_CONFIG to the selected serialized 2-D engine configuration}"

EXPECTED_ENV=${EXPECTED_ENV:-arrhenius-sharp-front-v10}
if [[ "${CONDA_DEFAULT_ENV:-}" != "$EXPECTED_ENV" ]]; then
  echo "ERROR: activate conda environment $EXPECTED_ENV" >&2
  exit 2
fi

OUTROOT=${OUTROOT:-runs/v10_2_15_stage3_four_option_monotonic_500um_theta45_1x_v1}
MAX_JOBS=${MAX_JOBS:-2}
STEPS=${STEPS:-300000}
TARGET_EXT_UM=${TARGET_EXT_UM:-500}
THETA=${THETA:-45}
SKIP_FINISHED=${SKIP_FINISHED:-1}
RUNTIME_PREFLIGHT=${RUNTIME_PREFLIGHT:-1}
MECHANICS_ROOT=${MECHANICS_ROOT:-$OUTROOT/mechanics}
FAMILY=${SIGNED_KERNEL_FAMILY_JSON:-$MECHANICS_ROOT/v10_2_14_active_only_campaign_family.json}
PREFLIGHT_ROOT=${PREFLIGHT_ROOT:-${OUTROOT}_runtime_preflight_v1}
PREFLIGHT_MARKER="$PREFLIGHT_ROOT/RUNTIME_PREFLIGHT_PASSED"
STATUS_FILE="$OUTROOT/overnight_status.json"
PID_FILE="$OUTROOT/overnight_launcher.pid"
PHASE=starting

mkdir -p "$MECHANICS_ROOT" "$OUTROOT"
echo "$$" > "$PID_FILE"

write_status() {
  local state=$1
  local message=$2
  local exit_code=${3:-}
  STATUS_FILE="$STATUS_FILE" STATE="$state" MESSAGE="$message" EXIT_CODE="$exit_code" \
  OUTROOT_VALUE="$OUTROOT" FAMILY_VALUE="$FAMILY" MAX_JOBS_VALUE="$MAX_JOBS" \
  PREFLIGHT_ROOT_VALUE="$PREFLIGHT_ROOT" LAUNCHER_PID="$$" python - <<'PY'
import json
import os
from datetime import datetime, timezone
from pathlib import Path

path = Path(os.environ["STATUS_FILE"])
payload = {
    "schema": "v10.2.15_stage3_overnight_status",
    "state": os.environ["STATE"],
    "message": os.environ["MESSAGE"],
    "updated_utc": datetime.now(timezone.utc).isoformat(),
    "launcher_pid": int(os.environ["LAUNCHER_PID"]),
    "outroot": os.environ["OUTROOT_VALUE"],
    "family": os.environ["FAMILY_VALUE"],
    "preflight_root": os.environ["PREFLIGHT_ROOT_VALUE"],
    "max_jobs": int(os.environ["MAX_JOBS_VALUE"]),
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

PHASE=atlas_assembly
write_status assembling "Assembling active-only kernel from completed E-state mechanics"
if [[ ! -f "$FAMILY" ]]; then
  echo "[stage3] assembling active-only kernel from completed E-state mechanics"
  python scripts/build_v10_2_14_campaign_ready_active_only_atlas_v2.py \
    --load-invariance-root "$LOAD_INVARIANCE_ROOT" \
    --engine-config "$ENGINE_CONFIG" \
    --out "$FAMILY"
else
  echo "[stage3] reusing kernel family: $FAMILY"
fi

if [[ "$RUNTIME_PREFLIGHT" == "1" && ! -f "$PREFLIGHT_MARKER" ]]; then
  PHASE=runtime_preflight
  write_status preflighting "Initializing all four Stage 3 MPZ grids with the assembled atlas"
  echo "[stage3] runtime preflight: four options at 700 K, one FEM step each"
  rm -rf "$PREFLIGHT_ROOT"
  set +e
  env \
    MODE=smoke \
    SIGNED_KERNEL_FAMILY_JSON="$FAMILY" \
    OUTROOT="$PREFLIGHT_ROOT" \
    MAX_JOBS=1 \
    TEMPS_SMOKE=700 \
    STEPS_SMOKE=1 \
    TARGET_EXT_UM_SMOKE=5 \
    SAVE_SNAPSHOTS_SMOKE=0 \
    SNAPSHOT_BY_EXT_UM_SMOKE=5 \
    THETA="$THETA" \
    SKIP_FINISHED=0 \
    bash scripts/run_v10_2_15_stage3_monotonic_temperature_sweep_macos.sh
  preflight_rc=$?
  set -e
  if [[ "$preflight_rc" -ne 0 ]]; then
    echo "[stage3] runtime preflight failed; full campaign was not started" >&2
    python scripts/status_v10_2_15_stage3.py \
      --outroot "$PREFLIGHT_ROOT" \
      --failed-tail 80 \
      --tail 5 || true
    exit "$preflight_rc"
  fi
  touch "$PREFLIGHT_MARKER"
  echo "[stage3] runtime preflight passed for 200-bin and 80-bin options"
fi

PHASE=campaign
write_status running "Running 40-case full 2-D Stage 3 campaign"
echo "[stage3] starting 40-case full 2-D campaign"
echo "[stage3] outroot=$OUTROOT max_jobs=$MAX_JOBS family=$FAMILY"

set +e
env \
  MODE=full \
  SIGNED_KERNEL_FAMILY_JSON="$FAMILY" \
  OUTROOT="$OUTROOT" \
  MAX_JOBS="$MAX_JOBS" \
  STEPS="$STEPS" \
  TARGET_EXT_UM="$TARGET_EXT_UM" \
  THETA="$THETA" \
  SKIP_FINISHED="$SKIP_FINISHED" \
  bash scripts/run_v10_2_15_stage3_monotonic_temperature_sweep_macos.sh
rc=$?
set -e

if [[ "$rc" -eq 0 ]]; then
  PHASE=complete
  write_status complete "All Stage 3 cases finished and campaign summary was written" 0
  trap - EXIT
  exit 0
fi

PHASE=campaign
write_status failed "Stage 3 campaign exited with failures" "$rc"
trap - EXIT
exit "$rc"
