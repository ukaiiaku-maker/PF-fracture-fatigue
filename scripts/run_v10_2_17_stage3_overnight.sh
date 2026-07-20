#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$ROOT"

EXPECTED_ENV=${EXPECTED_ENV:-arrhenius-sharp-front-v10}
if [[ "${CONDA_DEFAULT_ENV:-}" != "$EXPECTED_ENV" ]]; then
  echo "ERROR: activate conda environment $EXPECTED_ENV" >&2
  exit 2
fi

OUTROOT=${OUTROOT:-runs/v10_2_17_stage3_final_signed_stochastic_500um_theta45_1x_v1}
BUNDLED_FAMILY=${BUNDLED_FAMILY:-$ROOT/runtime_inputs/v10_2_17/v10_2_14_active_only_campaign_family.json}
if [[ -z "${FAMILY_JSON:-}" ]]; then
  if [[ -f "$BUNDLED_FAMILY" ]]; then
    FAMILY_JSON="$BUNDLED_FAMILY"
  else
    FAMILY_JSON="$OUTROOT/mechanics/v10_2_14_active_only_campaign_family.json"
  fi
fi
LOAD_INVARIANCE_ROOT=${LOAD_INVARIANCE_ROOT:-$ROOT/runtime_inputs/v10_2_17/v10_2_14_active_load_invariance_700K_all_states_v1}
ENGINE_CONFIG=${ENGINE_CONFIG:-$ROOT/runtime_inputs/v10_2_17/v10_2_3_2d_engine_config.json}
MAX_JOBS=${MAX_JOBS:-2}
STEPS=${STEPS:-300000}
TARGET_EXT_UM=${TARGET_EXT_UM:-500}
THETA=${THETA:-45}
SKIP_FINISHED=${SKIP_FINISHED:-1}
BASE_HAZARD_SEED=${BASE_HAZARD_SEED:-1720}
STATUS_FILE="$OUTROOT/overnight_status.json"
PID_FILE="$OUTROOT/overnight_launcher.pid"
PHASE=starting

mkdir -p "$OUTROOT" "$(dirname "$FAMILY_JSON")"
echo "$$" > "$PID_FILE"

write_status() {
  local state=$1
  local message=$2
  local exit_code=${3:-}
  STATUS_FILE="$STATUS_FILE" STATE="$state" MESSAGE="$message" EXIT_CODE="$exit_code" \
  OUTROOT_VALUE="$OUTROOT" FAMILY_VALUE="$FAMILY_JSON" MAX_JOBS_VALUE="$MAX_JOBS" \
  LAUNCHER_PID="$$" ROOT_VALUE="$ROOT" python - <<'PY'
import json
import os
from datetime import datetime, timezone
from pathlib import Path

path = Path(os.environ["STATUS_FILE"])
family = Path(os.environ["FAMILY_VALUE"]).expanduser().resolve()
root = Path(os.environ["ROOT_VALUE"]).resolve()
try:
    family.relative_to(root)
    local_family = True
except ValueError:
    local_family = False
payload = {
    "schema": "v10.2.17_stage3_final_signed_stochastic_overnight_status",
    "state": os.environ["STATE"],
    "message": os.environ["MESSAGE"],
    "updated_utc": datetime.now(timezone.utc).isoformat(),
    "launcher_pid": int(os.environ["LAUNCHER_PID"]),
    "outroot": os.environ["OUTROOT_VALUE"],
    "max_jobs": int(os.environ["MAX_JOBS_VALUE"]),
    "entry": "arrhenius_fracture.sharp_front_v10_2_17",
    "signed_engine": "arrhenius_fracture.state_resolved_signed_engine_v10214",
    "signed_family": str(family),
    "signed_family_inside_repository": local_family,
    "external_mechanics_inputs_required_for_this_launch": False,
    "cleavage_hazard_mode": "exponential",
    "event_length_mode": "threshold_scaled",
    "constitutive_K_shield_cap_applied": False,
    "wake_shielding_enabled": False,
}
if os.environ.get("EXIT_CODE", ""):
    payload["exit_code"] = int(os.environ["EXIT_CODE"])
path.parent.mkdir(parents=True, exist_ok=True)
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

PHASE=mechanics_family
if [[ -f "$FAMILY_JSON" ]]; then
  write_status auditing "Reusing the existing local signed-family runtime artifact"
  echo "[stage3] reusing signed family: $FAMILY_JSON"
else
  write_status assembling "Assembling the signed family because no local runtime artifact exists"
  if [[ ! -d "$LOAD_INVARIANCE_ROOT" ]]; then
    echo "ERROR: signed family is missing and LOAD_INVARIANCE_ROOT does not exist: $LOAD_INVARIANCE_ROOT" >&2
    echo "Freeze the existing family before deleting legacy mechanics inputs." >&2
    exit 2
  fi
  if [[ ! -f "$ENGINE_CONFIG" ]]; then
    echo "ERROR: signed family is missing and ENGINE_CONFIG does not exist: $ENGINE_CONFIG" >&2
    echo "Freeze the existing family before deleting legacy mechanics inputs." >&2
    exit 2
  fi
  python scripts/build_v10_2_14_campaign_ready_active_only_atlas_v2.py \
    --load-invariance-root "$LOAD_INVARIANCE_ROOT" \
    --engine-config "$ENGINE_CONFIG" \
    --out "$FAMILY_JSON"
fi

PHASE=stack_audit
write_status auditing "Auditing the local signed stochastic execution stack"
SIGNED_KERNEL_FAMILY_JSON="$FAMILY_JSON" python - <<'PY'
import os
from pathlib import Path
import arrhenius_fracture
from arrhenius_fracture.signed_kernel_family_v10214 import (
    ActiveOnlySigned2DShieldingKernelFamily,
)
from arrhenius_fracture.state_resolved_signed_engine_v10214 import (
    StateResolvedSignedBurgersTipEngine,
)
import arrhenius_fracture.sharp_front_v10_2_17 as entry

root = Path.cwd().resolve()
package = Path(arrhenius_fracture.__file__).resolve().parent
if package != root / "arrhenius_fracture":
    raise SystemExit(f"stale editable import: expected {root / 'arrhenius_fracture'}, got {package}")
family = ActiveOnlySigned2DShieldingKernelFamily.from_json(
    os.environ["SIGNED_KERNEL_FAMILY_JSON"]
)
assert family.metadata.get("production_parameterization_allowed") is True
assert family.metadata.get("constitutive_K_shield_cap_present") is False
assert family.metadata.get("active_kernel_mechanically_measured") is True
assert family.metadata.get("wake_kernel_mechanically_measured") is False
assert family.metadata.get("wake_shielding_supported") is False
assert entry.StateResolvedSignedBurgersTipEngine is StateResolvedSignedBurgersTipEngine
assert entry.FINAL_ENGINE.endswith("state_resolved_signed_engine_v10214")
assert entry.FINAL_FAMILY.endswith("signed_kernel_family_v10214")
print(
    "stack audit passed: "
    f"package={package} states={len(family.states)} engine={entry.FINAL_ENGINE} "
    "hazard=exponential event_length=threshold_scaled Kcap=off wake=off"
)
PY

PHASE=campaign
write_status running "Running the 40-case final signed stochastic Stage 3 campaign"
echo "[stage3] starting 40-case final signed stochastic campaign"
echo "[stage3] entry=arrhenius_fracture.sharp_front_v10_2_17"
echo "[stage3] family=$FAMILY_JSON"
echo "[stage3] hazard=exponential event_length=threshold_scaled outroot=$OUTROOT max_jobs=$MAX_JOBS"

set +e
env \
  MODE=full \
  OUTROOT="$OUTROOT" \
  SIGNED_KERNEL_FAMILY_JSON="$FAMILY_JSON" \
  MAX_JOBS="$MAX_JOBS" \
  STEPS="$STEPS" \
  TARGET_EXT_UM="$TARGET_EXT_UM" \
  THETA="$THETA" \
  SKIP_FINISHED="$SKIP_FINISHED" \
  BASE_HAZARD_SEED="$BASE_HAZARD_SEED" \
  bash scripts/run_v10_2_17_stage3_monotonic_temperature_sweep.sh
rc=$?
set -e

if [[ "$rc" -eq 0 ]]; then
  PHASE=complete
  write_status complete "All final signed stochastic Stage 3 cases finished and the campaign summary was written" 0
  trap - EXIT
  exit 0
fi

PHASE=campaign
write_status failed "Final signed stochastic Stage 3 campaign exited with failures" "$rc"
trap - EXIT
exit "$rc"
