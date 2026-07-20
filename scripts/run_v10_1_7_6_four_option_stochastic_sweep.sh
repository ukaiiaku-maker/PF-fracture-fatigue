#!/usr/bin/env bash
set -u
set -o pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$ROOT"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1

PYTHON_BIN=${PYTHON_BIN:-python}
EXPECTED_ENV=${EXPECTED_ENV:-arrhenius-sharp-front-v10}
if [[ "${CONDA_DEFAULT_ENV:-}" != "$EXPECTED_ENV" ]]; then
  echo "ERROR: activate conda environment $EXPECTED_ENV" >&2
  exit 2
fi

OUTROOT=${OUTROOT:-runs/v10_1_7_6_four_option_stochastic_500um_theta45_v1}
PARAMETER_REGISTRY=${PARAMETER_REGISTRY:-$ROOT/arrhenius_fracture/data/materials/MPZ_v9_11_1_parameter_registry.csv}
OPTIONS=${OPTIONS:-"ceramic_primary weakT_primary dbtt_primary peak_primary"}
TEMPS=${TEMPS:-"300 400 500 600 700 800 900 1000 1100 1200"}
THETA=${THETA:-45}
TARGET_EXT_UM=${TARGET_EXT_UM:-500}
STEPS=${STEPS:-300000}
MAX_JOBS=${MAX_JOBS:-2}
BASE_HAZARD_SEED=${BASE_HAZARD_SEED:-1720}
SKIP_FINISHED=${SKIP_FINISHED:-1}
PRINT_EVERY=${PRINT_EVERY:-200}
SAVE_SNAPSHOTS=${SAVE_SNAPSHOTS:-0}
SNAPSHOT_BY_EXT_UM=${SNAPSHOT_BY_EXT_UM:-25}
NO_PLOTS=${NO_PLOTS:-1}

# Exact working v10.1.7.1/4/5 production controls.
NX=${NX:-48}
NY=${NY:-96}
TIP_H_FINE=${TIP_H_FINE:-5e-7}
TIP_RATIO=${TIP_RATIO:-1.2}
DU=${DU:-2e-7}
DT=${DT:-8.4}
N_STAGGER=${N_STAGGER:-2}
DA_CHECKPOINT_M=${DA_CHECKPOINT_M:-5e-6}
WAKE_LENGTH_UM=${WAKE_LENGTH_UM:-100}
WAKE_N_BINS=${WAKE_N_BINS:-0}
EVENT_TARGET=${EVENT_TARGET:-0.05}
PACKET_LENGTH_M=${PACKET_LENGTH_M:-2.5e-10}
KINETIC_MAX_ACTION_SUBSTEP=${KINETIC_MAX_ACTION_SUBSTEP:-0.01}
KINETIC_MAX_TRANSLATION_SUBSTEP_M=${KINETIC_MAX_TRANSLATION_SUBSTEP_M:-5e-8}

export CAMPAIGN_BACKSTRESS_SCALE=${CAMPAIGN_BACKSTRESS_SCALE:-1.0}
export CAMPAIGN_REFRESH_SCALE=${CAMPAIGN_REFRESH_SCALE:-1.0}
export CLEAVAGE_HAZARD_MODE=exponential
export CLEAVAGE_EVENT_LENGTH_MODE=threshold_scaled
export CLEAVAGE_EVENT_MIN_FACTOR=${CLEAVAGE_EVENT_MIN_FACTOR:-0.5}
export CLEAVAGE_EVENT_MAX_FACTOR=${CLEAVAGE_EVENT_MAX_FACTOR:-4.0}
export CLEAVAGE_EVENT_SUBSEGMENT_FRACTION=${CLEAVAGE_EVENT_SUBSEGMENT_FRACTION:-0.1}
export ANISOTROPIC_USE_AVALANCHE_BACKEND=1
export ANISOTROPIC_TRANSPORT_MODE=validated_scalar
export ANISOTROPIC_EMISSION_ENABLED=1

mkdir -p "$OUTROOT"
PLAN="$OUTROOT/campaign_plan.tsv"
STATUS="$OUTROOT/campaign_status.json"

OPTIONS="$OPTIONS" TEMPS="$TEMPS" THETA="$THETA" OUTROOT="$OUTROOT" \
PARAMETER_REGISTRY="$PARAMETER_REGISTRY" BASE_HAZARD_SEED="$BASE_HAZARD_SEED" \
"$PYTHON_BIN" - <<'PY' > "$PLAN"
import os
from pathlib import Path
from arrhenius_fracture.parameter_registry_v9111 import CANONICAL_OPTIONS, select_option
options = os.environ["OPTIONS"].split()
temps = [int(float(value)) for value in os.environ["TEMPS"].split()]
if tuple(options) != CANONICAL_OPTIONS:
    raise SystemExit(f"expected options {CANONICAL_OPTIONS}; got {tuple(options)}")
if temps != list(range(300, 1201, 100)):
    raise SystemExit(f"expected temperatures 300..1200 by 100 K; got {temps}")
root = Path(os.environ["OUTROOT"])
theta = float(os.environ["THETA"])
base = int(os.environ["BASE_HAZARD_SEED"])
print("option_key\tcandidate_id\ttemperature_K\tmpz_length_um\tmpz_n_bins\thazard_seed\tcase_root")
for oi, option in enumerate(options):
    selected = select_option(option, os.environ["PARAMETER_REGISTRY"])
    for ti, temperature in enumerate(temps):
        seed = base + 1000 * oi + ti
        case_root = root / option / f"T{temperature}K_th{theta:g}"
        print(
            f"{option}\t{selected.candidate_id}\t{temperature}\t"
            f"{selected.mpz_length_um:g}\t{selected.mpz_n_bins}\t{seed}\t{case_root}"
        )
PY

N_CASES=$(($(wc -l < "$PLAN") - 1))
if [[ "$N_CASES" -ne 40 ]]; then
  echo "ERROR: expected 40 cases; found $N_CASES" >&2
  exit 2
fi

write_status() {
  local state=$1
  local message=$2
  STATE="$state" MESSAGE="$message" STATUS="$STATUS" OUTROOT="$OUTROOT" \
  "$PYTHON_BIN" - <<'PY'
import json, os
from datetime import datetime, timezone
from pathlib import Path
root = Path(os.environ["OUTROOT"])
payload = {
    "schema": "v10.1.7.6_four_option_stochastic_campaign_status",
    "state": os.environ["STATE"],
    "message": os.environ["MESSAGE"],
    "updated_utc": datetime.now(timezone.utc).isoformat(),
    "working_entry": "arrhenius_fracture.sharp_front_v10_1_7_5",
    "parameter_overlay_entry": "arrhenius_fracture.sharp_front_v10_1_7_6",
    "atlas_used": False,
    "mechanics_validation_stage": False,
    "cleavage_hazard_mode": "exponential",
    "cleavage_event_length_mode": "threshold_scaled",
    "planned_cases": 40,
    "complete_processes": len(list(root.glob("*/T*_th*/RUN_COMPLETE"))),
    "failed_processes": len(list(root.glob("*/T*_th*/RUN_FAILED"))),
}
Path(os.environ["STATUS"]).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
PY
}

cat <<EOF
v10.1.7.6 four-option stochastic monotonic campaign
  working code:  arrhenius_fracture.sharp_front_v10_1_7_5
  overlay:       exact v9.11.1 material row + recommended MPZ grid only
  atlas:         none
  hazard:        exponential
  event length:  threshold_scaled, bounds ${CLEAVAGE_EVENT_MIN_FACTOR}-${CLEAVAGE_EVENT_MAX_FACTOR}
  cases:         $N_CASES
  target:        ${TARGET_EXT_UM} um
  theta:         ${THETA} deg
  max jobs:      $MAX_JOBS
  output:        $OUTROOT
EOF
write_status running "Running 40-case stochastic parameter-overlay campaign"

PIDS=()
LABELS=()
FAILURES=0

reap_finished() {
  local next_pids=()
  local next_labels=()
  local i pid label rc
  for ((i=0; i<${#PIDS[@]}; i++)); do
    pid=${PIDS[$i]}
    label=${LABELS[$i]}
    if kill -0 "$pid" 2>/dev/null; then
      next_pids+=("$pid")
      next_labels+=("$label")
    else
      wait "$pid"
      rc=$?
      if [[ "$rc" -ne 0 ]]; then
        echo "FAILED: $label (exit $rc)" >&2
        FAILURES=$((FAILURES + 1))
      else
        echo "FINISHED: $label"
      fi
    fi
  done
  if [[ ${#next_pids[@]} -gt 0 ]]; then
    PIDS=("${next_pids[@]}")
    LABELS=("${next_labels[@]}")
  else
    PIDS=()
    LABELS=()
  fi
  write_status running "Running 40-case stochastic parameter-overlay campaign"
}

wait_for_slot() {
  while [[ ${#PIDS[@]} -ge $MAX_JOBS ]]; do
    sleep 2
    reap_finished
  done
}

run_case() {
  local option=$1 candidate=$2 temperature=$3 mpz_length=$4 mpz_bins=$5 seed=$6 case_root=$7
  local log="$case_root/run.log"
  mkdir -p "$case_root"
  if [[ "$SKIP_FINISHED" == 1 && -f "$case_root/RUN_COMPLETE" ]]; then
    echo "SKIP complete: $option T=${temperature}K"
    return 0
  fi
  rm -f "$case_root/RUN_COMPLETE" "$case_root/RUN_FAILED" "$case_root/exit_code.txt"

  local cmd=(
    "$PYTHON_BIN" -u -m arrhenius_fracture.sharp_front_v10_1_7_6
    --parameter-registry "$PARAMETER_REGISTRY"
    --parameter-option "$option"
    --mode 2d
    --temperatures "$temperature"
    --bulk-plasticity-mode tip_only
    --directional-j-mode root_signed
    --tip-kinetics-mode moving_velocity
    --tip-source-model continuum
    --tip-plasticity --active-shielding --signed-active-shielding
    --mobile-shield-fraction 0
    --kinetic-packet-length-m "$PACKET_LENGTH_M"
    --kinetic-max-action-substep "$KINETIC_MAX_ACTION_SUBSTEP"
    --kinetic-max-translation-substep-m "$KINETIC_MAX_TRANSLATION_SUBSTEP_M"
    --steps "$STEPS" --nx "$NX" --ny "$NY"
    --dU "$DU" --dt "$DT" --n-stagger "$N_STAGGER"
    --tip-h-fine "$TIP_H_FINE" --tip-ratio "$TIP_RATIO"
    --da-phys "$DA_CHECKPOINT_M"
    --target-crack-extension-um "$TARGET_EXT_UM"
    --wake-length-um "$WAKE_LENGTH_UM" --wake-n-bins "$WAKE_N_BINS"
    --no-wake-shielding
    --crystal-aniso --crystal-compete --crystal-theta-deg "$THETA"
    --crystal-material w --j-decomposition cluster
    --max-fronts 1 --crack-backend sharp_wake
    --adaptive-events --adaptive-event-target "$EVENT_TARGET"
    --print-every "$PRINT_EVERY"
    --save-snapshots "$SAVE_SNAPSHOTS"
    --snapshot-by-crack-extension-um "$SNAPSHOT_BY_EXT_UM"
    --out "$case_root"
  )
  if [[ "$NO_PLOTS" == 1 ]]; then cmd+=(--no-plots); fi

  {
    echo '#!/usr/bin/env bash'
    printf 'CLEAVAGE_HAZARD_MODE=exponential '
    printf 'CLEAVAGE_EVENT_LENGTH_MODE=threshold_scaled '
    printf 'CLEAVAGE_HAZARD_SEED=%q ' "$seed"
    printf 'ANISOTROPIC_TRANSPORT_MODE=validated_scalar '
    printf '%q ' "${cmd[@]}"
    printf '\n'
  } > "$case_root/command.sh"
  chmod +x "$case_root/command.sh"

  echo "START: $option candidate=$candidate T=${temperature}K mpz=${mpz_length}um/${mpz_bins}bins seed=$seed"
  CLEAVAGE_HAZARD_SEED="$seed" "${cmd[@]}" > "$log" 2>&1
  local rc=$?
  echo "$rc" > "$case_root/exit_code.txt"
  if [[ "$rc" -eq 0 ]]; then
    echo complete > "$case_root/RUN_COMPLETE"
    return 0
  fi
  echo "simulation_exit_$rc" > "$case_root/RUN_FAILED"
  return "$rc"
}

exec 3< "$PLAN"
IFS=$'\t' read -r _header <&3
while IFS=$'\t' read -r option candidate temperature mpz_length mpz_bins seed case_root <&3; do
  [[ -n "$option" ]] || continue
  wait_for_slot
  run_case "$option" "$candidate" "$temperature" "$mpz_length" "$mpz_bins" "$seed" "$case_root" &
  PIDS+=("$!")
  LABELS+=("$option/T${temperature}K")
done
exec 3<&-

while [[ ${#PIDS[@]} -gt 0 ]]; do
  sleep 2
  reap_finished
done

OUTROOT="$OUTROOT" "$PYTHON_BIN" - <<'PY'
import csv, json, os
from pathlib import Path
root = Path(os.environ["OUTROOT"])
rows = []
for case in sorted(root.glob("*/T*K_th*")):
    summary_path = case / "summary.json"
    selection_path = case / "v10_1_7_6_parameter_selection.json"
    if not summary_path.is_file():
        continue
    summary = json.loads(summary_path.read_text())
    row = summary[0] if isinstance(summary, list) and summary else {}
    selection = json.loads(selection_path.read_text()) if selection_path.is_file() else {}
    rows.append({
        "option_key": selection.get("option_key", case.parent.name),
        "candidate_id": selection.get("candidate_id", ""),
        "temperature_K": row.get("T"),
        "Kc_first_MPa_sqrt_m": row.get("Kc_first_MPa_sqrt_m"),
        "n_advances": row.get("n_advances"),
        "n_geometry_events": row.get("n_geometry_events"),
        "mode": row.get("mode"),
        "case_root": str(case),
    })
(root / "campaign_summary.json").write_text(json.dumps(rows, indent=2, sort_keys=True) + "\n")
columns = ["option_key", "candidate_id", "temperature_K", "Kc_first_MPa_sqrt_m", "n_advances", "n_geometry_events", "mode", "case_root"]
with (root / "campaign_summary.csv").open("w", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=columns)
    writer.writeheader()
    writer.writerows(rows)
print(json.dumps({"summarized_cases": len(rows)}, sort_keys=True))
PY

if [[ "$FAILURES" -eq 0 ]]; then
  write_status complete "All 40 stochastic parameter-overlay processes exited successfully"
else
  write_status failed "$FAILURES stochastic parameter-overlay processes failed"
fi

echo "campaign finished: failures=$FAILURES outroot=$OUTROOT"
exit "$FAILURES"
