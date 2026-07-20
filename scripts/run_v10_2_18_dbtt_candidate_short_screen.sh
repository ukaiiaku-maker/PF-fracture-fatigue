#!/usr/bin/env bash
set -u
set -o pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$ROOT"

PYTHON_BIN=${PYTHON_BIN:-python}
CONDA_ENV=${CONDA_ENV:-arrhenius-sharp-front-v10}
if [[ "${CONDA_DEFAULT_ENV:-}" != "$CONDA_ENV" ]]; then
  echo "ERROR: activate conda environment '$CONDA_ENV' before running" >&2
  exit 2
fi

OUTROOT=${OUTROOT:-runs/v10_2_18_dbtt_candidate_short_screen_20um_theta45_v1}
FAMILY_JSON=${FAMILY_JSON:-$ROOT/runtime_inputs/v10_2_17/v10_2_14_active_only_campaign_family.json}
PARAMETER_REGISTRY=${PARAMETER_REGISTRY:-$ROOT/arrhenius_fracture/data/materials/MPZ_v9_11_1_parameter_registry.csv}
OPTIONS=${OPTIONS:-"dbtt_primary dbtt_broad_shielding dbtt_intrinsic_control dbtt_moderate_shielding_reference"}
TEMPS=${TEMPS:-"300 400 500 600 700 800 900 1000 1100 1200"}
TARGET_EXT_UM=${TARGET_EXT_UM:-20}
THETA=${THETA:-45}
STEPS=${STEPS:-100000}
MAX_JOBS=${MAX_JOBS:-2}
BASE_HAZARD_SEED=${BASE_HAZARD_SEED:-1720}
SKIP_FINISHED=${SKIP_FINISHED:-1}
NO_PLOTS=${NO_PLOTS:-1}
PRINT_EVERY=${PRINT_EVERY:-50}

# Exact v10.2.17 Stage-3 controls; only the selected DBTT row changes.
NX=${NX:-36}
NY=${NY:-72}
DU=${DU:-2e-7}
DT=${DT:-8.4}
N_STAGGER=${N_STAGGER:-2}
TIP_H_FINE=${TIP_H_FINE:-1e-6}
TIP_RATIO=${TIP_RATIO:-1.20}
DA_PHYS_M=${DA_PHYS_M:-5e-6}
EVENT_TARGET=${EVENT_TARGET:-0.15}
SAVE_SNAPSHOTS=${SAVE_SNAPSHOTS:-0}
SNAPSHOT_BY_EXT_UM=${SNAPSHOT_BY_EXT_UM:-5}
SNAPSHOT_COLS=${SNAPSHOT_COLS:-6}
CLEAVAGE_EVENT_MIN_FACTOR=${CLEAVAGE_EVENT_MIN_FACTOR:-0.5}
CLEAVAGE_EVENT_MAX_FACTOR=${CLEAVAGE_EVENT_MAX_FACTOR:-4.0}
CLEAVAGE_EVENT_SUBSEGMENT_FRACTION=${CLEAVAGE_EVENT_SUBSEGMENT_FRACTION:-0.1}

if [[ ! -f "$FAMILY_JSON" ]]; then
  echo "ERROR: frozen signed family not found: $FAMILY_JSON" >&2
  exit 2
fi
if [[ ! -f "$PARAMETER_REGISTRY" ]]; then
  echo "ERROR: parameter registry not found: $PARAMETER_REGISTRY" >&2
  exit 2
fi

export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
export PARAMETER_CAMPAIGN=1
export CLEAVAGE_HAZARD_MODE=exponential
export CLEAVAGE_EVENT_LENGTH_MODE=threshold_scaled
export CLEAVAGE_EVENT_MIN_FACTOR
export CLEAVAGE_EVENT_MAX_FACTOR
export CLEAVAGE_EVENT_SUBSEGMENT_FRACTION
export ANISOTROPIC_TRANSPORT_MODE=validated_scalar
export ANISOTROPIC_USE_AVALANCHE_BACKEND=1
export ANISOTROPIC_EMISSION_ENABLED=1

mkdir -p "$OUTROOT"
PLAN="$OUTROOT/dbtt_candidate_screen_plan.tsv"

OPTIONS="$OPTIONS" TEMPS="$TEMPS" THETA="$THETA" OUTROOT="$OUTROOT" \
PARAMETER_REGISTRY="$PARAMETER_REGISTRY" BASE_HAZARD_SEED="$BASE_HAZARD_SEED" \
"$PYTHON_BIN" - <<'PY' > "$PLAN"
import os
from pathlib import Path
from arrhenius_fracture.parameter_registry_v9111 import select_option
from arrhenius_fracture.sharp_front_v10_2_18 import DBTT_SCREEN_OPTIONS

options = os.environ["OPTIONS"].split()
temps = [float(v) for v in os.environ["TEMPS"].split()]
if tuple(options) != DBTT_SCREEN_OPTIONS:
    raise SystemExit(f"expected DBTT options {DBTT_SCREEN_OPTIONS}; got {tuple(options)}")
if temps != [float(v) for v in range(300, 1201, 100)]:
    raise SystemExit(f"expected temperatures 300..1200 by 100 K; got {temps}")
root = Path(os.environ["OUTROOT"])
theta = float(os.environ["THETA"])
base = int(os.environ["BASE_HAZARD_SEED"])
registry = os.environ["PARAMETER_REGISTRY"]
print("option_key\tcandidate_id\ttemperature_K\tmpz_length_um\tmpz_n_bins\thazard_seed\tcase_root")
for option in options:
    selected = select_option(option, registry, canonical_stage3_only=False)
    for temperature in temps:
        # Common random numbers: each candidate receives the same first-passage
        # threshold stream at a given temperature, reducing stochastic comparison noise.
        seed = base + int(round(temperature))
        tag = f"T{int(round(temperature))}K_th{theta:g}_seed{seed}"
        case_root = root / option / tag
        print(
            f"{option}\t{selected.candidate_id}\t{temperature:g}\t"
            f"{selected.mpz_length_um:g}\t{selected.mpz_n_bins}\t{seed}\t{case_root}"
        )
PY

N_CASES=$(($(wc -l < "$PLAN") - 1))
if [[ "$N_CASES" -ne 40 ]]; then
  echo "ERROR: expected 40 screening cases; found $N_CASES" >&2
  exit 2
fi

cat <<EOF
v10.2.18 alternate DBTT short-distance screen
  base physics:   arrhenius_fracture.sharp_front_v10_2_17
  selected rows:  $OPTIONS
  temperatures:   $TEMPS
  target:         $TARGET_EXT_UM um
  common seeds:   same hazard seed across candidates at each temperature
  theta:          $THETA deg
  max jobs:       $MAX_JOBS
  solver plots:   $([[ "$NO_PLOTS" == 1 ]] && echo off || echo on)
  output:         $OUTROOT
EOF

PIDS=()
LABELS=()
FAILURES=0

reap_finished() {
  local next_pids=() next_labels=()
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
  local command_file="$case_root/command.sh"
  local rc complete

  mkdir -p "$case_root"
  if [[ "$SKIP_FINISHED" == 1 && -f "$case_root/stage3_case_status.json" ]]; then
    complete=$(CASE_ROOT="$case_root" "$PYTHON_BIN" - <<'PY'
import json, os
from pathlib import Path
p = Path(os.environ["CASE_ROOT"]) / "stage3_case_status.json"
try:
    print("1" if json.loads(p.read_text()).get("complete") is True else "0")
except Exception:
    print("0")
PY
)
    if [[ "$complete" == 1 ]]; then
      echo "SKIP verified complete: $option T=${temperature}K seed=$seed"
      return 0
    fi
  fi

  rm -f "$case_root/RUN_FAILED" "$case_root/exit_code.txt"
  local cmd=(
    "$PYTHON_BIN" -u -m arrhenius_fracture.sharp_front_v10_2_18
    --signed-kernel-family "$FAMILY_JSON"
    --mode 2d
    --parameter-registry "$PARAMETER_REGISTRY"
    --parameter-option "$option"
    --temperatures "$temperature"
    --steps "$STEPS"
    --nx "$NX" --ny "$NY"
    --dU "$DU" --dt "$DT" --n-stagger "$N_STAGGER"
    --tip-h-fine "$TIP_H_FINE" --tip-ratio "$TIP_RATIO"
    --da-phys "$DA_PHYS_M"
    --target-crack-extension-um "$TARGET_EXT_UM"
    --front-state-model moving_pz
    --tip-source-model continuum
    --tip-kinetics-mode moving_velocity
    --bulk-plasticity-mode tip_only
    --directional-j-mode root_signed
    --tip-plasticity
    --active-shielding
    --signed-active-shielding
    --mobile-shield-fraction 0
    --no-wake-shielding
    --crystal-aniso --crystal-compete
    --crystal-theta-deg "$THETA"
    --crystal-material w
    --j-decomposition cluster
    --max-fronts 1
    --crack-backend sharp_wake
    --adaptive-events --adaptive-event-target "$EVENT_TARGET"
    --print-every "$PRINT_EVERY"
    --save-snapshots "$SAVE_SNAPSHOTS"
    --snapshot-by-crack-extension-um "$SNAPSHOT_BY_EXT_UM"
    --snapshot-cols "$SNAPSHOT_COLS"
    --out "$case_root"
  )
  if [[ "$NO_PLOTS" == 1 ]]; then
    cmd+=(--no-plots)
  fi

  {
    echo '#!/usr/bin/env bash'
    printf 'CLEAVAGE_HAZARD_MODE=exponential '
    printf 'CLEAVAGE_HAZARD_SEED=%q ' "$seed"
    printf 'CLEAVAGE_EVENT_LENGTH_MODE=threshold_scaled '
    printf 'ANISOTROPIC_TRANSPORT_MODE=validated_scalar '
    printf 'ANISOTROPIC_USE_AVALANCHE_BACKEND=1 '
    printf '%q ' "${cmd[@]}"
    printf '\n'
  } > "$command_file"
  chmod +x "$command_file"

  echo "START: $option candidate=$candidate T=${temperature}K seed=$seed mpz=${mpz_length}um/${mpz_bins}bins"
  env CLEAVAGE_HAZARD_SEED="$seed" "${cmd[@]}" > "$log" 2>&1
  rc=$?
  echo "$rc" > "$case_root/exit_code.txt"
  if [[ "$rc" -ne 0 ]]; then
    echo "simulation_exit_$rc" > "$case_root/RUN_FAILED"
    return "$rc"
  fi

  "$PYTHON_BIN" scripts/classify_v10_2_15_stage3_case.py \
    --case-root "$case_root" \
    --target-extension-um "$TARGET_EXT_UM" >> "$log" 2>&1 || return $?

  complete=$(CASE_ROOT="$case_root" "$PYTHON_BIN" - <<'PY'
import json, os
from pathlib import Path
p = Path(os.environ["CASE_ROOT"]) / "stage3_case_status.json"
print("1" if json.loads(p.read_text()).get("complete") is True else "0")
PY
)
  if [[ "$complete" != 1 ]]; then
    echo "incomplete_short_screen" > "$case_root/RUN_FAILED"
    return 1
  fi
  rm -f "$case_root/RUN_FAILED"
}

exec 3< "$PLAN"
IFS=$'\t' read -r _header <&3
while IFS=$'\t' read -r option candidate temperature mpz_length mpz_bins seed case_root <&3; do
  [[ -n "$option" ]] || continue
  wait_for_slot
  run_case "$option" "$candidate" "$temperature" "$mpz_length" "$mpz_bins" "$seed" "$case_root" &
  PIDS+=("$!")
  LABELS+=("$option/T${temperature}K/seed${seed}")
done
exec 3<&-

while [[ ${#PIDS[@]} -gt 0 ]]; do
  sleep 2
  reap_finished
done

echo "DBTT candidate screen finished: failures=$FAILURES output=$OUTROOT"
if [[ "$FAILURES" -ne 0 ]]; then
  exit 1
fi

"$PYTHON_BIN" scripts/plot_v10_2_18_dbtt_candidate_screen.py --outroot "$OUTROOT"
