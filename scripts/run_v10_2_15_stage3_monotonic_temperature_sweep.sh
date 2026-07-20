#!/usr/bin/env bash
set -u
set -o pipefail

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$ROOT_DIR"

MODE=${MODE:-full}
PYTHON_BIN=${PYTHON_BIN:-python}
CONDA_ENV=${CONDA_ENV:-arrhenius-sharp-front-v10}
ACTIVE_CONDA_ENV=${CONDA_DEFAULT_ENV:-}
OUTROOT=${OUTROOT:-runs/v10_2_15_stage3_existing_2d_parameter_overlay_500um_theta45_1x_v1}
PARAMETER_REGISTRY=${PARAMETER_REGISTRY:-$ROOT_DIR/arrhenius_fracture/data/materials/MPZ_v9_11_1_parameter_registry.csv}
OPTIONS=${OPTIONS:-"ceramic_primary weakT_primary dbtt_primary peak_primary"}
TEMPS=${TEMPS:-"300 400 500 600 700 800 900 1000 1100 1200"}
THETA=${THETA:-45}
TARGET_EXT_UM=${TARGET_EXT_UM:-500}
DA_PHYS_M=${DA_PHYS_M:-5e-6}
DU=${DU:-2e-7}
DT=${DT:-8.4}
STEPS=${STEPS:-300000}
NX=${NX:-36}
NY=${NY:-72}
N_STAGGER=${N_STAGGER:-2}
TIP_H_FINE=${TIP_H_FINE:-1e-6}
TIP_RATIO=${TIP_RATIO:-1.20}
EVENT_TARGET=${EVENT_TARGET:-0.15}
MAX_JOBS=${MAX_JOBS:-2}
PRINT_EVERY=${PRINT_EVERY:-50}
SAVE_SNAPSHOTS=${SAVE_SNAPSHOTS:-11}
SNAPSHOT_BY_EXT_UM=${SNAPSHOT_BY_EXT_UM:-50}
SNAPSHOT_COLS=${SNAPSHOT_COLS:-6}
NO_PLOTS=${NO_PLOTS:-1}
SKIP_FINISHED=${SKIP_FINISHED:-1}
ALLOW_PARTIAL=${ALLOW_PARTIAL:-0}
DRY_RUN=${DRY_RUN:-0}
EXTRA_ARGS=${EXTRA_ARGS:-}

case "$MODE" in
  full) ;;
  smoke)
    TEMPS=${TEMPS_SMOKE:-700}
    TARGET_EXT_UM=${TARGET_EXT_UM_SMOKE:-20}
    STEPS=${STEPS_SMOKE:-12000}
    SAVE_SNAPSHOTS=${SAVE_SNAPSHOTS_SMOKE:-3}
    SNAPSHOT_BY_EXT_UM=${SNAPSHOT_BY_EXT_UM_SMOKE:-5}
    ALLOW_PARTIAL=1
    ;;
  plan) DRY_RUN=1 ;;
  *)
    echo "ERROR: MODE must be full, smoke, or plan; got $MODE" >&2
    exit 2
    ;;
esac

if [[ "$ACTIVE_CONDA_ENV" != "$CONDA_ENV" ]]; then
  echo "ERROR: activate conda environment '$CONDA_ENV' before running." >&2
  echo "Current environment: '${ACTIVE_CONDA_ENV:-none}'" >&2
  exit 2
fi
if [[ ! -f "$PARAMETER_REGISTRY" ]]; then
  echo "ERROR: parameter registry not found: $PARAMETER_REGISTRY" >&2
  exit 2
fi
if ! [[ "$MAX_JOBS" =~ ^[1-9][0-9]*$ ]]; then
  echo "ERROR: MAX_JOBS must be a positive integer" >&2
  exit 2
fi

export PYTHONPATH="$ROOT_DIR${PYTHONPATH:+:$PYTHONPATH}"
export CLEAVAGE_HAZARD_MODE=deterministic
export CLEAVAGE_EVENT_LENGTH_MODE=fixed
export ANISOTROPIC_TRANSPORT_MODE=validated_scalar
export ANISOTROPIC_USE_AVALANCHE_BACKEND=1

"$PYTHON_BIN" - <<'PY'
import arrhenius_fracture
from pathlib import Path
expected = Path.cwd() / "arrhenius_fracture"
actual = Path(arrhenius_fracture.__file__).resolve().parent
if arrhenius_fracture.PROJECT_ID != "PF-fracture-fatigue":
    raise SystemExit(f"wrong project imported: {arrhenius_fracture.PROJECT_ID!r}")
if actual != expected.resolve():
    raise SystemExit(f"stale editable import: expected {expected.resolve()}, got {actual}")
print(f"project={arrhenius_fracture.PROJECT_ID} release={arrhenius_fracture.PROJECT_RELEASE} package={actual}")
PY

mkdir -p "$OUTROOT"
PLAN="$OUTROOT/stage3_campaign_plan.tsv"
OPTIONS="$OPTIONS" TEMPS="$TEMPS" THETA="$THETA" OUTROOT="$OUTROOT" \
PARAMETER_REGISTRY="$PARAMETER_REGISTRY" ALLOW_PARTIAL="$ALLOW_PARTIAL" \
"$PYTHON_BIN" - <<'PY' > "$PLAN"
import os
from pathlib import Path
from arrhenius_fracture.parameter_registry_v9111 import CANONICAL_STAGE3_OPTIONS, select_option
options = os.environ["OPTIONS"].split()
temps = [float(value) for value in os.environ["TEMPS"].split()]
registry = os.environ["PARAMETER_REGISTRY"]
theta = float(os.environ["THETA"])
outroot = Path(os.environ["OUTROOT"])
allow_partial = os.environ.get("ALLOW_PARTIAL", "0") == "1"
if len(options) != len(set(options)):
    raise SystemExit("duplicate option in OPTIONS")
if len(temps) != len(set(temps)):
    raise SystemExit("duplicate temperature in TEMPS")
if not allow_partial:
    if tuple(options) != CANONICAL_STAGE3_OPTIONS:
        raise SystemExit(f"full Stage 3 requires options {CANONICAL_STAGE3_OPTIONS}; got {tuple(options)}")
    expected_temps = [float(value) for value in range(300, 1201, 100)]
    if temps != expected_temps:
        raise SystemExit(f"full Stage 3 requires temperatures {expected_temps}; got {temps}")
print("option_key\tcandidate_id\ttemperature_K\tmpz_length_um\tmpz_n_bins\tcase_root")
for option in options:
    selected = select_option(option, registry, canonical_stage3_only=True)
    for temperature in temps:
        tag = f"T{int(round(temperature))}K_th{theta:g}"
        case_root = outroot / selected.option_key / tag
        print(f"{selected.option_key}\t{selected.candidate_id}\t{temperature:g}\t{selected.mpz_length_um:g}\t{selected.mpz_n_bins}\t{case_root}")
PY

N_CASES=$(($(wc -l < "$PLAN") - 1))
if [[ "$MODE" == "full" && "$ALLOW_PARTIAL" != "1" && "$N_CASES" -ne 40 ]]; then
  echo "ERROR: full Stage 3 plan must contain exactly 40 cases; found $N_CASES" >&2
  exit 2
fi

cat <<EOF
Stage 3 existing-2D parameter-overlay plan
  final entry:   arrhenius_fracture.sharp_front_v10_1_7_5
  physics:       unchanged; exact material-manifest overlay only
  mode:          $MODE
  cases:         $N_CASES
  options:       $OPTIONS
  temperatures:  $TEMPS
  target:        $TARGET_EXT_UM um
  theta:         $THETA deg
  max jobs:      $MAX_JOBS
  output:        $OUTROOT
  plan:          $PLAN
EOF

if [[ "$DRY_RUN" == "1" ]]; then
  cat "$PLAN"
  exit 0
fi

PIDS=()
LABELS=()
FAILURES=0

terminate_children() {
  local pid
  if [[ ${#PIDS[@]} -gt 0 ]]; then
    for pid in "${PIDS[@]}"; do
      kill "$pid" 2>/dev/null || true
    done
  fi
}
trap terminate_children INT TERM

reap_finished() {
  local new_pids=()
  local new_labels=()
  local i pid label rc
  for ((i=0; i<${#PIDS[@]}; i++)); do
    pid=${PIDS[$i]}
    label=${LABELS[$i]}
    if kill -0 "$pid" 2>/dev/null; then
      new_pids+=("$pid")
      new_labels+=("$label")
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
  if [[ ${#new_pids[@]} -gt 0 ]]; then
    PIDS=("${new_pids[@]}")
    LABELS=("${new_labels[@]}")
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
  local option=$1
  local candidate=$2
  local temperature=$3
  local mpz_length=$4
  local mpz_bins=$5
  local case_root=$6
  local log="$case_root/run.log"
  local command_file="$case_root/command.sh"
  local rc

  mkdir -p "$case_root"
  if [[ "$SKIP_FINISHED" == "1" && -f "$case_root/stage3_case_status.json" ]]; then
    echo "SKIP finished: $option T=${temperature}K"
    return 0
  fi
  rm -f "$case_root/RUN_FAILED" "$case_root/exit_code.txt"

  local cmd=(
    "$PYTHON_BIN" -m arrhenius_fracture.sharp_front_v10_2_15
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
  if [[ "$NO_PLOTS" == "1" ]]; then
    cmd+=(--no-plots)
  fi
  if [[ -n "$EXTRA_ARGS" ]]; then
    local extra_words
    read -r -a extra_words <<< "$EXTRA_ARGS"
    cmd+=("${extra_words[@]}")
  fi

  {
    echo '#!/usr/bin/env bash'
    printf 'CLEAVAGE_HAZARD_MODE=deterministic CLEAVAGE_EVENT_LENGTH_MODE=fixed '
    printf 'ANISOTROPIC_TRANSPORT_MODE=validated_scalar '
    printf '%q ' "${cmd[@]}"
    printf '\n'
  } > "$command_file"
  chmod +x "$command_file"

  echo "START: $option candidate=$candidate T=${temperature}K mpz=${mpz_length}um/${mpz_bins}bins entry=v10.1.7.5"
  "${cmd[@]}" > "$log" 2>&1
  rc=$?
  echo "$rc" > "$case_root/exit_code.txt"
  if [[ "$rc" -ne 0 ]]; then
    echo "simulation_exit_$rc" > "$case_root/RUN_FAILED"
    return "$rc"
  fi
  rm -f "$case_root/RUN_FAILED"
  "$PYTHON_BIN" scripts/classify_v10_2_15_stage3_case.py \
    --case-root "$case_root" \
    --target-extension-um "$TARGET_EXT_UM" \
    >> "$log" 2>&1
}

exec 3< "$PLAN"
IFS=$'\t' read -r _header <&3
while IFS=$'\t' read -r option candidate temperature mpz_length mpz_bins case_root <&3; do
  [[ -n "$option" ]] || continue
  wait_for_slot
  run_case "$option" "$candidate" "$temperature" "$mpz_length" "$mpz_bins" "$case_root" &
  PIDS+=("$!")
  LABELS+=("$option/T${temperature}K")
done
exec 3<&-

while [[ ${#PIDS[@]} -gt 0 ]]; do
  sleep 2
  reap_finished
done

"$PYTHON_BIN" scripts/summarize_v10_2_15_stage3.py --outroot "$OUTROOT" || FAILURES=$((FAILURES + 1))

echo "Stage 3 runner finished: failures=$FAILURES output=$OUTROOT"
if [[ "$FAILURES" -ne 0 ]]; then
  exit 1
fi
