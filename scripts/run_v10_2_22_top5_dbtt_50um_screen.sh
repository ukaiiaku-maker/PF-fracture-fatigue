#!/usr/bin/env bash
set -u
set -o pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$ROOT"

PYTHON_BIN=${PYTHON_BIN:-python}
CONDA_ENV=${CONDA_ENV:-arrhenius-sharp-front-v10}
if [[ "${CONDA_DEFAULT_ENV:-}" != "$CONDA_ENV" ]]; then
  echo "ERROR: activate conda environment '$CONDA_ENV'" >&2
  exit 2
fi

OUTROOT=${OUTROOT:-runs/v10_2_22_top5_dbtt_physical_width_50um_theta45_v1}
FAMILY_JSON=${FAMILY_JSON:-$ROOT/runtime_inputs/v10_2_17/v10_2_14_active_only_campaign_family.json}
REGISTRY=${REGISTRY:-$ROOT/arrhenius_fracture/data/materials/v10_2_22_v912_top5_persistent_site_registry.csv}
OPTIONS=${OPTIONS:-"v912_top1_peak_persistent_sites v912_peak_0314_persistent_sites v912_peak_0162_persistent_sites v912_peak_0118_persistent_sites v912_plateau_0403_persistent_sites"}
TEMPS=${TEMPS:-"300 400 500 600 700 800 900 1000 1100 1200"}
MAX_JOBS=${MAX_JOBS:-2}
HAZARD_SEED=${HAZARD_SEED:-3621}
TARGET_EXT_UM=${TARGET_EXT_UM:-50}
STEPS=${STEPS:-100000}
THETA=${THETA:-45}
PERSISTENT_SOURCE_MIN_WIDTH_UM=${PERSISTENT_SOURCE_MIN_WIDTH_UM:-0}

for required in "$FAMILY_JSON" "$REGISTRY"; do
  [[ -f "$required" ]] || { echo "ERROR: missing $required" >&2; exit 2; }
done

export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
export PARAMETER_CAMPAIGN=1
export CLEAVAGE_HAZARD_MODE=exponential
export CLEAVAGE_EVENT_LENGTH_MODE=threshold_scaled
export CLEAVAGE_EVENT_MIN_FACTOR=${CLEAVAGE_EVENT_MIN_FACTOR:-0.5}
export CLEAVAGE_EVENT_MAX_FACTOR=${CLEAVAGE_EVENT_MAX_FACTOR:-4.0}
export CLEAVAGE_EVENT_SUBSEGMENT_FRACTION=${CLEAVAGE_EVENT_SUBSEGMENT_FRACTION:-0.1}
export ANISOTROPIC_TRANSPORT_MODE=validated_scalar
export ANISOTROPIC_USE_AVALANCHE_BACKEND=1
export ANISOTROPIC_EMISSION_ENABLED=1
export PERSISTENT_SOURCE_MIN_WIDTH_UM

mkdir -p "$OUTROOT"

run_case() {
  local option=$1
  local T=$2
  local case_root="$OUTROOT/$option/T${T}K_th${THETA}_seed${HAZARD_SEED}"
  local log="$case_root/run.log"
  mkdir -p "$case_root"

  if [[ -f "$case_root/stage3_case_status.json" ]]; then
    local complete
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
      echo "SKIP complete: option=${option} T=${T}K"
      return 0
    fi
  fi

  rm -f "$case_root/RUN_FAILED" "$case_root/exit_code.txt"
  local cmd=(
    "$PYTHON_BIN" -u -m arrhenius_fracture.sharp_front_v10_2_22_audited
    --signed-kernel-family "$FAMILY_JSON"
    --mode 2d
    --parameter-registry "$REGISTRY"
    --parameter-option "$option"
    --temperatures "$T"
    --steps "$STEPS"
    --nx 36 --ny 72
    --dU 2e-7 --dt 8.4 --n-stagger 2
    --tip-h-fine 1e-6 --tip-ratio 1.20
    --da-phys 5e-6
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
    --adaptive-events --adaptive-event-target 0.15
    --print-every 50
    --save-snapshots 0
    --no-plots
    --out "$case_root"
  )

  {
    echo '#!/usr/bin/env bash'
    printf 'PERSISTENT_SOURCE_MIN_WIDTH_UM=%q ' "$PERSISTENT_SOURCE_MIN_WIDTH_UM"
    printf 'CLEAVAGE_HAZARD_SEED=%q ' "$HAZARD_SEED"
    printf '%q ' "${cmd[@]}"
    printf '\n'
  } > "$case_root/command.sh"
  chmod +x "$case_root/command.sh"

  echo "START: option=${option} T=${T}K seed=${HAZARD_SEED}"
  env \
    CLEAVAGE_HAZARD_SEED="$HAZARD_SEED" \
    PERSISTENT_SOURCE_MIN_WIDTH_UM="$PERSISTENT_SOURCE_MIN_WIDTH_UM" \
    "${cmd[@]}" > "$log" 2>&1
  local rc=$?
  echo "$rc" > "$case_root/exit_code.txt"
  if [[ "$rc" -ne 0 ]]; then
    echo "simulation_exit_$rc" > "$case_root/RUN_FAILED"
    echo "--- option=${option} T=${T}K failure log tail ---" >&2
    tail -n 100 "$log" >&2 || true
    echo "--- end failure log tail ---" >&2
    return "$rc"
  fi

  "$PYTHON_BIN" scripts/classify_v10_2_15_stage3_case.py \
    --case-root "$case_root" \
    --target-extension-um "$TARGET_EXT_UM" >> "$log" 2>&1 || {
      echo "classification_failed" > "$case_root/RUN_FAILED"
      tail -n 100 "$log" >&2 || true
      return 1
    }

  local complete
  complete=$(CASE_ROOT="$case_root" "$PYTHON_BIN" - <<'PY'
import json, os
from pathlib import Path
p = Path(os.environ["CASE_ROOT"]) / "stage3_case_status.json"
print("1" if json.loads(p.read_text()).get("complete") is True else "0")
PY
)
  if [[ "$complete" != 1 ]]; then
    echo "incomplete_screen" > "$case_root/RUN_FAILED"
    tail -n 100 "$log" >&2 || true
    return 1
  fi
  rm -f "$case_root/RUN_FAILED"
  echo "FINISHED: option=${option} T=${T}K"
}

pids=()
labels=()
failures=0

reap() {
  local -a next_pids next_labels
  next_pids=()
  next_labels=()
  local i rc
  for ((i=0; i<${#pids[@]}; i++)); do
    if kill -0 "${pids[$i]}" 2>/dev/null; then
      next_pids+=("${pids[$i]}")
      next_labels+=("${labels[$i]}")
    else
      wait "${pids[$i]}"; rc=$?
      if [[ "$rc" -ne 0 ]]; then
        echo "FAILED: ${labels[$i]}" >&2
        failures=$((failures + 1))
      fi
    fi
  done
  pids=()
  labels=()
  if [[ ${#next_pids[@]} -gt 0 ]]; then
    pids=("${next_pids[@]}")
    labels=("${next_labels[@]}")
  fi
}

for option in $OPTIONS; do
  for T in $TEMPS; do
    while [[ ${#pids[@]} -ge $MAX_JOBS ]]; do sleep 2; reap; done
    run_case "$option" "$T" &
    pids+=("$!")
    labels+=("${option}:T${T}K")
  done
done
while [[ ${#pids[@]} -gt 0 ]]; do sleep 2; reap; done

complete_count=$(find "$OUTROOT" -type f -name COMPLETE | wc -l | tr -d ' ')
failed_count=$(find "$OUTROOT" -type f -name RUN_FAILED | wc -l | tr -d ' ')

"$PYTHON_BIN" scripts/plot_v10_2_22_dbtt_rcurves.py --outroot "$OUTROOT" || {
  echo "ERROR: R-curve postprocessing failed" >&2
  exit 1
}

echo "Screen complete: complete=$complete_count failed=$failed_count output=$OUTROOT"
[[ "$failures" -eq 0 && "$failed_count" -eq 0 ]] || exit 1
