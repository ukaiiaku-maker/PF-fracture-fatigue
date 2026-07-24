#!/usr/bin/env bash
set -u
set -o pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$ROOT"

PYTHON_BIN=${PYTHON_BIN:-python}
CONDA_ENV=${CONDA_ENV:-arrhenius-fem-czm}
if [[ "${CONDA_DEFAULT_ENV:-}" != "$CONDA_ENV" ]]; then
  echo "ERROR: activate conda environment '$CONDA_ENV'" >&2
  exit 2
fi

OUTROOT=${OUTROOT:-runs/v10_2_24_v913_upper_shelf_top10_50um_theta45_seed3621_v1}
FAMILY_JSON=${FAMILY_JSON:-$ROOT/runtime_inputs/v10_2_17/v10_2_14_active_only_campaign_family.json}
REGISTRY=${REGISTRY:-$ROOT/arrhenius_fracture/data/materials/v10_2_24_v913_top10_upper_shelf_registry.csv}
ONE_D_REFERENCE=${ONE_D_REFERENCE:-$ROOT/arrhenius_fracture/data/materials/v10_2_24_v913_top10_upper_shelf_1d_reference.csv}
SELECTION_RECORD=${SELECTION_RECORD:-$ROOT/arrhenius_fracture/data/materials/v10_2_24_v913_upper_shelf_selection.json}
OPTIONS=${OPTIONS:-"v913_shelf01_0086420_persistent_sites v913_shelf02_0009771_persistent_sites v913_shelf03_0088403_persistent_sites v913_shelf04_0202500_persistent_sites v913_shelf05_0196804_persistent_sites v913_shelf06_0162507_persistent_sites v913_shelf07_0027268_persistent_sites v913_shelf08_0011131_persistent_sites v913_shelf09_0073460_persistent_sites v913_shelf10_0134035_persistent_sites"}
TEMPS=${TEMPS:-"700 800 900 950 1000 1050 1100 1200 1300 1400"}
MAX_JOBS=${MAX_JOBS:-2}
HAZARD_SEED=${HAZARD_SEED:-3621}
TARGET_EXT_UM=${TARGET_EXT_UM:-50}
STEPS=${STEPS:-100000}
THETA=${THETA:-45}
PERSISTENT_SOURCE_MIN_WIDTH_UM=${PERSISTENT_SOURCE_MIN_WIDTH_UM:-0}

for required in "$FAMILY_JSON" "$REGISTRY" "$ONE_D_REFERENCE" "$SELECTION_RECORD"; do
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

"$PYTHON_BIN" - <<'PY'
import json
from pathlib import Path
import arrhenius_fracture
from arrhenius_fracture.sharp_front_v10_2_24 import (
    SELECTION_RECORD,
    VALID_OPTIONS,
)
expected = (Path.cwd() / "arrhenius_fracture").resolve()
actual = Path(arrhenius_fracture.__file__).resolve().parent
if actual != expected:
    raise SystemExit(f"stale/wrong package import: expected {expected}, got {actual}")
if len(VALID_OPTIONS) != 10:
    raise SystemExit(f"expected 10 v10.2.24 options; got {len(VALID_OPTIONS)}")
selection = json.loads(SELECTION_RECORD.read_text())
if selection.get("eligible_candidates") != 32:
    raise SystemExit("upper-shelf selection record must contain the 32-candidate pool")
if len(selection.get("selected", [])) != 10:
    raise SystemExit("upper-shelf selection record must contain exactly 10 transfers")
print(
    f"V10224_PREFLIGHT project={arrhenius_fracture.PROJECT_ID} "
    f"eligible=32 selected={len(VALID_OPTIONS)} package={actual}"
)
PY

mkdir -p "$OUTROOT"
cp "$SELECTION_RECORD" "$OUTROOT/v10_2_24_v913_upper_shelf_selection.json"

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
    "$PYTHON_BIN" -u -m arrhenius_fracture.sharp_front_v10_2_24_audited
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
    tail -n 100 "$log" >&2 || true
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

"$PYTHON_BIN" scripts/compare_v10_2_24_upper_shelf_1d_2d.py \
  --two-d-summary "$OUTROOT/v10_2_22_dbtt_50um_screen_summary.csv" \
  --one-d-reference "$ONE_D_REFERENCE" \
  --out-dir "$OUTROOT/comparison_1d_2d" || {
    echo "ERROR: upper-shelf 1-D/2-D comparison failed" >&2
    exit 1
  }

echo "v10.2.24 upper-shelf screen complete: complete=$complete_count failed=$failed_count output=$OUTROOT"
[[ "$failures" -eq 0 && "$failed_count" -eq 0 ]] || exit 1
