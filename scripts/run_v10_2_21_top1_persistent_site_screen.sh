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

MODE=${MODE:-smoke}
FAMILY_JSON=${FAMILY_JSON:-$ROOT/runtime_inputs/v10_2_17/v10_2_14_active_only_campaign_family.json}
PARAMETER_REGISTRY=${PARAMETER_REGISTRY:-$ROOT/arrhenius_fracture/data/materials/v10_2_21_v912_top1_persistent_site_registry.csv}
OPTION=${OPTION:-v912_top1_peak_persistent_sites}
THETA=${THETA:-45}
MAX_JOBS=${MAX_JOBS:-2}
BASE_HAZARD_SEED=${BASE_HAZARD_SEED:-2721}
SKIP_FINISHED=${SKIP_FINISHED:-1}
NO_PLOTS=${NO_PLOTS:-1}
PRINT_EVERY=${PRINT_EVERY:-50}

case "$MODE" in
  smoke)
    TEMPS=${TEMPS:-"700 800 900 1000 1100 1200"}
    TARGET_EXT_UM=${TARGET_EXT_UM:-10}
    STEPS=${STEPS:-100000}
    OUTROOT=${OUTROOT:-runs/v10_2_21_top1_persistent_sites_smoke_10um_theta45_v1}
    ;;
  full)
    TEMPS=${TEMPS:-"300 400 500 600 700 800 900 1000 1100 1200"}
    TARGET_EXT_UM=${TARGET_EXT_UM:-50}
    STEPS=${STEPS:-200000}
    OUTROOT=${OUTROOT:-runs/v10_2_21_top1_persistent_sites_50um_theta45_v1}
    ;;
  *)
    echo "ERROR: MODE must be smoke or full" >&2
    exit 2
    ;;
esac

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
PLAN="$OUTROOT/top1_persistent_site_plan.tsv"

TEMPS="$TEMPS" THETA="$THETA" OUTROOT="$OUTROOT" OPTION="$OPTION" \
PARAMETER_REGISTRY="$PARAMETER_REGISTRY" BASE_HAZARD_SEED="$BASE_HAZARD_SEED" \
"$PYTHON_BIN" - <<'PY' > "$PLAN"
import os
from pathlib import Path
from arrhenius_fracture.parameter_registry_v9111 import select_option

temps = [float(v) for v in os.environ["TEMPS"].split()]
option = os.environ["OPTION"]
selected = select_option(option, os.environ["PARAMETER_REGISTRY"], canonical_stage3_only=False)
root = Path(os.environ["OUTROOT"])
theta = float(os.environ["THETA"])
base = int(os.environ["BASE_HAZARD_SEED"])
print("option_key\tcandidate_id\ttemperature_K\thazard_seed\tcase_root")
for temperature in temps:
    seed = base + int(round(temperature))
    tag = f"T{int(round(temperature))}K_th{theta:g}_seed{seed}"
    print(f"{option}\t{selected.candidate_id}\t{temperature:g}\t{seed}\t{root / tag}")
PY

N_CASES=$(($(wc -l < "$PLAN") - 1))
if [[ "$N_CASES" -lt 1 ]]; then
  echo "ERROR: no cases in plan" >&2
  exit 2
fi

cat <<EOF
v10.2.21 top-ranked persistent-site DBTT/peak screen
  base mechanics:      frozen v10.2.18/v10.2.17 signed 2-D stack
  candidate:           v912_targeted_local_peak_013476_0368
  source inventory:    OFF
  source refresh:      OFF
  persistent sites:    rho_site * c_arc * r_tip * w_eff
  backstress gate:     ON
  dynamic blunting:    ON
  advance resharpening: ON
  explicit recovery:   OFF
  temperatures:        $TEMPS
  target:              $TARGET_EXT_UM um
  theta:               $THETA deg
  max jobs:            $MAX_JOBS
  output:              $OUTROOT
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
  local option=$1 candidate=$2 temperature=$3 seed=$4 case_root=$5
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
      echo "SKIP verified complete: T=${temperature}K seed=$seed"
      return 0
    fi
  fi

  rm -f "$case_root/RUN_FAILED" "$case_root/exit_code.txt"
  local cmd=(
    "$PYTHON_BIN" -u -m arrhenius_fracture.sharp_front_v10_2_21
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

  echo "START: candidate=$candidate T=${temperature}K seed=$seed"
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
    echo "incomplete_persistent_site_screen" > "$case_root/RUN_FAILED"
    return 1
  fi
  rm -f "$case_root/RUN_FAILED"
}

exec 3< "$PLAN"
IFS=$'\t' read -r _header <&3
while IFS=$'\t' read -r option candidate temperature seed case_root <&3; do
  [[ -n "$option" ]] || continue
  wait_for_slot
  run_case "$option" "$candidate" "$temperature" "$seed" "$case_root" &
  PIDS+=("$!")
  LABELS+=("T${temperature}K/seed${seed}")
done
exec 3<&-

while [[ ${#PIDS[@]} -gt 0 ]]; do
  sleep 2
  reap_finished
done

echo "Persistent-site screen finished: failures=$FAILURES output=$OUTROOT"
if [[ "$FAILURES" -ne 0 ]]; then
  exit 1
fi

OUTROOT="$OUTROOT" "$PYTHON_BIN" - <<'PY'
import csv, json, os
from pathlib import Path
import numpy as np

root = Path(os.environ["OUTROOT"])
E = 410.0e9
nu = 0.28
Eprime = E / (1.0 - nu**2)
rows = []
for case in sorted(root.glob("T*K_th*_seed*")):
    summary_path = case / "summary.json"
    if not summary_path.is_file():
        continue
    summary = json.loads(summary_path.read_text())[0]
    T = float(summary["T"])
    K0 = summary.get("Kc_first_MPa_sqrt_m")
    K0 = float(K0) if K0 is not None else float("nan")
    Jc = (K0 * 1.0e6) ** 2 / Eprime if np.isfinite(K0) else float("nan")
    step_files = sorted(case.glob("steps_*K.csv"))
    work = float("nan")
    K_end = float("nan")
    if len(step_files) == 1:
        data = np.genfromtxt(step_files[0], delimiter=",", names=True, dtype=float)
        data = np.atleast_1d(data)
        names = set(data.dtype.names or ())
        if {"Uapp_m", "Ftop_N"} <= names:
            u = np.asarray(data["Uapp_m"], float)
            f = np.asarray(data["Ftop_N"], float)
            ok = np.isfinite(u) & np.isfinite(f)
            if np.count_nonzero(ok) >= 2:
                work = float(np.trapz(f[ok], u[ok]))
        if "KJ_Pa_sqrtm" in names:
            k = np.asarray(data["KJ_Pa_sqrtm"], float)
            k = k[np.isfinite(k)]
            if k.size:
                K_end = float(k[-1] / 1.0e6)
    rows.append({
        "temperature_K": T,
        "K_initial_MPa_sqrt_m": K0,
        "J_at_cleavage_J_per_m2": Jc,
        "K_end_MPa_sqrt_m": K_end,
        "load_displacement_work_J_per_unit_thickness": work,
        "n_advances": summary.get("n_advances"),
        "mode_label_legacy_tip_only": summary.get("mode"),
        "case_root": str(case),
    })
rows.sort(key=lambda r: r["temperature_K"])
out = root / "v10_2_21_top1_temperature_summary.csv"
with out.open("w", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=list(rows[0]) if rows else ["temperature_K"])
    writer.writeheader()
    writer.writerows(rows)
print(f"Wrote {out}")
PY
