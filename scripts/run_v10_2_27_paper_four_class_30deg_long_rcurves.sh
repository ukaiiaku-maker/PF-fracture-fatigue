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

OUTROOT=${OUTROOT:-runs/v10_2_27_paper_four_class_1000um_theta30_varseed_base3621_v1}
FAMILY_JSON=${FAMILY_JSON:-$ROOT/runtime_inputs/v10_2_17/v10_2_14_active_only_campaign_family.json}
REGISTRY=${REGISTRY:-$ROOT/arrhenius_fracture/data/materials/v10_2_27_paper_four_class_registry.csv}
SELECTION=${SELECTION:-$ROOT/arrhenius_fracture/data/materials/v10_2_27_paper_four_class_selection.json}
OPTIONS=${OPTIONS:-"v913_paper_peak01_0242980_persistent_sites v913_paper_dbtt01_0202500_persistent_sites v913_paper_weakT01_0257068_persistent_sites v913_paper_ceramic01_0189364_persistent_sites"}
TEMPS=${TEMPS:-"300 600 800 900 950 1000 1050 1100 1150 1200 1250 1300"}
MAX_JOBS=${MAX_JOBS:-2}
HAZARD_SEED=${HAZARD_SEED:-3621}
SEED_OPTION_STRIDE=${SEED_OPTION_STRIDE:-1000000}
SEED_TEMPERATURE_STRIDE=${SEED_TEMPERATURE_STRIDE:-1009}
TARGET_EXT_UM=${TARGET_EXT_UM:-1000}
STEPS=${STEPS:-2000000}
THETA=${THETA:-30}
SAVE_SNAPSHOTS=${SAVE_SNAPSHOTS:-20}
SNAPSHOT_COLS=${SNAPSHOT_COLS:-5}
PERSISTENT_SOURCE_MIN_WIDTH_UM=${PERSISTENT_SOURCE_MIN_WIDTH_UM:-0}
SKIP_FINISHED=${SKIP_FINISHED:-1}
RESTART_INCOMPLETE=${RESTART_INCOMPLETE:-1}

"$PYTHON_BIN" scripts/install_v10_2_27_four_class_registry.py
"$PYTHON_BIN" scripts/install_v10_2_27_four_class_registry.py --check-only

for required in "$FAMILY_JSON" "$REGISTRY" "$SELECTION"; do
  [[ -f "$required" ]] || { echo "ERROR: missing $required" >&2; exit 2; }
done

case "$MAX_JOBS" in
  ''|*[!0-9]*) echo "ERROR: MAX_JOBS must be a positive integer" >&2; exit 2 ;;
esac
[[ "$MAX_JOBS" -ge 1 ]] || { echo "ERROR: MAX_JOBS must be >= 1" >&2; exit 2; }

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

OUTROOT="$OUTROOT" OPTIONS="$OPTIONS" TEMPS="$TEMPS" TARGET_EXT_UM="$TARGET_EXT_UM" \
THETA="$THETA" HAZARD_SEED="$HAZARD_SEED" SAVE_SNAPSHOTS="$SAVE_SNAPSHOTS" \
SNAPSHOT_COLS="$SNAPSHOT_COLS" REGISTRY="$REGISTRY" SELECTION="$SELECTION" \
FAMILY_JSON="$FAMILY_JSON" STEPS="$STEPS" SEED_OPTION_STRIDE="$SEED_OPTION_STRIDE" \
SEED_TEMPERATURE_STRIDE="$SEED_TEMPERATURE_STRIDE" \
CLEAVAGE_EVENT_MIN_FACTOR="$CLEAVAGE_EVENT_MIN_FACTOR" \
CLEAVAGE_EVENT_MAX_FACTOR="$CLEAVAGE_EVENT_MAX_FACTOR" \
CLEAVAGE_EVENT_SUBSEGMENT_FRACTION="$CLEAVAGE_EVENT_SUBSEGMENT_FRACTION" \
"$PYTHON_BIN" - <<'PY'
import csv
import json
import os
from pathlib import Path

root = Path(os.environ["OUTROOT"]).resolve()
options = os.environ["OPTIONS"].split()
temperatures = [float(value) for value in os.environ["TEMPS"].split()]
expected_options = [
    "v913_paper_peak01_0242980_persistent_sites",
    "v913_paper_dbtt01_0202500_persistent_sites",
    "v913_paper_weakT01_0257068_persistent_sites",
    "v913_paper_ceramic01_0189364_persistent_sites",
]
if options != expected_options:
    raise SystemExit(f"ERROR: OPTIONS must retain canonical order: {expected_options}")

registry_path = Path(os.environ["REGISTRY"]).resolve()
with registry_path.open(newline="") as stream:
    registry_rows = list(csv.DictReader(stream))
registry_options = [row["option_key"] for row in registry_rows]
if registry_options != expected_options:
    raise SystemExit(
        f"ERROR: generated registry option order mismatch: {registry_options}"
    )

selection_path = Path(os.environ["SELECTION"]).resolve()
selection = json.loads(selection_path.read_text())
if selection.get("canonical_option_order") != expected_options:
    raise SystemExit("ERROR: selection metadata option order mismatch")

base = int(os.environ["HAZARD_SEED"])
option_stride = int(os.environ["SEED_OPTION_STRIDE"])
temperature_stride = int(os.environ["SEED_TEMPERATURE_STRIDE"])
seed_rows = []
for option_index, option in enumerate(options):
    for temperature_index, temperature in enumerate(temperatures):
        seed_rows.append(
            {
                "option_index": option_index,
                "temperature_index": temperature_index,
                "option": option,
                "temperature_K": temperature,
                "seed": (
                    base
                    + option_index * option_stride
                    + temperature_index * temperature_stride
                ),
            }
        )
seeds = [row["seed"] for row in seed_rows]
if len(seeds) != len(set(seeds)):
    raise SystemExit("ERROR: case-seed mapping is not unique")

seed_map = root / "v10_2_27_case_seed_map.csv"
with seed_map.open("w", newline="") as stream:
    writer = csv.DictWriter(stream, fieldnames=list(seed_rows[0]))
    writer.writeheader()
    writer.writerows(seed_rows)

payload = {
    "schema": "v10.2.27_paper_four_class_30deg_long_rcurve_campaign_v1",
    "model_entry": "arrhenius_fracture.sharp_front_v10_2_27_audited",
    "parameter_registry": str(registry_path),
    "selection_record": str(selection_path),
    "signed_kernel_family": str(Path(os.environ["FAMILY_JSON"]).resolve()),
    "options": options,
    "temperatures_K": temperatures,
    "planned_case_count": len(seed_rows),
    "target_crack_extension_um": float(os.environ["TARGET_EXT_UM"]),
    "maximum_steps": int(os.environ["STEPS"]),
    "crystal_theta_deg": float(os.environ["THETA"]),
    "stochastic_cleavage_hazard": True,
    "cleavage_hazard_mode": "exponential",
    "cleavage_event_length_mode": "threshold_scaled",
    "cleavage_event_min_factor": float(os.environ["CLEAVAGE_EVENT_MIN_FACTOR"]),
    "cleavage_event_max_factor": float(os.environ["CLEAVAGE_EVENT_MAX_FACTOR"]),
    "cleavage_event_subsegment_fraction": float(
        os.environ["CLEAVAGE_EVENT_SUBSEGMENT_FRACTION"]
    ),
    "base_cleavage_hazard_seed": base,
    "common_random_numbers": False,
    "case_seed_rule": (
        "base_seed + option_index*seed_option_stride + "
        "temperature_index*seed_temperature_stride"
    ),
    "seed_option_stride": option_stride,
    "seed_temperature_stride": temperature_stride,
    "case_seed_map": str(seed_map.resolve()),
    "save_snapshots": int(os.environ["SAVE_SNAPSHOTS"]),
    "snapshot_columns": int(os.environ["SNAPSHOT_COLS"]),
    "persistent_source_min_width_um": float(
        os.environ.get("PERSISTENT_SOURCE_MIN_WIDTH_UM", "0")
    ),
    "physics_contract": {
        "persistent_sites": True,
        "finite_source_inventory": False,
        "source_depletion_on_emission": False,
        "source_refresh_on_crack_advance": False,
        "explicit_recovery": False,
        "backstress_limited_emission": True,
        "physical_front_width_grid_independent": True,
        "signed_mobile_and_retained_state": True,
        "mobile_shield_fraction": 0.0,
        "wake_shielding": False,
        "maximum_fronts": 1,
    },
    "requested_primary_outputs": [
        "field snapshots throughout crack growth",
        "event-resolved K versus crack extension",
        "per-candidate temperature overlays",
        "per-temperature four-class overlays",
        "target-aware K checkpoints",
    ],
}
(root / "v10_2_27_campaign_manifest.json").write_text(
    json.dumps(payload, indent=2, sort_keys=True) + "\n"
)
PY

candidate_for_option() {
  case "$1" in
    v913_paper_peak01_0242980_persistent_sites)
      printf '%s\n' "v913_zeroD_sobol_0242980"
      ;;
    v913_paper_dbtt01_0202500_persistent_sites)
      printf '%s\n' "v913_zeroD_sobol_0202500"
      ;;
    v913_paper_weakT01_0257068_persistent_sites)
      printf '%s\n' "v913_zeroD_sobol_0257068"
      ;;
    v913_paper_ceramic01_0189364_persistent_sites)
      printf '%s\n' "v913_zeroD_sobol_0189364"
      ;;
    *)
      echo "ERROR: unknown option: $1" >&2
      return 2
      ;;
  esac
}

write_case_contract() {
  local case_root=$1
  local option=$2
  local candidate=$3
  local T=$4
  local seed=$5
  CASE_ROOT="$case_root" OPTION="$option" CANDIDATE="$candidate" T="$T" \
  CASE_SEED="$seed" TARGET_EXT_UM="$TARGET_EXT_UM" THETA="$THETA" \
  SAVE_SNAPSHOTS="$SAVE_SNAPSHOTS" SNAPSHOT_COLS="$SNAPSHOT_COLS" \
  "$PYTHON_BIN" - <<'PY'
import json
import os
from pathlib import Path

payload = {
    "schema": "v10.2.27_case_contract_v1",
    "option": os.environ["OPTION"],
    "candidate_id": os.environ["CANDIDATE"],
    "temperature_K": float(os.environ["T"]),
    "seed": int(os.environ["CASE_SEED"]),
    "target_extension_um": float(os.environ["TARGET_EXT_UM"]),
    "theta_deg": float(os.environ["THETA"]),
    "save_snapshots": int(os.environ["SAVE_SNAPSHOTS"]),
    "snapshot_columns": int(os.environ["SNAPSHOT_COLS"]),
    "stochastic_cleavage_hazard": True,
    "common_random_numbers": False,
}
path = Path(os.environ["CASE_ROOT"]) / "v10_2_27_case_contract.json"
path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
PY
}

verified_complete() {
  local case_root=$1
  local option=$2
  local candidate=$3
  local T=$4
  local seed=$5
  CASE_ROOT="$case_root" OPTION="$option" CANDIDATE="$candidate" T="$T" \
  CASE_SEED="$seed" TARGET_EXT_UM="$TARGET_EXT_UM" THETA="$THETA" \
  SAVE_SNAPSHOTS="$SAVE_SNAPSHOTS" SNAPSHOT_COLS="$SNAPSHOT_COLS" \
  "$PYTHON_BIN" - <<'PY'
import json
import math
import os
from pathlib import Path

root = Path(os.environ["CASE_ROOT"])
required = [
    root / "COMPLETE",
    root / "stage3_case_status.json",
    root / "v10_2_27_case_contract.json",
    root / "v10_2_27_paper_four_class_parameter_transfer.json",
    root / "command.sh",
]
if any(not path.is_file() for path in required) or (root / "RUN_FAILED").exists():
    raise SystemExit(1)

expected = {
    "option": os.environ["OPTION"],
    "candidate_id": os.environ["CANDIDATE"],
    "temperature_K": float(os.environ["T"]),
    "seed": int(os.environ["CASE_SEED"]),
    "target_extension_um": float(os.environ["TARGET_EXT_UM"]),
    "theta_deg": float(os.environ["THETA"]),
    "save_snapshots": int(os.environ["SAVE_SNAPSHOTS"]),
    "snapshot_columns": int(os.environ["SNAPSHOT_COLS"]),
    "stochastic_cleavage_hazard": True,
    "common_random_numbers": False,
}
contract = json.loads((root / "v10_2_27_case_contract.json").read_text())
for key, value in expected.items():
    actual = contract.get(key)
    if isinstance(value, float):
        if actual is None or not math.isclose(float(actual), value, rel_tol=0.0, abs_tol=1e-12):
            raise SystemExit(1)
    elif actual != value:
        raise SystemExit(1)

status = json.loads((root / "stage3_case_status.json").read_text())
if status.get("complete") is not True:
    raise SystemExit(1)
if not math.isclose(
    float(status.get("target_extension_um", float("nan"))),
    expected["target_extension_um"],
    rel_tol=0.0,
    abs_tol=1e-12,
):
    raise SystemExit(1)
if not math.isclose(
    float(status.get("temperature_K", float("nan"))),
    expected["temperature_K"],
    rel_tol=0.0,
    abs_tol=1e-12,
):
    raise SystemExit(1)

transfer = json.loads(
    (root / "v10_2_27_paper_four_class_parameter_transfer.json").read_text()
)
if transfer.get("selected_option") != expected["option"]:
    raise SystemExit(1)
if transfer.get("selected_candidate") != expected["candidate_id"]:
    raise SystemExit(1)
if transfer.get("mechanics_changed") is not False:
    raise SystemExit(1)
if transfer.get("persistent_sites") is not True:
    raise SystemExit(1)
if transfer.get("finite_source_inventory") is not False:
    raise SystemExit(1)
if transfer.get("source_refresh") is not False:
    raise SystemExit(1)
if transfer.get("explicit_recovery") is not False:
    raise SystemExit(1)
if transfer.get("front_width_grid_independent") is not True:
    raise SystemExit(1)

command = (root / "command.sh").read_text()
tokens = [
    f"--parameter-option {expected['option']}",
    f"--temperatures {os.environ['T']}",
    f"--target-crack-extension-um {os.environ['TARGET_EXT_UM']}",
    f"--crystal-theta-deg {os.environ['THETA']}",
    f"CLEAVAGE_HAZARD_SEED={expected['seed']}",
]
if any(token not in command for token in tokens):
    raise SystemExit(1)
raise SystemExit(0)
PY
}

run_case() {
  local option=$1
  local T=$2
  local case_seed=$3
  local candidate
  candidate=$(candidate_for_option "$option") || return $?
  local case_root="$OUTROOT/$option/T${T}K_th${THETA}_seed${case_seed}"
  local log="$case_root/run.log"

  if [[ "$SKIP_FINISHED" == 1 && -d "$case_root" ]]; then
    if verified_complete "$case_root" "$option" "$candidate" "$T" "$case_seed"; then
      echo "SKIP verified complete: option=${option} T=${T}K seed=${case_seed}"
      return 0
    fi
    if [[ -f "$case_root/COMPLETE" ]]; then
      echo "ERROR: complete-looking case failed contract verification: $case_root" >&2
      return 3
    fi
  fi

  if [[ -d "$case_root" ]]; then
    if [[ "$RESTART_INCOMPLETE" != 1 ]]; then
      echo "ERROR: incomplete case exists and RESTART_INCOMPLETE!=1: $case_root" >&2
      return 3
    fi
    echo "REMOVE incomplete case before clean restart: $case_root"
    rm -rf "$case_root"
  fi
  mkdir -p "$case_root"
  write_case_contract "$case_root" "$option" "$candidate" "$T" "$case_seed"

  local cmd=(
    "$PYTHON_BIN" -u -m arrhenius_fracture.sharp_front_v10_2_27_audited
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
    --print-every 200
    --save-snapshots "$SAVE_SNAPSHOTS"
    --snapshot-cols "$SNAPSHOT_COLS"
    --out "$case_root"
  )

  {
    echo '#!/usr/bin/env bash'
    printf 'PERSISTENT_SOURCE_MIN_WIDTH_UM=%q ' "$PERSISTENT_SOURCE_MIN_WIDTH_UM"
    printf 'CLEAVAGE_HAZARD_SEED=%q ' "$case_seed"
    printf '%q ' "${cmd[@]}"
    printf '\n'
  } > "$case_root/command.sh"
  chmod +x "$case_root/command.sh"

  echo "START: option=${option} T=${T}K seed=${case_seed} target=${TARGET_EXT_UM}um theta=${THETA}"
  env \
    CLEAVAGE_HAZARD_SEED="$case_seed" \
    PERSISTENT_SOURCE_MIN_WIDTH_UM="$PERSISTENT_SOURCE_MIN_WIDTH_UM" \
    "${cmd[@]}" > "$log" 2>&1
  local rc=$?
  echo "$rc" > "$case_root/exit_code.txt"
  if [[ "$rc" -ne 0 ]]; then
    echo "simulation_exit_$rc" > "$case_root/RUN_FAILED"
    echo "--- option=${option} T=${T}K seed=${case_seed} failure log tail ---" >&2
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

  if ! verified_complete "$case_root" "$option" "$candidate" "$T" "$case_seed"; then
    echo "postrun_contract_verification_failed" > "$case_root/RUN_FAILED"
    tail -n 100 "$log" >&2 || true
    return 1
  fi

  rm -f "$case_root/RUN_FAILED"
  echo "FINISHED: option=${option} T=${T}K seed=${case_seed}"
}

pids=()
labels=()
failures=0

wait_one() {
  local pid=${pids[0]}
  local label=${labels[0]}
  local rc=0
  wait "$pid" || rc=$?
  if [[ "$rc" -ne 0 ]]; then
    echo "FAILED: $label (exit=$rc)" >&2
    failures=$((failures + 1))
  fi
  pids=("${pids[@]:1}")
  labels=("${labels[@]:1}")
}

option_index=0
for option in $OPTIONS; do
  temperature_index=0
  for T in $TEMPS; do
    case_seed=$((
      HAZARD_SEED
      + option_index * SEED_OPTION_STRIDE
      + temperature_index * SEED_TEMPERATURE_STRIDE
    ))

    while [[ ${#pids[@]} -ge $MAX_JOBS ]]; do
      wait_one
    done
    run_case "$option" "$T" "$case_seed" &
    pids+=("$!")
    labels+=("${option}:T${T}K:seed${case_seed}")
    temperature_index=$((temperature_index + 1))
  done
  option_index=$((option_index + 1))
done
while [[ ${#pids[@]} -gt 0 ]]; do
  wait_one
done

"$PYTHON_BIN" scripts/plot_v10_2_27_paper_four_class_rcurves.py \
  --outroot "$OUTROOT" \
  --target-extension-um "$TARGET_EXT_UM" || {
    echo "ERROR: four-class R-curve postprocessing failed" >&2
    exit 1
  }

OUTROOT="$OUTROOT" "$PYTHON_BIN" - <<'PY'
import csv
import json
import os
from pathlib import Path

root = Path(os.environ["OUTROOT"]).resolve()
manifest = json.loads((root / "v10_2_27_campaign_manifest.json").read_text())
with (root / "v10_2_27_case_seed_map.csv").open(newline="") as stream:
    cases = list(csv.DictReader(stream))

records = []
for case in cases:
    option = case["option"]
    temperature = float(case["temperature_K"])
    temperature_tag = f"{temperature:g}"
    theta_tag = f"{float(manifest['crystal_theta_deg']):g}"
    seed = int(case["seed"])
    case_root = root / option / f"T{temperature_tag}K_th{theta_tag}_seed{seed}"
    status_path = case_root / "stage3_case_status.json"
    status = json.loads(status_path.read_text()) if status_path.is_file() else {}
    records.append(
        {
            "option": option,
            "temperature_K": temperature,
            "seed": seed,
            "case_root": str(case_root),
            "complete": (
                status.get("complete") is True
                and (case_root / "COMPLETE").is_file()
                and not (case_root / "RUN_FAILED").exists()
            ),
            "status": status.get("status", "missing"),
            "projected_extension_um": status.get("projected_extension_um"),
        }
    )

complete = sum(record["complete"] for record in records)
failed = len(records) - complete
payload = {
    "schema": "v10.2.27_campaign_acceptance_v1",
    "planned_cases": len(records),
    "complete_cases": complete,
    "failed_or_incomplete_cases": failed,
    "all_cases_complete": failed == 0,
    "unique_seed_count": len({record["seed"] for record in records}),
    "theta_deg": manifest["crystal_theta_deg"],
    "target_extension_um": manifest["target_crack_extension_um"],
    "records": records,
}
(root / "v10_2_27_campaign_acceptance.json").write_text(
    json.dumps(payload, indent=2, sort_keys=True) + "\n"
)
print(
    f"Campaign acceptance: planned={len(records)} complete={complete} "
    f"failed_or_incomplete={failed}"
)
if failed:
    raise SystemExit(1)
PY

echo "Campaign complete: failures=$failures output=$OUTROOT"
[[ "$failures" -eq 0 ]] || exit 1
