#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$ROOT"

PYTHON_BIN=${PYTHON_BIN:-python}
CONDA_ENV=${CONDA_ENV:-arrhenius-sharp-front-v10}
if [[ "${CONDA_DEFAULT_ENV:-}" != "$CONDA_ENV" ]]; then
  echo "ERROR: activate conda environment '$CONDA_ENV'" >&2
  exit 2
fi

OUTROOT=${OUTROOT:-$ROOT/runs/v10_2_27_paper_four_class_1000um_theta30_frontfix_family35710f_varseed3621_v1}
TEMPS=${TEMPS:-"300 600 800 900 950 1000 1050 1100 1150 1200 1250 1300"}
TARGET_EXT_UM=${TARGET_EXT_UM:-1000}
THETA=${THETA:-30}
STEPS=${STEPS:-2000000}
MAX_JOBS=${MAX_JOBS:-2}
HAZARD_SEED=${HAZARD_SEED:-3621}
SEED_OPTION_STRIDE=${SEED_OPTION_STRIDE:-1000000}
SEED_TEMPERATURE_STRIDE=${SEED_TEMPERATURE_STRIDE:-1009}
SAVE_SNAPSHOTS=${SAVE_SNAPSHOTS:-20}
SNAPSHOT_COLS=${SNAPSHOT_COLS:-5}
SKIP_FINISHED=${SKIP_FINISHED:-1}
RESTART_INCOMPLETE=${RESTART_INCOMPLETE:-1}
PERSISTENT_SOURCE_MIN_WIDTH_UM=${PERSISTENT_SOURCE_MIN_WIDTH_UM:-0}
DA_PHYS_UM=${DA_PHYS_UM:-5}
CLEAVAGE_EVENT_MIN_FACTOR=${CLEAVAGE_EVENT_MIN_FACTOR:-0.5}
CLEAVAGE_EVENT_MAX_FACTOR=${CLEAVAGE_EVENT_MAX_FACTOR:-4.0}
KERNEL_MARGIN_EVENTS=${KERNEL_MARGIN_EVENTS:-1}
BRANCHING_MODE=${BRANCHING_MODE:-single_front}
MAX_FRONTS=${MAX_FRONTS:-1}

OPTIONS="v913_paper_peak01_0242980_persistent_sites v913_paper_dbtt01_0202500_persistent_sites v913_paper_weakT01_0129902_persistent_sites v913_paper_ceramic01_0077080_persistent_sites"
OLD_WEAK="v913_paper_weakT01_0257068_persistent_sites"
OLD_CERAMIC="v913_paper_ceramic01_0189364_persistent_sites"
BASE_RUNNER="$ROOT/scripts/run_v10_2_27_paper_four_class_30deg_long_rcurves.sh"

[[ -f "$BASE_RUNNER" ]] || { echo "ERROR: missing base runner: $BASE_RUNNER" >&2; exit 2; }
[[ -d "$OUTROOT" ]] || { echo "ERROR: campaign root does not exist: $OUTROOT" >&2; exit 2; }
if [[ -e "$OUTROOT/$OLD_WEAK" || -e "$OUTROOT/$OLD_CERAMIC" ]]; then
  echo "ERROR: superseded weakT/ceramic directories remain inside the campaign root." >&2
  echo "Archive them outside OUTROOT before launching replacement cases." >&2
  exit 2
fi

FAMILY_JSON=$(
  PYTHON_BIN="$PYTHON_BIN" \
  FAMILY_JSON="${FAMILY_JSON:-}" \
  MECHANICAL_CONFIG="${MECHANICAL_CONFIG:-}" \
  MECHANICAL_PROFILE="${MECHANICAL_PROFILE:-v10_2_27_default_single_front_frontfix}" \
  KERNEL_RESOLUTION_MODE="${KERNEL_RESOLUTION_MODE:-auto}" \
  KERNEL_BUILD_COMMAND="${KERNEL_BUILD_COMMAND:-}" \
  KERNEL_SNAPSHOT_ARCHIVE="${KERNEL_SNAPSHOT_ARCHIVE:-}" \
  KERNEL_LOAD_INVARIANCE_ARCHIVE="${KERNEL_LOAD_INVARIANCE_ARCHIVE:-}" \
  INITIAL_CRACK_LENGTH_UM="${INITIAL_CRACK_LENGTH_UM:-}" \
  KERNEL_INTERACTION_LENGTH_UM="${KERNEL_INTERACTION_LENGTH_UM:-}" \
  TEMPERATURE_DEPENDENT_MECHANICS="${TEMPERATURE_DEPENDENT_MECHANICS:-0}" \
  KERNEL_TEMPERATURE_K="${KERNEL_TEMPERATURE_K:-}" \
  TARGET_EXT_UM="$TARGET_EXT_UM" \
  THETA="$THETA" \
  BRANCHING_MODE="$BRANCHING_MODE" \
  MAX_FRONTS="$MAX_FRONTS" \
  DA_PHYS_UM="$DA_PHYS_UM" \
  CLEAVAGE_EVENT_MIN_FACTOR="$CLEAVAGE_EVENT_MIN_FACTOR" \
  CLEAVAGE_EVENT_MAX_FACTOR="$CLEAVAGE_EVENT_MAX_FACTOR" \
  KERNEL_MARGIN_EVENTS="$KERNEL_MARGIN_EVENTS" \
  bash scripts/resolve_v10_2_27_kernel_for_runner.sh
)
export FAMILY_JSON

echo "Resolved signed-kernel family: $FAMILY_JSON"

"$PYTHON_BIN" scripts/install_v10_2_27_four_class_registry.py
"$PYTHON_BIN" scripts/install_v10_2_27_four_class_registry.py --check-only

OUTROOT="$OUTROOT" TEMPS="$TEMPS" TARGET_EXT_UM="$TARGET_EXT_UM" THETA="$THETA" \
HAZARD_SEED="$HAZARD_SEED" SEED_OPTION_STRIDE="$SEED_OPTION_STRIDE" \
SEED_TEMPERATURE_STRIDE="$SEED_TEMPERATURE_STRIDE" "$PYTHON_BIN" - <<'PY'
import json
import math
import os
from pathlib import Path

root = Path(os.environ["OUTROOT"]).resolve()
temperatures = [float(value) for value in os.environ["TEMPS"].split()]
target = float(os.environ["TARGET_EXT_UM"])
theta = float(os.environ["THETA"])
base = int(os.environ["HAZARD_SEED"])
option_stride = int(os.environ["SEED_OPTION_STRIDE"])
temperature_stride = int(os.environ["SEED_TEMPERATURE_STRIDE"])
accepted = (
    ("v913_paper_peak01_0242980_persistent_sites", "v913_zeroD_sobol_0242980", 0),
    ("v913_paper_dbtt01_0202500_persistent_sites", "v913_zeroD_sobol_0202500", 1),
)
errors = []
for option, candidate, option_index in accepted:
    for temperature_index, temperature in enumerate(temperatures):
        seed = base + option_index * option_stride + temperature_index * temperature_stride
        case = root / option / f"T{temperature:g}K_th{theta:g}_seed{seed}"
        required = (
            case / "COMPLETE",
            case / "stage3_case_status.json",
            case / "v10_2_27_case_contract.json",
            case / "v10_2_27_paper_four_class_parameter_transfer.json",
        )
        missing = [str(path) for path in required if not path.is_file()]
        if missing or (case / "RUN_FAILED").exists():
            errors.append({"case": str(case), "missing": missing, "failed": (case / "RUN_FAILED").exists()})
            continue
        status = json.loads((case / "stage3_case_status.json").read_text())
        contract = json.loads((case / "v10_2_27_case_contract.json").read_text())
        transfer = json.loads((case / "v10_2_27_paper_four_class_parameter_transfer.json").read_text())
        checks = {
            "complete": status.get("complete") is True,
            "option": contract.get("option") == option,
            "candidate": contract.get("candidate_id") == candidate,
            "seed": int(contract.get("seed", -1)) == seed,
            "temperature": math.isclose(float(contract.get("temperature_K", float("nan"))), temperature, rel_tol=0.0, abs_tol=1e-12),
            "target": math.isclose(float(contract.get("target_extension_um", float("nan"))), target, rel_tol=0.0, abs_tol=1e-12),
            "theta": math.isclose(float(contract.get("theta_deg", float("nan"))), theta, rel_tol=0.0, abs_tol=1e-12),
            "transfer_option": transfer.get("selected_option") == option,
            "transfer_candidate": transfer.get("selected_candidate") == candidate,
        }
        if not all(checks.values()):
            errors.append({"case": str(case), "checks": checks})
if errors:
    print(json.dumps(errors, indent=2))
    raise SystemExit("ACCEPTED PEAK/DBTT PREFLIGHT FAILED")
print("Accepted peak/DBTT preflight passed: 24 cases")
PY

PATCHED_RUNNER=$(mktemp "$ROOT/scripts/.v10_2_27_final_four_class_runner.XXXXXX.sh")
trap 'rm -f "$PATCHED_RUNNER"' EXIT

BASE_RUNNER="$BASE_RUNNER" PATCHED_RUNNER="$PATCHED_RUNNER" "$PYTHON_BIN" - <<'PY'
import os
from pathlib import Path

source = Path(os.environ["BASE_RUNNER"])
target = Path(os.environ["PATCHED_RUNNER"])
text = source.read_text()
replacements = {
    "v913_paper_weakT01_0257068_persistent_sites": "v913_paper_weakT01_0129902_persistent_sites",
    "v913_zeroD_sobol_0257068": "v913_zeroD_sobol_0129902",
    "v913_paper_ceramic01_0189364_persistent_sites": "v913_paper_ceramic01_0077080_persistent_sites",
    "v913_zeroD_sobol_0189364": "v913_zeroD_sobol_0077080",
}
for old, new in replacements.items():
    count = text.count(old)
    if count == 0:
        raise SystemExit(f"base runner no longer contains expected token: {old}")
    text = text.replace(old, new)
target.write_text(text)
target.chmod(0o755)
PY

env \
  OUTROOT="$OUTROOT" \
  FAMILY_JSON="$FAMILY_JSON" \
  OPTIONS="$OPTIONS" \
  TEMPS="$TEMPS" \
  TARGET_EXT_UM="$TARGET_EXT_UM" \
  THETA="$THETA" \
  STEPS="$STEPS" \
  MAX_JOBS="$MAX_JOBS" \
  HAZARD_SEED="$HAZARD_SEED" \
  SEED_OPTION_STRIDE="$SEED_OPTION_STRIDE" \
  SEED_TEMPERATURE_STRIDE="$SEED_TEMPERATURE_STRIDE" \
  SAVE_SNAPSHOTS="$SAVE_SNAPSHOTS" \
  SNAPSHOT_COLS="$SNAPSHOT_COLS" \
  SKIP_FINISHED="$SKIP_FINISHED" \
  RESTART_INCOMPLETE="$RESTART_INCOMPLETE" \
  PERSISTENT_SOURCE_MIN_WIDTH_UM="$PERSISTENT_SOURCE_MIN_WIDTH_UM" \
  bash "$PATCHED_RUNNER"

"$PYTHON_BIN" scripts/plot_v10_2_27_paper_four_class_K_vs_temperature.py \
  --outroot "$OUTROOT" \
  --target-extension-um "$TARGET_EXT_UM"

"$PYTHON_BIN" scripts/plot_v10_2_27_paper_four_class_J_energy_vs_temperature.py \
  --outroot "$OUTROOT" \
  --target-extension-um "$TARGET_EXT_UM"

OUTROOT="$OUTROOT" TEMPS="$TEMPS" TARGET_EXT_UM="$TARGET_EXT_UM" THETA="$THETA" \
HAZARD_SEED="$HAZARD_SEED" SEED_OPTION_STRIDE="$SEED_OPTION_STRIDE" \
SEED_TEMPERATURE_STRIDE="$SEED_TEMPERATURE_STRIDE" "$PYTHON_BIN" - <<'PY'
import json
import os
from pathlib import Path
import numpy as np

root = Path(os.environ["OUTROOT"]).resolve()
temperatures = [float(value) for value in os.environ["TEMPS"].split()]
theta = float(os.environ["THETA"])
base = int(os.environ["HAZARD_SEED"])
option_stride = int(os.environ["SEED_OPTION_STRIDE"])
temperature_stride = int(os.environ["SEED_TEMPERATURE_STRIDE"])
replacements = (
    ("v913_paper_weakT01_0129902_persistent_sites", 2),
    ("v913_paper_ceramic01_0077080_persistent_sites", 3),
)
errors = []
for option, option_index in replacements:
    for temperature_index, temperature in enumerate(temperatures):
        seed = base + option_index * option_stride + temperature_index * temperature_stride
        case = root / option / f"T{temperature:g}K_th{theta:g}_seed{seed}"
        audit_path = case / "v10_2_27_energy_ledger_output_audit.json"
        steps = sorted(case.glob("steps_*K.csv"))
        if not (case / "COMPLETE").is_file() or (case / "RUN_FAILED").exists():
            errors.append({"case": str(case), "reason": "not complete"})
            continue
        if not audit_path.is_file() or len(steps) != 1:
            errors.append({"case": str(case), "reason": "missing energy audit or steps CSV"})
            continue
        audit = json.loads(audit_path.read_text())
        if int(audit.get("record_count", 0)) < 1:
            errors.append({"case": str(case), "reason": "empty energy audit"})
            continue
        data = np.genfromtxt(steps[0], delimiter=",", names=True, dtype=float)
        names = set(data.dtype.names or ())
        required = {
            "J_effective_direct_J_per_m2",
            "J_signed_direct_J_per_m2",
            "W_ext_cumulative_J_per_m",
            "U_elastic_J_per_m",
            "W_bulk_plastic_cumulative_J_per_m",
            "W_tip_emit_cumulative_J_per_m",
            "W_fracture_residual_cumulative_J_per_m",
        }
        missing = sorted(required - names)
        if missing:
            errors.append({"case": str(case), "missing_energy_columns": missing})
if errors:
    print(json.dumps(errors, indent=2))
    raise SystemExit("REPLACEMENT ENERGY-LEDGER ACCEPTANCE FAILED")
print("Replacement weakT/ceramic energy-ledger acceptance passed: 24 cases")
PY

"$PYTHON_BIN" scripts/postprocess_v10_2_27_final_four_class.py \
  --outroot "$OUTROOT" \
  --target-extension-um "$TARGET_EXT_UM"

echo "Final replacement campaign accepted: $OUTROOT"
