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

: "${PEAK_REFERENCE_ROOT:?Set PEAK_REFERENCE_ROOT to the matching monotonic peak case}"
: "${DBTT_REFERENCE_ROOT:?Set DBTT_REFERENCE_ROOT to the matching monotonic DBTT case}"
: "${WEAKT_REFERENCE_ROOT:?Set WEAKT_REFERENCE_ROOT to the matching monotonic weak-T case}"
: "${CERAMIC_REFERENCE_ROOT:?Set CERAMIC_REFERENCE_ROOT to the matching monotonic ceramic-like case}"

OUTROOT=${OUTROOT:-runs/v10_2_30_four_class_three_deltaK_energy_gate_qualification_v1}
KERNEL_CACHE_ROOT=${KERNEL_CACHE_ROOT:-runs/v10_2_28_kernel_cache}
FAMILY_JSON=${FAMILY_JSON:-}
TEMPERATURE_K=${TEMPERATURE_K:-300}
R_RATIO=${R_RATIO:-0.1}
FREQUENCY_HZ=${FREQUENCY_HZ:-1000}
DELTA_K_FRACTIONS=${DELTA_K_FRACTIONS:-"0.55 0.75 0.95"}
CONVERGENCE_FRACTION=${CONVERGENCE_FRACTION:-0.95}
HORIZON=${HORIZON:-1e10}
TARGET_EXT_UM=${TARGET_EXT_UM:-25}
STEPS=${STEPS:-20000}
TARGET_INCREMENT=${TARGET_INCREMENT:-0.10}
TARGET_DB=${TARGET_DB:-0.10}
ENERGY_GATE_TRIAL_FRACTION=${ENERGY_GATE_TRIAL_FRACTION:-0.10}
CONVERGENCE_TRIAL_FRACTION=${CONVERGENCE_TRIAL_FRACTION:-0.05}
CONVERGENCE_REL_TOL=${CONVERGENCE_REL_TOL:-0.10}
REQUIRE_HIGH_DRIVE_FULL_PROPOSAL=${REQUIRE_HIGH_DRIVE_FULL_PROPOSAL:-1}
FORCE=${FORCE:-0}

PEAK_DELTA_K_FRACTIONS=${PEAK_DELTA_K_FRACTIONS:-$DELTA_K_FRACTIONS}
DBTT_DELTA_K_FRACTIONS=${DBTT_DELTA_K_FRACTIONS:-$DELTA_K_FRACTIONS}
WEAKT_DELTA_K_FRACTIONS=${WEAKT_DELTA_K_FRACTIONS:-$DELTA_K_FRACTIONS}
CERAMIC_DELTA_K_FRACTIONS=${CERAMIC_DELTA_K_FRACTIONS:-$DELTA_K_FRACTIONS}

PEAK_CONVERGENCE_FRACTION=${PEAK_CONVERGENCE_FRACTION:-$CONVERGENCE_FRACTION}
DBTT_CONVERGENCE_FRACTION=${DBTT_CONVERGENCE_FRACTION:-$CONVERGENCE_FRACTION}
WEAKT_CONVERGENCE_FRACTION=${WEAKT_CONVERGENCE_FRACTION:-$CONVERGENCE_FRACTION}
CERAMIC_CONVERGENCE_FRACTION=${CERAMIC_CONVERGENCE_FRACTION:-$CONVERGENCE_FRACTION}

BASE_HAZARD_SEED=${BASE_HAZARD_SEED:-1720}
SEED_OPTION_STRIDE=${SEED_OPTION_STRIDE:-1000003}
PEAK_HAZARD_SEED=${PEAK_HAZARD_SEED:-$BASE_HAZARD_SEED}
DBTT_HAZARD_SEED=${DBTT_HAZARD_SEED:-$((BASE_HAZARD_SEED + SEED_OPTION_STRIDE))}
WEAKT_HAZARD_SEED=${WEAKT_HAZARD_SEED:-$((BASE_HAZARD_SEED + 2 * SEED_OPTION_STRIDE))}
CERAMIC_HAZARD_SEED=${CERAMIC_HAZARD_SEED:-$((BASE_HAZARD_SEED + 3 * SEED_OPTION_STRIDE))}

mkdir -p "$OUTROOT"

run_family() {
  local label=$1
  local option=$2
  local reference_root=$3
  local fractions=$4
  local convergence_fraction=$5
  local hazard_seed=$6
  local family_out="$OUTROOT/$label"

  [[ -d "$reference_root" ]] || {
    echo "ERROR: missing $label monotonic reference root: $reference_root" >&2
    exit 2
  }

  mkdir -p "$family_out"
  echo
  echo "QUALIFY: class=$label option=$option T=${TEMPERATURE_K}K seed=$hazard_seed"
  echo "  reference=$reference_root"
  echo "  fractions=$fractions convergence_fraction=$convergence_fraction"

  REFERENCE_ROOT="$reference_root" \
  OUTROOT="$family_out" \
  KERNEL_CACHE_ROOT="$KERNEL_CACHE_ROOT" \
  FAMILY_JSON="$FAMILY_JSON" \
  PARAMETER_OPTION="$option" \
  TEMPERATURE_K="$TEMPERATURE_K" \
  R_RATIO="$R_RATIO" \
  FREQUENCY_HZ="$FREQUENCY_HZ" \
  DELTA_K_FRACTIONS="$fractions" \
  HAZARD_SEED="$hazard_seed" \
  HORIZON="$HORIZON" \
  TARGET_EXT_UM="$TARGET_EXT_UM" \
  STEPS="$STEPS" \
  TARGET_INCREMENT="$TARGET_INCREMENT" \
  TARGET_DB="$TARGET_DB" \
  ENERGY_GATE_TRIAL_FRACTION="$ENERGY_GATE_TRIAL_FRACTION" \
  CONVERGENCE_FRACTION="$convergence_fraction" \
  CONVERGENCE_TRIAL_FRACTION="$CONVERGENCE_TRIAL_FRACTION" \
  CONVERGENCE_REL_TOL="$CONVERGENCE_REL_TOL" \
  REQUIRE_BRACKET=1 \
  REQUIRE_TRUNCATION=0 \
  REQUIRE_HIGH_DRIVE_FULL_PROPOSAL="$REQUIRE_HIGH_DRIVE_FULL_PROPOSAL" \
  FORCE="$FORCE" \
  bash scripts/run_v10_2_30_three_deltaK_energy_gate_qualification.sh \
    2>&1 | tee "$family_out/four_class_launcher.log"
}

run_family \
  peak \
  v913_paper_peak01_0242980_persistent_sites \
  "$PEAK_REFERENCE_ROOT" \
  "$PEAK_DELTA_K_FRACTIONS" \
  "$PEAK_CONVERGENCE_FRACTION" \
  "$PEAK_HAZARD_SEED"

run_family \
  dbtt \
  v913_paper_dbtt01_0202500_persistent_sites \
  "$DBTT_REFERENCE_ROOT" \
  "$DBTT_DELTA_K_FRACTIONS" \
  "$DBTT_CONVERGENCE_FRACTION" \
  "$DBTT_HAZARD_SEED"

run_family \
  weakT \
  v913_paper_weakT01_0129902_persistent_sites \
  "$WEAKT_REFERENCE_ROOT" \
  "$WEAKT_DELTA_K_FRACTIONS" \
  "$WEAKT_CONVERGENCE_FRACTION" \
  "$WEAKT_HAZARD_SEED"

run_family \
  ceramic \
  v913_paper_ceramic01_0077080_persistent_sites \
  "$CERAMIC_REFERENCE_ROOT" \
  "$CERAMIC_DELTA_K_FRACTIONS" \
  "$CERAMIC_CONVERGENCE_FRACTION" \
  "$CERAMIC_HAZARD_SEED"

"$PYTHON_BIN" scripts/analyze_v10_2_30_energy_gated_qualification.py "$OUTROOT" \
  | tee "$OUTROOT/four_class_qualification_analysis.log"

OUTROOT="$OUTROOT" TEMPERATURE_K="$TEMPERATURE_K" R_RATIO="$R_RATIO" \
FREQUENCY_HZ="$FREQUENCY_HZ" "$PYTHON_BIN" - <<'PY'
import json
import math
import os
from pathlib import Path

root = Path(os.environ["OUTROOT"])
expected_temperature = float(os.environ["TEMPERATURE_K"])
expected_R = float(os.environ["R_RATIO"])
expected_frequency = float(os.environ["FREQUENCY_HZ"])
expected = {
    "peak": "v913_paper_peak01_0242980_persistent_sites",
    "dbtt": "v913_paper_dbtt01_0202500_persistent_sites",
    "weakT": "v913_paper_weakT01_0129902_persistent_sites",
    "ceramic": "v913_paper_ceramic01_0077080_persistent_sites",
}
summary_path = root / "v10_2_30_qualification_summary.json"
summary = json.loads(summary_path.read_text())
rows = list(summary.get("cases", []))
errors = []

observed_options = {str(row.get("parameter_option")) for row in rows}
expected_options = set(expected.values())
if observed_options != expected_options:
    errors.append(
        "canonical option set mismatch: "
        f"expected={sorted(expected_options)} observed={sorted(observed_options)}"
    )

class_results = {}
class_seeds = {}
for label, option in expected.items():
    class_rows = [row for row in rows if row.get("parameter_option") == option]
    propagated = [row for row in class_rows if int(row.get("committed_events", 0)) > 0]
    censored = [row for row in class_rows if int(row.get("committed_events", 0)) == 0]
    failed = [row for row in class_rows if row.get("pass") is not True]
    seeds = {int(row.get("hazard_seed", 0)) for row in class_rows}
    class_seeds[label] = sorted(seeds)

    if not class_rows:
        errors.append(f"{label}: no qualification cases found")
    if not propagated:
        errors.append(f"{label}: no propagated qualification case")
    if not censored:
        errors.append(f"{label}: no censored qualification case")
    if failed:
        errors.append(f"{label}: {len(failed)} case audits failed")
    if len(seeds) != 1:
        errors.append(f"{label}: expected one class-local common seed, observed {sorted(seeds)}")

    for row in class_rows:
        if not math.isclose(float(row.get("temperature_K", math.nan)), expected_temperature):
            errors.append(f"{label}: temperature mismatch")
            break
        if not math.isclose(float(row.get("R", math.nan)), expected_R, rel_tol=0.0, abs_tol=1e-14):
            errors.append(f"{label}: R-ratio mismatch")
            break
        if not math.isclose(
            float(row.get("frequency_Hz", math.nan)),
            expected_frequency,
            rel_tol=0.0,
            abs_tol=1e-9,
        ):
            errors.append(f"{label}: frequency mismatch")
            break
        if not math.isclose(float(row.get("crystal_theta_deg", math.nan)), 30.0):
            errors.append(f"{label}: theta is not 30 degrees")
            break

    convergence_path = root / label / "v10_2_30_event_length_convergence.json"
    if not convergence_path.is_file():
        errors.append(f"{label}: missing trial-fraction convergence audit")
        convergence = {}
    else:
        convergence = json.loads(convergence_path.read_text())
        if convergence.get("pass") is not True:
            errors.append(f"{label}: trial-fraction convergence failed")

    class_results[label] = {
        "parameter_option": option,
        "case_count": len(class_rows),
        "propagated_case_count": len(propagated),
        "censored_case_count": len(censored),
        "truncated_event_count": sum(
            int(row.get("truncated_events", 0)) for row in class_rows
        ),
        "hazard_seeds": sorted(seeds),
        "convergence_pass": convergence.get("pass") is True,
    }

nonempty_seed_sets = [tuple(values) for values in class_seeds.values() if values]
if len(set(nonempty_seed_sets)) != len(nonempty_seed_sets):
    errors.append("class seed namespaces are not unique")

total_truncated = sum(int(row.get("truncated_events", 0)) for row in rows)
if total_truncated <= 0:
    errors.append("no positive committed event was energy-truncated in the four-class gate")

payload = {
    "schema": "v10.2.30_four_class_three_deltaK_energy_gate_qualification",
    "root": str(root),
    "temperature_K": expected_temperature,
    "R": expected_R,
    "frequency_Hz": expected_frequency,
    "canonical_options_complete": observed_options == expected_options,
    "total_case_count": len(rows),
    "total_truncated_event_count": total_truncated,
    "class_results": class_results,
    "errors": errors,
    "pass": not errors,
}
(root / "v10_2_30_four_class_qualification_gate.json").write_text(
    json.dumps(payload, indent=2, sort_keys=True) + "\n"
)
print(json.dumps(payload, indent=2, sort_keys=True))
if errors:
    raise SystemExit(1)
PY

echo "V10_2_30_FOUR_CLASS_THREE_DELTAK_QUALIFICATION_COMPLETE: $OUTROOT"
