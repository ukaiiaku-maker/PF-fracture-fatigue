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

: "${REFERENCE_ROOT:?Set REFERENCE_ROOT to the matching monotonic case containing steps_TTTTK.csv}"

OUTROOT=${OUTROOT:-runs/v10_2_30_three_deltaK_energy_gate_qualification_v1}
KERNEL_CACHE_ROOT=${KERNEL_CACHE_ROOT:-runs/v10_2_28_kernel_cache}
FAMILY_JSON=${FAMILY_JSON:-}
PARAMETER_OPTION=${PARAMETER_OPTION:-v913_paper_dbtt01_0202500_persistent_sites}
TEMPERATURE_K=${TEMPERATURE_K:-300}
R_RATIO=${R_RATIO:-0.1}
FREQUENCY_HZ=${FREQUENCY_HZ:-1000}
DELTA_K_FRACTIONS=${DELTA_K_FRACTIONS:-"0.55 0.75 0.95"}
HAZARD_SEED=${HAZARD_SEED:-1720}
HORIZON=${HORIZON:-1e10}
TARGET_EXT_UM=${TARGET_EXT_UM:-25}
STEPS=${STEPS:-20000}
TARGET_INCREMENT=${TARGET_INCREMENT:-0.10}
TARGET_DB=${TARGET_DB:-0.10}
ENERGY_GATE_TRIAL_FRACTION=${ENERGY_GATE_TRIAL_FRACTION:-0.10}
CONVERGENCE_FRACTION=${CONVERGENCE_FRACTION:-0.95}
CONVERGENCE_TRIAL_FRACTION=${CONVERGENCE_TRIAL_FRACTION:-0.05}
CONVERGENCE_REL_TOL=${CONVERGENCE_REL_TOL:-0.10}
REQUIRE_BRACKET=${REQUIRE_BRACKET:-1}
REQUIRE_TRUNCATION=${REQUIRE_TRUNCATION:-0}
REQUIRE_HIGH_DRIVE_FULL_PROPOSAL=${REQUIRE_HIGH_DRIVE_FULL_PROPOSAL:-1}
FORCE=${FORCE:-0}

mkdir -p "$OUTROOT"

"$PYTHON_BIN" scripts/install_v10_2_27_four_class_registry.py
"$PYTHON_BIN" scripts/install_v10_2_27_four_class_registry.py --check-only

"$PYTHON_BIN" -m pytest -q --tb=short \
  tests/test_v10_2_30_hazard_energy_gate.py \
  tests/test_v10_2_30_entry_wiring.py \
  tests/test_v10_2_29_event_cycle_accounting.py \
  tests/test_v10_2_29_coupled_transient_prepost.py

if [[ -z "$FAMILY_JSON" ]]; then
  resolution="$OUTROOT/v10_2_30_kernel_resolution.json"
  "$PYTHON_BIN" scripts/ensure_v10_2_28_signed_kernel.py \
    --theta-deg 30 \
    --target-extension-um "$TARGET_EXT_UM" \
    --branching-mode single_front \
    --maximum-fronts 1 \
    --process-zone-length-um 50 \
    --process-zone-bins 80 \
    --mesh-nx 36 \
    --mesh-ny 72 \
    --tip-h-fine-um 1 \
    --tip-ratio 1.20 \
    --da-phys-um 5 \
    --mode auto \
    --cache-root "$KERNEL_CACHE_ROOT" \
    --json > "$resolution.tmp"
  mv "$resolution.tmp" "$resolution"
  FAMILY_JSON=$(RESOLUTION="$resolution" "$PYTHON_BIN" - <<'PY'
import json
import os
from pathlib import Path
resolution = json.loads(Path(os.environ["RESOLUTION"]).read_text())
family = Path(resolution["family"]).resolve()
if not family.is_file():
    raise SystemExit(f"ERROR: resolved family is missing: {family}")
print(family)
PY
  )
fi

[[ -f "$FAMILY_JSON" ]] || {
  echo "ERROR: missing FAMILY_JSON=$FAMILY_JSON" >&2
  exit 2
}

REFERENCE_STEPS="$REFERENCE_ROOT/steps_$(printf '%04d' "$TEMPERATURE_K")K.csv"
[[ -f "$REFERENCE_STEPS" ]] || {
  echo "ERROR: missing monotonic reference $REFERENCE_STEPS" >&2
  exit 2
}

DELTA_K_CRIT_MPA=$(REFERENCE_STEPS="$REFERENCE_STEPS" R_RATIO="$R_RATIO" "$PYTHON_BIN" - <<'PY'
import os
import numpy as np
path = os.environ["REFERENCE_STEPS"]
R = float(os.environ["R_RATIO"])
rows = np.atleast_1d(np.genfromtxt(path, delimiter=",", names=True))
fired = rows[np.asarray(rows["n_fire"], float) > 0.0]
if fired.size == 0:
    raise SystemExit("ERROR: monotonic reference has no first-passage event")
Kcrit = float(np.atleast_1d(fired)[0]["KJ_Pa_sqrtm"]) / 1.0e6
print(f"{(1.0 - R) * Kcrit:.17g}")
PY
)

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
export KERNEL_STRICT_FAMILY_OVERRIDE=1
export SIGNED_KERNEL_FAMILY_JSON="$FAMILY_JSON"
export PERSISTENT_SOURCE_MIN_WIDTH_UM=${PERSISTENT_SOURCE_MIN_WIDTH_UM:-0}
export V10230_ENERGY_GATE_ENABLED=1
export V10230_ENERGY_GATE_BISECTION_ITERATIONS=${V10230_ENERGY_GATE_BISECTION_ITERATIONS:-24}
export V10230_ENERGY_GATE_RELATIVE_TOL=${V10230_ENERGY_GATE_RELATIVE_TOL:-1e-8}
export V10230_ENERGY_GATE_ABSOLUTE_TOL_J_PER_M=${V10230_ENERGY_GATE_ABSOLUTE_TOL_J_PER_M:-1e-12}

common_args=(
  --signed-kernel-family "$FAMILY_JSON"
  --mode 2d
  --temperatures "$TEMPERATURE_K"
  --nx 36 --ny 72
  --dt 8.4 --n-stagger 2
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
  --crystal-theta-deg 30
  --crystal-material w
  --j-decomposition cluster
  --max-fronts 1
  --crack-backend sharp_wake
  --dU 2e-7
  --fatigue-cycles
  --fatigue-hold-load
  --R "$R_RATIO"
  --frequency-Hz "$FREQUENCY_HZ"
  --cycle-block-mode hazard_limited
  --min-block-cycles 1e-6
  --target-dB "$TARGET_DB"
  --target-dN-store "$TARGET_INCREMENT"
  --target-dN-emit "$TARGET_INCREMENT"
  --target-dN-mobile "$TARGET_INCREMENT"
  --target-dN-escape "$TARGET_INCREMENT"
  --target-dN-peierls "$TARGET_INCREMENT"
  --target-dN-taylor "$TARGET_INCREMENT"
  --n-phase 48
  --max-da-per-block-um 5
  --adaptive-events --adaptive-event-target 0.2
  --print-every 100
  --save-snapshots 0
  --no-plots
  --parameter-option "$PARAMETER_OPTION"
)

run_case() {
  local fraction=$1
  local trial_fraction=$2
  local label=$3
  local deltaK
  deltaK=$(FRACTION="$fraction" DELTA_K_CRIT_MPA="$DELTA_K_CRIT_MPA" "$PYTHON_BIN" - <<'PY'
import os
fraction = float(os.environ["FRACTION"])
critical = float(os.environ["DELTA_K_CRIT_MPA"])
if not (0.0 < fraction <= 1.0):
    raise SystemExit("ERROR: DeltaK fraction must lie in (0,1]")
print(f"{fraction * critical:.17g}")
PY
  )
  local ftag
  local ttag
  ftag=$(printf '%g' "$fraction" | tr '.' 'p')
  ttag=$(printf '%g' "$trial_fraction" | tr '.' 'p')
  local case_out="$OUTROOT/$label/fraction_${ftag}/trial_${ttag}/seed_${HAZARD_SEED}"
  if [[ "$FORCE" == 1 ]]; then
    rm -rf "$case_out"
  fi
  if [[ -f "$case_out/v10_2_30_fixed_deltaK_control.json" ]]; then
    echo "SKIP existing case: $case_out"
    return
  fi
  mkdir -p "$case_out"
  CASE_OUT="$case_out" FRACTION="$fraction" DELTAK="$deltaK" TRIAL="$trial_fraction" \
  OPTION="$PARAMETER_OPTION" TEMPERATURE="$TEMPERATURE_K" SEED="$HAZARD_SEED" \
  "$PYTHON_BIN" - <<'PY'
import json
import os
from pathlib import Path
payload = {
    "schema": "v10.2.30_energy_gate_qualification_case",
    "parameter_option": os.environ["OPTION"],
    "temperature_K": float(os.environ["TEMPERATURE"]),
    "deltaK_fraction": float(os.environ["FRACTION"]),
    "target_deltaK_MPa_sqrt_m": float(os.environ["DELTAK"]),
    "energy_gate_trial_fraction": float(os.environ["TRIAL"]),
    "hazard_seed": int(os.environ["SEED"]),
}
Path(os.environ["CASE_OUT"]).joinpath("qualification_case.json").write_text(
    json.dumps(payload, indent=2, sort_keys=True) + "\n"
)
PY
  echo
  echo "CASE: label=$label fraction=$fraction DeltaK=$deltaK trial_fraction=$trial_fraction seed=$HAZARD_SEED"
  env \
    CLEAVAGE_HAZARD_SEED="$HAZARD_SEED" \
    V10230_ENERGY_GATE_TRIAL_FRACTION="$trial_fraction" \
    "$PYTHON_BIN" -u -m \
      arrhenius_fracture.sharp_front_v10_2_30_fixed_deltaK \
      "${common_args[@]}" \
      --target-deltaK-MPa-sqrt-m "$deltaK" \
      --steps "$STEPS" \
      --cycles-max "$HORIZON" \
      --block-cycles "$HORIZON" \
      --max-block-cycles "$HORIZON" \
      --out "$case_out" \
      2>&1 | tee "$case_out/run.log"
  test -s "$case_out/v10_2_30_fixed_deltaK_control.json"
  test -s "$case_out/v10_2_30_hazard_energy_gate_audit.json"
}

for fraction in $DELTA_K_FRACTIONS; do
  run_case "$fraction" "$ENERGY_GATE_TRIAL_FRACTION" primary
done
run_case "$CONVERGENCE_FRACTION" "$CONVERGENCE_TRIAL_FRACTION" convergence

analysis_args=("$OUTROOT")
if [[ "$REQUIRE_BRACKET" == 1 ]]; then
  analysis_args+=(--require-bracket)
fi
if [[ "$REQUIRE_TRUNCATION" == 1 ]]; then
  analysis_args+=(--require-truncation)
fi
"$PYTHON_BIN" scripts/analyze_v10_2_30_energy_gated_qualification.py "${analysis_args[@]}" \
  | tee "$OUTROOT/qualification_analysis.log"

OUTROOT="$OUTROOT" PRIMARY_TRIAL="$ENERGY_GATE_TRIAL_FRACTION" \
REFINED_TRIAL="$CONVERGENCE_TRIAL_FRACTION" FRACTION="$CONVERGENCE_FRACTION" \
REL_TOL="$CONVERGENCE_REL_TOL" REQUIRE_FULL="$REQUIRE_HIGH_DRIVE_FULL_PROPOSAL" \
"$PYTHON_BIN" - <<'PY'
import json
import math
import os
from pathlib import Path
root = Path(os.environ["OUTROOT"])
fraction = float(os.environ["FRACTION"])
primary_trial = float(os.environ["PRIMARY_TRIAL"])
refined_trial = float(os.environ["REFINED_TRIAL"])
rel_tol = float(os.environ["REL_TOL"])
require_full = int(os.environ["REQUIRE_FULL"])

def tag(value):
    return f"{value:g}".replace(".", "p")

def committed(label, trial):
    path = root / label / f"fraction_{tag(fraction)}" / f"trial_{tag(trial)}"
    controls = list(path.rglob("v10_2_30_fixed_deltaK_control.json"))
    if len(controls) != 1:
        raise SystemExit(f"ERROR: expected one control below {path}, found {len(controls)}")
    case = controls[0].parent
    events_path = case / "hazard_energy_gated_events_v10_2_30.json"
    events = json.loads(events_path.read_text()) if events_path.is_file() else []
    return case, [
        row for row in events
        if bool(row.get("inserted", False)) and float(row.get("event_advance_m", 0.0)) > 0.0
    ]

case_a, events_a = committed("primary", primary_trial)
case_b, events_b = committed("convergence", refined_trial)
errors = []
if not events_a or not events_b:
    errors.append("convergence fraction did not propagate in both trial-fraction runs")
else:
    proposal_a = float(events_a[0]["stochastic_proposed_event_length_m"])
    proposal_b = float(events_b[0]["stochastic_proposed_event_length_m"])
    actual_a = float(events_a[0]["event_advance_m"])
    actual_b = float(events_b[0]["event_advance_m"])
    if not math.isclose(proposal_a, proposal_b, rel_tol=1e-12, abs_tol=1e-15):
        errors.append("common-seed stochastic proposal changed between numerical trials")
    relative = abs(actual_a - actual_b) / max(abs(actual_a), abs(actual_b), 1e-300)
    if relative > rel_tol:
        errors.append(f"first committed event failed trial-fraction convergence: {relative} > {rel_tol}")
    if require_full and actual_a < proposal_a * (1.0 - 1e-8):
        errors.append("high-drive primary case did not recover the full stochastic proposal")
    if require_full and actual_b < proposal_b * (1.0 - 1e-8):
        errors.append("high-drive refined case did not recover the full stochastic proposal")

payload = {
    "schema": "v10.2.30_event_length_trial_fraction_convergence",
    "fraction": fraction,
    "primary_case": str(case_a),
    "refined_case": str(case_b),
    "primary_trial_fraction": primary_trial,
    "refined_trial_fraction": refined_trial,
    "relative_tolerance": rel_tol,
    "errors": errors,
    "pass": not errors,
}
(root / "v10_2_30_event_length_convergence.json").write_text(
    json.dumps(payload, indent=2, sort_keys=True) + "\n"
)
print(json.dumps(payload, indent=2, sort_keys=True))
if errors:
    raise SystemExit(1)
PY

echo "V10_2_30_THREE_DELTAK_QUALIFICATION_COMPLETE: $OUTROOT"
