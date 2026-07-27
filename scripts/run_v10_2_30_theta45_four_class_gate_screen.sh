#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$ROOT"

PYTHON_BIN=${PYTHON_BIN:-python}
CONDA_ENV=${CONDA_ENV:-arrhenius-sharp-front-v10}
[[ "${CONDA_DEFAULT_ENV:-}" == "$CONDA_ENV" ]] || {
  echo "ERROR: activate conda environment '$CONDA_ENV'" >&2
  exit 2
}

OUTROOT=${OUTROOT:-runs/v10_2_30_theta45_four_class_T900_20um_gate_screen_v1}
KERNEL_CACHE_ROOT=${KERNEL_CACHE_ROOT:-runs/v10_2_28_kernel_cache}
OPTIONS="v913_paper_peak01_0242980_persistent_sites v913_paper_dbtt01_0202500_persistent_sites v913_paper_weakT01_0129902_persistent_sites v913_paper_ceramic01_0077080_persistent_sites"
TEMPS=900
TARGET_EXT_UM=20
THETA=45
LOADING_RATE_FACTOR=1
DU_M=${DU_M:-2e-7}
BASE_DT_S=${BASE_DT_S:-8.4}
STEPS=${STEPS:-200000}
MAX_JOBS=1
SAVE_SNAPSHOTS=${SAVE_SNAPSHOTS:-4}
SNAPSHOT_COLS=${SNAPSHOT_COLS:-4}
HAZARD_SEED=${HAZARD_SEED:-3621}
SEED_OPTION_STRIDE=${SEED_OPTION_STRIDE:-1000000}
SEED_TEMPERATURE_STRIDE=${SEED_TEMPERATURE_STRIDE:-1009}
PERSISTENT_SOURCE_MIN_WIDTH_UM=${PERSISTENT_SOURCE_MIN_WIDTH_UM:-0}
SKIP_FINISHED=${SKIP_FINISHED:-1}
RESTART_INCOMPLETE=${RESTART_INCOMPLETE:-1}
PREFLIGHT_ONLY=${PREFLIGHT_ONLY:-0}

if [[ "$PREFLIGHT_ONLY" != 1 && -e "$OUTROOT" && "$SKIP_FINISHED" != 1 ]]; then
  echo "ERROR: screen output exists and SKIP_FINISHED!=1: $OUTROOT" >&2
  exit 2
fi

env \
  OUTROOT="$OUTROOT" \
  KERNEL_CACHE_ROOT="$KERNEL_CACHE_ROOT" \
  LOADING_RATE_FACTOR="$LOADING_RATE_FACTOR" \
  DU_M="$DU_M" \
  BASE_DT_S="$BASE_DT_S" \
  THETA="$THETA" \
  OPTIONS="$OPTIONS" \
  TEMPS="$TEMPS" \
  TARGET_EXT_UM="$TARGET_EXT_UM" \
  STEPS="$STEPS" \
  MAX_JOBS="$MAX_JOBS" \
  SAVE_SNAPSHOTS="$SAVE_SNAPSHOTS" \
  SNAPSHOT_COLS="$SNAPSHOT_COLS" \
  HAZARD_SEED="$HAZARD_SEED" \
  SEED_OPTION_STRIDE="$SEED_OPTION_STRIDE" \
  SEED_TEMPERATURE_STRIDE="$SEED_TEMPERATURE_STRIDE" \
  PERSISTENT_SOURCE_MIN_WIDTH_UM="$PERSISTENT_SOURCE_MIN_WIDTH_UM" \
  SKIP_FINISHED="$SKIP_FINISHED" \
  RESTART_INCOMPLETE="$RESTART_INCOMPLETE" \
  PREFLIGHT_ONLY="$PREFLIGHT_ONLY" \
  bash scripts/run_v10_2_30_paper_four_class_orientation_rate.sh

if [[ "$PREFLIGHT_ONLY" == 1 ]]; then
  echo "Gate screen preflight complete: $OUTROOT"
  exit 0
fi

OUTROOT="$OUTROOT" "$PYTHON_BIN" - <<'PY'
import csv
import json
import math
import os
from pathlib import Path

root = Path(os.environ["OUTROOT"]).resolve()
expected_options = [
    "v913_paper_peak01_0242980_persistent_sites",
    "v913_paper_dbtt01_0202500_persistent_sites",
    "v913_paper_weakT01_0129902_persistent_sites",
    "v913_paper_ceramic01_0077080_persistent_sites",
]
lock = json.loads(
    (root / "v10_2_30_hazard_energy_gate_campaign_lock.json").read_text()
)
assert lock["model_entry"] == (
    "arrhenius_fracture."
    "sharp_front_v10_2_30_hazard_energy_gated_audited"
)
assert lock["options"] == expected_options
assert lock["temperatures_K"] == [900.0]
assert lock["planned_case_count"] == 4
assert lock["hazard_energy_gate"] is True
assert lock["absolute_athermal_Gc"] is False
assert lock["gate_resolution"] == "every_internal_Strang_microstep"
assert math.isclose(lock["target_crack_extension_um"], 20.0)
assert math.isclose(lock["crystal_theta_deg"], 45.0)
assert math.isclose(lock["loading_rate_factor"], 1.0)

with (root / "v10_2_27_case_seed_map.csv").open(newline="") as stream:
    rows = list(csv.DictReader(stream))
assert len(rows) == 4
assert [row["option"] for row in rows] == expected_options

summary = []
for row in rows:
    option = row["option"]
    seed = int(row["seed"])
    case = root / option / f"T900K_th45_seed{seed}"
    assert (case / "COMPLETE").is_file(), case
    assert not (case / "RUN_FAILED").exists(), case

    status = json.loads((case / "stage3_case_status.json").read_text())
    assert status["complete"] is True
    assert float(status["projected_extension_um"]) >= 19.5

    transfer = json.loads(
        (case / "v10_2_27_paper_four_class_parameter_transfer.json").read_text()
    )
    assert transfer["selected_option"] == option
    assert transfer["persistent_sites"] is True
    assert transfer["finite_source_inventory"] is False
    assert transfer["source_refresh"] is False
    assert transfer["explicit_recovery"] is False

    audit = json.loads(
        (case / "v10_2_30_hazard_energy_gate_audit.json").read_text()
    )
    assert audit["event_initiation"] == "Arrhenius_first_passage_only"
    assert audit["absolute_athermal_Gc"] is False
    assert audit["gate_resolution"] == "every_internal_Strang_microstep"
    assert audit["fixed_DeltaK_energy_scaling"] == "(K_event/K_probe)^2"

    events = json.loads(
        (case / "stochastic_avalanche_geometry_events.json").read_text()
    )
    assert events, case
    accepted_total = 0.0
    proposed_total = 0.0
    minimum_gate = 1.0
    for event in events:
        assert event["hazard_energy_gate_active"] is True
        gate = event["hazard_energy_gate"]
        proposed = float(gate["proposed_event_advance_m"])
        accepted = float(gate["accepted_event_advance_m"])
        available = float(gate["energy_available_integrated_J_per_m"])
        dissipated = float(gate["energy_dissipated_integrated_J_per_m"])
        assert proposed > 0.0
        assert accepted > 0.0
        assert accepted <= proposed * (1.0 + 1.0e-10)
        assert float(gate["gamma_rel"]) > 0.0
        assert float(gate["DeltaG_cleave_eff_eV"]) > 0.0
        tolerance = 1.0e-10 * max(abs(available), abs(dissipated), 1.0)
        assert dissipated <= available + tolerance
        assert gate["integrated_energy_balance_pass"] is True
        accepted_total += accepted
        proposed_total += proposed
        minimum_gate = min(
            minimum_gate,
            float(gate.get("effective_event_gate_fraction", 1.0)),
        )

    summary.append(
        {
            "option": option,
            "seed": seed,
            "projected_extension_um": float(status["projected_extension_um"]),
            "event_count": len(events),
            "proposed_event_length_total_um": proposed_total * 1.0e6,
            "accepted_event_length_total_um": accepted_total * 1.0e6,
            "accepted_over_proposed": accepted_total / proposed_total,
            "minimum_effective_gate_fraction": minimum_gate,
        }
    )

output = {
    "schema": "v10.2.30_theta45_four_class_gate_screen_acceptance_v1",
    "all_four_parameterizations_verified": True,
    "absolute_athermal_Gc": False,
    "event_energy_balance_pass": True,
    "records": summary,
}
(root / "v10_2_30_gate_screen_acceptance.json").write_text(
    json.dumps(output, indent=2, sort_keys=True) + "\n"
)
print(json.dumps(output, indent=2, sort_keys=True))
PY

echo "PASS: v10.2.30 four-class gate screen accepted: $OUTROOT"
