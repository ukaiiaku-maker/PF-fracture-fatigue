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

SWEEP_ROOT=${SWEEP_ROOT:-runs/v10_2_30_theta45_hazard_energy_loading_rate_sweep_base3621_v1}
KERNEL_CACHE_ROOT=${KERNEL_CACHE_ROOT:-runs/v10_2_28_kernel_cache}
RATE_FACTORS=${RATE_FACTORS:-"1 100 0.01"}
THETA=${THETA:-45}
OPTIONS=${OPTIONS:-"v913_paper_peak01_0242980_persistent_sites v913_paper_dbtt01_0202500_persistent_sites v913_paper_weakT01_0129902_persistent_sites v913_paper_ceramic01_0077080_persistent_sites"}
TEMPS=${TEMPS:-"300 600 800 900 950 1000 1050 1100 1150 1200 1250 1300"}
TARGET_EXT_UM=${TARGET_EXT_UM:-1000}
STEPS=${STEPS:-2000000}
MAX_JOBS=${MAX_JOBS:-1}
SAVE_SNAPSHOTS=${SAVE_SNAPSHOTS:-20}
SNAPSHOT_COLS=${SNAPSHOT_COLS:-5}
HAZARD_SEED=${HAZARD_SEED:-3621}
SEED_OPTION_STRIDE=${SEED_OPTION_STRIDE:-1000000}
SEED_TEMPERATURE_STRIDE=${SEED_TEMPERATURE_STRIDE:-1009}
PERSISTENT_SOURCE_MIN_WIDTH_UM=${PERSISTENT_SOURCE_MIN_WIDTH_UM:-0}
SKIP_FINISHED=${SKIP_FINISHED:-1}
RESTART_INCOMPLETE=${RESTART_INCOMPLETE:-1}
DU_M=${DU_M:-2e-7}
BASE_DT_S=${BASE_DT_S:-8.4}
PREFLIGHT_ONLY=${PREFLIGHT_ONLY:-0}

THETA="$THETA" TARGET_EXT_UM="$TARGET_EXT_UM" MAX_JOBS="$MAX_JOBS" \
"$PYTHON_BIN" - <<'PY'
import math
import os

theta = float(os.environ["THETA"])
target = float(os.environ["TARGET_EXT_UM"])
max_jobs = int(os.environ["MAX_JOBS"])
if not math.isclose(theta, 45.0, rel_tol=0.0, abs_tol=1e-12):
    raise SystemExit("ERROR: this sweep is fixed to theta=45 degrees")
if not math.isclose(target, 1000.0, rel_tol=0.0, abs_tol=1e-12):
    raise SystemExit("ERROR: production rate sweep is fixed to 1000 um projected extension")
if max_jobs != 1:
    raise SystemExit("ERROR: theta45 gated rate sweep requires MAX_JOBS=1")
PY

mkdir -p "$SWEEP_ROOT"

SWEEP_ROOT="$SWEEP_ROOT" RATE_FACTORS="$RATE_FACTORS" OPTIONS="$OPTIONS" \
TEMPS="$TEMPS" THETA="$THETA" TARGET_EXT_UM="$TARGET_EXT_UM" \
HAZARD_SEED="$HAZARD_SEED" SEED_OPTION_STRIDE="$SEED_OPTION_STRIDE" \
SEED_TEMPERATURE_STRIDE="$SEED_TEMPERATURE_STRIDE" DU_M="$DU_M" \
BASE_DT_S="$BASE_DT_S" "$PYTHON_BIN" - <<'PY'
import json
import os
from pathlib import Path

from arrhenius_fracture.loading_rate_v10228 import resolve_loading_rate

root = Path(os.environ["SWEEP_ROOT"]).resolve()
factors = os.environ["RATE_FACTORS"].split()
rates = []
for value in factors:
    spec = resolve_loading_rate(
        float(value),
        nominal_dU_m=float(os.environ["DU_M"]),
        base_dt_s=float(os.environ["BASE_DT_S"]),
    )
    rates.append(
        {
            "loading_rate_factor": spec.loading_rate_factor,
            "rate_tag": spec.rate_tag,
            "nominal_dU_m": spec.nominal_dU_m,
            "base_dt_s": spec.base_dt_s,
            "nominal_dt_s": spec.nominal_dt_s,
            "nominal_opening_rate_m_per_s": spec.nominal_opening_rate_m_per_s,
            "outroot": str((root / spec.rate_tag).resolve()),
        }
    )
if len({item["rate_tag"] for item in rates}) != len(rates):
    raise SystemExit("ERROR: duplicate loading-rate factors")
payload = {
    "schema": "v10.2.30_theta45_hazard_energy_loading_rate_sweep_v1",
    "model_entry": (
        "arrhenius_fracture."
        "sharp_front_v10_2_30_hazard_energy_gated_audited"
    ),
    "hazard_energy_gate": True,
    "absolute_athermal_Gc": False,
    "hazard_dissipation_density": (
        "gamma_rel*m*DeltaG_cleave_eff(T,sigma)/b^2"
    ),
    "anisotropic_hazard_scaling": (
        "sigma_hazard=sigma_physical/sqrt(gamma_rel)"
    ),
    "fixed_DeltaK_energy_scaling": "(K_event/K_probe)^2",
    "gate_resolution": "every_internal_Strang_microstep",
    "crystal_theta_deg": float(os.environ["THETA"]),
    "target_quantity": "projected_ligament_extension",
    "target_crack_extension_um": float(os.environ["TARGET_EXT_UM"]),
    "options": os.environ["OPTIONS"].split(),
    "temperatures_K": [float(value) for value in os.environ["TEMPS"].split()],
    "rates": rates,
    "planned_case_count": (
        len(rates)
        * len(os.environ["OPTIONS"].split())
        * len(os.environ["TEMPS"].split())
    ),
    "common_random_numbers_across_loading_rates": True,
    "base_cleavage_hazard_seed": int(os.environ["HAZARD_SEED"]),
    "seed_option_stride": int(os.environ["SEED_OPTION_STRIDE"]),
    "seed_temperature_stride": int(os.environ["SEED_TEMPERATURE_STRIDE"]),
    "execution": "sequential_rates_and_one_case_at_a_time",
}
(root / "v10_2_30_theta45_loading_rate_sweep_manifest.json").write_text(
    json.dumps(payload, indent=2, sort_keys=True) + "\n"
)
print(json.dumps(payload, indent=2, sort_keys=True))
PY

for factor in $RATE_FACTORS; do
  RATE_TAG=$("$PYTHON_BIN" -m arrhenius_fracture.loading_rate_v10228 \
    --factor "$factor" --dU-m "$DU_M" --base-dt-s "$BASE_DT_S" --format tag)
  RATE_OUTROOT="$SWEEP_ROOT/$RATE_TAG"

  echo
  echo "START RATE: factor=$factor tag=$RATE_TAG outroot=$RATE_OUTROOT"

  env \
    OUTROOT="$RATE_OUTROOT" \
    KERNEL_CACHE_ROOT="$KERNEL_CACHE_ROOT" \
    LOADING_RATE_FACTOR="$factor" \
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

  echo "FINISHED RATE: factor=$factor tag=$RATE_TAG"
done

echo "Theta45 v10.2.30 gated loading-rate sweep complete: $SWEEP_ROOT"
