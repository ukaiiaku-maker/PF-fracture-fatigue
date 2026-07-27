#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$ROOT"

CONDA_ENV=${CONDA_ENV:-arrhenius-sharp-front-v10}
if [[ "${CONDA_DEFAULT_ENV:-}" != "$CONDA_ENV" ]]; then
  echo "ERROR: activate conda environment '$CONDA_ENV'" >&2
  exit 2
fi

python -m compileall -q \
  arrhenius_fracture/hazard_energy_gate_v10230.py \
  arrhenius_fracture/hazard_energy_observer_v10230.py \
  arrhenius_fracture/hazard_energy_observed_engine_v10230.py \
  arrhenius_fracture/hazard_energy_differential_engine_v10230.py \
  arrhenius_fracture/hazard_energy_backend_audit_v10230.py \
  arrhenius_fracture/sharp_front_v10_2_30_hazard_energy_gated.py \
  arrhenius_fracture/sharp_front_v10_2_30_hazard_energy_gated_audited.py \
  tests/test_v10_2_30_hazard_energy_gate.py \
  tests/test_v10_2_30_differential_gate_contract.py \
  tests/test_v10_2_30_dynamic_hazard_dissipation.py

bash -n \
  scripts/run_v10_2_28_paper_four_class_1000um_orientation_rate.sh \
  scripts/run_v10_2_28_theta45_loading_rate_sweep.sh \
  scripts/run_v10_2_29_300K_growth_gate.sh \
  scripts/run_v10_2_29_300K_validation.sh

python -m pytest -q \
  tests/test_v10_2_28_direct_kernel_provider.py \
  tests/test_v10_2_28_four_class_1000um_launcher.py \
  tests/test_v10_2_28_orientation_concurrent_mktemp.py \
  tests/test_v10_2_28_projected_kernel_coordinate.py \
  tests/test_v10_2_28_theta45_loading_rate_sweep.py \
  tests/test_v10_2_29_controller_delegate.py \
  tests/test_v10_2_29_driver_cycle_patch.py \
  tests/test_v10_2_29_event_cycle_accounting.py \
  tests/test_v10_2_29_fatigue_entry_contract.py \
  tests/test_v10_2_29_fatigue_growth_extractor.py \
  tests/test_v10_2_29_persistent_cycle_audit.py \
  tests/test_v10_2_29_validation_scripts.py \
  tests/test_v10_2_30_hazard_energy_gate.py \
  tests/test_v10_2_30_differential_gate_contract.py \
  tests/test_v10_2_30_dynamic_hazard_dissipation.py
