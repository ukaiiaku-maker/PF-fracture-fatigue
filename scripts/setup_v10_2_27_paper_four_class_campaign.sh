#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$ROOT"

PYTHON_BIN=${PYTHON_BIN:-python}
CONDA_ENV=${CONDA_ENV:-arrhenius-sharp-front-v10}

if [[ "${CONDA_DEFAULT_ENV:-}" != "$CONDA_ENV" ]]; then
  echo "ERROR: activate conda environment '$CONDA_ENV' first" >&2
  exit 2
fi

"$PYTHON_BIN" -m pip install -e ".[test]"
"$PYTHON_BIN" scripts/install_v10_2_27_four_class_registry.py
"$PYTHON_BIN" scripts/install_v10_2_27_four_class_registry.py --check-only
"$PYTHON_BIN" -m pytest -q \
  tests/test_v10_2_27_paper_four_class_campaign.py \
  tests/test_v10_2_27_signed_kernel_coverage.py \
  tests/test_v10_2_27_extended_active_only_atlas.py \
  tests/test_v10_2_27_front_direction_fix.py \
  tests/test_v10_2_27_zero_event_summary.py \
  tests/test_v10_2_27_K_vs_temperature_postprocess.py \
  tests/test_v10_2_27_J_vs_temperature_postprocess.py \
  tests/test_v10_2_27_kernel_resolution.py \
  tests/test_v10_2_27_kernel_runtime_contract.py

echo "v10.2.27 four-class campaign installation and preflight checks passed."
