#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$ROOT"

CONDA_ENV=${CONDA_ENV:-arrhenius-sharp-front-v10}
[[ "${CONDA_DEFAULT_ENV:-}" == "$CONDA_ENV" ]] || {
  echo "ERROR: activate conda environment '$CONDA_ENV'" >&2
  exit 2
}

bash scripts/validate_v10_2_30_hazard_energy_gate.sh
python -m pytest -q \
  tests/test_v10_4_bulk_peierls_taylor.py \
  tests/test_v10_4_provenance.py \
  tests/test_v10_4_1_detailed_balance.py
