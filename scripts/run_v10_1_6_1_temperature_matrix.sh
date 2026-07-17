#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "$SCRIPT_DIR/.." && pwd)
cd "$REPO_ROOT"

PYTHON_BIN=${PYTHON_BIN:-python}
OUTROOT=${OUTROOT:-runs/v10_1_6_emergent_temperature_matrix_50um_v1}

"$PYTHON_BIN" scripts/repair_v10_1_6_matrix_audit_aliases.py --root "$OUTROOT"
exec bash scripts/run_v10_1_6_temperature_matrix.sh
