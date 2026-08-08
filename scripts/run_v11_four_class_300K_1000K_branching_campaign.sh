#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$ROOT"
PYTHON_BIN=${PYTHON_BIN:-/opt/homebrew/Caskroom/miniconda/base/envs/arrhenius-sharp-front-v10/bin/python}
CAMPAIGN_ROOT=${CAMPAIGN_ROOT:-$ROOT/runs/v11_four_class_300K_1000K_theta30_seed3621_1000um}
MAX_JOBS=${MAX_JOBS:-2}
KERNEL_CACHE_ROOT=${KERNEL_CACHE_ROOT:-$ROOT/runs/v11_direct_kernel_cache_causal}
FAMILY_JSON=${FAMILY_JSON:-$KERNEL_CACHE_ROOT/e2e456c75935a60a56ccbd9f3b7392036db704de3c2ec6b88e6f0e7bfb127070/family.json}
[[ -x "$PYTHON_BIN" ]] || { echo "ERROR: missing Python: $PYTHON_BIN" >&2; exit 2; }
[[ -f "$FAMILY_JSON" ]] || { echo "ERROR: missing kernel family: $FAMILY_JSON" >&2; exit 2; }
IMPORT_PATH=$($PYTHON_BIN -c 'import pathlib,arrhenius_fracture; print(pathlib.Path(arrhenius_fracture.__file__).resolve().parent)')
[[ "$IMPORT_PATH" == "$ROOT/arrhenius_fracture" ]] || { echo "ERROR: wrong import: $IMPORT_PATH" >&2; exit 2; }
export CONDA_ENV=arrhenius-sharp-front-v10 CONDA_DEFAULT_ENV=arrhenius-sharp-front-v10 PYTHON_BIN KERNEL_CACHE_ROOT
exec "$PYTHON_BIN" scripts/supervise_v11_four_class_branching_campaign.py --root "$CAMPAIGN_ROOT" --family "$FAMILY_JSON" --max-jobs "$MAX_JOBS"
