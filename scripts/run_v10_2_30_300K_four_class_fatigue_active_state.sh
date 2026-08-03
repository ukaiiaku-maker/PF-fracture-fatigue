#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$ROOT"

PATCH_DIR="$ROOT/scripts/v10230_active_state_runtime"
export PYTHONPATH="$PATCH_DIR:$ROOT${PYTHONPATH:+:$PYTHONPATH}"
export V10230_ACTIVE_STATE_BLOCK_CONTROL=1
export V10230_VHCF_RELATIVE_CYCLE_TOL=${V10230_VHCF_RELATIVE_CYCLE_TOL:-1e-4}

python - <<'PY'
from arrhenius_fracture import active_state_block_control_v10230 as patch

patch.install_active_state_block_control()
audit = patch.audit_payload()
if audit["installed"] is not True:
    raise SystemExit("ERROR: active-state block control did not install")
if audit["cumulative_flux_ledgers_are_block_limiters"] is not False:
    raise SystemExit("ERROR: cumulative flux ledgers remain active block limiters")
print("v10.2.30 active-state VHCF block control verified")
print("  limits=cleavage_clock,mobile_pz,stored_pz")
print("  emitted/escaped totals=diagnostic-only")
PY

exec bash scripts/run_v10_2_30_300K_four_class_fatigue.sh "$@"
