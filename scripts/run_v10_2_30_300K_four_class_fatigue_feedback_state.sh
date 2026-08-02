#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$ROOT"

PATCH_DIR="$ROOT/scripts/v10230_active_state_runtime"
export PYTHONPATH="$PATCH_DIR:$ROOT${PYTHONPATH:+:$PYTHONPATH}"
export V10230_ACTIVE_STATE_BLOCK_CONTROL=1
export V10230_FEEDBACK_STATE_BLOCK_CONTROL=1
export V10230_VHCF_RELATIVE_CYCLE_TOL=${V10230_VHCF_RELATIVE_CYCLE_TOL:-1e-4}
# A positive absolute cutoff bypasses sigma/log-lambda checks in the inherited
# low-hazard branch. Zero keeps those feedback tolerances active for every
# positive cleavage rate while still allowing exactly zero hazard intervals.
export V10229_COUPLED_HAZARD_ABS_DB_TOL=0

python - <<'PY'
from arrhenius_fracture import active_state_block_control_v10230 as active
from arrhenius_fracture import feedback_state_block_control_v10230 as feedback

active.install_active_state_block_control()
feedback.install_feedback_state_block_control()
audit = feedback.audit_payload()
if audit["installed"] is not True:
    raise SystemExit("ERROR: feedback-state block control did not install")
if audit["raw_population_counts_are_block_limiters"] is not False:
    raise SystemExit("ERROR: raw population counts remain block limiters")
print("v10.2.30 feedback-state VHCF block control verified")
print("  outer limit=cleavage_clock")
print("  state feedback=sigma_tip,lambda_cleave through coupled quadrature")
print("  mobile/retained populations=fully evolved and diagnostic")
print("  absolute_dB_bypass=off")
PY

exec bash scripts/run_v10_2_30_300K_four_class_fatigue.sh "$@"
