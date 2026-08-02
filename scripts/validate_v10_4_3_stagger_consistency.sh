#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$ROOT"

PYTHON_BIN=${PYTHON_BIN:-python}

"$PYTHON_BIN" - <<'PY'
from importlib.metadata import version

observed = version("arrhenius-sharp-front-mpz")
expected = "10.4.3"
print(f"Package: {observed}")
if observed != expected:
    raise SystemExit(
        f"ERROR: expected editable package {expected}; run python -m pip install -e . --no-deps"
    )
PY

"$PYTHON_BIN" -m compileall -q arrhenius_fracture scripts tests

"$PYTHON_BIN" -m pytest -q \
  tests/test_v10_4_2_directional_j_positive.py \
  tests/test_v10_4_2_plastic_flow_terminal.py \
  tests/test_v10_4_2_reuse_aware_launcher.py \
  tests/test_v10_4_2_launcher_adapter.py \
  tests/test_v10_4_bulk_peierls_taylor.py \
  tests/test_v10_4_1_detailed_balance.py

GENERATED=$(mktemp "${TMPDIR:-/tmp}/v1043-final-scheduler.XXXXXX.sh")
trap 'rm -f "$GENERATED"' EXIT

"$PYTHON_BIN" scripts/build_v10_4_2_reuse_aware_launcher.py \
  --source scripts/run_v10_2_28_paper_four_class_theta30_1000um.sh \
  --output "$GENERATED"

bash -n "$GENERATED"
grep -q 'RERUN_REQUIRED_STAGGER_TIME_CORRECTION' "$GENERATED"
grep -q 'SKIP_REUSED_VERIFIED' "$GENERATED"
grep -q 'failed_or_incomplete_cases' "$GENERATED"

"$PYTHON_BIN" - <<'PY'
from pathlib import Path

source = Path("arrhenius_fracture/sharp_front.py").read_text()
from arrhenius_fracture.plastic_flow_stagger_consistent_v1043 import transform_source

transformed = transform_source(source)
required = {
    "step-state snapshot": "ep_gp_step0_v1043 = ep_gp.copy()",
    "density snapshot": "rho_gp_step0_v1043 = rho_gp.copy()",
    "re-based update": "ep_gp_step0_v1043, rho_gp_step0_v1043",
    "final equilibrium": "Close the staggered step with a",
    "converged work ledger": (
        "constitutive_dWp_accepted_gp_converged_stagger_rebased_state"
    ),
    "positive directional J": "J_positive = max(J_signed, 0.0)",
}
missing = [label for label, token in required.items() if token not in transformed]
if missing:
    raise SystemExit(f"ERROR: transformed source missing invariants: {missing}")
print("v10.4.3 transformed-source invariants verified")
PY

printf '%s\n' 'v10.4.3 targeted validation passed'
