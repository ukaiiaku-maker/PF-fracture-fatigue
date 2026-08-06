#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$ROOT"

CONDA_ENV=${CONDA_ENV:-arrhenius-sharp-front-v10}

if [[ "${USE_CURRENT_PYTHON:-0}" == 1 ]]; then
  PY=(python)
elif [[ "${CONDA_DEFAULT_ENV:-}" == "$CONDA_ENV" ]]; then
  PY=(python)
else
  command -v conda >/dev/null 2>&1 || {
    echo "ERROR: conda is unavailable and '$CONDA_ENV' is not active" >&2
    exit 2
  }
  PY=(conda run --no-capture-output -n "$CONDA_ENV" python)
fi

printf 'Repository: %s\n' "$ROOT"
printf 'Branch:     %s\n' "$(git branch --show-current)"
printf 'HEAD:       %s\n' "$(git rev-parse HEAD)"
printf 'Python:     '
"${PY[@]}" -c 'import sys; print(sys.executable)'

EXPECTED_BRANCH=${EXPECTED_BRANCH:-v10.4.3-plastic-dominance-censor}
if [[ "${SKIP_BRANCH_CHECK:-0}" != 1 ]] \
  && [[ "$(git branch --show-current)" != "$EXPECTED_BRANCH" ]]; then
  echo "ERROR: expected branch '$EXPECTED_BRANCH'" >&2
  exit 2
fi

if [[ -n "$(git status --porcelain)" && "${ALLOW_DIRTY:-0}" != 1 ]]; then
  echo "ERROR: working tree is dirty; set ALLOW_DIRTY=1 only for deliberate development" >&2
  git status --short >&2
  exit 2
fi

git diff --check
"${PY[@]}" -m compileall -q arrhenius_fracture scripts tests

TMP=$(mktemp "${TMPDIR:-/tmp}/v1043-launcher.XXXXXX.sh")
trap 'rm -f "$TMP"' EXIT
"${PY[@]}" scripts/build_v10_4_3_plastic_dominance_launcher.py \
  --source scripts/run_v10_2_28_paper_four_class_theta30_1000um.sh \
  --output "$TMP"
bash -n "$TMP"
bash -n scripts/launch_v10_4_3_plastic_dominance_campaign.sh

"${PY[@]}" -m pytest -q \
  tests/test_v10_4_2_directional_j_positive.py \
  tests/test_v10_4_2_reuse_aware_launcher.py \
  tests/test_v10_4_2_plastic_flow_terminal.py \
  tests/test_v10_4_2_launcher_adapter.py \
  tests/test_v10_4_3_plastic_dominance.py \
  tests/test_v10_4_3_fresh_campaign_launcher.py

"${PY[@]}" - <<'PY'
from pathlib import Path
import arrhenius_fracture
from arrhenius_fracture.plastic_dominance_runtime_v1043 import transform_source

root = Path.cwd()
assert arrhenius_fracture.PROJECT_ID == "PF-fracture-fatigue"
assert arrhenius_fracture.PROJECT_RELEASE == "10.4.3"
source = (root / "arrhenius_fracture" / "sharp_front.py").read_text()
transformed = transform_source(source)
compile(transformed, "sharp_front.py[v10.4.3-validation]", "exec")
required = [
    "ep_gp_step0_v1043.copy()",
    "rho_gp_step0_v1043.copy()",
    "constitutive_dWp_accepted_gp_final_stagger_iterate",
    "plastic_flow_candidate_latest.json",
    "v10.4.3_plastic_dominance_terminal_audit_v1",
    "future_fracture_beyond_terminal_resolved",
]
missing = [token for token in required if token not in transformed]
if missing:
    raise SystemExit(f"missing v10.4.3 transformed-source tokens: {missing}")
print("v10.4.3 transformed production source validated")
PY

echo "VALIDATION PASSED: v10.4.3 is ready for a fresh pilot or fresh 48-case campaign"
