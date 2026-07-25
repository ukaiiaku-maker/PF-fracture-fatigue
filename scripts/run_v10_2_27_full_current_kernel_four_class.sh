#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$ROOT"

PYTHON_BIN=${PYTHON_BIN:-python}
BASE_RUNNER="$ROOT/scripts/run_v10_2_27_paper_four_class_30deg_long_rcurves.sh"

[[ -f "$BASE_RUNNER" ]] || {
  echo "ERROR: missing historical base runner: $BASE_RUNNER" >&2
  exit 2
}
: "${FAMILY_JSON:?Resolve and export FAMILY_JSON before launching the full campaign}"
[[ -f "$FAMILY_JSON" ]] || {
  echo "ERROR: resolved signed-kernel family is missing: $FAMILY_JSON" >&2
  exit 2
}

# Keep the historical base runner byte-for-byte stable. Materialize a temporary
# production runner by replacing only the superseded weak-T and ceramic option
# identifiers and candidate fingerprints. All mechanics, seed indexing, case
# contracts, solver arguments, and acceptance logic remain inherited unchanged.
PATCHED_RUNNER=$(mktemp "$ROOT/scripts/.v10_2_27_full_current_kernel.XXXXXX.sh")
trap 'rm -f "$PATCHED_RUNNER"' EXIT

BASE_RUNNER="$BASE_RUNNER" PATCHED_RUNNER="$PATCHED_RUNNER" "$PYTHON_BIN" - <<'PY'
import os
from pathlib import Path

source = Path(os.environ["BASE_RUNNER"])
target = Path(os.environ["PATCHED_RUNNER"])
text = source.read_text()
replacements = {
    "v913_paper_weakT01_0257068_persistent_sites": (
        "v913_paper_weakT01_0129902_persistent_sites"
    ),
    "v913_zeroD_sobol_0257068": "v913_zeroD_sobol_0129902",
    "v913_paper_ceramic01_0189364_persistent_sites": (
        "v913_paper_ceramic01_0077080_persistent_sites"
    ),
    "v913_zeroD_sobol_0189364": "v913_zeroD_sobol_0077080",
}
for old, new in replacements.items():
    count = text.count(old)
    if count == 0:
        raise SystemExit(
            f"historical base runner no longer contains expected token: {old}"
        )
    text = text.replace(old, new)
for obsolete in replacements:
    if obsolete in text:
        raise SystemExit(f"superseded token remains after materialization: {obsolete}")
target.write_text(text)
target.chmod(0o755)
PY

OPTIONS="v913_paper_peak01_0242980_persistent_sites v913_paper_dbtt01_0202500_persistent_sites v913_paper_weakT01_0129902_persistent_sites v913_paper_ceramic01_0077080_persistent_sites" \
FAMILY_JSON="$FAMILY_JSON" \
bash "$PATCHED_RUNNER" "$@"
