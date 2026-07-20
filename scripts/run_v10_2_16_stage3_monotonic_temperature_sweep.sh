#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
SOURCE="$ROOT_DIR/scripts/run_v10_2_15_stage3_monotonic_temperature_sweep.sh"
GENERATED="$ROOT_DIR/scripts/.run_v10_2_16_generated_$$.sh"

export OUTROOT=${OUTROOT:-runs/v10_2_16_stage3_continuum_four_option_100um_theta45_v1}
export TARGET_EXT_UM=${TARGET_EXT_UM:-100}
export PRINT_EVERY=${PRINT_EVERY:-25}
export PYTHONUNBUFFERED=1

python - "$SOURCE" "$GENERATED" <<'PY'
from pathlib import Path
import sys

source = Path(sys.argv[1])
target = Path(sys.argv[2])
text = source.read_text()
replacements = {
    "arrhenius_fracture.sharp_front_v10_2_15":
        "arrhenius_fracture.sharp_front_v10_2_16",
    "Stage 3 existing-2D parameter-overlay plan":
        "v10.2.16 Stage 3 continuum-source parameter-overlay plan",
    "physics:       unchanged; exact material-manifest overlay only":
        "physics:       accepted 2-D continuum source; exact manifest overlay only",
    "entry=v10.1.7.5": "entry=v10.2.16",
    "Stage 3 runner finished:": "v10.2.16 Stage 3 runner finished:",
}
for old, new in replacements.items():
    if old not in text:
        raise SystemExit(f"upstream launcher changed; missing expected text: {old!r}")
    text = text.replace(old, new)

old_run = '''  "${cmd[@]}" > "$log" 2>&1
  rc=$?
'''
new_run = '''  "${cmd[@]}" 2>&1 | "$PYTHON_BIN" -u -c '\''
import sys
prefix = sys.argv[1]
for line in sys.stdin:
    print(f"[{prefix}] {line}", end="", flush=True)
'\'' "$option/T${temperature}K" | tee "$log"
  rc=${PIPESTATUS[0]}
'''
if old_run not in text:
    raise SystemExit("upstream launcher changed; solver redirection block not found")
text = text.replace(old_run, new_run, 1)
target.write_text(text)
target.chmod(0o755)
PY

set +e
bash "$GENERATED"
rc=$?
set -e
rm -f "$GENERATED"
exit "$rc"
