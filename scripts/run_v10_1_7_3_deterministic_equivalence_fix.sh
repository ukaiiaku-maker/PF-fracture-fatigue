#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "$SCRIPT_DIR/.." && pwd)
cd "$REPO_ROOT"

PYTHON_BIN=${PYTHON_BIN:-python}
CLASS=${CLASS:-DBTT}
TEMP_K=${TEMP_K:-700}
TARGET_EXT_UM=${TARGET_EXT_UM:-200}

GIT_HEAD=$(git rev-parse HEAD)
GIT_SHORT=$(git rev-parse --short HEAD)
if [[ -n "$(git status --porcelain --untracked-files=no)" ]]; then
  GIT_DIRTY=1
else
  GIT_DIRTY=0
fi

# Build a stable signature from every pilot input that can alter the trajectory.
# The code SHA covers changed defaults; explicit environment overrides are listed
# so two configurations at the same SHA cannot silently share completed outputs.
CONFIG_SIGNATURE=$(
  PILOT_GIT_HEAD="$GIT_HEAD" PILOT_GIT_DIRTY="$GIT_DIRTY" \
  "$PYTHON_BIN" - <<'PY'
import hashlib
import json
import os

keys = [
    "CLASS", "TEMP_K", "SEEDS", "TARGET_EXT_UM", "STEPS", "PRINT_EVERY",
    "CAMPAIGN_BACKSTRESS_SCALE", "CAMPAIGN_REFRESH_SCALE",
    "K_FIRST_MAX_MPA_SQRT_M", "NX", "NY", "TIP_H_FINE", "TIP_RATIO",
    "DU", "DT", "N_STAGGER", "DA_CHECKPOINT_M", "MPZ_LENGTH_UM",
    "MPZ_N_BINS", "WAKE_LENGTH_UM", "WAKE_N_BINS", "THETA",
    "EVENT_TARGET", "SAVE_SNAPSHOTS", "PACKET_LENGTH_M",
    "MOBILE_SHIELD_FRACTION", "KINETIC_MAX_ACTION_SUBSTEP",
    "KINETIC_MAX_TRANSLATION_SUBSTEP_M", "EVENT_MIN_FACTOR",
    "EVENT_MAX_FACTOR", "EVENT_SUBSEGMENT_FRACTION",
]
payload = {
    "git_head": os.environ["PILOT_GIT_HEAD"],
    "git_dirty": int(os.environ["PILOT_GIT_DIRTY"]),
    "overrides": {key: os.environ.get(key, "<runner-default>") for key in keys},
}
print(hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:12])
PY
)

BASE_OUTROOT=${OUTROOT_BASE:-runs/v10_1_7_3_stochastic_avalanche_equivalence_fix_${CLASS}_${TEMP_K}K_${TARGET_EXT_UM}um}
OUTROOT="${BASE_OUTROOT}_${GIT_SHORT}_${CONFIG_SIGNATURE}"

# Fresh calculations are the default for this regression audit. Set FORCE=0 only
# to resume outputs carrying the same SHA/configuration-keyed root.
FORCE=${FORCE:-1}
export OUTROOT FORCE CLASS TEMP_K TARGET_EXT_UM

mkdir -p "$OUTROOT"
PROVENANCE="$OUTROOT/pilot_provenance.json"
PILOT_STATUS=RUNNING \
PILOT_GIT_HEAD="$GIT_HEAD" \
PILOT_GIT_SHORT="$GIT_SHORT" \
PILOT_GIT_DIRTY="$GIT_DIRTY" \
PILOT_CONFIG_SIGNATURE="$CONFIG_SIGNATURE" \
PILOT_OUTROOT="$OUTROOT" \
PILOT_FORCE="$FORCE" \
"$PYTHON_BIN" - "$PROVENANCE" <<'PY'
import json
import os
import pathlib
import sys
from datetime import datetime, timezone

path = pathlib.Path(sys.argv[1])
payload = {
    "schema": "v10.1.7.3_deterministic_equivalence_fix_provenance",
    "status": os.environ["PILOT_STATUS"],
    "git_head": os.environ["PILOT_GIT_HEAD"],
    "git_short": os.environ["PILOT_GIT_SHORT"],
    "git_dirty": bool(int(os.environ["PILOT_GIT_DIRTY"])),
    "configuration_signature": os.environ["PILOT_CONFIG_SIGNATURE"],
    "outroot": os.environ["PILOT_OUTROOT"],
    "force_fresh": bool(int(os.environ["PILOT_FORCE"])),
    "started_utc": datetime.now(timezone.utc).isoformat(),
}
path.write_text(json.dumps(payload, indent=2, sort_keys=True))
PY

printf 'EQUIVALENCE FIX PILOT\n'
printf '  git_head=%s dirty=%s\n' "$GIT_HEAD" "$GIT_DIRTY"
printf '  configuration_signature=%s\n' "$CONFIG_SIGNATURE"
printf '  outroot=%s\n' "$OUTROOT"
printf '  force=%s\n' "$FORCE"

set +e
bash scripts/run_v10_1_7_3_stochastic_avalanche_pilot.sh
rc=$?
set -e

PILOT_STATUS=$([[ "$rc" -eq 0 ]] && printf COMPLETE || printf FAILED) \
PILOT_RC="$rc" \
"$PYTHON_BIN" - "$PROVENANCE" <<'PY'
import json
import os
import pathlib
import sys
from datetime import datetime, timezone

path = pathlib.Path(sys.argv[1])
payload = json.loads(path.read_text())
payload["status"] = os.environ["PILOT_STATUS"]
payload["return_code"] = int(os.environ["PILOT_RC"])
payload["finished_utc"] = datetime.now(timezone.utc).isoformat()
path.write_text(json.dumps(payload, indent=2, sort_keys=True))
PY

exit "$rc"
