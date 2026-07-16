#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)

TEMPS=${TEMPS:-700} \
CLASSES=${CLASSES:-"ceramic weakT DBTT"} \
TARGET_EXT_UM=${TARGET_EXT_UM:-50} \
STEPS=${STEPS:-50000} \
NX=${NX:-48} \
NY=${NY:-96} \
TIP_H_FINE=${TIP_H_FINE:-5e-7} \
DA_PHYS_M=${DA_PHYS_M:-5e-6} \
WAKE_SHIELDING=${WAKE_SHIELDING:-1} \
SAVE_SNAPSHOTS=${SAVE_SNAPSHOTS:-10} \
SNAPSHOT_BY_EXT_UM=${SNAPSHOT_BY_EXT_UM:-5} \
OUTROOT_BASE=${OUTROOT_BASE:-runs/v10_0_2_three_class_700K_50um_wake_on_v1} \
bash "$SCRIPT_DIR/run_v10_0_2_three_class_progression.sh"
