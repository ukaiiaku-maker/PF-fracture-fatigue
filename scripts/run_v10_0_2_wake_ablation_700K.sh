#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
TARGET_EXT_UM=${TARGET_EXT_UM:-50}
STEPS=${STEPS:-50000}

for WAKE in 1 0; do
  OUTROOT_BASE=${OUTROOT_BASE_PREFIX:-runs/v10_0_2_wake_ablation_700K_${TARGET_EXT_UM}um_v1}_wake_${WAKE} \
  TEMPS=700 \
  CLASSES=${CLASSES:-"ceramic weakT DBTT"} \
  TARGET_EXT_UM="$TARGET_EXT_UM" \
  STEPS="$STEPS" \
  NX=${NX:-48} \
  NY=${NY:-96} \
  TIP_H_FINE=${TIP_H_FINE:-5e-7} \
  DA_PHYS_M=5e-6 \
  WAKE_SHIELDING="$WAKE" \
  SAVE_SNAPSHOTS=${SAVE_SNAPSHOTS:-10} \
  SNAPSHOT_BY_EXT_UM=${SNAPSHOT_BY_EXT_UM:-5} \
  bash "$SCRIPT_DIR/run_v10_0_2_three_class_progression.sh"
done
