#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd); cd "$ROOT"
CAMPAIGN=${CAMPAIGN:-runs/v10_2_31_endurance_knee_ABCD_sparse2D_v1}
PYTHON_BIN=${PYTHON_BIN:-python}; MAX_PARALLEL=${MAX_PARALLEL:-2}
cases=(
"B_0658 transition_f1p00 v914_endurance_knee_0658 23.14916412935705 100"
"B_0658 transition_f1p02 v914_endurance_knee_0658 23.61214741194419 100"
"B_0658 transition_f1p04 v914_endurance_knee_0658 24.07513069453133 100"
)
for spec in "${cases[@]}"; do
  read -r cls label option dk target <<<"$spec"; out="$CAMPAIGN/$cls/$label"
  [[ -e "$out/exit_code.txt" ]] && continue
  while (( $(jobs -rp | wc -l) >= MAX_PARALLEL )); do oldest=$(jobs -rp | head -n 1); wait "$oldest" || true; done
  (set +e; env PYTHON_BIN="$PYTHON_BIN" PARAMETER_OPTION="$option" DELTA_K_MPA_SQRT_M="$dk" OUTROOT="$out" CYCLES_MAX=1e14 TARGET_EXT_UM="$target" STEPS=20000 bash scripts/run_v10_2_31_sparse_case.sh; rc=$?; printf '%s\n' "$rc" > "$out/exit_code.txt"; exit "$rc") &
done
rc=0; for pid in $(jobs -rp); do wait "$pid" || rc=1; done; exit "$rc"
