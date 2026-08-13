#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd); cd "$ROOT"
CAMPAIGN=${CAMPAIGN:-runs/v10_2_31_endurance_knee_ABCD_sparse2D_v1}
PYTHON_BIN=${PYTHON_BIN:-python}; MAX_PARALLEL=${MAX_PARALLEL:-2}
cases=(
"A_0462 low v914_endurance_knee_0462 8.12138534732442 10"
"A_0462 knee v914_endurance_knee_0462 8.319467916771359 100"
"A_0462 high v914_endurance_knee_0462 11.884954166816225 100"
"B_0658 low v914_endurance_knee_0658 20.83424771642134 10"
"B_0658 knee v914_endurance_knee_0658 22.686180846769904 100"
"B_0658 high v914_endurance_knee_0658 27.778996955228454 100"
"C_0554 low v914_endurance_knee_0554 13.364427958809149 10"
"C_0554 knee v914_endurance_knee_0554 13.49807223839724 100"
"C_0554 high v914_endurance_knee_0554 16.03731355057098 100"
"D_0133 low v914_endurance_knee_0133 33.29694002901198 10"
"D_0133 knee v914_endurance_knee_0133 34.53016003008649 100"
"D_0133 high v914_endurance_knee_0133 44.39592003868263 100"
)
mkdir -p "$CAMPAIGN"
for spec in "${cases[@]}"; do
  read -r cls regime option dk target <<<"$spec"; out="$CAMPAIGN/$cls/$regime"
  [[ -e "$out/exit_code.txt" ]] && continue
  while (( $(jobs -rp | wc -l) >= MAX_PARALLEL )); do
    oldest=$(jobs -rp | head -n 1); wait "$oldest" || true
  done
  mkdir -p "$CAMPAIGN/$cls"
  (set +e; env PYTHON_BIN="$PYTHON_BIN" PARAMETER_OPTION="$option" DELTA_K_MPA_SQRT_M="$dk" OUTROOT="$out" CYCLES_MAX=1e14 TARGET_EXT_UM="$target" STEPS=20000 bash scripts/run_v10_2_31_sparse_case.sh; rc=$?; printf '%s\n' "$rc" > "$out/exit_code.txt"; exit "$rc") &
done
rc=0; for pid in $(jobs -rp); do wait "$pid" || rc=1; done; exit "$rc"
