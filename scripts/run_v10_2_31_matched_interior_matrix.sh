#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd); cd "$ROOT"
CAMPAIGN=${CAMPAIGN:-runs/v10_2_31_endurance_knee_ABCD_sparse2D_v1}
PYTHON_BIN=${PYTHON_BIN:-python}; MAX_PARALLEL=${MAX_PARALLEL:-2}
cases=(
"B_0658 matched_f1p05 v914_endurance_knee_0658 24.30662233582490 100"
"B_0658 matched_f1p10 v914_endurance_knee_0658 25.46408054229275 100"
"C_0554 matched_f1p05 v914_endurance_knee_0554 14.03327916774710 100"
"D_0133 matched_f1p60 v914_endurance_knee_0133 39.46215114549567 100"
"D_0133 low_33p297_continued v914_endurance_knee_0133 33.29694002901198 100"
)
mkdir -p "$CAMPAIGN"
for spec in "${cases[@]}"; do
  read -r cls label option dk target <<<"$spec"; out="$CAMPAIGN/$cls/$label"
  [[ -e "$out/exit_code.txt" ]] && continue
  while (( $(jobs -rp | wc -l) >= MAX_PARALLEL )); do oldest=$(jobs -rp | head -n 1); wait "$oldest" || true; done
  mkdir -p "$CAMPAIGN/$cls"
  (set +e; env PYTHON_BIN="$PYTHON_BIN" PARAMETER_OPTION="$option" DELTA_K_MPA_SQRT_M="$dk" OUTROOT="$out" CYCLES_MAX=1e14 TARGET_EXT_UM="$target" STEPS=20000 bash scripts/run_v10_2_31_sparse_case.sh; rc=$?; printf '%s\n' "$rc" > "$out/exit_code.txt"; exit "$rc") &
done
rc=0; for pid in $(jobs -rp); do wait "$pid" || rc=1; done; exit "$rc"
