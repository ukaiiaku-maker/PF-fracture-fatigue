#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd); cd "$ROOT"
CAMPAIGN=${CAMPAIGN:-runs/v10_2_31_endurance_knee_ABCD_sparse2D_v1}
PYTHON_BIN=${PYTHON_BIN:-python}; MAX_PARALLEL=${MAX_PARALLEL:-2}

# Dimensional DeltaK points selected only after the primary sparse matrix showed
# shape disagreement.  The spacing is coarse on each upper branch and tightens
# around the observed transition/arrest region.  Low-side points use the
# qualified physical cycle censor and a short extension target.
cases=(
"A_0462 adaptive_7p70 v914_endurance_knee_0462 7.70 10"
"A_0462 adaptive_9p00 v914_endurance_knee_0462 9.00 100"
"A_0462 adaptive_10p50 v914_endurance_knee_0462 10.50 100"
"B_0658 adaptive_20p00 v914_endurance_knee_0658 20.00 10"
"B_0658 adaptive_21p30 v914_endurance_knee_0658 21.30 100"
"B_0658 adaptive_21p80 v914_endurance_knee_0658 21.80 100"
"B_0658 adaptive_24p50 v914_endurance_knee_0658 24.50 100"
"B_0658 adaptive_26p00 v914_endurance_knee_0658 26.00 100"
"C_0554 adaptive_12p50 v914_endurance_knee_0554 12.50 10"
"C_0554 adaptive_13p00 v914_endurance_knee_0554 13.00 100"
"C_0554 adaptive_14p00 v914_endurance_knee_0554 14.00 100"
"C_0554 adaptive_15p00 v914_endurance_knee_0554 15.00 100"
"D_0133 adaptive_32p00 v914_endurance_knee_0133 32.00 10"
"D_0133 adaptive_33p90 v914_endurance_knee_0133 33.90 100"
"D_0133 adaptive_36p00 v914_endurance_knee_0133 36.00 100"
"D_0133 adaptive_40p00 v914_endurance_knee_0133 40.00 100"
)

mkdir -p "$CAMPAIGN"
for spec in "${cases[@]}"; do
  read -r cls label option dk target <<<"$spec"; out="$CAMPAIGN/$cls/$label"
  [[ -e "$out/exit_code.txt" ]] && continue
  while (( $(jobs -rp | wc -l) >= MAX_PARALLEL )); do
    oldest=$(jobs -rp | head -n 1); wait "$oldest" || true
  done
  mkdir -p "$CAMPAIGN/$cls"
  (set +e; env PYTHON_BIN="$PYTHON_BIN" PARAMETER_OPTION="$option" DELTA_K_MPA_SQRT_M="$dk" OUTROOT="$out" CYCLES_MAX=1e14 TARGET_EXT_UM="$target" STEPS=20000 bash scripts/run_v10_2_31_sparse_case.sh; rc=$?; printf '%s\n' "$rc" > "$out/exit_code.txt"; exit "$rc") &
done
rc=0; for pid in $(jobs -rp); do wait "$pid" || rc=1; done; exit "$rc"
