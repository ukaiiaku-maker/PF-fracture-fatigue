#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$ROOT"

EXPECTED_HEAD=${EXPECTED_HEAD:?}
OUT_BASE=${OUT_BASE:-runs/v10_2_32_endurance_knee_ABCD_hybrid_HCF_LCF_v1}
MAX_PARALLEL=${MAX_PARALLEL:-2}
MIN_FREE_GIB=${MIN_FREE_GIB:-3}
LOCAL_REGISTRY=arrhenius_fracture/data/materials/v10_2_31_endurance_knee_ABCD_registry.csv
CANONICAL_REGISTRY=arrhenius_fracture/data/materials/v10_2_27_paper_four_class_registry.csv

# class|parameter option|normalized f|dimensional DeltaK|cycles cap
CASES=(
  'A|v914_endurance_knee_0462|3|29.712385417040565|1200'
  'A|v914_endurance_knee_0462|10|99.04128472346855|1200'
  'A|v914_endurance_knee_0462|20|198.0825694469371|1200'
  'B|v914_endurance_knee_0658|0.96|22.22319756418277|1200'
  'B|v914_endurance_knee_0658|2|46.29832825871409|100'
  'B|v914_endurance_knee_0658|5|115.74582064678523|100'
  'C|v914_endurance_knee_0554|1.2|16.03731355057098|100'
  'C|v914_endurance_knee_0554|3|40.09328387642745|100'
  'C|v914_endurance_knee_0554|10|133.64427958809148|100'
  'D|v914_endurance_knee_0133|2|49.3288000429807|200'
  'D|v914_endurance_knee_0133|5|123.32200010745175|1200'
  'D|v914_endurance_knee_0133|10|246.6440002149035|1200'
  'DBTT|v913_paper_dbtt01_0202500_persistent_sites|1.1|23.12783841641128|100'
  'DBTT|v913_paper_dbtt01_0202500_persistent_sites|5|105.1265382564149|100'
  'Peak|v913_paper_peak01_0242980_persistent_sites|1.1|23.41850111155525|500'
  'Peak|v913_paper_peak01_0242980_persistent_sites|5|106.44773232525111|100'
)

free_gib() { df -g /Volumes/Data | awk 'NR==2 {print $4}'; }
run_one() {
  IFS='|' read -r cls option fraction delta_k cycles_max <<<"$1"
  local label out registry
  label=${fraction//./p}
  out="$OUT_BASE/${cls}_${option##*_}/f${label}_explicit"
  if [[ -e "$out/developed_fatigue_growth_summary.json" ]]; then
    echo "already complete: $out"
    return 0
  fi
  [[ ! -e "$out" ]] || { echo "incomplete existing output requires audit: $out" >&2; return 2; }
  registry=$LOCAL_REGISTRY
  [[ "$cls" == DBTT || "$cls" == Peak ]] && registry=$CANONICAL_REGISTRY
  env EXPECTED_HEAD="$EXPECTED_HEAD" CLASS_LABEL="$cls" NORMALIZED_F="$fraction" \
    PARAMETER_OPTION="$option" DELTA_K_MPA_SQRT_M="$delta_k" OUTROOT="$out" \
    REGISTRY="$registry" CYCLES_MAX="$cycles_max" TARGET_EXT_UM=100 N_PHASE=32 \
    HAZARD_SEED=1720 bash scripts/run_v10_2_32_explicit_sparse_case.sh
}
export -f run_one free_gib
export EXPECTED_HEAD OUT_BASE LOCAL_REGISTRY CANONICAL_REGISTRY MIN_FREE_GIB

pids=()
for spec in "${CASES[@]}"; do
  if (( ${#pids[@]} >= MAX_PARALLEL )); then
    wait "${pids[0]}"
    pids=("${pids[@]:1}")
  fi
  if (( $(free_gib) < MIN_FREE_GIB )); then
    echo "disk safety gate: less than ${MIN_FREE_GIB} GiB free" >&2
    exit 3
  fi
  run_one "$spec" &
  pids+=("$!")
done
for pid in "${pids[@]}"; do wait "$pid"; done
