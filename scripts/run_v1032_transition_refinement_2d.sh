#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$ROOT"

EXPECTED_HEAD=${EXPECTED_HEAD:?}
MATRIX_CSV=${MATRIX_CSV:-runtime_inputs/v10_2_32/transition_refinement_2d_explicit.csv}
OUT_BASE=${OUT_BASE:-runs/v10_2_32_HCF_LCF_transition_refinement_v2}
MAX_PARALLEL=${MAX_PARALLEL:-2}
MIN_FREE_GIB=${MIN_FREE_GIB:-5}
REGISTRY=${REGISTRY:-arrhenius_fracture/data/materials/v10_2_27_paper_four_class_registry.csv}

[[ $(git rev-parse HEAD) == "$EXPECTED_HEAD" ]] || { echo "HEAD mismatch" >&2; exit 2; }
[[ -z $(git status --porcelain) ]] || { echo "authoritative launch requires clean worktree" >&2; exit 2; }
[[ -f "$MATRIX_CSV" ]] || { echo "missing matrix: $MATRIX_CSV" >&2; exit 2; }

free_gib() { df -g /Volumes/Data | awk 'NR==2 {print $4}'; }
run_one() {
  local family=$1 option=$2 candidate=$3 fraction=$4 delta_k=$5 seed=$6 max_cycles=$7
  local label=${fraction//./p}
  local short=${candidate##*_}
  local out="$OUT_BASE/${family}_${short}/f${label}_explicit"
  if [[ -f "$out/developed_fatigue_growth_summary.json" ]]; then
    echo "already terminal: $out"
    return 0
  fi
  [[ ! -e "$out" ]] || { echo "existing nonterminal output requires explicit audit: $out" >&2; return 2; }
  env EXPECTED_HEAD="$EXPECTED_HEAD" CLASS_LABEL="$family" NORMALIZED_F="$fraction" \
    PARAMETER_OPTION="$option" DELTA_K_MPA_SQRT_M="$delta_k" OUTROOT="$out" \
    REGISTRY="$REGISTRY" CYCLES_MAX="$max_cycles" TARGET_EXT_UM=100 N_PHASE=32 \
    HAZARD_SEED="$seed" bash scripts/run_v10_2_32_explicit_sparse_case.sh
}

pids=()
while IFS=, read -r family option candidate fraction delta_k seed max_cycles; do
  [[ "$family" == family || -z "$family" ]] && continue
  while (( ${#pids[@]} >= MAX_PARALLEL )); do
    wait "${pids[0]}"
    pids=("${pids[@]:1}")
  done
  if (( $(free_gib) < MIN_FREE_GIB )); then
    echo "disk safety gate: less than ${MIN_FREE_GIB} GiB free" >&2
    exit 3
  fi
  run_one "$family" "$option" "$candidate" "$fraction" "$delta_k" "$seed" "$max_cycles" &
  pids+=("$!")
done < "$MATRIX_CSV"
for pid in "${pids[@]}"; do wait "$pid"; done
