#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$ROOT"

EXPECTED_HEAD=${EXPECTED_HEAD:?}
MATRIX_CSV=${MATRIX_CSV:?}
OUT_BASE=${OUT_BASE:-runs/v914_HCF_LCF_transition_refinement_v2}
MAX_PARALLEL=${MAX_PARALLEL:-4}
MIN_FREE_GIB=${MIN_FREE_GIB:-5}
PYTHON_BIN=${PYTHON_BIN:-/opt/homebrew/Caskroom/miniconda/base/envs/arrhenius-sharp-front-v10-codex/bin/python}
V914=${V914:-/Volumes/Data/Data/Nanopillar_calculation/Arrhenius_FEM_CZM_MPZ_v9_14_cyclic_fatigue_knee_search}
REGISTRY=${REGISTRY:-$V914/runtime_inputs/v914/canonical_four_class_registry.csv}
PHYSICS=${PHYSICS:-$V914/mpz_v9_13_v10222_transfer_common_physics.json}

[[ $(git rev-parse HEAD) == "$EXPECTED_HEAD" ]] || { echo "HEAD mismatch" >&2; exit 2; }
[[ -z $(git status --porcelain) ]] || { echo "authoritative launch requires clean worktree" >&2; exit 2; }
[[ -f "$MATRIX_CSV" ]] || { echo "missing matrix: $MATRIX_CSV" >&2; exit 2; }

free_gib() { df -g /Volumes/Data | awk 'NR==2 {print $4}'; }
run_one() {
  local family=$1 candidate=$2 fraction=$3 delta_k=$4 seed=$5 max_cycles=$6 mode=${7:-explicit}
  local label=${fraction//./p}
  local short=${candidate##*_}
  local out="$OUT_BASE/${family}_${short}/f${label}_${mode}"
  if [[ -f "$out/result.json" && -f "$out/run_contract.json" ]]; then
    echo "already terminal: $out"
    return 0
  fi
  [[ ! -e "$out" ]] || { echo "existing nonterminal output requires explicit audit: $out" >&2; return 2; }
  env PYTHONPATH="$V914:$ROOT/scripts" "$PYTHON_BIN" scripts/run_v1032_explicit_lcf.py \
    --registry "$REGISTRY" --physics "$PHYSICS" --candidate "$candidate" \
    --deltaK "$delta_k" --mode "$mode" --phase-steps 32 --target-um 100 \
    --maximum-cycles "$max_cycles" --seed "$seed" --normalized-f "$fraction" \
    --expected-head "$EXPECTED_HEAD" --checkpoint-cycle-interval 10 \
    --state-history-cycle-interval 10 --out "$out"
}

pids=()
while IFS=, read -r family candidate fraction delta_k seed max_cycles mode; do
  [[ "$family" == family || -z "$family" ]] && continue
  while (( ${#pids[@]} >= MAX_PARALLEL )); do
    wait "${pids[0]}"
    pids=("${pids[@]:1}")
  done
  if (( $(free_gib) < MIN_FREE_GIB )); then
    echo "disk safety gate: less than ${MIN_FREE_GIB} GiB free" >&2
    exit 3
  fi
  run_one "$family" "$candidate" "$fraction" "$delta_k" "$seed" "$max_cycles" "${mode:-explicit}" &
  pids+=("$!")
done < "$MATRIX_CSV"
for pid in "${pids[@]}"; do wait "$pid"; done
