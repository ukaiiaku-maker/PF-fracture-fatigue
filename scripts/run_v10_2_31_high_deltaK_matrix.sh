#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd); cd "$ROOT"
CAMPAIGN=${CAMPAIGN:-runs/v10_2_31_endurance_knee_ABCD_high_deltaK_v1}
PYTHON_BIN=${PYTHON_BIN:-/opt/homebrew/Caskroom/miniconda/base/envs/arrhenius-sharp-front-v10-codex/bin/python}
MAX_PARALLEL=${MAX_PARALLEL:-2}
EXPECTED_BRANCH=${EXPECTED_BRANCH:-codex/v10.2.31-endurance-knee-ABCD-validation}
EXPECTED_HEAD=${EXPECTED_HEAD:?set EXPECTED_HEAD to the committed launch HEAD}
[[ $(git branch --show-current) == "$EXPECTED_BRANCH" ]] || { echo "wrong branch" >&2; exit 2; }
[[ $(git rev-parse HEAD) == "$EXPECTED_HEAD" ]] || { echo "wrong HEAD" >&2; exit 2; }
[[ -z $(git status --porcelain) ]] || { echo "worktree is not clean" >&2; exit 2; }

# H1/H2 choices follow the immutable v9.14 1-D high-side scouts.  A/C/D
# plateau, so H2 is a deliberately aggressive plateau-persistence probe;
# B uses modest increments because its existing 2-D rate is already few-cycle.
cases=(
  "A_0462 H1_intermediate_high v914_endurance_knee_0462 29.712385417040565 3.0"
  "A_0462 H2_few_cycle v914_endurance_knee_0462 99.04128472346855 10.0"
  "B_0658 H1_intermediate_high v914_endurance_knee_0658 30.093913368164165 1.3"
  "B_0658 H2_few_cycle v914_endurance_knee_0658 46.29832825871410 2.0"
  "C_0554 H1_intermediate_high v914_endurance_knee_0554 40.09328387642745 3.0"
  "C_0554 H2_few_cycle v914_endurance_knee_0554 133.6422795880915 10.0"
  "D_0133 H1_intermediate_high v914_endurance_knee_0133 98.65760008596142 4.0"
  "D_0133 H2_few_cycle v914_endurance_knee_0133 246.64400021490355 10.0"
)

mkdir -p "$CAMPAIGN"
printf '%s\n' "$EXPECTED_HEAD" > "$CAMPAIGN/launch_head.txt"
for spec in "${cases[@]}"; do
  read -r cls label option dk fraction <<<"$spec"
  out="$CAMPAIGN/$cls/$label"
  [[ ! -e "$out" ]] || { echo "refusing to overwrite $out" >&2; exit 2; }
  while (( $(jobs -rp | wc -l) >= MAX_PARALLEL )); do
    oldest=$(jobs -rp | head -n 1); wait "$oldest" || true
  done
  mkdir -p "$CAMPAIGN/$cls"
  (
    set +e
    env PYTHON_BIN="$PYTHON_BIN" PARAMETER_OPTION="$option" \
      DELTA_K_MPA_SQRT_M="$dk" OUTROOT="$out" CYCLES_MAX=1e6 \
      TARGET_EXT_UM=100 STEPS=20000 HAZARD_SEED=1720 \
      bash scripts/run_v10_2_31_sparse_case.sh
    rc=$?
    printf '%s\n' "$rc" > "$out/exit_code.txt"
    printf '%s\n' "$fraction" > "$out/normalized_fraction.txt"
    exit "$rc"
  ) &
done
rc=0
for pid in $(jobs -rp); do wait "$pid" || rc=1; done
exit "$rc"
