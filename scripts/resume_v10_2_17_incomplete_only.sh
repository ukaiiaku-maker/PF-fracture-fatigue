#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$ROOT"

EXPECTED_ENV=${EXPECTED_ENV:-arrhenius-sharp-front-v10}
if [[ "${CONDA_DEFAULT_ENV:-}" != "$EXPECTED_ENV" ]]; then
  echo "ERROR: activate conda environment $EXPECTED_ENV" >&2
  exit 2
fi

OUTROOT=${OUTROOT:?Set OUTROOT to the existing v10.2.17 campaign directory}
TARGET_EXT_UM=${TARGET_EXT_UM:-100}
FAMILY_JSON=${FAMILY_JSON:-$ROOT/runtime_inputs/v10_2_17/v10_2_14_active_only_campaign_family.json}
BASE_HAZARD_SEED=${BASE_HAZARD_SEED:-1720}
STEPS=${STEPS:-300000}
THETA=${THETA:-45}
NO_PLOTS=${NO_PLOTS:-0}
KEEP_INCOMPLETE_BACKUPS=${KEEP_INCOMPLETE_BACKUPS:-1}
SCAN_ONLY=${SCAN_ONLY:-0}

PLAN="$OUTROOT/stage3_campaign_plan.tsv"
RESUME_PLAN="$OUTROOT/resume_incomplete_plan.tsv"
FULL_PLAN_BACKUP="$OUTROOT/stage3_campaign_plan.full_resume_backup.tsv"

if [[ ! -f "$FAMILY_JSON" ]]; then
  echo "ERROR: frozen family not found: $FAMILY_JSON" >&2
  exit 2
fi
if [[ ! -f "$PLAN" ]]; then
  echo "ERROR: campaign plan not found: $PLAN" >&2
  exit 2
fi

printf 'option_key\tcandidate_id\ttemperature_K\tmpz_length_um\tmpz_n_bins\thazard_seed\tcase_root\treason\n' > "$RESUME_PLAN"

complete_count=0
resume_count=0

exec 3< "$PLAN"
IFS=$'\t' read -r _header <&3
while IFS=$'\t' read -r option candidate temperature mpz_length mpz_bins hazard_seed case_root <&3; do
  [[ -n "$option" ]] || continue
  reason="missing_or_unclassifiable_output"

  if [[ -f "$case_root/summary.json" ]]; then
    if python scripts/classify_v10_2_15_stage3_case.py \
      --case-root "$case_root" \
      --target-extension-um "$TARGET_EXT_UM" \
      > "$case_root/resume_classification.json" 2> "$case_root/resume_classification.err"; then
      complete=$(CASE_ROOT="$case_root" python - <<'PY'
import json
import os
from pathlib import Path
payload = json.loads((Path(os.environ["CASE_ROOT"]) / "stage3_case_status.json").read_text())
print("1" if payload.get("complete") is True else "0")
PY
)
      if [[ "$complete" == 1 ]]; then
        rm -f "$case_root/RUN_FAILED" "$case_root/exit_code.txt"
        echo "SKIP verified complete: $option T=${temperature}K seed=$hazard_seed"
        complete_count=$((complete_count + 1))
        continue
      fi
      reason=$(CASE_ROOT="$case_root" python - <<'PY'
import json
import os
from pathlib import Path
payload = json.loads((Path(os.environ["CASE_ROOT"]) / "stage3_case_status.json").read_text())
print(payload.get("status", "classified_incomplete"))
PY
)
    else
      reason="classification_failed"
    fi
  fi

  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "$option" "$candidate" "$temperature" "$mpz_length" "$mpz_bins" \
    "$hazard_seed" "$case_root" "$reason" >> "$RESUME_PLAN"
  resume_count=$((resume_count + 1))
done
exec 3<&-

echo "Verified complete cases: $complete_count"
echo "Cases requiring rerun:   $resume_count"
echo "Resume plan:             $RESUME_PLAN"

if [[ "$SCAN_ONLY" == 1 ]]; then
  echo "SCAN_ONLY=1: no simulations were started."
  exit 0
fi

if [[ "$resume_count" -eq 0 ]]; then
  python scripts/summarize_v10_2_15_stage3.py --outroot "$OUTROOT"
  echo "No simulations require rerun."
  exit 0
fi

cp -p "$PLAN" "$FULL_PLAN_BACKUP"
restore_full_plan() {
  if [[ -f "$FULL_PLAN_BACKUP" ]]; then
    cp -p "$FULL_PLAN_BACKUP" "$PLAN"
  fi
}
trap restore_full_plan EXIT INT TERM

exec 4< "$RESUME_PLAN"
IFS=$'\t' read -r _header <&4
while IFS=$'\t' read -r option candidate temperature mpz_length mpz_bins hazard_seed case_root reason <&4; do
  [[ -n "$option" ]] || continue

  if [[ -d "$case_root" ]]; then
    stamp=$(date +%Y%m%d_%H%M%S)
    backup="${case_root}.incomplete_backup_${stamp}"
    if [[ "$KEEP_INCOMPLETE_BACKUPS" == 1 ]]; then
      mv "$case_root" "$backup"
      echo "Preserved incomplete output: $backup"
    else
      rm -rf "$case_root"
      echo "Removed incomplete output: $case_root"
    fi
  fi

  case "$option" in
    ceramic_primary) option_offset=0 ;;
    weakT_primary) option_offset=10000 ;;
    dbtt_primary) option_offset=20000 ;;
    peak_primary) option_offset=30000 ;;
    *) echo "ERROR: unknown option $option" >&2; exit 2 ;;
  esac
  case_base_seed=$((BASE_HAZARD_SEED + option_offset))

  echo "RERUN: $option candidate=$candidate T=${temperature}K original_seed=$hazard_seed reason=$reason"
  env \
    MODE=full \
    ALLOW_PARTIAL=1 \
    OPTIONS="$option" \
    TEMPS="$temperature" \
    OUTROOT="$OUTROOT" \
    SIGNED_KERNEL_FAMILY_JSON="$FAMILY_JSON" \
    MAX_JOBS=1 \
    STEPS="$STEPS" \
    TARGET_EXT_UM="$TARGET_EXT_UM" \
    THETA="$THETA" \
    BASE_HAZARD_SEED="$case_base_seed" \
    SKIP_FINISHED=0 \
    NO_PLOTS="$NO_PLOTS" \
    bash scripts/run_v10_2_17_stage3_monotonic_temperature_sweep.sh

done
exec 4<&-

restore_full_plan
trap - EXIT INT TERM
python scripts/summarize_v10_2_15_stage3.py --outroot "$OUTROOT"

echo "Incomplete-only resume finished."
