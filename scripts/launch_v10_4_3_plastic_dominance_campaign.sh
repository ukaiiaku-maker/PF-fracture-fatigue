#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$ROOT"

MODE=${1:-pilot}
BRANCH=${BRANCH:-v10.4.3-plastic-dominance-censor}
CONDA_ENV=${CONDA_ENV:-arrhenius-sharp-front-v10}
OUTROOT=${OUTROOT:-/Volumes/Data/Data/Nanopillar_calculation/PF-fracture-fatigue_v10_2_21_persistent_sites_top1/runs/v10_4_3_theta0_rate1x_bulk_PT_positiveJ_plastic_dominance_fresh48_base3621_v1}
MAX_JOBS=${MAX_JOBS:-2}

case "$MODE" in
  pilot|full) ;;
  *)
    echo "Usage: $0 {pilot|full}" >&2
    echo "  pilot: run one canonical v10.4.3 case in the fresh campaign root" >&2
    echo "  full:  run/resume all 48 canonical v10.4.3 cases" >&2
    exit 2
    ;;
esac

CURRENT_BRANCH=$(git branch --show-current)
HEAD=$(git rev-parse HEAD)
if [[ "$CURRENT_BRANCH" != "$BRANCH" ]]; then
  echo "ERROR: expected branch '$BRANCH'; found '$CURRENT_BRANCH'" >&2
  exit 2
fi
if [[ -n "$(git status --porcelain)" ]]; then
  echo "ERROR: working tree is dirty" >&2
  git status --short >&2
  exit 2
fi

if [[ -e "$OUTROOT" && ! -d "$OUTROOT" ]]; then
  echo "ERROR: OUTROOT exists but is not a directory: $OUTROOT" >&2
  exit 2
fi
mkdir -p "$OUTROOT"

# This campaign intentionally recomputes all 48 cases with v10.4.3.  Do not
# allow audited v10.4.1/v10.4.2 materializations or symlinked terminal markers
# to enter the root accidentally.
if find "$OUTROOT" -type f -name v10_4_2_reuse_audit.json -print -quit | grep -q .; then
  echo "ERROR: inherited v10.4.2 reuse audit found under fresh campaign root" >&2
  echo "Use a new OUTROOT; inherited-case reuse is forbidden for this campaign." >&2
  exit 2
fi
if find "$OUTROOT" -type l -print -quit | grep -q .; then
  echo "ERROR: symbolic link found under fresh campaign root" >&2
  echo "Fresh v10.4.3 outcomes must be generated locally, not materialized." >&2
  exit 2
fi

INTENT_FILE="$OUTROOT/v10_4_3_fresh48_campaign_intent.json"
OUTROOT="$OUTROOT" INTENT_FILE="$INTENT_FILE" CURRENT_BRANCH="$CURRENT_BRANCH" \
HEAD="$HEAD" python - <<'PY'
import json
import os
from pathlib import Path

root = Path(os.environ["OUTROOT"]).resolve()
intent_path = Path(os.environ["INTENT_FILE"])
expected = {
    "schema": "v10.4.3_fresh48_campaign_intent_v1",
    "branch": os.environ["CURRENT_BRANCH"],
    "commit": os.environ["HEAD"],
    "planned_case_count": 48,
    "inherited_reuse_permitted": False,
    "all_cases_recomputed_with_v10_4_3": True,
}

if intent_path.is_file():
    actual = json.loads(intent_path.read_text())
    for key, value in expected.items():
        if actual.get(key) != value:
            raise SystemExit(
                f"ERROR: campaign intent mismatch for {key}: "
                f"expected {value!r}, found {actual.get(key)!r}"
            )
else:
    allowed = {".DS_Store"}
    unexpected = [
        path.name for path in root.iterdir()
        if path.name not in allowed
    ]
    if unexpected:
        raise SystemExit(
            "ERROR: fresh campaign root is nonempty but has no intent record: "
            + ", ".join(sorted(unexpected)[:10])
        )
    tmp = intent_path.with_suffix(intent_path.suffix + ".tmp")
    tmp.write_text(json.dumps(expected, indent=2, sort_keys=True) + "\n")
    tmp.replace(intent_path)
PY

PID_FILE=${PID_FILE:-$OUTROOT/v10_4_3_campaign.pid}
LOG_DIR=${LOG_DIR:-$OUTROOT/v10_4_3_logs}
mkdir -p "$LOG_DIR"

if [[ -f "$PID_FILE" ]]; then
  old_pid=$(tr -dc '0-9' < "$PID_FILE")
  if [[ -n "$old_pid" ]] && kill -0 "$old_pid" 2>/dev/null; then
    echo "ERROR: campaign launcher is already alive: pid=$old_pid" >&2
    exit 3
  fi
  if [[ "${CLEAR_STALE_PID:-0}" != 1 ]]; then
    echo "ERROR: stale PID file exists: $PID_FILE" >&2
    echo "Set CLEAR_STALE_PID=1 only after checking the process tree." >&2
    exit 3
  fi
  rm -f "$PID_FILE"
fi

echo $$ > "$PID_FILE"
cleanup_pid() {
  if [[ -f "$PID_FILE" ]] && [[ "$(cat "$PID_FILE" 2>/dev/null)" == "$$" ]]; then
    rm -f "$PID_FILE"
  fi
}
trap cleanup_pid EXIT INT TERM

if [[ "${SKIP_VALIDATION:-0}" != 1 ]]; then
  CONDA_ENV="$CONDA_ENV" EXPECTED_BRANCH="$BRANCH" \
    bash scripts/validate_v10_4_3_plastic_dominance.sh
fi

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG="$LOG_DIR/${MODE}_${TIMESTAMP}.log"
FILTER_OPTION=""
FILTER_TEMPERATURE=""

case "$MODE" in
  pilot)
    [[ "${APPROVE_PILOT:-}" == YES ]] || {
      echo "ERROR: pilot requires APPROVE_PILOT=YES" >&2
      exit 2
    }
    : "${PILOT_OPTION:?Set PILOT_OPTION to one canonical material option}"
    : "${PILOT_TEMPERATURE:?Set PILOT_TEMPERATURE to one canonical temperature}"
    FILTER_OPTION=$PILOT_OPTION
    FILTER_TEMPERATURE=$PILOT_TEMPERATURE
    MAX_JOBS=1
    ;;
  full)
    [[ "${APPROVE_FULL_CAMPAIGN:-}" == YES ]] || {
      echo "ERROR: full launch requires APPROVE_FULL_CAMPAIGN=YES" >&2
      exit 2
    }
    FILTER_OPTION=""
    FILTER_TEMPERATURE=""
    ;;
esac

cat <<EOF
v10.4.3 fresh campaign launch
  mode:        $MODE
  branch:      $CURRENT_BRANCH
  commit:      $HEAD
  output root: $OUTROOT
  log:         $LOG
  max jobs:    $MAX_JOBS
  reuse:       forbidden; all 48 cases use v10.4.3
  option:      ${FILTER_OPTION:-all four canonical options}
  temperature: ${FILTER_TEMPERATURE:-all twelve canonical temperatures}
EOF

set +e
CONDA_ENV="$CONDA_ENV" \
OUTROOT="$OUTROOT" \
MAX_JOBS="$MAX_JOBS" \
CASE_FILTER_OPTION="$FILTER_OPTION" \
CASE_FILTER_TEMPERATURE="$FILTER_TEMPERATURE" \
SKIP_FINISHED=1 \
RESTART_INCOMPLETE=1 \
bash scripts/run_v10_4_3_paper_four_class_orientation_rate.sh \
  2>&1 | tee "$LOG"
rc=${PIPESTATUS[0]}
set -e

if [[ "$rc" -ne 0 ]]; then
  echo "ERROR: $MODE launch failed with exit=$rc; log=$LOG" >&2
  exit "$rc"
fi

if grep -Eq 'FAILED_REUSE_VERIFICATION|(^|[[:space:]])FAILED:|RUN_FAILED' "$LOG"; then
  echo "ERROR: failure marker found in successful shell return; log=$LOG" >&2
  exit 4
fi
if grep -Fq 'SKIP_REUSED_VERIFIED' "$LOG"; then
  echo "ERROR: inherited-case reuse occurred in a fresh48 campaign" >&2
  exit 4
fi

if [[ "$MODE" == pilot ]]; then
  grep -Fq 'START:' "$LOG" || {
    echo "ERROR: fresh pilot did not launch the v10.4.3 solver" >&2
    exit 4
  }
  grep -Fq 'Campaign acceptance: planned=1 complete=1 failed_or_incomplete=0' "$LOG" || {
    echo "ERROR: one-case pilot acceptance contract was not met" >&2
    exit 4
  }
  grep -Fq 'Campaign complete: failures=0' "$LOG" || {
    echo "ERROR: pilot scheduler reported failures" >&2
    exit 4
  }
  echo "FRESH PILOT PASSED: inspect terminal and candidate metrics with the monitor"
else
  grep -Fq 'Campaign acceptance: planned=48 complete=48 failed_or_incomplete=0' "$LOG" || {
    echo "ERROR: full matrix did not finish all 48 canonical outcomes" >&2
    exit 4
  }
  grep -Fq 'Campaign complete: failures=0' "$LOG" || {
    echo "ERROR: full scheduler reported failures" >&2
    exit 4
  }
  echo "FULL FRESH CAMPAIGN PASSED: 48/48 v10.4.3 cases reached accepted outcomes"
fi

echo "Log: $LOG"
