#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$ROOT"

MODE=${1:-smoke}
BRANCH=${BRANCH:-v10.4.3-plastic-dominance-censor}
CONDA_ENV=${CONDA_ENV:-arrhenius-sharp-front-v10}
OUTROOT=${OUTROOT:-/Volumes/Data/Data/Nanopillar_calculation/PF-fracture-fatigue_v10_2_21_persistent_sites_top1/runs/v10_4_2_theta0_rate1x_bulk_PT_positiveJ_plastic_terminal_four_class_1000um_reuse17_base3621_v1}
MAX_JOBS=${MAX_JOBS:-2}

case "$MODE" in
  smoke|pilot|full) ;;
  *)
    echo "Usage: $0 {smoke|pilot|full}" >&2
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
if [[ ! -d "$OUTROOT" ]]; then
  echo "ERROR: campaign root does not exist: $OUTROOT" >&2
  echo "The default must already contain the 17 audited materialized cases." >&2
  exit 2
fi

PID_FILE=${PID_FILE:-$OUTROOT/v10_4_3_campaign.pid}
LOG_DIR=${LOG_DIR:-$OUTROOT/v10_4_3_logs}
SMOKE_OK=${SMOKE_OK:-$OUTROOT/v10_4_3_reuse_smoke_ok.json}
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
  smoke)
    FILTER_OPTION=${SMOKE_OPTION:-v913_paper_peak01_0242980_persistent_sites}
    FILTER_TEMPERATURE=${SMOKE_TEMPERATURE:-300}
    MAX_JOBS=1
    ;;
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
    [[ -f "$SMOKE_OK" ]] || {
      echo "ERROR: missing successful reuse smoke record: $SMOKE_OK" >&2
      exit 2
    }
    grep -Fq "\"commit\": \"$HEAD\"" "$SMOKE_OK" || {
      echo "ERROR: reuse smoke was not validated at current HEAD $HEAD" >&2
      exit 2
    }
    FILTER_OPTION=""
    FILTER_TEMPERATURE=""
    ;;
esac

cat <<EOF
v10.4.3 campaign launch
  mode:        $MODE
  branch:      $CURRENT_BRANCH
  commit:      $HEAD
  output root: $OUTROOT
  log:         $LOG
  max jobs:    $MAX_JOBS
  option:      ${FILTER_OPTION:-all four canonical options}
  temperature: ${FILTER_TEMPERATURE:-all twelve canonical temperatures}
EOF

set +e
CONDA_ENV="$CONDA_ENV" \
OUTROOT="$OUTROOT" \
MAX_JOBS="$MAX_JOBS" \
CASE_FILTER_OPTION="$FILTER_OPTION" \
CASE_FILTER_TEMPERATURE="$FILTER_TEMPERATURE" \
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

if [[ "$MODE" == smoke ]]; then
  grep -Fq 'SKIP_REUSED_VERIFIED' "$LOG" || {
    echo "ERROR: smoke did not execute the audited reuse path" >&2
    exit 4
  }
  grep -Fq 'SKIP verified complete:' "$LOG" || {
    echo "ERROR: scheduler did not classify the inherited case as a clean skip" >&2
    exit 4
  }
  if grep -Fq 'START:' "$LOG"; then
    echo "ERROR: smoke launched a solver instead of reusing the case" >&2
    exit 4
  fi
  grep -Fq 'Campaign acceptance: planned=1 complete=1 failed_or_incomplete=0' "$LOG" || {
    echo "ERROR: one-case acceptance contract was not met" >&2
    exit 4
  }
  grep -Fq 'Campaign complete: failures=0' "$LOG" || {
    echo "ERROR: scheduler failure counter was not reconciled" >&2
    exit 4
  }
  cat > "$SMOKE_OK" <<EOF
{
  "schema": "v10.4.3_reuse_smoke_result_v1",
  "branch": "$CURRENT_BRANCH",
  "commit": "$HEAD",
  "option": "$FILTER_OPTION",
  "temperature_K": $FILTER_TEMPERATURE,
  "expected_seed": 3621,
  "solver_launched": false,
  "planned": 1,
  "complete": 1,
  "failed_or_incomplete": 0,
  "failures": 0,
  "log": "$LOG"
}
EOF
  echo "REUSE SMOKE PASSED: $SMOKE_OK"
elif [[ "$MODE" == pilot ]]; then
  if ! grep -Eq 'START:|SKIP_REUSED_VERIFIED' "$LOG"; then
    echo "ERROR: pilot neither launched nor reused a canonical case" >&2
    exit 4
  fi
  echo "PILOT PASSED SHELL AND ACCEPTANCE GATES: inspect candidate/terminal metrics with the monitor"
else
  grep -Fq 'Campaign acceptance: planned=48 complete=48 failed_or_incomplete=0' "$LOG" || {
    echo "ERROR: full matrix did not finish all 48 canonical outcomes" >&2
    exit 4
  }
  grep -Fq 'Campaign complete: failures=0' "$LOG" || {
    echo "ERROR: full scheduler reported failures" >&2
    exit 4
  }
  echo "FULL CAMPAIGN PASSED: 48/48 cases reached accepted terminal outcomes"
fi

echo "Log: $LOG"
