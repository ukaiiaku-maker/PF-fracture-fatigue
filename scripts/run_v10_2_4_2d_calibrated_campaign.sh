#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "$SCRIPT_DIR/.." && pwd)
cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1

PYTHON_BIN=${PYTHON_BIN:-python}
OUTROOT=${OUTROOT:-runs/v10_2_4_2d_calibrated_reduced_campaign_v1}
SAMPLES=${SAMPLES:-256}
SEED=${SEED:-1201}
WORKERS=${WORKERS:-2}
PROMOTE=${PROMOTE:-16}
KDOT=${KDOT:-0.005}
KMAX=${KMAX:-80}
DK=${DK:-0.05}
TRANSPORT_MODE=${TRANSPORT_MODE:-validated_scalar}
THETA=${THETA:-45}
TARGET_EXT_UM=${TARGET_EXT_UM:-5}

stamp() { date '+%Y-%m-%d %H:%M:%S'; }
report() { printf '[%s] %s\n' "$(stamp)" "$*"; }

test ! -e "$OUTROOT" && test ! -e "${OUTROOT}.log" || {
  echo "ERROR: output or log already exists:"
  echo "  $OUTROOT"
  echo "  ${OUTROOT}.log"
  exit 1
}

mkdir -p "$OUTROOT/atlas_traces"

CASES=(
  "DBTT_A0002333 400"
  "DBTT_A0002333 500"
  "DBTT_A0003837 900"
  "DBTT_A0003837 1000"
  "DBTT_A0002277 600"
  "DBTT_A0002277 700"
)

TRACE_ARGS=()
for spec in "${CASES[@]}"; do
  read -r candidate temperature <<< "$spec"
  trace_root="$OUTROOT/atlas_traces/${candidate}_${temperature}K"
  report "ATLAS TRACE START candidate=$candidate T=${temperature}K"
  CANDIDATE="$candidate" \
  TEMP_K="$temperature" \
  THETA="$THETA" \
  TARGET_EXT_UM="$TARGET_EXT_UM" \
  TRANSPORT_MODE="$TRANSPORT_MODE" \
  OUTROOT="$trace_root" \
  bash scripts/run_v10_2_3_shared_state_equivalence_gate.sh \
    > "$OUTROOT/atlas_traces/${candidate}_${temperature}K.console.log" 2>&1
  report "ATLAS TRACE COMPLETE candidate=$candidate T=${temperature}K"
  TRACE_ARGS+=(--trace-root "$trace_root")
done

ATLAS="$OUTROOT/mechanical_atlas.csv"
report "MECHANICAL ATLAS BUILD START"
"$PYTHON_BIN" -u scripts/build_v10_2_4_mechanical_atlas.py \
  "${TRACE_ARGS[@]}" \
  --checkpoint-da-um 5 \
  --out "$ATLAS" \
  > "$OUTROOT/mechanical_atlas_build.log" 2>&1
report "MECHANICAL ATLAS BUILD COMPLETE atlas=$ATLAS"

CAMPAIGN="$OUTROOT/reduced_campaign"
report "REDUCED CAMPAIGN START samples=$SAMPLES workers=$WORKERS"
"$PYTHON_BIN" -u scripts/run_v10_2_4_reduced_campaign.py \
  --atlas "$ATLAS" \
  --out "$CAMPAIGN" \
  --samples "$SAMPLES" \
  --seed "$SEED" \
  --workers "$WORKERS" \
  --promote "$PROMOTE" \
  --Kdot "$KDOT" \
  --Kmax "$KMAX" \
  --dK "$DK" \
  --target-extension-um "$TARGET_EXT_UM" \
  --checkpoint-da-um 5 \
  --transport-mode "$TRANSPORT_MODE" \
  2>&1 | tee "$OUTROOT/reduced_campaign.console.log"

for required in \
  mechanical_atlas.csv \
  mechanical_atlas.csv.json \
  reduced_campaign/candidate_scores.csv \
  reduced_campaign/promoted_candidates.csv \
  reduced_campaign/campaign_assessment.json; do
  test -f "$OUTROOT/$required" || {
    echo "ERROR: required v10.2.4 output is missing: $OUTROOT/$required"
    exit 1
  }
done

report "V10.2.4 2-D-CALIBRATED REDUCED CAMPAIGN COMPLETE root=$OUTROOT"
