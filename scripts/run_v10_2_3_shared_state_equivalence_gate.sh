#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "$SCRIPT_DIR/.." && pwd)
cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1

PYTHON_BIN=${PYTHON_BIN:-python}
CANDIDATE=${CANDIDATE:-DBTT_A0002277}
TEMP_K=${TEMP_K:-600}
THETA=${THETA:-45}
TARGET_EXT_UM=${TARGET_EXT_UM:-5}
TRANSPORT_MODE=${TRANSPORT_MODE:-validated_scalar}
OUTROOT=${OUTROOT:-runs/v10_2_3_shared_state_equivalence_${CANDIDATE}_${TEMP_K}K_v1}

STEPS=${STEPS:-4000}
PRINT_EVERY=${PRINT_EVERY:-100}
NX=${NX:-48}
NY=${NY:-96}
TIP_H_FINE=${TIP_H_FINE:-5e-7}
TIP_RATIO=${TIP_RATIO:-1.2}
DU=${DU:-2e-7}
DT=${DT:-8.4}
N_STAGGER=${N_STAGGER:-2}
DA_CHECKPOINT_M=${DA_CHECKPOINT_M:-5e-6}
MPZ_LENGTH_UM=${MPZ_LENGTH_UM:-100}
MPZ_N_BINS=${MPZ_N_BINS:-200}
WAKE_LENGTH_UM=${WAKE_LENGTH_UM:-100}
WAKE_N_BINS=${WAKE_N_BINS:-0}
PACKET_LENGTH_M=${PACKET_LENGTH_M:-2.5e-10}
KINETIC_MAX_ACTION_SUBSTEP=${KINETIC_MAX_ACTION_SUBSTEP:-0.01}
KINETIC_MAX_TRANSLATION_SUBSTEP_M=${KINETIC_MAX_TRANSLATION_SUBSTEP_M:-5e-8}

MANIFEST="$REPO_ROOT/arrhenius_fracture/data/materials/fallback_dbtt/${CANDIDATE}.csv"
TWO_D="$OUTROOT/two_d"
REPLAY="$OUTROOT/replay"

stamp() { date '+%Y-%m-%d %H:%M:%S'; }
report() { printf '[%s] %s\n' "$(stamp)" "$*"; }

case "$CANDIDATE" in
  DBTT_A0002333|DBTT_A0003837|DBTT_A0002277) ;;
  *)
    echo "ERROR: unsupported preserved fallback candidate: $CANDIDATE" >&2
    exit 2
    ;;
esac

test -f "$MANIFEST" || {
  echo "ERROR: fallback manifest is missing: $MANIFEST"
  exit 1
}

test ! -e "$OUTROOT" && test ! -e "${OUTROOT}.log" || {
  echo "ERROR: output or external log already exists:"
  echo "  $OUTROOT"
  echo "  ${OUTROOT}.log"
  exit 1
}
mkdir -p "$TWO_D"

report "2-D TRACE START candidate=$CANDIDATE T=${TEMP_K}K theta=${THETA} transport=$TRANSPORT_MODE"
env \
  CLEAVAGE_HAZARD_MODE=deterministic \
  CLEAVAGE_HAZARD_SEED=0 \
  CLEAVAGE_EVENT_LENGTH_MODE=fixed \
  ANISOTROPIC_EMISSION_ENABLED=1 \
  ANISOTROPIC_USE_AVALANCHE_BACKEND=0 \
  ANISOTROPIC_TRANSPORT_MODE="$TRANSPORT_MODE" \
  ANISOTROPIC_CRYSTAL_THETA_DEG="$THETA" \
  ANISOTROPIC_SHARED_FOREST_DENSITY=1 \
  ANISOTROPIC_REQUIRE_RELIABLE_PROBE=1 \
  "$PYTHON_BIN" -u -m arrhenius_fracture.sharp_front_v10_2_3 \
    --mode 2d \
    --material-manifest "$MANIFEST" \
    --temperatures "$TEMP_K" \
    --bulk-plasticity-mode tip_only \
    --directional-j-mode root_signed \
    --tip-kinetics-mode moving_velocity \
    --tip-source-model continuum \
    --tip-plasticity \
    --active-shielding \
    --signed-active-shielding \
    --mobile-shield-fraction 1 \
    --kinetic-packet-length-m "$PACKET_LENGTH_M" \
    --kinetic-max-action-substep "$KINETIC_MAX_ACTION_SUBSTEP" \
    --kinetic-max-translation-substep-m "$KINETIC_MAX_TRANSLATION_SUBSTEP_M" \
    --steps "$STEPS" \
    --nx "$NX" --ny "$NY" \
    --dU "$DU" --dt "$DT" --n-stagger "$N_STAGGER" \
    --tip-h-fine "$TIP_H_FINE" --tip-ratio "$TIP_RATIO" \
    --da-phys "$DA_CHECKPOINT_M" \
    --target-crack-extension-um "$TARGET_EXT_UM" \
    --mpz-length-um "$MPZ_LENGTH_UM" --mpz-n-bins "$MPZ_N_BINS" \
    --wake-length-um "$WAKE_LENGTH_UM" --wake-n-bins "$WAKE_N_BINS" \
    --no-wake-shielding \
    --crack-backend sharp_wake \
    --crystal-aniso \
    --crystal-compete \
    --crystal-theta-deg "$THETA" \
    --crystal-material w \
    --j-decomposition cluster \
    --max-fronts 1 \
    --adaptive-events \
    --print-every "$PRINT_EVERY" \
    --save-snapshots 0 \
    --no-plots \
    --out "$TWO_D" \
    > "$TWO_D/two_d_trace.log" 2>&1
report "2-D TRACE COMPLETE"

for required in \
  v10_2_3_2d_replay_schedule.csv \
  v10_2_3_2d_final_state.npz \
  v10_2_3_2d_engine_config.json \
  v10_2_3_2d_state_trace.json \
  v10_2_2_physical_shielding.json; do
  test -f "$TWO_D/$required" || {
    echo "ERROR: required 2-D trace output is missing: $TWO_D/$required"
    tail -n 100 "$TWO_D/two_d_trace.log" || true
    exit 1
  }
done

report "SHARED-STATE REPLAY START"
"$PYTHON_BIN" -u scripts/compare_v10_2_3_shared_state_replay.py \
  --trace-root "$TWO_D" \
  --candidate "$CANDIDATE" \
  --mode full \
  --out "$REPLAY" \
  --relative-tolerance 1e-10 \
  --absolute-tolerance 1e-10 \
  > "$OUTROOT/replay_comparison.log" 2>&1 || {
    report "SHARED-STATE REPLAY FAILED equivalence criteria"
    tail -n 120 "$OUTROOT/replay_comparison.log" || true
    exit 1
  }
report "SHARED-STATE REPLAY PASSED"

"$PYTHON_BIN" - "$OUTROOT" "$CANDIDATE" "$TEMP_K" "$THETA" <<'PY'
import json
import pathlib
import sys

root = pathlib.Path(sys.argv[1])
assessment = json.loads(
    (root / "replay" / "shared_state_replay_equivalence.json").read_text()
)
summary = {
    "schema": "v10.2.3_shared_state_equivalence_gate",
    "candidate_id": sys.argv[2],
    "temperature_K": float(sys.argv[3]),
    "theta_deg": float(sys.argv[4]),
    "passed": bool(assessment["passed"]),
    "maximum_array_relative_error": assessment["maximum_array_relative_error"],
    "maximum_array_absolute_error": assessment["maximum_array_absolute_error"],
    "maximum_scalar_absolute_error": assessment["maximum_scalar_absolute_error"],
    "all_fired_flags_match": assessment["all_fired_flags_match"],
    "raw_equals_effective_when_shielding_active": assessment[
        "raw_equals_effective_when_shielding_active"
    ],
}
(root / "shared_state_equivalence_gate.json").write_text(json.dumps(summary, indent=2))
print(json.dumps(summary, indent=2))
PY

report "V10.2.3 SHARED-STATE EQUIVALENCE GATE COMPLETE root=$OUTROOT"
