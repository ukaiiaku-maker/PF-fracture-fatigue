#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "$SCRIPT_DIR/.." && pwd)
cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1

PYTHON_BIN=${PYTHON_BIN:-python}
CLASS=${CLASS:-DBTT}
TEMP_K=${TEMP_K:-700}
THETA=${THETA:-45}
STEPS=${STEPS:-2000}
TARGET_EXT_UM=${TARGET_EXT_UM:-25}
CYCLES_MAX=${CYCLES_MAX:-1e10}
MAX_BLOCK_CYCLES=${MAX_BLOCK_CYCLES:-1e8}
MIN_BLOCK_CYCLES=${MIN_BLOCK_CYCLES:-1e-6}
TARGET_DB=${TARGET_DB:-0.1}
TARGET_DN_EMIT=${TARGET_DN_EMIT:-0.25}
TARGET_DN_STORE=${TARGET_DN_STORE:-0.25}
N_PHASE=${N_PHASE:-48}
FREQUENCY_HZ=${FREQUENCY_HZ:-1000}
R=${R:-0.1}
HAZARD_SEED=${HAZARD_SEED:-1720}
TRANSPORT_MODE=${TRANSPORT_MODE:-validated_scalar}
REQUIRE_EVENT=${REQUIRE_EVENT:-1}
FORCE=${FORCE:-1}

NX=${NX:-48}
NY=${NY:-96}
TIP_H_FINE=${TIP_H_FINE:-5e-7}
TIP_RATIO=${TIP_RATIO:-1.2}
DU=${DU:-2e-7}
DT=${DT:-8.4}
DA_CHECKPOINT_M=${DA_CHECKPOINT_M:-5e-6}
MPZ_LENGTH_UM=${MPZ_LENGTH_UM:-100}
MPZ_N_BINS=${MPZ_N_BINS:-200}
WAKE_LENGTH_UM=${WAKE_LENGTH_UM:-100}
WAKE_N_BINS=${WAKE_N_BINS:-0}

export CLEAVAGE_HAZARD_MODE=${CLEAVAGE_HAZARD_MODE:-exponential}
export CLEAVAGE_HAZARD_SEED="$HAZARD_SEED"
export CLEAVAGE_EVENT_LENGTH_MODE=${CLEAVAGE_EVENT_LENGTH_MODE:-threshold_scaled}
export CLEAVAGE_EVENT_MIN_FACTOR=${CLEAVAGE_EVENT_MIN_FACTOR:-0.5}
export CLEAVAGE_EVENT_MAX_FACTOR=${CLEAVAGE_EVENT_MAX_FACTOR:-4.0}
export CLEAVAGE_EVENT_SUBSEGMENT_FRACTION=${CLEAVAGE_EVENT_SUBSEGMENT_FRACTION:-0.1}
export ANISOTROPIC_USE_AVALANCHE_BACKEND=1
export ANISOTROPIC_TRANSPORT_MODE="$TRANSPORT_MODE"
export ANISOTROPIC_EMISSION_ENABLED=1

GIT_SHORT=$(git rev-parse --short HEAD)
OUTROOT=${OUTROOT:-runs/v10_2_0_fatigue_smoke_${CLASS}_${TEMP_K}K_th${THETA}_${GIT_SHORT}}
if [[ "$FORCE" == 1 ]]; then rm -rf "$OUTROOT"; fi
mkdir -p "$OUTROOT"

"$PYTHON_BIN" -u -m arrhenius_fracture.sharp_front_v10_2_0 \
  --mode 2d \
  --fatigue-cycles \
  --material-class "$CLASS" \
  --temperatures "$TEMP_K" \
  --bulk-plasticity-mode tip_only \
  --directional-j-mode root_signed \
  --tip-kinetics-mode moving_velocity \
  --tip-source-model continuum \
  --tip-plasticity \
  --active-shielding \
  --signed-active-shielding \
  --mobile-shield-fraction 1 \
  --no-wake-shielding \
  --crack-backend sharp_wake \
  --crystal-aniso \
  --crystal-compete \
  --crystal-theta-deg "$THETA" \
  --crystal-material w \
  --j-decomposition cluster \
  --max-fronts 1 \
  --steps "$STEPS" \
  --nx "$NX" --ny "$NY" \
  --tip-h-fine "$TIP_H_FINE" \
  --tip-ratio "$TIP_RATIO" \
  --dU "$DU" --dt "$DT" --n-stagger 2 \
  --da-phys "$DA_CHECKPOINT_M" \
  --target-crack-extension-um "$TARGET_EXT_UM" \
  --mpz-length-um "$MPZ_LENGTH_UM" \
  --mpz-n-bins "$MPZ_N_BINS" \
  --wake-length-um "$WAKE_LENGTH_UM" \
  --wake-n-bins "$WAKE_N_BINS" \
  --R "$R" \
  --frequency-Hz "$FREQUENCY_HZ" \
  --cycles-max "$CYCLES_MAX" \
  --block-cycles "$MAX_BLOCK_CYCLES" \
  --max-block-cycles "$MAX_BLOCK_CYCLES" \
  --min-block-cycles "$MIN_BLOCK_CYCLES" \
  --cycle-block-mode hazard_limited \
  --target-dB "$TARGET_DB" \
  --target-dN-emit "$TARGET_DN_EMIT" \
  --target-dN-store "$TARGET_DN_STORE" \
  --n-phase "$N_PHASE" \
  --no-cyclic-mechanics \
  --adaptive-events \
  --adaptive-event-target 0.2 \
  --print-every 25 \
  --save-snapshots 0 \
  --no-plots \
  --out "$OUTROOT"

"$PYTHON_BIN" - "$OUTROOT" "$REQUIRE_EVENT" <<'PY'
import json, pathlib, sys
root = pathlib.Path(sys.argv[1])
require_event = bool(int(sys.argv[2]))
mode = json.loads((root / "v10_2_0_fatigue_reintegration.json").read_text())
assert mode["fatigue_cycles_enabled"] is True
assert mode["fatigue_front_dispatch"] == "native_moving_mpz_cycle_step_waveform"
assert mode["cleavage_hazard_mode"] == "exponential"
assert mode["event_length_mode"] == "threshold_scaled"

step_paths = sorted(root.glob("steps_*K.csv"))
assert len(step_paths) == 1, step_paths
lines = [line.strip() for line in step_paths[0].read_text().splitlines() if line.strip()]
header = [item.strip() for item in lines[0].lstrip("# ").split(",")]
rows = [dict(zip(header, line.split(","))) for line in lines[1:]]
cycle_key = "fatigue_cycles" if "fatigue_cycles" in header else "cycles"
assert cycle_key in header, header
cycles = [float(row[cycle_key]) for row in rows]
assert any(value > 0.0 for value in cycles), cycles[:5]

kinetic = json.loads((root / "kinetic_tip_cell_audit_v101.json").read_text())
records = kinetic.get("records", [])
fatigue = [row for row in records if row.get("fatigue_native_moving_mpz_dispatch")]
assert fatigue, "no native moving-MPZ fatigue audit records"
assert all(float(row.get("fatigue_cycles", 0.0)) > 0.0 for row in fatigue)
fired = [row for row in fatigue if bool(row.get("fired", False))]
if require_event:
    assert fired, "fatigue smoke produced no stochastic cleavage event"
if fired:
    assert all(float(row.get("avalanche_event_advance_m", 0.0)) > 0.0 for row in fired)
    geometry = json.loads((root / "stochastic_avalanche_geometry_events.json").read_text())
    assert geometry and len(geometry) == len(fired), (len(geometry), len(fired))
    assert all(float(row["event_advance_m"]) > 0.0 for row in geometry)
print(json.dumps({
    "fatigue_blocks": len(fatigue),
    "cycles_total": sum(float(row["fatigue_cycles"]) for row in fatigue),
    "stochastic_events": len(fired),
}, indent=2))
PY

printf 'v10.2.0 fatigue smoke complete: %s\n' "$OUTROOT"
