#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "$SCRIPT_DIR/.." && pwd)
cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1

PYTHON_BIN=${PYTHON_BIN:-python}
MODE=${MODE:-smoke}
CLASS=${CLASS:-DBTT}
TEMP_K=${TEMP_K:-700}
THETA=${THETA:-45}
R=${R:-0.1}
FREQUENCY_HZ=${FREQUENCY_HZ:-1000}
TRANSPORT_MODE=${TRANSPORT_MODE:-validated_scalar}
FORCE=${FORCE:-1}

case "$MODE" in
  smoke)
    DELTA_KS_MPA=${DELTA_KS_MPA:-"24 27"}
    SEEDS=${SEEDS:-1720}
    TARGET_EXT_UM=${TARGET_EXT_UM:-25}
    CYCLES_MAX=${CYCLES_MAX:-1e9}
    STEPS=${STEPS:-4000}
    ;;
  pilot)
    DELTA_KS_MPA=${DELTA_KS_MPA:-"21 24 27"}
    SEEDS=${SEEDS:-"1720 1721 1722"}
    TARGET_EXT_UM=${TARGET_EXT_UM:-100}
    CYCLES_MAX=${CYCLES_MAX:-1e11}
    STEPS=${STEPS:-20000}
    ;;
  *)
    echo "MODE must be smoke or pilot" >&2
    exit 2
    ;;
esac

MAX_BLOCK_CYCLES=${MAX_BLOCK_CYCLES:-1e8}
MIN_BLOCK_CYCLES=${MIN_BLOCK_CYCLES:-1e-6}
TARGET_DB=${TARGET_DB:-0.1}
# Tighter than v10.2.1 so the newly uncapped feedback is resolved rather than
# lagged across a large tau-leap block. This is a convergence control, not a cap.
TARGET_DN_EMIT=${TARGET_DN_EMIT:-0.05}
TARGET_DN_STORE=${TARGET_DN_STORE:-0.05}
N_PHASE=${N_PHASE:-48}
NX=${NX:-48}
NY=${NY:-96}
TIP_H_FINE=${TIP_H_FINE:-5e-7}
TIP_RATIO=${TIP_RATIO:-1.2}
DU_PROBE=${DU_PROBE:-2e-7}
DT=${DT:-8.4}
DA_CHECKPOINT_M=${DA_CHECKPOINT_M:-5e-6}
MPZ_LENGTH_UM=${MPZ_LENGTH_UM:-100}
MPZ_N_BINS=${MPZ_N_BINS:-200}
WAKE_LENGTH_UM=${WAKE_LENGTH_UM:-100}
WAKE_N_BINS=${WAKE_N_BINS:-0}

export CLEAVAGE_HAZARD_MODE=${CLEAVAGE_HAZARD_MODE:-exponential}
export CLEAVAGE_EVENT_LENGTH_MODE=${CLEAVAGE_EVENT_LENGTH_MODE:-threshold_scaled}
export CLEAVAGE_EVENT_MIN_FACTOR=${CLEAVAGE_EVENT_MIN_FACTOR:-0.5}
export CLEAVAGE_EVENT_MAX_FACTOR=${CLEAVAGE_EVENT_MAX_FACTOR:-4.0}
export CLEAVAGE_EVENT_SUBSEGMENT_FRACTION=${CLEAVAGE_EVENT_SUBSEGMENT_FRACTION:-0.1}
export ANISOTROPIC_USE_AVALANCHE_BACKEND=1
export ANISOTROPIC_TRANSPORT_MODE="$TRANSPORT_MODE"
export ANISOTROPIC_EMISSION_ENABLED=1

GIT_SHORT=$(git rev-parse --short HEAD)
OUTROOT=${OUTROOT:-runs/v10_2_2_uncapped_shielding_${MODE}_${CLASS}_${TEMP_K}K_th${THETA}_${GIT_SHORT}}
if [[ "$FORCE" == 1 ]]; then rm -rf "$OUTROOT"; fi
mkdir -p "$OUTROOT"

printf 'v10.2.2 uncapped physical-shielding sweep\n'
printf '  mode=%s class=%s T=%sK theta=%s R=%s f=%sHz\n' "$MODE" "$CLASS" "$TEMP_K" "$THETA" "$R" "$FREQUENCY_HZ"
printf '  DeltaK levels: %s\n' "$DELTA_KS_MPA"
printf '  seeds: %s\n' "$SEEDS"
printf '  constitutive K-shield cap: OFF; legacy manifest value is diagnostic only\n'

for DELTA_K in $DELTA_KS_MPA; do
  TAG=$(printf '%g' "$DELTA_K" | tr '.' 'p')
  for SEED in $SEEDS; do
    CASE_OUT="$OUTROOT/dK_${TAG}_MPa/seed_${SEED}"
    mkdir -p "$(dirname "$CASE_OUT")"
    export CLEAVAGE_HAZARD_SEED="$SEED"
    printf '\nCASE START DeltaK=%s MPa*sqrt(m) seed=%s out=%s\n' "$DELTA_K" "$SEED" "$CASE_OUT"

    "$PYTHON_BIN" -u -m arrhenius_fracture.sharp_front_v10_2_2 \
      --mode 2d \
      --fatigue-cycles \
      --target-deltaK-MPa-sqrt-m "$DELTA_K" \
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
      --dU "$DU_PROBE" --dt "$DT" --n-stagger 2 \
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
      --fatigue-hold-load \
      --adaptive-events \
      --adaptive-event-target 0.2 \
      --print-every 100 \
      --save-snapshots 0 \
      --no-plots \
      --out "$CASE_OUT"

    "$PYTHON_BIN" - "$CASE_OUT" "$DELTA_K" <<'PY'
import json, math, pathlib, sys
root = pathlib.Path(sys.argv[1])
target = float(sys.argv[2])
fixed = json.loads((root / "v10_2_1_fixed_deltaK_control.json").read_text())
shield = json.loads((root / "v10_2_2_physical_shielding.json").read_text())
assert math.isclose(float(fixed["target_deltaK_MPa_sqrt_m"]), target, rel_tol=0, abs_tol=1e-12)
assert fixed["fixed_deltaK_exact_within_relative_1e-12"] is True, fixed
assert shield["constitutive_K_shield_clip_applied"] is False, shield
assert shield["legacy_manifest_cap_used_in_kinetics"] is False, shield
assert shield["raw_equals_effective_within_relative_1e_12"] is True, shield
assert int(shield["n_shielding_samples"]) > 0, shield
print(json.dumps({
    "target_deltaK_MPa_sqrt_m": target,
    "seed": int(json.loads((root / "v10_2_0_fatigue_reintegration.json").read_text())["cleavage_hazard_seed"]),
    "events": int(fixed["stochastic_geometry_events"]),
    "status": fixed["censor_status"],
    "max_abs_Kshield_MPa_sqrt_m": shield["maximum_abs_raw_K_shield_Pa_sqrt_m"] / 1e6,
    "legacy_cap_exceedance_samples": shield["n_samples_above_legacy_cap_reference"],
    "max_raw_to_legacy_cap_ratio": shield["maximum_raw_to_legacy_cap_ratio"],
}, indent=2))
PY
  done
done

"$PYTHON_BIN" scripts/analyze_v10_2_1_fixed_deltaK_sweep.py "$OUTROOT"
printf '\nv10.2.2 uncapped physical-shielding sweep complete: %s\n' "$OUTROOT"
