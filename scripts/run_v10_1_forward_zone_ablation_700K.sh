#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "$SCRIPT_DIR/.." && pwd)
cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"

PYTHON_BIN=${PYTHON_BIN:-python}
T_K=${T_K:-700}
CLASSES=${CLASSES:-"ceramic weakT DBTT"}
MODES=${MODES:-"full active_shield_off plasticity_off"}
TARGET_EXT_UM=${TARGET_EXT_UM:-10}
STEPS=${STEPS:-20000}
OUTROOT=${OUTROOT:-runs/v10_1_forward_zone_ablation_700K_10um_v1}

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
THETA=${THETA:-45}
EVENT_TARGET=${EVENT_TARGET:-0.05}
PRINT_EVERY=${PRINT_EVERY:-25}
SAVE_SNAPSHOTS=${SAVE_SNAPSHOTS:-3}
SNAPSHOT_BY_EXT_UM=${SNAPSHOT_BY_EXT_UM:-5}

PACKET_LENGTH_M=${PACKET_LENGTH_M:-2.5e-10}
MOBILE_SHIELD_FRACTION=${MOBILE_SHIELD_FRACTION:-1.0}
KINETIC_MAX_ACTION_SUBSTEP=${KINETIC_MAX_ACTION_SUBSTEP:-0.02}
KINETIC_MAX_TRANSLATION_SUBSTEP_M=${KINETIC_MAX_TRANSLATION_SUBSTEP_M:-1e-7}

"$PYTHON_BIN" - "$DA_CHECKPOINT_M" <<'PY'
import math, sys
value = float(sys.argv[1])
if not math.isclose(value, 5.0e-6, rel_tol=0.0, abs_tol=1.0e-15):
    raise SystemExit(
        "v10.1 validation keeps the calibrated outer FEM checkpoint at 5e-6 m. "
        "Fine-scale advance is integrated internally by the moving-tip cell."
    )
PY

mkdir -p "$OUTROOT"
MANIFEST="$OUTROOT/forward_zone_ablation_manifest.tsv"
printf 'mode\tclass\ttemperature_K\ttip_plasticity\tactive_shielding\twake_shielding\tstatus\toutdir\n' > "$MANIFEST"

for MODE in $MODES; do
  case "$MODE" in
    full)
      kinetic_args=(--tip-plasticity --active-shielding)
      expect_plastic=1
      expect_active=1
      ;;
    active_shield_off)
      kinetic_args=(--tip-plasticity --no-active-shielding)
      expect_plastic=1
      expect_active=0
      ;;
    plasticity_off)
      kinetic_args=(--no-tip-plasticity --no-active-shielding)
      expect_plastic=0
      expect_active=0
      ;;
    *)
      echo "ERROR: unknown mode '$MODE'" >&2
      exit 2
      ;;
  esac

  for CLASS in $CLASSES; do
    OUTDIR="$OUTROOT/$MODE/$CLASS/T${T_K}_th${THETA}"
    mkdir -p "$OUTDIR"
    echo "========================================================================"
    echo "v10.1 moving-tip gate: mode=$MODE class=$CLASS T=${T_K}K target=${TARGET_EXT_UM}um"
    echo "out=$OUTDIR"
    echo "========================================================================"

    status=COMPLETE
    if ! "$PYTHON_BIN" -m arrhenius_fracture.sharp_front_v10_1 \
      --mode 2d --material-class "$CLASS" --temperatures "$T_K" \
      --bulk-plasticity-mode tip_only --directional-j-mode root_signed \
      --tip-kinetics-mode moving_velocity \
      "${kinetic_args[@]}" \
      --signed-active-shielding \
      --mobile-shield-fraction "$MOBILE_SHIELD_FRACTION" \
      --kinetic-packet-length-m "$PACKET_LENGTH_M" \
      --kinetic-max-action-substep "$KINETIC_MAX_ACTION_SUBSTEP" \
      --kinetic-max-translation-substep-m "$KINETIC_MAX_TRANSLATION_SUBSTEP_M" \
      --steps "$STEPS" --nx "$NX" --ny "$NY" \
      --dU "$DU" --dt "$DT" --n-stagger "$N_STAGGER" \
      --tip-h-fine "$TIP_H_FINE" --tip-ratio "$TIP_RATIO" \
      --da-phys "$DA_CHECKPOINT_M" --target-crack-extension-um "$TARGET_EXT_UM" \
      --mpz-length-um "$MPZ_LENGTH_UM" --mpz-n-bins "$MPZ_N_BINS" \
      --wake-length-um "$WAKE_LENGTH_UM" --wake-n-bins "$WAKE_N_BINS" \
      --no-wake-shielding \
      --crystal-aniso --crystal-compete --crystal-theta-deg "$THETA" \
      --crystal-material w --j-decomposition cluster \
      --max-fronts 1 --adaptive-events --adaptive-event-target "$EVENT_TARGET" \
      --print-every "$PRINT_EVERY" --save-snapshots "$SAVE_SNAPSHOTS" \
      --snapshot-by-crack-extension-um "$SNAPSHOT_BY_EXT_UM" \
      --out "$OUTDIR"; then
      status=FAILED
    fi

    "$PYTHON_BIN" - "$OUTDIR/v10_1_driver_modes.json" "$expect_plastic" "$expect_active" <<'PY'
import json, pathlib, sys
path = pathlib.Path(sys.argv[1])
expected_plastic = bool(int(sys.argv[2]))
expected_active = bool(int(sys.argv[3]))
data = json.loads(path.read_text())
assert data["tip_kinetics_mode"] == "moving_velocity", data
assert data["tip_plasticity_enabled"] is expected_plastic, data
assert data["active_shielding_enabled"] is expected_active, data
assert data["wake_shielding"] is False, data
PY

    printf '%s\t%s\t%s\t%s\t%s\t0\t%s\t%s\n' \
      "$MODE" "$CLASS" "$T_K" "$expect_plastic" "$expect_active" "$status" "$OUTDIR" \
      >> "$MANIFEST"

    if [[ "$status" != COMPLETE ]]; then
      echo "ERROR: $MODE/$CLASS failed; see $OUTDIR" >&2
      exit 1
    fi
  done
done

echo "wrote $MANIFEST"
