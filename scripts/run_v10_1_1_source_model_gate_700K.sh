#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "$SCRIPT_DIR/.." && pwd)
cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"

PYTHON_BIN=${PYTHON_BIN:-python}
T_K=${T_K:-700}
CLASSES=${CLASSES:-"weakT DBTT"}
SOURCE_MODELS=${SOURCE_MODELS:-"continuum finite_sites"}
TARGET_EXT_UM=${TARGET_EXT_UM:-10}
STEPS=${STEPS:-30000}
OUTROOT=${OUTROOT:-runs/v10_1_1_source_model_gate_700K_10um_v1}

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
KINETIC_MAX_ACTION_SUBSTEP=${KINETIC_MAX_ACTION_SUBSTEP:-0.01}
KINETIC_MAX_TRANSLATION_SUBSTEP_M=${KINETIC_MAX_TRANSLATION_SUBSTEP_M:-5e-8}

mkdir -p "$OUTROOT"
MANIFEST="$OUTROOT/source_model_manifest.tsv"
printf 'source_model\tclass\ttemperature_K\ttarget_ext_um\tstatus\toutdir\n' > "$MANIFEST"

for SOURCE_MODEL in $SOURCE_MODELS; do
  case "$SOURCE_MODEL" in
    continuum|finite_sites) ;;
    *) echo "ERROR: SOURCE_MODELS entries must be continuum or finite_sites" >&2; exit 2 ;;
  esac
  for CLASS in $CLASSES; do
    OUTDIR="$OUTROOT/$SOURCE_MODEL/$CLASS/T${T_K}_th${THETA}"
    mkdir -p "$OUTDIR"
    echo "========================================================================"
    echo "v10.1.1 source gate: source=$SOURCE_MODEL class=$CLASS T=${T_K}K target=${TARGET_EXT_UM}um"
    echo "out=$OUTDIR"
    echo "========================================================================"

    status=COMPLETE
    if ! "$PYTHON_BIN" -m arrhenius_fracture.sharp_front_v10_1 \
      --mode 2d --material-class "$CLASS" --temperatures "$T_K" \
      --bulk-plasticity-mode tip_only --directional-j-mode root_signed \
      --tip-kinetics-mode moving_velocity --tip-source-model "$SOURCE_MODEL" \
      --tip-plasticity --active-shielding --signed-active-shielding \
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

    "$PYTHON_BIN" - "$OUTDIR/v10_1_driver_modes.json" "$SOURCE_MODEL" <<'PY'
import json, pathlib, sys
path = pathlib.Path(sys.argv[1])
expected = sys.argv[2]
data = json.loads(path.read_text())
assert data["tip_source_model"] == expected, data
assert data["finite_distributed_source_inventory"] is False, data
PY

    printf '%s\t%s\t%s\t%s\t%s\t%s\n' \
      "$SOURCE_MODEL" "$CLASS" "$T_K" "$TARGET_EXT_UM" "$status" "$OUTDIR" \
      >> "$MANIFEST"

    if [[ "$status" != COMPLETE ]]; then
      echo "ERROR: $SOURCE_MODEL/$CLASS failed; see $OUTDIR" >&2
      exit 1
    fi
  done
done

echo "wrote $MANIFEST"
