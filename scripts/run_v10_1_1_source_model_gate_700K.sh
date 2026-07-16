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
# Validated moving-tip cases complete in <150 outer steps. A 500-step ceiling
# prevents a failed source law from loading indefinitely to nonphysical K.
STEPS=${STEPS:-500}
K_FIRST_MAX_MPA_SQRT_M=${K_FIRST_MAX_MPA_SQRT_M:-100}
OUTROOT=${OUTROOT:-runs/v10_1_3_source_model_gate_700K_10um_v1}

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
    echo "v10.1.3 source gate: source=$SOURCE_MODEL class=$CLASS T=${T_K}K target=${TARGET_EXT_UM}um"
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

    if [[ "$status" == COMPLETE ]]; then
      if ! "$PYTHON_BIN" - \
        "$OUTDIR/v10_1_driver_modes.json" \
        "$OUTDIR/summary.json" \
        "$OUTDIR/kinetic_tip_cell_audit_v101.json" \
        "$SOURCE_MODEL" \
        "$TARGET_EXT_UM" \
        "$DA_CHECKPOINT_M" \
        "$K_FIRST_MAX_MPA_SQRT_M" <<'PY'
import json, math, pathlib, sys

mode_path = pathlib.Path(sys.argv[1])
summary_path = pathlib.Path(sys.argv[2])
audit_path = pathlib.Path(sys.argv[3])
expected_source = sys.argv[4]
target_um = float(sys.argv[5])
da_m = float(sys.argv[6])
kmax = float(sys.argv[7])

modes = json.loads(mode_path.read_text())
assert modes["tip_source_model"] == expected_source, modes
assert modes["finite_distributed_source_inventory"] is False, modes

summary = json.loads(summary_path.read_text())
assert summary and isinstance(summary, list), summary
row = summary[0]
minimum_advances = max(1, math.ceil(target_um / (da_m * 1.0e6) - 1.0e-12))
assert int(row["n_advances"]) >= minimum_advances, row
kc = float(row["Kc_first_MPa_sqrt_m"])
assert math.isfinite(kc) and 0.0 < kc <= kmax, row

if expected_source == "continuum":
    audit = json.loads(audit_path.read_text())
    records = audit.get("records", [])
    assert records, audit
    required = (
        "tip_source_local_density_m2",
        "tip_source_backstress_shear_Pa",
        "tip_source_backstress_equivalent_Pa",
        "tip_source_effective_emission_stress_Pa",
    )
    for record in records:
        for key in required:
            value = float(record[key])
            assert math.isfinite(value) and value >= 0.0, (key, value, record)
    assert max(float(r["tip_source_backstress_equivalent_Pa"]) for r in records) > 0.0
PY
      then
        status=FAILED
      fi
    fi

    printf '%s\t%s\t%s\t%s\t%s\t%s\n' \
      "$SOURCE_MODEL" "$CLASS" "$T_K" "$TARGET_EXT_UM" "$status" "$OUTDIR" \
      >> "$MANIFEST"

    if [[ "$status" != COMPLETE ]]; then
      echo "ERROR: $SOURCE_MODEL/$CLASS failed or did not reach the required crack advance; see $OUTDIR" >&2
      exit 1
    fi
  done
done

echo "wrote $MANIFEST"
