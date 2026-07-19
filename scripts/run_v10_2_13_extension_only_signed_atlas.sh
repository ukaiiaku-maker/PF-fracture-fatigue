#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "$SCRIPT_DIR/.." && pwd)
cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1

MODE=${MODE:?Set MODE to discover, capture, load-invariance, normalization, build-review, or authorize}
PYTHON_BIN=${PYTHON_BIN:-python}

case "$MODE" in
  discover)
    : "${ATLAS_OUTROOT:?Set ATLAS_OUTROOT to a new reachable-state trace directory}"
    : "${RUN_OUT:?Set RUN_OUT to the underlying 2-D mechanics output directory}"
    : "${MATERIAL_MANIFEST:?Set MATERIAL_MANIFEST to the mechanics-control material CSV}"
    TEMPERATURES=${TEMPERATURES:?Set TEMPERATURES for trajectory discovery}
    EXTRA_ARGS=${EXTRA_ARGS:-}
    # shellcheck disable=SC2086
    exec "$PYTHON_BIN" -m arrhenius_fracture.sharp_front_v10_2_13_capture \
      --atlas-trajectory-only \
      --atlas-outroot "$ATLAS_OUTROOT" \
      --mode 2d --material-manifest "$MATERIAL_MANIFEST" \
      --temperatures $TEMPERATURES --out "$RUN_OUT" \
      --crystal-aniso --crystal-theta-deg "${THETA:-45}" \
      $EXTRA_ARGS
    ;;
  capture)
    : "${STATE_TABLE:?Set STATE_TABLE to the extension-only state table CSV}"
    : "${ATLAS_OUTROOT:?Set ATLAS_OUTROOT to a new snapshot directory}"
    : "${RUN_OUT:?Set RUN_OUT to the underlying 2-D mechanics output directory}"
    : "${MATERIAL_MANIFEST:?Set MATERIAL_MANIFEST to the mechanics-control material CSV}"
    TEMPERATURES=${TEMPERATURES:?Set TEMPERATURES to those present in STATE_TABLE}
    EXTRA_ARGS=${EXTRA_ARGS:-}
    MIN_ELEMENTS_PER_PZ=${MIN_ELEMENTS_PER_PZ:-3}
    # shellcheck disable=SC2086
    exec "$PYTHON_BIN" -m arrhenius_fracture.sharp_front_v10_2_13_capture \
      --atlas-state-table "$STATE_TABLE" \
      --atlas-outroot "$ATLAS_OUTROOT" \
      --minimum-elements-per-process-zone "$MIN_ELEMENTS_PER_PZ" \
      --mode 2d --material-manifest "$MATERIAL_MANIFEST" \
      --temperatures $TEMPERATURES --out "$RUN_OUT" \
      --crystal-aniso --crystal-theta-deg "${THETA:-45}" \
      $EXTRA_ARGS
    ;;
  load-invariance)
    : "${SNAPSHOT:?Set SNAPSHOT to one captured frozen-geometry state directory}"
    : "${OUTROOT:?Set OUTROOT to a new load-invariance output directory}"
    LOAD_SCALES=${LOAD_SCALES:-"0.5 1.0 1.5"}
    MAGNITUDES=${MAGNITUDES:-"0.25 0.50"}
    args=(
      --snapshot "$SNAPSHOT" --outroot "$OUTROOT"
      --load-scales "$LOAD_SCALES" --magnitudes "$MAGNITUDES"
      --linearity-tolerance "${LINEARITY_TOL:-0.03}"
      --load-invariance-tolerance "${LOAD_INVARIANCE_TOL:-0.05}"
    )
    if [[ -n "${RIBBON_WIDTH_M:-}" ]]; then
      args+=(--ribbon-width-m "$RIBBON_WIDTH_M")
    fi
    if [[ -n "${MINIMUM_STATION_SPACING_M:-}" ]]; then
      args+=(--minimum-station-spacing-m "$MINIMUM_STATION_SPACING_M")
    fi
    exec "$PYTHON_BIN" scripts/evaluate_v10_2_13_frozen_geometry_load_invariance.py "${args[@]}"
    ;;
  normalization)
    : "${ENGINE_CONFIG:?Set ENGINE_CONFIG to snapshot.json or a complete engine JSON}"
    : "${OUT:?Set OUT to a new normalization JSON}"
    : "${MINIMUM_SPACING_B:?Set MINIMUM_SPACING_B to the reviewed minimum source spacing in b}"
    : "${MAXIMUM_SPACING_B:?Set MAXIMUM_SPACING_B to the reviewed maximum source spacing in b}"
    args=(
      --engine-config "$ENGINE_CONFIG" --out "$OUT"
      --minimum-spacing-b "$MINIMUM_SPACING_B"
      --maximum-spacing-b "$MAXIMUM_SPACING_B"
    )
    if [[ -n "${SOURCE_REGION_LENGTH_M:-}" ]]; then
      args+=(--source-region-length-m "$SOURCE_REGION_LENGTH_M")
    fi
    exec "$PYTHON_BIN" scripts/build_v10_2_12_mechanics_normalization.py "${args[@]}"
    ;;
  build-review|authorize)
    : "${RESPONSES:?Set RESPONSES to load_scale=1 station-response CSVs}"
    : "${LOAD_INVARIANCE_REPORTS:?Set LOAD_INVARIANCE_REPORTS to the matching audit JSONs}"
    : "${NORMALIZATION:?Set NORMALIZATION to the mechanics normalization JSON}"
    : "${OUT:?Set OUT to a new atlas JSON}"
    args=(
      --normalization "$NORMALIZATION" --out "$OUT"
      --spatial-cross-validation-tolerance "${SPATIAL_CV_TOL:-0.10}"
    )
    for path in $RESPONSES; do
      args+=(--responses "$path")
    done
    for path in $LOAD_INVARIANCE_REPORTS; do
      args+=(--load-invariance "$path")
    done
    if [[ "$MODE" == "authorize" ]]; then
      : "${INDEPENDENT_REVIEW:?Set INDEPENDENT_REVIEW to the completed v10.2.13 review JSON}"
      args+=(--independent-review "$INDEPENDENT_REVIEW" --authorize-production-parameterization)
    elif [[ -n "${INDEPENDENT_REVIEW:-}" ]]; then
      args+=(--independent-review "$INDEPENDENT_REVIEW")
    fi
    exec "$PYTHON_BIN" scripts/build_v10_2_13_extension_only_real_signed_atlas.py "${args[@]}"
    ;;
  *)
    echo "ERROR: invalid MODE=$MODE" >&2
    exit 2
    ;;
esac
