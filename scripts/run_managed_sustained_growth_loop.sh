#!/bin/sh
set -eu

: "${SUSTAINED_GROWTH_RUN_OUT:?missing managed output directory}"
: "${PYTHON_BIN:?missing production Python}"

while true; do
  "$PYTHON_BIN" scripts/run_sustained_growth_candidate_v1.py
  if grep -Eq 'FIRST_8UM_PROMOTION_COMMITTED|PHYSICAL_HORIZON_REACHED|MINIMUM_STEP_EQUILIBRIUM_CERTIFICATE_UNAVAILABLE' \
      "$SUSTAINED_GROWTH_RUN_OUT/result.json"; then
    break
  fi
done
