#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$ROOT"

if [[ -n "${TEMPERATURE_K:-}" && "${TEMPERATURE_K}" != "300" ]]; then
  echo "ERROR: the current fatigue campaign is restricted to TEMPERATURE_K=300" >&2
  exit 2
fi

export TEMPERATURE_K=300
export V10230_VHCF_GROWTH_FACTOR=${V10230_VHCF_GROWTH_FACTOR:-16}
export V10230_VHCF_MAX_GROWTH_TRIALS=${V10230_VHCF_MAX_GROWTH_TRIALS:-32}
export V10230_VHCF_MAX_BISECTION_TRIALS=${V10230_VHCF_MAX_BISECTION_TRIALS:-48}
export V10230_VHCF_RELATIVE_CYCLE_TOL=${V10230_VHCF_RELATIVE_CYCLE_TOL:-1e-8}

echo "v10.2.30 room-temperature four-class fatigue qualification"
echo "  T=300 K only"
echo "  classes=peak,DBTT,weakT,ceramic"
echo "  block search=linear seed -> geometric bracket -> log-cycle refinement"

exec bash scripts/run_v10_2_30_four_class_three_deltaK_energy_gate_qualification.sh "$@"
