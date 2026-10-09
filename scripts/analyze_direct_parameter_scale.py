#!/usr/bin/env python3
"""Dimensional and inverse-action diagnostic for the frozen direct row."""
from __future__ import annotations

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from arrhenius_fracture.checkpoint_v11 import restore_checkpoint


def main():
    out = ROOT/"artifacts/informative_selected_natural_void_trajectory"
    state = restore_checkpoint(out/"step_004_restart.json")
    owner = state.void_state.dislocation_interaction_state
    transport = state.vacancy_transport_state_v2
    row = owner.parameter_row
    direct = {}
    for site in owner.sites:
        action = float(site.interaction_action)
        threshold = float(site.next_threshold())
        direct[site.site_id] = {
            "terminal_action": action,
            "next_stochastic_threshold": threshold,
            "reference_C_dd": row.interaction_geometry_factor_C_dd,
            "inverse_C_dd_for_action_one_holding_history_fixed": (
                row.interaction_geometry_factor_C_dd/action),
            "inverse_C_dd_for_next_threshold_holding_history_fixed": (
                row.interaction_geometry_factor_C_dd*threshold/action),
            "calibration_interpretation": (
                "SCALE_DIAGNOSTIC_ONLY_NOT_A_PROPOSED_OR_CALIBRATED_ROW"),
        }
    vacancy = {site.site_id: {
        "terminal_birth_action": site.birth_action,
        "next_stochastic_threshold": site.next_birth_threshold(),
        "reserved_inventory_area_m2": site.reserved_inventory_area_m2,
    } for site in transport.site_states}
    payload = {
        "parameter_row_id": row.row_id,
        "calibration_status": row.calibration_status,
        "formula": row.interaction_attempt_law,
        "dimensional_analysis": {
            "rho_forest": "m^-2", "equivalent_plastic_strain_rate": "s^-1",
            "burgers_vector": "m", "Gamma_dd": "m^-3 s^-1",
            "C_dd_required_dimension": "dimensionless",
            "integration": "Gamma_dd times support volume gives s^-1",
            "support_volume_policy": "2-D support area times declared unit_thickness_m",
            "unit_thickness_m": row.unit_thickness_m,
            "population_partition": {
                "successful_direct_outcome_fraction": row.successful_direct_outcome_fraction,
                "interpretation": (
                    "multiplicative probability after unweighted interaction opportunity; "
                    "not an additional dislocation population"),
            },
        },
        "direct_sites": direct,
        "vacancy_sites": vacancy,
        "decision": (
            "NO_NEW_DIRECT_PARAMETER_ROW: inverse values require an approximately "
            "10^16-fold change and there is no independent calibration for C_dd, "
            "support normalization, or successful-outcome probability"
        ),
    }
    (out/"direct_parameter_scale.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True)+"\n")


if __name__ == "__main__":
    main()
