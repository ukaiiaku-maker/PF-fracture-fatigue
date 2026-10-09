#!/usr/bin/env python3
"""Frozen-history source/capture/sink/action diagnosis and next protocol."""
from __future__ import annotations

import json
import math
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from arrhenius_fracture.checkpoint_v11 import restore_checkpoint

OUT = ROOT / "artifacts/informative_selected_natural_void_trajectory"


def main():
    result = json.loads((OUT / "result.json").read_text())
    history = result["history"]
    state = restore_checkpoint(OUT / result["latest_valid_restart"])
    transport = state.vacancy_transport_state_v2
    direct_owner = state.void_state.dislocation_interaction_state
    ledgers = dict(transport.ledgers)

    tracked = (
        "plastic_production_area_m2",
        "dislocation_climb_annihilation_area_m2",
        "dislocation_climb_emission_area_m2",
        "external_surface_absorption_area_m2",
        "crack_surface_absorption_area_m2",
        "subgrid_capture_area_m2",
    )
    previous_ledgers = {key: 0.0 for key in tracked}
    previous_reserved = {site.site_id: 0.0 for site in transport.site_states}
    previous_birth = {site.site_id: 0.0 for site in transport.site_states}
    previous_direct = {site.site_id: 0.0 for site in direct_owner.sites}
    intervals = []
    for row in history:
        dt = float(row["dt_s"])
        current_ledgers = row["transport_ledgers"]
        ledger_increment = {
            key: float(current_ledgers[key])-previous_ledgers[key] for key in tracked
        }
        reserved_increment = {
            site_id: float(value)-previous_reserved[site_id]
            for site_id, value in row["site_reserved_inventory_area_m2"].items()
        }
        birth_increment = {
            site_id: float(value)-previous_birth[site_id]
            for site_id, value in row["vacancy_birth_action"].items()
        }
        direct_increment = {
            site_id: float(value)-previous_direct[site_id]
            for site_id, value in row["direct_interaction_action"].items()
        }
        production = ledger_increment["plastic_production_area_m2"]
        capture = math.fsum(reserved_increment.values())
        intervals.append({
            "step": row["step"], "dt_s": dt, "terminal_time_s": row["time_s"],
            "maximum_plastic_increment": row["maximum_plastic_increment"],
            "generation_rate_area_per_s": production/dt,
            "climb_loss_rate_area_per_s": (
                ledger_increment["dislocation_climb_annihilation_area_m2"]/dt
            ),
            "surface_loss_rate_area_per_s": (
                (ledger_increment["external_surface_absorption_area_m2"]
                 + ledger_increment["crack_surface_absorption_area_m2"])/dt
            ),
            "capture_rate_area_per_s": capture/dt,
            "capture_over_generation": capture/max(production, 1.0e-300),
            "climb_over_generation": (
                ledger_increment["dislocation_climb_annihilation_area_m2"]
                / max(production, 1.0e-300)
            ),
            "reserved_inventory_increment_area_m2": reserved_increment,
            "birth_action_increment": birth_increment,
            "birth_action_per_captured_area_m2": {
                site_id: birth_increment[site_id]/max(reserved_increment[site_id], 1e-300)
                for site_id in reserved_increment
            },
            "direct_action_increment": direct_increment,
            "direct_action_rate_per_s": {
                site_id: value/dt for site_id, value in direct_increment.items()
            },
        })
        previous_ledgers = {key: float(current_ledgers[key]) for key in tracked}
        previous_reserved = {
            key: float(value) for key, value in row[
                "site_reserved_inventory_area_m2"].items()
        }
        previous_birth = {
            key: float(value) for key, value in row["vacancy_birth_action"].items()
        }
        previous_direct = {
            key: float(value) for key, value in row["direct_interaction_action"].items()
        }

    final_field_inventory = float(np.sum(
        np.asarray(transport.vacancy_fraction_gp)
        * np.asarray(transport.transport_mesh.area_e)
    ))
    sites = {}
    terminal_interval = intervals[-1]
    transport_by_id = {site.site_id: site for site in transport.site_states}
    direct_by_id = {site.site_id: site for site in direct_owner.sites}
    for site_id, site in transport_by_id.items():
        threshold = float(site.next_birth_threshold())
        action = float(site.birth_action)
        reserved = float(site.reserved_inventory_area_m2)
        action_rate = terminal_interval["birth_action_increment"][site_id] / terminal_interval["dt_s"]
        capture_rate = terminal_interval["reserved_inventory_increment_area_m2"][site_id] / terminal_interval["dt_s"]
        direct_site = direct_by_id[site_id]
        sites[site_id] = {
            "reserved_inventory_area_m2": reserved,
            "one_hit_inventory_area_m2": (
                site.required_embryo_inventory_area_m2/site.required_hit_count
            ),
            "one_hit_inventory_fraction": (
                reserved/(site.required_embryo_inventory_area_m2/site.required_hit_count)
            ),
            "birth_action": action,
            "next_birth_threshold": threshold,
            "birth_threshold_fraction": action/threshold,
            "terminal_capture_rate_area_per_s": capture_rate,
            "terminal_birth_action_rate_per_s": action_rate,
            "fixed_terminal_rate_seconds_to_one_hit_inventory": (
                max(site.required_embryo_inventory_area_m2/site.required_hit_count-reserved, 0.0)
                / max(capture_rate, 1.0e-300)
            ),
            "fixed_terminal_rate_seconds_to_birth_threshold": (
                max(threshold-action, 0.0)/max(action_rate, 1.0e-300)
            ),
            "direct_action": float(direct_site.interaction_action),
            "next_direct_threshold": float(direct_site.next_threshold()),
            "direct_threshold_fraction": (
                float(direct_site.interaction_action)/float(direct_site.next_threshold())
            ),
        }

    production = float(ledgers["plastic_production_area_m2"])
    payload = {
        "schema": "informative.natural-void-source-budget-diagnosis/1",
        "classification": "FROZEN_ACCEPTED_HISTORY_DIAGNOSTIC_NOT_A_TRAJECTORY",
        "reference_result": "result.json",
        "accepted_checkpoint": result["latest_valid_restart"],
        "reference_parameters_unchanged": True,
        "terminal_budget": {
            "plastic_generation_area_m2": production,
            "climb_annihilation_area_m2": ledgers[
                "dislocation_climb_annihilation_area_m2"],
            "climb_over_generation": ledgers[
                "dislocation_climb_annihilation_area_m2"]/production,
            "surface_absorption_area_m2": (
                ledgers["external_surface_absorption_area_m2"]
                + ledgers["crack_surface_absorption_area_m2"]
            ),
            "captured_reserved_area_m2": ledgers["subgrid_capture_area_m2"],
            "capture_over_generation": ledgers["subgrid_capture_area_m2"]/production,
            "field_inventory_area_m2": final_field_inventory,
            "initial_field_inventory_area_m2": ledgers[
                "initial_owned_inventory_area_m2"],
        },
        "accepted_interval_rates": intervals,
        "site_bottlenecks": sites,
        "direct_reference_diagnosis": {
            "row_id": direct_owner.parameter_row.row_id,
            "C_dd": direct_owner.parameter_row.interaction_geometry_factor_C_dd,
            "calibration_status": direct_owner.parameter_row.calibration_status,
            "decision": (
                "KEEP_REFERENCE_UNCHANGED; direct exposure is negligible and no "
                "population/thickness-normalized sensitivity row is justified yet"
            ),
        },
        "prospectively_selected_next_protocol": {
            "protocol_id": "REFERENCE_CONTINUATION_CAPTURE_VS_ACTION_10S_V1",
            "selection_status": "SELECTED_BEFORE_FULL_OUTCOME_RUN",
            "start_checkpoint": result["latest_valid_restart"],
            "material_family": result["material_family"],
            "vacancy_energy_variant": result["vacancy_energy_variant"],
            "temperature_K": result["temperature_K"],
            "load_policy": "UNCHANGED_ACCEPTED_HOLD_PROTOCOL_LOAD_FACTOR_1",
            "additional_step_durations_s": [1.0, 2.0, 3.0, 4.0],
            "maximum_additional_duration_s": 10.0,
            "stop_conditions": [
                "FIRST_ACTUAL_EVENT", "TEN_SECONDS_ADDITIONAL_ACCEPTED_TIME",
                "INVENTORY_OR_ACTION_CONSERVATION_FAILURE",
            ],
            "decision_observables": [
                "one_hit_inventory_fraction", "birth_threshold_fraction",
                "climb_over_generation", "capture_over_generation",
            ],
            "interpretation": (
                "Tests whether a first-hit inventory becomes available while "
                "successful-birth action remains limiting; it does not alter "
                "climb, vacancy energies, C_dd, thresholds, or RNG"
            ),
            "full_outcome_run_started": False,
        },
    }
    (OUT / "source_capture_action_diagnosis.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps({
        "climb_over_generation": payload["terminal_budget"]["climb_over_generation"],
        "capture_over_generation": payload["terminal_budget"]["capture_over_generation"],
        "site_bottlenecks": sites,
        "selected_protocol": payload["prospectively_selected_next_protocol"]["protocol_id"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
