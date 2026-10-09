#!/usr/bin/env python3
"""Honest bounded DBTT reference run with the uncalibrated vacancy row."""
from __future__ import annotations

import json
from pathlib import Path
import sys
from collections.abc import Mapping

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from run_production_vacancy_transport_one_void import (
    physical_transport_interval_from_real_history,
    real_accepted_mechanical_interval,
)


def _plain(value):
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_plain(item) for item in value]
    if hasattr(value, "item"):
        try:
            return value.item()
        except ValueError:
            pass
    return value


def main():
    mechanics = real_accepted_mechanical_interval()
    result = physical_transport_interval_from_real_history(mechanics)
    if not result["passed"]:
        raise RuntimeError(result["blocking_reason"])
    state = mechanics["state_n_plus_1"]
    transport = state.vacancy_transport_state_v2
    report = {
        "material_row": "DBTT",
        "temperature_K": 900.0,
        "load_history": result["load_time_history"],
        "legacy_emission_cleavage_audit": {
            "emission_semantics": "continuous MPZ evolution; no discrete threshold",
            "cleavage_owner": "DirectionalCompetitionState",
            "front_process": result["global_scheduler"]["events"][0].get(
                "legacy_front_process", ()),
        },
        "plastic_vacancy_production_area_m2": transport.ledgers[
            "plastic_production_area_m2"],
        "vacancy_transport_surface_losses_area_m2": {
            "crack": result["crack_surface_absorption_area_m2"],
            "external": result["external_surface_absorption_area_m2"],
            "climb_annihilation": result["climb_annihilation_area_m2"],
            "climb_emission": result["climb_emission_area_m2"],
        },
        "site_inventory_area_m2": result["captured_embryo_inventory_area_m2"],
        "selected_event": result["selected_event"],
        "selected_event_sequence": [row["selected_event"] for row in
                                    result["global_scheduler"]["events"]],
        "length_ledgers": dict(state.void_state.length_ledgers),
        "conservation_residual_area_m2": result["conservation_residual_area_m2"],
        "physical_classification": result["outcome"],
        "vacancy_parameter_status": transport.parameter_row.calibration_status,
    }
    print(json.dumps(_plain(report), sort_keys=True, indent=2))
    print(result["outcome"])


if __name__ == "__main__":
    main()
