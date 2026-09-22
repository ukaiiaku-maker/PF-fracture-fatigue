#!/usr/bin/env python3
"""Prospective source/capture/sink screen for the next natural-void protocol."""
from __future__ import annotations

import json
import math
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from arrhenius_fracture.bulk_plasticity_v104 import (
    PlasticityScope, accepted_bulk_load_step, vacancy_source_diagnostics,
)
from arrhenius_fracture.crack_network_v11 import ROOT_BRANCH_ID
from arrhenius_fracture.dislocation_interaction_voiding_v1 import accepted_interaction_opportunity
from arrhenius_fracture.vacancy_coupling_v1 import equivalent_plastic_strain_rate_s
from arrhenius_fracture.voiding_production_v5 import (
    advance_production_vacancy_transport_interval, crack_tip_tensor,
    directional_clock_rates,
)
from scripts.run_active_two_mechanism_natural_void_study import finite, make_initial

OUT = ROOT/"artifacts/informative_natural_void_protocol_screen_1ms"
TEMPERATURE_K = 1200.0
DURATION_S = 1.0e-3
LOAD_FACTORS = (2.0, 3.0, 4.0, 6.0)
PLASTIC_INCREMENT_LIMIT = 1.0e-2
DILUTE_CONCENTRATION_LIMIT = 1.0e-2


def dump(path, value):
    path.write_text(json.dumps(finite(value), indent=2, sort_keys=True)+"\n")


def run(load_factor):
    initial = make_initial("combined")
    mechanical, plastic = accepted_bulk_load_step(
        initial, scope=PlasticityScope.FULL_FIELD_PT_EVOLVING_RHO,
        load_factor=load_factor, duration_s=DURATION_S,
        temperature_K=TEMPERATURE_K, maximum_stagger_iterations=80,
    )
    epdot = equivalent_plastic_strain_rate_s(
        initial.ep_gp, mechanical.ep_gp, DURATION_S)
    source = vacancy_source_diagnostics(initial, mechanical, DURATION_S)
    transported, audit = advance_production_vacancy_transport_interval(
        initial, mechanical, duration_s=DURATION_S)
    transport = transported.vacancy_transport_state_v2
    solver = dict(transport.last_solver_audit)
    direct = [accepted_interaction_opportunity(
        mechanical, site, temperature_K=TEMPERATURE_K,
        accepted_equivalent_plastic_strain_rate_s=epdot,
    ) for site in mechanical.void_state.dislocation_interaction_state.sites]
    tensor, _ = crack_tip_tensor(mechanical, branch_id=ROOT_BRANCH_ID)
    fracture = directional_clock_rates(
        mechanical, tensor, temperature_K=TEMPERATURE_K)
    finite_crossings = [float(row["crossing_time_s"]) for row in fracture
                        if math.isfinite(float(row["crossing_time_s"]))]
    row = {
        "load_factor": load_factor, "temperature_K": TEMPERATURE_K,
        "duration_s": DURATION_S,
        "maximum_plastic_increment": float(np.max(epdot)*DURATION_S),
        "plastic_work_J_per_m": float(plastic["accepted_endpoint_path_plastic_work_J_per_m"]),
        "plastic_source_fraction_area": float(
            source["integrated_source_fraction_area_per_s"]*DURATION_S),
        "site_capture_fraction_area": solver.get("site_capture_fraction_area"),
        "climb_annihilation_fraction_area": solver.get("climb_annihilation_fraction_area"),
        "surface_sink_fraction_area": solver.get("surface_sink_fraction_area"),
        "maximum_vacancy_fraction": float(np.max(transport.vacancy_fraction_gp)),
        "site_reserved_inventory_area_m2": [
            site.reserved_inventory_area_m2 for site in transport.site_states],
        "site_birth_action": [site.birth_action for site in transport.site_states],
        "direct_successful_rate_s": [
            item["successful_direct_interaction_rate_s"] for item in direct],
        "direct_unweighted_rate_s": [
            item["support_integrated_interaction_rate_s"] for item in direct],
        "local_fracture_first_crossing_s": (
            min(finite_crossings) if finite_crossings else math.inf),
        "plastic_increment_valid": float(np.max(epdot)*DURATION_S) <= PLASTIC_INCREMENT_LIMIT,
        "dilute_concentration_valid": (
            float(np.max(transport.vacancy_fraction_gp)) <= DILUTE_CONCENTRATION_LIMIT),
        "transport_event": audit["event"],
    }
    return row


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for factor in LOAD_FACTORS:
        try:
            row = run(factor)
        except Exception as exc:
            row = {"load_factor": factor, "status": "FAIL_CLOSED",
                   "error_type": type(exc).__name__, "error": str(exc)}
        rows.append(row)
        dump(OUT/f"load_{factor:g}.json", row)
        print("NATURAL_SCREEN_COMPLETE", factor, row.get("status", "AVAILABLE"), flush=True)
    payload = {
        "classification": "PROSPECTIVE_UNSCALED_PROTOCOL_SCREEN",
        "vacancy_energy_variant": "combined",
        "temperature_K": TEMPERATURE_K, "duration_s": DURATION_S,
        "load_factors_declared_before_outcomes": LOAD_FACTORS,
        "plastic_increment_limit": PLASTIC_INCREMENT_LIMIT,
        "dilute_concentration_limit": DILUTE_CONCENTRATION_LIMIT,
        "selection_policy": (
            "highest site capture among rows satisfying plastic and dilute bounds; "
            "canonical fracture confirmation required before trajectory"
        ),
        "rows": rows,
    }
    dump(OUT/"screen.json", payload)


if __name__ == "__main__":
    main()
