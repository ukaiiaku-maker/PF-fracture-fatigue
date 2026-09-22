#!/usr/bin/env python3
"""Generate non-validation 97/101% load supports from the 8.47499-s state."""
from __future__ import annotations

from dataclasses import replace
import json
import math
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from arrhenius_fracture.bulk_plasticity_v104 import PlasticityScope, accepted_bulk_load_step
from arrhenius_fracture.checkpoint_v11 import restore_checkpoint, write_checkpoint
from arrhenius_fracture.crack_network_v11 import ROOT_BRANCH_ID
from arrhenius_fracture.dislocation_interaction_voiding_v1 import accepted_interaction_opportunity
from arrhenius_fracture.embryo_growth_v1 import source_local_fem_tensors
from arrhenius_fracture.global_physical_scheduler_v1 import advance_global_physical_interval
from arrhenius_fracture.topology_transaction_v11 import (
    complete_accepted_state_fingerprint, equilibrate_fixed_load_with_production_fem,
)
from arrhenius_fracture.vacancy_coupling_v1 import equivalent_plastic_strain_rate_s
from scripts.run_active_two_mechanism_natural_void_study import (
    canonical_drive_check, site_source_exposure,
)

RUN = ROOT/"runs/managed_sustained_growth_v1_20260920/output"
START = RUN/"restart_000008475ms.json"
OUT = ROOT/"artifacts/nearby_load_supports_v1"
BASELINE_OPENING_M = 4.0e-7
TEMPERATURE_K = 1200.0
DURATION_S = 0.1


def maximum_equivalent_plastic_strain(state) -> float:
    ep = np.asarray(state.ep_gp, float)
    return float(np.max(np.sqrt(ep[0]**2+ep[1]**2+2.0*ep[2]**2)))


def at_opening(state, opening_m: float):
    u = np.asarray(state.displacement, float).copy()
    top = 0.5*float(opening_m); bottom = -top
    u[2*np.asarray(state.boundary.top_nodes, int)+1] = top
    u[2*np.asarray(state.boundary.bot_nodes, int)+1] = bottom
    u[2*int(state.boundary.left_bot)+1] = bottom
    return equilibrate_fixed_load_with_production_fem(replace(state, displacement=u))


def site_rows(state):
    growth = {item.site_id: item for item in state.void_state.embryo_growth_states}
    rows = []
    for site_id in sorted(growth):
        item = growth[site_id]
        tensors, zz = source_local_fem_tensors(state, item)
        rows.append({
            "site_id": site_id, "radius_m": item.radius_m,
            "area_m2": item.area_m2,
            "effective_loop_count": item.cavity_emission_state.cumulative_loop_count_effective,
            "source_availability": [source.available_fraction
                                    for source in item.cavity_emission_state.sources],
            "source_tensors_xy_Pa": np.asarray(tensors).tolist(),
            "source_sigma_zz_Pa": np.asarray(zz).tolist(),
        })
    return rows


def inventory(state) -> float:
    transport = state.vacancy_transport_state_v2
    return float(np.dot(np.asarray(transport.transport_mesh.area_e),
                        np.asarray(transport.vacancy_fraction_gp)))


def mechanical_observables(state):
    eq = state.equilibrium_observables
    return {
        "top_reaction_N_per_m": eq.top_normal_resultant_N_per_m,
        "bottom_reaction_N_per_m": eq.bottom_normal_resultant_N_per_m,
        "mean_reaction_magnitude_N_per_m": eq.mean_reaction_magnitude_N_per_m,
        "compliance_m2_per_N": eq.compliance_m2_per_N,
        "recoverable_energy_J_per_m": eq.recoverable_elastic_energy_J_per_m,
        "total_internal_energy_J_per_m": eq.total_stored_internal_energy_J_per_m,
        "complete_force_accounting": state.energy_ledgers[
            "complete_equilibrium_force_accounting"],
    }


def run_case(reference, ratio: float, role: str):
    opening = BASELINE_OPENING_M*float(ratio)
    stepped = at_opening(reference, opening)
    mechanical, plastic = accepted_bulk_load_step(
        stepped, scope=PlasticityScope.FULL_FIELD_PT_EVOLVING_RHO,
        load_factor=1.0, duration_s=DURATION_S, temperature_K=TEMPERATURE_K,
        maximum_stagger_iterations=80)
    tensor, K, canonical, load_rows = canonical_drive_check(mechanical)
    exposure, source = site_source_exposure(stepped, mechanical, DURATION_S)
    epdot = equivalent_plastic_strain_rate_s(
        stepped.ep_gp, mechanical.ep_gp, DURATION_S)
    direct = [dict(accepted_interaction_opportunity(
        mechanical, site, temperature_K=TEMPERATURE_K,
        accepted_equivalent_plastic_strain_rate_s=epdot))
        for site in mechanical.void_state.dislocation_interaction_state.sites]
    final, scheduler = advance_global_physical_interval(
        stepped, mechanical, DURATION_S,
        {"kind": "hold", "temperature_K": TEMPERATURE_K,
         "stress_tensor_Pa": tensor,
         "K_effective_by_front": {ROOT_BRANCH_ID: K},
         "directional_load_rows_by_front": {ROOT_BRANCH_ID: load_rows},
         "maximum_events": 64})
    return final, {
        "role": role, "load_ratio": ratio, "opening_m": opening,
        "transition": "INSTANTANEOUS_DISPLACEMENT_STEP_AT_T0_THEN_0P1S_HOLD",
        "start_accepted_state_fingerprint": complete_accepted_state_fingerprint(stepped),
        "endpoint_accepted_state_fingerprint": complete_accepted_state_fingerprint(final),
        "initial_H": maximum_equivalent_plastic_strain(stepped),
        "endpoint_H": maximum_equivalent_plastic_strain(mechanical),
        "history_rate_s": (maximum_equivalent_plastic_strain(mechanical)
                           -maximum_equivalent_plastic_strain(stepped))/DURATION_S,
        "initial_mechanics": mechanical_observables(stepped),
        "endpoint_mechanics": mechanical_observables(mechanical),
        "canonical_drive": canonical,
        "canonical_K_Pa_sqrt_m": K,
        "canonical_source_tensor_Pa": np.asarray(tensor).tolist(),
        "plastic_work_J_per_m": plastic["accepted_endpoint_path_plastic_work_J_per_m"],
        "integrated_plastic_source_fraction_area": (
            source["integrated_source_fraction_area_per_s"]*DURATION_S),
        "site_source_exposure_fraction_area": exposure,
        "direct_opportunity": direct,
        "initial_vacancy_inventory_fraction_area": inventory(stepped),
        "final_vacancy_inventory_fraction_area": inventory(final),
        "initial_sites": site_rows(stepped), "final_sites": site_rows(final),
        "scheduler_events": [event["selected_event"] for event in scheduler["events"]
                             if event["selected_event"] is not None],
        "scheduler_audit": dict(scheduler),
    }


def main():
    wall = time.perf_counter()
    reference = restore_checkpoint(START)
    before = complete_accepted_state_fingerprint(reference)
    OUT.mkdir(parents=True, exist_ok=True)
    cases = []
    for ratio in (0.97, 1.01):
        final, row = run_case(reference, ratio, "TRAIN_LOAD_SUPPORT")
        name = f"support_{ratio:.2f}".replace(".", "p")+".json"
        manifest = write_checkpoint(final, OUT/name, compression="gzip")
        row["checkpoint"] = manifest
        cases.append(row)
    result = {
        "schema": "unified-2d.nearby-load-supports/1",
        "reference_checkpoint": str(START),
        "reference_state_preserved": (
            complete_accepted_state_fingerprint(reference) == before),
        "validation_ratio_excluded": 0.98,
        "duration_s": DURATION_S,
        "temperature_K": TEMPERATURE_K,
        "cases": cases, "wall_time_s": time.perf_counter()-wall,
    }
    (OUT/"supports.json").write_text(json.dumps(result, indent=2, sort_keys=True)+"\n")
    print(OUT/"supports.json")
    print("LOAD_SUPPORTS", [(row["load_ratio"], row["history_rate_s"]) for row in cases])


if __name__ == "__main__":
    main()
