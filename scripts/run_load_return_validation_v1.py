#!/usr/bin/env python3
"""Independent 2-D reference for the frozen 99%-opening then return protocol."""
from __future__ import annotations

from dataclasses import replace
from collections.abc import Mapping
from enum import Enum
import hashlib
import json
import os
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
from scripts.generate_load_aware_supports_v1 import (
    START, canonical_drive_check, inventory, maximum_equivalent_plastic_strain,
    mechanical_observables,
)
from scripts.run_active_two_mechanism_natural_void_study import site_source_exposure

PREDICTION = ROOT.parent/"oneD-v3-stateful-voiding-reduction/analysis_outputs/oneD_v3_state_dependent_transport_v1/prediction.json"
OUT = ROOT/"artifacts/load_return_validation_v1"
TEMPERATURE_K = 1200.0
DURATION_S = 0.1
DELTA0_M = 4.0e-7


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _jsonable(value):
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def _at_opening(state, opening_m: float):
    u = np.asarray(state.displacement, float).copy()
    top, bottom = 0.5*opening_m, -0.5*opening_m
    u[2*np.asarray(state.boundary.top_nodes, int)+1] = top
    u[2*np.asarray(state.boundary.bot_nodes, int)+1] = bottom
    u[2*int(state.boundary.left_bot)+1] = bottom
    return equilibrate_fixed_load_with_production_fem(replace(state, displacement=u))


def _site_rows(state):
    rows = []
    for item in state.void_state.embryo_growth_states:
        tensors, zz = source_local_fem_tensors(state, item)
        rows.append({
            "site_id": item.site_id, "radius_m": item.radius_m,
            "effective_loop_count": item.cavity_emission_state.cumulative_loop_count_effective,
            "source_availability": [source.available_fraction
                                    for source in item.cavity_emission_state.sources],
            "source_tensors_xy_Pa": np.asarray(tensors).tolist(),
            "source_sigma_zz_Pa": np.asarray(zz).tolist(),
        })
    return rows


def _budget(before, after):
    a, b = before.vacancy_transport_state_v2, after.vacancy_transport_state_v2
    keys = (
        "plastic_production_area_m2", "dislocation_climb_annihilation_area_m2",
        "dislocation_climb_emission_area_m2", "external_surface_absorption_area_m2",
        "crack_surface_absorption_area_m2", "subgrid_capture_area_m2",
        "reserved_embryo_inventory_area_m2", "active_cavity_inventory_area_m2",
    )
    return {key: float(b.ledgers[key]-a.ledgers[key]) for key in keys}


def _run_leg(start, ratio: float):
    opening = DELTA0_M*ratio
    stepped = _at_opening(start, opening)
    mechanical, plastic = accepted_bulk_load_step(
        stepped, scope=PlasticityScope.FULL_FIELD_PT_EVOLVING_RHO,
        load_factor=1.0, duration_s=DURATION_S, temperature_K=TEMPERATURE_K,
        maximum_stagger_iterations=80)
    tensor, K, canonical, load_rows = canonical_drive_check(mechanical)
    exposure, source = site_source_exposure(stepped, mechanical, DURATION_S)
    epdot = equivalent_plastic_strain_rate_s(stepped.ep_gp, mechanical.ep_gp, DURATION_S)
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
        "opening_ratio": ratio, "opening_m": opening, "duration_s": DURATION_S,
        "start_fingerprint": complete_accepted_state_fingerprint(stepped),
        "mechanical_fingerprint": complete_accepted_state_fingerprint(mechanical),
        "final_fingerprint": complete_accepted_state_fingerprint(final),
        "initial_H": maximum_equivalent_plastic_strain(stepped),
        "final_H": maximum_equivalent_plastic_strain(mechanical),
        "mechanics": mechanical_observables(mechanical),
        "canonical_drive": canonical, "canonical_K_Pa_sqrt_m": K,
        "canonical_source_tensor_Pa": np.asarray(tensor).tolist(),
        "plastic_work_J_per_m": plastic["accepted_endpoint_path_plastic_work_J_per_m"],
        "integrated_plastic_source_fraction_area": source[
            "integrated_source_fraction_area_per_s"]*DURATION_S,
        "site_source_exposure_fraction_area": exposure,
        "direct_opportunity": direct,
        "initial_field_inventory_area_m2": inventory(stepped),
        "final_field_inventory_area_m2": inventory(final),
        "field_inventory_change_area_m2": inventory(final)-inventory(stepped),
        "budget": _budget(stepped, final),
        "initial_sites": _site_rows(stepped), "final_sites": _site_rows(final),
        "events": [event["selected_event"] for event in scheduler["events"]
                   if event["selected_event"] is not None],
        "scheduler_audit": scheduler,
    }


def main():
    if not PREDICTION.exists():
        raise RuntimeError("OneD load-return prediction must be frozen first")
    prediction_sha = _sha(PREDICTION)
    prediction = json.loads(PREDICTION.read_text())
    if prediction["classification"] != "FROZEN_BEFORE_INDEPENDENT_0P99_RETURN_2D_REFERENCE":
        raise RuntimeError("wrong prospective prediction classification")
    wall = time.perf_counter()
    initial = restore_checkpoint(START)
    initial_fingerprint = complete_accepted_state_fingerprint(initial)
    after_first, first = _run_leg(initial, 0.99)
    after_return, second = _run_leg(after_first, 1.00)
    OUT.mkdir(parents=True, exist_ok=True)
    first_checkpoint = write_checkpoint(after_first, OUT/"after_0p99_hold.json", compression="gzip")
    final_checkpoint = write_checkpoint(after_return, OUT/"after_return_hold.json", compression="gzip")
    payload = {
        "schema": "unified-2d.load-return-validation/1",
        "classification": "INDEPENDENT_AFTER_ONED_PREDICTION_FROZEN",
        "prediction_path": str(PREDICTION),
        "prediction_sha256_before_reference": prediction_sha,
        "common_initial_checkpoint": str(START),
        "common_initial_fingerprint": initial_fingerprint,
        "reference_state_preserved": (
            complete_accepted_state_fingerprint(initial) == initial_fingerprint),
        "state_reset_between_legs": False,
        "first_leg": first, "return_leg": second,
        "first_checkpoint": first_checkpoint, "final_checkpoint": final_checkpoint,
        "wall_time_s": time.perf_counter()-wall,
    }
    temporary = OUT/"reference.json.tmp"
    temporary.write_text(json.dumps(_jsonable(payload), indent=2, sort_keys=True)+"\n")
    os.replace(temporary, OUT/"reference.json")
    print("INDEPENDENT_LOAD_RETURN_REFERENCE_COMPLETE", payload["wall_time_s"])
    print("FIRST_FIELD_CHANGE", first["field_inventory_change_area_m2"])
    print("RETURN_FIELD_CHANGE", second["field_inventory_change_area_m2"])


if __name__ == "__main__":
    main()
