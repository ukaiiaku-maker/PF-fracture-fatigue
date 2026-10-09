#!/usr/bin/env python3
"""Bounded common-protocol preflight for the four uncalibrated Ef/Em rows."""
from __future__ import annotations

from dataclasses import replace
import argparse
import hashlib
import json
import math
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from arrhenius_fracture.config import SimulationConfig
from arrhenius_fracture.crack_network_v11 import ROOT_BRANCH_ID
from arrhenius_fracture.emission_derived_plasticity import bind_dislocation_config_to_material_manifest
from arrhenius_fracture.fem import assemble_mechanics
from arrhenius_fracture.materials import PlasticityModel
from arrhenius_fracture.plasticity import update_plasticity
from arrhenius_fracture.topology_transaction_v11 import (
    complete_accepted_state_fingerprint, equilibrate_fixed_load_with_production_fem,
)
from arrhenius_fracture.unified_fracture_material_v5 import (
    material_bundle, material_manifest,
)
from arrhenius_fracture.vacancy_coupling_v1 import equivalent_plastic_strain_rate_s
from arrhenius_fracture.vacancy_transport_phase_field_v2 import (
    EV_J, KB_J_PER_K, VACANCY_ENERGY_STUDY_SPECS,
    equilibrium_vacancy_fraction,
)
from arrhenius_fracture.voiding_production_v5 import (
    _bundle_from_state, advance_production_vacancy_transport_interval,
    build_production_void_state, crack_tip_tensor, directional_clock_rates,
    fresh_sharp_front_engine,
)

TEMPERATURES_K = (900.0, 1200.0)
LOAD_FACTOR = 1.05
INTERVAL_S = 1.0e-10
SITES_M = ((6.0e-4, 6.0e-5), (7.5e-4, -6.0e-5))


def digest(value) -> str:
    return hashlib.sha256(json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False,
    ).encode()).hexdigest()


def accepted_plastic_endpoint(state, temperature_K):
    """Use the already-qualified production Peierls/Taylor endpoint update."""
    config = SimulationConfig()
    config.material = state.material
    config.dislocations.bulk_kinetics_model = "emission_derived_peierls_taylor_multihit"
    bind_dislocation_config_to_material_manifest(
        config.dislocations, material_manifest(_bundle_from_state(state)))
    loaded = equilibrate_fixed_load_with_production_fem(replace(
        state, displacement=np.asarray(state.displacement)*LOAD_FACTOR))
    _, _, sigma, *_ = assemble_mechanics(
        loaded.mesh, loaded.displacement, loaded.ep_gp, loaded.rho_gp,
        loaded.damage, loaded.elasticity_D, loaded.material,
        cohesive_network=loaded.cohesive_network)
    ep, rho, _, audit = update_plasticity(
        loaded.ep_gp.copy(), loaded.rho_gp.copy(), sigma, loaded.material,
        temperature_K, INTERVAL_S,
        PlasticityModel(config.plasticity_barrier, loaded.material),
        config.dislocations, return_info=True)
    endpoint = equilibrate_fixed_load_with_production_fem(replace(
        loaded, ep_gp=ep, rho_gp=rho))
    epdot = equivalent_plastic_strain_rate_s(state.ep_gp, endpoint.ep_gp, INTERVAL_S)
    return endpoint, epdot, audit


def finite(value):
    if isinstance(value, dict):
        return {str(key): finite(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [finite(item) for item in value]
    if isinstance(value, np.ndarray):
        return finite(value.tolist())
    if isinstance(value, np.generic):
        return finite(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return "Infinity" if value > 0 else "-Infinity"
    return value


def run_row(variant, temperature_K):
    state, _ = build_production_void_state(
        bundle=material_bundle("DBTT"), temperature_K=temperature_K,
        vacancy_energy_variant=variant, void_site_centers_m=SITES_M,
        stochastic=True, seed=3621)
    row = state.vacancy_transport_state_v2.parameter_row
    endpoint, epdot, plastic_audit = accepted_plastic_endpoint(state, temperature_K)
    transported, transport_audit = advance_production_vacancy_transport_interval(
        state, endpoint, duration_s=INTERVAL_S)
    transport = transported.vacancy_transport_state_v2
    solver = dict(transport.last_solver_audit)
    tensor, tensor_audit = crack_tip_tensor(endpoint, branch_id=ROOT_BRANCH_ID)
    cleavage = directional_clock_rates(endpoint, tensor, temperature_K=temperature_K)
    engine = fresh_sharp_front_engine(endpoint.material, _bundle_from_state(endpoint))
    sites = np.asarray(SITES_M, dtype=float)
    site_distance = float(np.linalg.norm(sites[1]-sites[0]))
    ledger_delta = {
        key: float(transport.ledgers[key]-state.vacancy_transport_state_v2.ledgers[key])
        for key in transport.ledgers
    }
    return finite({
        "variant": variant,
        "temperature_K": temperature_K,
        "parameter_row": row.__dict__,
        "parameter_row_fingerprint": row.fingerprint,
        "accepted_state_fingerprint": complete_accepted_state_fingerprint(state),
        "mechanical_endpoint_fingerprint": complete_accepted_state_fingerprint(endpoint),
        "transport_endpoint_fingerprint": complete_accepted_state_fingerprint(transported),
        "equilibrium_vacancy_fraction": equilibrium_vacancy_fraction(temperature_K, row),
        "diffusivity_m2_s": (row.vacancy_diffusivity_prefactor_m2_s*math.exp(
            -row.vacancy_migration_energy_eV*EV_J/(KB_J_PER_K*temperature_K))),
        "geometry": {
            "site_centers_m": SITES_M,
            "site_pair_distance_m": site_distance,
            "site_pair_distance_over_process_zone": site_distance/float(engine.f.L_pz),
            "process_zone_length_m": float(engine.f.L_pz),
            "canonical_r_tip_m": float(engine.r_eff()),
        },
        "common_protocol": {
            "load_factor": LOAD_FACTOR, "duration_s": INTERVAL_S,
            "plasticity_model": plastic_audit["bulk_kinetics_model"],
            "actual_evolving_mechanics": True, "physical_rate_scaling": 1.0,
        },
        "plastic": {
            "maximum_accepted_epdot_s": float(np.max(epdot)),
            "nonzero_element_count": int(np.count_nonzero(epdot)),
            "rho_min_m2": float(np.min(endpoint.rho_gp)),
            "rho_max_m2": float(np.max(endpoint.rho_gp)),
        },
        "transport": {
            "event": transport_audit["event"],
            "ledger_increment_area_m2": ledger_delta,
            "site_capture_fraction_area": solver["site_capture_fraction_area"],
            "site_reserved_inventory_area_m2": [
                site.reserved_inventory_area_m2 for site in transport.site_states],
            "site_birth_action": [site.birth_action for site in transport.site_states],
            "site_next_threshold": [site.next_birth_threshold() for site in transport.site_states],
            "plastic_source_fraction_area": solver["plastic_source_fraction_area"],
            "climb_annihilation_fraction_area": solver["climb_annihilation_fraction_area"],
            "climb_emission_fraction_area": solver["climb_emission_fraction_area"],
            "global_conservation_residual_area_m2": solver[
                "global_conservation_residual_area_m2"],
            "conservation_roundoff_budget_area_m2": solver[
                "conservation_roundoff_budget_area_m2"],
        },
        "cleavage": cleavage,
        "crack_tip_tensor_Pa": np.asarray(tensor),
        "crack_tip_tensor_audit": tensor_audit,
    })


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path,
                        default=ROOT/"artifacts/vacancy_energy_variant_preflight")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    rows = [run_row(variant, temperature)
            for variant in VACANCY_ENERGY_STUDY_SPECS
            for temperature in TEMPERATURES_K]
    payload = {
        "schema": "unified-2d.vacancy-energy-variant-preflight/1",
        "calibration_status": "UNCALIBRATED_COMPARATIVE_STUDY",
        "rows": rows,
    }
    payload["authentication_sha256"] = digest(payload)
    (args.out/"preflight.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True)+"\n")
    print("VACANCY_ENERGY_VARIANT_PREFLIGHT = PASS")
    for row in rows:
        print(row["variant"], int(row["temperature_K"]),
              row["equilibrium_vacancy_fraction"], row["diffusivity_m2_s"],
              row["plastic"]["maximum_accepted_epdot_s"],
              row["transport"]["event"])


if __name__ == "__main__":
    main()
