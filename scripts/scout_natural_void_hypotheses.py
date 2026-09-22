#!/usr/bin/env python3
"""Prospective bounded scout for one natural-voiding hypothesis row."""
from __future__ import annotations

from dataclasses import asdict, replace
from collections.abc import Mapping
import json
import math
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from arrhenius_fracture.bulk_plasticity_v104 import PlasticityScope, accepted_bulk_load_step
from arrhenius_fracture.crack_network_v11 import ROOT_BRANCH_ID
from arrhenius_fracture.dislocation_interaction_voiding_v1 import (
    DislocationInteractionParameterRowV1,
    accepted_interaction_opportunity,
)
from arrhenius_fracture.embryo_growth_v1 import (
    CavityEmissionParameterRowV1,
    cavity_surface_emission_rates,
    initialize_embryo_growth_state,
    loop_area_kinematics,
)
from arrhenius_fracture.unified_fracture_material_v5 import material_bundle
from arrhenius_fracture.vacancy_coupling_v1 import equivalent_plastic_strain_rate_s
from arrhenius_fracture.vacancy_transport_phase_field_v2 import accepted_elastoplastic_stress_3d
from arrhenius_fracture.voiding_production_v5 import (
    build_production_void_state,
    directional_sharp_front_rates,
    sharp_front_load_provider,
)


OUT = ROOT / "artifacts/natural_void_hypothesis_scout_v1"
TEMPERATURE_K = 1200.0
SEED = 3621
SITES_M = ((6.0e-4, 0.0), (7.5e-4, 0.0))
PARTITION = (0.50, 0.10, 0.40)
SCOUT_DT_S = 1.0e-5
LOAD_FACTOR = 1.5
C_DD_GRID = (1.0e-8, 1.0e-6, 1.0e-4)
BARRIER_GRID_EV = (0.25, 0.50, 0.75)
EMISSION_BARRIER_GRID_J = (1.2e-19, 1.8e-19, 2.6e-19)


def finite(value):
    if isinstance(value, Mapping):
        return {str(k): finite(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [finite(v) for v in value]
    if isinstance(value, np.ndarray):
        return finite(value.tolist())
    if isinstance(value, np.generic):
        return finite(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return "Infinity" if value > 0.0 else "-Infinity"
    return value


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    bundle = material_bundle("DBTT")
    initial, _ = build_production_void_state(
        bundle=bundle, temperature_K=TEMPERATURE_K,
        vacancy_energy_variant="combined", void_site_centers_m=SITES_M,
        stochastic=True, seed=SEED,
        direct_interaction_birth_enabled=True,
        cavity_surface_loop_emission_enabled=True,
        interaction_population_partition=PARTITION,
        bulk_initial_density_m2=float(bundle.fracture_row["rho_forest_floor_m2"]),
        initial_plastic_strain_gp=0.0,
    )
    mechanical, plastic = accepted_bulk_load_step(
        initial, scope=PlasticityScope.FULL_FIELD_PT_EVOLVING_RHO,
        load_factor=LOAD_FACTOR, duration_s=SCOUT_DT_S,
        temperature_K=TEMPERATURE_K, maximum_stagger_iterations=80,
    )
    epdot = equivalent_plastic_strain_rate_s(initial.ep_gp, mechanical.ep_gp, SCOUT_DT_S)
    provider = sharp_front_load_provider(
        mechanical, branch_id=ROOT_BRANCH_ID, matched_discretization=True,
    )
    fracture_rates = directional_sharp_front_rates(
        mechanical, provider["directional"], branch_id=ROOT_BRANCH_ID,
        temperature_K=TEMPERATURE_K,
    )
    crack_crossing = min(
        (float(row["crossing_time_s"]) for row in fracture_rates
         if row["directional_drive_status"] == "AVAILABLE"),
        default=math.inf,
    )
    stress3d = accepted_elastoplastic_stress_3d(mechanical)
    centers = np.asarray(mechanical.mesh.nodes)[np.asarray(mechanical.mesh.elems)].mean(axis=1)
    rows = []
    base_owner = mechanical.void_state.dislocation_interaction_state
    base_site = base_owner.sites[0]
    for C_dd in C_DD_GRID:
        for barrier_eV in BARRIER_GRID_EV:
            row = DislocationInteractionParameterRowV1(
                row_id=("hypothesis.elevated-mixed-v1:Cdd=" + f"{C_dd:.0e}"
                        + ":barrier-eV=" + f"{barrier_eV:.2f}"),
                interaction_geometry_factor_C_dd=C_dd,
                interaction_barrier_eV=barrier_eV,
            )
            sites = tuple(replace(
                site,
                interaction_barrier_eV=row.interaction_barrier_eV,
                required_interaction_hits=row.required_interaction_hits,
                direct_free_volume_per_hit_area_m2=row.direct_free_volume_per_hit_area_m2,
            ) for site in base_owner.sites)
            owner = replace(base_owner, parameter_row=row, sites=sites)
            trial = replace(
                mechanical,
                void_state=replace(mechanical.void_state, dislocation_interaction_state=owner),
            )
            audit = accepted_interaction_opportunity(
                trial, sites[0], temperature_K=TEMPERATURE_K,
                accepted_equivalent_plastic_strain_rate_s=epdot,
            )
            rate = float(audit["successful_direct_interaction_rate_s"])
            birth_estimate = float(sites[0].interaction_thresholds[-1]) / max(rate, 1.0e-300)
            embryo_area = row.required_interaction_hits * row.direct_free_volume_per_hit_area_m2
            embryo_radius = math.sqrt(embryo_area / math.pi)
            eid = int(np.argmin(np.linalg.norm(centers-np.asarray(sites[0].center_xy_m), axis=1)))
            tensor = np.array(((stress3d["sigma_xx_Pa"][eid], stress3d["sigma_xy_Pa"][eid]),
                               (stress3d["sigma_xy_Pa"][eid], stress3d["sigma_yy_Pa"][eid])))
            for emission_barrier_J in EMISSION_BARRIER_GRID_J:
                emission_row = CavityEmissionParameterRowV1(
                    row_id=("hypothesis.elevated-emission-v1:G0-J="
                            + f"{emission_barrier_J:.1e}"),
                    zero_stress_barrier_J=emission_barrier_J,
                    burgers_m=row.burgers_m,
                    model_thickness_m=row.unit_thickness_m,
                    loop_swept_area_m2=row.loop_swept_area_m2,
                    chi_loop=row.chi_loop,
                    loop_projection=row.loop_projection,
                )
                growth = initialize_embryo_growth_state(
                    site_id=sites[0].site_id, radius_m=embryo_radius,
                    initial_area_m2=0.0, dislocation_derived_area_m2=embryo_area,
                    direct_interaction_birth_hits=row.required_interaction_hits,
                    emission_enabled=True, row=emission_row, seed=SEED + 9001,
                )
                emission_rates, emission_audit = cavity_surface_emission_rates(
                    growth, stress_tensor_Pa=tensor,
                    sigma_zz_Pa=float(stress3d["sigma_zz_Pa"][eid]),
                    temperature_K=TEMPERATURE_K,
                )
                area_per_loop = loop_area_kinematics(emission_row, embryo_radius)[
                    "selected_area_m2"
                ]
                emission_area_rate = float(np.sum(emission_rates))*float(area_per_loop)
                promotion_area = math.pi*(8.0e-6)**2
                promotion_estimate = max(promotion_area-embryo_area, 0.0)/max(
                    emission_area_rate, 1.0e-300,
                )
                eligible = (
                    1.0e-5 <= birth_estimate <= 10.0
                    and birth_estimate < crack_crossing
                    and promotion_estimate <= 100.0
                )
                rows.append(finite({
                    "parameter_row": asdict(row),
                    "parameter_row_fingerprint": row.fingerprint,
                    "cavity_emission_parameter_row": asdict(emission_row),
                    "cavity_emission_parameter_row_fingerprint": emission_row.fingerprint,
                    "successful_direct_rate_s": rate,
                    "constant_endpoint_three_hit_time_estimate_s": birth_estimate,
                    "crack_crossing_time_s": crack_crossing,
                    "embryo_radius_from_three_hits_m": embryo_radius,
                    "emission_area_rate_at_endpoint_m2_s": emission_area_rate,
                    "constant_endpoint_promotion_time_estimate_s": promotion_estimate,
                    "thermal_success_probability": audit["thermal_success_probability"],
                    "support_volume_m3": audit["support_volume_m3"],
                    "emission_source_audit": emission_audit,
                    "eligible": eligible,
                }))
    eligible = [item for item in rows if item["eligible"]]
    if not eligible:
        (OUT / "no_selection_diagnostic.json").write_text(
            json.dumps(finite({"rows": rows}), indent=2, sort_keys=True) + "\n"
        )
        raise RuntimeError("bounded prospective grid contains no birth-and-growth candidate")
    # Conservative selection within the declared grid: smallest interaction
    # geometry factor, then largest barrier, among candidates passing the
    # prospective birth/growth/crack-ordering windows.
    selected = sorted(
        eligible,
        key=lambda item: (
            item["parameter_row"]["interaction_geometry_factor_C_dd"],
            -item["parameter_row"]["interaction_barrier_eV"],
            -item["cavity_emission_parameter_row"]["zero_stress_barrier_J"],
        ),
    )[0]
    payload = finite({
        "schema": "unified-2d.natural-void-hypothesis-scout/1",
        "classification": "PROSPECTIVE_UNCALIBRATED_PARAMETER_HYPOTHESIS",
        "selection_completed_before_full_trajectory": True,
        "temperature_K": TEMPERATURE_K,
        "geometry": {"site_centers_m": SITES_M, "aligned_mode_I": True},
        "loading": {"initial_load_factor": LOAD_FACTOR, "scout_duration_s": SCOUT_DT_S},
        "stochastic_seed": SEED,
        "parameter_grid": {"C_dd": C_DD_GRID, "interaction_barrier_eV": BARRIER_GRID_EV,
                           "cavity_emission_zero_stress_barrier_J": EMISSION_BARRIER_GRID_J},
        "selection_rule": (
            "pass declared birth/growth/crack-ordering windows; then smallest C_dd "
            "and largest barrier; no observed full-trajectory outcome used"
        ),
        "rows": rows,
        "selected_parameter_row": selected["parameter_row"],
        "selected_parameter_row_fingerprint": selected["parameter_row_fingerprint"],
        "selected_cavity_emission_parameter_row": selected[
            "cavity_emission_parameter_row"],
        "selected_cavity_emission_parameter_row_fingerprint": selected[
            "cavity_emission_parameter_row_fingerprint"],
        "reference_preserved": True,
        "vacancy_channel": "ACTIVE_COMBINED_EF2P4_EV_EM0P9_EV_HYPOTHESIS",
        "growth_channel": "EXISTING_CAVITY_SURFACE_LOOP_EMISSION",
        "plastic_work_J_per_m": plastic["accepted_endpoint_path_plastic_work_J_per_m"],
    })
    (OUT / "scout.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print("NATURAL_VOID_HYPOTHESIS_SCOUT = PASS")
    print("SELECTED_ROW", selected["parameter_row"]["row_id"])
    print("BIRTH_ESTIMATE_S", selected["constant_endpoint_three_hit_time_estimate_s"])
    print("PROMOTION_ESTIMATE_S", selected["constant_endpoint_promotion_time_estimate_s"])


if __name__ == "__main__":
    main()
