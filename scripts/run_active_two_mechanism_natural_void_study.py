#!/usr/bin/env python3
"""Active, unscaled two-mechanism natural-void comparison under finite creep."""
from __future__ import annotations

import argparse
from collections.abc import Mapping
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
from arrhenius_fracture.checkpoint_v11 import restore_checkpoint, write_checkpoint
from arrhenius_fracture.crack_network_v11 import ROOT_BRANCH_ID
from arrhenius_fracture.dislocation_interaction_voiding_v1 import accepted_interaction_opportunity
from arrhenius_fracture.global_physical_scheduler_v1 import advance_global_physical_interval
from arrhenius_fracture.topology_transaction_v11 import complete_accepted_state_fingerprint
from arrhenius_fracture.unified_fracture_material_v5 import material_bundle
from arrhenius_fracture.vacancy_coupling_v1 import equivalent_plastic_strain_rate_s
from arrhenius_fracture.vacancy_transport_phase_field_v2 import (
    VACANCY_ENERGY_STUDY_SPECS, equilibrium_vacancy_fraction,
)
from arrhenius_fracture.voiding_production_v5 import (
    build_production_void_state, crack_tip_tensor, directional_sharp_front_rates,
    sharp_front_load_provider,
)

TEMPERATURE_K = 1200.0
SITES_M = ((6.0e-4, 6.0e-5), (7.5e-4, -6.0e-5))
PARTITION = (0.50, 0.10, 0.40)
# Resolves the combined-row ~2.1e-4 s early relaxation scale, then continues
# through a full second of accepted, newly computed plastic evolution.
STEP_DURATIONS_S = (
    1e-5, 2e-5, 7e-5, 2e-4, 7e-4, 2e-3, 7e-3, 2e-2, 7e-2,
    0.1, 0.1, 0.1,
)


def finite(value):
    if isinstance(value, Mapping): return {str(k): finite(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)): return [finite(v) for v in value]
    if isinstance(value, np.ndarray): return finite(value.tolist())
    if isinstance(value, np.generic): return finite(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return "Infinity" if value > 0 else "-Infinity"
    return value


def make_initial(variant):
    bundle = material_bundle("DBTT")
    return build_production_void_state(
        bundle=bundle, temperature_K=TEMPERATURE_K,
        vacancy_energy_variant=variant, void_site_centers_m=SITES_M,
        stochastic=True, seed=3621,
        direct_interaction_birth_enabled=True,
        cavity_surface_loop_emission_enabled=True,
        interaction_population_partition=PARTITION,
        bulk_initial_density_m2=float(bundle.fracture_row["rho_forest_floor_m2"]),
        initial_plastic_strain_gp=0.0,
    )[0]


def canonical_drive_check(state):
    tensor, tensor_support = crack_tip_tensor(state, branch_id=ROOT_BRANCH_ID)
    # Probe local-contour availability without executing an incomplete
    # marginal family.  When local J is unavailable, the only consumed result
    # is the established qualified 4/5-level paired-energy family below; the
    # former default 2/3/4 family was always discarded before that retry.
    provider = sharp_front_load_provider(
        state, branch_id=ROOT_BRANCH_ID, matched_discretization=True,
        marginal_delta_a_m=(3.0e-5, 2.0e-5),
        marginal_mesh_levels=(4,),
        virtual_mechanical_trial_transport="deferred_read_only")
    if not any(row["directional_drive_status"] == "AVAILABLE"
               for row in provider["directional"]):
        repaired = sharp_front_load_provider(
            state, branch_id=ROOT_BRANCH_ID, matched_discretization=True,
            marginal_delta_a_m=(3.0e-5, 2.0e-5),
            marginal_mesh_levels=(4, 5),
            virtual_mechanical_trial_transport="deferred_read_only",
            mechanical_refinement_policy="legacy_accumulated_generations",
        )
        repaired["directional"] = [{**row, "representation_retry": {
            "trigger": "REQUIRED_DEFAULT_MARGINAL_DRIVE_UNAVAILABLE",
            "physical_perturbation_family_m": [3.0e-5, 2.0e-5],
            "mesh_levels": [4, 5],
            "tolerance_changed": False,
            "physical_time_advanced": False,
        }} for row in repaired["directional"]]
        provider = repaired
    rates = directional_sharp_front_rates(
        state, provider["directional"], branch_id=ROOT_BRANCH_ID,
        temperature_K=TEMPERATURE_K)
    available = [row for row in rates if row["directional_drive_status"] == "AVAILABLE"]
    if not available:
        raise RuntimeError(
            "REQUIRED_FRACTURE_OBSERVATION_UNAVAILABLE_AFTER_BOUNDED_REFINEMENT"
        )
    K = max((row["K_effective_Pa_sqrt_m"] for row in available), default=0.0)
    return tensor, K, finite({
        "adjacent_element_tensor_Pa": tensor,
        "adjacent_element_ids": tensor_support,
        "canonical_directional_rows": provider["directional"],
        "fracture_engine_rows": rates,
        "state_bound_K_effective_Pa_sqrt_m": K,
        "interpretation": (
            "CANONICAL_DRIVE_AVAILABLE" if available else
            "CANONICAL_MARGINAL_NOT_CONVERGED_FAIL_CLOSED"
        ),
    }), provider["directional"]


def site_source_exposure(previous, mechanical, dt):
    source = vacancy_source_diagnostics(previous, mechanical, dt)
    transport = previous.vacancy_transport_state_v2
    parent = np.asarray(transport.parent_mechanics_element_ids, int)
    centers = np.asarray(mechanical.mesh.nodes)[np.asarray(mechanical.mesh.elems)].mean(axis=1)[parent]
    area = np.asarray(transport.transport_mesh.area_e)
    total = np.asarray(source["total_rate_gp_s"])
    rows = {}
    for site in transport.site_states:
        radius = float(site.support_radius_m)
        mask = np.linalg.norm(centers-np.asarray(site.center_xy_m), axis=1) <= radius
        rows[site.site_id] = float(np.sum(area[mask]*total[mask])*dt)
    return rows, source


def run_variant(variant, outdir):
    initial = make_initial(variant)
    state = initial
    cumulative_work = 0.0
    cumulative_site_source = {site.site_id: 0.0 for site in state.vacancy_transport_state_v2.site_states}
    history = []
    canonical = None
    for index, dt in enumerate(STEP_DURATIONS_S):
        print("NATURAL_STEP", variant, index+1, dt, flush=True)
        previous = state
        mechanical, plastic = accepted_bulk_load_step(
            previous, scope=PlasticityScope.FULL_FIELD_PT_EVOLVING_RHO,
            load_factor=1.5 if index == 0 else 1.0, duration_s=dt,
            temperature_K=TEMPERATURE_K, maximum_stagger_iterations=80,
        )
        local_exposure, source = site_source_exposure(previous, mechanical, dt)
        for key, value in local_exposure.items(): cumulative_site_source[key] += value
        tensor, support = crack_tip_tensor(mechanical, branch_id=ROOT_BRANCH_ID)
        _, K_effective, canonical, load_rows = canonical_drive_check(mechanical)
        direct_previews = []
        epdot = equivalent_plastic_strain_rate_s(previous.ep_gp, mechanical.ep_gp, dt)
        for site in mechanical.void_state.dislocation_interaction_state.sites:
            direct_previews.append(accepted_interaction_opportunity(
                mechanical, site, temperature_K=TEMPERATURE_K,
                accepted_equivalent_plastic_strain_rate_s=epdot,
            ))
        state, scheduler = advance_global_physical_interval(
            previous, mechanical, dt,
            {"kind": "hold", "temperature_K": TEMPERATURE_K,
             "stress_tensor_Pa": tensor,
             "K_effective_by_front": {ROOT_BRANCH_ID: K_effective},
             "directional_load_rows_by_front": {ROOT_BRANCH_ID: load_rows},
             "maximum_events": 16},
        )
        cumulative_work += float(plastic["accepted_endpoint_path_plastic_work_J_per_m"])
        transport = state.vacancy_transport_state_v2
        solver = dict(transport.last_solver_audit)
        history.append(finite({
            "step": index+1, "dt_s": dt, "time_s": sum(STEP_DURATIONS_S[:index+1]),
            "accepted_mechanical_state_fingerprint": complete_accepted_state_fingerprint(
                mechanical),
            "canonical_drive_check": canonical,
            "maximum_cumulative_equivalent_plastic_strain": float(np.max(
                equivalent_plastic_strain_rate_s(initial.ep_gp, state.ep_gp, 1.0))),
            "cumulative_plastic_work_J_per_m": cumulative_work,
            "rho_min_m2": float(np.min(state.rho_gp)), "rho_max_m2": float(np.max(state.rho_gp)),
            "site_local_source_exposure_fraction_area": dict(cumulative_site_source),
            "vacancy_birth_action": {s.site_id: s.birth_action for s in transport.site_states},
            "site_reserved_inventory_area_m2": {s.site_id: s.reserved_inventory_area_m2 for s in transport.site_states},
            "direct_interaction_action": {s.site_id: s.interaction_action for s in state.void_state.dislocation_interaction_state.sites},
            "direct_opportunity": direct_previews,
            "integrated_source_fraction_area_per_s": source["integrated_source_fraction_area_per_s"],
            "transport_ledgers": dict(transport.ledgers),
            "local_equilibrium_fraction_min": solver.get("local_equilibrium_fraction_min"),
            "local_equilibrium_fraction_max": solver.get("local_equilibrium_fraction_max"),
            "supersaturation_fraction_area": solver.get("positive_supersaturation_fraction_area"),
            "eligible_owner_registry": [p for event in scheduler["events"] for p in event["eligible_previews"]],
            "all_owners_previewed": scheduler["all_owners_previewed"],
            "selected_events": [event["selected_event"] for event in scheduler["events"]],
            "solver_audit": solver,
        }))
    checkpoint = outdir/f"{variant}_latest_valid_restart.json"
    write_checkpoint(state, checkpoint, compression="gzip")
    restored = restore_checkpoint(checkpoint)
    transport = state.vacancy_transport_state_v2
    direct = state.void_state.dislocation_interaction_state
    field_path = outdir/f"{variant}_terminal_fields.npz"
    centers = np.asarray(transport.transport_mesh.nodes)[
        np.asarray(transport.transport_mesh.elems)].mean(axis=1)
    np.savez_compressed(
        field_path,
        transport_centers_xy_m=centers,
        vacancy_fraction_gp=np.asarray(transport.vacancy_fraction_gp),
        mechanics_ep_gp=np.asarray(state.ep_gp), rho_gp_m2=np.asarray(state.rho_gp),
        rt0_flux_vector_at_element_centroid=np.asarray(
            transport.last_solver_audit.get(
                "rt0_flux_vector_at_element_centroid", np.empty((0, 2)))),
    )
    selected = [item for row in history for item in row["selected_events"] if item]
    direct_summary = {}
    for site_index, site in enumerate(direct.sites):
        audits = [row["direct_opportunity"][site_index] for row in history]
        direct_summary[site.site_id] = {
            "unweighted_opportunity_integral": math.fsum(
                float(audit["support_integrated_interaction_rate_s"])*dt
                for audit, dt in zip(audits, STEP_DURATIONS_S)),
            "successful_action_integral": math.fsum(
                float(audit["successful_direct_interaction_rate_s"])*dt
                for audit, dt in zip(audits, STEP_DURATIONS_S)),
            "successful_outcome_fraction": float(
                direct.parameter_row.successful_direct_outcome_fraction),
            "terminal_thermal_success_probability": float(
                audits[-1]["thermal_success_probability"]),
            "support_radius_m": float(audits[-1]["support_radius_m"]),
            "support_area_m2": float(audits[-1]["support_area_m2"]),
            "support_volume_m3": float(audits[-1]["support_volume_m3"]),
            "opportunity_formula": audits[-1]["opportunity_formula"],
            "geometry_factor_C_dd": float(
                direct.parameter_row.interaction_geometry_factor_C_dd),
            "units_interpretation": (
                "Gamma_dd[m^-3 s^-1] integrated over unit-thickness support volume[m^3]"
            ),
        }
    if any(item == "DIRECTIONAL_CLEAVAGE" for item in selected): outcome = "DIRECT_CRACK_BEFORE_VOID"
    elif state.void_state.cavities: outcome = "VOID_BIRTH_BEFORE_CRACK"
    elif any(site.phase.value == "EMBRYO" for site in state.void_state.sites): outcome = "EMBRYO_PRESENT"
    elif all(site.birth_action == 0.0 for site in transport.site_states) and all(site.interaction_action == 0.0 for site in direct.sites): outcome = "NO_EVENT_WITHIN_WINDOW"
    else: outcome = "HAZARD_LIMITED"
    return finite({
        "variant": variant, "outcome": outcome,
        "configuration": {
            "vacancy_clustering_birth_enabled": state.void_state.embryo_growth_configuration.vacancy_clustering_birth_enabled,
            "direct_interaction_birth_enabled": state.void_state.embryo_growth_configuration.direct_interaction_birth_enabled,
            "cavity_surface_loop_emission_enabled": state.void_state.embryo_growth_configuration.cavity_surface_loop_emission_enabled,
            "direct_owner_enabled": direct.enabled,
            "interaction_population_partition_mobile_direct_benign": PARTITION,
            "bulk_initial_density_source": "rho_forest_floor_m2",
            "bulk_initial_density_m2": float(initial.rho_gp[0]),
            "initial_plastic_strain": "VIRGIN_ZERO",
            "plasticity_scope": PlasticityScope.FULL_FIELD_PT_EVOLVING_RHO.value,
        },
        "parameter_row": transport.parameter_row.__dict__,
        "equilibrium_vacancy_fraction": equilibrium_vacancy_fraction(TEMPERATURE_K, transport.parameter_row),
        "canonical_drive_check": canonical,
        "direct_channel_normalization": direct_summary,
        "initial_fingerprint": complete_accepted_state_fingerprint(initial),
        "final_fingerprint": complete_accepted_state_fingerprint(state),
        "restart_exact": complete_accepted_state_fingerprint(restored) == complete_accepted_state_fingerprint(state),
        "final_cavity_count": len(state.void_state.cavities),
        "prospective_second_site_retained": len(state.void_state.sites) >= 2,
        "history": history,
        "latest_restart": str(checkpoint.relative_to(ROOT)),
        "terminal_field_snapshot": str(field_path.relative_to(ROOT)),
    })


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=ROOT/"artifacts/active_two_mechanism_coupled_directional")
    args = parser.parse_args(); args.out.mkdir(parents=True, exist_ok=True)
    rows = [run_variant(name, args.out) for name in ("reference", "combined")]
    payload = finite({
        "schema": "unified-2d.active-two-mechanism-natural-void/2",
        "classification": "UNCALIBRATED_COMPARATIVE_PHYSICAL_STUDY",
        "protocol_selected_before_outcomes": True,
        "temperature_K": TEMPERATURE_K, "step_durations_s": STEP_DURATIONS_S,
        "total_duration_s": sum(STEP_DURATIONS_S), "physical_rate_scaling": 1.0,
        "rows": rows,
        "primary_trajectory_variants": ("reference", "combined"),
    })
    (args.out/"study.json").write_text(json.dumps(payload, indent=2, sort_keys=True)+"\n")
    print("ACTIVE_TWO_MECHANISM_NATURAL_VOID_STUDY = PASS")
    for row in rows:
        last = row["history"][-1]
        print(row["variant"], row["outcome"], row["final_cavity_count"],
              last["maximum_cumulative_equivalent_plastic_strain"],
              last["cumulative_plastic_work_J_per_m"], row["restart_exact"])


if __name__ == "__main__": main()
