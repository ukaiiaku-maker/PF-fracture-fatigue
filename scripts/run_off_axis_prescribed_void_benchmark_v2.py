#!/usr/bin/env python3
"""Certified finite-increment off-axis crack/void benchmark and scheduler gate."""
from __future__ import annotations

from dataclasses import replace
import argparse
from collections.abc import Mapping
import json
import math
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from arrhenius_fracture.adaptive_multitip_mesh_v11 import adapt_accepted_state_for_trials
from arrhenius_fracture.crack_network_v11 import ROOT_BRANCH_ID
from arrhenius_fracture.directional_competition_v11 import (
    DirectionalCompetitionState, tungsten_cleavage_candidates,
)
from arrhenius_fracture.global_physical_scheduler_v1 import advance_global_physical_interval
from arrhenius_fracture.embryo_growth_v1 import EmbryoGrowthConfiguration
from arrhenius_fracture.off_axis_void_steering import initial_steering_prediction
from arrhenius_fracture.topology_transaction_v11 import complete_accepted_state_fingerprint
from arrhenius_fracture.topology_transaction_v11 import equilibrate_fixed_load_with_production_fem
from arrhenius_fracture.unified_fracture_material_v5 import material_bundle
from arrhenius_fracture.voiding_production_v5 import (
    build_production_void_state, crack_tip_tensor, deterministic_trajectory,
    directional_sharp_front_rates, remesh_cavity, sharp_front_load_provider,
)
from arrhenius_fracture.voiding_v5 import Cavity2D, VoidPhase

TIP = (4.0e-4, 0.0)
ROOT_CRACK = (0.0, 0.0)
RADIUS = 5.5e-5
DELTA_A = 7.5e-5
KINETIC_DELTA_A = 2.0e-5
MAX_PHYSICAL_OBSERVATION_S = 1.0
CENTERS = {
    "above": (TIP[0]+3.0*RADIUS, 1.3*RADIUS),
    "below": (TIP[0]+3.0*RADIUS, -1.3*RADIUS),
}


def finite(value):
    if isinstance(value, Mapping): return {str(k): finite(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)): return [finite(v) for v in value]
    if isinstance(value, np.ndarray): return finite(value.tolist())
    if isinstance(value, np.generic): return finite(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return "Infinity" if value > 0 else "-Infinity"
    return value


def with_candidates(state):
    candidates = tungsten_cleavage_candidates(theta_deg=0.0, include_110=True)
    return replace(state, competition=DirectionalCompetitionState.initialize(
        candidates, global_hazard_seed=3621))


def _disable_lifecycle(state):
    direct = replace(state.void_state.dislocation_interaction_state, enabled=False)
    return replace(state, void_state=replace(
        state.void_state, dislocation_interaction_state=direct,
        embryo_growth_configuration=EmbryoGrowthConfiguration(
            vacancy_clustering_birth_enabled=False,
            direct_interaction_birth_enabled=False,
            cavity_surface_loop_emission_enabled=False)))


def initial_state(case, *, width=1.0e-3, height=1.0e-3, far_h=5.0e-5,
                  material_family="DBTT", load_scale=1.0,
                  cavity_centers=None, temperature_K=900.0):
    bundle = material_bundle(material_family)
    centers = CENTERS if cavity_centers is None else cavity_centers
    rho_bulk = float(bundle.fracture_row["rho_forest_floor_m2"])
    if case == "no_void":
        state, _ = build_production_void_state(
            bundle=bundle, enabled=True, crack_path_m=(ROOT_CRACK, TIP),
            cavity_center_m=(TIP[0]+3.0*RADIUS, 0.0), temperature_K=temperature_K,
            bulk_initial_density_m2=rho_bulk, initial_plastic_strain_gp=0.0,
            specimen_width_m=width, specimen_height_m=height, far_h_m=far_h)
        state = with_candidates(_disable_lifecycle(state))
        if float(load_scale) != 1.0:
            state = equilibrate_fixed_load_with_production_fem(replace(
                state, displacement=np.asarray(state.displacement)*float(load_scale)))
        return state
    state, hole = build_production_void_state(
        bundle=bundle, enabled=True,
        cavity_center_m=centers[case], crack_path_m=(ROOT_CRACK, TIP),
        cavity_geometry_radius_m=RADIUS, temperature_K=temperature_K,
        initial_vacancy_fraction_gp=None,
        bulk_initial_density_m2=rho_bulk, initial_plastic_strain_gp=0.0,
        specimen_width_m=width, specimen_height_m=height, far_h_m=far_h,
    )
    site = replace(state.void_state.sites[0], phase=VoidPhase.RESOLVED_VOID)
    area = math.pi*RADIUS**2
    cavity = Cavity2D(
        cavity_id="prescribed-hole:"+site.site_id,
        parent_site_id=site.site_id, center_m=centers[case], radius_m=RADIUS,
        area_m2=area, inventory_area_m2=area, phase=VoidPhase.RESOLVED_VOID,
        lineage=("PRESCRIBED_HOLE_MECHANICS_NOT_LIFECYCLE_BIRTH",),
    )
    prescribed = replace(
        state.void_state, sites=(site,), cavities=(cavity,),
        event_history=state.void_state.event_history+({
            "event": "PRESCRIBED_HOLE_MECHANICS_INITIALIZATION",
            "inventory_derived": False,
        },),
    )
    state = remesh_cavity(
        state, hole, prescribed, "prescribed-hole-mechanics-initialization")
    state = with_candidates(_disable_lifecycle(state))
    if float(load_scale) != 1.0:
        state = equilibrate_fixed_load_with_production_fem(replace(
            state, displacement=np.asarray(state.displacement)*float(load_scale)))
    return state


def solve_case(case, *, width=1.0e-3, height=1.0e-3, far_h=5.0e-5,
               run_scheduler=True, include_finite_diagnostic=True,
               material_family="DBTT", load_scale=1.0, cavity_centers=None,
               temperature_K=900.0, maximum_observation_s=MAX_PHYSICAL_OBSERVATION_S,
               observation_crossing_multiples=4.0, maximum_events=8,
               return_state=False):
    centers = CENTERS if cavity_centers is None else cavity_centers
    state = initial_state(
        case, width=width, height=height, far_h=far_h,
        material_family=material_family, load_scale=load_scale,
        cavity_centers=centers, temperature_K=temperature_K)
    candidates = state.competition.candidates
    mapping = {ROOT_BRANCH_ID: candidates}
    common, adaptation = adapt_accepted_state_for_trials(
        state, mapping, da_phys_m=DELTA_A, tip_h_fine_m=1.0e-6,
        contour_radius_m=5.0e-5, crack_band_radius_m=5.0e-7,
        accepted_load_m=4.0e-7*float(load_scale), maximum_levels=10,
    )
    finite_provider = None
    if include_finite_diagnostic:
        finite_provider = sharp_front_load_provider(
            common, branch_id=ROOT_BRANCH_ID, candidates=candidates,
            evaluate_overlap_marginals=True, marginal_delta_a_m=(DELTA_A,),
            marginal_mesh_levels=(4, 5), finite_increment_mode=True,
        )
    kinetic_provider = sharp_front_load_provider(
        common, branch_id=ROOT_BRANCH_ID, candidates=candidates,
        evaluate_overlap_marginals=True, marginal_delta_a_m=(KINETIC_DELTA_A,),
        marginal_mesh_levels=(4, 5), finite_increment_mode=True,
    )
    by_id = {candidate.candidate_id: candidate for candidate in candidates}
    def curve_from(provider, physical_increment):
        curve = []
        for row in provider["directional"]:
            candidate = by_id[row["candidate_id"]]
            certified = [trial for trial in row["marginal_trial_rows"]
                         if trial["status"] == "CERTIFIED_EXACT_FIXED_VOID_MARGINAL"]
            finest = max(certified, key=lambda item: item["local_mesh_level"], default=None)
            raw_G = None if finest is None else float(finest["G_marginal_J_per_m2"])
            curve.append(finite({
                "candidate_id": candidate.candidate_id,
                "plane_variant": candidate.plane_variant,
                "angle_deg": math.degrees(candidate.angle_rad),
                "gamma_rel": candidate.gamma_rel,
                "physical_increment_m": physical_increment,
                "all_trial_supports_certified": len(certified) == 2,
                "candidate_support_executable": bool(certified),
                "finite_increment_G_J_per_m2": raw_G,
                "G_over_gamma_J_per_m2": None if raw_G is None else raw_G/candidate.gamma_rel,
                "mesh_converged": row["marginal_convergence_passed"],
                "mesh_relative_error": row["marginal_mesh_relative_errors_by_delta"],
                "drive_source": row["drive_source"],
                "directional_drive_status": row["directional_drive_status"],
                "trial_rows": row["marginal_trial_rows"],
            }))
        return curve
    curve = ([] if finite_provider is None
             else curve_from(finite_provider, DELTA_A))
    kinetic_curve = curve_from(kinetic_provider, KINETIC_DELTA_A)
    rates = directional_sharp_front_rates(
        common, kinetic_provider["directional"], branch_id=ROOT_BRANCH_ID,
        temperature_K=temperature_K)
    available = [row for row in rates if row["directional_drive_status"] == "AVAILABLE"]
    tensor, tensor_support = crack_tip_tensor(common, branch_id=ROOT_BRANCH_ID)
    finite_crossings = [float(row["crossing_time_s"]) for row in available
                        if math.isfinite(float(row["crossing_time_s"]))]
    first_crossing = min(finite_crossings, default=math.inf)
    observation_window = (
        0.0 if not math.isfinite(first_crossing)
        else min(float(observation_crossing_multiples)*first_crossing,
                 float(maximum_observation_s))
    )
    row_cache = {
        complete_accepted_state_fingerprint(common): kinetic_provider["directional"]
    }
    def state_bound_rows(mechanical, active_tip_id):
        if active_tip_id is None:
            return None
        fingerprint = complete_accepted_state_fingerprint(mechanical)
        if fingerprint not in row_cache:
            row_cache[fingerprint] = sharp_front_load_provider(
                mechanical, branch_id=active_tip_id,
                candidates=mechanical.competition.candidates,
                evaluate_overlap_marginals=True,
                marginal_delta_a_m=(KINETIC_DELTA_A,),
                marginal_mesh_levels=(4, 5), finite_increment_mode=True,
            )["directional"]
        return row_cache[fingerprint]
    if run_scheduler and observation_window > 0.0:
        scheduled, audit = advance_global_physical_interval(
            common, common, observation_window,
            {"kind": "hold", "temperature_K": temperature_K,
             "directional_load_rows_for_state": state_bound_rows,
             "maximum_events": int(maximum_events)},
        )
    else:
        scheduled, audit = common, {"events": (), "all_owners_previewed": True}
    selected = [event["selected_event"] for event in audit["events"] if event["selected_event"]]
    result = finite({
        "case": case, "cavity_center_m": None if case == "no_void" else centers[case],
        "material_family": material_family, "temperature_K": temperature_K,
        "load_scale": load_scale,
        "prescribed_void_mechanics_benchmark": case != "no_void",
        "physical_increment_m": DELTA_A,
        "canonical_kinetic_increment_m": KINETIC_DELTA_A,
        "specimen_width_m": width, "specimen_height_m": height,
        "common_parent_fingerprint": complete_accepted_state_fingerprint(common),
        "common_parent_element_count": common.mesh.ne,
        "common_parent_refinement_levels": len(adaptation.lineages),
        "directional_curve": curve,
        "canonical_kinetic_directional_curve": kinetic_curve,
        "canonical_kinetic_rates": rates,
        "all_three_candidate_supports_executable": all(row["candidate_support_executable"] for row in curve),
        "all_three_directional_energies_mesh_converged": all(row["mesh_converged"] for row in curve),
        "all_three_canonical_kinetic_energies_mesh_converged": all(
            row["mesh_converged"] for row in kinetic_curve),
        "asymptotic_prediction": None if case == "no_void" else initial_steering_prediction(
            TIP[0], cavity_center_m=centers[case], cavity_radius_m=RADIUS,
            poisson_ratio=0.3),
        "scheduler": {
            "actual_global_scheduler_exercised": bool(run_scheduler and observation_window > 0.0),
            "eligible_owner_registry": [p for event in audit["events"] for p in event["eligible_previews"]],
            "all_owners_previewed": audit["all_owners_previewed"],
            "first_qualified_crossing_time_s": first_crossing,
            "finite_observation_window_s": observation_window,
            "state_bound_directional_observation_count": len(row_cache),
            "selected_events": selected,
            "path_or_stopping_condition": (
                "EVENT_COMMITTED" if selected else
                "SCHEDULER_NOT_REQUESTED" if not run_scheduler else
                "FINITE_WINDOW_NO_EVENT_CANONICAL_DRIVE_FAIL_CLOSED"
                if not available else
                "QUALIFIED_DRIVE_HAZARD_LIMITED_WITHIN_PHYSICAL_WINDOW"
            ),
            "terminal_crack_paths_m": [branch.path for branch in scheduled.crack_network.branches],
        },
        "adjacent_tip_tensor_Pa": tensor,
        "adjacent_tip_support_elements": tensor_support,
    })
    return (result, scheduled) if return_state else result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path,
                        default=ROOT/"artifacts/off_axis_prescribed_void_benchmark_v2")
    args = parser.parse_args(); args.out.mkdir(parents=True, exist_ok=True)
    cases = [solve_case(name) for name in ("no_void", "above", "below")]
    no_void = {row["plane_variant"]: row for row in cases[0]["directional_curve"]}
    for case in cases[1:]:
        for row in case["directional_curve"]:
            base = no_void[row["plane_variant"]]["finite_increment_G_J_per_m2"]
            value = row["finite_increment_G_J_per_m2"]
            row["normalized_void_induced_G_change"] = (
                None if base is None or value is None else (value-base)/max(abs(base), 1e-300)
            )
        kinetic_no_void = {
            row["plane_variant"]: row
            for row in cases[0]["canonical_kinetic_directional_curve"]
        }
        for row in case["canonical_kinetic_directional_curve"]:
            base = kinetic_no_void[row["plane_variant"]]["finite_increment_G_J_per_m2"]
            value = row["finite_increment_G_J_per_m2"]
            row["normalized_void_induced_G_change"] = (
                None if base is None or value is None
                else (value-base)/max(abs(base), 1e-300)
            )
    mechanics_ready = all(
        case["all_three_directional_energies_mesh_converged"]
        and case["all_three_canonical_kinetic_energies_mesh_converged"]
        for case in cases)
    outer_cases = []
    outer_comparisons = []
    if mechanics_ready:
        outer_cases = [solve_case(
            name, width=1.5e-3, height=1.5e-3, far_h=7.5e-5,
            run_scheduler=False, include_finite_diagnostic=False,
        ) for name in ("no_void", "above", "below")]
        outer_no_void = {
            row["plane_variant"]: row
            for row in outer_cases[0]["canonical_kinetic_directional_curve"]
        }
        central_no_void = {
            row["plane_variant"]: row
            for row in cases[0]["canonical_kinetic_directional_curve"]
        }
        for central, outer in zip(cases, outer_cases):
            central_rows = {row["plane_variant"]: row
                            for row in central["canonical_kinetic_directional_curve"]}
            for row in outer["canonical_kinetic_directional_curve"]:
                variant = row["plane_variant"]
                value = row["finite_increment_G_J_per_m2"]
                base = outer_no_void[variant]["finite_increment_G_J_per_m2"]
                central_value = central_rows[variant]["finite_increment_G_J_per_m2"]
                central_base = central_no_void[variant]["finite_increment_G_J_per_m2"]
                row["domain_matched_void_delta_G_J_per_m2"] = (
                    None if value is None or base is None else value-base)
                row["central_domain_void_delta_G_J_per_m2"] = (
                    None if central_value is None or central_base is None
                    else central_value-central_base)
                outer_delta = row["domain_matched_void_delta_G_J_per_m2"]
                central_delta = row["central_domain_void_delta_G_J_per_m2"]
                relative = (
                    None if outer_delta is None or central_delta is None else
                    abs(outer_delta-central_delta)/max(
                        abs(outer_delta), abs(central_delta), 1.0e-300)
                )
                row["outer_boundary_void_delta_G_relative_change"] = relative
                if outer["case"] != "no_void" and relative is not None:
                    outer_comparisons.append(relative)
    outer_limit = 0.10
    outer_passed = bool(
        mechanics_ready and outer_comparisons
        and max(outer_comparisons) <= outer_limit
    )
    payload = finite({
        "schema": "unified-2d.off-axis-prescribed-void-benchmark/2",
        "classification": "PRESCRIBED_VOID_MECHANICS_SEPARATE_FROM_NATURAL_VOIDING",
        "certificate_relaxed": False,
        "junction_local_scale_correction": "LOCAL_VERTEX_INCIDENT_EXACT_SUPPORT",
        "cases": cases,
        "outer_boundary_cases": outer_cases,
        "outer_boundary_void_delta_G_relative_limit": outer_limit,
        "outer_boundary_max_void_delta_G_relative_change": (
            None if not outer_comparisons else max(outer_comparisons)),
        "outer_boundary_status": (
            "PENDING_DIRECTIONAL_MESH_CONVERGENCE" if not mechanics_ready
            else "PASS" if outer_passed else
            "COMPLETED_SENSITIVITY_NOT_CONVERGED"
        ),
    })
    (args.out/"benchmark.json").write_text(json.dumps(payload, indent=2, sort_keys=True)+"\n")
    print("OFF_AXIS_CERTIFIED_CANDIDATE_SUPPORT = "+(
        "PASS" if all(case["all_three_candidate_supports_executable"] for case in cases) else "FAIL"))
    print("OFF_AXIS_FINITE_INCREMENT_ENERGY_CONVERGENCE = "+(
        "PASS" if all(case["all_three_directional_energies_mesh_converged"] for case in cases) else "PENDING"))
    for case in cases:
        print(case["case"], case["scheduler"]["path_or_stopping_condition"])


if __name__ == "__main__": main()
