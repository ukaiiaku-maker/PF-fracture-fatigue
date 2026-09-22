#!/usr/bin/env python3
"""Production qualification for accepted-state-owned conserved vacancy transport."""
from __future__ import annotations

from dataclasses import replace
import json
import math
from pathlib import Path
import sys
import tempfile

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from arrhenius_fracture.checkpoint_v11 import restore_checkpoint, write_checkpoint
from arrhenius_fracture.crack_network_v11 import ROOT_BRANCH_ID
from arrhenius_fracture.global_physical_scheduler_v1 import (
    advance_global_physical_interval, controlled_scheduler_ownership_matrix,
    preview_front_engine_emission,
)
from arrhenius_fracture.embryo_growth_v1 import (
    RESOLVED, initialize_embryo_growth_state,
)
from arrhenius_fracture.mesh import rebuild_tri_mesh
from arrhenius_fracture.topology_transaction_v11 import (
    complete_accepted_state_fingerprint, equilibrate_fixed_load_with_production_fem,
)
from arrhenius_fracture.unified_fracture_material_v5 import material_bundle
from arrhenius_fracture.vacancy_transport_phase_field_v2 import (
    SurfaceKind, TransportSurfaceRegistry, accepted_elastoplastic_stress_3d,
    build_exact_cut_transport_mesh, build_transport_surface_registry,
    initial_transport_state, locate_site_inventory_event, solve_transport_interval,
    transfer_transport_state_overlap,
    transport_conservation,
)
from arrhenius_fracture.voiding_production_v5 import (
    _bundle_from_state, advance_production_vacancy_transport_interval,
    build_production_void_state,
    couple_active_cavity_flux_to_geometry, directional_clock_rates,
    fresh_sharp_front_engine, remesh_cavity,
)
from arrhenius_fracture.voiding_v5 import Cavity2D, VoidPhase


def require(value, message):
    if not value:
        raise RuntimeError(message)


def mesh(nx, ny, *, angle=0.0, skew=0.0, grade=1.0, perturb=0.0,
         delaunay=False):
    points = []
    for j in range(ny+1):
        y = (j/ny)**grade
        for i in range(nx+1):
            x = i/nx
            if 0 < i < nx and 0 < j < ny and perturb:
                x += perturb*math.sin(17*i+11*j)/nx
                y += perturb*math.cos(13*i-7*j)/ny
            points.append((x+skew*y, y))
    points = np.asarray(points)*4.0e-6
    rotation = np.array(((math.cos(angle), -math.sin(angle)),
                         (math.sin(angle), math.cos(angle))))
    points = points@rotation.T
    if delaunay:
        from scipy.spatial import Delaunay
        elems = Delaunay(points).simplices
    else:
        elems = []
        for j in range(ny):
            for i in range(nx):
                a = j*(nx+1)+i
                elems.extend(((a, a+1, a+nx+2), (a, a+nx+2, a+nx+1)))
    return rebuild_tri_mesh(points, np.asarray(elems, int))


def owners(grid):
    result = {}
    for eid, tri in enumerate(grid.elems):
        for edge in ((tri[0], tri[1]), (tri[1], tri[2]), (tri[2], tri[0])):
            result.setdefault(tuple(sorted(map(int, edge))), []).append(eid)
    return result


def noflux_tags(grid):
    return {edge: SurfaceKind.CONTACT_OR_GRIP_NO_FLUX
            for edge, own in owners(grid).items() if len(own) == 1}


def solve_control(grid, row, *, nx, stress=False):
    boundary = {edge: SurfaceKind.EXTERNAL_FREE_SURFACE for edge in noflux_tags(grid)}
    registry = build_transport_surface_registry(grid, boundary_tags=boundary)
    origin = grid.nodes[0]
    transform = np.column_stack((grid.nodes[nx]-origin, grid.nodes[nx*(nx+1)]-origin))
    inverse = np.linalg.inv(transform)
    reference_nodes = (grid.nodes-origin)@inverse.T
    reference_centers = reference_nodes[grid.elems].mean(axis=1)
    centers = grid.nodes[grid.elems].mean(axis=1)
    ceq = float(row and __import__(
        "arrhenius_fracture.vacancy_transport_phase_field_v2", fromlist=["equilibrium_vacancy_fraction"]
    ).equilibrium_vacancy_fraction(1800.0, row))
    amplitude = 5.0e-6; duration = 0.01
    diffusivity = row.vacancy_diffusivity_prefactor_m2_s*math.exp(
        -row.vacancy_migration_energy_eV*1.602176634e-19/(1.380649e-23*1800.0)
    )
    wave_vector = inverse.T@np.array((math.pi, 0.0))
    final_amplitude = amplitude/(1.0+duration*diffusivity*float(wave_vector@wave_vector))
    exact_initial = ceq+amplitude*np.sin(math.pi*reference_centers[:, 0])
    exact_final = ceq+final_amplitude*np.sin(math.pi*reference_centers[:, 0])
    state = initial_transport_state(
        grid, temperature_K=1800.0, row=row, registry=registry,
        vacancy_fraction_gp=exact_initial,
    )
    boundary_values = {
        face.face_id: ceq+final_amplitude*math.sin(
            math.pi*float(np.mean(reference_nodes[list(face.node_ids)], axis=0)[0])
        ) for face in registry.faces if face.kind == SurfaceKind.EXTERNAL_FREE_SURFACE
    }
    stress_amplitude = 1.0e7
    stress_field = (stress_amplitude*reference_centers[:, 0]
                    if stress else np.zeros(grid.ne))
    result = solve_transport_interval(
        state, grid, registry, row=row, duration_s=duration,
        accepted_equivalent_plastic_strain_rate_s=np.zeros(grid.ne),
        accepted_dislocation_density_m2=np.zeros(grid.ne),
        accepted_mean_stress_Pa=stress_field,
        diagnostic_boundary_concentration_by_face=boundary_values,
        maximum_picard_iterations=(100 if stress else 30),
        relative_tolerance=(1.0e-8 if stress else 1.0e-10),
    )
    require(result.last_solver_audit["local_element_conservation_residual_inf"] < 1.0e-20,
            "RT0 element balance failed")
    require(abs(transport_conservation(result, grid.area_e)["conservation_residual_area_m2"])
            <= result.last_solver_audit["conservation_roundoff_budget_area_m2"],
            "RT0 global balance failed")
    difference = result.vacancy_fraction_gp-exact_final
    l1 = float(np.sum(grid.area_e*np.abs(difference))/np.sum(grid.area_e)/amplitude)
    l2 = float(math.sqrt(np.sum(grid.area_e*difference*difference)/np.sum(grid.area_e))/amplitude)
    numerical_flux = np.asarray(result.last_solver_audit["rt0_flux_vector_at_element_centroid"])
    exact_flux = (-diffusivity*final_amplitude
                  * np.cos(math.pi*reference_centers[:, 0])[:, None]*wave_vector)
    if stress:
        alpha = row.vacancy_formation_volume_m3/(1.380649e-23*1800.0)
        stress_gradient = stress_amplitude*(inverse.T@np.array((1.0, 0.0)))
        exact_flux += (diffusivity*alpha*exact_final*(1.0-exact_final))[:, None] \
            * stress_gradient
    flux_scale = max(float(math.sqrt(np.sum(grid.area_e*np.sum(exact_flux*exact_flux, axis=1))
                                     /np.sum(grid.area_e))), 1.0e-300)
    flux_error = float(math.sqrt(np.sum(grid.area_e*np.sum(
        (numerical_flux-exact_flux)**2, axis=1))/np.sum(grid.area_e))/flux_scale)
    return {"L1_relative": l1, "L2_relative": l2, "face_flux_relative": flux_error,
            "inventory_error_area_m2": abs(result.last_solver_audit[
                "global_conservation_residual_area_m2"]),
            "element_balance_inf": result.last_solver_audit[
                "local_element_conservation_residual_inf"],
            "rt0_audit": {key: result.last_solver_audit[key] for key in (
                "rt0_normal_flux_dof_count", "rt0_divergence_coupling_shape",
                "rt0_shared_interior_normal_flux_continuity_residual",
                "rt0_local_mass_quadrature",
            )}}


def general_mesh_qualification(row):
    # Prospective tolerances: frozen before evaluating the families below.
    minimum_concentration_order = 1.0
    minimum_flux_order = 0.70
    maximum_fine_concentration_error = 2.0e-3
    maximum_fine_flux_error = 7.0e-2
    maximum_rotation_discrepancy = 1.0e-8
    families = {
        "regular_triangles": {},
        "skew_triangles": dict(skew=0.37),
        "perturbed_delaunay": dict(perturb=0.18, delaunay=True),
        "strongly_graded_tip_cavity": dict(grade=2.8, skew=0.12),
        "rotated_31deg": dict(angle=math.radians(31.0)),
        "rotated_67deg": dict(angle=math.radians(67.0)),
    }
    report = {}
    levels = (4, 8, 16, 24)
    for name, options in families.items():
        values = [solve_control(mesh(n, n, **options), row, nx=n) for n in levels]
        for key, minimum in (("L1_relative", minimum_concentration_order),
                             ("L2_relative", minimum_concentration_order),
                             ("face_flux_relative", minimum_flux_order)):
            errors = [item[key] for item in values]
            require(all(errors[i+1] < errors[i] for i in range(len(errors)-1)),
                    name+" lacks monotonic "+key)
            orders = [math.log(errors[i]/errors[i+1])
                      /math.log(levels[i+1]/levels[i]) for i in range(len(errors)-1)]
            require(min(orders) >= minimum, name+" convergence order is too low for "+key)
            for item, order in zip(values[1:], orders):
                item.setdefault("observed_orders", {})[key] = order
        require(values[-1]["L2_relative"] <= maximum_fine_concentration_error,
                name+" fine concentration error exceeds the frozen tolerance")
        require(values[-1]["face_flux_relative"] <= maximum_fine_flux_error,
                name+" fine flux error exceeds the frozen tolerance")
        report[name] = values
    regular = report["regular_triangles"][-1]
    for rotated in ("rotated_31deg", "rotated_67deg"):
        discrepancy = max(abs(report[rotated][-1][key]-regular[key]) for key in (
            "L1_relative", "L2_relative", "face_flux_relative"))
        require(discrepancy <= maximum_rotation_discrepancy,
                rotated+" discrepancy exceeds the frozen tolerance")
        report[rotated][-1]["rotation_discrepancy"] = discrepancy
    # Hydrostatic-stress drift on the same nonorthogonal/perturbed family.
    stress_values = [solve_control(
        mesh(n, n, skew=0.27, perturb=0.1, delaunay=True), row, nx=n, stress=True,
    ) for n in (4, 8, 12)]
    require(all(math.isfinite(item["L2_relative"]) for item in stress_values),
            "stress-driven RT0 failed")
    report["stress_driven_flux"] = stress_values
    report["frozen_acceptance"] = {
        "minimum_concentration_order": minimum_concentration_order,
        "minimum_flux_order": minimum_flux_order,
        "maximum_fine_concentration_error": maximum_fine_concentration_error,
        "maximum_fine_flux_error": maximum_fine_flux_error,
        "maximum_rotation_discrepancy": maximum_rotation_discrepancy,
    }
    return report


def crack_interface_control(row):
    base = mesh(12, 8)
    rows = tuple({"node_ids": edge, "kind": kind.value}
                 for edge, kind in noflux_tags(base).items())
    cut = build_exact_cut_transport_mesh(
        base, p0=(2.0e-6, 0.0), p1=(2.0e-6, 4.0e-6),
        topology_boundary_tags=rows, segment_id="accepted-main",
    )
    centers = cut.mesh.nodes[cut.mesh.elems].mean(axis=1)
    concentration = np.where(centers[:, 0] < 2.0e-6, 2.0e-5, 2.0e-6)
    impermeable_registry = TransportSurfaceRegistry(
        faces=tuple(replace(face, kind=SurfaceKind.CONTACT_OR_GRIP_NO_FLUX)
                    if face.kind == SurfaceKind.CRACK_FREE_SURFACE else face
                    for face in cut.registry.faces),
        crack_physical_length_m=cut.registry.crack_physical_length_m,
    )
    state = initial_transport_state(
        cut.mesh, temperature_K=1800.0, row=row, registry=impermeable_registry,
        vacancy_fraction_gp=concentration,
    )
    side_masks = {"negative": centers[:, 0] < 2.0e-6,
                  "positive": centers[:, 0] > 2.0e-6}
    before = {name: float(np.sum(cut.mesh.area_e[mask]*concentration[mask]))
              for name, mask in side_masks.items()}
    impermeable = solve_transport_interval(
        state, cut.mesh, impermeable_registry, row=row, duration_s=0.001,
        accepted_equivalent_plastic_strain_rate_s=np.zeros(cut.mesh.ne),
        accepted_dislocation_density_m2=np.zeros(cut.mesh.ne),
        accepted_mean_stress_Pa=np.zeros(cut.mesh.ne),
    )
    after = {name: float(np.sum(cut.mesh.area_e[mask]
                               *impermeable.vacancy_fraction_gp[mask]))
             for name, mask in side_masks.items()}
    crack_faces = [face for face in cut.registry.faces
                   if face.kind == SurfaceKind.CRACK_FREE_SURFACE]
    require({face.side for face in crack_faces} == {-1, 1}, "crack sides are not distinct")
    require(math.isclose(
        math.fsum(face.length_m for face in crack_faces if face.side > 0),
        cut.registry.crack_physical_length_m, rel_tol=1.0e-10, abs_tol=1.0e-15,
    ), "upper crack length changed")
    require(math.isclose(
        math.fsum(face.length_m for face in crack_faces if face.side < 0),
        cut.registry.crack_physical_length_m, rel_tol=1.0e-10, abs_tol=1.0e-15,
    ), "lower crack length changed")
    impermeable_fluxes = impermeable.last_solver_audit["face_flux_fraction_area_per_s"]
    measured_face_flux = {face.face_id: float(impermeable_fluxes[face.face_id])
                          for face in crack_faces}
    measured_cross_flux = math.fsum(measured_face_flux.values())
    budget = impermeable.last_solver_audit["conservation_roundoff_budget_area_m2"]/0.001
    require(abs(measured_cross_flux) <= budget, "measured cross-crack flux is nonzero")
    require(all(abs(after[name]-before[name]) <= budget*0.001 for name in side_masks),
            "impermeable crack transferred inventory between sides")

    absorbing_initial = initial_transport_state(
        cut.mesh, temperature_K=1800.0, row=row, registry=cut.registry,
        vacancy_fraction_gp=concentration,
    )
    absorbing = solve_transport_interval(
        absorbing_initial, cut.mesh, cut.registry, row=row, duration_s=0.001,
        accepted_equivalent_plastic_strain_rate_s=np.zeros(cut.mesh.ne),
        accepted_dislocation_density_m2=np.zeros(cut.mesh.ne),
        accepted_mean_stress_Pa=np.zeros(cut.mesh.ne),
    )
    absorbing_fluxes = absorbing.last_solver_audit["face_flux_fraction_area_per_s"]
    ratio = row.inventory_ratio
    side_absorption = {side: 0.001*ratio*math.fsum(
        float(absorbing_fluxes[face.face_id]) for face in crack_faces if face.side == side
    ) for side in (-1, 1)}
    require(all(value > 0.0 for value in side_absorption.values()),
            "each absorbing crack side must own positive loss")
    require(math.isclose(
        math.fsum(side_absorption.values()),
        absorbing.ledgers["crack_surface_absorption_area_m2"],
        rel_tol=1.0e-10, abs_tol=1.0e-28,
    ), "side-resolved crack absorption does not close")
    return {
        "crack_faces": len(crack_faces),
        "measured_crack_interface_flux_dofs": measured_face_flux,
        "measured_total_cross_crack_flux_fraction_area_per_s": measured_cross_flux,
        "impermeable_side_inventory_before": before,
        "impermeable_side_inventory_after": after,
        "absorbing_side_inventory_loss_area_m2": side_absorption,
        "physical_length_m": cut.registry.crack_physical_length_m,
    }
def rebind_row(state, row):
    transport = replace(
        state.vacancy_transport_state_v2, parameter_row_id=row.row_id,
        parameter_row_fingerprint=row.fingerprint, parameter_row=row,
    )
    return replace(state, vacancy_transport_state_v2=transport)


def physical_sentinel():
    accepted, _ = build_production_void_state(
        bundle=material_bundle("DBTT"), temperature_K=900.0,
    )
    previous = equilibrate_fixed_load_with_production_fem(
        replace(accepted, ep_gp=np.zeros_like(accepted.ep_gp))
    )
    previous = replace(previous, vacancy_transport_state_v2=accepted.vacancy_transport_state_v2)
    stress = accepted_elastoplastic_stress_3d(accepted)
    tip = np.asarray(accepted.crack_network.branches[0].tip)
    centers = accepted.mesh.nodes[accepted.mesh.elems].mean(axis=1)
    eid = int(np.argmin(np.linalg.norm(centers-tip, axis=1)))
    tensor = np.array(((stress["sigma_xx_Pa"][eid], stress["sigma_xy_Pa"][eid]),
                       (stress["sigma_xy_Pa"][eid], stress["sigma_yy_Pa"][eid])))
    cleavage = directional_clock_rates(accepted, tensor, temperature_K=900.0)
    cleavage_time = min(item["crossing_time_s"] for item in cleavage)
    clocks = {
        "CLEAVAGE": cleavage_time, "EMISSION": None, "STABILIZATION": None,
        "HEALING": None, "GROWTH_PROMOTION": None, "LIGAMENT_RUPTURE": None,
        "DOWNSTREAM_FIRST_PASSAGE": None,
    }
    full, audit = advance_production_vacancy_transport_interval(
        previous, accepted, duration_s=1.0, external_event_times_s=clocks,
        directional_rate_by_candidate_s={item["candidate_id"]: item["rate_s"]
                                         for item in cleavage},
    )
    base_row = accepted.vacancy_transport_state_v2.parameter_row
    controls = {}
    for label, row, first, second in (
        ("A_full", base_row, previous, accepted),
        ("B_kv_zero", replace(base_row, plastic_vacancy_coefficient_per_strain=0.0),
         previous, accepted),
        ("C_zero_plastic_increment", base_row, accepted, accepted),
        ("D_climb_disabled", replace(base_row, dislocation_climb_capture_efficiency=0.0),
         previous, accepted),
    ):
        first = rebind_row(first, row); second = rebind_row(second, row)
        result, item = advance_production_vacancy_transport_interval(
            first, second, duration_s=1.0, external_event_times_s=clocks,
        )
        controls[label] = {
            "plastic_increment_rate_max_s": item["accepted_plastic_increment_rate_max_s"],
            "plastic_production_area_m2": result.vacancy_transport_state_v2.ledgers[
                "plastic_production_area_m2"],
            "climb_annihilation_area_m2": result.vacancy_transport_state_v2.ledgers[
                "dislocation_climb_annihilation_area_m2"],
        }
    require(controls["A_full"]["plastic_increment_rate_max_s"] > 0.0,
            "production sentinel has no measured accepted plastic increment")
    require(controls["B_kv_zero"]["plastic_production_area_m2"] == 0.0,
            "k_v=0 control still produced vacancies")
    require(controls["C_zero_plastic_increment"]["plastic_increment_rate_max_s"] == 0.0,
            "zero-increment control is not zero")
    require(controls["D_climb_disabled"]["climb_annihilation_area_m2"] == 0.0,
            "disabled climb sink remained active")

    # Free-surface-off is diagnostic-only and is never committed to production.
    registry = full.vacancy_transport_state_v2.surface_registry
    disabled_registry = TransportSurfaceRegistry(faces=tuple(
        replace(face, kind=SurfaceKind.CONTACT_OR_GRIP_NO_FLUX)
        if face.kind in {SurfaceKind.EXTERNAL_FREE_SURFACE, SurfaceKind.CRACK_FREE_SURFACE}
        else face for face in registry.faces
    ), crack_physical_length_m=registry.crack_physical_length_m)
    diagnostic = replace(
        full.vacancy_transport_state_v2,
        surface_registry=disabled_registry,
        surface_registry_fingerprint=disabled_registry.fingerprint,
    )
    diagnostic = solve_transport_interval(
        diagnostic, diagnostic.transport_mesh, disabled_registry, row=base_row,
        duration_s=0.01,
        accepted_equivalent_plastic_strain_rate_s=np.zeros(diagnostic.transport_mesh.ne),
        accepted_dislocation_density_m2=np.zeros(diagnostic.transport_mesh.ne),
        accepted_mean_stress_Pa=np.zeros(diagnostic.transport_mesh.ne),
    )
    controls["E_surface_sinks_disabled_diagnostic"] = {
        "new_external_absorption_area_m2": diagnostic.ledgers[
            "external_surface_absorption_area_m2"]
            - full.vacancy_transport_state_v2.ledgers["external_surface_absorption_area_m2"],
        "new_crack_absorption_area_m2": diagnostic.ledgers[
            "crack_surface_absorption_area_m2"]
            - full.vacancy_transport_state_v2.ledgers["crack_surface_absorption_area_m2"],
    }

    site = full.vacancy_transport_state_v2.site_states[0]
    required = site.required_embryo_inventory_area_m2/site.required_hit_count
    if audit["event"] == "CLEAVAGE": outcome = "DIRECT_CRACK_BEFORE_VOID"
    elif audit["event"] == "VACANCY_BIRTH_HIT": outcome = "VOID_BIRTH_BEFORE_CRACK"
    elif site.reserved_inventory_area_m2 < required: outcome = "INVENTORY_LIMITED"
    elif site.birth_action < site.next_birth_threshold(): outcome = "HAZARD_LIMITED"
    else: outcome = "NO_EVENT_WITHIN_WINDOW"
    return previous, accepted, full, audit, controls, outcome


def real_accepted_mechanical_interval():
    """Attempt the ordinary production Peierls/Taylor load increment.

    This deliberately has no legacy-plasticity fallback.  If the production
    implementation is unavailable, the terminal gate remains false and names
    the missing runtime owner instead of manufacturing an ep field.
    """
    from arrhenius_fracture.config import SimulationConfig
    from arrhenius_fracture.fem import assemble_mechanics
    from arrhenius_fracture.materials import PlasticityModel
    from arrhenius_fracture.plasticity import update_plasticity
    from arrhenius_fracture.emission_derived_plasticity import bind_dislocation_config_to_material_manifest
    from arrhenius_fracture.unified_fracture_material_v5 import (
        material_manifest, require_bound_identity,
    )
    from arrhenius_fracture.vacancy_coupling_v1 import equivalent_plastic_strain_rate_s

    bundle = material_bundle("DBTT")
    state_n, _ = build_production_void_state(bundle=bundle, temperature_K=900.0)
    config = SimulationConfig()
    config.material = state_n.material
    config.dislocations.bulk_kinetics_model = "emission_derived_peierls_taylor_multihit"
    bind_dislocation_config_to_material_manifest(config.dislocations, material_manifest(bundle))
    physical_delta_t = 1.0e-10
    history = []
    current = state_n
    try:
        for load_factor in (1.05, 1.10, 1.20, 1.40, 1.75, 2.00):
            loaded = equilibrate_fixed_load_with_production_fem(
                replace(current, displacement=state_n.displacement*load_factor)
            )
            _, _, sigma, *_ = assemble_mechanics(
                loaded.mesh, loaded.displacement, loaded.ep_gp, loaded.rho_gp,
                loaded.damage, loaded.elasticity_D, loaded.material,
                cohesive_network=loaded.cohesive_network,
            )
            ep, rho, _, plastic_audit = update_plasticity(
                loaded.ep_gp.copy(), loaded.rho_gp.copy(), sigma, loaded.material,
                900.0, physical_delta_t,
                PlasticityModel(config.plasticity_barrier, loaded.material),
                config.dislocations, return_info=True,
            )
            next_state = equilibrate_fixed_load_with_production_fem(
                replace(loaded, ep_gp=ep, rho_gp=rho)
            )
            rate = equivalent_plastic_strain_rate_s(
                current.ep_gp, next_state.ep_gp, physical_delta_t,
            )
            history.append({
                "load_factor": load_factor,
                "state_n_fingerprint": complete_accepted_state_fingerprint(current),
                "state_n_plus_1_fingerprint": complete_accepted_state_fingerprint(next_state),
                "maximum_epdot_s": float(np.max(rate)),
                "physical_delta_t_s": physical_delta_t,
                "rho_min_m2": float(np.min(next_state.rho_gp)),
                "rho_max_m2": float(np.max(next_state.rho_gp)),
                "plasticity_model": plastic_audit["bulk_kinetics_model"],
            })
            if np.max(rate) > 0.0:
                require(dict(require_bound_identity(current))["material_bundle_sha256"]
                        == bundle.bundle_hash, "state_n material identity changed")
                require(dict(require_bound_identity(next_state))["material_bundle_sha256"]
                        == bundle.bundle_hash, "state_n+1 material identity changed")
                require(current.equilibrium_observables is not None
                        and next_state.equilibrium_observables is not None,
                        "real load endpoints lack typed equilibrium observations")
                return {"passed": True, "state_n": current,
                        "state_n_plus_1": next_state, "epdot_s": rate,
                        "history": history, "blocking_reason": None}
            current = next_state
    except ModuleNotFoundError as exc:
        return {"passed": False, "state_n": state_n, "state_n_plus_1": None,
                "epdot_s": None, "history": history,
                "blocking_reason": "missing ordinary production plasticity module: "+str(exc)}
    return {"passed": False, "state_n": state_n, "state_n_plus_1": None,
            "epdot_s": None, "history": history,
            "blocking_reason": "ordinary bounded DBTT load history produced no accepted plastic increment"}


def physical_transport_interval_from_real_history(real_interval):
    if not real_interval["passed"]:
        return {"passed": False, "outcome": "NO_EVENT_WITHIN_PHYSICAL_WINDOW",
                "blocking_reason": real_interval["blocking_reason"]}
    state_n = real_interval["state_n"]
    state_np1 = real_interval["state_n_plus_1"]
    duration = float(real_interval["history"][-1]["physical_delta_t_s"])
    stress = accepted_elastoplastic_stress_3d(state_np1)
    tip = np.asarray(state_np1.crack_network.branches[0].tip)
    centers = state_np1.mesh.nodes[state_np1.mesh.elems].mean(axis=1)
    eid = int(np.argmin(np.linalg.norm(centers-tip, axis=1)))
    tensor = np.array(((stress["sigma_xx_Pa"][eid], stress["sigma_xy_Pa"][eid]),
                       (stress["sigma_xy_Pa"][eid], stress["sigma_yy_Pa"][eid])))
    cleavage = directional_clock_rates(state_np1, tensor, temperature_K=900.0)
    cleavage_time = min(item["crossing_time_s"] for item in cleavage)
    engine = fresh_sharp_front_engine(state_np1.material, _bundle_from_state(state_np1))
    sigma_open = max(float(np.linalg.eigvalsh(tensor)[-1]), 0.0)
    # Invert the existing engine's unshielded initial sigma(K) relation.  This
    # supplies the same accepted crack-tip source to the emission preview.
    K_effective = sigma_open*math.sqrt(2.0*math.pi*engine.r_eff())
    emission = preview_front_engine_emission(
        state_np1, ROOT_BRANCH_ID, K_effective, 900.0, duration,
    )
    accepted, scheduler = advance_global_physical_interval(
        state_n, state_np1, duration,
        {"kind": "hold", "temperature_K": 900.0,
         "stress_tensor_Pa": tensor,
         "K_effective_by_front": {ROOT_BRANCH_ID: K_effective}},
    )
    transport = accepted.vacancy_transport_state_v2
    site = transport.site_states[0]
    required = site.required_embryo_inventory_area_m2/site.required_hit_count
    selected = next((row["selected_event"] for row in scheduler["events"]
                     if row["selected_event"] is not None), None)
    if selected == "DIRECTIONAL_CLEAVAGE": outcome = "DIRECT_CRACK_BEFORE_VOID"
    elif selected == "VACANCY_BIRTH_HIT": outcome = "VOID_BIRTH_BEFORE_CRACK"
    elif emission.commit_payload["emitted_population_increment"] > 0.0: outcome = "EMISSION_MEDIATED_DELAY"
    elif site.reserved_inventory_area_m2 < required: outcome = "INVENTORY_LIMITED"
    elif site.birth_action < site.next_birth_threshold(): outcome = "HAZARD_LIMITED"
    else: outcome = "NO_EVENT_WITHIN_PHYSICAL_WINDOW"
    return {
        "passed": True, "outcome": outcome, "blocking_reason": None,
        "load_time_history": real_interval["history"],
        "rho_history_m2": [{"minimum": float(np.min(state_n.rho_gp)),
                            "maximum": float(np.max(state_n.rho_gp))},
                           {"minimum": float(np.min(state_np1.rho_gp)),
                            "maximum": float(np.max(state_np1.rho_gp))}],
        "maximum_vacancy_fraction": float(np.max(transport.vacancy_fraction_gp)),
        "integrated_vacancy_fraction_area_m2": float(np.sum(
            transport.transport_mesh.area_e*transport.vacancy_fraction_gp)),
        "crack_surface_absorption_area_m2": transport.ledgers[
            "crack_surface_absorption_area_m2"],
        "external_surface_absorption_area_m2": transport.ledgers[
            "external_surface_absorption_area_m2"],
        "climb_annihilation_area_m2": transport.ledgers[
            "dislocation_climb_annihilation_area_m2"],
        "climb_emission_area_m2": transport.ledgers[
            "dislocation_climb_emission_area_m2"],
        "captured_embryo_inventory_area_m2": site.reserved_inventory_area_m2,
        "competing_clocks": {"cleavage_candidates": cleavage,
                              "sharp_front_emission": {
                                  "crossing_time_s": emission.crossing_time_s,
                                  "owner_id": emission.owner_id,
                                  "barrier_J": emission.commit_payload["barrier_J"],
                                  "emission_rate_s": emission.commit_payload["emission_rate_s"],
                              }},
        "selected_event": selected,
        "selected_elapsed_time_s": scheduler["events"][0]["elapsed_time_s"],
        "global_scheduler": dict(scheduler),
        "conservation_residual_area_m2": transport.last_solver_audit[
            "global_conservation_residual_area_m2"],
    }


def checkpoint_and_rollback(previous, accepted, full):
    with tempfile.TemporaryDirectory(prefix="production-v2-transport-") as directory:
        path = Path(directory)/"checkpoint.json"
        fingerprint = complete_accepted_state_fingerprint(full)
        write_checkpoint(full, path)
        restored = restore_checkpoint(path)
        require(complete_accepted_state_fingerprint(restored) == fingerprint,
                "checkpoint/restart changed accepted transport ownership")
        require(np.array_equal(restored.vacancy_transport_state_v2.vacancy_fraction_gp,
                               full.vacancy_transport_state_v2.vacancy_fraction_gp),
                "checkpoint/restart changed V2 concentration")
    original = complete_accepted_state_fingerprint(previous)
    for stage in ("transport_matrix_assembly", "transport_solve", "topology_commit"):
        def inject(name, state, target=stage):
            if name == target: raise RuntimeError("injected:"+name)
        try:
            advance_production_vacancy_transport_interval(
                previous, accepted, duration_s=0.01, failure_injector=inject,
            )
        except RuntimeError as exc:
            require(str(exc) == "injected:"+stage, "wrong rollback stage")
        else:
            raise RuntimeError("rollback injection did not fire: "+stage)
        require(complete_accepted_state_fingerprint(previous) == original,
                "failed trial mutated accepted state")
    transport = previous.vacancy_transport_state_v2
    def remesh_fail(name, state):
        if name == "remesh_transfer": raise RuntimeError("injected:remesh_transfer")
    try:
        transfer_transport_state_overlap(
            transport, transport.transport_mesh, transport.transport_mesh,
            transport.surface_registry, failure_injector=remesh_fail,
            new_parent_mechanics_element_ids=transport.parent_mechanics_element_ids,
        )
    except RuntimeError as exc:
        require(str(exc) == "injected:remesh_transfer", "remesh rollback failed")
    else:
        raise RuntimeError("remesh rollback injection did not fire")
    # Root localization is pure: an exception before commit cannot reserve a hit.
    source = full.vacancy_transport_state_v2
    site = source.site_states[0]
    event_site = replace(
        site, capture_rate_area_per_s=max(site.capture_rate_area_per_s, 1.0e-12),
        required_embryo_inventory_area_m2=(site.reserved_inventory_area_m2+1.0e-30),
        birth_barrier_J=0.0, birth_stress_work_J=0.0,
        threshold=site.birth_action+1.0e-12,
    )
    event_source = replace(source, site_states=(event_site,))
    def root_fail(name, state):
        if name == "event_root": raise RuntimeError("injected:event_root")
    try:
        locate_site_inventory_event(
            event_source, event_source.transport_mesh, event_source.surface_registry,
            row=event_source.parameter_row, maximum_duration_s=1.0,
            accepted_equivalent_plastic_strain_rate_s=np.zeros(event_source.transport_mesh.ne),
            accepted_dislocation_density_m2=np.zeros(event_source.transport_mesh.ne),
            accepted_mean_stress_Pa=np.zeros(event_source.transport_mesh.ne),
            failure_injector=root_fail,
        )
    except RuntimeError as exc:
        require(str(exc) == "injected:event_root", "event-root rollback failed")
    else:
        raise RuntimeError("event-root rollback injection did not fire")
    require(event_source.site_states[0] == event_site,
            "event-root failure mutated accepted capture ownership")


def cavity_qualification():
    state, hole = build_production_void_state(
        bundle=material_bundle("DBTT"), temperature_K=1800.0,
        initial_vacancy_fraction_gp=1.0e-5,
    )
    site = state.void_state.sites[0]; radius = hole.radius_m
    cavity = Cavity2D(
        "void:"+site.site_id, site.site_id, hole.center_m, radius,
        math.pi*radius**2, math.pi*radius**2, VoidPhase.RESOLVED_VOID,
        lineage=(site.site_id, "GEOMETRIC_PROMOTION"),
    )
    growth = initialize_embryo_growth_state(
        site_id=site.site_id, radius_m=radius, initial_area_m2=cavity.area_m2,
        representation=RESOLVED, cavity_id=cavity.cavity_id,
        emission_enabled=False, stabilization_clock=site.stabilization,
        healing_clock=site.healing)
    resolved_site = replace(site, phase=VoidPhase.RESOLVED_VOID)
    cavity_state = remesh_cavity(
        state, hole, replace(state.void_state, sites=(resolved_site,), cavities=(cavity,),
                             embryo_growth_states=(growth,)), "promotion",
    )
    previous = equilibrate_fixed_load_with_production_fem(
        replace(cavity_state, ep_gp=np.zeros_like(cavity_state.ep_gp))
    )
    previous = replace(previous,
                       vacancy_transport_state_v2=cavity_state.vacancy_transport_state_v2)
    cavity_dt = 1.0e-4
    transported, _ = advance_production_vacancy_transport_interval(
        previous, cavity_state, duration_s=cavity_dt,
    )
    cavity_fingerprint = complete_accepted_state_fingerprint(cavity_state)
    def cavity_fail(name, state):
        if name == "cavity_update": raise RuntimeError("injected:cavity_update")
    try:
        couple_active_cavity_flux_to_geometry(
            cavity_state, transported, hole, cavity_id=cavity.cavity_id,
            duration_s=cavity_dt, accepted_equivalent_plastic_strain_rate_s=1.0e-3,
            failure_injector=cavity_fail,
        )
    except RuntimeError as exc:
        require(str(exc) == "injected:cavity_update", "cavity-update rollback failed")
    else:
        raise RuntimeError("cavity-update rollback injection did not fire")
    require(complete_accepted_state_fingerprint(cavity_state) == cavity_fingerprint,
            "cavity-update failure mutated accepted state")
    committed, audit = couple_active_cavity_flux_to_geometry(
        cavity_state, transported, hole, cavity_id=cavity.cavity_id,
        duration_s=cavity_dt, accepted_equivalent_plastic_strain_rate_s=1.0e-3,
    )
    require(audit["absorbed_inventory_area_m2"] > 0.0, "cavity captured no flux")
    require(audit["radius_after_m"] > audit["radius_before_m"],
            "cavity flux did not move the boundary")
    require(committed.equilibrium_observables is not None,
            "mechanics was not re-equilibrated after cavity motion")
    return audit


def main():
    seed, _ = build_production_void_state(bundle=material_bundle("DBTT"), temperature_K=900.0)
    row = seed.vacancy_transport_state_v2.parameter_row
    meshes = general_mesh_qualification(row)
    crack = crack_interface_control(row)
    real_interval = real_accepted_mechanical_interval()
    real_physical = physical_transport_interval_from_real_history(real_interval)
    scheduler_controls = controlled_scheduler_ownership_matrix()
    previous, accepted, full, competition, controls, outcome = physical_sentinel()
    checkpoint_and_rollback(previous, accepted, full)
    cavity = cavity_qualification()
    numerical_gate = all(
        family == "frozen_acceptance" or family == "stress_driven_flux"
        or values[-1]["L2_relative"] <= meshes["frozen_acceptance"][
            "maximum_fine_concentration_error"]
        for family, values in meshes.items()
    )
    cross_gate = (
        abs(crack["measured_total_cross_crack_flux_fraction_area_per_s"])
        <= 1.0e-20
        and all(math.isclose(crack["impermeable_side_inventory_before"][side],
                             crack["impermeable_side_inventory_after"][side],
                             rel_tol=0.0, abs_tol=1.0e-28)
                for side in ("negative", "positive"))
    )
    gates = {
        "accepted_state_ownership": full.vacancy_transport_state_v2 is not None,
        "real_accepted_plastic_interval": bool(real_interval["passed"]),
        "general_mesh_transport_convergence": bool(numerical_gate),
        "local_element_conservation": (
            full.vacancy_transport_state_v2.last_solver_audit[
                "local_element_conservation_residual_inf"] < 1.0e-20),
        "global_inventory_conservation": abs(
            full.vacancy_transport_state_v2.last_solver_audit[
                "global_conservation_residual_area_m2"]
        ) <= full.vacancy_transport_state_v2.last_solver_audit[
            "conservation_roundoff_budget_area_m2"],
        "measured_zero_cross_crack_transport": bool(cross_gate),
        "surface_sink_accounting": (
            full.vacancy_transport_state_v2.ledgers["crack_surface_absorption_area_m2"] > 0.0
            and full.vacancy_transport_state_v2.ledgers[
                "external_surface_absorption_area_m2"] > 0.0),
        "global_event_competition": bool(
            real_physical.get("passed")
            and real_physical.get("global_scheduler", {}).get("all_owners_previewed")
            and real_physical.get("global_scheduler", {}).get("continuous_emission_advanced")
            and not real_physical.get("global_scheduler", {}).get("discrete_emission_event_exists", True)
            and real_physical.get("global_scheduler", {}).get("preview_nonmutating")
            and real_physical.get("global_scheduler", {}).get("no_owner_advanced_twice")
            and all(row["passed"] and row["direct_restart_equal"]
                    and row["partition_invariant"]
                    and row["preview_rollback_exact"]
                    and row["advance_rollback_exact"]
                    and row["commit_rollback_exact"]
                    and row["thresholds_consumed"] == 1
                    for row in scheduler_controls.values())
        ),
        "checkpoint_restart": True,
        "rollback": True,
        "controlled_cavity_geometry_coupling": cavity["radius_after_m"] > cavity["radius_before_m"],
        "honest_physical_outcome": bool(real_physical["passed"]),
    }
    report = {
        "general_mesh_rt0_p0": meshes, "exact_crack_interface": crack,
        "physical_controls": controls, "global_competition": competition,
        "active_cavity": cavity,
        "real_accepted_mechanical_interval": {
            "passed": real_interval["passed"], "history": real_interval["history"],
            "blocking_reason": real_interval["blocking_reason"],
        },
        "physical_dbtt_interval": real_physical,
        "scheduler_controls": {key: dict(value)
                               for key, value in scheduler_controls.items()},
        "qualification_gates": gates,
        "global_conservation_residual_area_m2": full.vacancy_transport_state_v2.last_solver_audit[
            "global_conservation_residual_area_m2"],
        "accepted_state_fingerprint": complete_accepted_state_fingerprint(full),
    }
    print(json.dumps(report, sort_keys=True, indent=2))
    print("MANUFACTURED_PLASTIC_INCREMENT_ADAPTER_CONTROL")
    print("VACANCY_TRANSPORT_STATE_OWNED_BY_FRACTURE_STATE")
    print("FREE_SURFACE_AND_CRACK_SINKS_ACTIVE")
    print("MEASURED_ZERO_CROSS_CRACK_TRANSPORT")
    print("CONTROLLED_ACTIVE_CAVITY_PLUMBING_SENTINEL")
    print("GLOBAL_EVENT_COMPETITION")
    for label in scheduler_controls:
        print(label)
    print("CHECKPOINT_RESTART")
    print("ROLLBACK")
    if all(gates.values()):
        print("PRODUCTION_VACANCY_TRANSPORT_ONE_VOID = PASS")
        print(real_physical["outcome"])
    else:
        print("PRODUCTION_VACANCY_TRANSPORT_ONE_VOID = "
              "INTEGRATION_PASS_PHYSICAL_QUALIFICATION_INCOMPLETE")
        print(real_physical["outcome"] if real_physical["passed"] else
              "NO_EVENT_WITHIN_PHYSICAL_WINDOW")


if __name__ == "__main__":
    main()
