#!/usr/bin/env python3
"""Bounded V2 vacancy-transport qualification and one physical one-void study."""
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

from arrhenius_fracture.config import ElasticProperties
from arrhenius_fracture.fem import assemble_mechanics
from arrhenius_fracture.mesh import rebuild_tri_mesh
from arrhenius_fracture.unified_fracture_material_v5 import material_bundle
from arrhenius_fracture.vacancy_transport_phase_field_v2 import (
    LocalizedCaptureSite,
    SurfaceKind,
    VacancyTransportParameterRowV2,
    accepted_plane_strain_mean_stress_Pa,
    accepted_elastoplastic_stress_3d,
    build_exact_cut_transport_mesh,
    build_transport_surface_registry,
    compete_on_common_transport_clock,
    initial_transport_state,
    restore_transport_checkpoint,
    solve_transport_interval,
    transfer_transport_state_overlap,
    transport_conservation,
    vacancy_transport_refinement_indicators,
    write_transport_checkpoint,
)
from arrhenius_fracture.voiding_production_v5 import build_production_void_state


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def structured_mesh(nx, ny, lx=4.0e-6, ly=2.0e-6):
    nodes = np.array([(i*lx/nx, j*ly/ny) for j in range(ny+1) for i in range(nx+1)])
    elems = []
    for j in range(ny):
        for i in range(nx):
            a = j*(nx+1)+i
            elems.extend(((a, a+1, a+nx+2), (a, a+nx+2, a+nx+1)))
    return rebuild_tri_mesh(nodes, np.asarray(elems))


def edge_owners(mesh):
    owners = {}
    for eid, triangle in enumerate(mesh.elems):
        for a, b in ((triangle[0], triangle[1]), (triangle[1], triangle[2]),
                     (triangle[2], triangle[0])):
            owners.setdefault(tuple(sorted((int(a), int(b)))), []).append(eid)
    return owners


def rectangular_tags(mesh, *, right=SurfaceKind.CONTACT_OR_GRIP_NO_FLUX):
    nodes = np.asarray(mesh.nodes)
    xmin, ymin = nodes.min(axis=0); xmax, ymax = nodes.max(axis=0)
    result = {}
    for edge, owners in edge_owners(mesh).items():
        if len(owners) != 1:
            continue
        center = nodes[list(edge)].mean(axis=0)
        if np.isclose(center[0], xmin):
            kind = SurfaceKind.SYMMETRY_NO_FLUX
        elif np.isclose(center[0], xmax):
            kind = right
        elif np.isclose(center[1], ymin) or np.isclose(center[1], ymax):
            kind = SurfaceKind.CONTACT_OR_GRIP_NO_FLUX
        else:
            raise RuntimeError("unclassified manufactured boundary")
        result[edge] = kind
    return result


def production_tags(state):
    top = set(map(int, state.boundary.top_nodes))
    bottom = set(map(int, state.boundary.bot_nodes))
    result = {}
    for edge, owners in edge_owners(state.mesh).items():
        if len(owners) != 1:
            continue
        nodes = set(edge)
        result[edge] = (
            SurfaceKind.CONTACT_OR_GRIP_NO_FLUX
            if nodes <= top or nodes <= bottom
            else SurfaceKind.EXTERNAL_FREE_SURFACE
        )
    return result


def solve(state, mesh, registry, row, duration, epdot, rho=None, stress=None):
    zeros = np.zeros(mesh.ne)
    return solve_transport_interval(
        state, mesh, registry, row=row, duration_s=duration,
        accepted_equivalent_plastic_strain_rate_s=epdot,
        accepted_dislocation_density_m2=zeros if rho is None else rho,
        accepted_mean_stress_Pa=zeros if stress is None else stress,
    )


def numerical_qualification():
    material = ElasticProperties(E=210e9, nu=0.3, b=2.74e-10, Tm=3695.0)
    row = VacancyTransportParameterRowV2.from_material(material)
    mesh = structured_mesh(4, 2)
    noflux = build_transport_surface_registry(mesh, boundary_tags=rectangular_tags(mesh))
    initial = initial_transport_state(
        mesh, temperature_K=1800.0, row=row, registry=noflux,
        vacancy_fraction_gp=np.full(mesh.ne, 1.0e-5),
    )

    uniform = solve(initial, mesh, noflux, row, 0.01, np.zeros(mesh.ne))
    require(np.allclose(uniform.vacancy_fraction_gp, initial.vacancy_fraction_gp,
                        atol=2e-20, rtol=0.0), "uniform diffusion control failed")

    sourced = solve(initial, mesh, noflux, row, 0.01, np.full(mesh.ne, 0.5))
    require(np.min(sourced.vacancy_fraction_gp) > 1.0e-5,
            "accepted plastic source control failed")

    climbed = solve(initial, mesh, noflux, row, 0.01, np.zeros(mesh.ne),
                    rho=np.full(mesh.ne, 1.0e16))
    require(np.max(climbed.vacancy_fraction_gp) < 1.0e-5,
            "dislocation climb sink control failed")

    exposed_tags = rectangular_tags(mesh, right=SurfaceKind.EXTERNAL_FREE_SURFACE)
    exposed_cut = build_exact_cut_transport_mesh(
        mesh, p0=[2e-6, 0.25e-6], p1=[2e-6, 1.75e-6],
        topology_boundary_tags=tuple(
            {"node_ids": edge, "kind": kind.value} for edge, kind in exposed_tags.items()
        ), segment_id="control-crack",
    )
    exposed_mesh, exposed = exposed_cut.mesh, exposed_cut.registry
    exposed_initial = initial_transport_state(
        exposed_mesh, temperature_K=1800.0, row=row, registry=exposed,
        vacancy_fraction_gp=np.full(exposed_mesh.ne, 1.0e-5),
    )
    absorbed = solve(exposed_initial, exposed_mesh, exposed, row, 0.01,
                     np.zeros(exposed_mesh.ne))
    require(absorbed.ledgers["crack_surface_absorption_area_m2"] > 0.0,
            "internal crack sink control failed")
    require(absorbed.ledgers["external_surface_absorption_area_m2"] > 0.0,
            "external surface sink control failed")
    require(exposed.count(SurfaceKind.CRACK_FREE_SURFACE) >= 2,
            "exact cut did not create independent crack faces")

    fine = structured_mesh(8, 4)
    fine_registry = build_transport_surface_registry(fine, boundary_tags=rectangular_tags(fine))
    transferred, transfer_audit = transfer_transport_state_overlap(
        uniform, mesh, fine, fine_registry,
    )
    conservation = transport_conservation(transferred, fine.area_e)
    require(abs(conservation["conservation_residual_area_m2"]) <= 1.0e-24,
            "adaptive transport transfer failed conservation")
    indicators = vacancy_transport_refinement_indicators(
        fine, fine_registry, vacancy_fraction_gp=transferred.vacancy_fraction_gp,
        plastic_source_rate_s=np.ones(fine.ne), climb_exchange_rate_s=np.zeros(fine.ne),
        duration_s=0.01, diffusivity_m2_s=1.0e-12,
    )
    require(indicators["marked_element_ids"], "transport adaptation produced no marks")

    site = LocalizedCaptureSite(
        site_id="controlled-site", center_xy_m=(2e-6, 1e-6), support_radius_m=0.8e-6,
        capture_rate_area_per_s=1.0e-12, required_embryo_inventory_area_m2=1.0e-18,
        birth_barrier_J=0.0, threshold=0.25, rng_fingerprint="controlled-fixed",
    )
    competition_initial = initial_transport_state(
        mesh, temperature_K=1800.0, row=row, registry=noflux,
        vacancy_fraction_gp=np.full(mesh.ne, 1.0e-5), site_states=(site,),
    )
    competition = compete_on_common_transport_clock(
        competition_initial, mesh, noflux, row=row, maximum_duration_s=0.1,
        accepted_equivalent_plastic_strain_rate_s=np.zeros(mesh.ne),
        accepted_dislocation_density_m2=np.zeros(mesh.ne),
        accepted_mean_stress_Pa=np.zeros(mesh.ne),
        external_event_times_s={"ORDINARY_CLEAVAGE": 0.025},
    )
    require(competition["event"] == "ORDINARY_CLEAVAGE",
            "common-time external/vacancy competition failed")
    require(competition["transport_state"].site_states[0].reserved_inventory_area_m2 > 0.0,
            "losing site clock did not advance through common elapsed time")

    with tempfile.TemporaryDirectory(prefix="vacancy-phase-field-v2-") as directory:
        checkpoint = Path(directory)/"accepted.json"
        write_transport_checkpoint(absorbed, exposed, checkpoint)
        restarted, restarted_registry = restore_transport_checkpoint(checkpoint)
        require(np.array_equal(restarted.vacancy_fraction_gp, absorbed.vacancy_fraction_gp),
                "transport checkpoint changed concentration")
        require(restarted_registry.fingerprint == exposed.fingerprint,
                "transport checkpoint changed surface ownership")

    return row, {
        "plastic_production_area_m2": sourced.ledgers["plastic_production_area_m2"],
        "climb_annihilation_area_m2": climbed.ledgers["dislocation_climb_annihilation_area_m2"],
        "crack_absorption_area_m2": absorbed.ledgers["crack_surface_absorption_area_m2"],
        "external_absorption_area_m2": absorbed.ledgers["external_surface_absorption_area_m2"],
        "remesh_method": transfer_audit["method"],
        "common_time_winner": competition["event"],
    }


def physical_one_void_study():
    bundle = material_bundle("DBTT")
    accepted, _ = build_production_void_state(
        bundle=bundle, temperature_K=900.0, initial_vacancy_fraction_gp=None,
    )
    transport = accepted.vacancy_transport_state_v2
    row = transport.parameter_row
    registry = transport.surface_registry
    transport_mesh = transport.transport_mesh
    site_owner = accepted.void_state.sites[0]
    site = LocalizedCaptureSite(
        site_id=site_owner.site_id, center_xy_m=site_owner.center_m,
        support_radius_m=5.0e-5,
        capture_rate_area_per_s=(
            row.vacancy_diffusivity_prefactor_m2_s * math.exp(
                -row.vacancy_migration_energy_eV*1.602176634e-19/(1.380649e-23*900.0)
            )
        ),
        required_embryo_inventory_area_m2=2.0e-20,
        hit_count=site_owner.hits, required_hit_count=site_owner.required_hits,
        birth_action=site_owner.birth.accumulated,
        threshold=site_owner.birth.threshold, rng_fingerprint="state-owned-seed-3621",
    )
    transport = replace(transport, site_states=(site,))
    stress3d = accepted_elastoplastic_stress_3d(accepted)
    parent = transport.parent_mechanics_element_ids
    mean_stress = np.asarray(stress3d["mean_stress_Pa"])[parent]
    # Previous and accepted mechanical states are identical in this bounded
    # physical hold: consequently the accepted plastic increment is exactly
    # zero.  No rate is extrapolated to manufacture a void.
    competition = compete_on_common_transport_clock(
        transport, transport_mesh, registry, row=row, maximum_duration_s=0.1,
        accepted_equivalent_plastic_strain_rate_s=np.zeros(transport_mesh.ne),
        accepted_dislocation_density_m2=np.asarray(accepted.rho_gp)[parent],
        accepted_mean_stress_Pa=mean_stress,
        external_event_times_s={},
    )
    final = competition["transport_state"]
    outcome = "NO_EVENT_WITHIN_PHYSICAL_WINDOW"
    report = {
        "outcome": outcome,
        "physical_window_s": competition["elapsed_time_s"],
        "material_bundle_id": bundle.bundle_id,
        "transport_parameter_row_id": row.row_id,
        "transport_parameter_fingerprint": row.fingerprint,
        "surface_registry_fingerprint": registry.fingerprint,
        "crack_face_count": registry.count(SurfaceKind.CRACK_FREE_SURFACE),
        "external_free_surface_face_count": registry.count(SurfaceKind.EXTERNAL_FREE_SURFACE),
        "maximum_vacancy_fraction": float(np.max(final.vacancy_fraction_gp)),
        "reserved_embryo_inventory_area_m2": final.site_states[0].reserved_inventory_area_m2,
        "required_embryo_inventory_area_m2": final.site_states[0].required_embryo_inventory_area_m2,
        "transport_conservation_residual_area_m2": final.last_solver_audit[
            "global_conservation_residual_area_m2"
        ],
        "accepted_plastic_increment_rate_max_s": 0.0,
        "void_event_forced": False,
    }
    return outcome, report


def main():
    _, qualification = numerical_qualification()
    outcome, physical = physical_one_void_study()
    print("VACANCY_PHASE_FIELD_INITIALIZED")
    print("LOCAL_PLASTIC_VACANCY_SOURCE")
    print("DISLOCATION_CLIMB_SINK")
    print("VACANCY_TRANSPORT_SOLVED")
    print("CRACK_SURFACE_SINK_ACTIVE")
    print("EXTERNAL_FREE_SURFACE_SINK_ACTIVE")
    print("ADAPTIVE_REMESH_CONSERVATIVE")
    print("COMMON_TIME_EVENT_COMPETITION")
    print("VACANCY_PHASE_FIELD_NUMERICS = PASS")
    print(outcome)
    print(json.dumps({"qualification": qualification, "physical_study": physical},
                     indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
