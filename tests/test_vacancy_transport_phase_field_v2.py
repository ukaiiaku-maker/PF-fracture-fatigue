from __future__ import annotations

from dataclasses import replace
import math

import numpy as np
import pytest

from arrhenius_fracture.config import ElasticProperties
from arrhenius_fracture.mesh import rebuild_tri_mesh
from arrhenius_fracture.vacancy_transport_phase_field_v2 import (
    LOCAL_REACTION_V1_LEGACY,
    SCHEMA,
    LocalizedCaptureSite,
    SurfaceKind,
    VacancyTransportParameterRowV2,
    accepted_plane_strain_mean_stress_Pa,
    build_transport_surface_registry,
    build_exact_cut_transport_mesh,
    build_exact_cut_transport_mesh_for_network,
    cavity_growth_rate_from_transport,
    compete_on_common_transport_clock,
    dilute_flux_1d,
    initial_transport_state,
    heal_embryo_reservation,
    locate_site_inventory_event,
    rebind_registry_after_accepted_topology,
    restore_transport_checkpoint,
    solve_transport_interval,
    stabilize_reserved_embryo,
    promote_subgrid_inventory,
    thermodynamic_flux_1d,
    transfer_transport_state_overlap,
    transport_conservation,
    vacancy_energy_study_row,
    vacancy_transport_refinement_indicators,
    write_transport_checkpoint,
)


def structured_mesh(nx=4, ny=2, lx=4.0e-6, ly=2.0e-6):
    nodes = np.array([(i*lx/nx, j*ly/ny) for j in range(ny+1) for i in range(nx+1)])
    elems = []
    for j in range(ny):
        for i in range(nx):
            a = j*(nx+1)+i
            elems.extend(((a, a+1, a+nx+2), (a, a+nx+2, a+nx+1)))
    return rebuild_tri_mesh(nodes, np.asarray(elems))


def boundary_edges(mesh):
    owners = {}
    for eid, tri in enumerate(mesh.elems):
        for a, b in ((tri[0], tri[1]), (tri[1], tri[2]), (tri[2], tri[0])):
            owners.setdefault(tuple(sorted((int(a), int(b)))), []).append(eid)
    return {edge for edge, value in owners.items() if len(value) == 1}


def tags(mesh, *, left=SurfaceKind.SYMMETRY_NO_FLUX,
         right=SurfaceKind.CONTACT_OR_GRIP_NO_FLUX,
         top=SurfaceKind.CONTACT_OR_GRIP_NO_FLUX,
         bottom=SurfaceKind.CONTACT_OR_GRIP_NO_FLUX):
    xy = np.asarray(mesh.nodes)
    xmin, ymin = xy.min(axis=0); xmax, ymax = xy.max(axis=0)
    result = {}
    for edge in boundary_edges(mesh):
        center = xy[list(edge)].mean(axis=0)
        if np.isclose(center[0], xmin): kind = left
        elif np.isclose(center[0], xmax): kind = right
        elif np.isclose(center[1], ymin): kind = bottom
        elif np.isclose(center[1], ymax): kind = top
        else: raise AssertionError("unclassified boundary")
        result[edge] = kind
    return result


def exact_cut(mesh, p0, p1, segment_id="accepted"):
    boundary = tags(mesh)
    rows = tuple({"node_ids": edge, "kind": kind.value} for edge, kind in boundary.items())
    return build_exact_cut_transport_mesh(
        mesh, p0=p0, p1=p1, topology_boundary_tags=rows, segment_id=segment_id,
    )


@pytest.fixture
def row():
    return VacancyTransportParameterRowV2.from_material(
        ElasticProperties(E=210.0e9, nu=0.3, b=2.74e-10, Tm=3695.0)
    )


def advance(state, mesh, registry, row, duration, *, epdot=None, rho=None, stress=None):
    zeros = np.zeros(mesh.ne)
    return solve_transport_interval(
        state, mesh, registry, row=row, duration_s=duration,
        accepted_equivalent_plastic_strain_rate_s=zeros if epdot is None else epdot,
        accepted_dislocation_density_m2=zeros if rho is None else rho,
        accepted_mean_stress_Pa=zeros if stress is None else stress,
    )


def test_schema_reference_values_and_legacy_identity(row):
    assert SCHEMA == "unified-2d.vacancy-transport-phase-field/2"
    assert LOCAL_REACTION_V1_LEGACY == "LOCAL_REACTION_V1_LEGACY"
    assert row.vacancy_formation_energy_eV == 3.2
    assert row.vacancy_migration_energy_eV == 1.7
    assert row.vacancy_diffusivity_prefactor_m2_s == 1.0e-6
    assert row.atomic_volume_m3 == row.vacancy_formation_volume_m3
    assert row.calibration_status == "REFERENCE_UNCALIBRATED"


def test_four_vacancy_energy_study_rows_are_explicit_and_distinct():
    material = ElasticProperties(E=210.0e9, nu=0.3, b=2.74e-10, Tm=3695.0)
    expected = {
        "reference": (3.2, 1.7),
        "low_formation": (2.4, 1.7),
        "low_migration": (3.2, 0.9),
        "combined": (2.4, 0.9),
    }
    rows = {name: vacancy_energy_study_row(material, name) for name in expected}
    assert {(row.vacancy_formation_energy_eV, row.vacancy_migration_energy_eV)
            for row in rows.values()} == set(expected.values())
    assert len({row.fingerprint for row in rows.values()}) == 4
    assert all(row.calibration_status == "REFERENCE_UNCALIBRATED"
               for row in rows.values())
    for name, pair in expected.items():
        assert (rows[name].vacancy_formation_energy_eV,
                rows[name].vacancy_migration_energy_eV) == pair


def test_dilute_flux_identity_and_stress_direction(row):
    full = thermodynamic_flux_1d(1.0e-10, 2.0e-5, 3.0e13, 1200.0, row)
    dilute = dilute_flux_1d(1.0e-10, 2.0e-5, 3.0e13, 1200.0, row)
    assert full == pytest.approx(dilute, rel=2.0e-10, abs=1.0e-40)
    # Positive tensile mean-stress gradient attracts vacancies (+x flux).
    assert thermodynamic_flux_1d(1.0e-5, 0.0, 3.0e15, 1200.0, row) > 0.0


def test_plane_strain_mean_stress_uses_sigma_zz():
    sigma = np.array([[3.0, 6.0], [2.0, 4.0], [100.0, -20.0]])
    assert np.allclose(accepted_plane_strain_mean_stress_Pa(sigma, 0.3), [2.1666666667, 4.3333333333])


def test_uniform_no_flux_field_and_local_balance(row):
    mesh = structured_mesh()
    registry = build_transport_surface_registry(mesh, boundary_tags=tags(mesh))
    state = initial_transport_state(
        mesh, temperature_K=1200.0, row=row, registry=registry,
        vacancy_fraction_gp=np.full(mesh.ne, 1.0e-5),
    )
    result = advance(state, mesh, registry, row, 0.3)
    assert np.allclose(result.vacancy_fraction_gp, state.vacancy_fraction_gp,
                       rtol=0.0, atol=2.0e-20)
    assert result.last_solver_audit["local_element_conservation_residual_inf"] < 1.0e-25
    assert abs(transport_conservation(result, mesh.area_e)["conservation_residual_area_m2"]) < 1.0e-28


def test_uniform_climb_and_plastic_source_converge_to_analytic_solutions(row):
    mesh = structured_mesh(2, 1)
    registry = build_transport_surface_registry(mesh, boundary_tags=tags(mesh))
    c0 = 1.0e-5
    initial = initial_transport_state(mesh, temperature_K=1600.0, row=row, registry=registry,
                                      vacancy_fraction_gp=np.full(mesh.ne, c0))
    rho = np.full(mesh.ne, 5.0e15)
    # Backward Euler should converge first order to exact climb relaxation.
    coarse = advance(initial, mesh, registry, row, 0.02, rho=rho)
    fine = initial
    for _ in range(20): fine = advance(fine, mesh, registry, row, 0.001, rho=rho)
    D = fine.last_solver_audit["diffusivity_m2_s"]
    from arrhenius_fracture.vacancy_transport_phase_field_v2 import equilibrium_vacancy_fraction
    ceq = equilibrium_vacancy_fraction(1600.0, row)
    exact = ceq+(c0-ceq)*math.exp(-D*row.dislocation_climb_capture_efficiency*rho[0]*0.02)
    assert abs(fine.vacancy_fraction_gp[0]-exact) < abs(coarse.vacancy_fraction_gp[0]-exact)

    source_initial = initial_transport_state(mesh, temperature_K=1600.0, row=row, registry=registry,
                                             vacancy_fraction_gp=np.full(mesh.ne, c0))
    epdot = np.full(mesh.ne, 0.4)
    fine = source_initial
    for _ in range(40): fine = advance(fine, mesh, registry, row, 0.0005, epdot=epdot)
    source0 = row.plastic_vacancy_coefficient_per_strain*epdot[0]
    exact = 1.0-(1.0-c0)*math.exp(-source0*0.02)
    assert fine.vacancy_fraction_gp[0] == pytest.approx(exact, rel=1.0e-8)


def test_one_and_two_perfect_sink_slab_deplete_and_account(row):
    mesh = structured_mesh(12, 1, lx=12.0e-6, ly=1.0e-6)
    one_registry = build_transport_surface_registry(
        mesh, boundary_tags=tags(mesh, right=SurfaceKind.EXTERNAL_FREE_SURFACE)
    )
    c0 = np.full(mesh.ne, 1.0e-5)
    one = initial_transport_state(mesh, temperature_K=1800.0, row=row, registry=one_registry,
                                  vacancy_fraction_gp=c0)
    one = advance(one, mesh, one_registry, row, 0.1)
    centers = mesh.nodes[mesh.elems].mean(axis=1)[:, 0]
    assert np.mean(one.vacancy_fraction_gp[centers > 10e-6]) < np.mean(one.vacancy_fraction_gp[centers < 2e-6])
    assert one.ledgers["external_surface_absorption_area_m2"] > 0.0

    two_registry = build_transport_surface_registry(
        mesh, boundary_tags=tags(mesh, left=SurfaceKind.EXTERNAL_FREE_SURFACE,
                                 right=SurfaceKind.EXTERNAL_FREE_SURFACE)
    )
    two = initial_transport_state(mesh, temperature_K=1800.0, row=row, registry=two_registry,
                                  vacancy_fraction_gp=c0)
    two = advance(two, mesh, two_registry, row, 0.1)
    assert np.sum(two.vacancy_fraction_gp*mesh.area_e) < np.sum(one.vacancy_fraction_gp*mesh.area_e)


def test_one_dimensional_slab_matches_one_and_two_sink_series(row):
    mesh = structured_mesh(80, 4, lx=20e-6, ly=1e-6)
    c0 = 1e-5
    from arrhenius_fracture.vacancy_transport_phase_field_v2 import equilibrium_vacancy_fraction
    ceq = equilibrium_vacancy_fraction(1800.0, row)
    results = []
    for two_sinks in (False, True):
        boundary = tags(mesh, right=SurfaceKind.EXTERNAL_FREE_SURFACE)
        if two_sinks:
            x = mesh.nodes[:, 0]
            for edge in list(boundary):
                if np.isclose(np.mean(x[list(edge)]), 0.0):
                    boundary[edge] = SurfaceKind.EXTERNAL_FREE_SURFACE
        registry = build_transport_surface_registry(mesh, boundary_tags=boundary)
        state = initial_transport_state(mesh, temperature_K=1800.0, row=row, registry=registry,
                                        vacancy_fraction_gp=np.full(mesh.ne, c0))
        for _ in range(100):
            state = advance(state, mesh, registry, row, 0.01)
        results.append(state)
    D = results[0].last_solver_audit["diffusivity_m2_s"]
    L = 20e-6
    one_factor = math.fsum(
        2.0/(((n+0.5)*math.pi)**2)*math.exp(-D*((n+0.5)*math.pi/L)**2)
        for n in range(1000)
    )
    two_factor = math.fsum(
        8.0/(n*n*math.pi**2)*math.exp(-D*(n*math.pi/L)**2)
        for n in range(1, 1000, 2)
    )
    for state, factor in zip(results, (one_factor, two_factor)):
        average = float(np.sum(state.vacancy_fraction_gp*mesh.area_e)/np.sum(mesh.area_e))
        exact = ceq+(c0-ceq)*factor
        assert average == pytest.approx(exact, rel=0.01)


def test_embedded_crack_has_two_faces_blocks_cross_flux_and_uses_physical_length(row):
    mesh = structured_mesh(8, 4, lx=8e-6, ly=4e-6)
    cut = exact_cut(mesh, [4e-6, 0.5e-6], [4e-6, 3.5e-6], "accepted-main")
    mesh, registry = cut.mesh, cut.registry
    crack_faces = [face for face in registry.faces if face.kind == SurfaceKind.CRACK_FREE_SURFACE]
    assert math.fsum(face.length_m for face in crack_faces) == pytest.approx(2.0*3e-6, rel=1e-12)
    assert registry.crack_physical_length_m == pytest.approx(3e-6)
    assert not registry.blocked_interior_edges
    state = initial_transport_state(mesh, temperature_K=1800.0, row=row, registry=registry,
                                    vacancy_fraction_gp=np.full(mesh.ne, 1e-5))
    result = advance(state, mesh, registry, row, 0.02)
    assert result.ledgers["crack_surface_absorption_area_m2"] > 0.0
    assert all(face.node_ids is not None for face in registry.faces)


def test_conforming_and_embedded_crack_absorption_parity(row):
    mesh = structured_mesh(32, 16, lx=8e-6, ly=4e-6)
    absorption = []
    for x in (4.0e-6, 4.13e-6):
        cut = exact_cut(mesh, [x, 0.5e-6], [x, 3.5e-6], "same-physical-crack")
        state = initial_transport_state(cut.mesh, temperature_K=1800.0, row=row, registry=cut.registry,
                                        vacancy_fraction_gp=np.full(cut.mesh.ne, 1e-5))
        state = advance(state, cut.mesh, cut.registry, row, 0.01)
        absorption.append(state.ledgers["crack_surface_absorption_area_m2"])
    assert absorption[1] == pytest.approx(absorption[0], rel=0.03)


def test_exact_cut_supports_owned_branched_crack_network(row):
    mechanics = structured_mesh(8, 8, lx=8e-6, ly=8e-6)
    boundary = tuple({"node_ids": edge, "kind": kind.value}
                     for edge, kind in tags(mechanics).items())
    segments = (
        ([1e-6, 4e-6], [4e-6, 4e-6], "root"),
        ([4e-6, 4e-6], [7e-6, 6e-6], "child-up"),
        ([4e-6, 4e-6], [7e-6, 2e-6], "child-down"),
    )
    cut = build_exact_cut_transport_mesh_for_network(
        mechanics, crack_segments=segments, topology_boundary_tags=boundary,
    )
    expected = math.fsum(math.dist(first, second) for first, second, _ in segments)
    assert cut.registry.crack_physical_length_m == pytest.approx(expected)
    for _, _, identity in segments:
        for side in (-1, 1):
            assert math.fsum(face.length_m for face in cut.registry.faces
                             if face.segment_id == identity and face.side == side) > 0.0
    state = initial_transport_state(
        cut.mesh, temperature_K=1800.0, row=row, registry=cut.registry,
        vacancy_fraction_gp=np.full(cut.mesh.ne, 1e-5),
    )
    result = advance(state, cut.mesh, cut.registry, row, 0.001)
    assert result.ledgers["crack_surface_absorption_area_m2"] > 0.0


def test_cavity_flux_goes_to_cavity_inventory_not_generic_surface(row):
    mesh = structured_mesh(4, 2)
    cavity_tags = tags(mesh, right=SurfaceKind.ACTIVE_CAVITY_SURFACE)
    registry = build_transport_surface_registry(mesh, boundary_tags=cavity_tags)
    state = initial_transport_state(mesh, temperature_K=1800.0, row=row, registry=registry,
                                    vacancy_fraction_gp=np.full(mesh.ne, 1e-5))
    result = advance(state, mesh, registry, row, 0.05)
    absorbed = result.ledgers["active_cavity_absorption_area_m2"]
    assert absorbed > 0.0
    assert result.ledgers["active_cavity_inventory_area_m2"] == pytest.approx(absorbed)
    assert result.ledgers["external_surface_absorption_area_m2"] == 0.0
    growth = cavity_growth_rate_from_transport(
        absorbed_inventory_rate_area_m2_s=absorbed/0.05, radius_m=1e-7,
        equivalent_plastic_strain_rate_s=0.0, plastic_accommodation_coefficient=1.0,
    )
    assert growth["Rdot_transport_m_s"] > 0.0
    assert growth["Rdot_m_s"] == 0.0
    assert growth["limiting_branch"] == "PLASTIC_ACCOMMODATION_LIMITED"


def test_site_capture_is_local_conservative_and_inventory_gates_hit(row):
    mesh = structured_mesh(8, 4, lx=8e-6, ly=4e-6)
    registry = build_transport_surface_registry(mesh, boundary_tags=tags(mesh))
    site = LocalizedCaptureSite(
        site_id="s0", center_xy_m=(4e-6, 2e-6), support_radius_m=1.2e-6,
        capture_rate_area_per_s=2e-12, required_embryo_inventory_area_m2=1e-18,
        birth_barrier_J=0.0, threshold=0.2, rng_fingerprint="frozen",
    )
    state = initial_transport_state(mesh, temperature_K=1600.0, row=row, registry=registry,
                                    vacancy_fraction_gp=np.full(mesh.ne, 1e-5), site_states=(site,))
    event_time, result = locate_site_inventory_event(
        state, mesh, registry, row=row, maximum_duration_s=1.0,
        accepted_equivalent_plastic_strain_rate_s=np.zeros(mesh.ne),
        accepted_dislocation_density_m2=np.zeros(mesh.ne),
        accepted_mean_stress_Pa=np.zeros(mesh.ne), time_tolerance_s=1e-7,
    )
    assert event_time is not None
    assert result.site_states[0].hit_count == 1
    assert result.site_states[0].reserved_inventory_area_m2 >= 1e-18
    assert result.site_states[0].birth_action >= result.site_states[0].threshold
    assert abs(transport_conservation(result, mesh.area_e)["conservation_residual_area_m2"]) < 1e-27
    shorter_partition_time, _ = locate_site_inventory_event(
        state, mesh, registry, row=row, maximum_duration_s=0.5,
        accepted_equivalent_plastic_strain_rate_s=np.zeros(mesh.ne),
        accepted_dislocation_density_m2=np.zeros(mesh.ne),
        accepted_mean_stress_Pa=np.zeros(mesh.ne), time_tolerance_s=1e-7,
    )
    assert shorter_partition_time == pytest.approx(event_time, abs=2e-7)
    def fail_event(stage, _trial):
        if stage == "event_root":
            raise RuntimeError("injected-event-root")
    with pytest.raises(RuntimeError, match="injected-event-root"):
        locate_site_inventory_event(
            state, mesh, registry, row=row, maximum_duration_s=0.5,
            accepted_equivalent_plastic_strain_rate_s=np.zeros(mesh.ne),
            accepted_dislocation_density_m2=np.zeros(mesh.ne),
            accepted_mean_stress_Pa=np.zeros(mesh.ne), time_tolerance_s=1e-7,
            failure_injector=fail_event,
        )
    assert state.site_states[0].reserved_inventory_area_m2 == 0.0

    competition = compete_on_common_transport_clock(
        state, mesh, registry, row=row, maximum_duration_s=1.0,
        accepted_equivalent_plastic_strain_rate_s=np.zeros(mesh.ne),
        accepted_dislocation_density_m2=np.zeros(mesh.ne),
        accepted_mean_stress_Pa=np.zeros(mesh.ne),
        external_event_times_s={"ORDINARY_CLEAVAGE": 0.25*event_time},
        time_tolerance_s=1e-7,
    )
    assert competition["event"] == "ORDINARY_CLEAVAGE"
    assert competition["transport_state"].site_states[0].reserved_inventory_area_m2 > 0.0
    assert competition["transport_state"].site_states[0].hit_count == 0

    stabilized = stabilize_reserved_embryo(result, "s0")
    amount = stabilized.ledgers["subgrid_cavity_inventory_area_m2"]
    promoted = promote_subgrid_inventory(stabilized, amount)
    assert promoted.ledgers["subgrid_cavity_inventory_area_m2"] == pytest.approx(0.0, abs=1e-30)
    assert promoted.ledgers["active_cavity_inventory_area_m2"] == pytest.approx(amount)
    assert abs(transport_conservation(promoted, mesh.area_e)["conservation_residual_area_m2"]) < 1e-27


def test_healed_embryo_returns_or_explicitly_annihilates_inventory(row):
    mesh = structured_mesh(8, 4, lx=8e-6, ly=4e-6)
    registry = build_transport_surface_registry(mesh, boundary_tags=tags(mesh))
    site = LocalizedCaptureSite(
        site_id="heal", center_xy_m=(4e-6, 2e-6), support_radius_m=1.2e-6,
        capture_rate_area_per_s=0.0, reserved_inventory_area_m2=1e-18,
        required_embryo_inventory_area_m2=2e-18,
    )
    state = initial_transport_state(mesh, temperature_K=1600.0, row=row, registry=registry,
                                    vacancy_fraction_gp=np.full(mesh.ne, 1e-5), site_states=(site,))
    returned = heal_embryo_reservation(state, mesh, "heal")
    assert returned.site_states[0].reserved_inventory_area_m2 == 0.0
    assert np.sum(returned.vacancy_fraction_gp*mesh.area_e) > np.sum(state.vacancy_fraction_gp*mesh.area_e)
    assert abs(transport_conservation(returned, mesh.area_e)["conservation_residual_area_m2"]) < 1e-27
    annihilated = heal_embryo_reservation(
        state, mesh, "heal", disposition="EXPLICIT_ANNIHILATION_LEDGER",
    )
    assert annihilated.ledgers["embryo_healing_annihilation_area_m2"] == pytest.approx(1e-18)
    assert abs(transport_conservation(annihilated, mesh.area_e)["conservation_residual_area_m2"]) < 1e-27


def test_refinement_indicators_and_overlap_transfer(row):
    coarse = structured_mesh(2, 1)
    fine = structured_mesh(4, 2)
    coarse_registry = build_transport_surface_registry(coarse, boundary_tags=tags(coarse))
    fine_registry = build_transport_surface_registry(fine, boundary_tags=tags(fine))
    state = initial_transport_state(coarse, temperature_K=1200.0, row=row, registry=coarse_registry,
                                    vacancy_fraction_gp=np.linspace(1e-6, 2e-6, coarse.ne))
    transferred, audit = transfer_transport_state_overlap(state, coarse, fine, fine_registry)
    assert audit["flux_transferred"] is False
    assert audit["surface_registry_rebuilt"] is True
    assert np.sum(state.vacancy_fraction_gp*coarse.area_e) == pytest.approx(
        np.sum(transferred.vacancy_fraction_gp*fine.area_e), abs=1e-28
    )
    assert abs(audit["roundoff_correction_fraction_area"]) <= \
        audit["transfer_acceptance_budget_fraction_area"]
    changed_by_correction = ([] if audit["correction_target_element_id"] is None
                             else [audit["correction_target_element_id"]])
    assert len(changed_by_correction) <= 1
    assert audit["coverage_defect_area_m2"] <= \
        audit["coverage_roundoff_budget_area_m2"]
    indicators = vacancy_transport_refinement_indicators(
        fine, fine_registry, vacancy_fraction_gp=transferred.vacancy_fraction_gp,
        plastic_source_rate_s=np.ones(fine.ne), climb_exchange_rate_s=np.zeros(fine.ne),
        duration_s=0.01, diffusivity_m2_s=1e-12,
    )
    assert indicators["marked_element_ids"]


def test_overlap_transfer_rejects_geometric_defect_above_operand_budget(
        row, monkeypatch):
    import arrhenius_fracture.voiding_production_v5 as production

    coarse = structured_mesh(2, 1)
    fine = structured_mesh(4, 2)
    coarse_registry = build_transport_surface_registry(coarse, boundary_tags=tags(coarse))
    fine_registry = build_transport_surface_registry(fine, boundary_tags=tags(fine))
    state = initial_transport_state(
        coarse, temperature_K=1200.0, row=row, registry=coarse_registry,
        vacancy_fraction_gp=np.linspace(1e-6, 2e-6, coarse.ne))
    exact_overlap = production._triangle_overlap_area

    def defective_overlap(*args, **kwargs):
        return 0.99*exact_overlap(*args, **kwargs)

    monkeypatch.setattr(production, "_triangle_overlap_area", defective_overlap)
    with pytest.raises(RuntimeError, match="coverage exceeds operand-aware budget"):
        transfer_transport_state_overlap(state, coarse, fine, fine_registry)


def test_moving_crack_registry_updates_only_after_rebind(row):
    mesh = structured_mesh(8, 4, lx=8e-6, ly=4e-6)
    registry0 = build_transport_surface_registry(mesh, boundary_tags=tags(mesh))
    state = initial_transport_state(mesh, temperature_K=1500.0, row=row, registry=registry0,
                                    vacancy_fraction_gp=np.full(mesh.ne, 1e-5))
    cut = exact_cut(mesh, [2e-6, 2e-6], [4e-6, 2e-6], "accepted-advance")
    registry1 = cut.registry
    with pytest.raises(RuntimeError, match="stale"):
        advance(state, mesh, registry1, row, 0.01)
    rebound, _ = transfer_transport_state_overlap(
        state, mesh, cut.mesh, registry1,
        new_parent_mechanics_element_ids=cut.parent_mechanics_element_ids,
    )
    result = advance(rebound, cut.mesh, registry1, row, 0.01)
    assert result.ledgers["crack_surface_absorption_area_m2"] > 0.0


def test_checkpoint_restart_and_failure_rollback(tmp_path, row):
    mesh = structured_mesh()
    registry = build_transport_surface_registry(mesh, boundary_tags=tags(mesh))
    state = initial_transport_state(mesh, temperature_K=1400.0, row=row, registry=registry,
                                    vacancy_fraction_gp=np.full(mesh.ne, 1e-5))
    accepted = advance(state, mesh, registry, row, 0.01)
    target = tmp_path/"transport.json"
    write_transport_checkpoint(accepted, registry, target)
    restored, restored_registry = restore_transport_checkpoint(target)
    assert np.array_equal(restored.vacancy_fraction_gp, accepted.vacancy_fraction_gp)
    assert restored_registry.fingerprint == registry.fingerprint
    assert dict(restored.ledgers) == dict(accepted.ledgers)
    continued = advance(accepted, mesh, registry, row, 0.01)
    restarted_continued = advance(restored, mesh, restored_registry, row, 0.01)
    assert np.array_equal(continued.vacancy_fraction_gp, restarted_continued.vacancy_fraction_gp)
    assert dict(continued.ledgers) == dict(restarted_continued.ledgers)

    before = accepted.vacancy_fraction_gp.copy(), dict(accepted.ledgers)
    def fail(stage, _state):
        if stage == "transport_solve": raise RuntimeError("injected")
    with pytest.raises(RuntimeError, match="injected"):
        solve_transport_interval(
            accepted, mesh, registry, row=row, duration_s=0.01,
            accepted_equivalent_plastic_strain_rate_s=np.zeros(mesh.ne),
            accepted_dislocation_density_m2=np.zeros(mesh.ne),
            accepted_mean_stress_Pa=np.zeros(mesh.ne), failure_injector=fail,
        )
    assert np.array_equal(accepted.vacancy_fraction_gp, before[0])
    assert dict(accepted.ledgers) == before[1]


def test_explicit_boundary_tags_are_mandatory():
    mesh = structured_mesh(1, 1)
    with pytest.raises(ValueError, match="explicit transport boundary tags incomplete"):
        build_transport_surface_registry(mesh, boundary_tags={})


@pytest.mark.parametrize("stage", [
    "source_assembly", "surface_classification", "transport_matrix_assembly",
    "transport_solve", "site_capture", "cavity_flux_integration",
])
def test_transport_failure_stages_rollback_exactly(stage, row):
    mesh = structured_mesh(2, 1)
    registry = build_transport_surface_registry(mesh, boundary_tags=tags(mesh))
    state = initial_transport_state(mesh, temperature_K=1400.0, row=row, registry=registry,
                                    vacancy_fraction_gp=np.full(mesh.ne, 1e-5))
    before = state.vacancy_fraction_gp.copy(), dict(state.ledgers)
    def fail(name, _trial):
        if name == stage:
            raise RuntimeError("injected-"+stage)
    with pytest.raises(RuntimeError, match="injected"):
        solve_transport_interval(
            state, mesh, registry, row=row, duration_s=0.01,
            accepted_equivalent_plastic_strain_rate_s=np.zeros(mesh.ne),
            accepted_dislocation_density_m2=np.zeros(mesh.ne),
            accepted_mean_stress_Pa=np.zeros(mesh.ne), failure_injector=fail,
        )
    assert np.array_equal(state.vacancy_fraction_gp, before[0])
    assert dict(state.ledgers) == before[1]


def test_remesh_topology_and_checkpoint_publication_failures_rollback(tmp_path, row):
    coarse = structured_mesh(2, 1)
    fine = structured_mesh(4, 2)
    registry = build_transport_surface_registry(coarse, boundary_tags=tags(coarse))
    fine_registry = build_transport_surface_registry(fine, boundary_tags=tags(fine))
    state = initial_transport_state(coarse, temperature_K=1400.0, row=row, registry=registry,
                                    vacancy_fraction_gp=np.full(coarse.ne, 1e-5))
    def fail(name, _trial):
        raise RuntimeError("injected-"+name)
    with pytest.raises(RuntimeError, match="remesh_transfer"):
        transfer_transport_state_overlap(
            state, coarse, fine, fine_registry, failure_injector=fail,
        )
    with pytest.raises(RuntimeError, match="topology_commit"):
        rebind_registry_after_accepted_topology(state, fine_registry, failure_injector=fail)
    target = tmp_path/"not-published.json"
    with pytest.raises(RuntimeError, match="checkpoint_publication"):
        write_transport_checkpoint(state, registry, target, failure_injector=fail)
    assert not target.exists()
    assert np.array_equal(state.vacancy_fraction_gp, np.full(coarse.ne, 1e-5))
