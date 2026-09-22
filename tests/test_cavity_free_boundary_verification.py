from dataclasses import replace

import pytest

from arrhenius_fracture.cavity_free_boundary_verification import (
    CavityFreeBoundaryVerification,
    verify_cavity_free_boundary,
)


@pytest.fixture(scope="module")
def connected_dbtt_state():
    from arrhenius_fracture.unified_fracture_material_v5 import material_bundle
    from arrhenius_fracture.voiding_production_v5 import (
        deterministic_trajectory,
        equilibrate_fixed_load_with_production_fem,
        ligament_transaction,
    )

    state, _ = deterministic_trajectory(
        bundle=material_bundle("DBTT"), stop_before_ligament=True,
    )
    state = equilibrate_fixed_load_with_production_fem(
        replace(state, displacement=2.0 * state.displacement)
    )
    state, result = ligament_transaction(state)
    assert result.accepted
    return state


def test_traction_free_hole_passes_weak_contract(connected_dbtt_state):
    result = verify_cavity_free_boundary(connected_dbtt_state)
    assert isinstance(result, CavityFreeBoundaryVerification)
    assert result.passed
    assert all(result.predicates.values())


def test_intentionally_applied_cavity_traction_fails(connected_dbtt_state):
    baseline = verify_cavity_free_boundary(connected_dbtt_state)
    loads = [(1.0, 0.0)] + [(0.0, 0.0)] * (len(baseline.cavity_boundary_edge_ids) - 1)
    result = verify_cavity_free_boundary(
        connected_dbtt_state,
        prescribed_neumann_load_N_per_m_by_edge=loads,
    )
    assert not result.passed
    assert not result.predicates["zero_prescribed_neumann_load"]


def test_accidentally_constrained_cavity_node_fails(connected_dbtt_state):
    baseline = verify_cavity_free_boundary(connected_dbtt_state)
    result = verify_cavity_free_boundary(
        connected_dbtt_state,
        additional_constrained_boundary_nodes=(baseline.boundary_node_ids[0],),
    )
    assert not result.passed
    assert not result.predicates["no_unjustified_cavity_dirichlet_constraint"]


def test_residual_bridge_across_cavity_fails(connected_dbtt_state):
    result = verify_cavity_free_boundary(
        connected_dbtt_state, injected_bridge_element_ids=(0,),
    )
    assert not result.passed
    assert result.residual_bridge_element_ids == (0,)
    assert not result.predicates["no_cohesive_or_residual_bridge"]


@pytest.mark.parametrize("mutation", ("duplicate", "missing"))
def test_duplicated_or_missing_cavity_edge_fails(connected_dbtt_state, mutation):
    baseline = verify_cavity_free_boundary(connected_dbtt_state)
    edges = baseline.cavity_boundary_edge_ids
    supplied = edges + edges[:1] if mutation == "duplicate" else edges[1:]
    result = verify_cavity_free_boundary(connected_dbtt_state, boundary_edges=supplied)
    assert not result.passed
    assert not result.predicates["complete_unique_boundary_identity"]


def test_normal_and_edge_order_reversal_are_invariant(connected_dbtt_state):
    baseline = verify_cavity_free_boundary(connected_dbtt_state)
    reversed_edges = tuple((b, a) for a, b in reversed(baseline.cavity_boundary_edge_ids))
    result = verify_cavity_free_boundary(connected_dbtt_state, boundary_edges=reversed_edges)
    assert result == baseline
