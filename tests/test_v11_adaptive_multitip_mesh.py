from dataclasses import replace

import numpy as np

from arrhenius_fracture.adaptive_multitip_mesh_v11 import (
    _subdivide, active_tip_hbar, adapt_accepted_state_for_trials,
    diagnose_underresolved_trial_geometry,
    mark_multitip_trial_support, mark_underresolved_trial_geometry,
    mesh_fingerprint, refine_accepted_state,
)
from arrhenius_fracture.config import ElasticProperties
from arrhenius_fracture.crack_network_v11 import CrackBranchState, CrackNetworkState
from arrhenius_fracture.directional_competition_v11 import DirectionalCompetitionState, tungsten_cleavage_candidates
from arrhenius_fracture.fem import assemble_mechanics, plane_strain_D
from arrhenius_fracture.mesh import BoundaryData, rebuild_tri_mesh
from arrhenius_fracture.topology_transaction_v11 import LiveFEMTopologyState


def fixture_state(damage_value=0.0):
    nodes = np.array(((0., 0.), (1., 0.), (1., 1.), (0., 1.)))
    elems = np.array(((0, 1, 2), (0, 2, 3)))
    mesh = rebuild_tri_mesh(nodes, elems, tip_centers=((0.75, 0.5),))
    boundary = BoundaryData(np.array((2, 3)), np.array((0, 1)), 0, 1, np.array((0,)))
    branch = CrackBranchState("b00000000", None, 0, 0, ((0., .5), (.75, .5)), (0.,))
    network = CrackNetworkState((branch,))
    candidates = tungsten_cleavage_candidates(theta_deg=45)
    competition = DirectionalCompetitionState.initialize(candidates, global_hazard_seed=9)
    # Exact affine P1 field.
    uv = np.column_stack((2 * nodes[:, 0] - nodes[:, 1], 3 * nodes[:, 1] + nodes[:, 0]))
    mat = ElasticProperties(); D = plane_strain_D(mat)
    state = LiveFEMTopologyState(
        mesh, boundary, np.full(mesh.nn, damage_value), uv.reshape(-1),
        np.array(((0.01, 0.02), (0.03, 0.04), (0.005, 0.006))),
        np.array((4., 7.)), D, mat, None, network, competition,
        {}, {}, {}, {}, {}, 0.0,
    )
    return state, candidates


def energy(state):
    *_, psi = assemble_mechanics(
        state.mesh, state.displacement, state.ep_gp, state.rho_gp,
        state.damage, state.elasticity_D, state.material,
    )
    return float(np.sum(psi * state.mesh.area_e))


def test_nested_prolongation_preserves_old_nodes_parent_histories_and_energy():
    state, _ = fixture_state()
    refined, lineage = refine_accepted_state(
        state, marked_parent_elements=(0,), active_tip_ids=("b00000000",), generation=1, operation_index=1,
    )
    assert np.array_equal(refined.mesh.nodes[:state.mesh.nn], state.mesh.nodes)
    assert np.array_equal(refined.displacement.reshape(-1, 2)[:state.mesh.nn], state.displacement.reshape(-1, 2))
    for node, (a, b, wa, wb) in lineage.new_node_parent_interpolation.items():
        expected = wa * state.displacement.reshape(-1, 2)[a] + wb * state.displacement.reshape(-1, 2)[b]
        assert np.array_equal(refined.displacement.reshape(-1, 2)[node], expected)
    for parent, children in lineage.parent_to_child_element_map.items():
        assert np.all(refined.ep_gp[:, children] == state.ep_gp[:, parent, None])
        assert np.all(refined.rho_gp[list(children)] == state.rho_gp[parent])
        assert np.isclose(np.sum(refined.mesh.area_e[list(children)]), state.mesh.area_e[parent])
    assert np.isclose(energy(refined), energy(state), rtol=2e-15, atol=1e-12)


def test_intact_and_fully_cracked_parent_fields_are_inherited_exactly():
    for value in (0.0, 1.0):
        state, _ = fixture_state(value)
        refined, _ = refine_accepted_state(state, marked_parent_elements=(0, 1), active_tip_ids=("b00000000",), generation=1, operation_index=1)
        assert np.all(refined.damage == value)
        assert np.isclose(energy(refined), energy(state), rtol=2e-15, atol=1e-12)


def test_mixed_nodal_crack_band_uses_exact_parent_material_inheritance():
    state, _ = fixture_state()
    state = replace(state, damage=np.array((1.0, 1.0, 0.0, 0.0)))
    refined, lineage = refine_accepted_state(
        state, marked_parent_elements=(0, 1), active_tip_ids=("b00000000",),
        generation=1, operation_index=1,
    )
    parent_damage = np.mean(state.damage[state.mesh.elems], axis=1)
    for parent, children in lineage.parent_to_child_element_map.items():
        assert np.all(refined.mesh.element_damage_gp[list(children)] == parent_damage[parent])
    assert np.isclose(energy(refined), energy(state), rtol=2e-15, atol=1e-12)


def test_boundary_edge_midpoints_inherit_classification():
    state, _ = fixture_state()
    refined, lineage = refine_accepted_state(state, marked_parent_elements=(0, 1), active_tip_ids=("b00000000",), generation=1, operation_index=1)
    labels = lineage.boundary_inheritance_map
    assert sum(label == "top" for label in labels.values()) == 1
    assert sum(label == "bottom" for label in labels.values()) == 1
    assert set(state.boundary.top_nodes).issubset(set(refined.boundary.top_nodes))
    assert set(state.boundary.bot_nodes).issubset(set(refined.boundary.bot_nodes))


def test_refinement_and_marked_union_are_order_independent():
    state, candidates = fixture_state()
    by_tip_a = {"b00000000": tuple(candidates)}
    by_tip_b = {"b00000000": tuple(reversed(candidates))}
    marked_a = mark_multitip_trial_support(state.mesh, state.crack_network, by_tip_a, da_phys_m=.1, contour_radius_m=.15, crack_band_radius_m=.02)
    marked_b = mark_multitip_trial_support(state.mesh, state.crack_network, by_tip_b, da_phys_m=.1, contour_radius_m=.15, crack_band_radius_m=.02)
    assert marked_a == marked_b
    one, lineage_one = refine_accepted_state(state, marked_parent_elements=marked_a, active_tip_ids=("b00000000",), generation=1, operation_index=1)
    two, lineage_two = refine_accepted_state(state, marked_parent_elements=reversed(marked_a), active_tip_ids=("b00000000",), generation=1, operation_index=1)
    assert mesh_fingerprint(one.mesh) == mesh_fingerprint(two.mesh)
    assert lineage_one.current_mesh_fingerprint == lineage_two.current_mesh_fingerprint


def test_longest_edge_closure_is_deterministic_and_splits_the_controlling_edge():
    nodes = np.array(((0.0, 0.0), (8.0, 0.0), (0.1, 0.01), (8.0, 1.0)))
    elems = np.array(((0, 1, 2), (1, 3, 2)))
    first = _subdivide(nodes, elems, (0,), longest_edge_closure=True)
    second = _subdivide(nodes, elems, reversed((0,)), longest_edge_closure=True)
    np.testing.assert_array_equal(first[0], second[0])
    np.testing.assert_array_equal(first[1], second[1])
    # The midpoint of the marked triangle's longest edge (0, 1) must exist.
    midpoint = first[2]
    assert (0, 1) in midpoint
    np.testing.assert_array_equal(first[0][midpoint[(0, 1)]], (4.0, 0.0))


def test_longest_edge_refinement_preserves_stage_a_energy_exactly():
    state, _ = fixture_state()
    refined, _ = refine_accepted_state(
        state, marked_parent_elements=(0,), active_tip_ids=("b00000000",),
        generation=1, operation_index=1, longest_edge_closure=True,
    )
    assert np.isclose(energy(refined), energy(state), rtol=2e-15, atol=1e-12)


def test_local_resolution_contract_ignores_long_thin_nonintersecting_edge():
    state, candidates = fixture_state()
    # A vanishing-area conformity sliver has a huge global edge, but it is away
    # from both the physical proposal and the 0.15-radius J patches.
    nodes = np.vstack((state.mesh.nodes, ((4.0, 4.0), (14.0, 4.0), (4.0, 4.000001))))
    elems = np.vstack((state.mesh.elems, ((4, 5, 6),)))
    mesh = rebuild_tri_mesh(nodes, elems, tip_centers=((0.75, 0.5),))
    marked = mark_underresolved_trial_geometry(
        mesh, state.crack_network, {"b00000000": tuple(candidates)},
        da_phys_m=0.1, contour_radius_m=0.15, target_resolution_m=0.2,
    )
    assert mesh.ne - 1 not in marked


def test_local_resolution_contract_is_candidate_enumeration_invariant():
    state, candidates = fixture_state()
    first = mark_underresolved_trial_geometry(
        state.mesh, state.crack_network, {"b00000000": tuple(candidates)},
        da_phys_m=0.1, contour_radius_m=0.15, target_resolution_m=0.2,
    )
    second = mark_underresolved_trial_geometry(
        state.mesh, state.crack_network, {"b00000000": tuple(reversed(candidates))},
        da_phys_m=0.1, contour_radius_m=0.15, target_resolution_m=0.2,
    )
    assert first == second


def test_fixed_physical_tip_hbar_is_not_diluted_by_global_element_count():
    state, _ = fixture_state()
    expected = active_tip_hbar(state, contour_radius_m=0.3)
    nodes = state.mesh.nodes.tolist()
    elems = state.mesh.elems.tolist()
    for index in range(30):
        base = len(nodes); x = 10.0 + index
        nodes.extend(((x, 10.0), (x + 0.1, 10.0), (x, 10.1)))
        elems.append((base, base + 1, base + 2))
    mesh = rebuild_tri_mesh(np.asarray(nodes), np.asarray(elems), tip_centers=((0.75, 0.5),))
    expanded = replace(
        state, mesh=mesh, damage=np.pad(state.damage, (0, mesh.nn - state.mesh.nn)),
        displacement=np.pad(state.displacement, (0, 2 * (mesh.nn - state.mesh.nn))),
        ep_gp=np.pad(state.ep_gp, ((0, 0), (0, mesh.ne - state.mesh.ne))),
        rho_gp=np.pad(state.rho_gp, (0, mesh.ne - state.mesh.ne)),
    )
    assert active_tip_hbar(expanded, contour_radius_m=0.3) == expected


def test_reason_resolved_marks_are_local_and_below_threshold_children_are_not_remarked():
    state, candidates = fixture_state()
    audit = diagnose_underresolved_trial_geometry(
        state.mesh, state.crack_network, {"b00000000": tuple(candidates)},
        da_phys_m=0.1, contour_radius_m=0.15, target_resolution_m=0.2,
    )
    assert audit.marked_element_ids
    assert all(record.controlling_metric_m > record.threshold_m for record in audit.records)
    assert all(record.distance_to_current_tip_m <= 0.3 for record in audit.records)


def test_final_mesh_is_validated_after_last_permitted_refinement_level():
    state, candidates = fixture_state()
    # The fixture already satisfies this deliberately loose local contract, so
    # maximum_levels=0 exercises the same post-loop validation path used after
    # a real eighth and final refinement.
    adapted, audit = adapt_accepted_state_for_trials(
        state, {"b00000000": tuple(candidates)}, da_phys_m=0.1,
        tip_h_fine_m=1.0, contour_radius_m=0.3,
        crack_band_radius_m=0.0, accepted_load_m=0.0,
        maximum_levels=0,
    )
    assert adapted.mesh is state.mesh
    assert audit.lineages == ()
    assert min(audit.trial_changed_element_count.values()) > 0
