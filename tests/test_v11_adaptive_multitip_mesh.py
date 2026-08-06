from dataclasses import replace

import numpy as np

from arrhenius_fracture.adaptive_multitip_mesh_v11 import (
    mark_multitip_trial_support, mesh_fingerprint, refine_accepted_state,
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
