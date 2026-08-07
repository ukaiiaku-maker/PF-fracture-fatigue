from dataclasses import dataclass, replace
from types import SimpleNamespace

import numpy as np

from arrhenius_fracture.causal_sharp_wake_v11 import (
    CRACK_REPRESENTATION,
    apply_causal_segment,
    causal_segment_support,
)
from arrhenius_fracture.crack_backend import SharpWakeBackend
from arrhenius_fracture.crack_network_v11 import CrackNetworkState
from arrhenius_fracture.coalescence import segment_intersection_first
from arrhenius_fracture.topology_transaction_v11 import LiveFEMTopologyState


@dataclass(frozen=True)
class FixtureMesh:
    nodes: np.ndarray
    elems: np.ndarray
    area_e: np.ndarray
    element_damage_gp: np.ndarray


def strip_mesh():
    nodes = np.array([
        [0.0, -1.0], [0.0, 0.0], [0.0, 1.0],
        [1.0, -1.0], [1.0, 0.0], [1.0, 1.0],
        [2.0, -1.0], [2.0, 0.0], [2.0, 1.0],
        [3.0, -1.0], [3.0, 0.0], [3.0, 1.0],
    ])
    elems = np.array([
        [0, 3, 1], [1, 3, 4], [1, 4, 2], [2, 4, 5],
        [3, 6, 4], [4, 6, 7], [4, 7, 5], [5, 7, 8],
        [6, 9, 7], [7, 9, 10], [7, 10, 8], [8, 10, 11],
    ])
    return FixtureMesh(nodes, elems, np.full(len(elems), 0.5), np.zeros(len(elems)))


def state():
    mesh = strip_mesh()
    return LiveFEMTopologyState(
        mesh=mesh, boundary={}, damage=np.zeros(mesh.nodes.shape[0]),
        displacement=np.zeros(2 * mesh.nodes.shape[0]), ep_gp=np.zeros((3, len(mesh.elems))),
        rho_gp=np.ones(len(mesh.elems)), elasticity_D=np.eye(3), material={},
        cohesive_network=None,
        crack_network=CrackNetworkState.one_tip(((0.0, 0.0), (1.0, 0.0))),
        competition=SimpleNamespace(), tip_process_state={}, junction_process_state={},
        energy_ledgers={}, rng_state={}, event_counters={}, stored_energy_J_per_m=0.0,
    )


def test_causal_selector_has_no_endpoint_only_forward_kill():
    mesh = strip_mesh()
    selected, lengths = causal_segment_support(mesh, np.array([0.0, 0.0]), np.array([1.0, 0.0]))
    assert selected.size > 0
    assert np.all(lengths > 0.0)
    # Every selected triangle has vertices at or behind the advancing endpoint.
    assert np.all(np.min(mesh.nodes[mesh.elems[selected], 0], axis=1) < 1.0)
    assert not np.any(np.all(mesh.nodes[mesh.elems[selected], 0] >= 1.0, axis=1))


def test_repeated_advances_always_change_additional_p0_stiffness():
    accepted = state()
    first, audit1 = apply_causal_segment(accepted, np.array([0.0, 0.0]), np.array([1.0, 0.0]))
    second, audit2 = apply_causal_segment(first, np.array([1.0, 0.0]), np.array([2.0, 0.0]))
    third, audit3 = apply_causal_segment(second, np.array([2.0, 0.0]), np.array([3.0, 0.0]))
    assert CRACK_REPRESENTATION == "sharp_wake_causal_v11"
    assert audit1.mechanically_resolved
    assert audit2.mechanically_resolved
    assert audit3.mechanically_resolved
    assert audit1.newly_degraded_element_count > 0
    assert audit2.newly_degraded_element_count > 0
    assert audit3.newly_degraded_element_count > 0
    assert np.count_nonzero(third.mesh.element_damage_gp) == sum(
        audit.newly_degraded_element_count for audit in (audit1, audit2, audit3)
    )


def test_nodal_visual_projection_does_not_propagate_p0_stiffness():
    accepted = state()
    trial, audit = apply_causal_segment(
        accepted, np.array([0.1, -0.2]), np.array([0.9, 0.2])
    )
    selected = np.asarray(audit.selected_element_ids)
    nodal_neighbours = np.flatnonzero(
        np.any(np.isin(trial.mesh.elems, np.unique(trial.mesh.elems[selected])), axis=1)
    )
    halo_only = np.setdiff1d(nodal_neighbours, selected)
    assert halo_only.size > 0
    assert np.all(trial.mesh.element_damage_gp[halo_only] == 0.0)


def test_old_radius_halo_kills_more_than_causal_support():
    accepted = state()
    p0, p1 = np.array([0.1, -0.2]), np.array([0.9, 0.2])
    causal, audit = apply_causal_segment(accepted, p0, p1)
    old = SharpWakeBackend().advance(
        mesh=accepted.mesh, boundary=accepted.boundary, damage=accepted.damage,
        displacement=accepted.displacement, p0=p0, p1=p1, kill_r=1.0,
    )
    assert np.count_nonzero(old.mesh.element_damage_gp) > np.count_nonzero(
        causal.mesh.element_damage_gp
    )
    assert audit.newly_degraded_element_count == np.count_nonzero(
        causal.mesh.element_damage_gp
    )


def test_collinear_existing_crack_spanning_candidate_is_a_physical_hit():
    hit = segment_intersection_first(
        np.array([1.0, 0.0]), np.array([2.0, 0.0]),
        np.array([0.0, 0.0]), np.array([3.0, 0.0]),
    )
    assert hit is not None
    assert 0.0 < hit[0] < 1.0e-6


def test_valid_submicron_triangle_is_not_misclassified_as_degenerate():
    mesh = FixtureMesh(
        nodes=np.array([[0.0, 0.0], [2.0e-9, 0.0], [0.0, 1.0e-9]]),
        elems=np.array([[0, 1, 2]]), area_e=np.array([1.0e-18]),
        element_damage_gp=np.zeros(1),
    )
    selected, represented = causal_segment_support(
        mesh, np.array([0.1e-9, 0.1e-9]), np.array([1.0e-9, 0.1e-9])
    )
    np.testing.assert_array_equal(selected, [0])
    assert represented[0] > 0.0
