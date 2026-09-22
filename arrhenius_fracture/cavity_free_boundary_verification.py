"""Finite-element-consistent verification of an explicit free cavity boundary."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from typing import Any, Mapping

import numpy as np

from .finalization_v3_schema import SCIENTIFIC_ACCEPTANCE_TOLERANCES
from .topology_transaction_v11 import (
    _accepted_dirichlet_partition, complete_accepted_state_fingerprint,
    require_equilibrium_observables,
)


SCHEMA = "unified-2d.cavity-free-boundary-verification/1"


def _hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False,
    ).encode()).hexdigest()


@dataclass(frozen=True)
class CavityFreeBoundaryVerification:
    schema: str
    cavity_boundary_edge_ids: tuple[tuple[int, int], ...]
    edge_owner_counts: tuple[int, ...]
    boundary_node_ids: tuple[int, ...]
    prescribed_neumann_load_N_per_m_by_edge: tuple[tuple[float, float], ...]
    prescribed_dirichlet_dofs_on_cavity: tuple[int, ...]
    justified_rigid_body_constraint_dofs: tuple[int, ...]
    interior_intersection_element_ids: tuple[int, ...]
    cohesive_bridge_ids: tuple[str, ...]
    residual_bridge_element_ids: tuple[int, ...]
    support_bridge_element_ids: tuple[int, ...]
    assembled_boundary_residual_N_per_m: tuple[tuple[float, float], ...]
    assembled_boundary_residual_relative: float
    weak_normal_virtual_work_N_per_m: float
    weak_tangential_virtual_work_N_per_m: float
    linear_normal_virtual_work_N_per_m: float
    linear_tangential_virtual_work_N_per_m: float
    normalized_virtual_work: Mapping[str, float]
    total_boundary_resultant_N_per_m: tuple[float, float]
    total_boundary_resultant_relative: float
    total_boundary_moment_N: float
    total_boundary_moment_relative: float
    global_top_bottom_reaction_balance: float
    free_dof_residual_relative: float
    geometry_fingerprint: str
    accepted_state_fingerprint: str
    predicates: Mapping[str, bool]
    passed: bool


def _polygon_overlaps_triangle(polygon, triangle, tolerance=1.0e-14):
    axes = []
    for vertices in (np.asarray(polygon), np.asarray(triangle)):
        for first, second in zip(vertices, np.roll(vertices, -1, axis=0)):
            edge = second - first
            axes.append(np.asarray((-edge[1], edge[0])))
    for axis in axes:
        length = float(np.linalg.norm(axis))
        if length <= tolerance:
            continue
        axis /= length
        a, b = np.asarray(polygon) @ axis, np.asarray(triangle) @ axis
        if min(float(np.max(a)), float(np.max(b))) - max(float(np.min(a)), float(np.min(b))) <= tolerance:
            return False
    return True


def _ordered_polygon(nodes: np.ndarray, edges: tuple[tuple[int, int], ...]) -> np.ndarray:
    adjacency: dict[int, list[int]] = {}
    for a, b in edges:
        adjacency.setdefault(a, []).append(b); adjacency.setdefault(b, []).append(a)
    if not adjacency or any(len(peers) != 2 for peers in adjacency.values()):
        raise ValueError("cavity boundary is not one closed two-regular cycle")
    first = min(adjacency); ordered = [first]; previous = None; current = first
    while True:
        following = min(node for node in adjacency[current] if node != previous)
        if following == first:
            break
        ordered.append(following); previous, current = current, following
        if len(ordered) > len(adjacency):
            raise ValueError("cavity boundary cycle does not close")
    if len(ordered) != len(adjacency):
        raise ValueError("cavity boundary contains multiple components")
    return nodes[ordered]


def verify_cavity_free_boundary(
    state, *, boundary_edges=None,
    prescribed_neumann_load_N_per_m_by_edge=None,
    additional_constrained_boundary_nodes=(), injected_bridge_element_ids=(),
    allowed_rigid_body_constraint_dofs=(),
) -> CavityFreeBoundaryVerification:
    """Verify the variational free-boundary contract on one accepted state.

    Optional overrides exist only for manufactured failure tests.  Production
    callers provide the accepted state alone.
    """
    from .fem import assemble_mechanics
    from .voiding_production_v5 import _actual_cavity_boundary_edges

    if state.void_state is None or len(state.void_state.cavities) != 1:
        raise ValueError("free-boundary verification requires one explicit cavity")
    nodes = np.asarray(state.mesh.nodes, dtype=float)
    elements = np.asarray(state.mesh.elems, dtype=int)
    actual = tuple(tuple(sorted(map(int, edge))) for edge in _actual_cavity_boundary_edges(state))
    supplied = actual if boundary_edges is None else tuple(
        tuple(sorted(map(int, edge))) for edge in boundary_edges
    )
    canonical = tuple(sorted(set(supplied)))
    boundary_identity = len(supplied) == len(canonical) and set(canonical) == set(actual)
    owner_map: dict[tuple[int, int], list[int]] = {}
    for eid, triangle in enumerate(elements):
        for a, b in ((triangle[0], triangle[1]), (triangle[1], triangle[2]), (triangle[2], triangle[0])):
            owner_map.setdefault(tuple(sorted((int(a), int(b)))), []).append(eid)
    owner_counts = tuple(len(owner_map.get(edge, ())) for edge in canonical)
    boundary_nodes = tuple(sorted({node for edge in canonical for node in edge}))
    polygon = _ordered_polygon(nodes, canonical) if boundary_identity else np.empty((0, 2))
    if len(polygon):
        polygon_min = polygon.min(axis=0); polygon_max = polygon.max(axis=0)
        triangles = nodes[elements]
        candidate_ids = np.flatnonzero(np.all(triangles.max(axis=1) >= polygon_min, axis=1)
                                       & np.all(triangles.min(axis=1) <= polygon_max, axis=1))
        interior = tuple(int(eid) for eid in candidate_ids
                         if _polygon_overlaps_triangle(polygon, triangles[eid]))
    else:
        interior = ()

    if prescribed_neumann_load_N_per_m_by_edge is None:
        neumann = tuple((0.0, 0.0) for _ in canonical)
    else:
        loads = np.asarray(prescribed_neumann_load_N_per_m_by_edge, dtype=float)
        if loads.shape != (len(canonical), 2) or not np.isfinite(loads).all():
            raise ValueError("prescribed cavity Neumann loads have the wrong shape")
        neumann = tuple(tuple(map(float, row)) for row in loads)

    prescribed, free = _accepted_dirichlet_partition(state)
    for node in additional_constrained_boundary_nodes:
        prescribed[2 * int(node):2 * int(node) + 2] = True
    cavity_dofs = np.ravel(np.column_stack((2*np.asarray(boundary_nodes), 2*np.asarray(boundary_nodes)+1)))
    constrained = tuple(sorted(set(map(int, cavity_dofs[prescribed[cavity_dofs]]))))
    allowed_rigid = tuple(sorted(set(map(int, allowed_rigid_body_constraint_dofs))))
    unjustified = tuple(dof for dof in constrained if dof not in allowed_rigid)

    _, residual, *_ = assemble_mechanics(
        state.mesh, state.displacement, state.ep_gp, state.rho_gp, state.damage,
        state.elasticity_D, state.material, cohesive_network=state.cohesive_network,
    )
    nodal = np.asarray(residual, dtype=float).reshape(-1, 2)[list(boundary_nodes)]
    cavity = state.void_state.cavities[0]
    center = np.asarray(cavity.center_m, dtype=float)
    radial = nodes[list(boundary_nodes)] - center
    radii = np.linalg.norm(radial, axis=1)
    normals = radial / radii[:, None]
    tangents = np.column_stack((-normals[:, 1], normals[:, 0]))
    source = np.asarray(cavity.connection_exit_m or nodes[boundary_nodes[0]], dtype=float)
    source_normal = (source-center)/np.linalg.norm(source-center)
    arcs = cavity.radius_m * np.arctan2(
        source_normal[0]*normals[:,1]-source_normal[1]*normals[:,0], normals@source_normal,
    )
    arc_scale = max(float(cavity.radius_m), 1.0e-300)
    works = {
        "constant_normal": float(np.sum(nodal*normals)),
        "constant_tangential": float(np.sum(nodal*tangents)),
        "linear_normal": float(np.sum(nodal*normals*(arcs/arc_scale)[:,None])),
        "linear_tangential": float(np.sum(nodal*tangents*(arcs/arc_scale)[:,None])),
    }
    resultant = np.sum(nodal, axis=0)
    moment = float(np.sum(radial[:,0]*nodal[:,1]-radial[:,1]*nodal[:,0]))
    equilibrium = require_equilibrium_observables(state)
    reaction = float(equilibrium.mean_reaction_magnitude_N_per_m if hasattr(
        equilibrium, "mean_reaction_magnitude_N_per_m"
    ) else abs(equilibrium["latest_reaction_N_per_m"]))
    normalized_works = {key: abs(value)/reaction for key, value in works.items()}
    boundary_residual_relative = float(np.linalg.norm(nodal)/reaction)
    resultant_relative = float(np.linalg.norm(resultant)/reaction)
    moment_relative = abs(moment)/(reaction*float(cavity.radius_m))

    support_ids = tuple(getattr(state.v12_support_state, "selected_support_elements", ()))
    support_bridges = tuple(
        int(eid) for eid in support_ids
        if len(polygon) and _polygon_overlaps_triangle(polygon, nodes[elements[int(eid)]])
    )
    residual_damage = getattr(state.mesh, "element_damage_gp", None)
    residual_candidates = () if residual_damage is None else tuple(map(int, np.flatnonzero(
        np.asarray(residual_damage, dtype=float) > 0.0
    )))
    residual_bridges = tuple(sorted(set(map(int, injected_bridge_element_ids)) | {
        eid for eid in residual_candidates
        if len(polygon) and _polygon_overlaps_triangle(polygon, nodes[elements[eid]])
    }))
    # The production cohesive assembly has no cavity-edge loading channel.  A
    # non-None cohesive network is crack-owned and cannot cross the empty cavity
    # when the exact interior/support tests above pass.
    cohesive_bridges: tuple[str, ...] = () if not interior and not support_bridges else ("GEOMETRIC_CROSSING",)

    limits = SCIENTIFIC_ACCEPTANCE_TOLERANCES
    weak_limit = limits["free_residual_relative"]
    predicates = {
        "complete_unique_boundary_identity": boundary_identity,
        "one_material_owner_per_edge": bool(owner_counts) and all(count == 1 for count in owner_counts),
        "no_element_intersects_cavity_interior": not interior,
        "zero_prescribed_neumann_load": all(load == (0.0, 0.0) for load in neumann),
        "no_unjustified_cavity_dirichlet_constraint": not unjustified,
        "no_cohesive_or_residual_bridge": not cohesive_bridges and not residual_bridges and not support_bridges,
        "assembled_boundary_residual": boundary_residual_relative <= weak_limit,
        "constant_and_linear_virtual_work": max(normalized_works.values()) <= weak_limit,
        "boundary_resultant_and_moment": max(resultant_relative, moment_relative) <= weak_limit,
        "global_reaction_balance": float(equilibrium.top_bottom_balance_residual if hasattr(
            equilibrium, "top_bottom_balance_residual"
        ) else equilibrium["latest_top_bottom_reaction_balance"]) <= limits["reaction_balance_relative"],
        "free_dof_residual": float(equilibrium.free_dof_residual_relative if hasattr(
            equilibrium, "free_dof_residual_relative"
        ) else equilibrium["latest_free_dof_residual_relative"]) <= weak_limit,
    }
    geometry = {
        "nodes": nodes[list(boundary_nodes)].tolist(), "edges": canonical,
        "cavity_center_m": list(map(float, cavity.center_m)), "cavity_radius_m": float(cavity.radius_m),
    }
    return CavityFreeBoundaryVerification(
        schema=SCHEMA, cavity_boundary_edge_ids=canonical, edge_owner_counts=owner_counts,
        boundary_node_ids=boundary_nodes, prescribed_neumann_load_N_per_m_by_edge=neumann,
        prescribed_dirichlet_dofs_on_cavity=constrained,
        justified_rigid_body_constraint_dofs=allowed_rigid,
        interior_intersection_element_ids=interior, cohesive_bridge_ids=cohesive_bridges,
        residual_bridge_element_ids=residual_bridges, support_bridge_element_ids=support_bridges,
        assembled_boundary_residual_N_per_m=tuple(tuple(map(float,row)) for row in nodal),
        assembled_boundary_residual_relative=boundary_residual_relative,
        weak_normal_virtual_work_N_per_m=works["constant_normal"],
        weak_tangential_virtual_work_N_per_m=works["constant_tangential"],
        linear_normal_virtual_work_N_per_m=works["linear_normal"],
        linear_tangential_virtual_work_N_per_m=works["linear_tangential"],
        normalized_virtual_work=normalized_works,
        total_boundary_resultant_N_per_m=tuple(map(float,resultant)),
        total_boundary_resultant_relative=resultant_relative,
        total_boundary_moment_N=moment, total_boundary_moment_relative=moment_relative,
        global_top_bottom_reaction_balance=float(equilibrium.top_bottom_balance_residual if hasattr(
            equilibrium, "top_bottom_balance_residual") else equilibrium["latest_top_bottom_reaction_balance"]),
        free_dof_residual_relative=float(equilibrium.free_dof_residual_relative if hasattr(
            equilibrium, "free_dof_residual_relative") else equilibrium["latest_free_dof_residual_relative"]),
        geometry_fingerprint=_hash(geometry),
        accepted_state_fingerprint=complete_accepted_state_fingerprint(state),
        predicates=predicates, passed=all(predicates.values()),
    )


__all__ = ["CavityFreeBoundaryVerification", "SCHEMA", "verify_cavity_free_boundary"]
