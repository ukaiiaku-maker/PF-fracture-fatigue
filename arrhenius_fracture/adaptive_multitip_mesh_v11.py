"""Nested, deterministic conforming h-refinement for accepted v11 FEM states."""
from __future__ import annotations

from dataclasses import dataclass, replace
from enum import IntFlag
import hashlib
import json
import math
from typing import Iterable, Mapping

import numpy as np

from .mesh import BoundaryData, _estimate_hbar_tip, rebuild_tri_mesh


Edge = tuple[int, int]


def _edge(a: int, b: int) -> Edge:
    return (a, b) if a < b else (b, a)


def _cross2(a: np.ndarray, b: np.ndarray) -> float:
    return float(a[0] * b[1] - a[1] * b[0])


def _oriented(nodes: np.ndarray, vertices: Iterable[int], sign: float) -> tuple[int, int, int]:
    a, b, c = tuple(int(v) for v in vertices)
    cross = _cross2(nodes[b] - nodes[a], nodes[c] - nodes[a])
    return (a, b, c) if cross * sign > 0.0 else (a, c, b)


def mesh_fingerprint(mesh) -> str:
    digest = hashlib.sha256()
    digest.update(np.ascontiguousarray(mesh.nodes).tobytes())
    digest.update(np.ascontiguousarray(mesh.elems).tobytes())
    return digest.hexdigest()


@dataclass(frozen=True)
class RefinementLineage:
    mesh_generation: int
    parent_mesh_fingerprint: str
    current_mesh_fingerprint: str
    refinement_operation_index: int
    refined_parent_element_ids: tuple[int, ...]
    parent_to_child_element_map: Mapping[int, tuple[int, ...]]
    new_node_parent_interpolation: Mapping[int, tuple[int, int, float, float]]
    boundary_inheritance_map: Mapping[int, str]
    damage_inheritance_map: Mapping[int, tuple[int, int, float, float]]
    elements_added_by_conformity: int

    def to_dict(self) -> dict:
        return json.loads(json.dumps(self, default=lambda value: value.__dict__, sort_keys=True))


@dataclass(frozen=True)
class AdaptationAudit:
    lineages: tuple[RefinementLineage, ...]
    parent_energy_J_per_m: float
    prolonged_energy_J_per_m: float
    refined_equilibrium_energy_J_per_m: float
    parent_vs_prolonged_relative_error: float
    refined_equilibrium_relative_correction: float
    reaction_prolongated_N_per_m: float
    reaction_refined_equilibrium_N_per_m: float
    active_tip_hbar_m: Mapping[str, float]
    trial_changed_element_count: Mapping[str, int]
    refinement_marking_diagnostics: Mapping[str, object] | None = None

    def to_dict(self) -> dict:
        return json.loads(json.dumps(self, default=lambda value: value.__dict__, sort_keys=True))


class MarkingReason(IntFlag):
    CANDIDATE_SEGMENT_INTERSECTION = 1
    CANDIDATE_CRACK_NORMAL_SPAN = 2
    CURRENT_TIP_J_SUPPORT_AREA = 4
    CANDIDATE_ENDPOINT_J_SUPPORT_AREA = 8
    ACTIVE_TIP_HBAR = 16
    CONFORMITY_CLOSURE = 32


class TrialVisibilityFailure(RuntimeError):
    """Fail-closed frozen-state diagnosis for an invisible topology trial."""

    def __init__(self, message: str, *, state, diagnostics: Mapping[str, object]):
        super().__init__(message)
        self.state = state
        self.diagnostics = dict(diagnostics)


@dataclass(frozen=True)
class PhysicalMarkRecord:
    element_id: int
    tip_id: str
    candidate_id: str | None
    reason_bitmask: int
    area_m2: float
    equivalent_size_m: float
    candidate_direction_projection_span_m: float | None
    candidate_normal_projection_span_m: float | None
    segment_intersection_length_m: float | None
    distance_to_current_tip_m: float
    distance_to_candidate_endpoint_m: float | None
    inside_current_tip_J_support: bool
    inside_candidate_endpoint_J_support: bool
    controlling_metric_m: float
    threshold_m: float

    @property
    def reasons(self) -> tuple[str, ...]:
        bits = MarkingReason(self.reason_bitmask)
        return tuple(reason.name.lower() for reason in MarkingReason if reason in bits)

    def to_dict(self) -> dict[str, object]:
        return {**self.__dict__, "reasons": self.reasons}


@dataclass(frozen=True)
class MarkingAudit:
    marked_element_ids: tuple[int, ...]
    records: tuple[PhysicalMarkRecord, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "marked_element_ids": list(self.marked_element_ids),
            "records": [record.to_dict() for record in self.records],
        }


@dataclass
class _NestedRefinementProgressGuard:
    """Fail closed only when local refinement ceases to improve its metrics."""
    maximum_stalled_levels: int = 3
    previous: tuple[float, float, float] | None = None
    stalled_levels: int = 0

    def observe(self, *, marked_area_m2: float, maximum_metric_m: float,
                maximum_tip_hbar_m: float) -> None:
        current = (float(marked_area_m2), float(maximum_metric_m), float(maximum_tip_hbar_m))
        if self.previous is not None:
            progress = any(
                now < before * (1.0 - 1.0e-12)
                for now, before in zip(current, self.previous)
                if before > 0.0
            )
            self.stalled_levels = 0 if progress else self.stalled_levels + 1
            if self.stalled_levels >= self.maximum_stalled_levels:
                raise RuntimeError(
                    "nested_refinement_no_measurable_progress: "
                    f"previous={self.previous!r} current={current!r} "
                    f"stalled_levels={self.stalled_levels}"
                )
        self.previous = current


def _subdivide(
    nodes: np.ndarray, elems: np.ndarray, marked: Iterable[int], *,
    longest_edge_closure: bool = False,
):
    marked_ids = tuple(sorted(set(int(index) for index in marked)))
    if any(index < 0 or index >= len(elems) for index in marked_ids):
        raise ValueError("marked parent element index is out of range")
    split_edges: set[Edge] = set()
    def longest(triangle) -> Edge:
        a, b, c = (int(v) for v in triangle)
        edges = (_edge(a, b), _edge(b, c), _edge(c, a))
        return min(
            edges,
            key=lambda item: (-float(np.linalg.norm(nodes[item[1]] - nodes[item[0]])), item),
        )

    for index in marked_ids:
        a, b, c = (int(v) for v in elems[index])
        if longest_edge_closure:
            split_edges.add(longest((a, b, c)))
        else:
            split_edges.update((_edge(a, b), _edge(b, c), _edge(c, a)))
    if longest_edge_closure:
        # Conformity closure: a triangle touched by a split edge is never bisected
        # across a shorter edge while retaining its longest edge.  This is the
        # deterministic longest-edge propagation needed to prevent arbitrarily
        # thin green-conformity children.
        changed = True
        while changed:
            changed = False
            for triangle in elems:
                a, b, c = (int(v) for v in triangle)
                edges = (_edge(a, b), _edge(b, c), _edge(c, a))
                if any(item in split_edges for item in edges):
                    edge = longest(triangle)
                    if edge not in split_edges:
                        split_edges.add(edge); changed = True
    new_nodes = [tuple(point) for point in np.asarray(nodes, dtype=float)]
    midpoint: dict[Edge, int] = {}
    interpolation: dict[int, tuple[int, int, float, float]] = {}
    for edge in sorted(split_edges):
        node_id = len(new_nodes)
        midpoint[edge] = node_id
        new_nodes.append(tuple(0.5 * (nodes[edge[0]] + nodes[edge[1]])))
        interpolation[node_id] = (edge[0], edge[1], 0.5, 0.5)
    points = np.asarray(new_nodes, dtype=float)
    children: list[tuple[int, int, int]] = []
    parent_map: dict[int, tuple[int, ...]] = {}
    conformity = 0
    for parent_id, triangle in enumerate(elems):
        a, b, c = (int(v) for v in triangle)
        sign = math.copysign(1.0, _cross2(nodes[b] - nodes[a], nodes[c] - nodes[a]))
        present = {edge for edge in (_edge(a, b), _edge(b, c), _edge(c, a)) if edge in midpoint}
        start = len(children)
        if not present:
            children.append((a, b, c))
        elif len(present) == 1:
            x, y = next(iter(present)); z = next(v for v in (a, b, c) if v not in (x, y)); m = midpoint[_edge(x, y)]
            children.extend((_oriented(points, (x, m, z), sign), _oriented(points, (m, y, z), sign)))
        elif len(present) == 2:
            edges = tuple(sorted(present))
            shared = next(v for v in edges[0] if v in edges[1])
            outer = sorted(v for edge in edges for v in edge if v != shared)
            x, y = outer; mx = midpoint[_edge(shared, x)]; my = midpoint[_edge(shared, y)]
            children.extend((
                _oriented(points, (shared, mx, my), sign),
                _oriented(points, (mx, x, y), sign),
                _oriented(points, (mx, y, my), sign),
            ))
        else:
            mab, mbc, mca = midpoint[_edge(a, b)], midpoint[_edge(b, c)], midpoint[_edge(c, a)]
            children.extend((
                _oriented(points, (a, mab, mca), sign),
                _oriented(points, (mab, b, mbc), sign),
                _oriented(points, (mca, mbc, c), sign),
                _oriented(points, (mab, mbc, mca), sign),
            ))
        parent_map[parent_id] = tuple(range(start, len(children)))
        if parent_id not in marked_ids and len(children) - start > 1:
            conformity += len(children) - start - 1
    return points, np.asarray(children, dtype=int), midpoint, interpolation, parent_map, conformity


def refine_accepted_state(
    state, *, marked_parent_elements: Iterable[int], active_tip_ids: Iterable[str],
    generation: int, operation_index: int, longest_edge_closure: bool = False,
):
    old = state.mesh
    marked = tuple(sorted(set(int(value) for value in marked_parent_elements)))
    nodes, elems, midpoint, interpolation, parent_map, conformity = _subdivide(
        old.nodes, old.elems, marked, longest_edge_closure=longest_edge_closure,
    )
    tips = tuple(sorted(set(str(value) for value in active_tip_ids)))
    centers = np.asarray([state.crack_network.branch(tip).tip for tip in tips], dtype=float)
    mesh = rebuild_tri_mesh(nodes, elems, tip_centers=centers)

    old_u = np.asarray(state.displacement).reshape(old.nn, 2)
    u = np.empty((mesh.nn, 2), dtype=float); u[:old.nn] = old_u
    damage = np.empty(mesh.nn, dtype=float); damage[:old.nn] = state.damage
    for node, (a, b, wa, wb) in interpolation.items():
        u[node] = wa * old_u[a] + wb * old_u[b]
        damage[node] = wa * state.damage[a] + wb * state.damage[b]

    ep = np.empty((state.ep_gp.shape[0], mesh.ne), dtype=float)
    rho = np.empty(mesh.ne, dtype=float)
    old_element_damage = getattr(old, "element_damage_gp", None)
    if old_element_damage is None:
        old_element_damage = np.mean(np.asarray(state.damage)[old.elems], axis=1)
    element_damage = np.empty(mesh.ne, dtype=float)
    for parent, child_ids in parent_map.items():
        ep[:, child_ids] = np.asarray(state.ep_gp)[:, parent, None]
        rho[list(child_ids)] = float(state.rho_gp[parent])
        element_damage[list(child_ids)] = float(old_element_damage[parent])
    mesh = replace(mesh, element_damage_gp=element_damage)

    inherited: dict[int, str] = {}
    old_sets = {
        "top": set(int(v) for v in state.boundary.top_nodes),
        "bottom": set(int(v) for v in state.boundary.bot_nodes),
        "notch": set(int(v) for v in state.boundary.notch_nodes),
    }
    new_sets = {name: set(values) for name, values in old_sets.items()}
    for edge, node in midpoint.items():
        labels = [name for name, values in old_sets.items() if edge[0] in values and edge[1] in values]
        for name in labels:
            new_sets[name].add(node)
        inherited[node] = "+".join(sorted(labels)) if labels else "interior"
    boundary = BoundaryData(
        top_nodes=np.asarray(sorted(new_sets["top"]), dtype=int),
        bot_nodes=np.asarray(sorted(new_sets["bottom"]), dtype=int),
        left_bot=int(state.boundary.left_bot), right_bot=int(state.boundary.right_bot),
        notch_nodes=np.asarray(sorted(new_sets["notch"]), dtype=int),
    )
    result = replace(state, mesh=mesh, boundary=boundary, displacement=u.reshape(-1), damage=damage, ep_gp=ep, rho_gp=rho)
    lineage = RefinementLineage(
        mesh_generation=int(generation), parent_mesh_fingerprint=mesh_fingerprint(old),
        current_mesh_fingerprint=mesh_fingerprint(mesh), refinement_operation_index=int(operation_index),
        refined_parent_element_ids=marked, parent_to_child_element_map=parent_map,
        new_node_parent_interpolation=interpolation, boundary_inheritance_map=inherited,
        damage_inheritance_map=interpolation, elements_added_by_conformity=conformity,
    )
    return result, lineage


def _distance_to_segment(points: np.ndarray, p0: np.ndarray, p1: np.ndarray) -> np.ndarray:
    segment = p1 - p0; length2 = float(segment @ segment)
    if length2 <= 0.0:
        return np.linalg.norm(points - p0, axis=1)
    t = np.clip(((points - p0) @ segment) / length2, 0.0, 1.0)
    return np.linalg.norm(points - (p0 + t[:, None] * segment), axis=1)


def mark_multitip_trial_support(mesh, network, candidates_by_tip, *, da_phys_m: float, contour_radius_m: float, crack_band_radius_m: float) -> tuple[int, ...]:
    from .causal_sharp_wake_v11 import causal_segment_support
    centroids = mesh.nodes[mesh.elems].mean(axis=1)
    element_radius = np.sqrt(np.maximum(mesh.area_e, 0.0))
    marked = np.zeros(mesh.ne, dtype=bool)
    support = max(float(crack_band_radius_m), 0.0)
    for tip_id in sorted(network.active_tip_ids):
        tip = np.asarray(network.branch(tip_id).tip, dtype=float)
        marked |= np.linalg.norm(centroids - tip, axis=1) <= float(contour_radius_m) + element_radius
        for candidate in sorted(candidates_by_tip[tip_id], key=lambda item: item.candidate_id):
            end = tip + float(da_phys_m) * np.asarray(candidate.direction_xy, dtype=float)
            # Centroid/radius corridor tests can miss long, low-area conformity
            # triangles.  Every element intersected by the physical proposal is
            # therefore included explicitly before the causal P0 trial.
            intersected, _ = causal_segment_support(mesh, tip, end)
            marked[intersected] = True
            marked |= _distance_to_segment(centroids, tip, end) <= support + element_radius
            # Every sibling trial must also land in an already qualified J
            # support patch.  Refining this union before A1/A2/A12 avoids a
            # post-commit discretization correction at the newly advanced tip.
            marked |= np.linalg.norm(centroids - end, axis=1) <= float(contour_radius_m) + element_radius
    return tuple(int(value) for value in np.flatnonzero(marked))


def _stored_energy_and_reaction(state) -> tuple[float, float]:
    from .fem import assemble_mechanics, elastic_energy_densities
    _, residual, sigma, *_ = assemble_mechanics(
        state.mesh, state.displacement, state.ep_gp, state.rho_gp, state.damage,
        state.elasticity_D, state.material, cohesive_network=state.cohesive_network,
    )
    density, _ = elastic_energy_densities(
        state.mesh, state.displacement, state.ep_gp, sigma, state.elasticity_D,
    )
    top = np.asarray(state.boundary.top_nodes, dtype=int)
    reaction = float(np.sum(residual[2 * top + 1])) if top.size else 0.0
    return float(np.sum(density * state.mesh.area_e)), reaction


def active_tip_hbar(state, *, contour_radius_m: float | None = None) -> dict[str, float]:
    if contour_radius_m is not None:
        centroids = state.mesh.nodes[state.mesh.elems].mean(axis=1)
        equivalent = np.sqrt(4.0 * np.maximum(state.mesh.area_e, 0.0) / math.pi)
        result = {}
        for tip in state.crack_network.active_tip_ids:
            point = np.asarray(state.crack_network.branch(tip).tip, dtype=float)
            distance = np.linalg.norm(centroids - point, axis=1)
            patch = np.flatnonzero(distance <= float(contour_radius_m))
            if not patch.size:
                patch = np.asarray((int(np.argmin(distance)),), dtype=int)
            # This is the controlling J-support local scale: every element in
            # the fixed physical patch must satisfy the same area-equivalent
            # contract used by the marker.  Unlike the legacy nearest-2%
            # statistic, its physical sampling region cannot grow with ne.
            result[tip] = float(np.max(equivalent[patch]))
        return result
    return {
        tip: _estimate_hbar_tip(
            state.mesh.nodes, state.mesh.elems, *state.crack_network.branch(tip).tip,
        )
        for tip in state.crack_network.active_tip_ids
    }


def _mean_edge_length(mesh) -> np.ndarray:
    triangles = mesh.nodes[mesh.elems]
    return (
        np.linalg.norm(triangles[:, 1] - triangles[:, 0], axis=1)
        + np.linalg.norm(triangles[:, 2] - triangles[:, 1], axis=1)
        + np.linalg.norm(triangles[:, 0] - triangles[:, 2], axis=1)
    ) / 3.0


def mark_underresolved_trial_geometry(
    mesh, network, candidates_by_tip, *, da_phys_m: float,
    contour_radius_m: float, target_resolution_m: float,
) -> tuple[int, ...]:
    """Mark only geometry that controls a causal crack/J trial.

    Candidate-crossed elements are measured by the represented tangent length
    and their exact vertex-projection span in the crack-normal direction. Tip and
    candidate-endpoint J patches are measured by element area-equivalent
    diameter, the scale controlling their integrated FEM contribution.  A long
    conformity edge with vanishing altitude therefore cannot veto an otherwise
    resolved trial merely because its global edge maximum is large.
    """
    return diagnose_underresolved_trial_geometry(
        mesh, network, candidates_by_tip, da_phys_m=da_phys_m,
        contour_radius_m=contour_radius_m,
        target_resolution_m=target_resolution_m,
    ).marked_element_ids


def diagnose_underresolved_trial_geometry(
    mesh, network, candidates_by_tip, *, da_phys_m: float,
    contour_radius_m: float, target_resolution_m: float,
) -> MarkingAudit:
    """Return deterministic, reason-resolved physical refinement marks."""
    from .causal_sharp_wake_v11 import causal_segment_support
    target = float(target_resolution_m)
    centroids = mesh.nodes[mesh.elems].mean(axis=1)
    equivalent_diameter = np.sqrt(4.0 * np.maximum(mesh.area_e, 0.0) / math.pi)
    associations: dict[tuple[int, str, str | None], dict[str, object]] = {}

    def add(
        element_id: int, tip_id: str, candidate_id: str | None,
        reason: MarkingReason, *, tip: np.ndarray, end: np.ndarray | None = None,
        tangent_span: float | None = None, normal_span: float | None = None,
        intersection_length: float | None = None, metric: float,
    ) -> None:
        element_id = int(element_id)
        key = (element_id, tip_id, candidate_id)
        endpoint_distance = None if end is None else float(np.linalg.norm(centroids[element_id] - end))
        current_distance = float(np.linalg.norm(centroids[element_id] - tip))
        row = associations.get(key)
        if row is None:
            row = {
                "element_id": element_id, "tip_id": tip_id,
                "candidate_id": candidate_id, "reason_bitmask": 0,
                "area_m2": float(mesh.area_e[element_id]),
                "equivalent_size_m": float(equivalent_diameter[element_id]),
                "candidate_direction_projection_span_m": tangent_span,
                "candidate_normal_projection_span_m": normal_span,
                "segment_intersection_length_m": intersection_length,
                "distance_to_current_tip_m": current_distance,
                "distance_to_candidate_endpoint_m": endpoint_distance,
                "inside_current_tip_J_support": current_distance <= float(contour_radius_m),
                "inside_candidate_endpoint_J_support": (
                    endpoint_distance is not None and endpoint_distance <= float(contour_radius_m)
                ),
                "controlling_metric_m": float(metric), "threshold_m": target,
            }
            associations[key] = row
        row["reason_bitmask"] = int(row["reason_bitmask"]) | int(reason)
        row["controlling_metric_m"] = max(float(row["controlling_metric_m"]), float(metric))

    for tip_id in sorted(network.active_tip_ids):
        tip = np.asarray(network.branch(tip_id).tip, dtype=float)
        current_patch = np.linalg.norm(centroids - tip, axis=1) <= float(contour_radius_m)
        for element_id in np.flatnonzero(current_patch & (equivalent_diameter > target)):
            add(element_id, tip_id, None, MarkingReason.CURRENT_TIP_J_SUPPORT_AREA,
                tip=tip, metric=float(equivalent_diameter[element_id]))
        for candidate in sorted(candidates_by_tip[tip_id], key=lambda item: item.candidate_id):
            end = tip + float(da_phys_m) * np.asarray(candidate.direction_xy, dtype=float)
            endpoint_patch = np.linalg.norm(centroids - end, axis=1) <= float(contour_radius_m)
            for element_id in np.flatnonzero(endpoint_patch & (equivalent_diameter > target)):
                add(element_id, tip_id, candidate.candidate_id,
                    MarkingReason.CANDIDATE_ENDPOINT_J_SUPPORT_AREA,
                    tip=tip, end=end, metric=float(equivalent_diameter[element_id]))
            intersected, lengths = causal_segment_support(mesh, tip, end)
            if intersected.size:
                tangent = (end - tip) / max(float(np.linalg.norm(end - tip)), np.finfo(float).tiny)
                normal = np.array((-tangent[1], tangent[0]))
                vertices = mesh.nodes[mesh.elems[intersected]]
                normal_coordinates = vertices @ normal
                tangent_coordinates = vertices @ tangent
                normal_width = np.ptp(normal_coordinates, axis=1)
                tangent_width = np.ptp(tangent_coordinates, axis=1)
                for local, element_id in enumerate(intersected):
                    if lengths[local] > target:
                        add(element_id, tip_id, candidate.candidate_id,
                            MarkingReason.CANDIDATE_SEGMENT_INTERSECTION,
                            tip=tip, end=end, tangent_span=float(tangent_width[local]),
                            normal_span=float(normal_width[local]),
                            intersection_length=float(lengths[local]), metric=float(lengths[local]))
                    if normal_width[local] > target:
                        add(element_id, tip_id, candidate.candidate_id,
                            MarkingReason.CANDIDATE_CRACK_NORMAL_SPAN,
                            tip=tip, end=end, tangent_span=float(tangent_width[local]),
                            normal_span=float(normal_width[local]),
                            intersection_length=float(lengths[local]), metric=float(normal_width[local]))
    # One element can belong both to current-tip support (candidate id None)
    # and to candidate support.  Use an explicit total ordering rather than
    # asking Python to compare None with str.
    ordered_keys = sorted(
        associations,
        key=lambda item: (item[0], item[1], item[2] is not None, item[2] or ""),
    )
    records = tuple(PhysicalMarkRecord(**associations[key]) for key in ordered_keys)
    return MarkingAudit(
        tuple(sorted({record.element_id for record in records})), records,
    )


def trial_stiffness_visibility(state, candidates_by_tip, *, da_phys_m: float, crack_band_radius_m: float) -> dict[str, int]:
    from .causal_sharp_wake_v11 import apply_causal_segment
    result: dict[str, int] = {}
    for tip_id in sorted(state.crack_network.active_tip_ids):
        tip = np.asarray(state.crack_network.branch(tip_id).tip, dtype=float)
        for candidate in sorted(candidates_by_tip[tip_id], key=lambda item: item.candidate_id):
            end = tip + float(da_phys_m) * np.asarray(candidate.direction_xy, dtype=float)
            _, audit = apply_causal_segment(state, tip, end)
            result[f"{tip_id}|{candidate.candidate_id}"] = audit.newly_degraded_element_count
    return result


def zero_visibility_reasons(state, candidates_by_tip, *, da_phys_m: float) -> dict[str, str]:
    """Classify zero-visibility proposals without changing the accepted state.

    A P0 wake element can extend beyond the physical endpoint that first
    degraded it.  Nested refinement deliberately inherits that committed
    damage, so refinement cannot create new stiffness visibility inside such
    material.  Reporting this as a resolution-marker inconsistency is both
    misleading and non-actionable.
    """
    from .causal_sharp_wake_v11 import causal_segment_support, element_damage

    damage = element_damage(state.mesh, state.damage)
    result: dict[str, str] = {}
    for tip_id in sorted(state.crack_network.active_tip_ids):
        tip = np.asarray(state.crack_network.branch(tip_id).tip, dtype=float)
        for candidate in sorted(candidates_by_tip[tip_id], key=lambda item: item.candidate_id):
            end = tip + float(da_phys_m) * np.asarray(candidate.direction_xy, dtype=float)
            selected, _ = causal_segment_support(state.mesh, tip, end)
            key = f"{tip_id}|{candidate.candidate_id}"
            if not selected.size:
                result[key] = "admissible_segment_has_no_discrete_causal_stiffness_support"
            elif np.all(damage[selected] >= 1.0):
                result[key] = "candidate_segment_already_in_committed_wake_material"
    return result


def adapt_accepted_state_for_trials(
    state, candidates_by_tip, *, da_phys_m: float, tip_h_fine_m: float,
    contour_radius_m: float, crack_band_radius_m: float, accepted_load_m: float,
    starting_generation: int = 0, starting_operation_index: int = 0,
    maximum_levels: int = 32,
):
    """Proactively refine one accepted discretization for all sibling trials."""
    from .fem import assemble_mechanics, solve_dirichlet

    target_hbar = max(float(tip_h_fine_m) * 1.5, float(da_phys_m) / 5.0)
    parent_energy, _ = _stored_energy_and_reaction(state)
    current = state
    lineages = []
    marking_levels: list[dict[str, object]] = []
    root_lineage = {element_id: element_id for element_id in range(state.mesh.ne)}
    unique_roots: set[int] = set()
    resolution_gate_passed = False
    progress_guard = _NestedRefinementProgressGuard()
    for level in range(1, int(maximum_levels) + 1):
        hbars = active_tip_hbar(current, contour_radius_m=contour_radius_m)
        marking = diagnose_underresolved_trial_geometry(
            current.mesh, current.crack_network, candidates_by_tip,
            da_phys_m=da_phys_m, contour_radius_m=contour_radius_m,
            target_resolution_m=target_hbar,
        )
        marked = marking.marked_element_ids
        reason_counts: dict[str, int] = {}
        association_counts: dict[tuple[str, str, str], int] = {}
        for record in marking.records:
            for reason in record.reasons:
                reason_counts[reason] = reason_counts.get(reason, 0) + 1
                key = (record.tip_id, record.candidate_id or "-", reason)
                association_counts[key] = association_counts.get(key, 0) + 1
        roots = {root_lineage[element_id] for element_id in marked}
        unique_roots.update(roots)
        if marked:
            points = current.mesh.nodes[current.mesh.elems[list(marked)]].reshape(-1, 2)
            bounding_box = {
                "minimum_xy_m": points.min(axis=0).tolist(),
                "maximum_xy_m": points.max(axis=0).tolist(),
            }
            marked_area = float(np.sum(current.mesh.area_e[list(marked)]))
        else:
            bounding_box = None; marked_area = 0.0
        maximum_metric = max(
            (float(record.controlling_metric_m) for record in marking.records),
            default=0.0,
        )
        marking_levels.append({
            "level": level - 1, "element_count_before": current.mesh.ne,
            "physical_mark_count": len(marked),
            "physical_marked_area_m2": marked_area,
            "marked_region_bounding_box": bounding_box,
            "unique_initial_parent_elements_affected": len(roots),
            "counts_by_reason": dict(sorted(reason_counts.items())),
            "counts_by_tip_candidate_reason": [
                {"tip_id": key[0], "candidate_id": key[1], "reason": key[2], "count": value}
                for key, value in sorted(association_counts.items())
            ],
            "records": [
                {**record.to_dict(), "parent_lineage_root_element_id": root_lineage[record.element_id]}
                for record in marking.records
            ],
        })
        visible = trial_stiffness_visibility(
            current, candidates_by_tip, da_phys_m=da_phys_m,
            crack_band_radius_m=max(float(current.mesh.hbar_tip), float(crack_band_radius_m)),
        )
        if max(hbars.values(), default=0.0) <= target_hbar and min(visible.values(), default=1) > 0 and not marked:
            resolution_gate_passed = True
            break
        if not marked:
            zero_reasons = zero_visibility_reasons(
                current, candidates_by_tip, da_phys_m=da_phys_m,
            )
            failure_diagnostics = {
                "schema": "v11.frozen-zero-visibility-diagnosis/1",
                "active_tip_hbar_m": dict(hbars),
                "target_resolution_m": target_hbar,
                "trial_changed_element_count": dict(visible),
                "zero_visibility_reasons": dict(zero_reasons),
                "final_marking": marking.to_dict(),
                "refinement_levels": marking_levels,
                "physical_time_or_rng_advanced": False,
            }
            committed_wake = sorted(
                key for key, reason in zero_reasons.items()
                if reason == "candidate_segment_already_in_committed_wake_material"
            )
            if committed_wake:
                raise TrialVisibilityFailure(
                    "candidate_segment_already_in_committed_wake_material: "
                    + ",".join(committed_wake), state=current,
                    diagnostics=failure_diagnostics,
                )
            unsupported = sorted(
                key for key, reason in zero_reasons.items()
                if reason == "admissible_segment_has_no_discrete_causal_stiffness_support"
            )
            if unsupported:
                raise TrialVisibilityFailure(
                    "candidate_segment_has_no_discrete_causal_stiffness_support: "
                    + ",".join(unsupported), state=current,
                    diagnostics=failure_diagnostics,
                )
            raise TrialVisibilityFailure(
                "active_tip_resolution_marker_inconsistency: "
                f"hbar={max(hbars.values(), default=0.0):.17g} target={target_hbar:.17g}",
                state=current, diagnostics=failure_diagnostics,
            )
        progress_guard.observe(
            marked_area_m2=marked_area,
            maximum_metric_m=maximum_metric,
            maximum_tip_hbar_m=max(hbars.values(), default=0.0),
        )
        refined, lineage = refine_accepted_state(
            current, marked_parent_elements=marked,
            active_tip_ids=current.crack_network.active_tip_ids,
            generation=starting_generation + level,
            operation_index=starting_operation_index + level,
            longest_edge_closure=True,
        )
        before, _ = _stored_energy_and_reaction(current)
        prolonged, _ = _stored_energy_and_reaction(refined)
        tolerance = 1e-12 * max(abs(before), abs(prolonged), 1.0)
        if abs(prolonged - before) > tolerance:
            raise RuntimeError(
                "nested_refinement_stage_a_energy_parity_failure: "
                f"parent={before:.17g} prolonged={prolonged:.17g}"
            )
        current = refined; lineages.append(lineage)
        next_roots = {}
        for parent, children in lineage.parent_to_child_element_map.items():
            for child in children:
                next_roots[int(child)] = root_lineage[parent]
        root_lineage = next_roots

    if not resolution_gate_passed:
        # A mesh refined on the final permitted level has not yet passed
        # through the loop header again.  Validate that resulting mesh before
        # declaring exhaustion; this is not an additional refinement level.
        final_hbars = active_tip_hbar(current, contour_radius_m=contour_radius_m)
        final_marking = diagnose_underresolved_trial_geometry(
            current.mesh, current.crack_network, candidates_by_tip,
            da_phys_m=da_phys_m, contour_radius_m=contour_radius_m,
            target_resolution_m=target_hbar,
        )
        final_visible = trial_stiffness_visibility(
            current, candidates_by_tip, da_phys_m=da_phys_m,
            crack_band_radius_m=max(float(current.mesh.hbar_tip), float(crack_band_radius_m)),
        )
        if (
            max(final_hbars.values(), default=0.0) > target_hbar
            or min(final_visible.values(), default=1) <= 0
            or final_marking.marked_element_ids
        ):
            raise RuntimeError("nested_refinement_maximum_levels_exceeded")
        marking_levels.append({
            "level": int(maximum_levels), "element_count_before": current.mesh.ne,
            "physical_mark_count": 0, "physical_marked_area_m2": 0.0,
            "marked_region_bounding_box": None,
            "unique_initial_parent_elements_affected": 0,
            "counts_by_reason": {}, "counts_by_tip_candidate_reason": [],
            "records": [],
        })

    visibility = trial_stiffness_visibility(
        current, candidates_by_tip, da_phys_m=da_phys_m,
        crack_band_radius_m=max(float(current.mesh.hbar_tip), float(crack_band_radius_m)),
    )
    if min(visibility.values(), default=0) <= 0:
        invisible = sorted(key for key, value in visibility.items() if value <= 0)
        raise RuntimeError(f"sharp_wake_trial_not_mechanically_resolved: {invisible}")
    prolonged_energy, prolonged_reaction = _stored_energy_and_reaction(current)
    K, residual, *_ = assemble_mechanics(
        current.mesh, current.displacement, current.ep_gp, current.rho_gp,
        current.damage, current.elasticity_D, current.material,
        cohesive_network=current.cohesive_network,
    )
    displacement, equilibrium_reaction = solve_dirichlet(
        K, residual, current.displacement, current.boundary,
        0.5 * float(accepted_load_m), -0.5 * float(accepted_load_m),
    )
    current = replace(current, displacement=displacement)
    equilibrium_energy, _ = _stored_energy_and_reaction(current)
    current = replace(current, stored_energy_J_per_m=equilibrium_energy)
    denominator = max(abs(parent_energy), 1e-300)
    return current, AdaptationAudit(
        lineages=tuple(lineages), parent_energy_J_per_m=parent_energy,
        prolonged_energy_J_per_m=prolonged_energy,
        refined_equilibrium_energy_J_per_m=equilibrium_energy,
        parent_vs_prolonged_relative_error=(prolonged_energy - parent_energy) / denominator,
        refined_equilibrium_relative_correction=(equilibrium_energy - prolonged_energy) / max(abs(prolonged_energy), 1e-300),
        reaction_prolongated_N_per_m=prolonged_reaction,
        reaction_refined_equilibrium_N_per_m=float(equilibrium_reaction),
        active_tip_hbar_m=active_tip_hbar(current, contour_radius_m=contour_radius_m),
        trial_changed_element_count=visibility,
        refinement_marking_diagnostics={
            "schema": "v11.reason-resolved-adaptation-marking/1",
            "cumulative_mark_operations": sum(
                int(item["physical_mark_count"]) for item in marking_levels
            ),
            "unique_initial_parent_elements_affected": len(unique_roots),
            "levels": marking_levels,
        },
    )


__all__ = [
    "AdaptationAudit", "MarkingAudit", "MarkingReason", "PhysicalMarkRecord",
    "RefinementLineage", "active_tip_hbar", "adapt_accepted_state_for_trials",
    "diagnose_underresolved_trial_geometry", "mark_multitip_trial_support",
    "mesh_fingerprint", "refine_accepted_state", "trial_stiffness_visibility",
    "TrialVisibilityFailure", "zero_visibility_reasons",
]
