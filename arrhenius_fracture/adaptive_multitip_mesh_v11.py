"""Nested, deterministic conforming h-refinement for accepted v11 FEM states."""
from __future__ import annotations

from dataclasses import dataclass, replace
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

    def to_dict(self) -> dict:
        return json.loads(json.dumps(self, default=lambda value: value.__dict__, sort_keys=True))


def _subdivide(nodes: np.ndarray, elems: np.ndarray, marked: Iterable[int]):
    marked_ids = tuple(sorted(set(int(index) for index in marked)))
    if any(index < 0 or index >= len(elems) for index in marked_ids):
        raise ValueError("marked parent element index is out of range")
    split_edges: set[Edge] = set()
    for index in marked_ids:
        a, b, c = (int(v) for v in elems[index])
        split_edges.update((_edge(a, b), _edge(b, c), _edge(c, a)))
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


def refine_accepted_state(state, *, marked_parent_elements: Iterable[int], active_tip_ids: Iterable[str], generation: int, operation_index: int):
    old = state.mesh
    marked = tuple(sorted(set(int(value) for value in marked_parent_elements)))
    nodes, elems, midpoint, interpolation, parent_map, conformity = _subdivide(old.nodes, old.elems, marked)
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
    centroids = mesh.nodes[mesh.elems].mean(axis=1)
    element_radius = np.sqrt(np.maximum(mesh.area_e, 0.0))
    marked = np.zeros(mesh.ne, dtype=bool)
    support = max(float(crack_band_radius_m), 0.0)
    for tip_id in sorted(network.active_tip_ids):
        tip = np.asarray(network.branch(tip_id).tip, dtype=float)
        marked |= np.linalg.norm(centroids - tip, axis=1) <= float(contour_radius_m) + element_radius
        for candidate in sorted(candidates_by_tip[tip_id], key=lambda item: item.candidate_id):
            end = tip + float(da_phys_m) * np.asarray(candidate.direction_xy, dtype=float)
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


def active_tip_hbar(state) -> dict[str, float]:
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


def trial_stiffness_visibility(state, candidates_by_tip, *, da_phys_m: float, crack_band_radius_m: float) -> dict[str, int]:
    from .crack_backend import SharpWakeBackend
    inherited = getattr(state.mesh, "element_damage_gp", None)
    if inherited is None:
        inherited = np.mean(np.asarray(state.damage)[state.mesh.elems], axis=1)
    result: dict[str, int] = {}
    for tip_id in sorted(state.crack_network.active_tip_ids):
        tip = np.asarray(state.crack_network.branch(tip_id).tip, dtype=float)
        for candidate in sorted(candidates_by_tip[tip_id], key=lambda item: item.candidate_id):
            end = tip + float(da_phys_m) * np.asarray(candidate.direction_xy, dtype=float)
            trial = SharpWakeBackend().advance(
                mesh=state.mesh, boundary=state.boundary, damage=state.damage,
                displacement=state.displacement, p0=tip, p1=end,
                kill_r=float(crack_band_radius_m),
            )
            after = getattr(trial.mesh, "element_damage_gp", None)
            if after is None:
                after = np.mean(np.asarray(trial.damage)[trial.mesh.elems], axis=1)
            result[f"{tip_id}|{candidate.candidate_id}"] = int(np.count_nonzero(np.asarray(after) != inherited))
    return result


def adapt_accepted_state_for_trials(
    state, candidates_by_tip, *, da_phys_m: float, tip_h_fine_m: float,
    contour_radius_m: float, crack_band_radius_m: float, accepted_load_m: float,
    starting_generation: int = 0, starting_operation_index: int = 0,
    maximum_levels: int = 8,
):
    """Proactively refine one accepted discretization for all sibling trials."""
    from .fem import assemble_mechanics, solve_dirichlet

    target_hbar = max(float(tip_h_fine_m) * 1.5, float(da_phys_m) / 5.0)
    parent_energy, _ = _stored_energy_and_reaction(state)
    current = state
    lineages = []
    for level in range(1, int(maximum_levels) + 1):
        hbars = active_tip_hbar(current)
        support = mark_multitip_trial_support(
            current.mesh, current.crack_network, candidates_by_tip,
            da_phys_m=da_phys_m, contour_radius_m=contour_radius_m,
            crack_band_radius_m=max(float(current.mesh.hbar_tip), float(crack_band_radius_m)),
        )
        mean_edge = _mean_edge_length(current.mesh)
        marked = tuple(index for index in support if mean_edge[index] > target_hbar)
        visible = trial_stiffness_visibility(
            current, candidates_by_tip, da_phys_m=da_phys_m,
            crack_band_radius_m=max(float(current.mesh.hbar_tip), float(crack_band_radius_m)),
        )
        if max(hbars.values(), default=0.0) <= target_hbar and min(visible.values(), default=1) > 0 and not marked:
            break
        if not marked:
            marked = support
        refined, lineage = refine_accepted_state(
            current, marked_parent_elements=marked,
            active_tip_ids=current.crack_network.active_tip_ids,
            generation=starting_generation + level,
            operation_index=starting_operation_index + level,
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
    else:
        raise RuntimeError("nested_refinement_maximum_levels_exceeded")

    visibility = trial_stiffness_visibility(
        current, candidates_by_tip, da_phys_m=da_phys_m,
        crack_band_radius_m=max(float(current.mesh.hbar_tip), float(crack_band_radius_m)),
    )
    if min(visibility.values(), default=0) <= 0:
        invisible = sorted(key for key, value in visibility.items() if value <= 0)
        raise RuntimeError(f"unresolved_trial_stiffness_topology: {invisible}")
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
        active_tip_hbar_m=active_tip_hbar(current),
        trial_changed_element_count=visibility,
    )


__all__ = [
    "AdaptationAudit", "RefinementLineage", "active_tip_hbar",
    "adapt_accepted_state_for_trials", "mark_multitip_trial_support",
    "mesh_fingerprint", "refine_accepted_state", "trial_stiffness_visibility",
]
