"""Executable one-void trajectory owned by the V12 production FEM state."""
from __future__ import annotations

from dataclasses import asdict, replace
import hashlib
import json
import math
from pathlib import Path
import subprocess
from typing import Any

import numpy as np
from scipy.interpolate import LinearNDInterpolator
from scipy.spatial import cKDTree

from .crack_network_v11 import CrackBranchState, CrackNetworkState, ROOT_BRANCH_ID
from .directional_competition_v11 import (
    CleavageCandidate, DirectionalCompetitionState, commit_directional_interval,
    construct_action_proposals, preview_directional_interval,
    select_temporal_or_degenerate_proposal, tungsten_cleavage_candidates,
)
from .explicit_cavity_v5 import build_explicit_hole_mesh, fill_explicit_hole_mesh, triangle_intersects_open_disk
from .mesh import BoundaryData, rebuild_tri_mesh
from .fem import assemble_mechanics, plane_strain_D
from .sharp_front import (
    FrontConfig, FrontEngine, default_cleavage_barrier,
    default_emission_barrier, make_emergent_config,
)
from .hazard_energy_event_gate_v10230 import hazard_resistance_J_per_m2
from .mechanically_separating_sharp_wake_v12 import (
    certification_arcs, independent_intact_path_certificate,
)
from .sharp_wake_backend_v12 import V12_MODEL_ID
from .topology_transaction_v11 import (
    LiveFEMTopologyState, TopologyArm, apply_v12_production_trial_geometry,
    complete_accepted_state_fingerprint, equilibrate_fixed_load_with_production_fem,
    execute_topology_trial, initialize_mechanically_separating_v12,
    remesh_mechanically_separating_v12,
)
from .voiding_v5 import (
    Cavity2D, HazardClock, ProductionVoidState, VoidPhase, VoidSite, VoidingConfig,
    advance_site, arrhenius_rates, create_subgrid_cavity, grow_cavity_2d,
    grow_cavity_from_rate,
    promote_cavity, replace_cavity, update_cavity_growth,
)

SCHEMA = "v12.production-one-void-trajectory/5"


def _head():
    return subprocess.check_output(("git", "rev-parse", "HEAD"), cwd=Path(__file__).resolve().parents[1], text=True).strip()


def _geometry(radius_m=5.0e-5, center_m=(7.0e-4, 0.0)):
    if center_m[1] < 0.0:
        positive_hole, positive_filled = _geometry(radius_m, (center_m[0], -center_m[1]))
        def mirrored(value):
            nodes = np.asarray(value.mesh.nodes).copy(); nodes[:, 1] *= -1.0
            elems = np.asarray(value.mesh.elems)[:, [0, 2, 1]]
            mesh = rebuild_tri_mesh(nodes, elems, tip_centers=np.asarray(center_m))
            bottom = np.asarray(value.boundary.top_nodes)
            boundary = BoundaryData(
                np.asarray(value.boundary.bot_nodes), bottom,
                int(bottom[np.argmin(nodes[bottom, 0])]), int(bottom[np.argmax(nodes[bottom, 0])]),
                np.asarray(value.boundary.notch_nodes),
            )
            return replace(value, mesh=mesh, boundary=boundary,
                           center_m=(float(center_m[0]), float(center_m[1])))
        return mirrored(positive_hole), mirrored(positive_filled)
    hole = build_explicit_hole_mesh(1.0e-3, 1.0e-3, center_m, radius_m, 5.0e-5, 32, radial_layers_override=12)
    return hole, fill_explicit_hole_mesh(hole)


def _insert_point_in_mesh(mesh, point):
    point = np.asarray(point, dtype=float)
    nodes = np.asarray(mesh.nodes)
    if float(np.min(np.linalg.norm(nodes - point, axis=1))) <= 1.0e-12:
        return mesh
    owners = []
    for index, triangle_ids in enumerate(np.asarray(mesh.elems, dtype=int)):
        triangle = nodes[triangle_ids]
        matrix = np.column_stack((triangle[1] - triangle[0], triangle[2] - triangle[0]))
        if abs(np.linalg.det(matrix)) <= 1.0e-24: continue
        uv = np.linalg.solve(matrix, point - triangle[0])
        if uv[0] >= -1.0e-12 and uv[1] >= -1.0e-12 and uv.sum() <= 1.0 + 1.0e-12:
            owners.append((index, uv))
    if not owners:
        raise ValueError("fixed crack tip is outside the specimen mesh")
    new = len(nodes)
    source = np.asarray(mesh.elems, dtype=int)
    strict = [(index, uv) for index, uv in owners
              if uv[0] > 1.0e-10 and uv[1] > 1.0e-10 and uv.sum() < 1.0 - 1.0e-10]
    replacements = []
    removed = set()
    if strict:
        owner = strict[0][0]
        a, b, c = map(int, source[owner])
        removed.add(owner)
        replacements.extend(((a, b, new), (b, c, new), (c, a, new)))
    else:
        # A point on an existing edge must split both incident triangles.  A
        # one-sided split creates a hanging node and makes mirrored fixed-crack
        # meshes depend on which owner happens to be encountered first.
        candidate_edges = []
        for owner, _ in owners:
            triangle_ids = source[owner]
            for u, v in ((triangle_ids[0], triangle_ids[1]),
                         (triangle_ids[1], triangle_ids[2]),
                         (triangle_ids[2], triangle_ids[0])):
                edge = nodes[v] - nodes[u]
                scale = max(float(np.dot(edge, edge)), 1.0e-300)
                fraction = float(np.dot(point - nodes[u], edge) / scale)
                offset = point - nodes[u]
                distance = abs(float(edge[0] * offset[1] - edge[1] * offset[0])) / math.sqrt(scale)
                if -1.0e-12 <= fraction <= 1.0 + 1.0e-12 and distance <= 1.0e-12:
                    candidate_edges.append(tuple(sorted((int(u), int(v)))))
        edge = min(candidate_edges)
        for owner, triangle_ids in enumerate(source):
            if not set(edge).issubset(map(int, triangle_ids)):
                continue
            removed.add(owner)
            u, v = edge
            w = next(int(value) for value in triangle_ids if int(value) not in edge)
            original = nodes[triangle_ids]
            oa, ob = original[1] - original[0], original[2] - original[0]
            original_sign = oa[0] * ob[1] - oa[1] * ob[0]
            for tri in ((u, new, w), (new, v, w)):
                coordinates = np.vstack((nodes[tri[0]], point, nodes[tri[2]])) if tri[0] == u else np.vstack((point, nodes[tri[1]], nodes[tri[2]]))
                ca, cb = coordinates[1] - coordinates[0], coordinates[2] - coordinates[0]
                sign = ca[0] * cb[1] - ca[1] * cb[0]
                replacements.append(tri if sign * original_sign > 0.0 else (tri[0], tri[2], tri[1]))
    elems = np.delete(source, sorted(removed), axis=0)
    elems = np.vstack((elems, replacements))
    return rebuild_tri_mesh(np.vstack((nodes, point)), elems, tip_centers=point)


def _external_free_root_context(mesh, boundary, root):
    counts = {}
    for triangle in np.asarray(mesh.elems, dtype=int):
        for a, b in ((triangle[0], triangle[1]), (triangle[1], triangle[2]), (triangle[2], triangle[0])):
            edge = tuple(sorted((int(a), int(b))))
            counts[edge] = counts.get(edge, 0) + 1
    prescribed_nodes = set(map(int, np.asarray(boundary.top_nodes))) | set(
        map(int, np.asarray(boundary.bot_nodes))
    )
    point = np.asarray(root, dtype=float)
    free_edges = [edge for edge, count in counts.items()
                  if count == 1 and not set(edge).issubset(prescribed_nodes)]
    incident = []
    for edge in free_edges:
        if set(edge).issubset(prescribed_nodes):
            continue
        a, b = np.asarray(mesh.nodes)[list(edge)]
        delta = b - a
        fraction = float((point - a) @ delta / max(delta @ delta, 1.0e-300))
        distance = float(np.linalg.norm(point - (a + np.clip(fraction, 0.0, 1.0) * delta)))
        if distance <= 1.0e-12 and -1.0e-12 <= fraction <= 1.0 + 1.0e-12:
            incident.append(edge)
    if not incident:
        return {}
    component = set(incident)
    frontier = set(node for edge in incident for node in edge)
    changed = True
    while changed:
        changed = False
        for edge in free_edges:
            if edge in component or not frontier.intersection(edge):
                continue
            component.add(edge); frontier.update(edge); changed = True
    return {"b00000000:arc0": ({
        "endpoint": "start", "endpoint_role": "physical_root",
        "endpoint_coordinate_m": tuple(map(float, root)),
        "boundary_kind": "external_free_surface",
        "boundary_component_id": "external-free-component-at-root",
        "boundary_edge_ids": tuple(sorted(component)), "cavity_id": None,
        "tangent_enters_or_approaches_solid": True,
    },)}


def _refine_state_around_graph(state, levels):
    from .adaptive_multitip_mesh_v11 import refine_accepted_state
    result = state
    for level in range(int(levels)):
        centroids = np.asarray(result.mesh.nodes)[np.asarray(result.mesh.elems)].mean(axis=1)
        local_h = np.sqrt(np.maximum(np.asarray(result.mesh.area_e), 1.0e-300))
        marked = np.zeros(result.mesh.ne, dtype=bool)
        for branch in result.crack_network.branches:
            for first, second in zip(branch.path, branch.path[1:]):
                a, b = np.asarray(first), np.asarray(second)
                delta = b - a
                fraction = np.clip(((centroids - a) @ delta) /
                                   max(float(delta @ delta), 1.0e-300), 0.0, 1.0)
                distance = np.linalg.norm(centroids - (a + fraction[:, None] * delta), axis=1)
                marked |= distance <= 2.0 * local_h
        result, _ = refine_accepted_state(
            result, marked_parent_elements=tuple(np.flatnonzero(marked)),
            active_tip_ids=result.crack_network.active_tip_ids,
            generation=level + 1, operation_index=level + 1,
        )
    return result


def build_production_void_state(*, enabled=True, stochastic=False, seed=3621,
                                cavity_center_m=(7.0e-4, 0.0), crack_path_m=None,
                                cleavage_theta_deg=0.0):
    hole, filled = _geometry(center_m=cavity_center_m)
    mesh = filled.mesh
    ray = 16
    if crack_path_m is None:
        start = tuple(map(float, mesh.nodes[12 * 32 + ray]))
        tip = tuple(map(float, mesh.nodes[3 * 32 + ray]))
        crack_path = (start, tip)
    else:
        crack_path = tuple(tuple(map(float, point)) for point in crack_path_m)
        if len(crack_path) < 2:
            raise ValueError("fixed crack path requires at least two points")
        start, tip = crack_path[0], crack_path[-1]
        for point in crack_path:
            mesh = _insert_point_in_mesh(mesh, point)
            hole = replace(hole, mesh=_insert_point_in_mesh(hole.mesh, point))
        filled = replace(filled, mesh=mesh)
    cfg = make_emergent_config()
    element_x = np.asarray(mesh.nodes)[np.asarray(mesh.elems)].mean(axis=1)[:, 0]
    u = np.zeros(mesh.ndof)
    u[2 * np.asarray(filled.boundary.top_nodes) + 1] = 2.0e-7
    u[2 * np.asarray(filled.boundary.bot_nodes) + 1] = -2.0e-7
    void_state = None
    if enabled:
        rng = np.random.default_rng(seed)
        thresholds = tuple(float(rng.exponential()) for _ in range(3)) if stochastic else (0.25, 0.25, 0.35)
        site = VoidSite(
            "site-1", tuple(map(float, cavity_center_m)), VoidPhase.AVAILABLE_SITE, 0, 2, 0.8,
            HazardClock(0.0, thresholds[0]), HazardClock(0.0, thresholds[1]), HazardClock(0.0, thresholds[2]),
        )
        void_state = ProductionVoidState((site,), rng_state=rng.bit_generator.state)
    state = LiveFEMTopologyState(
        mesh, filled.boundary, np.zeros(mesh.nn), u,
        np.vstack((np.full(mesh.ne, 1.0e-5), np.full(mesh.ne, -0.5e-5),
                   np.full(mesh.ne, 0.25e-5 * (1.0 if cavity_center_m[1] >= 0.0 else -1.0)))),
        1.0e12 + 2.0e11 * element_x / 1.0e-3,
        plane_strain_D(cfg.material), cfg.material, None,
        CrackNetworkState.one_tip(crack_path), DirectionalCompetitionState.initialize(
            tungsten_cleavage_candidates(theta_deg=cleavage_theta_deg), global_hazard_seed=seed,
        ),
        {"retained": 4.0, "mobile": 1.0}, {
            "source_state": {"density": 3.0, "clock": 0.125},
            "boundary_terminal_context": _external_free_root_context(mesh, filled.boundary, start),
        },
        {"emission_work": 1.0}, np.random.default_rng(seed).bit_generator.state,
        {"accepted_steps": 0, "topology_actions": 0, "mesh_generation": 0}, 0.0,
        void_state=void_state,
    )
    if abs(float(cleavage_theta_deg)) > 1.0e-12:
        hole_fields = _project_fields(state, hole.mesh)
        hole_state = replace(
            state, mesh=hole.mesh, boundary=hole.boundary,
            damage=hole_fields["damage"], displacement=hole_fields["displacement"],
            ep_gp=hole_fields["ep_gp"], rho_gp=hole_fields["rho_gp"],
        )
        hole_state = _refine_state_around_graph(hole_state, 2)
        hole = replace(hole, mesh=hole_state.mesh, boundary=hole_state.boundary)
        state = _refine_state_around_graph(state, 2)
        junction = dict(state.junction_process_state)
        junction["boundary_terminal_context"] = _external_free_root_context(
            state.mesh, state.boundary, start,
        )
        state = replace(state, junction_process_state=junction)
    state = initialize_mechanically_separating_v12(
        state, source_commit=_head(), configuration={"voiding_enabled": enabled, "model": V12_MODEL_ID},
    )
    return equilibrate_fixed_load_with_production_fem(state), hole


def observables(state, operation):
    cavity = None if state.void_state is None or not state.void_state.cavities else state.void_state.cavities[0]
    site = None if state.void_state is None else state.void_state.sites[0]
    tri = np.asarray(state.mesh.nodes)[np.asarray(state.mesh.elems)]
    side = np.linalg.norm(tri[:, [1, 2, 0]] - tri[:, [0, 1, 2]], axis=2)
    avec, bvec = tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]
    area = np.abs(avec[:, 0] * bvec[:, 1] - avec[:, 1] * bvec[:, 0]) / 2.0
    quality = 4.0 * np.sqrt(3.0) * area / np.maximum(np.sum(side**2, axis=1), 1.0e-300)
    reaction = float(state.energy_ledgers.get("latest_reaction_N_per_m", 0.0))
    active_branches = [branch.branch_id for branch in state.crack_network.branches
                       if branch.status == "active"]
    support_active = [] if state.v12_support_state is None else list(state.v12_support_state.active_tip_identities)
    return {
        "operation": operation, "fingerprint": complete_accepted_state_fingerprint(state),
        "mesh_nodes": int(state.mesh.nn), "mesh_elements": int(state.mesh.ne),
        "graph_length_m": float(state.crack_network.total_physical_crack_length_m),
        "reaction_N_per_m": reaction,
        "compliance_m2_per_N": 4.0e-7 / max(abs(reaction), 1.0e-300),
        "energy_J_per_m": float(state.stored_energy_J_per_m),
        "full_residual_including_reactions_N_per_m": float(state.energy_ledgers.get("latest_residual_l2_N_per_m", 0.0)),
        "free_dof_residual_l2_N_per_m": float(state.energy_ledgers.get("latest_free_dof_residual_l2_N_per_m", 0.0)),
        "constrained_reaction_l2_N_per_m": float(state.energy_ledgers.get("latest_constrained_reaction_l2_N_per_m", 0.0)),
        "top_bottom_reaction_balance": float(state.energy_ledgers.get("latest_top_bottom_reaction_balance", 0.0)),
        "energy_reaction_identity": float(state.energy_ledgers.get("latest_energy_reaction_identity", 0.0)),
        "void_phase": None if cavity is None else cavity.phase.value,
        "site_phase": None if site is None else site.phase.value,
        "cavity_radius_m": None if cavity is None else cavity.radius_m,
        "cavity_area_m2": None if cavity is None else cavity.area_m2,
        "inventory_area_m2": None if cavity is None else cavity.inventory_area_m2,
        "available_defect_inventory_area_m2": None if state.void_state is None else state.void_state.available_defect_inventory_area_m2,
        "consumed_defect_inventory_area_m2": None if state.void_state is None else state.void_state.consumed_defect_inventory_area_m2,
        "void_event_history": [] if state.void_state is None else list(state.void_state.event_history),
        "length_ledgers": {} if state.void_state is None else dict(state.void_state.length_ledgers),
        "event_counters": dict(state.event_counters),
        "active_crack_branch_ids": active_branches,
        "support_active_tip_ids": support_active,
        "mesh_minimum_quality": float(np.min(quality)),
        "mesh_maximum_aspect_ratio": float(np.max(side, axis=1).max() / max(np.min(side), 1.0e-300)),
        "field_transfer_audit": state.junction_process_state.get("latest_void_remesh_audit"),
        "closed_cavity_boundary_cycle_certificate": state.junction_process_state.get("latest_closed_cavity_boundary_cycle_certificate"),
        "crack_void_connection_certificate": state.junction_process_state.get("latest_crack_void_connection_certificate"),
    }


def local_site_tensor(state):
    _, _, sigma, *_ = assemble_mechanics(
        state.mesh, state.displacement, state.ep_gp, state.rho_gp, state.damage,
        state.elasticity_D, state.material, cohesive_network=state.cohesive_network,
    )
    center = np.asarray(state.void_state.sites[0].center_m)
    centroids = state.mesh.nodes[state.mesh.elems].mean(axis=1)
    element = int(np.argmin(np.linalg.norm(centroids - center, axis=1)))
    return np.array([[sigma[0, element], sigma[2, element]], [sigma[2, element], sigma[1, element]]])


def crack_tip_tensor(state, *, branch_id):
    """Return a geometry-weighted tensor recovery at an aligned crack-tip node."""
    _, _, sigma, *_ = assemble_mechanics(
        state.mesh, state.displacement, state.ep_gp, state.rho_gp, state.damage,
        state.elasticity_D, state.material, cohesive_network=state.cohesive_network,
    )
    tip = np.asarray(state.crack_network.branch(branch_id).tip)
    nodes = np.asarray(state.mesh.nodes)
    node = int(np.argmin(np.linalg.norm(nodes - tip, axis=1)))
    if np.linalg.norm(nodes[node] - tip) > 1.0e-12:
        raise RuntimeError("crack-tip tensor recovery requires an aligned mesh node")
    elements = np.flatnonzero(np.any(np.asarray(state.mesh.elems) == node, axis=1))
    weights = np.asarray(state.mesh.area_e)[elements]
    weights = weights / np.sum(weights)
    recovered = np.array([
        [np.sum(weights * sigma[0, elements]), np.sum(weights * sigma[2, elements])],
        [np.sum(weights * sigma[2, elements]), np.sum(weights * sigma[1, elements])],
    ])
    return recovered, tuple(map(int, elements))


def _first_ray_cavity_intersection(state, start, direction):
    """Intersect a candidate ray with the actual polygonal cavity boundary."""
    cavity = state.void_state.cavities[0]
    nodes = np.asarray(state.mesh.nodes)
    radii = np.linalg.norm(nodes - np.asarray(cavity.center_m), axis=1)
    boundary = nodes[np.flatnonzero(radii <= cavity.radius_m * 1.02)]
    angles = np.arctan2(boundary[:, 1] - cavity.center_m[1], boundary[:, 0] - cavity.center_m[0])
    boundary = boundary[np.argsort(angles)]
    origin = np.asarray(start, dtype=float)
    ray = np.asarray(direction, dtype=float)
    intersections = []
    for first, second in zip(boundary, np.vstack((boundary[1:], boundary[:1]))):
        edge = second - first
        matrix = np.column_stack((ray, -edge))
        if abs(np.linalg.det(matrix)) <= 1.0e-18:
            continue
        distance, fraction = np.linalg.solve(matrix, first - origin)
        if distance > 1.0e-14 and -1.0e-12 <= fraction <= 1.0 + 1.0e-12:
            intersections.append((float(distance), origin + distance * ray))
    if not intersections:
        return None
    return tuple(map(float, min(intersections, key=lambda item: item[0])[1]))


def _conform_cavity_intersection(state, intersection, *, failure_injector=None):
    """Insert an interior ray/cavity-edge intersection as an explicit node."""
    point = np.asarray(intersection, dtype=float)
    nodes = np.asarray(state.mesh.nodes)
    nearest = int(np.argmin(np.linalg.norm(nodes - point, axis=1)))
    if np.linalg.norm(nodes[nearest] - point) <= 1.0e-12:
        return state, nearest, False
    counts: dict[tuple[int, int], int] = {}
    owners: dict[tuple[int, int], list[int]] = {}
    for element, triangle in enumerate(np.asarray(state.mesh.elems, dtype=int)):
        for a, b in ((triangle[0], triangle[1]), (triangle[1], triangle[2]), (triangle[2], triangle[0])):
            edge = tuple(sorted((int(a), int(b))))
            counts[edge] = counts.get(edge, 0) + 1
            owners.setdefault(edge, []).append(element)
    cavity = state.void_state.cavities[0]
    center = np.asarray(cavity.center_m)
    candidates = []
    for edge, count in counts.items():
        if count != 1:
            continue
        a, b = nodes[list(edge)]
        if max(abs(np.linalg.norm(a - center) - cavity.radius_m),
               abs(np.linalg.norm(b - center) - cavity.radius_m)) > cavity.radius_m * 0.02:
            continue
        delta = b - a
        fraction = float((point - a) @ delta / max(delta @ delta, 1.0e-300))
        distance = float(np.linalg.norm(point - (a + np.clip(fraction, 0.0, 1.0) * delta)))
        if 1.0e-12 < fraction < 1.0 - 1.0e-12:
            candidates.append((distance, edge, owners[edge][0]))
    if not candidates or min(candidates)[0] > 1.0e-11:
        raise RuntimeError("ray intersection is not on a splittable cavity boundary edge")
    _, edge, owner = min(candidates)
    triangle = list(map(int, state.mesh.elems[owner]))
    a, b = edge
    third = next(node for node in triangle if node not in edge)
    new_node = len(nodes)
    elems = np.delete(np.asarray(state.mesh.elems, dtype=int), owner, axis=0)
    elems = np.vstack((elems, (a, new_node, third), (new_node, b, third)))
    mesh = rebuild_tri_mesh(np.vstack((nodes, point)), elems, tip_centers=np.asarray(point))
    fields = _project_fields(state, mesh)
    rebuilt = remesh_mechanically_separating_v12(
        state, mesh=mesh, boundary=state.boundary, transferred_fields=fields,
        source_commit=_head(), configuration={"event": "CAVITY_EDGE_INTERSECTION_INSERTION"},
        transaction_identity="cavity-edge-intersection", failure_injector=failure_injector,
    )
    return rebuilt, new_node, True


def _conform_bulk_point(state, point, *, identity, failure_injector=None):
    mesh = _insert_point_in_mesh(state.mesh, point)
    if mesh is state.mesh:
        return state
    fields = _project_fields(state, mesh)
    return remesh_mechanically_separating_v12(
        state, mesh=mesh, boundary=state.boundary, transferred_fields=fields,
        source_commit=_head(), configuration={"event": identity},
        transaction_identity=identity.lower().replace("_", "-"),
        failure_injector=failure_injector,
    )


def cavity_boundary_tensor(state, *, boundary_node: int | None = None,
                           boundary_element: int | None = None):
    """Return the most tensile resolved tensor on the explicit cavity boundary.

    First passage is local, so circumferential averaging (which can cancel the
    tensile hot spot) is not a valid kinetic input.
    """
    _, _, sigma, *_ = assemble_mechanics(
        state.mesh, state.displacement, state.ep_gp, state.rho_gp, state.damage,
        state.elasticity_D, state.material, cohesive_network=state.cohesive_network,
    )
    cavity = state.void_state.cavities[0]
    center = np.asarray(cavity.center_m)
    radii = np.linalg.norm(np.asarray(state.mesh.nodes) - center, axis=1)
    boundary_nodes = np.flatnonzero(radii <= cavity.radius_m * 1.02)
    selected_nodes = boundary_nodes if boundary_node is None else np.asarray((int(boundary_node),))
    if boundary_node is not None and int(boundary_node) not in set(map(int, boundary_nodes)):
        raise ValueError("requested tensor node is not on the cavity boundary")
    elements = np.flatnonzero(np.any(np.isin(state.mesh.elems, selected_nodes), axis=1))
    if boundary_element is not None:
        if boundary_node is None:
            raise ValueError("fixed boundary element requires a fixed boundary node")
        if int(boundary_element) not in set(map(int, elements)):
            raise ValueError("requested tensor element is not incident to the fixed boundary node")
        elements = np.asarray((int(boundary_element),))
    tensors = np.asarray([
        [[sigma[0, element], sigma[2, element]],
         [sigma[2, element], sigma[1, element]]]
        for element in elements
    ])
    selected = int(np.argmax(np.linalg.eigvalsh(tensors)[:, -1]))
    return tensors[selected], (int(elements[selected]),)


def cavity_boundary_recovery_operator(state, boundary_node: int) -> dict[str, Any]:
    """Freeze a geometry-only, single-element boundary stress sampler."""
    elements = np.flatnonzero(np.any(np.asarray(state.mesh.elems) == int(boundary_node), axis=1))
    if not len(elements):
        raise ValueError("boundary node has no incident element")
    selected = int(np.min(elements))
    return {"boundary_node_id": int(boundary_node), "selected_element_id": selected,
            "recovery_operator_id": f"incident-element:{selected}", "recovery_weights": [1.0]}


def cavity_free_surface_certificate(state) -> dict[str, Any]:
    """Certify one connected, closed mesh-boundary cycle at the cavity."""
    cavity = state.void_state.cavities[0]
    nodes = np.asarray(state.mesh.nodes)
    radii = np.linalg.norm(nodes - np.asarray(cavity.center_m), axis=1)
    cavity_nodes = set(map(int, np.flatnonzero(radii <= cavity.radius_m * 1.02)))
    counts: dict[tuple[int, int], int] = {}
    for triangle in np.asarray(state.mesh.elems, dtype=int):
        for a, b in ((triangle[0], triangle[1]), (triangle[1], triangle[2]), (triangle[2], triangle[0])):
            edge = tuple(sorted((int(a), int(b))))
            counts[edge] = counts.get(edge, 0) + 1
    edges = tuple(edge for edge, count in counts.items() if count == 1 and set(edge).issubset(cavity_nodes))
    adjacency = {node: set() for edge in edges for node in edge}
    for a, b in edges:
        adjacency[a].add(b)
        adjacency[b].add(a)
    visited = set()
    stack = [next(iter(adjacency))] if adjacency else []
    while stack:
        node = stack.pop()
        if node not in visited:
            visited.add(node)
            stack.extend(adjacency[node] - visited)
    passed = len(edges) >= 16 and visited == set(adjacency) and all(len(peers) == 2 for peers in adjacency.values())
    return {"passed": passed, "boundary_edge_count": len(edges),
            "boundary_node_ids": sorted(adjacency), "boundary_edge_ids": [list(edge) for edge in edges],
            "boundary_node_count": len(adjacency),
            "connected_component_count": 1 if passed else None,
            "topology": "single_closed_cavity_boundary_cycle"}


def _polygon_boundary_arc_length(state, first_point, second_point) -> float:
    cycle = cavity_free_surface_certificate(state)
    nodes = np.asarray(state.mesh.nodes)
    first = min(cycle["boundary_node_ids"], key=lambda node: np.linalg.norm(nodes[node] - first_point))
    second = min(cycle["boundary_node_ids"], key=lambda node: np.linalg.norm(nodes[node] - second_point))
    adjacency = {node: [] for node in cycle["boundary_node_ids"]}
    for a, b in cycle["boundary_edge_ids"]:
        length = float(np.linalg.norm(nodes[a] - nodes[b]))
        adjacency[a].append((b, length)); adjacency[b].append((a, length))
    lengths = []
    for initial in adjacency[first]:
        previous, current, total = first, initial[0], initial[1]
        while current != second:
            following = next(item for item in adjacency[current] if item[0] != previous)
            previous, current, total = current, following[0], total + following[1]
        lengths.append(total)
    return min(lengths)


def _ordered_cavity_polygon(state, cycle):
    adjacency = {int(node): [] for node in cycle["boundary_node_ids"]}
    for a, b in cycle["boundary_edge_ids"]:
        adjacency[int(a)].append(int(b)); adjacency[int(b)].append(int(a))
    first = min(adjacency)
    ordered = [first]
    previous, current = None, first
    while True:
        choices = sorted(node for node in adjacency[current] if node != previous)
        following = choices[0]
        if following == first:
            break
        ordered.append(following)
        previous, current = current, following
    return np.asarray(state.mesh.nodes, dtype=float)[ordered]


def _convex_polygon_interior_overlaps_triangle(polygon, triangle, tolerance=1.0e-14):
    """Exact edge-axis test; boundary contact alone is not interior overlap."""
    polygon = np.asarray(polygon, dtype=float)
    triangle = np.asarray(triangle, dtype=float)
    axes = []
    for vertices in (polygon, triangle):
        for first, second in zip(vertices, np.roll(vertices, -1, axis=0)):
            edge = second - first
            axes.append(np.asarray((-edge[1], edge[0]), dtype=float))
    for axis in axes:
        norm = float(np.linalg.norm(axis))
        if norm <= tolerance:
            continue
        axis /= norm
        poly_projection = polygon @ axis
        tri_projection = triangle @ axis
        overlap = min(float(np.max(poly_projection)), float(np.max(tri_projection))) - max(
            float(np.min(poly_projection)), float(np.min(tri_projection))
        )
        if overlap <= tolerance:
            return False
    return True


def crack_void_connection_certificate(state, *, branch_id: str, cavity_id: str,
                                      intended_intersection) -> dict[str, Any]:
    """Certify combined graph/cavity incidence, clearance, and free boundary."""
    cavities = [item for item in state.void_state.cavities if item.cavity_id == cavity_id]
    if len(cavities) != 1:
        raise ValueError("combined topology certificate requires one exact cavity identity")
    cavity = cavities[0]
    branch = state.crack_network.branch(branch_id)
    endpoint = np.asarray(branch.tip)
    intended = np.asarray(intended_intersection, dtype=float)
    center = np.asarray(cavity.center_m)
    cycle = cavity_free_surface_certificate(state)
    endpoint_matches = bool(np.linalg.norm(endpoint - intended) <= 1.0e-12)
    endpoint_on_boundary = bool(abs(np.linalg.norm(endpoint - center) - cavity.radius_m) <= cavity.radius_m * 0.02)
    def segment_open_disk_distance(first, second):
        a = np.asarray(first); b = np.asarray(second); delta = b - a
        fraction = float(np.clip((center - a) @ delta / max(delta @ delta, 1.0e-300), 0.0, 1.0))
        return float(np.linalg.norm(a + fraction * delta - center))
    segment_rows = [{"branch_id": item.branch_id, "segment_index": index,
                     "minimum_center_distance_m": segment_open_disk_distance(first, second),
                     "intersects_cavity_open_disk": segment_open_disk_distance(first, second) < cavity.radius_m * (1.0 - 1.0e-10)}
                    for item in state.crack_network.branches
                    for index, (first, second) in enumerate(zip(item.path, item.path[1:]))]
    graph_outside = not any(row["intersects_cavity_open_disk"] for row in segment_rows)
    support = state.v12_support_state
    support_ids = () if support is None else support.selected_support_elements
    centroids = np.asarray(state.mesh.nodes)[np.asarray(state.mesh.elems)].mean(axis=1)
    open_disk_overlap_ids = [int(element) for element in support_ids if triangle_intersects_open_disk(
        np.asarray(state.mesh.nodes)[np.asarray(state.mesh.elems)[int(element)]], center, cavity.radius_m,
    )]
    cavity_polygon = _ordered_cavity_polygon(state, cycle)
    polygon_overlap_ids = [int(element) for element in support_ids
                           if _convex_polygon_interior_overlaps_triangle(
                               cavity_polygon,
                               np.asarray(state.mesh.nodes)[np.asarray(state.mesh.elems)[int(element)]],
                           )]
    support_outside = not polygon_overlap_ids
    # Retain sampled centerline coverage as a supplemental diagnostic, while
    # the pass/fail decision uses the independent V12 intact path certificate.
    ligament_start = np.asarray(branch.path[-2])
    samples = np.linspace(ligament_start, endpoint, 65)[1:-1]
    support_triangles = np.asarray(state.mesh.nodes)[np.asarray(state.mesh.elems)[list(support_ids)]] if support_ids else np.empty((0, 3, 2))
    def point_in_triangle(point, triangle):
        matrix = np.column_stack((triangle[1] - triangle[0], triangle[2] - triangle[0]))
        if abs(np.linalg.det(matrix)) <= 1.0e-24: return False
        uv = np.linalg.solve(matrix, point - triangle[0])
        return uv[0] >= -1.0e-10 and uv[1] >= -1.0e-10 and uv.sum() <= 1.0 + 1.0e-10
    covered = [any(point_in_triangle(sample, triangle) for triangle in support_triangles) for sample in samples]
    uncovered_indices = [index for index, value in enumerate(covered) if not value]
    no_solid_bridge = not uncovered_indices
    intact_certificate = (independent_intact_path_certificate(
        state.mesh, state.crack_network, support_ids,
        allow_boundary_clip_for_screen=True,
    ) if support_ids else {
        "intact_cross_graph_path_exists": True,
        "insufficient_seed_segment_ids": ("all:no-support",),
        "bridge_node_ids": (), "bridge_element_ids": (),
        "edge_cut_certificates": (), "certificate_fingerprint": "NO_SUPPORT",
    })
    intact_certificate = {
        key: ([asdict(item) for item in value] if key == "edge_cut_certificates" else value)
        for key, value in intact_certificate.items()
    }
    exact_no_bridge = bool(
        not intact_certificate["intact_cross_graph_path_exists"]
        and not intact_certificate["insufficient_seed_segment_ids"]
    )
    # Build and traverse the actual branch/cavity incidence graph.
    component_graph = {"cavity:" + cavity_id: set()}
    incidence_edges = []
    boundary_nodes = np.asarray(cycle["boundary_node_ids"], dtype=int)
    boundary_points = np.asarray(state.mesh.nodes)[boundary_nodes]
    intersected_edge_id = None
    for item in state.crack_network.branches:
        key = "branch:" + item.branch_id; component_graph.setdefault(key, set())
        incident_points = (item.path[0], item.tip)
        incident = any(len(boundary_points) and float(np.min(
            np.linalg.norm(boundary_points - np.asarray(point), axis=1)
        )) <= 1.0e-12 for point in incident_points)
        if incident:
            cavity_key = "cavity:" + cavity_id
            component_graph[key].add(cavity_key); component_graph[cavity_key].add(key)
            incidence_edges.append([key, cavity_key])
    unseen = set(component_graph); components = []
    while unseen:
        todo = [min(unseen)]; component = []
        while todo:
            key = todo.pop()
            if key not in unseen: continue
            unseen.remove(key); component.append(key)
            todo.extend(sorted(component_graph[key], reverse=True))
        components.append(sorted(component))
    combined_count = len(components)
    for edge in cycle["boundary_edge_ids"]:
        a, b = np.asarray(state.mesh.nodes)[edge]
        if np.linalg.norm(endpoint - a) <= 1.0e-12 or np.linalg.norm(endpoint - b) <= 1.0e-12:
            intersected_edge_id = edge; break
    passed = bool(cycle["passed"] and endpoint_matches and endpoint_on_boundary
                  and graph_outside and support_outside and no_solid_bridge
                  and exact_no_bridge and combined_count == 1)
    return {
        "passed": passed, "cavity_id": cavity_id, "branch_id": branch_id,
        "branch_endpoint_m": endpoint.tolist(), "intended_intersection_m": intended.tolist(),
        "endpoint_matches_intersection": endpoint_matches, "endpoint_on_cavity_boundary": endpoint_on_boundary,
        "no_surviving_solid_ligament_bridge": no_solid_bridge,
        "exact_no_intact_node_or_element_path": exact_no_bridge,
        "independent_intact_path_certificate": intact_certificate,
        "bridge_search_sample_count": len(samples), "bridge_search_uncovered_sample_indices": uncovered_indices,
        "intersected_cavity_edge_id": intersected_edge_id,
        "crack_graph_outside_cavity": graph_outside, "wake_support_outside_cavity": support_outside,
        "crack_segment_cavity_intersections": segment_rows,
        "support_triangle_cavity_overlap_element_ids": polygon_overlap_ids,
        "supplemental_open_disk_overlap_element_ids": open_disk_overlap_ids,
        "exact_triangle_polygon_interior_overlap_absent": not polygon_overlap_ids,
        "combined_incidence_edges": incidence_edges, "combined_components": components,
        "active_tip_ids": list(state.crack_network.active_tip_ids),
        "combined_incidence_component_count": combined_count,
        "closed_cavity_boundary_cycle": cycle,
        "topology": "connected_crack_graph_and_traction_free_cavity_cycle",
    }


def _project_fields(state, mesh):
    old_nodes = np.asarray(state.mesh.nodes); new_nodes = np.asarray(mesh.nodes)
    _, node_parent = cKDTree(old_nodes).query(new_nodes)
    old_centers = old_nodes[state.mesh.elems].mean(axis=1)
    new_centers = new_nodes[mesh.elems].mean(axis=1)
    _, element_parent = cKDTree(old_centers).query(new_centers)
    def p1(values):
        source = np.asarray(values, dtype=float)
        projected = np.asarray(LinearNDInterpolator(old_nodes, source)(new_nodes))
        missing = ~np.all(np.isfinite(projected), axis=1) if projected.ndim == 2 else ~np.isfinite(projected)
        projected[missing] = source[node_parent[missing]]
        return projected

    ep = np.asarray(state.ep_gp, dtype=float)[:, element_parent]
    rho = np.asarray(state.rho_gp, dtype=float)[element_parent]
    new_area = np.asarray(mesh.area_e, dtype=float)
    old_area = np.asarray(state.mesh.area_e, dtype=float)
    for component in range(ep.shape[0]):
        source_integral = float(np.sum(np.asarray(state.ep_gp)[component] * old_area))
        projected_integral = float(np.sum(ep[component] * new_area))
        if abs(projected_integral) > 1.0e-300:
            ep[component] *= source_integral / projected_integral
    source_rho = float(np.sum(np.asarray(state.rho_gp) * old_area))
    projected_rho = float(np.sum(rho * new_area))
    if abs(projected_rho) > 1.0e-300:
        rho *= source_rho / projected_rho
    return {
        "damage": np.clip(p1(np.asarray(state.damage)), 0.0, 1.0),
        "displacement": p1(np.asarray(state.displacement).reshape(-1, 2)).reshape(-1),
        "ep_gp": ep,
        "rho_gp": rho,
        "tip_process_state": state.tip_process_state,
        "source_state": state.junction_process_state.get("source_state", {}),
    }


def _grow_hole_boundary(hole, radius_m):
    nodes = np.asarray(hole.mesh.nodes).copy()
    count = len(hole.prescribed_polygon_nodes)
    center = np.asarray(hole.center_m)
    theta = np.arctan2(nodes[:count, 1] - center[1], nodes[:count, 0] - center[0])
    polygon_radius = float(radius_m) / math.cos(math.pi / count)
    nodes[:count] = center + polygon_radius * np.c_[np.cos(theta), np.sin(theta)]
    mesh = rebuild_tri_mesh(nodes, np.asarray(hole.mesh.elems), tip_centers=np.asarray(hole.center_m))
    return replace(hole, mesh=mesh, radius_m=float(radius_m))


def remesh_cavity(state, hole, void_state, identity, operation_log=None, failure_stage=None):
    operations = operation_log if operation_log is not None else []
    trial = replace(state, void_state=void_state)
    fields = _project_fields(trial, hole.mesh)
    transfer_audit = {
        "source_displacement_l2_m": float(np.linalg.norm(state.displacement)),
        "projected_displacement_l2_m": float(np.linalg.norm(fields["displacement"])),
        "source_plastic_history_l2": float(np.linalg.norm(state.ep_gp)),
        "projected_plastic_history_l2": float(np.linalg.norm(fields["ep_gp"])),
        "source_density_l2_m2": float(np.linalg.norm(state.rho_gp)),
        "projected_density_l2_m2": float(np.linalg.norm(fields["rho_gp"])),
        "plastic_integral_error": float(np.max(np.abs(
            np.sum(fields["ep_gp"] * hole.mesh.area_e[None, :], axis=1)
            - np.sum(state.ep_gp * state.mesh.area_e[None, :], axis=1)
        ))),
        "density_integral_error": float(abs(
            np.sum(fields["rho_gp"] * hole.mesh.area_e)
            - np.sum(state.rho_gp * state.mesh.area_e)
        )),
        "projected_fields_nonzero": bool(
            np.linalg.norm(fields["displacement"]) > 0.0 and np.linalg.norm(fields["ep_gp"]) > 0.0
            and np.linalg.norm(fields["rho_gp"]) > 0.0
        ),
    }
    def inject(stage, current):
        operations.append(stage)
        if stage == failure_stage: raise RuntimeError("injected:" + stage)
    remesh_junction = dict(trial.junction_process_state)
    remesh_junction["boundary_terminal_context"] = _external_free_root_context(
        hole.mesh, hole.boundary, trial.crack_network.branch(ROOT_BRANCH_ID).root,
    )
    trial = replace(trial, junction_process_state=remesh_junction)
    rebuilt = remesh_mechanically_separating_v12(
        trial, mesh=hole.mesh, boundary=hole.boundary, transferred_fields=fields,
        source_commit=_head(), configuration={"voiding_enabled": True},
        transaction_identity=identity, failure_injector=inject,
    )
    junction = dict(rebuilt.junction_process_state)
    junction["latest_void_remesh_audit"] = transfer_audit
    rebuilt = replace(rebuilt, junction_process_state=junction)
    solved = equilibrate_fixed_load_with_production_fem(rebuilt)
    operations.append("equilibrium")
    if failure_stage == "equilibrium": raise RuntimeError("injected:equilibrium")
    return solved


def _tensor_fingerprint(tensor) -> str:
    return hashlib.sha256(np.asarray(tensor, dtype=np.float64).tobytes()).hexdigest()


def _source_identity(state, tensor, *, source_kind, source_front_id=None,
                     source_cavity_id=None, source_boundary_site_id=None,
                     source_position_m=None, source_probe_identity=None):
    return {
        "source_kind": source_kind,
        "source_front_id": source_front_id,
        "source_cavity_id": source_cavity_id,
        "source_boundary_site_id": source_boundary_site_id,
        "source_position_m": None if source_position_m is None else list(map(float, source_position_m)),
        "source_geometry_generation": int(state.crack_network.geometry_generation),
        "source_tensor_fingerprint": _tensor_fingerprint(tensor),
        "source_probe_identity": source_probe_identity,
    }


def _transition_competition_source(state, source, *, candidates=None):
    """Archive completed events and start source-owned fresh clocks."""
    junction = dict(state.junction_process_state)
    provenance = dict(junction.get("directional_event_provenance", {}))
    stale = list(junction.get("geometrically_stale_directional_events", ()))
    for event in state.competition.pending_events:
        record = dict(provenance[event.event_id])
        record.update({"status": "COMPLETED_BUT_GEOMETRICALLY_STALE",
                       "stale_at_geometry_generation": int(state.crack_network.geometry_generation)})
        provenance[event.event_id] = record
        stale.append(record)
    seed_payload = json.dumps({"source": source, "seed": state.competition.global_hazard_seed},
                              sort_keys=True, separators=(",", ":")).encode()
    seed = int.from_bytes(hashlib.sha256(seed_payload).digest()[:8], "big") & ((1 << 63) - 1)
    competition = DirectionalCompetitionState.initialize(
        state.competition.candidates if candidates is None else candidates,
        global_hazard_seed=seed,
    )
    source = dict(source)
    declared = {row["candidate_id"]: dict(row)
                for row in source.get("candidate_source_states", ())}
    source["candidate_source_states"] = tuple({
        **declared.get(candidate.candidate_id, {}),
        "candidate_id": candidate.candidate_id,
        "direction_xy": list(candidate.direction_xy),
        "normal_xy": list(candidate.normal_xy),
        "threshold_identity": {
            "threshold_process": hazard.threshold_process,
            "threshold_seed": hazard.threshold_seed,
            "threshold_action": hazard.current_threshold_action,
        },
        "rng_provenance": {"source_seed": seed,
                           "global_hazard_seed": competition.global_hazard_seed},
    } for candidate, hazard in zip(competition.candidates, competition.hazard_states))
    junction.update({"directional_event_provenance": provenance,
                     "geometrically_stale_directional_events": tuple(stale),
                     "active_event_source": source})
    return replace(state, competition=competition, junction_process_state=junction)


def _mark_consumed_event_provenance(state, event_ids):
    junction = dict(state.junction_process_state)
    provenance = dict(junction.get("directional_event_provenance", {}))
    for event_id in event_ids:
        if event_id in provenance:
            provenance[event_id] = {**provenance[event_id], "status": "CONSUMED_AT_OWNED_SOURCE"}
    junction["directional_event_provenance"] = provenance
    return replace(state, junction_process_state=junction)


def _mesh_aligned_principal_candidate(state, start, tensor, *, variant, target_distance_m,
                                      allow_kinetically_dormant=False):
    eigenvalues, eigenvectors = np.linalg.eigh(np.asarray(tensor, dtype=float))
    normal = eigenvectors[:, int(np.argmax(eigenvalues))]
    ideal = np.asarray((normal[1], -normal[0]))
    if ideal[0] < 0.0:
        ideal *= -1.0
    delta = np.asarray(state.mesh.nodes) - np.asarray(start)
    distance = np.linalg.norm(delta, axis=1)
    unit = delta / np.maximum(distance[:, None], 1.0e-300)
    forward = unit @ ideal
    normals = np.column_stack((-unit[:, 1], unit[:, 0]))
    resolved_opening = np.einsum("ni,ij,nj->n", normals, np.asarray(tensor, dtype=float), normals)
    geometric = ((distance >= 0.5 * target_distance_m)
                 & (distance <= 1.5 * target_distance_m)
                 & (forward > 0.8)
                 & (np.asarray(state.mesh.nodes)[:, 0] > 1.0e-12)
                 & (np.asarray(state.mesh.nodes)[:, 0] < 1.0e-3 - 1.0e-12)
                 & (np.abs(np.asarray(state.mesh.nodes)[:, 1]) < 5.0e-4 - 1.0e-12))
    tensile = np.flatnonzero(geometric & (resolved_opening > 0.0))
    eligible = tensile if len(tensile) or not allow_kinetically_dormant else np.flatnonzero(geometric)
    if not len(eligible):
        raise RuntimeError("no mesh-aligned forward cleavage endpoint")
    opening_scale = max(float(np.max(np.maximum(resolved_opening[eligible], 0.0))), 1.0)
    score = (1.0 - forward[eligible]) + 0.1 * np.abs(
        distance[eligible] - target_distance_m
    ) / target_distance_m + 0.1 * (1.0 - resolved_opening[eligible] / opening_scale)
    endpoint = int(eligible[int(np.argmin(score))])
    direction = unit[endpoint]
    aligned_normal = np.asarray((-direction[1], direction[0]))
    candidate = CleavageCandidate.create(
        plane_family="cleavage", plane_variant=variant,
        direction_xy=direction, normal_xy=aligned_normal, gamma_rel=1.0,
        orientation_convention="V5 mesh-aligned principal-opening renewal",
    )
    return candidate, tuple(map(float, state.mesh.nodes[endpoint]))


def _mesh_endpoint_on_direction(state, start, direction, target_distance_m):
    direction = np.asarray(direction, dtype=float)
    direction /= np.linalg.norm(direction)
    delta = np.asarray(state.mesh.nodes) - np.asarray(start)
    axial = delta @ direction
    normal = np.abs(delta[:, 0] * direction[1] - delta[:, 1] * direction[0])
    eligible = np.flatnonzero((axial > 0.5 * target_distance_m)
                              & (axial < 1.5 * target_distance_m)
                              & (normal <= 1.0e-10)
                              & (np.asarray(state.mesh.nodes)[:, 0] < 1.0e-3 - 1.0e-12)
                              & (np.abs(np.asarray(state.mesh.nodes)[:, 1]) < 5.0e-4 - 1.0e-12))
    if not len(eligible):
        return None
    node = int(eligible[np.argmin(np.abs(axial[eligible] - target_distance_m))])
    return tuple(map(float, state.mesh.nodes[node]))


def _complete_next_clock(state, stress_tensor_Pa, *, source_kind="sharp_front",
                         source_front_id=None, source_cavity_id=None,
                         source_boundary_site_id=None, source_position_m=None,
                         source_probe_identity=None, temperature_K=900.0,
                         maximum_advance_duration_s=None):
    """Advance all directional clocks through one common earliest-event time."""
    material = state.material
    engine = FrontEngine(
        FrontConfig(), default_cleavage_barrier(), default_emission_barrier(material.b),
        material.G, material.nu, material.b,
    )
    stress = np.asarray(stress_tensor_Pa, dtype=float).reshape(2, 2)
    junction = dict(state.junction_process_state)
    start_time = float(junction.get("production_time_s", 0.0))
    source = _source_identity(
        state, stress, source_kind=source_kind, source_front_id=source_front_id,
        source_cavity_id=source_cavity_id, source_boundary_site_id=source_boundary_site_id,
        source_position_m=source_position_m, source_probe_identity=source_probe_identity,
    )
    rates = []
    hazards = list(state.competition.hazard_states)
    crossing_times = []
    for candidate, hazard in zip(state.competition.candidates, hazards):
        normal = np.asarray(candidate.normal_xy, dtype=float)
        resolved_opening = max(float(normal @ stress @ normal), 0.0)
        raw_rate, _, barrier = engine.lambda_cleave(resolved_opening, temperature_K)
        zero_tolerance = 1.0e-12 * max(float(np.linalg.norm(stress, ord=2)), 1.0)
        rate = 0.0 if resolved_opening <= zero_tolerance else max(float(raw_rate), 0.0)
        remaining = max(hazard.current_threshold_action - hazard.action, 0.0)
        crossing = math.inf if rate <= 0.0 else remaining / rate
        crossing_times.append(crossing)
        rates.append({"candidate_id": candidate.candidate_id, "raw_rate_s": float(raw_rate),
                      "effective_rate_s": rate, "rate_s": rate,
                      "resolved_opening_stress_Pa": resolved_opening,
                      "hazard_barrier_J": barrier, "crossing_time_s": crossing})
    duration = min(crossing_times)
    if not math.isfinite(duration):
        duration = 0.0 if maximum_advance_duration_s is None else float(maximum_advance_duration_s)
        if duration < 0.0 or not math.isfinite(duration):
            raise ValueError("maximum_advance_duration_s must be finite and nonnegative")
        for rate in rates:
            rate.update({"common_advance_duration_s": duration, "emitted_event_ids": [],
                         "winner": False,
                         "instantaneous_status": "ZERO_DOWNSTREAM_DRIVE"})
        junction.update({
            "production_time_s": start_time + duration,
            "active_event_source": source,
            "latest_directional_clock_status": "NO_KINETICALLY_ACTIVE_CANDIDATE",
        })
        return replace(state, junction_process_state=junction), rates
    if maximum_advance_duration_s is not None:
        bounded = float(maximum_advance_duration_s)
        if bounded < 0.0 or not math.isfinite(bounded):
            raise ValueError("maximum_advance_duration_s must be finite and nonnegative")
        duration = min(duration, bounded)
    emitted_winner_ids = []
    for index, (hazard, rate) in enumerate(zip(hazards, rates)):
        preview = preview_directional_interval(
            hazard, lambda_per_s=rate["rate_s"], start_time_s=start_time,
            duration_s=duration,
        )
        hazards[index] = commit_directional_interval(hazard, preview)
        rate["common_advance_duration_s"] = duration
        rate["emitted_event_ids"] = [event.event_id for event in preview.completed_events]
        rate["winner"] = bool(preview.completed_events)
        if preview.completed_events:
            emitted_winner_ids.append(rate["candidate_id"])
    pending_new = {
        event.candidate_id for before, after in zip(state.competition.hazard_states, hazards)
        for event in after.pending_events[len(before.pending_events):]
    }
    if pending_new != set(emitted_winner_ids):
        raise AssertionError("pending events do not match emitted first-passage winners")
    provenance = dict(junction.get("directional_event_provenance", {}))
    threshold_by_candidate = {hazard.candidate_id: {
        "threshold_process": hazard.threshold_process,
        "threshold_seed": hazard.threshold_seed,
        "threshold_action": hazard.current_threshold_action,
    } for hazard in state.competition.hazard_states}
    for hazard in hazards:
        before_ids = {event.event_id for old in state.competition.hazard_states
                      for event in old.pending_events}
        for event in hazard.pending_events:
            if event.event_id in before_ids:
                continue
            provenance[event.event_id] = {
                **source, "completion_time_s": event.completion_time_s,
                "candidate_id": event.candidate_id,
                "threshold_identity": threshold_by_candidate[event.candidate_id],
                "status": "PENDING_SOURCE_VALIDATION",
            }
    junction.update({"production_time_s": start_time + duration,
                     "directional_event_provenance": provenance,
                     "active_event_source": source,
                     "latest_directional_clock_status": (
                         "FIRST_PASSAGE_COMPLETED" if emitted_winner_ids
                         else "KINETICALLY_ACTIVE_NO_COMPLETION"
                     )})
    return replace(state, competition=replace(state.competition, hazard_states=tuple(hazards)),
                   junction_process_state=junction), rates


def _select_emitted_proposal(state, audit, *, eligible_candidate_ids=None):
    emitted_ids = {event_id for row in audit for event_id in row["emitted_event_ids"]}
    eligible = None if eligible_candidate_ids is None else set(eligible_candidate_ids)
    proposals = tuple(
        proposal for proposal in construct_action_proposals(
            state.competition.hazard_states, correlation_interval_s=0.0,
        )
        if set(proposal.member_event_ids).issubset(emitted_ids)
        and proposal.action_type == "one_arm"
        and (eligible is None or set(proposal.member_candidate_ids).issubset(eligible))
    )
    proposal = select_temporal_or_degenerate_proposal(
        proposals, global_hazard_seed=state.competition.global_hazard_seed,
        competition_event_index=state.competition.competition_event_index,
    )
    winners = {row["candidate_id"] for row in audit if row["winner"]}
    if not set(proposal.member_candidate_ids).issubset(winners):
        raise AssertionError("selected proposal is not owned by emitted winners")
    return proposal


def ligament_transaction(state, *, failure_stage=None, operation_log=None):
    cavity = state.void_state.cavities[0]
    start = state.crack_network.branch(ROOT_BRANCH_ID).tip
    tensor, source_element = crack_tip_tensor(state, branch_id=ROOT_BRANCH_ID)
    state, cleavage_audit = _complete_next_clock(
        state, tensor, source_kind="sharp_front", source_front_id=ROOT_BRANCH_ID,
        source_position_m=start, source_probe_identity={"kind": "crack_tip_tensor",
                                                        "element_ids": list(source_element)},
    )
    intersections = {candidate.candidate_id: _first_ray_cavity_intersection(state, start, candidate.direction_xy)
                     for candidate in state.competition.candidates}
    eligible_ids = {candidate_id for candidate_id, intersection in intersections.items() if intersection is not None}
    proposal = _select_emitted_proposal(state, cleavage_audit, eligible_candidate_ids=eligible_ids)
    candidate = next(item for item in state.competition.candidates if item.candidate_id == proposal.member_candidate_ids[0])
    end = intersections[candidate.candidate_id]
    if end is None:
        raise RuntimeError("selected candidate ray does not intersect the cavity polygon")
    engine = FrontEngine(FrontConfig(), default_cleavage_barrier(), default_emission_barrier(state.material.b),
                         state.material.G, state.material.nu, state.material.b)
    winner = next(item for item in cleavage_audit if item["candidate_id"] == candidate.candidate_id)
    barrier = winner["hazard_barrier_J"]
    resistance = hazard_resistance_J_per_m2(
        barrier_J=barrier, cooperative_hits=engine.f.m_hits,
        burgers_vector_m=engine.b, gamma_relative=candidate.gamma_rel,
    )
    arm = TopologyArm(
        proposal.member_candidate_ids[0], ROOT_BRANCH_ID, start, end, math.dist(start, end),
        resistance * math.dist(start, end), event_classification="physical_cleavage",
        candidate_direction_xy=candidate.direction_xy, first_intersection_xy_m=end,
    )
    operations = operation_log if operation_log is not None else []
    def inject(stage, current):
        operations.append(stage)
        if stage == failure_stage: raise RuntimeError("injected:" + stage)
    def geometry(trial, arms):
        trial, aligned_node, inserted = _conform_cavity_intersection(
            trial, end, failure_injector=inject,
        )
        inject("intersection_alignment", trial)
        exit_point = _first_ray_cavity_intersection(trial, end, candidate.direction_xy)
        if exit_point is None:
            raise RuntimeError("connected cavity has no far-side chord intersection")
        trial, exit_node, exit_inserted = _conform_cavity_intersection(
            trial, exit_point, failure_injector=inject,
        )
        exit_point = tuple(map(float, trial.mesh.nodes[exit_node]))
        inject("exit_alignment", trial)
        realized = apply_v12_production_trial_geometry(
            trial, arms, source_commit=_head(), configuration={"event": "CRACK_TO_VOID_LIGAMENT"},
            transaction_identity="ligament", failure_injector=inject,
        )
        connected = replace(
            realized.void_state.cavities[0], phase=VoidPhase.CONNECTED_VOID,
            lineage=realized.void_state.cavities[0].lineage + ("CRACK_TO_VOID_LIGAMENT",),
            connection_entry_m=end, connection_exit_m=exit_point,
            connection_direction_xy=candidate.direction_xy,
        )
        void_state = replace_cavity(realized.void_state, connected)
        inject("cavity_phase_update", replace(realized, void_state=void_state))
        ledgers = dict(void_state.length_ledgers)
        ligament_length = math.dist(start, end)
        physical_span = math.dist(end, exit_point)
        projected_span = float(np.asarray(exit_point)[0] - np.asarray(end)[0])
        increments = {
            "fractured_ligament_length_m": ligament_length,
            "active_front_coordinate_advance_m": ligament_length,
            "physical_active_front_travel_m": ligament_length,
            "projected_fractured_length_m": end[0] - start[0],
            "projected_front_advance_m": end[0] - start[0],
            "preexisting_void_free_span_m": physical_span,
            "connected_void_free_span_m": physical_span,
            "projected_connected_void_free_span_m": projected_span,
            "connected_free_surface_extent_m": _polygon_boundary_arc_length(realized, end, exit_point),
        }
        for name, increment in increments.items():
            ledgers[name] += increment
            void_state = replace(void_state, length_ledgers=dict(ledgers))
            inject("length_ledger_update:" + name, replace(realized, void_state=void_state))
        void_state = replace(
            void_state,
            event_history=void_state.event_history + ({"event": "CRACK_TO_VOID_LIGAMENT",
                                                        "candidate_id": candidate.candidate_id,
                                                        "source_element": source_element},),
        )
        realized = replace(realized, void_state=void_state)
        root = realized.crack_network.branch(ROOT_BRANCH_ID)
        dormant_network = replace(
            realized.crack_network,
            branches=tuple(replace(
                branch, status="arrested",
                local_state={**branch.local_state,
                             "terminal_boundary_kind": "traction_free_cavity"},
            ) if branch.branch_id == ROOT_BRANCH_ID else branch
                           for branch in realized.crack_network.branches),
        )
        realized = replace(realized, crack_network=dormant_network)
        inject("root_status_change", realized)
        cycle_before_rebuild = cavity_free_surface_certificate(realized)
        contexts = {key: tuple(value) for key, value in
                    realized.junction_process_state.get("boundary_terminal_context", {}).items()}
        for arc_start, arc_end, arc_id in certification_arcs(dormant_network):
            if np.linalg.norm(np.asarray(arc_end) - np.asarray(end)) <= 1.0e-12:
                endpoint_name = "end"
            elif np.linalg.norm(np.asarray(arc_start) - np.asarray(end)) <= 1.0e-12:
                endpoint_name = "start"
            else:
                continue
            cavity_context = {
                "endpoint": endpoint_name, "endpoint_role": "inactive_terminal",
                "endpoint_coordinate_m": tuple(map(float, end)),
                "boundary_kind": "cavity_free_surface",
                "boundary_component_id": "cavity-cycle:" + cavity.cavity_id,
                "boundary_edge_ids": tuple(tuple(map(int, edge))
                                           for edge in cycle_before_rebuild["boundary_edge_ids"]),
                "cavity_id": cavity.cavity_id,
                "certified_cavity_id": cavity.cavity_id,
                "cavity_cycle_certified": cycle_before_rebuild["passed"],
                "tangent_enters_or_approaches_solid": True,
            }
            contexts[arc_id] = contexts.get(arc_id, ()) + (cavity_context,)
        junction = dict(realized.junction_process_state)
        junction["boundary_terminal_context"] = contexts
        realized = replace(realized, junction_process_state=junction)
        realized = initialize_mechanically_separating_v12(
            realized, source_commit=_head(),
            configuration={"event": "CRACK_TO_VOID_LIGAMENT", "root_status": "arrested"},
            transaction_identity="ligament-connected-dormant",
        )
        inject("dormant_support_rebuild", realized)
        cycle = cavity_free_surface_certificate(realized)
        certificate = crack_void_connection_certificate(
            realized, branch_id=ROOT_BRANCH_ID, cavity_id=cavity.cavity_id,
            intended_intersection=end,
        )
        if not certificate["passed"]:
            raise RuntimeError("connected free surface is not certified: " + repr(certificate))
        junction = dict(realized.junction_process_state)
        junction["latest_intersection_alignment"] = {
            "intersection_m": list(end), "aligned_node_id_before_refinement": aligned_node,
            "interior_edge_split_performed": inserted,
            "exit_intersection_m": list(exit_point), "exit_aligned_node_id": exit_node,
            "exit_interior_edge_split_performed": exit_inserted,
            "accepted_mesh_has_aligned_node": bool(np.min(np.linalg.norm(np.asarray(realized.mesh.nodes) - np.asarray(end), axis=1)) <= 1.0e-12),
        }
        junction["latest_closed_cavity_boundary_cycle_certificate"] = cycle
        junction["latest_crack_void_connection_certificate"] = certificate
        realized = replace(realized, junction_process_state=junction)
        inject("connected_surface_certification", realized)
        return realized
    result = execute_topology_trial(
        state, proposal, (arm,), apply_trial_geometry=geometry,
        equilibrate_fixed_load=equilibrate_fixed_load_with_production_fem,
        network_geometry_already_realized=True, failure_injector=inject,
    )
    if not result.accepted: raise RuntimeError("ligament event rejected")
    result_state = _mark_consumed_event_provenance(result.state, proposal.member_event_ids)
    cavity = result_state.void_state.cavities[0]
    exit_nodes = np.asarray(result_state.mesh.nodes)
    exit_node = int(np.argmin(np.linalg.norm(exit_nodes - np.asarray(cavity.connection_exit_m), axis=1)))
    surface_tensor, boundary_elements = cavity_boundary_tensor(result_state, boundary_node=exit_node)
    origin = np.asarray(cavity.connection_exit_m, dtype=float)
    candidate_inventory = tuple(result_state.competition.candidates)
    candidate_rows = []
    endpoints = {}
    for surface_candidate in candidate_inventory:
        direction = np.asarray(surface_candidate.direction_xy, dtype=float)
        direction /= np.linalg.norm(direction)
        endpoint = origin + 7.5e-5 * direction
        if not (1.0e-12 < endpoint[0] < 1.0e-3 - 1.0e-12
                and abs(endpoint[1]) < 5.0e-4 - 1.0e-12):
            continue
        normal = np.asarray(surface_candidate.normal_xy, dtype=float)
        opening = max(float(normal @ surface_tensor @ normal), 0.0)
        endpoints[surface_candidate.candidate_id] = list(map(float, endpoint))
        candidate_rows.append({
            "candidate_id": surface_candidate.candidate_id,
            "direction_xy": list(surface_candidate.direction_xy),
            "normal_xy": list(surface_candidate.normal_xy),
            "tangent_xy": list(surface_candidate.direction_xy),
            "planned_endpoint_m": endpoints[surface_candidate.candidate_id],
            "geometry_status": "GEOMETRICALLY_VALID_KINETICALLY_DORMANT"
                               if opening <= 0.0 else "GEOMETRICALLY_VALID_KINETICALLY_ACTIVE",
            "instantaneous_status": "ZERO_DOWNSTREAM_DRIVE"
                                    if opening <= 0.0 else "POSITIVE_DOWNSTREAM_DRIVE",
            "resolved_opening_stress_Pa": opening,
            "effective_rate_s": 0.0 if opening <= 0.0 else None,
            "crossing_time_s": "infinity" if opening <= 0.0 else None,
        })
    candidate_inventory = tuple(candidate for candidate in candidate_inventory
                                if candidate.candidate_id in endpoints)
    if not candidate_inventory:
        raise RuntimeError("connected cavity has no geometrically admissible downstream direction")
    surface_source = _source_identity(
        result_state, surface_tensor, source_kind="cavity_surface",
        source_cavity_id=cavity.cavity_id, source_boundary_site_id="connection_exit",
        source_position_m=cavity.connection_exit_m,
        source_probe_identity={"kind": "direct_cavity_boundary_tensor",
                               "boundary_node_id": exit_node,
                               "element_ids": list(boundary_elements)},
    )
    surface_source["candidate_source_states"] = tuple(candidate_rows)
    surface_source["next_candidate_endpoints_m"] = endpoints
    transitioned = _transition_competition_source(
        result_state, surface_source, candidates=candidate_inventory,
    )
    return transitioned, replace(result, state=transitioned)


def downstream_front_transaction(state, *, continuation=False, failure_stage=None, operation_log=None):
    child_id = "void-front-1"
    cavity = state.void_state.cavities[0]
    nodes = np.asarray(state.mesh.nodes)
    active_source = state.junction_process_state.get("active_event_source", {})
    if continuation:
        if state.crack_network.active_tip_ids != (child_id,):
            raise RuntimeError("continued propagation requires the sole active downstream child")
        start = state.crack_network.branch(child_id).tip
        tensor, boundary_elements = crack_tip_tensor(state, branch_id=child_id)
        source_kind = "sharp_front"
        probe_identity = {"kind": "child_crack_tip_tensor", "element_ids": list(boundary_elements)}
    else:
        if cavity.connection_exit_m is None:
            raise RuntimeError("downstream nucleation requires the stored connection exit")
        start = tuple(map(float, cavity.connection_exit_m))
        exit_node = int(np.argmin(np.linalg.norm(nodes - np.asarray(start), axis=1)))
        if np.linalg.norm(nodes[exit_node] - np.asarray(start)) > 1.0e-12:
            raise RuntimeError("stored connection exit is not an aligned cavity-boundary node")
        tensor, boundary_elements = cavity_boundary_tensor(state, boundary_node=exit_node)
        source_kind = "cavity_surface"
        probe_identity = {"kind": "direct_cavity_boundary_tensor", "boundary_node_id": exit_node,
                          "element_ids": list(boundary_elements)}
    state, cleavage_audit = _complete_next_clock(
        state, tensor, source_kind=source_kind,
        source_front_id=child_id if continuation else None,
        source_cavity_id=None if continuation else cavity.cavity_id,
        source_boundary_site_id=None if continuation else "connection_exit",
        source_position_m=start, source_probe_identity=probe_identity,
    )
    if not any(row["winner"] for row in cleavage_audit):
        return state, None, operation_log if operation_log is not None else [], {
            "status": "NO_KINETICALLY_ACTIVE_CANDIDATE",
            "tensor_Pa": tensor.tolist(), "boundary_element_ids": boundary_elements,
            "source_kind": source_kind, "source_front_id": child_id if continuation else None,
            "source_position_m": list(start), "source_probe_identity": probe_identity,
            "cleavage": cleavage_audit,
        }
    proposal = _select_emitted_proposal(state, cleavage_audit)
    candidate = next(item for item in state.competition.candidates
                     if item.candidate_id == proposal.member_candidate_ids[0])
    planned_endpoint = active_source.get("next_candidate_endpoints_m", {}).get(candidate.candidate_id)
    if planned_endpoint is None:
        planned_endpoint = active_source.get("next_mesh_aligned_endpoint_m")
    end = tuple(map(float, planned_endpoint or ()))
    if len(end) != 2:
        raise RuntimeError("event source lacks its mesh-aligned endpoint")
    if continuation:
        base_network = state.crack_network
    else:
        r_tip = max(float(state.mesh.hbar_tip), 10.0 * float(state.material.b))
        child = CrackBranchState(child_id, ROOT_BRANCH_ID, 1,
                                 int(state.event_counters.get("topology_actions", 0)) + 1,
                                 (start,), (candidate.angle_rad,), local_state={
                                     "nucleation_source": "direct_cavity_boundary_tensor",
                                     "upstream_lineage_branch_id": ROOT_BRANCH_ID,
                                     "active_source": "child_crack_tip_tensor",
                                     "r_tip_m": r_tip,
                                     "r_tip_initialization_policy": "fresh_moving_tip_renewal_no_historical_partition",
                                 })
        base_network = replace(state.crack_network, branches=state.crack_network.branches + (child,),
                               geometry_generation=state.crack_network.geometry_generation + 1,
                               branching_enabled=True)
    engine = FrontEngine(FrontConfig(), default_cleavage_barrier(), default_emission_barrier(state.material.b),
                         state.material.G, state.material.nu, state.material.b)
    winner = next(item for item in cleavage_audit if item["candidate_id"] == candidate.candidate_id)
    barrier = winner["hazard_barrier_J"]
    resistance = hazard_resistance_J_per_m2(
        barrier_J=barrier, cooperative_hits=engine.f.m_hits,
        burgers_vector_m=engine.b, gamma_relative=candidate.gamma_rel,
    )
    arm = TopologyArm(
        proposal.member_candidate_ids[0], child_id, start, end, math.dist(start, end),
        resistance * math.dist(start, end), event_classification="physical_cleavage",
        candidate_direction_xy=candidate.direction_xy, first_intersection_xy_m=end,
    )
    operations = operation_log if operation_log is not None else []
    def inject(stage, current):
        operations.append(stage)
        if stage == failure_stage:
            raise RuntimeError("injected:" + stage)
    def geometry(trial, arms):
        trial = _conform_bulk_point(
            trial, end,
            identity="CONTINUED_FRONT_ENDPOINT_CONFORMING" if continuation
                     else "DOWNSTREAM_FIRST_PASSAGE_ENDPOINT_CONFORMING",
            failure_injector=inject,
        )
        if not continuation:
            inject("downstream_child_activation", replace(trial, crack_network=base_network))
        realized = apply_v12_production_trial_geometry(
            replace(trial, crack_network=base_network), arms,
            source_commit=_head(), configuration={"event": "DOWNSTREAM_FRONT" if not continuation else "CONTINUED_FRONT"},
            transaction_identity="downstream-continued" if continuation else "downstream-first-passage",
            failure_injector=inject, refinement_levels=1,
        )
        cavity = realized.void_state.cavities[0]
        updated = replace(cavity, phase=VoidPhase.DOWNSTREAM_FRONT_ACTIVE,
                          lineage=cavity.lineage + (("CONTINUED_EVENT" if continuation else "DOWNSTREAM_FIRST_PASSAGE"),))
        void_state = replace_cavity(realized.void_state, updated)
        inject("downstream_phase_update", replace(realized, void_state=void_state))
        ledgers = dict(void_state.length_ledgers)
        projected = end[0] - start[0]
        physical_span = cavity.connection_entry_m is not None and cavity.connection_exit_m is not None
        chord_length = math.dist(cavity.connection_entry_m, cavity.connection_exit_m) if physical_span else 0.0
        chord_projected = (cavity.connection_exit_m[0] - cavity.connection_entry_m[0]) if physical_span else 0.0
        increments = {
            "ordinary_crack_fractured_length_m": math.dist(start, end),
            "active_front_coordinate_advance_m": math.dist(start, end) + (chord_length if not continuation else 0.0),
            "physical_active_front_travel_m": math.dist(start, end) + (chord_length if not continuation else 0.0),
            "projected_fractured_length_m": projected,
            "projected_front_advance_m": projected + (chord_projected if not continuation else 0.0),
            "projected_free_span_m": chord_projected if not continuation else 0.0,
            "traversed_void_free_span_m": chord_length if not continuation else 0.0,
        }
        for name, increment in increments.items():
            ledgers[name] += increment
            void_state = replace(void_state, length_ledgers=dict(ledgers))
            inject("downstream_length_ledger_update:" + name, replace(realized, void_state=void_state))
        void_state = replace(void_state, event_history=void_state.event_history + ({
            "event": "CONTINUED_ACCEPTED_EVENT" if continuation else "DOWNSTREAM_FIRST_PASSAGE",
            "candidate_id": candidate.candidate_id,
            "source_kind": source_kind, "source_front_id": child_id if continuation else None,
            "source_cavity_id": None if continuation else cavity.cavity_id,
        },))
        inject("downstream_event_history_update", replace(realized, void_state=void_state))
        tip_state = dict(realized.tip_process_state)
        if not continuation:
            tip_state.update({
                "active_branch_id": child_id,
                "by_branch": {child_id: {
                    "r_tip_m": r_tip,
                    "initialization_policy": "fresh_moving_tip_renewal_no_historical_partition",
                    "historical_state_imported": False,
                }},
            })
        realized = replace(realized, void_state=void_state, tip_process_state=tip_state)
        certificate = crack_void_connection_certificate(
            realized, branch_id=ROOT_BRANCH_ID, cavity_id=cavity.cavity_id,
            intended_intersection=cavity.connection_entry_m,
        )
        if not certificate["passed"]:
            raise RuntimeError("post-downstream combined topology is not certified: " + repr(certificate))
        junction = dict(realized.junction_process_state)
        junction["latest_crack_void_connection_certificate"] = certificate
        junction["latest_topology_certificate_stage"] = (
            "POST_CONTINUATION" if continuation else "ROOT_CAVITY_DOWNSTREAM_CHILD"
        )
        return replace(realized, junction_process_state=junction)
    result = execute_topology_trial(
        state, proposal, (arm,), apply_trial_geometry=geometry,
        equilibrate_fixed_load=equilibrate_fixed_load_with_production_fem,
        network_geometry_already_realized=True, failure_injector=inject,
    )
    if not result.accepted: raise RuntimeError("downstream front event rejected")
    accepted = _mark_consumed_event_provenance(result.state, proposal.member_event_ids)
    if not continuation:
        child_tensor, _ = crack_tip_tensor(accepted, branch_id=child_id)
        child_tip = np.asarray(accepted.crack_network.branch(child_id).tip, dtype=float)
        child_candidates = tuple(accepted.competition.candidates)
        child_endpoints = {}
        child_rows = []
        for child_candidate in child_candidates:
            direction = np.asarray(child_candidate.direction_xy, dtype=float)
            endpoint = child_tip + 7.5e-5 * direction
            if not (1.0e-12 < endpoint[0] < 1.0e-3 - 1.0e-12
                    and abs(endpoint[1]) < 5.0e-4 - 1.0e-12):
                continue
            normal = np.asarray(child_candidate.normal_xy, dtype=float)
            opening = max(float(normal @ child_tensor @ normal), 0.0)
            child_endpoints[child_candidate.candidate_id] = list(map(float, endpoint))
            child_rows.append({
                "candidate_id": child_candidate.candidate_id,
                "direction_xy": list(child_candidate.direction_xy),
                "normal_xy": list(child_candidate.normal_xy),
                "tangent_xy": list(child_candidate.direction_xy),
                "planned_endpoint_m": child_endpoints[child_candidate.candidate_id],
                "geometry_status": "GEOMETRICALLY_VALID_KINETICALLY_DORMANT"
                                   if opening <= 0.0 else "GEOMETRICALLY_VALID_KINETICALLY_ACTIVE",
                "instantaneous_status": "ZERO_DOWNSTREAM_DRIVE"
                                        if opening <= 0.0 else "POSITIVE_DOWNSTREAM_DRIVE",
                "resolved_opening_stress_Pa": opening,
                "effective_rate_s": 0.0 if opening <= 0.0 else None,
                "crossing_time_s": "infinity" if opening <= 0.0 else None,
            })
        child_candidates = tuple(item for item in child_candidates
                                 if item.candidate_id in child_endpoints)
        if not child_candidates:
            raise RuntimeError("active child has no geometrically admissible continuation direction")
        child_source = _source_identity(
            accepted, child_tensor, source_kind="sharp_front", source_front_id=child_id,
            source_position_m=accepted.crack_network.branch(child_id).tip,
            source_probe_identity={"kind": "child_crack_tip_tensor"},
        )
        child_source["candidate_source_states"] = tuple(child_rows)
        child_source["next_candidate_endpoints_m"] = child_endpoints
        accepted = _transition_competition_source(
            accepted, child_source, candidates=child_candidates,
        )
        result = replace(result, state=accepted)
    cavity = accepted.void_state.cavities[0]
    return accepted, result, operations, {
        "tensor_Pa": tensor.tolist(), "boundary_element_ids": boundary_elements,
        "source_kind": source_kind, "source_front_id": child_id if continuation else None,
        "source_position_m": list(start), "source_probe_identity": probe_identity,
        "r_tip_m": accepted.tip_process_state.get("by_branch", {}).get(child_id, {}).get("r_tip_m"),
        "candidate_id": proposal.member_candidate_ids[0],
        "selected_proposal_candidate_ids": list(proposal.member_candidate_ids),
        "emitted_winner_candidate_ids": [row["candidate_id"] for row in cleavage_audit if row["winner"]],
        "barrier_candidate_id": winner["candidate_id"],
        "cavity_id": cavity.cavity_id, "cleavage": cleavage_audit,
    }


def deterministic_trajectory(*, stop_before_ligament=False, cavity_center_m=(7.0e-4, 0.0),
                             crack_path_m=None, cleavage_theta_deg=0.0):
    state, hole = build_production_void_state(enabled=True, cavity_center_m=cavity_center_m,
                                              crack_path_m=crack_path_m,
                                              cleavage_theta_deg=cleavage_theta_deg)
    cfg = VoidingConfig(enabled=True, promotion_radius_m=5.0e-5)
    rows = [observables(state, "available_site")]
    for label in ("multi_hit_1", "multi_hit_2"):
        tensor = local_site_tensor(state); rates = arrhenius_rates(cfg, temperature_K=900.0, stress_tensor_Pa=tensor)
        site = state.void_state.sites[0]
        dt = max(site.birth.threshold - site.birth.accumulated, 0.0) / (rates["birth_s"] * site.candidate_weight)
        void_state, events = advance_site(state.void_state, site.site_id, dt, rates=rates)
        state = equilibrate_fixed_load_with_production_fem(replace(state, void_state=void_state))
        rows.append({**observables(state, label), "local_tensor_Pa": tensor.tolist(), "rates": rates, "events": events})
    tensor = local_site_tensor(state)
    rates = arrhenius_rates(cfg, temperature_K=900.0, stress_tensor_Pa=tensor)
    site = state.void_state.sites[0]
    dt = site.stabilization.threshold / rates["stabilization_s"]
    void_state, events = advance_site(state.void_state, site.site_id, dt, rates=rates)
    state = equilibrate_fixed_load_with_production_fem(replace(state, void_state=void_state))
    rows.append({**observables(state, "stabilization"), "local_tensor_Pa": tensor.tolist(), "rates": rates, "events": events})
    state = replace(state, void_state=create_subgrid_cavity(state.void_state, "site-1", 2.5e-5))
    state = equilibrate_fixed_load_with_production_fem(state); rows.append(observables(state, "subgrid_void"))
    tensor = local_site_tensor(state)
    rates = arrhenius_rates(cfg, temperature_K=900.0, stress_tensor_Pa=tensor)
    growth_dt = 2.5e-5 / (cfg.radial_growth_scale_m * rates["series_limited_growth_s"])
    state = replace(state, void_state=update_cavity_growth(
        state.void_state, state.void_state.cavities[0].cavity_id,
        rates=rates, dt_s=growth_dt, radial_growth_scale_m=cfg.radial_growth_scale_m,
    ))
    grown = state.void_state.cavities[0]
    state = equilibrate_fixed_load_with_production_fem(state)
    rows.append({**observables(state, "subgrid_growth"), "rates": rates, "growth_dt_s": growth_dt})
    promoted = promote_cavity(state.void_state, grown.cavity_id, cfg.promotion_radius_m)
    operations = []
    state = remesh_cavity(state, hole, promoted, "promotion", operations)
    rows.append({**observables(state, "geometric_promotion"), "executed_operations": operations})
    grown_hole = _grow_hole_boundary(hole, 5.5e-5)
    tensor = cavity_boundary_tensor(state)[0]
    rates = arrhenius_rates(cfg, temperature_K=900.0, stress_tensor_Pa=tensor)
    growth_dt = 0.5e-5 / (cfg.radial_growth_scale_m * rates["series_limited_growth_s"])
    void_state = update_cavity_growth(
        state.void_state, state.void_state.cavities[0].cavity_id,
        rates=rates, dt_s=growth_dt, radial_growth_scale_m=cfg.radial_growth_scale_m,
    )
    state = remesh_cavity(state, grown_hole, void_state, "resolved-growth")
    rows.append({**observables(state, "resolved_growth"), "rates": rates, "growth_dt_s": growth_dt})
    if stop_before_ligament:
        return state, rows
    state, result = ligament_transaction(state)
    rows.append({
        **observables(state, "ligament_rupture"),
        "energy_release_J_per_m": result.energy_release_J_per_m,
        "hazard_dissipation_J_per_m": result.hazard_dissipation_J_per_m,
        "energy_margin_J_per_m": result.energy_margin_J_per_m,
        "event_classification": "physical_cleavage",
    })
    rows.append(observables(state, "connected_topology"))
    tensor, boundary_elements = cavity_boundary_tensor(state)
    rates = arrhenius_rates(cfg, temperature_K=900.0, stress_tensor_Pa=tensor)
    rows.append({**observables(state, "downstream_surface_probe"), "direct_cavity_boundary_tensor_Pa": tensor.tolist(),
                 "cavity_boundary_element_ids": boundary_elements,
                 "void_birth_rate_s": rates["birth_s"],
                 "classification": "PRE_CLEAVAGE_SURFACE_PROBE"})
    state, result, operations, causal = downstream_front_transaction(state)
    rows.append({**observables(state, "new_graph_front"), "executed_operations": operations,
                 "energy_release_J_per_m": result.energy_release_J_per_m,
                 "causal_first_passage": causal})
    state, result, operations, causal = downstream_front_transaction(state, continuation=True)
    rows.append({**observables(state, "continued_accepted_event"), "executed_operations": operations,
                 "energy_release_J_per_m": result.energy_release_J_per_m,
                 "causal_first_passage": causal})
    return state, rows


def natural_trajectory(seed=3621, steps=6):
    state, _ = build_production_void_state(enabled=True, stochastic=True, seed=seed)
    cfg = VoidingConfig(enabled=True)
    rows = []
    for step in range(steps):
        tensor = local_site_tensor(state); rates = arrhenius_rates(cfg, temperature_K=900.0, stress_tensor_Pa=tensor)
        state = equilibrate_fixed_load_with_production_fem(state)
        void_state, events = advance_site(state.void_state, "site-1", 1.0e-12, rates=rates)
        state = replace(state, void_state=void_state)
        rows.append({**observables(state, f"natural_step_{step+1}"), "local_tensor_Pa": tensor.tolist(), "rates": rates, "events": events})
    return state, rows
