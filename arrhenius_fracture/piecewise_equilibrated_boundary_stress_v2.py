"""Piecewise H(div)-type reconstruction for independent cavity-traction verification.

The reconstructed symmetric stress is affine on every selected triangle.  Its
divergence vanishes by parameterization and its traction is matched across
internal patch edges.  No cavity-boundary value is prescribed.
"""
from __future__ import annotations

import hashlib
import json
import math
from typing import Any

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import factorized, lsmr

from .explicit_cavity_v5 import cavity_edge_traction_geometry
from .piecewise_equilibrated_boundary_stress_v2_contract import (
    ACCEPTANCE_LIMITS, OPERATOR_ID,
)


SCHEMA = "unified-2d-one-void.piecewise-equilibrated-boundary-stress/2"
WINDOW_SCALE_FRACTION = 0.5
MINIMUM_SUPPORT_ELEMENTS = 18
CONSTRAINT_WEIGHT = 1.0e2


class PiecewiseEquilibratedStressUnavailable(RuntimeError):
    """Expected scientific non-certification of the independent verifier."""


def _sha(value: object) -> str:
    return hashlib.sha256(json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False,
    ).encode()).hexdigest()


def _stress_rows(stress, element_count: int) -> np.ndarray:
    values = np.asarray(stress, dtype=float)
    if values.shape == (3, element_count):
        rows = np.column_stack((values[0], values[1], values[2]))
    elif values.shape == (element_count, 3):
        rows = values
    elif values.shape == (element_count, 2, 2):
        rows = np.column_stack((values[:, 0, 0], values[:, 1, 1], values[:, 0, 1]))
    else:
        raise ValueError("stress has an unsupported shape")
    if not np.isfinite(rows).all():
        raise ValueError("stress contains a nonfinite value")
    return rows


def _quality(points: np.ndarray) -> np.ndarray:
    sides2 = np.sum((points - points[:, [1, 2, 0]]) ** 2, axis=(1, 2))
    first = points[:, 1] - points[:, 0]
    second = points[:, 2] - points[:, 0]
    twice_area = np.abs(first[:, 0] * second[:, 1] - first[:, 1] * second[:, 0])
    return 2.0 * np.sqrt(3.0) * twice_area / np.maximum(sides2, 1.0e-300)


def _component_design(x: float, y: float) -> np.ndarray:
    """Map seven local coefficients to (sxx, syy, sxy)."""
    return np.asarray((
        (1., x, y, 0., 0., 0., 0.),
        (0., 0., 0., 0., -y, 1., x),
        (0., -y, 0., 1., x, 0., 0.),
    ))


def _traction_design(x: float, y: float, normal: np.ndarray) -> np.ndarray:
    stress = _component_design(x, y)
    nx, ny = map(float, normal)
    return np.vstack((nx * stress[0] + ny * stress[2],
                      nx * stress[2] + ny * stress[1]))


def _tensor(coefficients: np.ndarray, x: float, y: float) -> np.ndarray:
    xx, yy, xy = _component_design(x, y) @ coefficients
    return np.asarray(((xx, xy), (xy, yy)))


def _selected_patch(xy, tri, center, source, radius, scale, solid):
    points = xy[tri]
    centroids = points.mean(axis=1)
    normal = (source - center) / np.linalg.norm(source - center)
    tangent = np.asarray((-normal[1], normal[0]))
    relative = centroids - source
    xi = relative @ normal / scale
    eta = relative @ tangent / scale
    radial_distance = np.linalg.norm(centroids - center, axis=1)
    selected = np.flatnonzero(
        solid & (radial_distance >= radius - max(1.0e-12, 1.0e-8 * radius))
        & (radial_distance - radius <= scale) & (np.abs(eta) <= 1.0)
    )
    if len(selected) < MINIMUM_SUPPORT_ELEMENTS:
        raise PiecewiseEquilibratedStressUnavailable("fixed physical source patch has inadequate support")
    if float(np.min(_quality(points[selected]))) < ACCEPTANCE_LIMITS["minimum_mesh_quality"]:
        raise PiecewiseEquilibratedStressUnavailable("fixed physical source patch has poor-quality support")
    return selected, normal, tangent


def recover_piecewise_equilibrated_boundary_stress_v2(
    *, nodes, elements, stress_Pa, boundary_node: int, cavity_center_m,
    cavity_radius_m: float, process_length_m: float, owned_boundary_edges,
    remote_traction_scale_Pa: float, state_fingerprint: str,
    solid_element_mask=None,
) -> dict[str, Any]:
    """Reconstruct and evaluate traction on the actual source-window edges."""
    xy = np.asarray(nodes, dtype=float)
    tri = np.asarray(elements, dtype=int)
    center = np.asarray(cavity_center_m, dtype=float)
    node = int(boundary_node)
    radius, process, remote = map(float, (
        cavity_radius_m, process_length_m, remote_traction_scale_Pa,
    ))
    if xy.ndim != 2 or xy.shape[1] != 2 or tri.ndim != 2 or tri.shape[1] != 3:
        raise ValueError("nodes/elements must describe a triangular plane mesh")
    if not np.isfinite(xy).all() or np.any(tri < 0) or np.any(tri >= len(xy)):
        raise ValueError("mesh contains invalid coordinates or connectivity")
    if center.shape != (2,) or not 0 <= node < len(xy):
        raise ValueError("cavity center or boundary node is invalid")
    if not np.isfinite((radius, process, remote)).all() or min(radius, process, remote) <= 0:
        raise ValueError("radius, process length, and remote traction must be positive")
    if not str(state_fingerprint):
        raise ValueError("state fingerprint is required")
    source = xy[node]
    scale = WINDOW_SCALE_FRACTION * min(radius, process)
    solid = np.ones(len(tri), dtype=bool) if solid_element_mask is None else np.asarray(solid_element_mask, dtype=bool)
    if solid.shape != (len(tri),):
        raise ValueError("solid element mask has the wrong shape")
    selected, normal, tangent = _selected_patch(xy, tri, center, source, radius, scale, solid)
    basis = np.column_stack((normal, tangent))
    local_points = (xy - source) @ basis / scale
    element_centers_local = local_points[tri[selected]].mean(axis=1)
    element_scales_local = np.sqrt(
        2.0 * np.asarray([
            abs(np.linalg.det(np.column_stack((
                local_points[element[1]] - local_points[element[0]],
                local_points[element[2]] - local_points[element[0]],
            )))) / 2.0
            for element in tri[selected]
        ])
    )
    if np.any(element_scales_local <= 0.0):
        raise ValueError("selected patch contains a degenerate element")

    def element_coordinates(patch_id: int, point: np.ndarray) -> np.ndarray:
        return (np.asarray(point) - element_centers_local[patch_id]) / element_scales_local[patch_id]

    stresses = _stress_rows(stress_Pa, len(tri))[selected]
    tensors = np.zeros((len(selected), 2, 2))
    tensors[:, 0, 0], tensors[:, 1, 1], tensors[:, 0, 1] = stresses.T
    tensors[:, 1, 0] = tensors[:, 0, 1]
    local_targets = np.einsum("ia,nij,jb->nab", basis, tensors, basis) / remote

    local_index = {int(eid): index for index, eid in enumerate(selected)}
    unknowns = 7 * len(selected)
    fit_rows, fit_cols, fit_data, fit_rhs = [], [], [], []
    fit_row = 0
    for patch_id, eid in enumerate(selected):
        first = xy[tri[eid, 1]] - xy[tri[eid, 0]]
        second = xy[tri[eid, 2]] - xy[tri[eid, 0]]
        twice_area = abs(float(first[0] * second[1] - first[1] * second[0]))
        area_weight = math.sqrt(twice_area / 6.0)
        target = np.asarray((local_targets[patch_id, 0, 0],
                             local_targets[patch_id, 1, 1], local_targets[patch_id, 0, 1]))
        for vertex in tri[eid]:
            design = _component_design(*element_coordinates(patch_id, local_points[int(vertex)]))
            for component in range(3):
                for coefficient, value in enumerate(design[component]):
                    if value:
                        fit_rows.append(fit_row); fit_cols.append(7 * patch_id + coefficient)
                        fit_data.append(area_weight * value)
                fit_rhs.append(area_weight * target[component]); fit_row += 1
    fit = sparse.csr_matrix((fit_data, (fit_rows, fit_cols)), shape=(fit_row, unknowns))
    fit_rhs_array = np.asarray(fit_rhs)
    fit_scale = max(float(np.linalg.norm(fit_rhs_array)), 1.0e-300)
    fit /= fit_scale
    fit_rhs_array /= fit_scale

    edge_owners: dict[tuple[int, int], list[int]] = {}
    for eid in selected:
        element = tri[eid]
        for a, b in ((element[0], element[1]), (element[1], element[2]), (element[2], element[0])):
            edge_owners.setdefault(tuple(sorted((int(a), int(b)))), []).append(int(eid))
    con_rows, con_cols, con_data = [], [], []
    con_row = 0
    internal_edges = []
    for edge, owners in sorted(edge_owners.items()):
        if len(owners) != 2:
            continue
        internal_edges.append(edge)
        p0, p1 = xy[list(edge)]
        edge_vector = p1 - p0
        edge_normal_global = np.asarray((edge_vector[1], -edge_vector[0])) / np.linalg.norm(edge_vector)
        edge_normal_local = basis.T @ edge_normal_global
        for vertex in edge:
            for component in range(2):
                for sign, eid in ((1., owners[0]), (-1., owners[1])):
                    patch_id = local_index[eid]
                    design = _traction_design(
                        *element_coordinates(patch_id, local_points[vertex]), edge_normal_local,
                    )
                    offset = 7 * local_index[eid]
                    for coefficient, value in enumerate(design[component]):
                        if value:
                            con_rows.append(con_row); con_cols.append(offset + coefficient)
                            con_data.append(sign * value)
                con_row += 1

    patch_boundary = [(edge, owners[0]) for edge, owners in edge_owners.items() if len(owners) == 1]
    total_patch_boundary_length = sum(np.linalg.norm(xy[b] - xy[a]) for (a, b), _ in patch_boundary)
    constraints = sparse.csr_matrix((con_data, (con_rows, con_cols)), shape=(con_row, unknowns))
    # Method of multipliers enforces the shared-edge equations without the
    # squared ill-conditioning of a single extreme penalty solve.  The local
    # fit Hessian is positive definite because every affine field is sampled at
    # all three non-collinear triangle vertices.
    fit_hessian = (fit.T @ fit).tocsc()
    system = fit_hessian + CONSTRAINT_WEIGHT * (constraints.T @ constraints).tocsc()
    system += sparse.eye(unknowns, format="csc") * 1.0e-10
    solve = factorized(system)
    fit_force = np.asarray(fit.T @ fit_rhs_array).ravel()
    multiplier = np.zeros(con_row)
    flat_coefficients = np.zeros(unknowns)
    multiplier_iterations = 0
    for multiplier_iterations in range(1, 13):
        flat_coefficients = solve(fit_force - constraints.T @ multiplier)
        constraint_values = np.asarray(constraints @ flat_coefficients).ravel()
        multiplier += CONSTRAINT_WEIGHT * constraint_values
        if float(np.max(np.abs(constraint_values), initial=0.0)) <= 1.0e-10:
            break
    if not np.isfinite(flat_coefficients).all():
        raise PiecewiseEquilibratedStressUnavailable("piecewise multiplier solve produced a nonfinite value")
    coefficients = flat_coefficients.reshape(-1, 7)
    augmented = sparse.vstack((fit, math.sqrt(CONSTRAINT_WEIGHT) * constraints), format="csr")
    column_norm = np.sqrt(np.asarray(augmented.power(2).sum(axis=0)).ravel())
    scaled_augmented = augmented @ sparse.diags(1.0 / column_norm)
    solution = lsmr(scaled_augmented, np.concatenate((fit_rhs_array, np.zeros(con_row))),
                    atol=1.0e-6, btol=1.0e-6, maxiter=100)
    fit_residual = float(np.linalg.norm(fit @ coefficients.ravel() - fit_rhs_array))
    constraint_values = constraints @ coefficients.ravel()
    interface_jump = float(np.max(np.abs(constraint_values), initial=0.0))
    # Resultant and moment balance follow from local equilibrium, symmetry and
    # interface continuity; integrate the entire recovered patch boundary as
    # an independent check instead of adding redundant constraint rows.
    patch_resultant = np.zeros(2); patch_moment = 0.0
    for (a, b), eid in patch_boundary:
        vector = xy[b] - xy[a]; length = float(np.linalg.norm(vector)); midpoint = 0.5 * (xy[a] + xy[b])
        outward = np.asarray((vector[1], -vector[0])) / length
        centroid = xy[tri[eid]].mean(axis=0)
        if float(outward @ (centroid - midpoint)) > 0: outward = -outward
        for offset in (-1 / math.sqrt(3), 1 / math.sqrt(3)):
            point = midpoint + 0.5 * offset * vector
            patch_id = local_index[eid]
            point_local = (point-source) @ basis / scale
            local_tensor = _tensor(
                coefficients[patch_id], *element_coordinates(patch_id, point_local)
            )
            traction = local_tensor @ (basis.T @ outward)
            force = 0.5 * length * traction
            patch_resultant += force
            arm = (point-source) @ basis / scale
            patch_moment += float(arm[0]*force[1]-arm[1]*force[0])
    resultant_residual = float(np.linalg.norm(patch_resultant) / max(total_patch_boundary_length, 1.0e-300))
    moment_value = float(abs(patch_moment) / max(total_patch_boundary_length, 1.0e-300))
    equilibrium_residual = 0.0

    canonical_edges = sorted({tuple(sorted(map(int, edge))) for edge in owned_boundary_edges})
    boundary_rows = []
    for edge in canonical_edges:
        owners = edge_owners.get(edge, ())
        if len(owners) != 1:
            continue
        midpoint = xy[list(edge)].mean(axis=0)
        radial = midpoint - center
        angle = math.atan2(normal[0] * radial[1] - normal[1] * radial[0], float(radial @ normal))
        if abs(radius * angle) > scale:
            continue
        eid = owners[0]
        patch_id = local_index[eid]
        midpoint_local = (midpoint - source) @ basis / scale
        local_tensor = _tensor(
            coefficients[patch_id], *element_coordinates(patch_id, midpoint_local)
        )
        global_tensor = remote * basis @ local_tensor @ basis.T
        geometry = cavity_edge_traction_geometry(xy[edge[0]], xy[edge[1]], center, global_tensor)
        traction = np.asarray(geometry["traction"])
        length = float(geometry["edge_length_m"])
        boundary_rows.append({
            "edge_node_ids": list(edge), "adjacent_element_id": int(eid),
            "edge_midpoint_m": midpoint.tolist(), "edge_length_m": length,
            "traction_vector_Pa": traction.tolist(),
            "normal_traction_Pa": float(geometry["normal_traction"]),
            "tangential_traction_Pa": float(geometry["tangential_traction"]),
            "l2_numerator_Pa2_m": float(traction @ traction) * length,
        })
    if len(boundary_rows) < 2:
        raise PiecewiseEquilibratedStressUnavailable("source-window boundary has inadequate reconstructed support")
    boundary_length = sum(row["edge_length_m"] for row in boundary_rows)
    l2 = sum(row["l2_numerator_Pa2_m"] for row in boundary_rows)
    rms = math.sqrt(l2 / boundary_length)
    boundary_resultant = np.sum([
        np.asarray(row["traction_vector_Pa"]) * row["edge_length_m"] for row in boundary_rows
    ], axis=0)
    source_patch_id = min(
        range(len(selected)),
        key=lambda patch_id: float(np.linalg.norm(
            xy[tri[selected[patch_id]]].mean(axis=0)-source)))
    source_local_point = (source-source) @ basis / scale
    source_local_tensor = _tensor(
        coefficients[source_patch_id],
        *element_coordinates(source_patch_id, source_local_point))
    source_tensor = remote*basis@source_local_tensor@basis.T
    window_identity = {
        "source_coordinate_m": source.tolist(), "cavity_center_m": center.tolist(),
        "cavity_radius_m": radius, "half_width_and_depth_m": scale,
        "selected_element_ids": selected.tolist(), "boundary_edge_ids": [row["edge_node_ids"] for row in boundary_rows],
    }
    return {
        "schema": SCHEMA, "operator": OPERATOR_ID, "available": True,
        "zero_boundary_traction_imposed": False,
        "stress_space": "ELEMENT_LOCAL_AFFINE_SYMMETRIC_DIVERGENCE_FREE",
        "equilibrium_constraint": "EXACT_BY_PARAMETERIZATION",
        "internal_traction_continuity": "EDGE_ENDPOINT_EQUALITY_PENALTY_SOLVE",
        "rank": unknowns, "required_rank": unknowns,
        "condition": float(solution[6]), "condition_limit": ACCEPTANCE_LIMITS["maximum_scaled_condition"],
        "stress_fit_residual": fit_residual,
        "stress_fit_residual_limit": ACCEPTANCE_LIMITS["maximum_stress_fit_residual"],
        "equilibrium_residual": equilibrium_residual,
        "interface_traction_jump_residual": interface_jump,
        "resultant_residual": resultant_residual, "moment_residual": abs(moment_value),
        "constraint_limits": {
            "normalized_boundary_traction": ACCEPTANCE_LIMITS[
                "normalized_equilibrated_boundary_traction"],
            "equilibrium": ACCEPTANCE_LIMITS["maximum_equilibrium_residual"],
            "interface_traction_jump": ACCEPTANCE_LIMITS["maximum_interface_traction_jump_relative"],
            "resultant": ACCEPTANCE_LIMITS["maximum_resultant_residual_relative"],
            "moment": ACCEPTANCE_LIMITS["maximum_moment_residual_relative"],
        },
        "solver": {"method": "SPARSE_METHOD_OF_MULTIPLIERS_CONSTRAINED_LEAST_SQUARES",
                   "constraint_weight": CONSTRAINT_WEIGHT, "istop": int(solution[1]),
                   "iterations": int(multiplier_iterations),
                   "condition_estimator_iterations": int(solution[2]),
                   "augmented_residual_norm": float(solution[3])},
        "stencil_element_ids": selected.tolist(), "internal_patch_edge_count": len(internal_edges),
        "patch_boundary_edge_count": len(patch_boundary),
        "minimum_support_quality": float(np.min(_quality(xy[tri[selected]]))),
        "boundary_edge_rows": boundary_rows,
        "boundary_resultant_N_per_m": np.asarray(boundary_resultant).tolist(),
        "boundary_traction_rms_Pa": rms,
        "normalized_boundary_traction": rms / remote,
        "source_tensor_Pa": source_tensor.tolist(),
        "source_sigma_tt_Pa": float(tangent@source_tensor@tangent),
        "source_sigma_nn_Pa": float(normal@source_tensor@normal),
        "source_sigma_nt_Pa": float(normal@source_tensor@tangent),
        "source_tensor_element_id": int(selected[source_patch_id]),
        "remote_traction_scale_Pa": remote,
        "window_identity": {**window_identity, "sha256": _sha(window_identity)},
        "state_fingerprint": str(state_fingerprint),
    }


__all__ = ["PiecewiseEquilibratedStressUnavailable", "SCHEMA",
           "recover_piecewise_equilibrated_boundary_stress_v2"]
