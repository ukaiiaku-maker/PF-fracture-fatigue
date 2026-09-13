"""Independent equilibrated recovery of cavity-boundary traction.

The fitted symmetric stress is generated from one Airy polynomial, so its
divergence is identically zero and its traction is continuous inside the
patch.  Boundary traction is never prescribed: it is evaluated after fitting
on the actual owned cavity edges.
"""
from __future__ import annotations

import hashlib
import json
from typing import Any

import numpy as np

from .production_closure_contract import FALLBACK_OPERATOR_ID


SCHEMA = "unified-2d-one-void.equilibrated-boundary-traction-recovery/1"
WINDOW_SCALE_FRACTION = 0.5
AIRY_MAXIMUM_DEGREE = 8
MINIMUM_SUPPORT_ELEMENTS = 18
MINIMUM_ELEMENT_QUALITY = 0.05
MAXIMUM_DESIGN_CONDITION = 1.0e6
MAXIMUM_FIT_RESIDUAL = 0.25


class EquilibratedBoundaryTractionUnavailable(RuntimeError):
    """Expected scientific non-certification of the equilibrated recovery."""


def _sha(value: object) -> str:
    return hashlib.sha256(json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False,
    ).encode()).hexdigest()


def _canonical_edges(edges) -> tuple[tuple[int, int], ...]:
    return tuple(sorted({tuple(sorted(map(int, edge))) for edge in edges}))


def _stress_rows(stress, element_count: int) -> np.ndarray:
    values = np.asarray(stress, dtype=float)
    if values.shape == (3, element_count):
        rows = np.column_stack((values[0], values[1], values[2]))
    elif values.shape == (element_count, 3):
        rows = values
    elif values.shape == (element_count, 2, 2):
        rows = np.column_stack((values[:, 0, 0], values[:, 1, 1], values[:, 0, 1]))
    else:
        raise ValueError("stress must have shape (3, ne), (ne, 3), or (ne, 2, 2)")
    if not np.isfinite(rows).all():
        raise ValueError("stress contains a nonfinite value")
    return rows


def _quality(points: np.ndarray) -> np.ndarray:
    sides2 = np.sum((points - points[:, [1, 2, 0]]) ** 2, axis=(1, 2))
    first = points[:, 1] - points[:, 0]
    second = points[:, 2] - points[:, 0]
    twice_area = np.abs(first[:, 0] * second[:, 1] - first[:, 1] * second[:, 0])
    return 2.0 * np.sqrt(3.0) * twice_area / np.maximum(sides2, 1.0e-300)


_AIRY_POWERS = tuple(
    (i, degree - i)
    for degree in range(2, AIRY_MAXIMUM_DEGREE + 1)
    for i in range(degree + 1)
)


def _derivative(power: tuple[int, int], xi: np.ndarray, eta: np.ndarray, dx: int, dy: int):
    i, j = power
    if i < dx or j < dy:
        return np.zeros_like(xi)
    factor = 1
    for offset in range(dx):
        factor *= i - offset
    for offset in range(dy):
        factor *= j - offset
    return factor * xi ** (i - dx) * eta ** (j - dy)


def _design(xi: np.ndarray, eta: np.ndarray) -> np.ndarray:
    """Rows are sigma_nn, sigma_tt, sigma_nt for each sampling point."""
    rows = []
    for x, y in zip(np.asarray(xi), np.asarray(eta)):
        xx = np.asarray((x,), dtype=float)
        yy = np.asarray((y,), dtype=float)
        rows.extend((
            [_derivative(power, xx, yy, 0, 2)[0] for power in _AIRY_POWERS],
            [_derivative(power, xx, yy, 2, 0)[0] for power in _AIRY_POWERS],
            [-_derivative(power, xx, yy, 1, 1)[0] for power in _AIRY_POWERS],
        ))
    return np.asarray(rows, dtype=float)


def _evaluate(coefficients: np.ndarray, xi: np.ndarray, eta: np.ndarray) -> np.ndarray:
    flat = (_design(xi, eta) @ coefficients).reshape(-1, 3)
    tensors = np.zeros((len(flat), 2, 2), dtype=float)
    tensors[:, 0, 0] = flat[:, 0]
    tensors[:, 1, 1] = flat[:, 1]
    tensors[:, 0, 1] = tensors[:, 1, 0] = flat[:, 2]
    return tensors


def recover_equilibrated_boundary_traction_v1(
    *, nodes, elements, stress_Pa, boundary_node: int, cavity_center_m,
    cavity_radius_m: float, process_length_m: float, owned_boundary_edges,
    remote_traction_scale_Pa: float, state_fingerprint: str,
    solid_element_mask=None,
) -> dict[str, Any]:
    """Fit and independently evaluate an equilibrated local stress field."""
    xy = np.asarray(nodes, dtype=float)
    tri = np.asarray(elements, dtype=int)
    if xy.ndim != 2 or xy.shape[1] != 2 or tri.ndim != 2 or tri.shape[1] != 3:
        raise ValueError("nodes/elements must describe a two-dimensional triangular mesh")
    if not np.isfinite(xy).all() or np.any(tri < 0) or np.any(tri >= len(xy)):
        raise ValueError("mesh contains invalid coordinates or connectivity")
    center = np.asarray(cavity_center_m, dtype=float)
    radius = float(cavity_radius_m)
    process_length = float(process_length_m)
    remote = float(remote_traction_scale_Pa)
    node = int(boundary_node)
    if center.shape != (2,) or node < 0 or node >= len(xy):
        raise ValueError("cavity center or boundary node is invalid")
    if not np.isfinite((radius, process_length, remote)).all() or min(radius, process_length, remote) <= 0.0:
        raise ValueError("cavity, process length, and reaction-derived traction scale must be positive")
    if not str(state_fingerprint):
        raise ValueError("state fingerprint is required")
    edges = _canonical_edges(owned_boundary_edges)
    if not edges or not any(node in edge for edge in edges):
        raise EquilibratedBoundaryTractionUnavailable("source node is absent from the owned cavity boundary")
    source = xy[node]
    radial = source - center
    distance = float(np.linalg.norm(radial))
    if abs(distance - radius) > max(1.0e-12, 1.0e-8 * radius):
        raise EquilibratedBoundaryTractionUnavailable("source node is inconsistent with the cavity radius")
    normal = radial / distance
    tangent = np.asarray((-normal[1], normal[0]))
    basis = np.column_stack((normal, tangent))
    scale = WINDOW_SCALE_FRACTION * min(radius, process_length)

    points = xy[tri]
    centroids = points.mean(axis=1)
    relative = centroids - source
    xi_all = relative @ normal / scale
    eta_all = relative @ tangent / scale
    radial_distance = np.linalg.norm(centroids - center, axis=1)
    solid = np.ones(len(tri), dtype=bool) if solid_element_mask is None else np.asarray(solid_element_mask, dtype=bool)
    if solid.shape != (len(tri),):
        raise ValueError("solid element mask has the wrong shape")
    selected = np.flatnonzero(
        solid & (radial_distance >= radius - max(1.0e-12, 1.0e-8 * radius))
        & (radial_distance - radius <= scale) & (np.abs(eta_all) <= 1.0)
    )
    if len(selected) < MINIMUM_SUPPORT_ELEMENTS:
        raise EquilibratedBoundaryTractionUnavailable("fixed physical source window has inadequate support")
    quality = _quality(points[selected])
    if float(np.min(quality)) < MINIMUM_ELEMENT_QUALITY:
        raise EquilibratedBoundaryTractionUnavailable("fixed physical source window has poor-quality support")

    first = points[selected, 1] - points[selected, 0]
    second = points[selected, 2] - points[selected, 0]
    areas = 0.5 * np.abs(first[:, 0] * second[:, 1] - first[:, 1] * second[:, 0])
    xi = xi_all[selected]
    eta = eta_all[selected]
    kernel = np.maximum(1.0 - eta**2, 0.0) ** 2 * np.maximum(1.0 - xi**2, 0.0) ** 2
    weights = areas * kernel
    positive = weights > np.finfo(float).eps * max(float(np.max(weights)), 1.0)
    if int(np.count_nonzero(positive)) < MINIMUM_SUPPORT_ELEMENTS:
        raise EquilibratedBoundaryTractionUnavailable("fixed physical source window has inadequate positive weight")
    selected = selected[positive]
    xi, eta, weights = xi[positive], eta[positive], weights[positive]
    weights = weights / np.sum(weights)

    global_rows = _stress_rows(stress_Pa, len(tri))[selected]
    global_tensors = np.zeros((len(selected), 2, 2), dtype=float)
    global_tensors[:, 0, 0] = global_rows[:, 0]
    global_tensors[:, 1, 1] = global_rows[:, 1]
    global_tensors[:, 0, 1] = global_tensors[:, 1, 0] = global_rows[:, 2]
    local_tensors = np.einsum("ai,nij,jb->nab", basis, global_tensors, basis)
    observed = np.column_stack((
        local_tensors[:, 0, 0], local_tensors[:, 1, 1], local_tensors[:, 0, 1],
    )).reshape(-1)
    design = _design(xi, eta)
    component_weights = np.repeat(weights, 3)
    weighted_design = np.sqrt(component_weights)[:, None] * design
    weighted_observed = np.sqrt(component_weights) * observed
    rank = int(np.linalg.matrix_rank(weighted_design))
    condition = float(np.linalg.cond(weighted_design))
    if rank < len(_AIRY_POWERS):
        raise EquilibratedBoundaryTractionUnavailable("equilibrated Airy design is rank deficient")
    if not np.isfinite(condition) or condition > MAXIMUM_DESIGN_CONDITION:
        raise EquilibratedBoundaryTractionUnavailable("equilibrated Airy design is poorly conditioned")
    coefficients = np.linalg.lstsq(weighted_design, weighted_observed, rcond=None)[0]
    predicted = design @ coefficients
    fit_residual = float(np.linalg.norm(np.sqrt(component_weights) * (predicted - observed)) /
                         max(np.linalg.norm(weighted_observed), 1.0e-300))
    if not np.isfinite(fit_residual) or fit_residual > MAXIMUM_FIT_RESIDUAL:
        raise EquilibratedBoundaryTractionUnavailable("equilibrated Airy fit residual exceeds its frozen bound")

    edge_array = np.asarray(edges, dtype=int)
    midpoints = xy[edge_array].mean(axis=1)
    edge_radial = midpoints - center
    angles = np.arctan2(
        normal[0] * edge_radial[:, 1] - normal[1] * edge_radial[:, 0],
        edge_radial @ normal,
    )
    keep = np.abs(radius * angles) <= scale
    boundary_edges = edge_array[keep]
    boundary_midpoints = midpoints[keep]
    if len(boundary_edges) < 2:
        raise EquilibratedBoundaryTractionUnavailable("actual source-window boundary has inadequate edges")
    edge_vectors = xy[boundary_edges[:, 1]] - xy[boundary_edges[:, 0]]
    lengths = np.linalg.norm(edge_vectors, axis=1)
    boundary_normals = boundary_midpoints - center
    boundary_normals /= np.linalg.norm(boundary_normals, axis=1)[:, None]
    boundary_relative = boundary_midpoints - source
    boundary_local = _evaluate(
        coefficients, boundary_relative @ normal / scale,
        boundary_relative @ tangent / scale,
    )
    boundary_global = np.einsum("ia,nab,jb->nij", basis, boundary_local, basis)
    traction = np.einsum("nij,nj->ni", boundary_global, boundary_normals)
    resultant = np.sum(lengths[:, None] * traction, axis=0)
    rms = float(np.sqrt(np.sum(lengths * np.sum(traction**2, axis=1)) / np.sum(lengths)))
    normalized = rms / remote
    source_local = _evaluate(coefficients, np.asarray((0.0,)), np.asarray((0.0,)))[0]
    source_global = basis @ source_local @ basis.T
    window_identity = {
        "source_coordinate_m": source.tolist(),
        "cavity_center_m": center.tolist(),
        "cavity_radius_m": radius,
        "half_width_and_depth_m": scale,
        "owned_boundary_coordinates_m": xy[np.unique(edge_array)].tolist(),
    }
    return {
        "schema": SCHEMA,
        "operator": FALLBACK_OPERATOR_ID,
        "available": True,
        "zero_boundary_traction_imposed": False,
        "equilibrium_constraint": "AIRY_STRESS_DIVERGENCE_IDENTICALLY_ZERO",
        "internal_traction_continuity": "SINGLE_CONTINUOUS_POLYNOMIAL_STRESS_FIELD",
        "airy_maximum_degree": AIRY_MAXIMUM_DEGREE,
        "rank": rank,
        "required_rank": len(_AIRY_POWERS),
        "condition": condition,
        "condition_limit": MAXIMUM_DESIGN_CONDITION,
        "fit_residual": fit_residual,
        "fit_residual_limit": MAXIMUM_FIT_RESIDUAL,
        "stencil_element_ids": selected.tolist(),
        "weights": weights.tolist(),
        "minimum_support_quality": float(np.min(_quality(points[selected]))),
        "source_tensor_Pa": source_global.tolist(),
        "boundary_edge_ids": boundary_edges.tolist(),
        "boundary_traction_vectors_Pa": traction.tolist(),
        "boundary_resultant_N_per_m": resultant.tolist(),
        "boundary_traction_rms_Pa": rms,
        "normalized_boundary_traction": normalized,
        "remote_traction_scale_Pa": remote,
        "window_identity": {**window_identity, "sha256": _sha(window_identity)},
        "state_fingerprint": str(state_fingerprint),
    }


__all__ = [
    "AIRY_MAXIMUM_DEGREE", "EquilibratedBoundaryTractionUnavailable",
    "MAXIMUM_DESIGN_CONDITION", "MAXIMUM_FIT_RESIDUAL", "SCHEMA",
    "recover_equilibrated_boundary_traction_v1",
]
