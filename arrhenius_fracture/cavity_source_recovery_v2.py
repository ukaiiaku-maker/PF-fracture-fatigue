"""Fixed-arc cavity stress recovery used by the V5 downstream source.

This contract is frozen before examining the retained DBTT readiness result.
The historical incident-CST maximum-principal sampler remains V1 evidence and
is not used by this operator.
"""
from __future__ import annotations

import hashlib
import json
from typing import Any

import numpy as np


RECOVERY_ID = "CAVITY_FIXED_ARC_PATCH_RECOVERY_V2"
V1_RECOVERY_STATUS = "FAIL_NONCONVERGENT"
PATCH_RING_COUNT = 2
MINIMUM_SUPPORT_ELEMENTS = 6
MINIMUM_ELEMENT_QUALITY = 0.05
MAXIMUM_DESIGN_CONDITION = 1.0e6
TENSOR_RELATIVE_TOLERANCE = 0.05
TRACTION_RESIDUAL_TOLERANCE = 0.05
MINIMUM_QUALITY_VALID_FINE_LEVELS = 2


class CavitySourceRecoveryUnavailable(RuntimeError):
    """Expected scientific non-certification of the fixed-arc recovery."""


def _tensor_rows(stress: np.ndarray, element_count: int) -> np.ndarray:
    values = np.asarray(stress, dtype=float)
    if values.shape == (3, element_count):
        return np.stack((values[0], values[1], values[2]), axis=1)
    if values.shape == (element_count, 3):
        return values
    if values.shape == (element_count, 2, 2):
        return np.stack((values[:, 0, 0], values[:, 1, 1], values[:, 0, 1]), axis=1)
    raise ValueError("stress must have shape (3, ne), (ne, 3), or (ne, 2, 2)")


def _triangle_quality(points: np.ndarray) -> np.ndarray:
    sides2 = np.sum((points - points[:, [1, 2, 0]]) ** 2, axis=(1, 2))
    cross = ((points[:, 1, 0] - points[:, 0, 0]) *
             (points[:, 2, 1] - points[:, 0, 1]) -
             (points[:, 1, 1] - points[:, 0, 1]) *
             (points[:, 2, 0] - points[:, 0, 0]))
    return 2.0 * np.sqrt(3.0) * np.abs(cross) / np.maximum(sides2, 1.0e-300)


def _canonical_edges(edges) -> tuple[tuple[int, int], ...]:
    return tuple(sorted({tuple(sorted((int(edge[0]), int(edge[1])))) for edge in edges}))


def recover_fixed_arc_patch_v2(
    *,
    nodes,
    elements,
    stress_Pa,
    boundary_node: int,
    cavity_id: str,
    cavity_center_m,
    cavity_radius_m: float,
    owned_boundary_edges,
    solid_element_mask=None,
) -> dict[str, Any]:
    """Recover the solid-side boundary tensor from a deterministic two-ring patch.

    The affine weighted least-squares fit is evaluated at the exact owned arc
    coordinate.  The independently recovered raw normal and shear components
    define the traction residual; the returned source tensor enforces the
    traction-free boundary limit while retaining the fitted tangential stress.
    """
    xy = np.asarray(nodes, dtype=float)
    tri = np.asarray(elements, dtype=int)
    if xy.ndim != 2 or xy.shape[1] != 2 or tri.ndim != 2 or tri.shape[1] != 3:
        raise ValueError("nodes/elements must describe a two-dimensional triangular mesh")
    node = int(boundary_node)
    if node < 0 or node >= len(xy):
        raise ValueError("boundary node is outside the mesh")
    center = np.asarray(cavity_center_m, dtype=float)
    radius = float(cavity_radius_m)
    if center.shape != (2,) or not np.isfinite(radius) or radius <= 0.0:
        raise ValueError("cavity center and radius are invalid")
    edges = _canonical_edges(owned_boundary_edges)
    if not edges or not any(node in edge for edge in edges):
        raise ValueError("the fixed arc node is absent from the owned cavity boundary")
    position = xy[node]
    radial = position - center
    radial_norm = float(np.linalg.norm(radial))
    if radial_norm <= 0.0 or radial_norm > 1.02 * radius:
        raise ValueError("the fixed arc coordinate is not on the declared cavity")
    normal = radial / radial_norm
    tangent = np.asarray((-normal[1], normal[0]))

    if solid_element_mask is None:
        solid = np.ones(len(tri), dtype=bool)
    else:
        solid = np.asarray(solid_element_mask, dtype=bool)
        if solid.shape != (len(tri),):
            raise ValueError("solid element mask has the wrong shape")
    incident = set(map(int, np.flatnonzero(solid & np.any(tri == node, axis=1))))
    if not incident:
        raise CavitySourceRecoveryUnavailable("fixed arc has no solid-side incident element")
    rings = {element: 1 for element in incident}
    selected = set(incident)
    frontier = set(incident)
    for ring in range(2, PATCH_RING_COUNT + 1):
        frontier_nodes = set(map(int, tri[sorted(frontier)].ravel()))
        following = set(map(int, np.flatnonzero(
            solid & np.any(np.isin(tri, tuple(frontier_nodes)), axis=1)
        ))) - selected
        for element in following:
            rings[element] = ring
        selected.update(following)
        frontier = following
    stencil = np.asarray(sorted(selected), dtype=int)
    if len(stencil) < MINIMUM_SUPPORT_ELEMENTS:
        raise CavitySourceRecoveryUnavailable("fixed arc patch has inadequate solid support")

    points = xy[tri[stencil]]
    quality = _triangle_quality(points)
    valid = quality >= MINIMUM_ELEMENT_QUALITY
    stencil = stencil[valid]
    points = points[valid]
    quality = quality[valid]
    if len(stencil) < MINIMUM_SUPPORT_ELEMENTS:
        raise CavitySourceRecoveryUnavailable("fixed arc patch lacks quality-valid support")
    centroids = points.mean(axis=1)
    offsets = centroids - position
    tangent_offset = offsets @ tangent
    normal_offset = offsets @ normal
    solid_side_tolerance = max(1.0e-12, 1.0e-8 * radius)
    solid_side = normal_offset >= -solid_side_tolerance
    stencil = stencil[solid_side]
    points = points[solid_side]
    quality = quality[solid_side]
    centroids = centroids[solid_side]
    tangent_offset = tangent_offset[solid_side]
    normal_offset = normal_offset[solid_side]
    if len(stencil) < MINIMUM_SUPPORT_ELEMENTS:
        raise CavitySourceRecoveryUnavailable("fixed arc patch lacks solid-side support")

    distances = np.linalg.norm(centroids - position, axis=1)
    patch_scale = float(np.max(distances))
    if not np.isfinite(patch_scale) or patch_scale <= 0.0:
        raise CavitySourceRecoveryUnavailable("fixed arc patch has zero spatial extent")
    design = np.column_stack((np.ones(len(stencil)),
                              tangent_offset / patch_scale,
                              normal_offset / patch_scale))
    first_side = points[:, 1] - points[:, 0]
    second_side = points[:, 2] - points[:, 0]
    areas = np.abs(first_side[:, 0] * second_side[:, 1] -
                   first_side[:, 1] * second_side[:, 0]) / 2.0
    raw_weights = areas / (1.0 + (distances / patch_scale) ** 2)
    weights = raw_weights / np.sum(raw_weights)
    weighted_design = np.sqrt(weights)[:, None] * design
    rank = int(np.linalg.matrix_rank(weighted_design))
    condition = float(np.linalg.cond(weighted_design))
    if rank < 3:
        raise CavitySourceRecoveryUnavailable("fixed arc patch fit is rank deficient")
    if not np.isfinite(condition) or condition > MAXIMUM_DESIGN_CONDITION:
        raise CavitySourceRecoveryUnavailable("fixed arc patch fit is poorly conditioned")

    stress_rows = _tensor_rows(stress_Pa, len(tri))[stencil]
    global_tensors = np.zeros((len(stencil), 2, 2), dtype=float)
    global_tensors[:, 0, 0] = stress_rows[:, 0]
    global_tensors[:, 1, 1] = stress_rows[:, 1]
    global_tensors[:, 0, 1] = global_tensors[:, 1, 0] = stress_rows[:, 2]
    local = np.column_stack((
        np.einsum("i,nij,j->n", normal, global_tensors, normal),
        np.einsum("i,nij,j->n", tangent, global_tensors, tangent),
        np.einsum("i,nij,j->n", normal, global_tensors, tangent),
    ))
    coefficients = np.linalg.lstsq(weighted_design, np.sqrt(weights)[:, None] * local,
                                    rcond=None)[0]
    raw_nn, raw_tt, raw_nt = map(float, coefficients[0])
    raw_local_norm = float(np.linalg.norm(((raw_nn, raw_nt), (raw_nt, raw_tt))))
    traction_residual = float(np.hypot(raw_nn, raw_nt) / max(raw_local_norm, 1.0e-300))
    recovered = raw_tt * np.outer(tangent, tangent)

    arc_payload = {
        "cavity_id": str(cavity_id),
        "cavity_center_m": center.tolist(),
        "cavity_radius_m": radius,
        "boundary_position_m": position.tolist(),
        "owned_boundary_edges": [list(edge) for edge in edges],
    }
    arc_fingerprint = hashlib.sha256(json.dumps(
        arc_payload, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()).hexdigest()
    return {
        "schema": "v5.cavity-fixed-arc-patch-recovery/2",
        "recovery_operator": RECOVERY_ID,
        "v1_incident_cst_max_principal_status": V1_RECOVERY_STATUS,
        "available": True,
        "physical_arc_identity": {**arc_payload, "sha256": arc_fingerprint},
        "boundary_node_id": node,
        "normal_xy": normal.tolist(),
        "tangent_xy": tangent.tolist(),
        "patch_ring_count": PATCH_RING_COUNT,
        "stencil_element_ids": stencil.tolist(),
        "stencil_ring_by_element": {str(int(element)): int(rings[int(element)]) for element in stencil},
        "weights": weights.tolist(),
        "minimum_element_quality": float(np.min(quality)),
        "design_rank": rank,
        "design_condition": condition,
        "maximum_design_condition": MAXIMUM_DESIGN_CONDITION,
        "patch_scale_m": patch_scale,
        "fit_basis": ["1", "tangent_offset/patch_scale", "normal_offset/patch_scale"],
        "raw_boundary_local_tensor_Pa": [[raw_nn, raw_nt], [raw_nt, raw_tt]],
        "raw_sigma_nn_Pa": raw_nn,
        "raw_sigma_tt_Pa": raw_tt,
        "raw_sigma_nt_Pa": raw_nt,
        "traction_free_enforcement": "sigma_nn=sigma_nt=0; sigma_tt=weighted-affine-solid-side-limit",
        "traction_residual": traction_residual,
        "tensor_Pa": recovered.tolist(),
        "sigma_nn_Pa": 0.0,
        "sigma_tt_Pa": raw_tt,
        "sigma_nt_Pa": 0.0,
    }
