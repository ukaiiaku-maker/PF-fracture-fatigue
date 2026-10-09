"""Fixed-physical-window cavity stress recovery for the V5 downstream source.

The operator and its physical window are frozen before the central DBTT V3
readiness evaluation. V1 and V2 remain retained evidence and are not modified.
"""
from __future__ import annotations

import hashlib
import json
from typing import Any, Mapping

import numpy as np


RECOVERY_ID = "CAVITY_FIXED_PHYSICAL_ARC_PATCH_RECOVERY_V3"
SCHEMA = "v5.cavity-fixed-physical-arc-patch-recovery/3"
WINDOW_SCALE_FRACTION = 0.5
POLYNOMIAL_ORDER = 2
MINIMUM_SUPPORT_ELEMENTS = 12
MINIMUM_ELEMENT_QUALITY = 0.05
MAXIMUM_DESIGN_CONDITION = 1.0e6
TENSOR_RELATIVE_TOLERANCE = 0.05
TRACTION_RESIDUAL_TOLERANCE = 0.05
MINIMUM_QUALITY_VALID_FINE_LEVELS = 2
REQUIRED_MATERIAL_FINGERPRINTS = (
    "core_model_id",
    "material_bundle_id",
    "material_bundle_sha256",
    "elasticity_fingerprint",
    "plasticity_fingerprint",
    "FrontConfig_fingerprint",
    "cleavage_barrier_fingerprint",
    "emission_barrier_fingerprint",
    "process_zone_fingerprint",
)


class CavitySourceRecoveryV3Unavailable(RuntimeError):
    """Expected scientific non-certification of the V3 recovery."""


def _canonical_json_sha256(value: object) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def _canonical_edges(edges) -> tuple[tuple[int, int], ...]:
    return tuple(sorted({tuple(sorted((int(edge[0]), int(edge[1])))) for edge in edges}))


def _tensor_rows(stress: np.ndarray, element_count: int) -> np.ndarray:
    values = np.asarray(stress, dtype=float)
    if values.shape == (3, element_count):
        rows = np.stack((values[0], values[1], values[2]), axis=1)
    elif values.shape == (element_count, 3):
        rows = values
    elif values.shape == (element_count, 2, 2):
        rows = np.stack(
            (values[:, 0, 0], values[:, 1, 1], values[:, 0, 1]), axis=1
        )
    else:
        raise ValueError("stress must have shape (3, ne), (ne, 3), or (ne, 2, 2)")
    if not np.isfinite(rows).all():
        raise ValueError("stress contains a nonfinite value")
    return rows


def _triangle_quality(points: np.ndarray) -> np.ndarray:
    sides2 = np.sum((points - points[:, [1, 2, 0]]) ** 2, axis=(1, 2))
    cross = (
        (points[:, 1, 0] - points[:, 0, 0])
        * (points[:, 2, 1] - points[:, 0, 1])
        - (points[:, 1, 1] - points[:, 0, 1])
        * (points[:, 2, 0] - points[:, 0, 0])
    )
    return 2.0 * np.sqrt(3.0) * np.abs(cross) / np.maximum(sides2, 1.0e-300)


def _physical_arc_coordinates(
    points: np.ndarray, center: np.ndarray, source_normal: np.ndarray, radius: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    radial = points - center
    radial_distance = np.linalg.norm(radial, axis=1)
    if np.any(radial_distance <= 0.0):
        raise CavitySourceRecoveryV3Unavailable(
            "physical arc window contains a point at the cavity center"
        )
    point_normals = radial / radial_distance[:, None]
    point_tangents = np.column_stack((-point_normals[:, 1], point_normals[:, 0]))
    cross = source_normal[0] * point_normals[:, 1] - source_normal[1] * point_normals[:, 0]
    dot = point_normals @ source_normal
    angle = np.arctan2(cross, dot)
    return radius * angle, radial_distance - radius, point_normals, point_tangents


def _weighted_fit(
    design: np.ndarray,
    values: np.ndarray,
    weights: np.ndarray,
    *,
    required_rank: int,
    name: str,
) -> tuple[np.ndarray, int, float, np.ndarray]:
    weighted = np.sqrt(weights)[:, None] * design
    rank = int(np.linalg.matrix_rank(weighted))
    condition = float(np.linalg.cond(weighted))
    if rank < required_rank:
        raise CavitySourceRecoveryV3Unavailable(f"{name} design is rank deficient")
    if not np.isfinite(condition) or condition > MAXIMUM_DESIGN_CONDITION:
        raise CavitySourceRecoveryV3Unavailable(f"{name} design is poorly conditioned")
    coefficients = np.linalg.lstsq(
        weighted, np.sqrt(weights) * values, rcond=None
    )[0]
    return coefficients, rank, condition, design @ coefficients


def recover_fixed_physical_arc_patch_v3(
    *,
    nodes,
    elements,
    stress_Pa,
    boundary_node: int,
    cavity_id: str,
    cavity_center_m,
    cavity_radius_m: float,
    process_length_m: float,
    poisson_ratio: float,
    owned_boundary_edges,
    material_fingerprints: Mapping[str, str],
    state_fingerprint: str,
    solid_element_mask=None,
) -> dict[str, Any]:
    """Recover the source tensor from one mesh-independent physical arc window."""
    xy = np.asarray(nodes, dtype=float)
    tri = np.asarray(elements, dtype=int)
    if xy.ndim != 2 or xy.shape[1] != 2 or tri.ndim != 2 or tri.shape[1] != 3:
        raise ValueError("nodes/elements must describe a two-dimensional triangular mesh")
    if not np.isfinite(xy).all() or np.any(tri < 0) or np.any(tri >= len(xy)):
        raise ValueError("mesh contains invalid coordinates or connectivity")
    node = int(boundary_node)
    if node < 0 or node >= len(xy):
        raise ValueError("boundary node is outside the mesh")
    center = np.asarray(cavity_center_m, dtype=float)
    radius = float(cavity_radius_m)
    process_length = float(process_length_m)
    nu = float(poisson_ratio)
    if (
        center.shape != (2,)
        or not np.isfinite(radius + process_length + nu)
        or radius <= 0.0
        or process_length <= 0.0
        or not -1.0 < nu < 0.5
    ):
        raise ValueError("cavity, process length, or plane-strain Poisson data are invalid")
    edges = _canonical_edges(owned_boundary_edges)
    if not edges or not any(node in edge for edge in edges):
        raise CavitySourceRecoveryV3Unavailable(
            "the exact source node is absent from the owned cavity boundary"
        )
    position = xy[node]
    source_radial = position - center
    source_distance = float(np.linalg.norm(source_radial))
    geometry_tolerance = max(1.0e-12, 1.0e-8 * radius)
    if abs(source_distance - radius) > geometry_tolerance:
        raise CavitySourceRecoveryV3Unavailable(
            "the exact source coordinate is inconsistent with the owned cavity"
        )
    normal = source_radial / source_distance
    tangent = np.asarray((-normal[1], normal[0]))

    identities = {str(key): str(value) for key, value in material_fingerprints.items()}
    missing = [key for key in REQUIRED_MATERIAL_FINGERPRINTS if not identities.get(key)]
    if missing or not str(state_fingerprint):
        raise CavitySourceRecoveryV3Unavailable(
            "incomplete material/state ownership: " + ",".join(missing or ("state_fingerprint",))
        )

    if solid_element_mask is None:
        solid = np.ones(len(tri), dtype=bool)
    else:
        solid = np.asarray(solid_element_mask, dtype=bool)
        if solid.shape != (len(tri),):
            raise ValueError("solid element mask has the wrong shape")
    points = xy[tri]
    centroids = points.mean(axis=1)
    s_all, n_all, point_normals, point_tangents = _physical_arc_coordinates(
        centroids, center, normal, radius
    )
    physical_scale = min(radius, process_length)
    tangential_half_width = WINDOW_SCALE_FRACTION * physical_scale
    normal_depth = WINDOW_SCALE_FRACTION * physical_scale
    solid_side_tolerance = max(1.0e-12, 1.0e-8 * physical_scale)
    selected = np.flatnonzero(
        solid
        & (np.abs(s_all) <= tangential_half_width)
        & (n_all >= -solid_side_tolerance)
        & (n_all <= normal_depth)
    )
    if len(selected) < MINIMUM_SUPPORT_ELEMENTS:
        raise CavitySourceRecoveryV3Unavailable(
            "fixed physical arc window has inadequate solid support"
        )
    selected_points = points[selected]
    quality = _triangle_quality(selected_points)
    if np.any(quality < MINIMUM_ELEMENT_QUALITY):
        raise CavitySourceRecoveryV3Unavailable(
            "fixed physical arc window contains poor-quality support"
        )

    s = s_all[selected]
    n = np.maximum(n_all[selected], 0.0)
    xi = s / tangential_half_width
    zeta = n / normal_depth
    design_tt = np.column_stack(
        (np.ones(len(selected)), xi, zeta, xi * xi, xi * zeta, zeta * zeta)
    )
    # Multiplication by n enforces sigma_nn(s,0)=sigma_nt(s,0)=0 exactly.
    design_traction = zeta[:, None] * np.column_stack(
        (np.ones(len(selected)), xi, zeta)
    )
    sides = selected_points[:, 1:] - selected_points[:, :1]
    areas = np.abs(
        sides[:, 0, 0] * sides[:, 1, 1] - sides[:, 0, 1] * sides[:, 1, 0]
    ) / 2.0
    kernel_s = np.maximum(1.0 - xi * xi, 0.0) ** 2
    kernel_n = np.maximum(1.0 - zeta * zeta, 0.0) ** 2
    raw_weights = areas * kernel_s * kernel_n
    if not np.isfinite(raw_weights).all() or float(np.sum(raw_weights)) <= 0.0:
        raise CavitySourceRecoveryV3Unavailable("fixed physical window has zero weight")
    weights = raw_weights / np.sum(raw_weights)

    rows = _tensor_rows(stress_Pa, len(tri))[selected]
    global_tensors = np.zeros((len(selected), 2, 2), dtype=float)
    global_tensors[:, 0, 0] = rows[:, 0]
    global_tensors[:, 1, 1] = rows[:, 1]
    global_tensors[:, 0, 1] = global_tensors[:, 1, 0] = rows[:, 2]
    local_nn = np.einsum(
        "ni,nij,nj->n", point_normals[selected], global_tensors, point_normals[selected]
    )
    local_tt = np.einsum(
        "ni,nij,nj->n", point_tangents[selected], global_tensors, point_tangents[selected]
    )
    local_nt = np.einsum(
        "ni,nij,nj->n", point_normals[selected], global_tensors, point_tangents[selected]
    )
    coefficient_tt, rank_tt, condition_tt, predicted_tt = _weighted_fit(
        design_tt, local_tt, weights, required_rank=6, name="sigma_tt"
    )
    coefficient_nn, rank_nn, condition_nn, predicted_nn = _weighted_fit(
        design_traction, local_nn, weights, required_rank=3, name="sigma_nn"
    )
    coefficient_nt, rank_nt, condition_nt, predicted_nt = _weighted_fit(
        design_traction, local_nt, weights, required_rank=3, name="sigma_nt"
    )
    observed = np.column_stack((local_nn, local_tt, local_nt))
    predicted = np.column_stack((predicted_nn, predicted_tt, predicted_nt))
    fit_residual = float(
        np.sqrt(np.sum(weights[:, None] * (predicted - observed) ** 2))
        / max(np.sqrt(np.sum(weights[:, None] * observed**2)), 1.0e-300)
    )

    sigma_tt = float(coefficient_tt[0])
    tensor = sigma_tt * np.outer(tangent, tangent)
    sigma_zz = nu * float(np.trace(tensor))
    mean_stress = (float(np.trace(tensor)) + sigma_zz) / 3.0
    traction_residual = 0.0
    window = {
        "definition": "abs(s)<=0.5*min(R_void,L_pz); 0<=n<=0.5*min(R_void,L_pz)",
        "coordinate_system": "circular-cavity arc length s and outward radial depth n",
        "cavity_radius_m": radius,
        "canonical_process_length_m": process_length,
        "physical_scale_m": physical_scale,
        "tangential_half_width_m": tangential_half_width,
        "normal_depth_m": normal_depth,
        "polynomial_order": POLYNOMIAL_ORDER,
        "sigma_tt_basis": ["1", "s/S", "n/N", "(s/S)^2", "s*n/(S*N)", "(n/N)^2"],
        "sigma_nn_sigma_nt_basis": ["(n/N)", "(n/N)*(s/S)", "(n/N)^2"],
        "weight": "triangle_area*(1-(s/S)^2)^2*(1-(n/N)^2)^2",
        "equilibrium_constraints": {
            "included": False,
            "reason": "not independently qualified for the curvilinear polynomial basis",
        },
    }
    physical_identity = {
        "cavity_id": str(cavity_id),
        "cavity_center_m": center.tolist(),
        "cavity_radius_m": radius,
        "boundary_position_m": position.tolist(),
        "normal_xy": normal.tolist(),
        "tangent_xy": tangent.tolist(),
        "window": window,
    }
    physical_identity["sha256"] = _canonical_json_sha256(physical_identity)
    return {
        "schema": SCHEMA,
        "recovery_operator": RECOVERY_ID,
        "available": True,
        "preserved_v1_status": "FAIL_NONCONVERGENT",
        "preserved_v2_operator": "CAVITY_FIXED_ARC_PATCH_RECOVERY_V2",
        "physical_arc_and_window_identity": physical_identity,
        "physical_arc_identity": physical_identity,
        "owned_boundary_edges": [list(edge) for edge in edges],
        "boundary_node_id": node,
        "element_ids": selected.tolist(),
        "stencil_element_ids": selected.tolist(),
        "sample_count": int(len(selected)),
        "element_centroids_global_m": centroids[selected].tolist(),
        "element_coordinates_s_n_m": np.column_stack((s, n)).tolist(),
        "weights": weights.tolist(),
        "minimum_element_quality": float(np.min(quality)),
        "rank": {"sigma_tt": rank_tt, "sigma_nn": rank_nn, "sigma_nt": rank_nt},
        "condition": {
            "sigma_tt": condition_tt,
            "sigma_nn": condition_nn,
            "sigma_nt": condition_nt,
            "maximum": max(condition_tt, condition_nn, condition_nt),
            "limit": MAXIMUM_DESIGN_CONDITION,
        },
        "coefficients_Pa": {
            "sigma_tt": coefficient_tt.tolist(),
            "sigma_nn": coefficient_nn.tolist(),
            "sigma_nt": coefficient_nt.tolist(),
        },
        "constrained_fit_residual": fit_residual,
        "traction_free_enforcement": "sigma_nn(s,0)=sigma_nt(s,0)=0",
        "traction_residual": traction_residual,
        "sigma_tt_Pa": sigma_tt,
        "sigma_nn_Pa": 0.0,
        "sigma_nt_Pa": 0.0,
        "tensor_Pa": tensor.tolist(),
        "plane_strain_sigma_zz_Pa": sigma_zz,
        "mean_stress_Pa": mean_stress,
        "plane_strain_constitutive_rule": "sigma_zz=nu*(sigma_xx+sigma_yy)",
        "material_fingerprints": identities,
        "state_fingerprint": str(state_fingerprint),
    }


__all__ = [
    "CavitySourceRecoveryV3Unavailable",
    "MAXIMUM_DESIGN_CONDITION",
    "MINIMUM_ELEMENT_QUALITY",
    "MINIMUM_QUALITY_VALID_FINE_LEVELS",
    "RECOVERY_ID",
    "TENSOR_RELATIVE_TOLERANCE",
    "TRACTION_RESIDUAL_TOLERANCE",
    "recover_fixed_physical_arc_patch_v3",
]
