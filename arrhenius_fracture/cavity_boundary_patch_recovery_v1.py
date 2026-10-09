"""Versioned, unconstrained, geometry-only boundary stress postprocessing.

An admissible regression is NOT a scientific source qualification. This module
never changes a FEM state and never owns a first-passage clock.
"""
from __future__ import annotations

import hashlib
import math
import numpy as np

OPERATOR_ID = "CAVITY_BOUNDARY_PATCH_RECOVERY_V1"
MAX_CONDITION_NUMBER = 1.0e6
ADJACENCY_RINGS = 2


class RecoveryUnavailable(ValueError):
    """Fail-closed geometric or numerical recovery rejection."""


def _fingerprint(array):
    value = np.ascontiguousarray(array)
    return hashlib.sha256(str((value.dtype.str, value.shape)).encode() + value.tobytes()).hexdigest()


def _stress_tensors(stress, ne):
    stress = np.asarray(stress, dtype=float)
    if stress.shape == (3, ne):
        out = np.empty((ne, 2, 2))
        out[:, 0, 0], out[:, 1, 1] = stress[0], stress[1]
        out[:, 0, 1] = out[:, 1, 0] = stress[2]
    elif stress.shape == (ne, 2, 2):
        out = stress.copy()
    else:
        raise RecoveryUnavailable("stress must be CST (3, ne) or (ne, 2, 2)")
    if not np.isfinite(out).all() or not np.array_equal(out, out.transpose(0, 2, 1)):
        raise RecoveryUnavailable("nonfinite or nonsymmetric stress")
    return out


def recover_boundary_tensor(nodes, elements, stress, *, boundary_edges, source_point,
                            normal, intact_mask=None, candidate_normals=()):
    """Recover a full tensor without constraints at an actual boundary site.

    ``normal`` is the fixed geometric radial convention, with tangent=(-ny,nx).
    Boundary edge order and endpoint order have no effect. An intact mask must
    exclude V12 support/damage elements; omission means every element is intact.
    """
    nodes = np.asarray(nodes, dtype=float)
    elements = np.asarray(elements, dtype=int)
    point = np.asarray(source_point, dtype=float)
    normal = np.asarray(normal, dtype=float)
    if (nodes.ndim != 2 or nodes.shape[1] != 2 or elements.ndim != 2 or
            elements.shape[1] != 3 or point.shape != (2,) or normal.shape != (2,) or
            not np.isfinite(nodes).all() or not np.isfinite(point).all() or
            not np.isfinite(normal).all() or abs(np.linalg.norm(normal)-1.) > 1e-12 or
            not len(elements) or np.min(elements) < 0 or np.max(elements) >= len(nodes)):
        raise RecoveryUnavailable("invalid mesh, source, or unit normal")
    raw = _stress_tensors(stress, len(elements))
    intact = np.ones(len(elements), dtype=bool) if intact_mask is None else np.asarray(intact_mask, dtype=bool)
    if intact.shape != (len(elements),):
        raise RecoveryUnavailable("invalid intact mask")
    tri = nodes[elements]
    cross = ((tri[:, 1, 0]-tri[:, 0, 0])*(tri[:, 2, 1]-tri[:, 0, 1]) -
             (tri[:, 1, 1]-tri[:, 0, 1])*(tri[:, 2, 0]-tri[:, 0, 0]))
    areas = np.abs(cross)*.5
    if np.any(areas <= 0):
        raise RecoveryUnavailable("degenerate element")
    owners = {}
    for eid, triangle in enumerate(elements):
        for a, b in ((triangle[0], triangle[1]), (triangle[1], triangle[2]), (triangle[2], triangle[0])):
            owners.setdefault(tuple(sorted((int(a), int(b)))), []).append(eid)
    boundary_edges = np.asarray(boundary_edges, dtype=int)
    if boundary_edges.ndim != 2 or boundary_edges.shape[1] != 2 or not len(boundary_edges):
        raise RecoveryUnavailable("invalid boundary edge array")
    selected_edges = sorted(set(tuple(sorted(map(int, edge))) for edge in boundary_edges))
    scale = max(float(np.ptp(nodes, axis=0).max()), np.finfo(float).tiny)
    geometric_eps = 64*np.finfo(float).eps*scale
    seeds, source_edges = set(), []
    for edge in selected_edges:
        if len(owners.get(edge, ())) != 1:
            raise RecoveryUnavailable("cavity edge does not have exactly one solid owner")
        a, b = nodes[list(edge)]
        delta = b-a
        fraction = float((point-a)@delta/(delta@delta))
        distance = np.linalg.norm(point-(a+np.clip(fraction, 0., 1.)*delta))
        if distance <= geometric_eps:
            eid = owners[edge][0]
            if intact[eid]:
                seeds.add(eid)
                source_edges.append(list(edge))
    if not seeds:
        raise RecoveryUnavailable("source is not on an intact owned boundary edge")
    adjacency = [set() for _ in elements]
    for ids in owners.values():
        if len(ids) == 2 and intact[ids[0]] and intact[ids[1]]:
            adjacency[ids[0]].add(ids[1]); adjacency[ids[1]].add(ids[0])
        elif len(ids) > 2:
            raise RecoveryUnavailable("nonmanifold mesh edge")
    patch, frontier = set(seeds), set(seeds)
    for _ in range(ADJACENCY_RINGS):
        frontier = set().union(*(adjacency[e] for e in sorted(frontier)))-patch if frontier else set()
        patch.update(frontier)
    ids = np.array(sorted(patch), dtype=int)
    if len(ids) < 3:
        raise RecoveryUnavailable("fewer than three intact samples")
    tangent = np.array([-normal[1], normal[0]])
    basis = np.column_stack((normal, tangent))
    offsets = tri[ids].mean(axis=1)-point
    distance_sq = np.sum(offsets**2, axis=1)
    length = math.sqrt(float(distance_sq.max()))
    local = offsets@basis/length
    design = np.column_stack((np.ones(len(ids)), local))
    weights = areas[ids]/(distance_sq+float(distance_sq.mean()))
    weights /= weights.sum()
    weighted_design = np.sqrt(weights)[:, None]*design
    u, singular, vt = np.linalg.svd(weighted_design, full_matrices=False)
    rank = int(np.sum(singular > max(weighted_design.shape)*np.finfo(float).eps*singular[0]))
    condition = float(singular[0]/singular[-1]) if singular[-1] else math.inf
    if rank != 3 or not math.isfinite(condition) or condition > MAX_CONDITION_NUMBER:
        raise RecoveryUnavailable(f"rank/conditioning rejected: rank={rank}, condition={condition}")
    inverse = (vt.T/singular)@u.T
    evaluation = inverse[0]*np.sqrt(weights)
    coefficients = inverse@(np.sqrt(weights)[:, None]*raw[ids].reshape(len(ids), 4))
    tensor = (evaluation@raw[ids].reshape(len(ids), 4)).reshape(2, 2)
    residual = design@coefficients-raw[ids].reshape(len(ids), 4)
    result = dict(operator_id=OPERATOR_ID, status="NUMERICALLY_ADMISSIBLE_NOT_SOURCE_QUALIFIED",
        source_point_m=point.tolist(), normal_xy=normal.tolist(), tangent_xy=tangent.tolist(),
        source_edge_node_ids=source_edges, seed_element_ids=sorted(seeds),
        element_ids=ids.tolist(), centroid_offsets_normal_tangent_scaled=local.tolist(),
        coordinate_scale_m=length, regression_weights=weights.tolist(),
        evaluation_weights=evaluation.tolist(), singular_values=singular.tolist(),
        rank=rank, condition_number=condition, weighted_residual_norm_Pa=float(np.linalg.norm(np.sqrt(weights)[:, None]*residual)),
        raw_CST_tensors_Pa=raw[ids].tolist(), tensor_Pa=tensor.tolist(),
        sigma_nn_Pa=float(normal@tensor@normal), sigma_tt_Pa=float(tangent@tensor@tangent),
        sigma_nt_Pa=float(normal@tensor@tangent), principal_stresses_Pa=np.linalg.eigvalsh(tensor).tolist(),
        mesh_nodes_fingerprint=_fingerprint(nodes), mesh_elements_fingerprint=_fingerprint(elements),
        intact_mask_fingerprint=_fingerprint(intact), full_raw_stress_fingerprint=_fingerprint(raw))
    planes = []
    for candidate in candidate_normals:
        n = np.asarray(candidate, dtype=float)
        if n.shape != (2,) or not np.isfinite(n).all() or abs(np.linalg.norm(n)-1.) > 1e-12:
            raise RecoveryUnavailable("invalid candidate plane normal")
        t = np.array([-n[1], n[0]])
        planes.append(dict(normal_xy=n.tolist(), tangent_xy=t.tolist(),
                           opening_Pa=float(n@tensor@n), shear_Pa=float(t@tensor@n)))
    result["candidate_planes"] = planes
    return result


def polygon_arc_source(nodes, boundary_edges, *, center, arc_fraction):
    """Exact polygon/radial-ray intersection, never a fabricated circular site."""
    nodes, center = np.asarray(nodes, dtype=float), np.asarray(center, dtype=float)
    theta = 2*math.pi*float(arc_fraction)
    normal = np.array([math.cos(theta), math.sin(theta)])
    hits = []
    for a, b in sorted(set(tuple(sorted(map(int, e))) for e in boundary_edges)):
        p, q = nodes[[a, b]]
        matrix = np.column_stack((normal, p-q))
        if abs(np.linalg.det(matrix)) <= np.finfo(float).eps*np.linalg.norm(p-q):
            continue
        distance, fraction = np.linalg.solve(matrix, p-center)
        if distance > 0 and -1e-12 <= fraction <= 1+1e-12:
            hits.append((float(distance), a, b))
    if not hits:
        raise RecoveryUnavailable("radial source ray misses cavity polygon")
    distance, _, _ = min(hits)
    return center+distance*normal, normal
