"""Boundary-fitted downstream source recovery on the accepted cavity polygon."""
from __future__ import annotations

import hashlib
import json
import math
from typing import Any, Mapping

import numpy as np

from .cavity_source_recovery_v3 import (
    MAXIMUM_DESIGN_CONDITION, MINIMUM_ELEMENT_QUALITY, MINIMUM_SUPPORT_ELEMENTS,
    WINDOW_SCALE_FRACTION, _tensor_rows, _triangle_quality, _weighted_fit,
)


SCHEMA = "v5.cavity-boundary-fitted-source/1"
RECOVERY_ID = "CAVITY_BOUNDARY_FITTED_PHYSICAL_PATCH_V1"
MAXIMUM_POINT_FACET_PAIRS_PER_BLOCK = 1_048_576


class CavityBoundaryFittedSourceUnavailable(RuntimeError):
    """Expected scientific non-certification of the polygon-fitted observer."""


def _sha(value: object) -> str:
    return hashlib.sha256(json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False,
    ).encode()).hexdigest()


def _ordered_cycle(edges, source_node: int) -> tuple[int, ...]:
    canonical = tuple(sorted({tuple(sorted(map(int, edge))) for edge in edges}))
    adjacency: dict[int, list[int]] = {}
    for a, b in canonical:
        adjacency.setdefault(a, []).append(b); adjacency.setdefault(b, []).append(a)
    if not canonical or source_node not in adjacency or any(len(row) != 2 for row in adjacency.values()):
        raise CavityBoundaryFittedSourceUnavailable(
            "owned cavity boundary is not one degree-two cycle containing the source"
        )
    ordered = [int(source_node)]
    previous = None
    current = int(source_node)
    while True:
        neighbors = sorted(adjacency[current])
        following = neighbors[0] if neighbors[0] != previous else neighbors[1]
        if following == source_node:
            break
        ordered.append(following)
        previous, current = current, following
        if len(ordered) > len(adjacency):
            raise CavityBoundaryFittedSourceUnavailable("owned cavity cycle traversal did not close")
    if len(ordered) != len(adjacency):
        raise CavityBoundaryFittedSourceUnavailable("owned cavity boundary has multiple components")
    return tuple(ordered)


def _nearest_polygon_coordinates(points, polygon, vectors, lengths):
    """Exact nearest-facet coordinates with bounded temporary storage.

    The production cavity cycle can contain O(10^3) refined facets while the
    mechanics mesh contains O(10^5) elements.  Broadcasting both complete
    axes creates multiple gigabyte arrays.  Blocking only the point axis does
    not change the calculation or tie-breaking order used by ``argmin``.
    """
    samples=np.asarray(points,float); facets=np.asarray(polygon,float)
    directions=np.asarray(vectors,float); facet_lengths=np.asarray(lengths,float)
    if samples.ndim!=2 or samples.shape[1]!=2 or facets.ndim!=2 or facets.shape[1]!=2:
        raise ValueError("points and polygon must be planar coordinate arrays")
    if len(facets)!=len(directions) or len(facets)!=len(facet_lengths) or not len(facets):
        raise ValueError("polygon facet arrays must be nonempty and aligned")
    block=max(1,MAXIMUM_POINT_FACET_PAIRS_PER_BLOCK//len(facets))
    nearest=np.empty(len(samples),dtype=int)
    nearest_fraction=np.empty(len(samples),dtype=float)
    nearest_projection=np.empty((len(samples),2),dtype=float)
    denominator=np.maximum(facet_lengths**2,1.0e-300)
    for start in range(0,len(samples),block):
        stop=min(start+block,len(samples)); current=samples[start:stop]
        relative=current[:,None,:]-facets[None,:,:]
        fraction=np.clip(
            np.einsum("nki,ki->nk",relative,directions)/denominator[None,:],
            0.0,1.0,
        )
        projection=facets[None,:,:]+fraction[:,:,None]*directions[None,:,:]
        squared=np.einsum("nki,nki->nk",current[:,None,:]-projection,
                          current[:,None,:]-projection)
        local=np.argmin(squared,axis=1); rows=np.arange(stop-start)
        nearest[start:stop]=local
        nearest_fraction[start:stop]=fraction[rows,local]
        nearest_projection[start:stop]=projection[rows,local]
    return nearest,nearest_fraction,nearest_projection


def recover_boundary_fitted_physical_patch_v1(
    *, nodes, elements, stress_Pa, boundary_node: int, cavity_id: str,
    cavity_center_m, cavity_radius_m: float, process_length_m: float,
    poisson_ratio: float, owned_boundary_edges, material_fingerprints: Mapping[str, str],
    state_fingerprint: str, solid_element_mask=None,
) -> dict[str, Any]:
    """Recover the source tensor in exact polygon arclength/normal coordinates."""
    xy = np.asarray(nodes, float); tri = np.asarray(elements, int)
    center = np.asarray(cavity_center_m, float); node = int(boundary_node)
    radius = float(cavity_radius_m); process = float(process_length_m); nu = float(poisson_ratio)
    if xy.ndim != 2 or xy.shape[1] != 2 or tri.ndim != 2 or tri.shape[1] != 3:
        raise ValueError("nodes/elements must describe a triangular plane mesh")
    if not (0 <= node < len(xy)) or center.shape != (2,):
        raise ValueError("source node or cavity center is invalid")
    if not np.isfinite(radius + process + nu) or radius <= 0.0 or process <= 0.0:
        raise ValueError("physical window inputs must be finite and positive")
    cycle = _ordered_cycle(owned_boundary_edges, node)
    polygon = xy[np.asarray(cycle, int)]
    following = np.roll(polygon, -1, axis=0)
    vectors = following - polygon
    lengths = np.linalg.norm(vectors, axis=1)
    if np.any(lengths <= 0.0):
        raise CavityBoundaryFittedSourceUnavailable("owned cavity cycle has a zero-length edge")
    perimeter = float(np.sum(lengths)); cumulative = np.concatenate(([0.0], np.cumsum(lengths)))
    tangents = vectors / lengths[:, None]
    normals = np.column_stack((-tangents[:, 1], tangents[:, 0]))
    midpoints = 0.5 * (polygon + following)
    inward = center[None, :] - midpoints
    normals[np.einsum("ij,ij->i", normals, inward) > 0.0] *= -1.0

    # The inserted connection source must be a genuine point on the accepted
    # cycle. Its two incident facets define one unique tangent line.
    incident = [index for index, value in enumerate(cycle)
                if value == node or cycle[(index + 1) % len(cycle)] == node]
    if len(incident) != 2:
        raise CavityBoundaryFittedSourceUnavailable("source does not own two polygon facets")
    t0, t1 = tangents[incident]
    if abs(float(t0[0]*t1[1]-t0[1]*t1[0])) > 1.0e-8:
        raise CavityBoundaryFittedSourceUnavailable(
            "source is a polygon corner rather than a boundary-fitted facet point"
        )
    source_tangent = t0 if float(t0 @ t1) > 0.0 else t0
    source_tangent /= np.linalg.norm(source_tangent)
    source_normal = np.asarray((-source_tangent[1], source_tangent[0]))
    if float(source_normal @ (xy[node]-center)) < 0.0:
        source_normal *= -1.0

    points = xy[tri]; centroids = points.mean(axis=1)
    nearest,nearest_fraction,nearest_projection=_nearest_polygon_coordinates(
        centroids,polygon,vectors,lengths,
    )
    point_normals = normals[nearest]
    point_tangents = tangents[nearest]
    n_all = np.einsum("ij,ij->i", centroids-nearest_projection, point_normals)
    arc = cumulative[nearest] + nearest_fraction*lengths[nearest]
    s_all = (arc + 0.5*perimeter) % perimeter - 0.5*perimeter

    physical_scale = min(radius, process)
    half_width = WINDOW_SCALE_FRACTION*physical_scale
    normal_depth = WINDOW_SCALE_FRACTION*physical_scale
    tolerance = max(1.0e-12, 1.0e-8*physical_scale)
    solid = (np.ones(len(tri), bool) if solid_element_mask is None
             else np.asarray(solid_element_mask, bool))
    if solid.shape != (len(tri),):
        raise ValueError("solid element mask has the wrong shape")
    selected = np.flatnonzero(
        solid & (np.abs(s_all) <= half_width)
        & (n_all >= -tolerance) & (n_all <= normal_depth)
    )
    if len(selected) < MINIMUM_SUPPORT_ELEMENTS:
        raise CavityBoundaryFittedSourceUnavailable("boundary-fitted window has inadequate support")
    quality = _triangle_quality(points[selected])
    if float(np.min(quality)) < MINIMUM_ELEMENT_QUALITY:
        raise CavityBoundaryFittedSourceUnavailable("boundary-fitted window has poor-quality support")
    s = s_all[selected]; n = np.maximum(n_all[selected], 0.0)
    xi = s/half_width; zeta = n/normal_depth
    design_tt = np.column_stack((np.ones(len(selected)), xi, zeta, xi*xi, xi*zeta, zeta*zeta))
    design_traction = zeta[:, None]*np.column_stack((np.ones(len(selected)), xi, zeta))
    sides = points[selected, 1:] - points[selected, :1]
    areas = np.abs(sides[:, 0, 0]*sides[:, 1, 1]-sides[:, 0, 1]*sides[:, 1, 0])/2.0
    raw_weights = areas*np.maximum(1.0-xi*xi, 0.0)**2*np.maximum(1.0-zeta*zeta, 0.0)**2
    if not np.isfinite(raw_weights).all() or float(np.sum(raw_weights)) <= 0.0:
        raise CavityBoundaryFittedSourceUnavailable("boundary-fitted window has zero weight")
    weights = raw_weights/np.sum(raw_weights)
    stress_rows = _tensor_rows(stress_Pa, len(tri))[selected]
    tensors = np.zeros((len(selected), 2, 2))
    tensors[:, 0, 0], tensors[:, 1, 1], tensors[:, 0, 1] = stress_rows.T
    tensors[:, 1, 0] = tensors[:, 0, 1]
    local_nn = np.einsum("ni,nij,nj->n", point_normals[selected], tensors, point_normals[selected])
    local_tt = np.einsum("ni,nij,nj->n", point_tangents[selected], tensors, point_tangents[selected])
    local_nt = np.einsum("ni,nij,nj->n", point_normals[selected], tensors, point_tangents[selected])
    try:
        ctt, rtt, ktt, ptt = _weighted_fit(design_tt, local_tt, weights, required_rank=6, name="sigma_tt")
        cnn, rnn, knn, pnn = _weighted_fit(design_traction, local_nn, weights, required_rank=3, name="sigma_nn")
        cnt, rnt, knt, pnt = _weighted_fit(design_traction, local_nt, weights, required_rank=3, name="sigma_nt")
    except RuntimeError as exc:
        raise CavityBoundaryFittedSourceUnavailable(str(exc)) from exc
    observed = np.column_stack((local_nn, local_tt, local_nt))
    predicted = np.column_stack((pnn, ptt, pnt))
    fit_residual = float(np.sqrt(np.sum(weights[:, None]*(predicted-observed)**2))
                         / max(np.sqrt(np.sum(weights[:, None]*observed**2)), 1.0e-300))
    sigma_tt = float(ctt[0])
    tensor = sigma_tt*np.outer(source_tangent, source_tangent)
    sigma_zz = nu*float(np.trace(tensor))
    identity = {
        "cavity_id": str(cavity_id), "boundary_node_id": node,
        "boundary_position_m": xy[node].tolist(),
        "owned_polygon_cycle_node_ids": list(cycle),
        "owned_polygon_perimeter_m": perimeter,
        "source_normal_xy": source_normal.tolist(),
        "source_tangent_xy": source_tangent.tolist(),
        "coordinate_system": "owned-polygon arclength and outward facet-normal distance",
        "tangential_half_width_m": half_width, "normal_depth_m": normal_depth,
    }
    identity["sha256"] = _sha(identity)
    return {
        "schema": SCHEMA, "recovery_operator": RECOVERY_ID, "available": True,
        "physical_arc_identity": identity, "boundary_node_id": node,
        "owned_boundary_edges": [list(map(int, edge)) for edge in owned_boundary_edges],
        "stencil_element_ids": selected.tolist(), "sample_count": int(len(selected)),
        "element_coordinates_s_n_m": np.column_stack((s, n)).tolist(),
        "weights": weights.tolist(), "minimum_element_quality": float(np.min(quality)),
        "rank": {"sigma_tt": rtt, "sigma_nn": rnn, "sigma_nt": rnt},
        "condition": {"sigma_tt": ktt, "sigma_nn": knn, "sigma_nt": knt,
                      "maximum": max(ktt, knn, knt), "limit": MAXIMUM_DESIGN_CONDITION},
        "constrained_fit_residual": fit_residual,
        "traction_free_enforcement": "sigma_nn(s,0)=sigma_nt(s,0)=0",
        "traction_residual": 0.0, "sigma_tt_Pa": sigma_tt,
        "sigma_nn_Pa": 0.0, "sigma_nt_Pa": 0.0, "tensor_Pa": tensor.tolist(),
        "plane_strain_sigma_zz_Pa": sigma_zz,
        "mean_stress_Pa": (float(np.trace(tensor))+sigma_zz)/3.0,
        "material_fingerprints": {str(k): str(v) for k, v in material_fingerprints.items()},
        "state_fingerprint": str(state_fingerprint),
    }


__all__ = ["CavityBoundaryFittedSourceUnavailable", "RECOVERY_ID", "SCHEMA",
           "recover_boundary_fitted_physical_patch_v1"]
