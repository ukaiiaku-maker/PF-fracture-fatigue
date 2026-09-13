"""Prospective shape-regular local cavity-source mesh for V5 readiness."""
from __future__ import annotations

from dataclasses import replace
import hashlib
import json
import math
from typing import Sequence

import numpy as np

from .explicit_cavity_v5 import (
    HoleMesh, _final_mesh_validation, build_explicit_hole_mesh,
)
from .mesh import BoundaryData, rebuild_tri_mesh


MESH_CONTRACT_ID = "CAVITY_SOURCE_SHAPE_REGULAR_LOCAL_PATCH_V5"
SCHEMA = "v5.cavity-source-shape-regular-local-patch/1"
POLYGON_SECTORS = 128
MATCHED_BOUNDARY_NODE_COUNT = 512
MATCHED_OUTER_RING_NODE_COUNT = 256
BASE_RADIAL_LAYERS = 48
LOCAL_LEVELS = {
    "A": {"first_strip_radial_subdivisions": 2, "eta_n_target": 0.06, "eta_t_target": 0.05},
    "B": {"first_strip_radial_subdivisions": 4, "eta_n_target": 0.03, "eta_t_target": 0.025},
    "C": {"first_strip_radial_subdivisions": 8, "eta_n_target": 0.015, "eta_t_target": 0.0125},
}
CONSTRUCTION_TARGET_MINIMUM_QUALITY = 0.10
ACCEPTANCE_MINIMUM_QUALITY = 0.05


def _sha256(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                 allow_nan=False).encode()).hexdigest()


def _subdivide_ring(points: np.ndarray, subdivisions: int) -> np.ndarray:
    rows = []
    for index, first in enumerate(points):
        second = points[(index + 1) % len(points)]
        rows.extend(first + (step / subdivisions) * (second - first)
                    for step in range(subdivisions))
    return np.asarray(rows, dtype=float)


def _orient(nodes: np.ndarray, triangle) -> tuple[int, int, int]:
    ids = tuple(map(int, triangle))
    xy = nodes[list(ids)]
    first, second = xy[1] - xy[0], xy[2] - xy[0]
    return ids if first[0] * second[1] - first[1] * second[0] > 0.0 else (ids[0], ids[2], ids[1])


def _connect_equal(nodes, inner, outer):
    out = []
    count = len(inner)
    for index in range(count):
        following = (index + 1) % count
        a, b = inner[index], inner[following]
        c, d = outer[following], outer[index]
        if index % 2:
            pairs = ((a, b, d), (b, c, d))
        else:
            pairs = ((a, b, c), (a, c, d))
        out.extend(_orient(nodes, item) for item in pairs)
    return out


def _connect_two_to_one(nodes, inner, outer):
    if len(inner) != 2 * len(outer):
        raise ValueError("transition rings must have a two-to-one node ratio")
    out = []
    for index in range(len(outer)):
        i0, i1, i2 = inner[2 * index], inner[2 * index + 1], inner[(2 * index + 2) % len(inner)]
        o0, o1 = outer[index], outer[(index + 1) % len(outer)]
        for triangle in ((o0, o1, i2), (o0, i2, i1), (o0, i1, i0)):
            out.append(_orient(nodes, triangle))
    return out


def _mesh_quality(mesh):
    points = np.asarray(mesh.nodes)[np.asarray(mesh.elems)]
    sides = np.linalg.norm(points - points[:, [1, 2, 0]], axis=2)
    return 4.0 * np.sqrt(3.0) * np.asarray(mesh.area_e) / np.maximum(np.sum(sides**2, axis=1), 1e-300)


def discrete_cavity_boundary_fingerprint(hole: HoleMesh) -> str:
    edges = sorted(tuple(sorted(map(int, edge))) for edge in np.asarray(hole.cavity_edges))
    nodes = sorted(set(item for edge in edges for item in edge))
    record = {
        "coordinates_m": np.asarray(hole.mesh.nodes)[nodes].tolist(),
        "edges_local": [[nodes.index(a), nodes.index(b)] for a, b in edges],
    }
    return _sha256(record)


def build_shape_regular_source_patch_hole_mesh(
    width_m: float, height_m: float, center_m: tuple[float, float], radius_m: float,
    far_h_m: float, *, local_level: str, polygon_sectors: int=POLYGON_SECTORS,
    base_radial_layers: int=BASE_RADIAL_LAYERS, source_direction_xy: Sequence[float]=(1.0, 0.0),
) -> HoleMesh:
    """Refine local resolution independently of one fixed polygon geometry."""
    if local_level not in LOCAL_LEVELS:
        raise ValueError("local level must be A, B, or C")
    if polygon_sectors not in (64, 128, 256):
        raise ValueError("polygon sectors must be one of the prospectively bounded levels")
    direction = np.asarray(source_direction_xy, dtype=float)
    direction /= np.linalg.norm(direction)
    source_angle = math.atan2(float(direction[1]), float(direction[0]))
    base = build_explicit_hole_mesh(
        width_m, height_m, center_m, radius_m, far_h_m, polygon_sectors,
        radial_layers_override=base_radial_layers,
        angular_phase_rad=source_angle - math.pi / polygon_sectors,
    )
    base_grid = np.asarray(base.mesh.nodes).reshape(base_radial_layers + 1, polygon_sectors, 2)
    boundary_subdivisions = MATCHED_BOUNDARY_NODE_COUNT // polygon_sectors
    outer_subdivisions = MATCHED_OUTER_RING_NODE_COUNT // polygon_sectors
    if boundary_subdivisions * polygon_sectors != MATCHED_BOUNDARY_NODE_COUNT:
        raise ValueError("polygon does not divide the frozen matched boundary resolution")
    fine_rings = [_subdivide_ring(base_grid[0], boundary_subdivisions)]
    first_outer = _subdivide_ring(base_grid[1], boundary_subdivisions)
    radial_scale = POLYGON_SECTORS // polygon_sectors
    if radial_scale < 1:
        radial_scale = 1
    radial_subdivisions = (
        LOCAL_LEVELS[local_level]["first_strip_radial_subdivisions"] * radial_scale
    )
    for step in range(1, radial_subdivisions + 1):
        fine_rings.append(fine_rings[0] + (step / radial_subdivisions) * (first_outer - fine_rings[0]))
    outer_rings = [
        _subdivide_ring(base_grid[layer], outer_subdivisions)
        for layer in range(2, base_radial_layers + 1)
    ]
    ring_points = fine_rings + outer_rings
    nodes = np.vstack(ring_points)
    rings = []
    offset = 0
    for points in ring_points:
        rings.append(np.arange(offset, offset + len(points), dtype=int))
        offset += len(points)
    elements = []
    for index in range(len(fine_rings) - 1):
        elements.extend(_connect_equal(nodes, rings[index], rings[index + 1]))
    transition = len(fine_rings) - 1
    elements.extend(_connect_two_to_one(nodes, rings[transition], rings[transition + 1]))
    for index in range(transition + 1, len(rings) - 1):
        elements.extend(_connect_equal(nodes, rings[index], rings[index + 1]))
    mesh = rebuild_tri_mesh(nodes, np.asarray(elements, dtype=int))
    cavity = np.column_stack((rings[0], np.roll(rings[0], -1)))
    exterior = np.column_stack((rings[-1], np.roll(rings[-1], -1)))
    tolerance = max(far_h_m * 0.1, 1.0e-12)
    x, y = nodes[:, 0], nodes[:, 1]
    top = np.flatnonzero(np.isclose(y, height_m / 2.0, atol=tolerance))
    bottom = np.flatnonzero(np.isclose(y, -height_m / 2.0, atol=tolerance))
    lb = int(np.argmin(x*x + (y + height_m/2.0)**2))
    rb = int(np.argmin((x-width_m)**2 + (y + height_m/2.0)**2))
    boundary = BoundaryData(top, bottom, lb, rb, np.array([], dtype=int))
    corner_offset = 0
    prescribed = rings[0][corner_offset::boundary_subdivisions]
    validation = {
        "geometry_contract": MESH_CONTRACT_ID,
        "schema": SCHEMA,
        "local_level": local_level,
        "polygon_segment_count": polygon_sectors,
        "polygon_angular_phase_rad": source_angle - math.pi / polygon_sectors,
        "polygon_apothem_m": float(radius_m),
        "polygon_circumradius_m": float(radius_m / math.cos(math.pi / polygon_sectors)),
        "boundary_subdivisions_per_facet": boundary_subdivisions,
        "outer_subdivisions_per_facet": outer_subdivisions,
        "first_strip_radial_subdivisions": radial_subdivisions,
        "radial_layers": base_radial_layers,
        "source_direction_xy": direction.tolist(),
        "source_coordinates_m": {
            "near": (np.asarray(center_m) - float(radius_m) * direction).tolist(),
            "far": (np.asarray(center_m) + float(radius_m) * direction).tolist(),
        },
        "construction_strategy": "DETERMINISTIC_TWO_TO_ONE_STRUCTURED_LOCAL_PATCH",
        "unconstrained_retriangulation": False,
    }
    hole = HoleMesh(mesh, boundary, center_m, radius_m, cavity, exterior, prescribed, validation)
    validation = _final_mesh_validation(hole)
    validation.update(hole.validation)
    validation["minimum_quality"] = float(np.min(_mesh_quality(mesh)))
    validation["discrete_cavity_boundary_fingerprint"] = discrete_cavity_boundary_fingerprint(hole)
    validation["polygon_geometry_fingerprint"] = _sha256({
        "center_m": list(center_m), "nominal_radius_m": radius_m,
        "polygon_sectors": polygon_sectors,
        "corner_coordinates_m": nodes[prescribed].tolist(),
    })
    return replace(hole, validation=validation)


__all__ = [
    "ACCEPTANCE_MINIMUM_QUALITY", "BASE_RADIAL_LAYERS",
    "CONSTRUCTION_TARGET_MINIMUM_QUALITY", "LOCAL_LEVELS",
    "MATCHED_BOUNDARY_NODE_COUNT", "MATCHED_OUTER_RING_NODE_COUNT", "MESH_CONTRACT_ID",
    "POLYGON_SECTORS", "build_shape_regular_source_patch_hole_mesh",
    "discrete_cavity_boundary_fingerprint",
]
