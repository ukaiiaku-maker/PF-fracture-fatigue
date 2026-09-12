"""Prospective source-conforming cavity geometry contract for V4 readiness."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from typing import Any, Mapping, Sequence

import numpy as np

from .cavity_source_recovery_v3 import (
    CavitySourceRecoveryV3Unavailable,
    recover_fixed_physical_arc_patch_v3,
)


GEOMETRY_ID = "CAVITY_SOURCE_CONFORMING_GEOMETRY_V4"
SCHEMA = "v5.cavity-source-conforming-geometry/4"
SOURCE_COORDINATE_TOLERANCE_RELATIVE = 1.0e-8
SOURCE_COORDINATE_TOLERANCE_ABSOLUTE_M = 1.0e-12
TENSOR_RELATIVE_TOLERANCE = 0.05
TRACTION_RESIDUAL_TOLERANCE = 0.05
ETA_N_MAX = 0.03
ETA_T_MAX = 0.025
MINIMUM_MESH_QUALITY = 0.05


class CavitySourceGeometryV4Unavailable(RuntimeError):
    """Expected scientific non-certification under the frozen V4 contract."""


def _sha256(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                 allow_nan=False).encode()).hexdigest()


def _canonical_edges(edges) -> tuple[tuple[int, int], ...]:
    return tuple(sorted({tuple(sorted(map(int, edge))) for edge in edges}))


def _polygon_measure(points: np.ndarray) -> tuple[float, float]:
    following = np.roll(points, -1, axis=0)
    area = 0.5 * abs(float(np.sum(points[:, 0] * following[:, 1]
                                  - points[:, 1] * following[:, 0])))
    perimeter = float(np.sum(np.linalg.norm(following - points, axis=1)))
    return area, perimeter


@dataclass(frozen=True)
class CavityRadiusConventionV4:
    nominal_circle_radius_m: float
    polygon_apothem_m: float
    polygon_circumradius_m: float
    boundary_segment_count: int
    polygon_angular_phase_rad: float
    actual_polygon_vertices_m: tuple[tuple[float, float], ...]
    actual_polygon_edges: tuple[tuple[int, int], ...]
    actual_fem_boundary_edges: tuple[tuple[int, int], ...]
    physical_source_coordinate_m: tuple[float, float]
    topological_connection_coordinate_m: tuple[float, float]
    cavity_area_convention: str = "pi_times_nominal_circle_radius_squared"
    void_kinetics_radius_convention: str = "nominal_physical_circle_radius"
    finite_element_boundary_convention: str = "actual_circumscribed_polygon"
    length_ledger_boundary_convention: str = "actual_source_conforming_polygon_intersections"

    def as_record(self) -> dict[str, Any]:
        record = dict(vars(self))
        record["schema"] = SCHEMA
        record["geometry_contract"] = GEOMETRY_ID
        record["nominal_circle_area_m2"] = math.pi * self.nominal_circle_radius_m**2
        vertices = np.asarray(self.actual_polygon_vertices_m, dtype=float)
        area, perimeter = _polygon_measure(vertices)
        record["finite_element_polygon_area_m2"] = area
        record["finite_element_polygon_perimeter_m"] = perimeter
        record["sha256"] = _sha256(record)
        return record


def radius_convention_from_hole(hole, *, source_direction_xy=(1.0, 0.0), side="far"):
    direction = np.asarray(source_direction_xy, dtype=float)
    direction /= np.linalg.norm(direction)
    if side not in ("near", "far"):
        raise ValueError("source side must be near or far")
    sign = -1.0 if side == "near" else 1.0
    center = np.asarray(hole.center_m, dtype=float)
    source = center + sign * float(hole.radius_m) * direction
    polygon_nodes = tuple(map(int, np.asarray(hole.prescribed_polygon_nodes)))
    vertices = tuple(tuple(map(float, hole.mesh.nodes[node])) for node in polygon_nodes)
    edges = tuple((polygon_nodes[index], polygon_nodes[(index + 1) % len(polygon_nodes)])
                  for index in range(len(polygon_nodes)))
    convention = CavityRadiusConventionV4(
        nominal_circle_radius_m=float(hole.radius_m),
        polygon_apothem_m=float(hole.radius_m),
        polygon_circumradius_m=float(hole.radius_m / math.cos(math.pi / len(polygon_nodes))),
        boundary_segment_count=len(polygon_nodes),
        polygon_angular_phase_rad=float(hole.validation["polygon_angular_phase_rad"]),
        actual_polygon_vertices_m=vertices,
        actual_polygon_edges=edges,
        actual_fem_boundary_edges=_canonical_edges(hole.cavity_edges),
        physical_source_coordinate_m=tuple(map(float, source)),
        topological_connection_coordinate_m=tuple(map(float, source)),
    )
    return convention


def source_node_certificate(*, nodes, elements, owned_boundary_edges, boundary_node,
                            cavity_center_m, cavity_radius_m) -> dict[str, Any]:
    xy = np.asarray(nodes, dtype=float)
    tri = np.asarray(elements, dtype=int)
    node = int(boundary_node)
    edges = _canonical_edges(owned_boundary_edges)
    incident = tuple(edge for edge in edges if node in edge)
    if len(incident) != 2:
        raise CavitySourceGeometryV4Unavailable(
            "source node does not have degree-two cavity-boundary incidence"
        )
    owners = {}
    for edge in incident:
        owners[edge] = sum(set(edge).issubset(set(map(int, row))) for row in tri)
    if any(value != 1 for value in owners.values()):
        raise CavitySourceGeometryV4Unavailable(
            "source node cavity facets do not have exactly one solid owner"
        )
    center = np.asarray(cavity_center_m, dtype=float)
    radial = xy[node] - center
    distance = float(np.linalg.norm(radial))
    radius = float(cavity_radius_m)
    tolerance = max(SOURCE_COORDINATE_TOLERANCE_ABSOLUTE_M,
                    SOURCE_COORDINATE_TOLERANCE_RELATIVE * radius)
    if abs(distance - radius) > tolerance:
        raise CavitySourceGeometryV4Unavailable(
            "source node is not at the nominal physical circle radius"
        )
    neighbors = [next(item for item in edge if item != node) for edge in incident]
    vectors = xy[neighbors] - xy[node]
    cross = float(vectors[0, 0] * vectors[1, 1] - vectors[0, 1] * vectors[1, 0])
    scale = max(float(np.linalg.norm(vectors[0]) * np.linalg.norm(vectors[1])), 1.0e-300)
    if abs(cross) > 1.0e-10 * scale or float(vectors[0] @ vectors[1]) >= 0.0:
        raise CavitySourceGeometryV4Unavailable(
            "source node is not a degree-two collinear facet midpoint"
        )
    normal = radial / distance
    tangent = np.asarray((-normal[1], normal[0]))
    certificate = {
        "schema": SCHEMA,
        "geometry_contract": GEOMETRY_ID,
        "boundary_node_id": node,
        "boundary_position_m": xy[node].tolist(),
        "nominal_circle_radius_m": radius,
        "radial_error_m": abs(distance - radius),
        "degree": 2,
        "incident_boundary_edges": [list(edge) for edge in incident],
        "incident_solid_owner_counts": {f"{a}:{b}": owners[(a, b)] for a, b in incident},
        "collinearity_residual": abs(cross) / scale,
        "normal_xy": normal.tolist(),
        "tangent_xy": tangent.tolist(),
        "unique_tangent_and_outward_normal": True,
    }
    certificate["sha256"] = _sha256(certificate)
    return certificate


def recover_source_conforming_geometry_v4(**kwargs) -> dict[str, Any]:
    """Certify geometry, then reuse the unchanged V3 fixed-window WLS recovery."""
    try:
        certificate = source_node_certificate(
            nodes=kwargs["nodes"], elements=kwargs["elements"],
            owned_boundary_edges=kwargs["owned_boundary_edges"],
            boundary_node=kwargs["boundary_node"], cavity_center_m=kwargs["cavity_center_m"],
            cavity_radius_m=kwargs["cavity_radius_m"],
        )
        recovery = recover_fixed_physical_arc_patch_v3(**kwargs)
    except CavitySourceRecoveryV3Unavailable as exc:
        raise CavitySourceGeometryV4Unavailable(str(exc)) from exc
    return {
        **recovery,
        "geometry_contract": GEOMETRY_ID,
        "geometry_contract_schema": SCHEMA,
        "source_node_certificate": certificate,
        "v3_fixed_physical_window_wls_reused_unchanged": True,
        "v3_geometry_check_relaxed": False,
        "frozen_acceptance_gates": {
            "tensor_relative": TENSOR_RELATIVE_TOLERANCE,
            "traction_residual": TRACTION_RESIDUAL_TOLERANCE,
            "eta_n_max": ETA_N_MAX,
            "eta_t_max": ETA_T_MAX,
            "minimum_mesh_quality": MINIMUM_MESH_QUALITY,
        },
    }


__all__ = [
    "CavityRadiusConventionV4", "CavitySourceGeometryV4Unavailable", "ETA_N_MAX",
    "ETA_T_MAX", "GEOMETRY_ID", "MINIMUM_MESH_QUALITY", "SCHEMA",
    "TENSOR_RELATIVE_TOLERANCE", "TRACTION_RESIDUAL_TOLERANCE",
    "radius_convention_from_hole", "recover_source_conforming_geometry_v4",
    "source_node_certificate",
]
