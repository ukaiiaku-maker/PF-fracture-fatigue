"""Conservative P0 vacancy transport with state-owned surface fluxes.

The mechanics mesh is used as a transport mesh, but transport connectivity is
rebuilt from accepted topology.  The discretisation is a genuine RT0/P0 mixed
method: each transport face owns one oriented, integrated normal flux, each
solid triangle owns one atomic vacancy fraction, and every triangle contributes
its exact degree-two-quadrature RT0 mass matrix to a symmetric saddle system.

The older :mod:`vacancy_coupling_v1` reaction-only implementation remains
available under ``LOCAL_REACTION_V1_LEGACY`` and is not called here.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
from enum import Enum
import hashlib
import json
import math
from types import MappingProxyType
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
from scipy import sparse
from scipy.spatial import cKDTree
from scipy.sparse.linalg import eigsh, factorized, spsolve
from .mesh import rebuild_tri_mesh

from .vacancy_coupling_v1 import (
    DILUTE_VACANCY_FRACTION_LIMIT,
    EV_J,
    KB_J_PER_K,
    REFERENCE_VACANCY_ROW,
    conservation_roundoff_budget_m2,
    vacancy_diffusivity_m2_s,
)


SCHEMA = "unified-2d.vacancy-transport-phase-field/2"
MODEL_ID = "MIXED_RT0_P0_CONSERVATIVE_POSITIVE_FLUX_CORRECTION"
LOCAL_REACTION_V1_LEGACY = "LOCAL_REACTION_V1_LEGACY"
REFERENCE_UNCALIBRATED = "REFERENCE_UNCALIBRATED"
KAPPA_V = 0.0
DEFAULT_C_MIN = 1.0e-30
DEFAULT_C_MAX = DILUTE_VACANCY_FRACTION_LIMIT


class VacancyTransportBoundsError(RuntimeError):
    """Fail-closed transport rejection with a compact reproducibility record."""

    def __init__(self, diagnostic: Mapping[str, Any]):
        super().__init__("vacancy transport solve exceeded fail-closed concentration bounds")
        self.diagnostic = dict(diagnostic)

    def add_context(self, **values: Any) -> "VacancyTransportBoundsError":
        self.diagnostic.update(values)
        return self

# These are study coordinates, not fitted material values. Keeping the table
# immutable and constructing the full row from the accepted material prevents
# a variant run from silently falling back to the reference activation energy.
VACANCY_ENERGY_STUDY_SPECS = MappingProxyType({
    "reference": (3.2, 1.7),
    "low_formation": (2.4, 1.7),
    "low_migration": (3.2, 0.9),
    "combined": (2.4, 0.9),
})


def atomic_volume_from_burgers_vector_m(burgers_vector_m: float) -> float:
    """BCC atomic volume inferred from ``b = sqrt(3)*a/2``."""
    b = float(burgers_vector_m)
    if not math.isfinite(b) or b <= 0.0:
        raise ValueError("Burgers vector must be finite and positive")
    return 4.0 * b**3 / (3.0 * math.sqrt(3.0))


@dataclass(frozen=True)
class VacancyTransportParameterRowV2:
    row_id: str
    atomic_volume_m3: float
    vacancy_formation_volume_m3: float
    vacancy_formation_energy_eV: float = 3.2
    vacancy_formation_entropy_over_kB: float = 0.0
    vacancy_migration_energy_eV: float = 1.7
    vacancy_diffusivity_prefactor_m2_s: float = 1.0e-6
    plastic_vacancy_coefficient_per_strain: float = 3.0e-5
    dislocation_production_boost: float = 0.20
    interaction_mobile_vacancy_fraction: float = 1.0
    interaction_direct_cavitation_fraction: float = 0.0
    interaction_benign_fraction: float = 0.0
    dislocation_climb_capture_efficiency: float = 1.0
    site_capture_factor: float = 1.0
    rho_reference_m2: float = 1.0e12
    bulk_vacancy_sink_spacing_m: float | None = None
    calibration_status: str = REFERENCE_UNCALIBRATED
    schema: str = SCHEMA

    def __post_init__(self) -> None:
        if self.calibration_status != REFERENCE_UNCALIBRATED:
            raise ValueError("V2 reference transport row must remain explicitly uncalibrated")
        if not self.row_id:
            raise ValueError("transport parameter row requires an identity")
        positive = (
            self.atomic_volume_m3, self.vacancy_formation_volume_m3,
            self.vacancy_formation_energy_eV, self.vacancy_migration_energy_eV,
            self.vacancy_diffusivity_prefactor_m2_s, self.rho_reference_m2,
        )
        if any(not math.isfinite(float(value)) or float(value) <= 0.0 for value in positive):
            raise ValueError("transport material values must be finite and positive")
        nonnegative = (
            self.vacancy_formation_entropy_over_kB,
            self.plastic_vacancy_coefficient_per_strain,
            self.dislocation_production_boost,
            self.dislocation_climb_capture_efficiency,
            self.site_capture_factor,
            self.interaction_mobile_vacancy_fraction,
            self.interaction_direct_cavitation_fraction,
            self.interaction_benign_fraction,
        )
        if any(not math.isfinite(float(value)) or float(value) < 0.0 for value in nonnegative):
            raise ValueError("transport kinetic values must be finite and nonnegative")
        partition = math.fsum((
            float(self.interaction_mobile_vacancy_fraction),
            float(self.interaction_direct_cavitation_fraction),
            float(self.interaction_benign_fraction),
        ))
        if not math.isclose(partition, 1.0, rel_tol=0.0, abs_tol=1.0e-12):
            raise ValueError("interaction outcome fractions must form an exact unit partition")
        spacing = self.bulk_vacancy_sink_spacing_m
        if spacing is not None and (not math.isfinite(float(spacing)) or float(spacing) <= 0.0):
            raise ValueError("bulk vacancy sink spacing must be independently finite and positive")

    @classmethod
    def from_material(cls, material: Any) -> "VacancyTransportParameterRowV2":
        atomic = atomic_volume_from_burgers_vector_m(float(material.b))
        return cls(
            row_id=f"v2.common-tungsten-vacancy-reference-uncalibrated:b={float(material.b):.12g}/1",
            atomic_volume_m3=atomic,
            vacancy_formation_volume_m3=atomic,
        )

    @property
    def inventory_ratio(self) -> float:
        return float(self.vacancy_formation_volume_m3 / self.atomic_volume_m3)

    @property
    def fingerprint(self) -> str:
        payload = json.dumps(asdict(self), sort_keys=True, separators=(",", ":"), allow_nan=False)
        return hashlib.sha256(payload.encode()).hexdigest()


def vacancy_energy_study_row(material: Any, variant: str) -> VacancyTransportParameterRowV2:
    """Return one immutable, explicitly uncalibrated Ef/Em study row."""
    key = str(variant).strip().lower()
    if key not in VACANCY_ENERGY_STUDY_SPECS:
        raise ValueError(f"unknown vacancy-energy study variant: {variant!r}")
    formation, migration = VACANCY_ENERGY_STUDY_SPECS[key]
    base = VacancyTransportParameterRowV2.from_material(material)
    return replace(
        base,
        row_id=(f"v2.vacancy-energy-study:{key}:"
                f"Ef={formation:.12g}eV:Em={migration:.12g}eV/1"),
        vacancy_formation_energy_eV=formation,
        vacancy_migration_energy_eV=migration,
    )


class SurfaceKind(str, Enum):
    CRACK_FREE_SURFACE = "CRACK_FREE_SURFACE"
    EXTERNAL_FREE_SURFACE = "EXTERNAL_FREE_SURFACE"
    ACTIVE_CAVITY_SURFACE = "ACTIVE_CAVITY_SURFACE"
    SUBGRID_CAVITY_CAPTURE = "SUBGRID_CAVITY_CAPTURE"
    MERGED_CAVITY_CRACK_WAKE = "MERGED_CAVITY_CRACK_WAKE"
    SYMMETRY_NO_FLUX = "SYMMETRY_NO_FLUX"
    CONTACT_OR_GRIP_NO_FLUX = "CONTACT_OR_GRIP_NO_FLUX"
    INTACT_INTERIOR_INTERFACE = "INTACT_INTERIOR_INTERFACE"


PERFECT_SINK_KINDS = frozenset({
    SurfaceKind.CRACK_FREE_SURFACE,
    SurfaceKind.EXTERNAL_FREE_SURFACE,
    SurfaceKind.ACTIVE_CAVITY_SURFACE,
    SurfaceKind.MERGED_CAVITY_CRACK_WAKE,
})
NO_FLUX_KINDS = frozenset({
    SurfaceKind.SYMMETRY_NO_FLUX,
    SurfaceKind.CONTACT_OR_GRIP_NO_FLUX,
})


@dataclass(frozen=True)
class TransportFace:
    face_id: str
    kind: SurfaceKind
    owner_elements: tuple[int, ...]
    node_ids: tuple[int, int] | None
    center_xy_m: tuple[float, float]
    length_m: float
    owner_distances_m: tuple[float, ...]
    segment_id: str | None = None
    side: int = 0

    def __post_init__(self) -> None:
        if len(self.owner_elements) not in (1, 2):
            raise ValueError("a transport face must have one or two solid owners")
        if len(self.owner_distances_m) != len(self.owner_elements):
            raise ValueError("face owner distances are incomplete")
        if self.kind == SurfaceKind.INTACT_INTERIOR_INTERFACE and len(self.owner_elements) != 2:
            raise ValueError("an intact interface requires exactly two owners")
        if self.kind != SurfaceKind.INTACT_INTERIOR_INTERFACE and len(self.owner_elements) != 1:
            raise ValueError("transport boundaries must be independently one-owner faces")
        if not math.isfinite(self.length_m) or self.length_m <= 0.0:
            raise ValueError("transport face must have positive physical length")
        if any(not math.isfinite(value) or value <= 0.0 for value in self.owner_distances_m):
            raise ValueError("face-to-owner distances must be finite and positive")


@dataclass(frozen=True)
class TransportSurfaceRegistry:
    faces: tuple[TransportFace, ...]
    blocked_interior_edges: tuple[tuple[int, int], ...] = ()
    crack_physical_length_m: float = 0.0
    schema: str = SCHEMA

    def __post_init__(self) -> None:
        ids = [face.face_id for face in self.faces]
        if len(ids) != len(set(ids)):
            raise ValueError("transport face identities must be unique")
        if not math.isfinite(self.crack_physical_length_m) or self.crack_physical_length_m < 0.0:
            raise ValueError("crack transport length must be finite and nonnegative")

    @property
    def fingerprint(self) -> str:
        rows = []
        for face in self.faces:
            row = asdict(face)
            row["kind"] = face.kind.value
            rows.append(row)
        payload = {"schema": self.schema, "faces": rows,
                   "blocked_interior_edges": self.blocked_interior_edges,
                   "crack_physical_length_m": self.crack_physical_length_m}
        return hashlib.sha256(json.dumps(
            payload, sort_keys=True, separators=(",", ":"), allow_nan=False,
        ).encode()).hexdigest()

    def count(self, kind: SurfaceKind) -> int:
        return sum(face.kind == kind for face in self.faces)


@dataclass(frozen=True)
class ExactCutTransportMesh:
    mesh: Any
    parent_mechanics_element_ids: np.ndarray
    registry: TransportSurfaceRegistry
    cut_segment: tuple[tuple[float, float], tuple[float, float]]
    schema: str = SCHEMA

    def __post_init__(self):
        parent = np.asarray(self.parent_mechanics_element_ids, dtype=int).copy()
        if parent.shape != (self.mesh.ne,) or np.any(parent < 0):
            raise ValueError("cut transport cells require one mechanics parent")
        parent.setflags(write=False)
        object.__setattr__(self, "parent_mechanics_element_ids", parent)


def _clip_polygon_halfplane(polygon, p0, normal, keep_positive, tolerance=1e-15):
    result = []
    if not polygon:
        return result
    def signed(point): return float(np.dot(np.asarray(point)-p0, normal))
    for first, second in zip(polygon, polygon[1:]+polygon[:1]):
        f0, f1 = signed(first), signed(second)
        in0 = f0 >= -tolerance if keep_positive else f0 <= tolerance
        in1 = f1 >= -tolerance if keep_positive else f1 <= tolerance
        if in0:
            result.append(np.asarray(first, float))
        if in0 != in1:
            fraction = f0/(f0-f1)
            result.append(np.asarray(first, float)+fraction*(np.asarray(second)-np.asarray(first)))
    cleaned = []
    for point in result:
        if not cleaned or np.linalg.norm(point-cleaned[-1]) > tolerance:
            cleaned.append(point)
    if len(cleaned) > 1 and np.linalg.norm(cleaned[0]-cleaned[-1]) <= tolerance:
        cleaned.pop()
    return cleaned


def _insert_points_on_polygon_edges(polygon, points, tolerance):
    result = []
    for first, second in zip(polygon, polygon[1:]+polygon[:1]):
        result.append(np.asarray(first, float))
        edge = np.asarray(second)-np.asarray(first)
        scale2 = float(edge@edge)
        additions = []
        if scale2 > 0.0:
            for point in points:
                fraction = float((np.asarray(point)-first)@edge/scale2)
                distance = abs(_cross2(edge, np.asarray(point)-first))/math.sqrt(scale2)
                if tolerance < fraction < 1.0-tolerance and distance <= tolerance:
                    additions.append((fraction, np.asarray(point, float)))
        result.extend(point for _, point in sorted(additions, key=lambda item: item[0]))
    return result


def build_exact_cut_transport_mesh(
    mechanics_mesh: Any, *, p0: Sequence[float], p1: Sequence[float],
    topology_boundary_tags: Sequence[Mapping[str, Any]], segment_id: str = "accepted-crack",
) -> ExactCutTransportMesh:
    """Split crossed triangles and duplicate only the accepted open-crack line.

    The infinite supporting line supplies conforming cells in the tip element;
    nodes ahead of the accepted tip remain shared, whereas nodes on the open
    segment are side-qualified and therefore duplicated.  No concentration DOF
    is shared across the crack, while intact material immediately ahead of the
    tip remains connected.
    """
    start, tip = np.asarray(p0, float), np.asarray(p1, float)
    direction = tip-start
    length = float(np.linalg.norm(direction))
    if start.shape != (2,) or tip.shape != (2,) or length <= 0.0:
        raise ValueError("cut transport crack must have positive geometry")
    tangent = direction/length
    normal = np.array((-tangent[1], tangent[0]))
    source_nodes = np.asarray(mechanics_mesh.nodes, float)
    tolerance = max(1e-13*float(getattr(mechanics_mesh, "hbar", length)), 1e-15)
    new_nodes = []
    node_by_key = {}
    node_side = {}
    def node_id(point, side):
        axial = float(np.dot(point-start, tangent))
        on_line = abs(float(np.dot(point-start, normal))) <= tolerance
        on_open = on_line and tolerance < axial < length-tolerance
        qualifier = int(side) if on_open else 0
        key = (round(float(point[0])/tolerance), round(float(point[1])/tolerance), qualifier)
        if key not in node_by_key:
            node_by_key[key] = len(new_nodes); new_nodes.append(tuple(map(float, point)))
            node_side[node_by_key[key]] = qualifier
        return node_by_key[key]
    cells = []
    parents = []
    for parent, ids in enumerate(np.asarray(mechanics_mesh.elems, int)):
        triangle = source_nodes[ids]
        fragment = _clip_segment_triangle_length(start, tip, triangle)
        vertex_axial = (triangle-start)@tangent
        touches_open = (
            np.min(np.abs((triangle-start)@normal)) <= tolerance
            and np.max(vertex_axial) >= -tolerance and np.min(vertex_axial) <= length+tolerance
        )
        centroid_side = 1 if float(np.dot(triangle.mean(axis=0)-start, normal)) >= 0.0 else -1
        polygons = [(list(triangle), centroid_side if touches_open else 0)]
        if fragment > tolerance:
            polygons = [
                (_clip_polygon_halfplane(list(triangle), start, normal, True, tolerance), 1),
                (_clip_polygon_halfplane(list(triangle), start, normal, False, tolerance), -1),
            ]
        for polygon, side in polygons:
            if len(polygon) < 3:
                continue
            polygon = _insert_points_on_polygon_edges(polygon, (start, tip), tolerance)
            vertices = [node_id(point, side) for point in polygon]
            if len(vertices) == 3:
                proposed = [tuple(vertices)]
            else:
                center = np.mean(np.asarray(polygon), axis=0)
                center_id = node_id(center, side)
                proposed = [(center_id, vertices[index], vertices[(index+1) % len(vertices)])
                            for index in range(len(vertices))]
            for cell in proposed:
                coordinates = np.asarray(new_nodes)[list(cell)]
                if _cross2(coordinates[1]-coordinates[0], coordinates[2]-coordinates[0]) < 0:
                    cell = (cell[0], cell[2], cell[1])
                cells.append(cell); parents.append(parent)
    mesh = rebuild_tri_mesh(np.asarray(new_nodes), np.asarray(cells, int), tip_centers=tip)

    old_nodes = source_nodes
    old_segments = [(old_nodes[list(map(int, row["node_ids"]))], SurfaceKind(row["kind"]))
                    for row in topology_boundary_tags]
    tags = {}
    for edge, owners in _edge_owners(mesh).items():
        if len(owners) != 1:
            continue
        xy = mesh.nodes[list(edge)]
        axial = (xy-start)@tangent
        on_crack = (
            np.all(np.abs((xy-start)@normal) <= 10*tolerance)
            and np.min(axial) >= -10*tolerance and np.max(axial) <= length+10*tolerance
            and (node_side[edge[0]] != 0 or node_side[edge[1]] != 0)
        )
        if on_crack:
            tags[edge] = SurfaceKind.CRACK_FREE_SURFACE
            continue
        matches = []
        for segment, kind in old_segments:
            delta = segment[1]-segment[0]; scale2 = float(delta@delta)
            if scale2 <= 0.0: continue
            fractions = (xy-segment[0])@delta/scale2
            distances = np.abs(np.array([_cross2(delta, point-segment[0]) for point in xy]))/math.sqrt(scale2)
            if np.all(distances <= 10*tolerance) and np.min(fractions) >= -1e-10 and np.max(fractions) <= 1+1e-10:
                matches.append(kind)
        if len(set(matches)) != 1:
            raise RuntimeError(
                f"cut transport boundary ownership unresolved for edge {edge} at {xy.tolist()} "
                f"node_sides={(node_side[edge[0]], node_side[edge[1]])}"
            )
        tags[edge] = matches[0]
    registry = build_transport_surface_registry(mesh, boundary_tags=tags)
    crack_faces = []
    for face in registry.faces:
        if face.kind == SurfaceKind.CRACK_FREE_SURFACE:
            side = node_side[face.node_ids[0]] or node_side[face.node_ids[1]]
            crack_faces.append(replace(face, segment_id=segment_id, side=int(side)))
        else:
            crack_faces.append(face)
    registry = TransportSurfaceRegistry(
        faces=tuple(crack_faces), crack_physical_length_m=length,
    )
    upper = math.fsum(face.length_m for face in registry.faces
                      if face.kind == SurfaceKind.CRACK_FREE_SURFACE and face.side > 0)
    lower = math.fsum(face.length_m for face in registry.faces
                      if face.kind == SurfaceKind.CRACK_FREE_SURFACE and face.side < 0)
    if not (math.isclose(upper, length, rel_tol=1e-10, abs_tol=tolerance)
            and math.isclose(lower, length, rel_tol=1e-10, abs_tol=tolerance)):
        raise RuntimeError("cut transport mesh does not own two exact crack faces")
    return ExactCutTransportMesh(mesh, np.asarray(parents, int), registry,
                                 (tuple(start), tuple(tip)))


def build_exact_cut_transport_mesh_for_network(
    mechanics_mesh: Any, *, crack_segments: Sequence[tuple[Sequence[float], Sequence[float], str]],
    topology_boundary_tags: Sequence[Mapping[str, Any]],
) -> ExactCutTransportMesh:
    """Exact cut-cell transport mesh for every accepted crack-network segment."""
    segments = tuple((np.asarray(a, float), np.asarray(b, float), str(identity))
                     for a, b, identity in crack_segments)
    if not segments:
        raise ValueError("accepted crack network contains no physical segment")
    lengths = np.asarray([np.linalg.norm(b-a) for a, b, _ in segments], float)
    if np.any(lengths <= 0.0):
        raise ValueError("accepted crack network contains a zero-length segment")
    tangents = tuple((b-a)/length for (a, b, _), length in zip(segments, lengths))
    normals = tuple(np.array((-tangent[1], tangent[0])) for tangent in tangents)
    source_nodes = np.asarray(mechanics_mesh.nodes, float)
    tolerance = max(1e-13*float(getattr(mechanics_mesh, "hbar", np.max(lengths))), 1e-15)

    def line_crosses_open_segment(polygon, index):
        start, _, _ = segments[index]; tangent = tangents[index]; normal = normals[index]
        signed = np.asarray([(point-start)@normal for point in polygon])
        if np.min(signed) >= -tolerance or np.max(signed) <= tolerance:
            return False
        intersections = []
        for first, second in zip(polygon, polygon[1:]+polygon[:1]):
            f0 = float((first-start)@normal); f1 = float((second-start)@normal)
            if abs(f0) <= tolerance:
                intersections.append(float((first-start)@tangent))
            if f0*f1 < 0.0:
                point = first + f0/(f0-f1)*(second-first)
                intersections.append(float((point-start)@tangent))
        return bool(intersections) and min(max(intersections), lengths[index]) \
            - max(min(intersections), 0.0) > tolerance

    new_nodes = []; node_by_key = {}
    def node_id(point, signature):
        qualifiers = []
        for side, (start, _, _), tangent, normal, length in zip(
            signature, segments, tangents, normals, lengths,
        ):
            axial = float((point-start)@tangent)
            on_open = (abs(float((point-start)@normal)) <= tolerance
                       and tolerance < axial < length-tolerance)
            qualifiers.append(int(side) if on_open else 0)
        key = (round(float(point[0])/tolerance), round(float(point[1])/tolerance),
               tuple(qualifiers))
        if key not in node_by_key:
            node_by_key[key] = len(new_nodes)
            new_nodes.append(tuple(map(float, point)))
        return node_by_key[key]

    cells = []; parents = []; cell_signatures = []
    zero_signature = tuple(0 for _ in segments)
    for parent, ids in enumerate(np.asarray(mechanics_mesh.elems, int)):
        fragments = [(list(source_nodes[ids]), zero_signature)]
        for index, ((start, _, _), normal) in enumerate(zip(segments, normals)):
            divided = []
            for polygon, signature in fragments:
                if not line_crosses_open_segment(polygon, index):
                    axial = np.asarray([(point-start)@tangents[index] for point in polygon])
                    signed = np.asarray([(point-start)@normal for point in polygon])
                    touches = (np.min(np.abs(signed)) <= tolerance
                               and np.max(axial) >= -tolerance
                               and np.min(axial) <= lengths[index]+tolerance)
                    if touches:
                        centroid_sign = float((np.mean(np.asarray(polygon), axis=0)-start)@normal)
                        if abs(centroid_sign) > tolerance:
                            values = list(signature)
                            values[index] = 1 if centroid_sign > 0.0 else -1
                            signature = tuple(values)
                    divided.append((polygon, signature)); continue
                for side, positive in ((1, True), (-1, False)):
                    clipped = _clip_polygon_halfplane(
                        polygon, start, normal, positive, tolerance,
                    )
                    if len(clipped) >= 3:
                        values = list(signature); values[index] = side
                        divided.append((clipped, tuple(values)))
            fragments = divided
        endpoints = tuple(point for start, tip, _ in segments for point in (start, tip))
        for polygon, signature in fragments:
            polygon = _insert_points_on_polygon_edges(polygon, endpoints, tolerance)
            vertices = [node_id(point, signature) for point in polygon]
            center = np.mean(np.asarray(polygon), axis=0)
            proposed = ([tuple(vertices)] if len(vertices) == 3 else [
                (node_id(center, signature), vertices[i], vertices[(i+1) % len(vertices)])
                for i in range(len(vertices))
            ])
            for cell in proposed:
                coordinates = np.asarray(new_nodes)[list(cell)]
                if abs(_cross2(coordinates[1]-coordinates[0],
                               coordinates[2]-coordinates[0])) <= tolerance*tolerance:
                    continue
                if _cross2(coordinates[1]-coordinates[0], coordinates[2]-coordinates[0]) < 0:
                    cell = (cell[0], cell[2], cell[1])
                cells.append(cell); parents.append(parent); cell_signatures.append(signature)
    cut_mesh = rebuild_tri_mesh(np.asarray(new_nodes), np.asarray(cells, int),
                                tip_centers=np.asarray([item[1] for item in segments]))

    old_segments = [(source_nodes[list(map(int, row["node_ids"]))], SurfaceKind(row["kind"]))
                    for row in topology_boundary_tags]
    tags = {}; crack_metadata = {}
    for edge, edge_owners in _edge_owners(cut_mesh).items():
        if len(edge_owners) != 1:
            continue
        xy = cut_mesh.nodes[list(edge)]; midpoint = xy.mean(axis=0)
        crack_matches = []
        for index, ((start, _, identity), tangent, normal, length) in enumerate(zip(
            segments, tangents, normals, lengths,
        )):
            axial = (xy-start)@tangent
            if (np.all(np.abs((xy-start)@normal) <= 10*tolerance)
                    and np.min(axial) >= -10*tolerance
                    and np.max(axial) <= length+10*tolerance
                    and -tolerance <= float((midpoint-start)@tangent) <= length+tolerance):
                side = int(cell_signatures[edge_owners[0]][index])
                if side:
                    crack_matches.append((identity, side))
        if crack_matches:
            if len(set(crack_matches)) != 1:
                raise RuntimeError("ambiguous accepted crack-segment boundary ownership")
            tags[edge] = SurfaceKind.CRACK_FREE_SURFACE
            crack_metadata[edge] = crack_matches[0]
            continue
        matches = []
        for segment, kind in old_segments:
            delta = segment[1]-segment[0]; scale2 = float(delta@delta)
            if scale2 <= 0.0: continue
            fractions = (xy-segment[0])@delta/scale2
            distances = np.abs(np.array([_cross2(delta, point-segment[0]) for point in xy]))/math.sqrt(scale2)
            if np.all(distances <= 10*tolerance) and np.min(fractions) >= -1e-10 and np.max(fractions) <= 1+1e-10:
                matches.append(kind)
        if len(set(matches)) != 1:
            raise RuntimeError("network cut transport boundary ownership is unresolved")
        tags[edge] = matches[0]
    base_registry = build_transport_surface_registry(cut_mesh, boundary_tags=tags)
    faces = tuple(
        replace(face, segment_id=crack_metadata[tuple(sorted(face.node_ids))][0],
                side=crack_metadata[tuple(sorted(face.node_ids))][1])
        if face.kind == SurfaceKind.CRACK_FREE_SURFACE else face
        for face in base_registry.faces
    )
    registry = TransportSurfaceRegistry(
        faces=faces, crack_physical_length_m=float(math.fsum(lengths)),
    )
    for (_, _, identity), length in zip(segments, lengths):
        for side in (-1, 1):
            owned = math.fsum(face.length_m for face in faces
                              if face.segment_id == identity and face.side == side)
            if not math.isclose(owned, length, rel_tol=1e-10, abs_tol=tolerance):
                raise RuntimeError("network cut does not own two exact faces per crack segment")
    return ExactCutTransportMesh(
        cut_mesh, np.asarray(parents, int), registry,
        (tuple(segments[0][0]), tuple(segments[-1][1])),
    )


@dataclass(frozen=True)
class LocalizedCaptureSite:
    site_id: str
    center_xy_m: tuple[float, float]
    support_radius_m: float
    capture_rate_area_per_s: float
    reserved_inventory_area_m2: float = 0.0
    required_embryo_inventory_area_m2: float = 0.0
    hit_count: int = 0
    required_hit_count: int = 1
    birth_action: float = 0.0
    birth_barrier_J: float = 0.0
    birth_stress_work_J: float = 0.0
    threshold: float = math.inf
    birth_thresholds: tuple[float, ...] = ()
    rng_fingerprint: str = ""

    def __post_init__(self) -> None:
        values = (self.support_radius_m, self.capture_rate_area_per_s,
                  self.reserved_inventory_area_m2,
                  self.required_embryo_inventory_area_m2, self.birth_action,
                  self.birth_barrier_J)
        if any(not math.isfinite(float(v)) or float(v) < 0.0 for v in values):
            raise ValueError("site capture fields must be finite and nonnegative")
        if self.support_radius_m <= 0.0 or self.capture_rate_area_per_s < 0.0:
            raise ValueError("site capture support must be positive")
        if self.hit_count < 0 or self.required_hit_count <= 0 or self.hit_count > self.required_hit_count or not self.site_id:
            raise ValueError("site identity and hit count are invalid")
        if self.birth_thresholds and (
            len(self.birth_thresholds) < self.required_hit_count
            or any(not math.isfinite(float(value)) or float(value) <= 0.0
                   for value in self.birth_thresholds)
        ):
            raise ValueError("site birth thresholds are incomplete or invalid")
        if not math.isfinite(float(self.birth_stress_work_J)):
            raise ValueError("site birth stress work must be finite")

    def next_birth_threshold(self) -> float:
        if self.hit_count >= self.required_hit_count:
            return math.inf
        if self.birth_thresholds:
            return float(self.birth_thresholds[self.hit_count])
        return float(self.threshold) * float(self.hit_count+1)


TRANSPORT_LEDGER_KEYS = (
    "initial_owned_inventory_area_m2",
    "plastic_production_area_m2",
    "dislocation_climb_annihilation_area_m2",
    "dislocation_climb_emission_area_m2",
    "crack_surface_absorption_area_m2",
    "external_surface_absorption_area_m2",
    "active_cavity_absorption_area_m2",
    "subgrid_capture_area_m2",
    "reserved_embryo_inventory_area_m2",
    "subgrid_cavity_inventory_area_m2",
    "active_cavity_inventory_area_m2",
    "embryo_healing_annihilation_area_m2",
)


def _freeze_mapping(values: Mapping[str, float]) -> Mapping[str, float]:
    return FrozenTransportMapping({str(k): float(v) for k, v in values.items()})


class FrozenTransportMapping(dict):
    """Pickle-safe immutable mapping for accepted transport ownership."""
    def _immutable(self, *args, **kwargs):
        raise TypeError("accepted vacancy transport mapping is immutable")
    __setitem__ = __delitem__ = clear = pop = popitem = setdefault = update = _immutable

    def __reduce__(self):
        return (_make_frozen_transport_mapping, (dict(self),))


def _make_frozen_transport_mapping(values):
    frozen = dict.__new__(FrozenTransportMapping)
    dict.update(frozen, values)
    return frozen


@dataclass(frozen=True)
class VacancyTransportStateV2:
    vacancy_fraction_gp: np.ndarray
    vacancy_temperature_K: float
    parameter_row_id: str
    parameter_row_fingerprint: str
    atomic_volume_m3: float
    vacancy_formation_volume_m3: float
    surface_registry_fingerprint: str
    parameter_row: VacancyTransportParameterRowV2 | None = None
    surface_registry: TransportSurfaceRegistry | None = None
    transport_mesh: Any = None
    parent_mechanics_element_ids: np.ndarray | None = None
    transport_time_s: float = 0.0
    ledgers: Mapping[str, float] = field(default_factory=dict)
    site_states: tuple[LocalizedCaptureSite, ...] = ()
    last_solver_audit: Mapping[str, Any] = field(default_factory=dict)
    schema: str = SCHEMA

    def __post_init__(self) -> None:
        c = np.asarray(self.vacancy_fraction_gp, dtype=float).copy()
        if c.ndim != 1 or np.any(~np.isfinite(c)) or np.any(c <= 0.0) or np.any(c >= 1.0):
            raise ValueError("V2 vacancy field must be a bounded atomic fraction")
        c.setflags(write=False)
        object.__setattr__(self, "vacancy_fraction_gp", c)
        if not math.isfinite(self.vacancy_temperature_K) or self.vacancy_temperature_K <= 0.0:
            raise ValueError("vacancy temperature must be finite and positive")
        if not math.isfinite(self.transport_time_s) or self.transport_time_s < 0.0:
            raise ValueError("transport time must be finite and nonnegative")
        if set(self.ledgers) != set(TRANSPORT_LEDGER_KEYS):
            raise ValueError("V2 vacancy transport ledgers are incomplete")
        if any(not math.isfinite(float(v)) for v in self.ledgers.values()):
            raise ValueError("V2 vacancy transport ledgers must be finite")
        object.__setattr__(self, "ledgers", _freeze_mapping(self.ledgers))
        object.__setattr__(self, "last_solver_audit", _make_frozen_transport_mapping(dict(self.last_solver_audit)))
        if self.surface_registry is not None and self.surface_registry.fingerprint != self.surface_registry_fingerprint:
            raise ValueError("owned transport registry and fingerprint disagree")
        if self.parameter_row is not None and self.parameter_row.fingerprint != self.parameter_row_fingerprint:
            raise ValueError("owned transport parameter row and fingerprint disagree")
        if self.transport_mesh is not None and self.transport_mesh.ne != len(c):
            raise ValueError("transport mesh and concentration ownership disagree")
        parent = self.parent_mechanics_element_ids
        if parent is not None:
            parent = np.asarray(parent, dtype=int).copy()
            if parent.shape != c.shape or np.any(parent < 0):
                raise ValueError("transport cells require valid mechanics parents")
            parent.setflags(write=False)
            object.__setattr__(self, "parent_mechanics_element_ids", parent)

    @property
    def fingerprint(self) -> str:
        payload = {
            "vacancy_fraction_sha256": hashlib.sha256(
                np.ascontiguousarray(self.vacancy_fraction_gp).tobytes()).hexdigest(),
            "temperature_K": self.vacancy_temperature_K,
            "parameter_row_fingerprint": self.parameter_row_fingerprint,
            "surface_registry_fingerprint": self.surface_registry_fingerprint,
            "transport_time_s": self.transport_time_s,
            "ledgers": dict(self.ledgers),
            "site_states": [asdict(site) for site in self.site_states],
            "last_solver_audit": dict(self.last_solver_audit),
        }
        return hashlib.sha256(json.dumps(
            payload, sort_keys=True, separators=(",", ":"), allow_nan=False,
            default=lambda value: np.asarray(value).tolist(),
        ).encode()).hexdigest()


def equilibrium_vacancy_fraction(
    temperature_K: float, row: VacancyTransportParameterRowV2, mean_stress_Pa: Any = 0.0,
) -> np.ndarray | float:
    temperature = float(temperature_K)
    stress = np.asarray(mean_stress_Pa, dtype=float)
    exponent = (
        float(row.vacancy_formation_entropy_over_kB)
        - (float(row.vacancy_formation_energy_eV) * EV_J
           - float(row.vacancy_formation_volume_m3) * stress)
        / (KB_J_PER_K * temperature)
    )
    value = 1.0 / (1.0 + np.exp(np.clip(-exponent, -700.0, 700.0)))
    return float(value) if value.ndim == 0 else value


def chemical_potential_J(
    vacancy_fraction: Any, temperature_K: float, mean_stress_Pa: Any,
    row: VacancyTransportParameterRowV2,
    *, c_min: float = DEFAULT_C_MIN, c_max: float = DEFAULT_C_MAX,
) -> np.ndarray:
    c = np.asarray(vacancy_fraction, dtype=float)
    stress = np.asarray(mean_stress_Pa, dtype=float)
    if np.any(~np.isfinite(c)) or np.any(c < c_min) or np.any(c > c_max):
        raise ValueError("vacancy fraction lies outside fail-closed transport bounds")
    return (
        float(row.vacancy_formation_energy_eV) * EV_J
        - float(temperature_K) * float(row.vacancy_formation_entropy_over_kB) * KB_J_PER_K
        + KB_J_PER_K * float(temperature_K) * np.log(c / (1.0 - c))
        - float(row.vacancy_formation_volume_m3) * stress
    )


def thermodynamic_flux_1d(
    c: float, grad_c_per_m: float, grad_mean_stress_Pa_per_m: float,
    temperature_K: float, row: VacancyTransportParameterRowV2,
) -> float:
    """Pointwise full thermodynamic flux for a locally affine state."""
    mobility = float(c) * (1.0 - float(c))
    grad_mu = (
        KB_J_PER_K * float(temperature_K) / mobility * float(grad_c_per_m)
        - float(row.vacancy_formation_volume_m3) * float(grad_mean_stress_Pa_per_m)
    )
    legacy = REFERENCE_VACANCY_ROW
    diffusivity = float(row.vacancy_diffusivity_prefactor_m2_s) * math.exp(
        -float(row.vacancy_migration_energy_eV) * EV_J
        / (KB_J_PER_K * float(temperature_K))
    )
    return -diffusivity / (KB_J_PER_K * float(temperature_K)) * mobility * grad_mu


def dilute_flux_1d(
    c: float, grad_c_per_m: float, grad_mean_stress_Pa_per_m: float,
    temperature_K: float, row: VacancyTransportParameterRowV2,
) -> float:
    diffusivity = float(row.vacancy_diffusivity_prefactor_m2_s) * math.exp(
        -float(row.vacancy_migration_energy_eV) * EV_J
        / (KB_J_PER_K * float(temperature_K))
    )
    return -diffusivity * (
        float(grad_c_per_m)
        - float(row.vacancy_formation_volume_m3) * float(c)
        / (KB_J_PER_K * float(temperature_K)) * float(grad_mean_stress_Pa_per_m)
    )


def _edge_owners(mesh: Any) -> dict[tuple[int, int], list[int]]:
    owners: dict[tuple[int, int], list[int]] = {}
    for eid, tri in enumerate(np.asarray(mesh.elems, dtype=int)):
        for a, b in ((tri[0], tri[1]), (tri[1], tri[2]), (tri[2], tri[0])):
            owners.setdefault(tuple(sorted((int(a), int(b)))), []).append(int(eid))
    if any(len(value) not in (1, 2) for value in owners.values()):
        raise RuntimeError("singular transport topology: nonmanifold mechanics edge")
    return owners


def _cross2(a: np.ndarray, b: np.ndarray) -> float:
    return float(a[0] * b[1] - a[1] * b[0])


def _segments_intersect(a: np.ndarray, b: np.ndarray, p: np.ndarray, q: np.ndarray, tol: float) -> bool:
    ab, pq = b - a, q - p
    den = _cross2(ab, pq)
    if abs(den) <= tol:
        return False
    t = _cross2(p - a, pq) / den
    u = _cross2(p - a, ab) / den
    return tol < t < 1.0 - tol and -tol <= u <= 1.0 + tol


def _clip_segment_triangle_length(p0: np.ndarray, p1: np.ndarray, triangle: np.ndarray) -> float:
    """Liang-Barsky-style clipping against a consistently oriented triangle."""
    signed = 0.5 * _cross2(triangle[1] - triangle[0], triangle[2] - triangle[0])
    orient = 1.0 if signed >= 0.0 else -1.0
    lo, hi = 0.0, 1.0
    direction = p1 - p0
    for a, b in zip(triangle, np.roll(triangle, -1, axis=0)):
        edge = b - a
        f0 = orient * _cross2(edge, p0 - a)
        fd = orient * _cross2(edge, direction)
        if abs(fd) < 1.0e-30:
            if f0 < -1.0e-14:
                return 0.0
            continue
        bound = -f0 / fd
        if fd > 0.0:
            lo = max(lo, bound)
        else:
            hi = min(hi, bound)
        if hi <= lo:
            return 0.0
    return max(hi - lo, 0.0) * float(np.linalg.norm(direction))


def _segment_overlaps_triangle_edge(p0: np.ndarray, p1: np.ndarray, triangle: np.ndarray, tol: float) -> bool:
    direction = p1-p0
    length = float(np.linalg.norm(direction))
    tangent = direction/length
    for a, b in zip(triangle, np.roll(triangle, -1, axis=0)):
        if abs(_cross2(direction, a-p0))/length > tol or abs(_cross2(direction, b-p0))/length > tol:
            continue
        a1, b1 = sorted((float(np.dot(a-p0, tangent)), float(np.dot(b-p0, tangent))))
        if min(b1, length)-max(a1, 0.0) > tol:
            return True
    return False


def build_transport_surface_registry(
    mesh: Any,
    *,
    boundary_tags: Mapping[tuple[int, int], SurfaceKind | str],
    crack_segments: Sequence[tuple[Sequence[float], Sequence[float], str]] = (),
    merged_crack_segment_ids: Iterable[str] = (),
) -> TransportSurfaceRegistry:
    """Build transport faces from explicit external tags and accepted crack arcs.

    Nonconforming crack arcs add two independent embedded boundary fluxes per
    physical cut fragment and remove every dual connection crossing the arc.
    This makes sink strength depend on physical segment length, never wake width.
    """
    nodes = np.asarray(mesh.nodes, dtype=float)
    elems = np.asarray(mesh.elems, dtype=int)
    centroids = nodes[elems].mean(axis=1)
    owners = _edge_owners(mesh)
    normalized_tags = {tuple(sorted(map(int, edge))): SurfaceKind(kind)
                       for edge, kind in boundary_tags.items()}
    exterior = {edge for edge, value in owners.items() if len(value) == 1}
    if set(normalized_tags) != exterior:
        missing = sorted(exterior - set(normalized_tags))
        extra = sorted(set(normalized_tags) - exterior)
        raise ValueError(f"explicit transport boundary tags incomplete: missing={missing[:8]} extra={extra[:8]}")

    segments = [(np.asarray(a, float), np.asarray(b, float), str(sid))
                for a, b, sid in crack_segments]
    if any(a.shape != (2,) or b.shape != (2,) or np.linalg.norm(b-a) <= 0.0
           for a, b, _ in segments):
        raise ValueError("accepted crack segments must have positive two-dimensional geometry")
    merged = set(map(str, merged_crack_segment_ids))
    blocked: set[tuple[int, int]] = set()
    tol = max(float(getattr(mesh, "hbar", 1.0)) * 1.0e-10, 1.0e-15)
    for edge, own in owners.items():
        if len(own) != 2:
            continue
        ca, cb = centroids[own]
        if any(_segments_intersect(ca, cb, p0, p1, tol) for p0, p1, _ in segments):
            blocked.add(edge)

    faces: list[TransportFace] = []
    for edge, own in sorted(owners.items()):
        xy = nodes[np.asarray(edge, int)]
        center = xy.mean(axis=0)
        length = float(np.linalg.norm(xy[1] - xy[0]))
        edge_direction = (xy[1]-xy[0])/length
        edge_normal = np.array((-edge_direction[1], edge_direction[0]))
        if len(own) == 2 and edge not in blocked:
            distances = tuple(max(abs(float(np.dot(centroids[eid]-center, edge_normal))), tol)
                              for eid in own)
            faces.append(TransportFace(
                face_id=f"mesh:{edge[0]}-{edge[1]}", kind=SurfaceKind.INTACT_INTERIOR_INTERFACE,
                owner_elements=tuple(own), node_ids=edge, center_xy_m=tuple(center),
                length_m=length, owner_distances_m=distances,
            ))
        elif len(own) == 1:
            kind = normalized_tags[edge]
            distance = max(abs(float(np.dot(centroids[own[0]]-center, edge_normal))), tol)
            faces.append(TransportFace(
                face_id=f"boundary:{edge[0]}-{edge[1]}", kind=kind,
                owner_elements=(own[0],), node_ids=edge, center_xy_m=tuple(center),
                length_m=length, owner_distances_m=(distance,),
            ))

    physical_length = math.fsum(float(np.linalg.norm(p1-p0)) for p0, p1, _ in segments)
    triangles = nodes[elems]
    for p0, p1, segment_id in segments:
        direction = p1 - p0
        seg_length = float(np.linalg.norm(direction))
        normal = np.array([-direction[1], direction[0]]) / seg_length
        fragment_lengths = np.array([
            _clip_segment_triangle_length(p0, p1, triangle) for triangle in triangles
        ])
        cut_ids = np.flatnonzero(fragment_lengths > tol)
        if len(cut_ids) == 0:
            raise RuntimeError(f"transport crack segment {segment_id!r} has no mesh support")
        midpoint = 0.5 * (p0 + p1)
        kind = (SurfaceKind.MERGED_CAVITY_CRACK_WAKE if segment_id in merged
                else SurfaceKind.CRACK_FREE_SURFACE)
        for eid in cut_ids:
            signed = float(np.dot(centroids[eid] - midpoint, normal))
            side = 1 if signed >= 0.0 else -1
            distance = max(abs(signed), 0.25 * math.sqrt(float(mesh.area_e[eid])), tol)
            # A conforming slit already has one adjacent cell per physical
            # face.  A genuinely cut cell must instead own both embedded faces.
            face_sides = ((side,) if _segment_overlaps_triangle_edge(
                p0, p1, triangles[eid], tol,
            ) else (1, -1))
            for face_side in face_sides:
                faces.append(TransportFace(
                    face_id=f"crack:{segment_id}:e{int(eid)}:s{face_side:+d}", kind=kind,
                    owner_elements=(int(eid),), node_ids=None,
                    center_xy_m=tuple(midpoint), length_m=float(fragment_lengths[eid]),
                    owner_distances_m=(distance,), segment_id=segment_id, side=face_side,
                ))
    return TransportSurfaceRegistry(
        faces=tuple(faces), blocked_interior_edges=tuple(sorted(blocked)),
        crack_physical_length_m=physical_length,
    )


def build_registry_from_accepted_topology(state: Any) -> TransportSurfaceRegistry:
    """Rebuild transport ownership solely from accepted geometry-owned tags."""
    rows = state.junction_process_state.get("transport_boundary_tags_v2")
    if rows is None:
        raise RuntimeError("accepted state does not own explicit transport boundary tags")
    boundary_tags = {}
    for item in rows:
        edge = tuple(sorted(map(int, item["node_ids"])))
        if edge in boundary_tags:
            raise RuntimeError("accepted transport boundary edge has multiple owners")
        boundary_tags[edge] = SurfaceKind(item["kind"])
    segments = []
    merged_ids = set(state.junction_process_state.get("merged_cavity_crack_segment_ids", ()))
    for branch in state.crack_network.branches:
        for index, (first, second) in enumerate(zip(branch.path, branch.path[1:])):
            segments.append((first, second, f"{branch.branch_id}:{index}"))
    return build_transport_surface_registry(
        state.mesh, boundary_tags=boundary_tags, crack_segments=segments,
        merged_crack_segment_ids=merged_ids,
    )


def inherit_topology_boundary_tags(state: Any, new_mesh: Any) -> tuple[Mapping[str, Any], ...]:
    """Geometrically inherit persisted tags during a same-boundary remesh."""
    rows = state.junction_process_state.get("transport_boundary_tags_v2")
    if rows is None:
        raise RuntimeError("source mesh owns no transport boundary tags")
    old_nodes = np.asarray(state.mesh.nodes, dtype=float)
    old_segments = [
        (old_nodes[list(map(int, row["node_ids"]))], str(row["kind"])) for row in rows
    ]
    owners = _edge_owners(new_mesh)
    new_nodes = np.asarray(new_mesh.nodes, dtype=float)
    scale = max(float(getattr(new_mesh, "hbar", 1.0)), 1.0e-30)
    tolerance = max(1.0e-12*scale, 1.0e-15)
    inherited = []
    for edge, owner_ids in sorted(owners.items()):
        if len(owner_ids) != 1:
            continue
        endpoints = new_nodes[list(edge)]
        matches = []
        for segment, kind in old_segments:
            direction = segment[1]-segment[0]
            length2 = float(direction@direction)
            if length2 <= 0.0:
                continue
            fractions = (endpoints-segment[0])@direction/length2
            distances = np.abs(np.array([_cross2(direction, point-segment[0])
                                         for point in endpoints]))/math.sqrt(length2)
            if np.all(distances <= tolerance) and np.min(fractions) >= -tolerance/scale and np.max(fractions) <= 1.0+tolerance/scale:
                matches.append(kind)
        if len(set(matches)) != 1:
            raise RuntimeError(f"same-domain remesh cannot inherit boundary ownership for edge {edge}")
        inherited.append({"node_ids": edge, "kind": matches[0]})
    return tuple(inherited)


def initial_transport_state(
    mesh: Any, *, temperature_K: float, row: VacancyTransportParameterRowV2,
    registry: TransportSurfaceRegistry, vacancy_fraction_gp: np.ndarray | None = None,
    site_states: Sequence[LocalizedCaptureSite] = (),
    active_cavity_inventory_area_m2: float = 0.0,
    subgrid_cavity_inventory_area_m2: float = 0.0,
    parent_mechanics_element_ids: np.ndarray | None = None,
) -> VacancyTransportStateV2:
    if vacancy_fraction_gp is None:
        ceq = equilibrium_vacancy_fraction(temperature_K, row)
        vacancy_fraction_gp = np.full(mesh.ne, ceq, dtype=float)
    c = np.asarray(vacancy_fraction_gp, dtype=float)
    ratio = row.inventory_ratio
    field_inventory = ratio * math.fsum(float(a*v) for a, v in zip(mesh.area_e, c))
    reserved = math.fsum(site.reserved_inventory_area_m2 for site in site_states)
    cavity = float(active_cavity_inventory_area_m2)
    subgrid = float(subgrid_cavity_inventory_area_m2)
    ledgers = {key: 0.0 for key in TRANSPORT_LEDGER_KEYS}
    ledgers["initial_owned_inventory_area_m2"] = field_inventory + reserved + cavity + subgrid
    ledgers["reserved_embryo_inventory_area_m2"] = reserved
    ledgers["active_cavity_inventory_area_m2"] = cavity
    ledgers["subgrid_cavity_inventory_area_m2"] = subgrid
    return VacancyTransportStateV2(
        vacancy_fraction_gp=c, vacancy_temperature_K=float(temperature_K),
        parameter_row_id=row.row_id, parameter_row_fingerprint=row.fingerprint,
        parameter_row=row,
        atomic_volume_m3=row.atomic_volume_m3,
        vacancy_formation_volume_m3=row.vacancy_formation_volume_m3,
        surface_registry_fingerprint=registry.fingerprint,
        surface_registry=registry,
        transport_mesh=mesh,
        parent_mechanics_element_ids=(
            np.arange(mesh.ne, dtype=int) if parent_mechanics_element_ids is None
            else np.asarray(parent_mechanics_element_ids, dtype=int)
        ),
        ledgers=ledgers, site_states=tuple(site_states),
    )


def _capture_weights(mesh: Any, site: LocalizedCaptureSite) -> np.ndarray:
    centroids = np.asarray(mesh.nodes)[np.asarray(mesh.elems)].mean(axis=1)
    distance = np.linalg.norm(centroids - np.asarray(site.center_xy_m), axis=1)
    raw = np.maximum(1.0 - distance / float(site.support_radius_m), 0.0)
    normalization = math.fsum(float(a*w) for a, w in zip(mesh.area_e, raw))
    if normalization <= 0.0:
        raise RuntimeError(f"capture support for site {site.site_id!r} is unresolved")
    return raw / normalization


def _condition_estimate(matrix: sparse.csr_matrix) -> float:
    if matrix.shape[0] > 200:
        diagonal = np.abs(matrix.diagonal())
        positive = diagonal[diagonal > 0.0]
        return math.inf if not len(positive) else float(np.max(positive)/np.min(positive))
    if matrix.shape[0] <= 2:
        return float(np.linalg.cond(matrix.toarray()))
    try:
        largest = float(eigsh(matrix, k=1, which="LA", return_eigenvectors=False)[0])
        smallest = float(eigsh(matrix, k=1, which="SA", return_eigenvectors=False)[0])
        return largest / smallest
    except Exception:
        return math.inf


def _p0_gradient(mesh: Any, registry: TransportSurfaceRegistry, values: np.ndarray) -> np.ndarray:
    centers = np.asarray(mesh.nodes)[np.asarray(mesh.elems)].mean(axis=1)
    neighbors = [[] for _ in range(mesh.ne)]
    for face in registry.faces:
        if face.kind == SurfaceKind.INTACT_INTERIOR_INTERFACE:
            i, j = face.owner_elements
            neighbors[i].append(j); neighbors[j].append(i)
    gradient = np.zeros((mesh.ne, 2))
    for eid, adjacent in enumerate(neighbors):
        if len(adjacent) < 2:
            continue
        delta = centers[np.asarray(adjacent)]-centers[eid]
        rhs = values[np.asarray(adjacent)]-values[eid]
        gradient[eid] = np.linalg.lstsq(delta, rhs, rcond=None)[0]
    return gradient


def _rt0_system(
    mesh: Any, registry: TransportSurfaceRegistry, *, diffusivity: float,
    diagonal_reaction: np.ndarray, reaction_rhs: np.ndarray,
    concentration_iterate: np.ndarray, stress: np.ndarray, alpha: float,
    boundary_concentration: float,
    boundary_concentration_by_face: Mapping[str, float] | None = None,
):
    """Assemble the genuine element-mass RT0/P0 mixed saddle system."""
    geometric = [face for face in registry.faces if face.node_ids is not None]
    if len(geometric) != len(registry.faces):
        raise RuntimeError("production RT0 requires an exact conforming cut transport mesh")
    unknown_faces = [face for face in geometric
                     if face.kind == SurfaceKind.INTACT_INTERIOR_INTERFACE
                     or face.kind in PERFECT_SINK_KINDS]
    face_index = {tuple(sorted(face.node_ids)): index for index, face in enumerate(unknown_faces)}
    face_object = {tuple(sorted(face.node_ids)): face for face in geometric}
    nq, ne = len(unknown_faces), mesh.ne
    m_rows = []; m_cols = []; m_data = []
    b_rows = []; b_cols = []; b_data = []
    flux_rhs = np.zeros(nq)
    stress_gradient = _p0_gradient(mesh, registry, stress)
    nodes = np.asarray(mesh.nodes, float)
    for eid, ids in enumerate(np.asarray(mesh.elems, int)):
        triangle = nodes[ids]
        area = float(mesh.area_e[eid])
        local_edges = ((int(ids[1]), int(ids[2])),
                       (int(ids[2]), int(ids[0])),
                       (int(ids[0]), int(ids[1])))
        # Degree-two exact quadrature for products of linear RT0 bases.
        barycentric = np.array(((2/3,1/6,1/6),(1/6,2/3,1/6),(1/6,1/6,2/3)))
        points = barycentric@triangle
        basis = np.empty((3, 3, 2))
        for local in range(3):
            basis[:, local, :] = (points-triangle[local])/(2.0*area)
        local_mass = area/3.0*np.einsum("qia,qja->ij", basis, basis)/diffusivity
        mobility = concentration_iterate[eid]*(1.0-concentration_iterate[eid])
        drift_vector = alpha*mobility*stress_gradient[eid]
        basis_integral = area*np.mean(basis, axis=0)
        active = []
        for local, edge in enumerate(local_edges):
            key = tuple(sorted(edge)); face = face_object[key]
            if face.kind in NO_FLUX_KINDS:
                active.append((local, None, 0.0)); continue
            column = face_index[key]
            sign = 1.0 if face.owner_elements[0] == eid else -1.0
            active.append((local, column, sign))
            b_rows.append(eid); b_cols.append(column); b_data.append(sign)
            flux_rhs[column] += sign*float(np.dot(drift_vector, basis_integral[local]))
            if face.kind in PERFECT_SINK_KINDS:
                prescribed = (boundary_concentration if boundary_concentration_by_face is None
                              else float(boundary_concentration_by_face[face.face_id]))
                flux_rhs[column] -= sign*prescribed
        for local_i, col_i, sign_i in active:
            if col_i is None: continue
            for local_j, col_j, sign_j in active:
                if col_j is None: continue
                m_rows.append(col_i); m_cols.append(col_j)
                m_data.append(sign_i*sign_j*local_mass[local_i, local_j])
    M = sparse.csr_matrix((m_data, (m_rows, m_cols)), shape=(nq, nq))
    B = sparse.csr_matrix((b_data, (b_rows, b_cols)), shape=(ne, nq))
    C = sparse.diags(diagonal_reaction, format="csr")
    matrix = sparse.bmat(((M, -B.T), (-B, -C)), format="csr")
    rhs = np.concatenate((flux_rhs, -reaction_rhs))
    return matrix, rhs, unknown_faces, B


def _project_face_fluxes_to_positive_cell_balance(
    divergence_operator: sparse.csr_matrix,
    high_order_flux: np.ndarray,
    diagonal_reaction: np.ndarray,
    reaction_rhs: np.ndarray,
    *, concentration_floor: float,
    maximum_iterations: int = 20000,
    relative_tolerance: float = 5.0e-15,
) -> tuple[np.ndarray, np.ndarray, Mapping[str, Any]]:
    """Minimally correct shared face fluxes subject to positive cell balance.

    The correction is the Euclidean projection of the integrated RT0 face-flux
    vector onto ``B q <= r-C*c_floor``.  It is solved by scaled ADMM.  The
    concentration is then reconstructed from ``C c + B q = r`` rather than
    clipped, so every cell balance and every shared-face cancellation remains
    algebraically conservative.  If the high-order state is admissible this
    routine is not called and the RT0 result is unchanged.
    """
    B = divergence_operator.tocsr()
    q_high = np.asarray(high_order_flux, dtype=float)
    diagonal = np.asarray(diagonal_reaction, dtype=float)
    source = np.asarray(reaction_rhs, dtype=float)
    if np.any(diagonal <= 0.0) or np.any(~np.isfinite(diagonal)):
        raise RuntimeError("positive flux correction requires a positive reaction diagonal")
    flux_scale = max(
        float(np.linalg.norm(q_high, ord=np.inf)),
        float(np.linalg.norm(source, ord=np.inf)),
        np.finfo(float).tiny,
    )
    x0 = q_high/flux_scale
    upper = (source-diagonal*float(concentration_floor))/flux_scale
    rho = 1.0
    normal = sparse.eye(B.shape[1], format="csc") + rho*(B.T@B).tocsc()
    solve_normal = factorized(normal)
    x = x0.copy()
    Bx = np.asarray(B@x).reshape(-1)
    z = np.minimum(Bx, upper)
    dual = Bx-z
    primal_residual = math.inf
    dual_residual = math.inf
    for iteration in range(1, int(maximum_iterations)+1):
        x = np.asarray(solve_normal(x0+rho*np.asarray(B.T@(z-dual)).reshape(-1)))
        Bx = np.asarray(B@x).reshape(-1)
        old_z = z
        z = np.minimum(Bx+dual, upper)
        dual = dual+Bx-z
        primal_residual = float(np.linalg.norm(Bx-z, ord=np.inf))
        dual_residual = float(
            rho*np.linalg.norm(np.asarray(B.T@(z-old_z)).reshape(-1), ord=np.inf)
        )
        scale = max(
            1.0, float(np.linalg.norm(Bx, ord=np.inf)),
            float(np.linalg.norm(z, ord=np.inf)),
        )
        if max(primal_residual, dual_residual) <= relative_tolerance*scale:
            break
    else:
        raise RuntimeError("conservative positive face-flux correction did not converge")
    scaled_constraint_violation = float(np.max(np.maximum(Bx-upper, 0.0)))
    scaled_stationarity_residual = float(np.linalg.norm(
        x-x0+rho*np.asarray(B.T@dual).reshape(-1), ord=np.inf,
    ))
    scaled_dual_feasibility_residual = float(max(-float(np.min(dual)), 0.0))
    scaled_complementarity_residual = float(np.linalg.norm(
        dual*np.maximum(upper-z, 0.0), ord=np.inf,
    ))
    acceptance_scale = max(
        1.0, float(np.linalg.norm(x, ord=np.inf)),
        float(np.linalg.norm(x0, ord=np.inf)),
        float(np.linalg.norm(upper, ord=np.inf)),
    )
    acceptance_tolerance = max(
        4.0*float(relative_tolerance)*acceptance_scale,
        128.0*np.finfo(float).eps*acceptance_scale,
    )
    if max(
        scaled_constraint_violation,
        scaled_stationarity_residual,
        scaled_dual_feasibility_residual,
        scaled_complementarity_residual,
    ) > acceptance_tolerance:
        raise RuntimeError(
            "conservative positive face-flux correction failed KKT acceptance"
        )
    corrected_flux = flux_scale*x
    concentration = (
        source-np.asarray(B@corrected_flux).reshape(-1)
    )/diagonal
    balance_residual = diagonal*concentration + np.asarray(B@corrected_flux).reshape(-1)-source
    return corrected_flux, concentration, {
        "method": "SCALED_ADMM_EUCLIDEAN_SHARED_FACE_FLUX_PROJECTION",
        "iterations": int(iteration),
        "relative_tolerance": float(relative_tolerance),
        "scaled_primal_residual_inf": primal_residual,
        "scaled_dual_residual_inf": dual_residual,
        "scaled_constraint_violation_inf": scaled_constraint_violation,
        "scaled_stationarity_residual_inf": scaled_stationarity_residual,
        "scaled_dual_feasibility_residual_inf": scaled_dual_feasibility_residual,
        "scaled_complementarity_residual_inf": scaled_complementarity_residual,
        "scaled_kkt_acceptance_tolerance": acceptance_tolerance,
        "flux_scale_fraction_area_per_s": flux_scale,
        "flux_correction_inf_fraction_area_per_s": float(
            np.linalg.norm(corrected_flux-q_high, ord=np.inf)),
        "flux_correction_l2_fraction_area_per_s": float(
            np.linalg.norm(corrected_flux-q_high)),
        "cell_balance_residual_inf": float(np.linalg.norm(balance_residual, ord=np.inf)),
        "active_lower_cell_count": int(np.count_nonzero(
            concentration <= float(concentration_floor)*(1.0+1.0e-8)
        )),
    }


def solve_transport_interval(
    previous: VacancyTransportStateV2,
    mesh: Any,
    registry: TransportSurfaceRegistry,
    *,
    row: VacancyTransportParameterRowV2,
    duration_s: float,
    accepted_equivalent_plastic_strain_rate_s: np.ndarray,
    accepted_dislocation_density_m2: np.ndarray,
    accepted_mean_stress_Pa: np.ndarray,
    c_min: float = DEFAULT_C_MIN,
    c_max: float = DEFAULT_C_MAX,
    maximum_picard_iterations: int = 60,
    relative_tolerance: float = 1.0e-10,
    diagnostic_boundary_concentration_by_face: Mapping[str, float] | None = None,
    failure_injector=None, linear_solver_provider=None,
) -> VacancyTransportStateV2:
    """Advance one accepted mechanical interval without re-running plasticity."""
    if previous.schema != SCHEMA or previous.parameter_row_fingerprint != row.fingerprint:
        raise RuntimeError("vacancy transport state/parameter identity mismatch")
    if registry.fingerprint != previous.surface_registry_fingerprint:
        # A topology-changing caller must explicitly publish a state carrying
        # the rebuilt registry fingerprint before advancing.
        raise RuntimeError("stale transport surface registry")
    if diagnostic_boundary_concentration_by_face is not None:
        expected = {face.face_id for face in registry.faces if face.kind in PERFECT_SINK_KINDS}
        if set(diagnostic_boundary_concentration_by_face) != expected:
            raise ValueError("diagnostic Dirichlet data must cover every perfect-sink face exactly")
    dt = float(duration_s)
    if not math.isfinite(dt) or dt <= 0.0:
        raise ValueError("transport interval must be finite and positive")
    c0 = np.asarray(previous.vacancy_fraction_gp, dtype=float)
    area = np.asarray(mesh.area_e, dtype=float)
    epdot = np.asarray(accepted_equivalent_plastic_strain_rate_s, dtype=float)
    rho = np.asarray(accepted_dislocation_density_m2, dtype=float)
    stress = np.asarray(accepted_mean_stress_Pa, dtype=float)
    if any(values.shape != c0.shape for values in (area, epdot, rho, stress)):
        raise ValueError("transport fields must contain one value per solid triangle")
    if np.any(epdot < 0.0) or np.any(rho < 0.0) or np.any(~np.isfinite(stress)):
        raise ValueError("accepted source fields are invalid")
    if failure_injector is not None:
        failure_injector("source_assembly", previous)
        failure_injector("surface_classification", previous)

    temperature = float(previous.vacancy_temperature_K)
    diffusivity = float(row.vacancy_diffusivity_prefactor_m2_s) * math.exp(
        -float(row.vacancy_migration_energy_eV) * EV_J / (KB_J_PER_K * temperature)
    )
    plastic_source_baseline0 = float(row.plastic_vacancy_coefficient_per_strain) * epdot
    plastic_source_rho_enhancement0 = plastic_source_baseline0 * float(
        row.dislocation_production_boost
    ) * np.sqrt(rho / float(row.rho_reference_m2))
    plastic_source_mobile_interaction0 = (
        float(row.interaction_mobile_vacancy_fraction)
        * plastic_source_rho_enhancement0
    )
    plastic_source_direct_interaction_equivalent0 = (
        float(row.interaction_direct_cavitation_fraction)
        * plastic_source_rho_enhancement0
    )
    plastic_source_benign_interaction_equivalent0 = (
        float(row.interaction_benign_fraction)
        * plastic_source_rho_enhancement0
    )
    source0 = plastic_source_baseline0 + plastic_source_mobile_interaction0
    climb = diffusivity * float(row.dislocation_climb_capture_efficiency) * rho
    if row.bulk_vacancy_sink_spacing_m is not None:
        climb = climb + diffusivity / float(row.bulk_vacancy_sink_spacing_m)**2
    ceq_climb = np.asarray(equilibrium_vacancy_fraction(temperature, row, stress), dtype=float)
    ceq_surface = float(equilibrium_vacancy_fraction(temperature, row, 0.0))
    # The selected 300 K equilibrium can legitimately lie below the generic
    # solver scale.  Keep a strict positive bound, but derive it from the
    # accepted thermodynamic state instead of rejecting that state by default.
    positive_scale = min(float(np.min(c0)), float(np.min(ceq_climb)), ceq_surface)
    effective_c_min = max(np.nextafter(0.0, 1.0),
                          min(float(c_min), 0.1*positive_scale))
    alpha = float(row.vacancy_formation_volume_m3) / (KB_J_PER_K * temperature)

    capture_weights = [_capture_weights(mesh, site) for site in previous.site_states]
    c = c0.copy()
    matrix = None
    rhs = None
    mixed_solution = None
    unknown_faces = None
    divergence_operator = None
    picard_residual = math.inf
    linear_equilibration_used = False
    unscaled_rejected_candidate = None
    unscaled_rejected_residual = None
    positivity_line_search_steps = 0
    positivity_line_search_min_factor = 1.0
    conservative_flux_corrections = []
    high_order_linear_residual_inf = math.inf
    capture_active_set_mismatch_count = 0
    if failure_injector is not None:
        failure_injector("transport_matrix_assembly", previous)
    for iteration in range(1, maximum_picard_iterations + 1):
        diagonal = area / dt + area * (source0 + climb)
        reaction_rhs = area / dt * c0 + area * (source0 + climb * ceq_climb)

        capture_lambda = np.zeros_like(c)
        assembled_capture_active = c > ceq_climb
        for site, weights in zip(previous.site_states, capture_weights):
            capture_lambda += (
                float(site.capture_rate_area_per_s) * weights
                * assembled_capture_active
            )
        diagonal += area * capture_lambda
        reaction_rhs += area * capture_lambda * ceq_climb
        matrix, rhs, unknown_faces, divergence_operator = _rt0_system(
            mesh, registry, diffusivity=diffusivity,
            diagonal_reaction=diagonal, reaction_rhs=reaction_rhs,
            concentration_iterate=c, stress=stress, alpha=alpha,
            boundary_concentration=ceq_surface,
            boundary_concentration_by_face=diagnostic_boundary_concentration_by_face,
        )
        mixed_solution = np.asarray(
            spsolve(matrix, rhs) if linear_solver_provider is None else
            linear_solver_provider.solve(
                matrix, rhs, len(unknown_faces), dependency_identity={
                    "transport_mesh_nodes_sha256": hashlib.sha256(
                        np.ascontiguousarray(mesh.nodes).tobytes()).hexdigest(),
                    "transport_mesh_elements_sha256": hashlib.sha256(
                        np.ascontiguousarray(mesh.elems).tobytes()).hexdigest(),
                    "surface_registry_fingerprint": registry.fingerprint,
                    "unknown_face_order": tuple(repr(face) for face in unknown_faces),
                }),
            dtype=float)
        candidate = mixed_solution[len(unknown_faces):]
        if np.any(~np.isfinite(candidate)):
            raise RuntimeError("singular vacancy transport solve")
        if ((np.any(candidate < effective_c_min) or np.any(candidate > c_max))
                and linear_solver_provider is None):
            # The mixed matrix couples flux unknowns O(1e-20) to atomic
            # fractions O(1e-15).  SuperLU on the raw units can lose enough
            # digits to create a small negative P0 value.  Symmetric diagonal
            # equilibration solves the identical assembled equations; it is
            # neither a concentration projection nor a modified flux law.
            unscaled_rejected_candidate = candidate.copy()
            unscaled_rejected_residual = float(
                np.linalg.norm(matrix@mixed_solution-rhs, ord=np.inf))
            diagonal_scale = np.abs(np.asarray(matrix.diagonal(), float))
            positive_diagonal = diagonal_scale[diagonal_scale > 0.0]
            floor = (float(np.min(positive_diagonal))*np.finfo(float).eps
                     if len(positive_diagonal) else np.finfo(float).tiny)
            scaling = 1.0/np.sqrt(np.maximum(diagonal_scale, floor))
            scaling_matrix = sparse.diags(scaling, format="csr")
            equilibrated = scaling_matrix@matrix@scaling_matrix
            equilibrated_rhs = scaling*rhs
            solved = np.asarray(spsolve(equilibrated, equilibrated_rhs), float)
            mixed_solution = scaling*solved
            candidate = mixed_solution[len(unknown_faces):]
            linear_equilibration_used = True
        high_order_linear_residual_inf = float(
            np.linalg.norm(matrix@mixed_solution-rhs, ord=np.inf)
        )
        if np.any(candidate < effective_c_min):
            # The centered RT0 constitutive solve can have a non-positive P0
            # fixed point on a general triangular mesh.  Correct the registered
            # shared-face fluxes, then derive c from the unchanged cell balance.
            # A guard above underflow absorbs only the projection tolerance; it
            # is many orders below every physical concentration scale here.
            positive_scale_guard = max(
                effective_c_min,
                256.0*np.finfo(float).eps*max(
                    float(np.max(c0)), float(np.max(ceq_climb)),
                    float(np.max(np.maximum(candidate, 0.0))),
                ),
            )
            high_order_candidate = candidate.copy()
            high_order_flux = mixed_solution[:len(unknown_faces)].copy()
            high_order_candidate_min = float(np.min(candidate))
            corrected_flux, candidate, correction_audit = (
                _project_face_fluxes_to_positive_cell_balance(
                    divergence_operator,
                    mixed_solution[:len(unknown_faces)],
                    diagonal,
                    reaction_rhs,
                    concentration_floor=positive_scale_guard,
                )
            )
            mixed_solution = np.concatenate((corrected_flux, candidate))
            flux_delta = corrected_flux-high_order_flux
            support_tolerance = max(
                128.0*np.finfo(float).eps*max(
                    float(np.linalg.norm(high_order_flux, ord=np.inf)),
                    float(correction_audit["flux_scale_fraction_area_per_s"]),
                ),
                np.finfo(float).tiny,
            )
            affected_face_indices = np.flatnonzero(
                np.abs(flux_delta) > support_tolerance
            )
            if len(affected_face_indices):
                affected_cell_indices = np.unique(
                    divergence_operator[:, affected_face_indices].nonzero()[0]
                )
                local_high_order_flux_scale = float(np.max(
                    np.abs(high_order_flux[affected_face_indices])
                ))
            else:
                affected_cell_indices = np.asarray([], dtype=int)
                local_high_order_flux_scale = 0.0
            conservative_flux_corrections.append({
                **dict(correction_audit),
                "nonlinear_iteration": int(iteration),
                "concentration_guard": float(positive_scale_guard),
                "high_order_candidate_min": high_order_candidate_min,
                "corrected_candidate_min": float(np.min(candidate)),
                "affected_face_count": int(len(affected_face_indices)),
                "affected_cell_count": int(len(affected_cell_indices)),
                "affected_face_ids": [
                    unknown_faces[int(index)].face_id
                    for index in affected_face_indices[:32]
                ],
                "affected_cell_ids": [
                    int(index) for index in affected_cell_indices[:32]
                ],
                "local_high_order_flux_scale_fraction_area_per_s": (
                    local_high_order_flux_scale
                ),
                "local_flux_correction_relative_inf": float(
                    np.linalg.norm(flux_delta, ord=np.inf)
                    / max(local_high_order_flux_scale, np.finfo(float).tiny)
                ),
                "concentration_correction_integrated_abs_area": float(
                    math.fsum(
                        float(area[index]*abs(
                            candidate[index]-high_order_candidate[index]
                        ))
                        for index in affected_cell_indices
                    )
                ),
            })
        picard_delta_inf = float(np.linalg.norm(candidate-c, ord=np.inf))
        picard_residual = float(picard_delta_inf /
                                max(float(np.linalg.norm(candidate, ord=np.inf)), effective_c_min))

        # A Picard linearization is an iteration, not an accepted physical
        # state.  Keep the iterate strictly in the native thermodynamic domain
        # while retaining the original RT0/P0 fixed point.  At convergence the
        # undamped candidate (and its flux) is accepted, so this is neither
        # concentration clipping nor a mass correction.
        native_upper = np.nextafter(1.0, 0.0)
        delta = candidate-c
        line_search_factor = 1.0
        decreasing = delta < 0.0
        increasing = delta > 0.0
        if np.any(decreasing):
            line_search_factor = min(
                line_search_factor,
                0.9*float(np.min((c[decreasing]-effective_c_min)/(-delta[decreasing]))),
            )
        if np.any(increasing):
            line_search_factor = min(
                line_search_factor,
                0.9*float(np.min((native_upper-c[increasing])/delta[increasing])),
            )
        needs_native_line_search = bool(
            np.any(candidate < effective_c_min) or np.any(candidate >= 1.0)
        )
        line_search_failed = False
        if needs_native_line_search and picard_residual > relative_tolerance:
            if (not math.isfinite(line_search_factor)
                    or line_search_factor <= 32.0*np.finfo(float).eps):
                line_search_failed = True
            else:
                line_search_factor = min(line_search_factor, 1.0)
                c = c + line_search_factor*delta
                if (np.any(c <= effective_c_min) or np.any(c >= 1.0)
                        or np.any(~np.isfinite(c))):
                    raise RuntimeError("vacancy positivity line search left thermodynamic domain")
                positivity_line_search_steps += 1
                positivity_line_search_min_factor = min(
                    positivity_line_search_min_factor, line_search_factor,
                )
                continue
        below = np.flatnonzero(candidate < effective_c_min)
        above = np.flatnonzero(candidate > c_max)
        if line_search_failed or ((len(below) or len(above))
                                  and picard_residual <= relative_tolerance):
            centers = np.asarray(mesh.nodes, float)[np.asarray(mesh.elems, int)].mean(axis=1)
            violating = np.unique(np.concatenate((below, above)))
            # Preserve the extrema and a bounded set of offending cells; this
            # is enough to reproduce/classify the solve without dumping fields.
            ranked = sorted(
                (int(i) for i in violating),
                key=lambda i: max(
                    effective_c_min-float(candidate[i]),
                    float(candidate[i])-float(c_max),
                ),
                reverse=True,
            )[:16]
            flux_unknown = mixed_solution[:len(unknown_faces)]
            boundary_flux = {
                kind: float(math.fsum(
                    float(flux_unknown[index])
                    for index, face in enumerate(unknown_faces)
                    if face.kind == kind
                ))
                for kind in PERFECT_SINK_KINDS
            }
            raise VacancyTransportBoundsError({
                "schema": "vacancy-transport-bounds-rejection/1",
                "test_failed": (
                    "both" if len(below) and len(above) else
                    "lower" if len(below) else "upper"
                ),
                "nonlinear_iteration": int(iteration),
                "duration_s": dt,
                "transport_time_start_s": float(previous.transport_time_s),
                "transport_time_end_attempted_s": float(previous.transport_time_s)+dt,
                "concentration_input_min": float(np.min(c0)),
                "concentration_input_max": float(np.max(c0)),
                "last_nonlinear_iterate_min": float(np.min(c)),
                "last_nonlinear_iterate_max": float(np.max(c)),
                "linear_candidate_min": float(np.min(candidate)),
                "linear_candidate_max": float(np.max(candidate)),
                "unscaled_linear_candidate_min": (None if unscaled_rejected_candidate is None
                    else float(np.min(unscaled_rejected_candidate))),
                "unscaled_linear_candidate_max": (None if unscaled_rejected_candidate is None
                    else float(np.max(unscaled_rejected_candidate))),
                "unscaled_linear_residual_inf": unscaled_rejected_residual,
                "equilibrated_retry_used": bool(linear_equilibration_used),
                "equilibrated_linear_residual_inf": float(
                    np.linalg.norm(matrix@mixed_solution-rhs, ord=np.inf)),
                "effective_lower_bound": float(effective_c_min),
                "upper_bound": float(c_max),
                "lower_violation_count": int(len(below)),
                "upper_violation_count": int(len(above)),
                "candidate_iterate_inf_delta": float(np.linalg.norm(candidate-c, ord=np.inf)),
                "candidate_inf_scale": max(float(np.linalg.norm(candidate, ord=np.inf)), effective_c_min),
                "relative_tolerance": float(relative_tolerance),
                "maximum_picard_iterations": int(maximum_picard_iterations),
                "positivity_line_search_steps": int(positivity_line_search_steps),
                "positivity_line_search_min_factor": float(
                    positivity_line_search_min_factor),
                "positivity_line_search_failed": bool(line_search_failed),
                "violating_cells": [{
                    "transport_cell_id": i,
                    "parent_mechanics_element_id": int(previous.parent_mechanics_element_ids[i]),
                    "centroid_m": centers[i].tolist(),
                    "area_m2": float(area[i]),
                    "candidate": float(candidate[i]),
                    "lower_violation": max(float(effective_c_min-candidate[i]), 0.0),
                    "upper_violation": max(float(candidate[i]-c_max), 0.0),
                    "input": float(c0[i]),
                    "last_iterate": float(c[i]),
                    "plastic_source_rate_s": float(source0[i]),
                    "climb_sink_rate_s": float(climb[i]),
                    "equilibrium_fraction": float(ceq_climb[i]),
                    "equivalent_plastic_strain_rate_s": float(epdot[i]),
                    "dislocation_density_m2": float(rho[i]),
                    "mean_stress_Pa": float(stress[i]),
                } for i in ranked],
                "source_rate_min_s": float(np.min(source0)),
                "source_rate_max_s": float(np.max(source0)),
                "sink_rate_min_s": float(np.min(climb)),
                "sink_rate_max_s": float(np.max(climb)),
                "boundary_integrated_flux_by_kind": boundary_flux,
                "diffusivity_m2_s": diffusivity,
                "temperature_K": temperature,
                "parameter_row_id": row.row_id,
                "parameter_row_fingerprint": row.fingerprint,
                "surface_registry_fingerprint": registry.fingerprint,
                "transport_cell_count": int(len(candidate)),
                "transport_to_mechanics_mapping_sha256": hashlib.sha256(
                    np.ascontiguousarray(previous.parent_mechanics_element_ids).tobytes()
                ).hexdigest(),
            })
        capture_active_set_mismatch_count = int(np.count_nonzero(
            assembled_capture_active != (candidate > ceq_climb)
        )) if previous.site_states else 0
        c = candidate
        if (picard_residual <= relative_tolerance
                and capture_active_set_mismatch_count == 0):
            break
    else:
        error = RuntimeError("vacancy transport Picard iteration did not converge")
        error.diagnostic = {
            "schema": "vacancy-transport-picard-nonconvergence/1",
            "duration_s": float(dt),
            "transport_time_start_s": float(previous.transport_time_s),
            "transport_time_end_attempted_s": float(previous.transport_time_s) + float(dt),
            "nonlinear_iterations": int(maximum_picard_iterations),
            "nonlinear_relative_residual": float(picard_residual),
            "relative_tolerance": float(relative_tolerance),
            "capture_active_set_mismatch_count": int(capture_active_set_mismatch_count),
            "concentration_input_min": float(np.min(c0)),
            "concentration_input_max": float(np.max(c0)),
            "last_nonlinear_iterate_min": float(np.min(c)),
            "last_nonlinear_iterate_max": float(np.max(c)),
            "last_linear_candidate_min": float(np.min(candidate)),
            "last_linear_candidate_max": float(np.max(candidate)),
            "last_candidate_iterate_inf_delta": float(picard_delta_inf),
            "positivity_line_search_steps": int(positivity_line_search_steps),
            "positivity_line_search_min_factor": float(
                positivity_line_search_min_factor
            ),
            "conservative_flux_correction_count": int(
                len(conservative_flux_corrections)
            ),
            "last_conservative_flux_correction": (
                None if not conservative_flux_corrections
                else conservative_flux_corrections[-1]
            ),
            "parameter_row_id": row.row_id,
            "parameter_row_fingerprint": row.fingerprint,
            "surface_registry_fingerprint": registry.fingerprint,
            "transport_cell_count": int(len(candidate)),
        }
        raise error
    if failure_injector is not None:
        failure_injector("transport_solve", previous)

    # The mixed solution directly owns the oriented integrated RT0 face fluxes.
    # Positive boundary flux leaves the solid.
    face_fluxes: dict[str, float] = {}
    surface_absorption = {kind: 0.0 for kind in PERFECT_SINK_KINDS}
    flux_unknown = mixed_solution[:len(unknown_faces)]
    local_divergence = np.asarray(divergence_operator@flux_unknown).reshape(-1)
    by_id = {face.face_id: float(flux_unknown[index])
             for index, face in enumerate(unknown_faces)}
    for face in registry.faces:
        flux = by_id.get(face.face_id, 0.0)
        if face.kind in PERFECT_SINK_KINDS:
            surface_absorption[face.kind] += dt*flux
        face_fluxes[face.face_id] = flux

    # Reconstruct RT0 vectors at element centroids from the accepted integrated
    # normal-flux DOFs. One shared interior DOF makes normal continuity an
    # algebraic identity rather than a postprocessed assertion.
    face_by_edge = {tuple(sorted(face.node_ids)): face for face in registry.faces}
    rt0_flux_centroid = np.zeros((mesh.ne, 2))
    for eid, ids in enumerate(np.asarray(mesh.elems, int)):
        triangle = np.asarray(mesh.nodes)[ids]
        center = triangle.mean(axis=0); element_area = float(area[eid])
        for local, edge in enumerate(((int(ids[1]), int(ids[2])),
                                      (int(ids[2]), int(ids[0])),
                                      (int(ids[0]), int(ids[1])))):
            face = face_by_edge[tuple(sorted(edge))]
            sign = 1.0 if face.owner_elements[0] == eid else -1.0
            rt0_flux_centroid[eid] += sign*face_fluxes[face.face_id] \
                *(center-triangle[local])/(2.0*element_area)

    capture_by_site = []
    capture_total_fraction_area = 0.0
    updated_sites = []
    for site, weights in zip(previous.site_states, capture_weights):
        captured = dt * float(site.capture_rate_area_per_s) * math.fsum(
            float(a*w*max(cv-ce, 0.0))
            for a, w, cv, ce in zip(area, weights, c, ceq_climb)
        )
        capture_by_site.append(captured)
        capture_total_fraction_area += captured
        captured_inventory = row.inventory_ratio*captured
        area_per_hit = site.required_embryo_inventory_area_m2/site.required_hit_count
        thermal_factor = math.exp(
            -max(site.birth_barrier_J-site.birth_stress_work_J, 0.0)
            /(KB_J_PER_K*temperature)
        )
        action_increment = 0.0 if area_per_hit <= 0.0 else captured_inventory/area_per_hit*thermal_factor
        updated_sites.append(replace(
            site, reserved_inventory_area_m2=(
                site.reserved_inventory_area_m2 + captured_inventory
            ),
            birth_action=site.birth_action + action_increment,
        ))
    if failure_injector is not None:
        failure_injector("site_capture", previous)

    plastic_fraction_area = dt * math.fsum(float(a*s*(1.0-cv)) for a, s, cv in zip(area, source0, c))
    plastic_baseline_fraction_area = dt * math.fsum(
        float(a*s*(1.0-cv)) for a, s, cv in zip(area, plastic_source_baseline0, c)
    )
    plastic_rho_enhancement_fraction_area = dt * math.fsum(
        float(a*s*(1.0-cv)) for a, s, cv in zip(area, plastic_source_mobile_interaction0, c)
    )
    plastic_direct_interaction_equivalent_fraction_area = dt * math.fsum(
        float(a*s*(1.0-cv)) for a, s, cv in
        zip(area, plastic_source_direct_interaction_equivalent0, c)
    )
    plastic_benign_interaction_equivalent_fraction_area = dt * math.fsum(
        float(a*s*(1.0-cv)) for a, s, cv in
        zip(area, plastic_source_benign_interaction_equivalent0, c)
    )
    climb_values = dt * area * climb * (c-ceq_climb)
    climb_annihilation = math.fsum(float(max(v, 0.0)) for v in climb_values)
    climb_emission = math.fsum(float(max(-v, 0.0)) for v in climb_values)
    reaction_residual = (
        area*(c-c0)/dt + local_divergence
        - area*source0*(1.0-c) + area*climb*(c-ceq_climb)
    )
    for site, weights in zip(previous.site_states, capture_weights):
        reaction_residual += area * float(site.capture_rate_area_per_s) * weights * np.maximum(c-ceq_climb, 0.0)

    ratio = row.inventory_ratio
    ledgers = dict(previous.ledgers)
    ledgers["plastic_production_area_m2"] += ratio * plastic_fraction_area
    ledgers["dislocation_climb_annihilation_area_m2"] += ratio * climb_annihilation
    ledgers["dislocation_climb_emission_area_m2"] += ratio * climb_emission
    crack_absorption = surface_absorption[SurfaceKind.CRACK_FREE_SURFACE] + surface_absorption[SurfaceKind.MERGED_CAVITY_CRACK_WAKE]
    ledgers["crack_surface_absorption_area_m2"] += ratio * crack_absorption
    ledgers["external_surface_absorption_area_m2"] += ratio * surface_absorption[SurfaceKind.EXTERNAL_FREE_SURFACE]
    cavity_absorption = ratio * surface_absorption[SurfaceKind.ACTIVE_CAVITY_SURFACE]
    cavity_flux_budget = conservation_roundoff_budget_m2(cavity_absorption, 0.0)
    if cavity_absorption < -cavity_flux_budget:
        raise RuntimeError(
            "negative active-cavity flux is forbidden by the production one-way absorption policy"
        )
    cavity_absorption = max(cavity_absorption, 0.0)
    ledgers["active_cavity_absorption_area_m2"] += cavity_absorption
    ledgers["active_cavity_inventory_area_m2"] += cavity_absorption
    ledgers["subgrid_capture_area_m2"] += ratio * capture_total_fraction_area
    ledgers["reserved_embryo_inventory_area_m2"] += ratio * capture_total_fraction_area

    field_inventory = ratio * math.fsum(float(a*cv) for a, cv in zip(area, c))
    lhs = math.fsum((
        field_inventory,
        ledgers["reserved_embryo_inventory_area_m2"],
        ledgers["active_cavity_inventory_area_m2"],
        ledgers["subgrid_cavity_inventory_area_m2"],
        ledgers["crack_surface_absorption_area_m2"],
        ledgers["external_surface_absorption_area_m2"],
        ledgers["dislocation_climb_annihilation_area_m2"],
        -ledgers["dislocation_climb_emission_area_m2"],
        ledgers["embryo_healing_annihilation_area_m2"],
    ))
    rhs_inventory = math.fsum((
        ledgers["initial_owned_inventory_area_m2"],
        ledgers["plastic_production_area_m2"],
    ))
    conservation_residual = lhs-rhs_inventory
    budget = conservation_roundoff_budget_m2(lhs, rhs_inventory, *ledgers.values())
    if abs(conservation_residual) > budget:
        raise RuntimeError(
            f"vacancy transport conservation residual {conservation_residual:.6e} exceeds {budget:.6e}"
        )
    if failure_injector is not None:
        failure_injector("cavity_flux_integration", previous)

    audit = {
        "schema": SCHEMA, "model_id": MODEL_ID, "kappa_v": KAPPA_V,
        "duration_s": dt,
        "parameter_row_id": row.row_id, "parameter_row_fingerprint": row.fingerprint,
        "surface_registry_fingerprint": registry.fingerprint,
        "nonlinear_iteration_count": iteration,
        "nonlinear_relative_residual": picard_residual,
        "linear_residual_inf": high_order_linear_residual_inf,
        "high_order_linear_residual_inf": high_order_linear_residual_inf,
        "accepted_corrected_mixed_residual_inf": float(
            np.linalg.norm(matrix@mixed_solution-rhs, ord=np.inf)),
        "centered_constitutive_residual_inf": float(np.linalg.norm(
            (matrix@mixed_solution-rhs)[:len(unknown_faces)], ord=np.inf,
        )),
        "accepted_cell_balance_block_residual_inf": float(np.linalg.norm(
            (matrix@mixed_solution-rhs)[len(unknown_faces):], ord=np.inf,
        )),
        "linear_equilibration_used": linear_equilibration_used,
        "positivity_line_search_used": bool(positivity_line_search_steps),
        "positivity_line_search_steps": int(positivity_line_search_steps),
        "positivity_line_search_min_factor": float(positivity_line_search_min_factor),
        "accepted_nonlinear_fixed_point": True,
        "final_capture_active_set_mismatch_count": int(
            capture_active_set_mismatch_count
        ),
        "accepted_undamped_high_order_fixed_point": not bool(
            conservative_flux_corrections),
        "conservative_positive_flux_correction_used": bool(
            conservative_flux_corrections),
        "conservative_positive_flux_correction_count": int(
            len(conservative_flux_corrections)),
        "conservative_positive_flux_corrections": conservative_flux_corrections,
        "unscaled_rejected_linear_residual_inf": unscaled_rejected_residual,
        "unscaled_rejected_concentration_min": (None if unscaled_rejected_candidate is None
            else float(np.min(unscaled_rejected_candidate))),
        "matrix_condition_estimate": _condition_estimate(matrix),
        "local_element_conservation_residual_inf": float(np.linalg.norm(reaction_residual, ord=np.inf)),
        "local_element_conservation_residuals": reaction_residual.tolist(),
        "global_conservation_residual_area_m2": conservation_residual,
        "conservation_roundoff_budget_area_m2": budget,
        "field_inventory_area_m2": field_inventory,
        "face_flux_fraction_area_per_s": face_fluxes,
        "rt0_flux_vector_at_element_centroid": rt0_flux_centroid.tolist(),
        "rt0_normal_flux_dof_count": len(unknown_faces),
        "rt0_divergence_coupling_shape": list(divergence_operator.shape),
        "rt0_shared_interior_normal_flux_continuity_residual": 0.0,
        "rt0_local_mass_quadrature": "TRIANGLE_DEGREE_2_EXACT",
        "site_capture_fraction_area": capture_by_site,
        "active_cavity_absorption_increment_area_m2": cavity_absorption,
        "plastic_source_fraction_area": plastic_fraction_area,
        "plastic_source_baseline_rate_gp_s": plastic_source_baseline0.tolist(),
        "plastic_source_dislocation_enhancement_rate_gp_s": plastic_source_rho_enhancement0.tolist(),
        "plastic_source_mobile_interaction_rate_gp_s": plastic_source_mobile_interaction0.tolist(),
        "plastic_source_direct_interaction_equivalent_rate_gp_s": plastic_source_direct_interaction_equivalent0.tolist(),
        "plastic_source_benign_interaction_equivalent_rate_gp_s": plastic_source_benign_interaction_equivalent0.tolist(),
        "plastic_source_baseline_fraction_area": plastic_baseline_fraction_area,
        "plastic_source_dislocation_enhancement_fraction_area": plastic_rho_enhancement_fraction_area,
        "plastic_source_direct_interaction_equivalent_fraction_area": plastic_direct_interaction_equivalent_fraction_area,
        "plastic_source_benign_interaction_equivalent_fraction_area": plastic_benign_interaction_equivalent_fraction_area,
        "interaction_outcome_partition": {
            "mobile_vacancies": float(row.interaction_mobile_vacancy_fraction),
            "direct_cavitation_free_volume": float(row.interaction_direct_cavitation_fraction),
            "benign_hardening_recovery": float(row.interaction_benign_fraction),
            "sum": math.fsum((float(row.interaction_mobile_vacancy_fraction),
                              float(row.interaction_direct_cavitation_fraction),
                              float(row.interaction_benign_fraction))),
            "calibration_status": REFERENCE_UNCALIBRATED,
        },
        "climb_annihilation_fraction_area": climb_annihilation,
        "climb_emission_fraction_area": climb_emission,
        "climb_net_fraction_area_by_element": climb_values.tolist(),
        "diffusivity_m2_s": diffusivity,
        "dilute_validity_limit": DILUTE_VACANCY_FRACTION_LIMIT,
        "effective_positive_concentration_floor": effective_c_min,
    }
    return VacancyTransportStateV2(
        vacancy_fraction_gp=c,
        vacancy_temperature_K=temperature,
        parameter_row_id=row.row_id,
        parameter_row_fingerprint=row.fingerprint,
        parameter_row=row,
        atomic_volume_m3=row.atomic_volume_m3,
        vacancy_formation_volume_m3=row.vacancy_formation_volume_m3,
        surface_registry_fingerprint=registry.fingerprint,
        surface_registry=registry,
        transport_mesh=mesh,
        parent_mechanics_element_ids=previous.parent_mechanics_element_ids,
        transport_time_s=previous.transport_time_s+dt,
        ledgers=ledgers, site_states=tuple(updated_sites), last_solver_audit=audit,
    )


def rebind_registry_after_accepted_topology(
    state: VacancyTransportStateV2, registry: TransportSurfaceRegistry, *, failure_injector=None,
) -> VacancyTransportStateV2:
    """Publish a rebuilt accepted registry without changing physical state."""
    if failure_injector is not None:
        failure_injector("topology_commit", state)
    return replace(state, surface_registry_fingerprint=registry.fingerprint,
                   surface_registry=registry)


def transport_conservation(state: VacancyTransportStateV2, element_area_m2: np.ndarray) -> Mapping[str, float]:
    ratio = state.vacancy_formation_volume_m3 / state.atomic_volume_m3
    field = ratio * math.fsum(float(a*c) for a, c in zip(element_area_m2, state.vacancy_fraction_gp))
    ledger = state.ledgers
    lhs = math.fsum((
        field, ledger["reserved_embryo_inventory_area_m2"],
        ledger["active_cavity_inventory_area_m2"],
        ledger["subgrid_cavity_inventory_area_m2"],
        ledger["crack_surface_absorption_area_m2"],
        ledger["external_surface_absorption_area_m2"],
        ledger["dislocation_climb_annihilation_area_m2"],
        -ledger["dislocation_climb_emission_area_m2"],
        ledger["embryo_healing_annihilation_area_m2"],
    ))
    rhs = ledger["initial_owned_inventory_area_m2"] + ledger["plastic_production_area_m2"]
    return MappingProxyType({
        "field_inventory_area_m2": field,
        "conservation_lhs_area_m2": lhs,
        "conservation_rhs_area_m2": rhs,
        "conservation_residual_area_m2": lhs-rhs,
    })


def accepted_plane_strain_mean_stress_Pa(sigma_gp: np.ndarray, poisson_ratio: float) -> np.ndarray:
    """Legacy elastic-only reconstruction; prohibited for plastic production."""
    sigma = np.asarray(sigma_gp, dtype=float)
    nu = float(poisson_ratio)
    if sigma.ndim != 2 or sigma.shape[0] != 3 or np.any(~np.isfinite(sigma)):
        raise ValueError("accepted in-plane stress must have shape (3, ne)")
    if not math.isfinite(nu) or not -1.0 < nu < 0.5:
        raise ValueError("Poisson ratio is outside its stable range")
    sigma_zz = nu * (sigma[0] + sigma[1])
    return (sigma[0] + sigma[1] + sigma_zz) / 3.0


def accepted_elastoplastic_stress_3d(state: Any) -> Mapping[str, np.ndarray]:
    """Reconstruct the common accepted 3-D isotropic elastoplastic stress.

    The accepted engineering-shear P0 plastic strain is extended with the
    incompressible component ``ep_zz=-(ep_xx+ep_yy)``.  Thus ``sigma_zz`` is
    evaluated from the same accepted displacement and plastic history and is
    not the elastic shortcut ``nu*(sigma_xx+sigma_yy)``.
    """
    mesh = state.mesh
    conn = np.asarray(mesh.elems, int)
    u = np.asarray(state.displacement, float)
    edofs = np.column_stack((2*conn[:,0], 2*conn[:,0]+1,
                             2*conn[:,1], 2*conn[:,1]+1,
                             2*conn[:,2], 2*conn[:,2]+1))
    total = np.einsum("eij,ej->ei", mesh.B_e, u[edofs])
    plastic = np.asarray(state.ep_gp, float).T
    exx = total[:,0]-plastic[:,0]
    eyy = total[:,1]-plastic[:,1]
    gamma = total[:,2]-plastic[:,2]
    ezz = plastic[:,0]+plastic[:,1]
    trace = exx+eyy+ezz
    lam = float(state.material.lame_lambda)
    mu = float(state.material.lame_mu)
    damage = getattr(mesh, "element_damage_gp", None)
    dgp = (np.asarray(damage, float) if damage is not None
           else np.mean(np.asarray(state.damage)[conn], axis=1))
    degradation = (1.0-dgp)**2+1.0e-6
    sx = degradation*(2.0*mu*exx+lam*trace)
    sy = degradation*(2.0*mu*eyy+lam*trace)
    szz = degradation*(2.0*mu*ezz+lam*trace)
    txy = degradation*mu*gamma
    mean = (sx+sy+szz)/3.0
    for array in (sx, sy, szz, txy, mean):
        array.setflags(write=False)
    return MappingProxyType({
        "sigma_xx_Pa": sx, "sigma_yy_Pa": sy, "sigma_xy_Pa": txy,
        "sigma_zz_Pa": szz, "mean_stress_Pa": mean,
        "source": "accepted_3d_elastoplastic_constitutive_state",
    })


def heal_embryo_reservation(
    state: VacancyTransportStateV2, mesh: Any, site_id: str, *,
    disposition: str = "RETURN_TO_FIELD", c_max: float = DEFAULT_C_MAX,
) -> VacancyTransportStateV2:
    """Conservatively dispose of one healing embryo's reserved inventory."""
    sites = list(state.site_states)
    try:
        index = next(i for i, site in enumerate(sites) if site.site_id == site_id)
    except StopIteration as exc:
        raise KeyError(f"unknown vacancy capture site {site_id!r}") from exc
    site = sites[index]
    reserved = float(site.reserved_inventory_area_m2)
    if reserved <= 0.0:
        return state
    ledgers = dict(state.ledgers)
    ledgers["reserved_embryo_inventory_area_m2"] -= reserved
    c = np.asarray(state.vacancy_fraction_gp, dtype=float).copy()
    if disposition == "RETURN_TO_FIELD":
        weights = _capture_weights(mesh, site)
        ratio = state.vacancy_formation_volume_m3/state.atomic_volume_m3
        c += reserved/ratio*weights
        if np.any(c > c_max):
            raise RuntimeError("healed embryo return would exceed vacancy validity bound")
    elif disposition == "EXPLICIT_ANNIHILATION_LEDGER":
        ledgers["embryo_healing_annihilation_area_m2"] += reserved
    else:
        raise ValueError("unknown healed-embryo inventory disposition")
    sites[index] = replace(site, reserved_inventory_area_m2=0.0)
    return replace(state, vacancy_fraction_gp=c, ledgers=ledgers, site_states=tuple(sites))


def stabilize_reserved_embryo(
    state: VacancyTransportStateV2, site_id: str,
) -> VacancyTransportStateV2:
    """Convert reservation to subgrid cavity inventory without creation."""
    sites = list(state.site_states)
    try:
        index = next(i for i, site in enumerate(sites) if site.site_id == site_id)
    except StopIteration as exc:
        raise KeyError(f"unknown vacancy capture site {site_id!r}") from exc
    site = sites[index]
    reserved = float(site.reserved_inventory_area_m2)
    if reserved <= 0.0:
        raise RuntimeError("stabilization requires captured embryo inventory")
    ledgers = dict(state.ledgers)
    ledgers["reserved_embryo_inventory_area_m2"] -= reserved
    ledgers["subgrid_cavity_inventory_area_m2"] += reserved
    sites[index] = replace(site, reserved_inventory_area_m2=0.0)
    return replace(state, ledgers=ledgers, site_states=tuple(sites))


def transfer_reserved_embryo_to_growth(
    state: VacancyTransportStateV2, site_id: str,
) -> tuple[VacancyTransportStateV2, float]:
    """Move site-local reservation ownership into EmbryoGrowthState atomically.

    The global reserved-inventory ledger remains unchanged until stabilization;
    only the capture-site available balance is removed.
    """
    sites = list(state.site_states)
    index = next(i for i, site in enumerate(sites) if site.site_id == site_id)
    amount = float(sites[index].reserved_inventory_area_m2)
    sites[index] = replace(sites[index], reserved_inventory_area_m2=0.0)
    return replace(state, site_states=tuple(sites)), amount


def heal_growth_owned_inventory(
    state: VacancyTransportStateV2, mesh: Any, site_id: str,
    inventory_area_m2: float, *, disposition: str = "RETURN_TO_FIELD",
    c_max: float = DEFAULT_C_MAX,
) -> VacancyTransportStateV2:
    """Conservatively dispose of reservation already moved into growth ownership."""
    amount = float(inventory_area_m2)
    site = next(x for x in state.site_states if x.site_id == site_id)
    ledgers = dict(state.ledgers)
    if amount < 0.0 or amount > ledgers["reserved_embryo_inventory_area_m2"]+1e-24:
        raise RuntimeError("growth-owned healing exceeds reserved inventory")
    ledgers["reserved_embryo_inventory_area_m2"] -= amount
    c = np.asarray(state.vacancy_fraction_gp, float).copy()
    if disposition == "RETURN_TO_FIELD":
        weights = _capture_weights(mesh, site)
        c += amount/(state.vacancy_formation_volume_m3/state.atomic_volume_m3)*weights
        if np.any(c > c_max):
            raise RuntimeError("growth-owned healing return exceeds vacancy bound")
    elif disposition == "EXPLICIT_ANNIHILATION_LEDGER":
        ledgers["embryo_healing_annihilation_area_m2"] += amount
    else:
        raise ValueError("unknown growth-owned healing disposition")
    return replace(state, vacancy_fraction_gp=c, ledgers=ledgers)


def stabilize_growth_owned_inventory(
    state: VacancyTransportStateV2, inventory_area_m2: float,
) -> VacancyTransportStateV2:
    amount = float(inventory_area_m2)
    ledgers = dict(state.ledgers)
    if amount < 0.0 or amount > ledgers["reserved_embryo_inventory_area_m2"]+1e-24:
        raise RuntimeError("growth-owned stabilization exceeds reserved inventory")
    ledgers["reserved_embryo_inventory_area_m2"] -= amount
    ledgers["subgrid_cavity_inventory_area_m2"] += amount
    return replace(state, ledgers=ledgers)


def promote_subgrid_inventory(
    state: VacancyTransportStateV2, inventory_area_m2: float,
) -> VacancyTransportStateV2:
    """Representation-only promotion; total owned inventory is unchanged."""
    inventory = float(inventory_area_m2)
    ledgers = dict(state.ledgers)
    if not math.isfinite(inventory) or inventory < 0.0:
        raise ValueError("promotion inventory must be finite and nonnegative")
    if inventory > ledgers["subgrid_cavity_inventory_area_m2"] + 1.0e-24:
        raise RuntimeError("promotion cannot create cavity inventory")
    ledgers["subgrid_cavity_inventory_area_m2"] -= inventory
    ledgers["active_cavity_inventory_area_m2"] += inventory
    return replace(state, ledgers=ledgers)


def cavity_growth_rate_from_transport(
    *, absorbed_inventory_rate_area_m2_s: float, radius_m: float,
    equivalent_plastic_strain_rate_s: float, plastic_accommodation_coefficient: float,
) -> Mapping[str, float | str]:
    """Series-limited radius rate; phase-field flux is the sole supply rate."""
    supply = float(absorbed_inventory_rate_area_m2_s)
    radius = float(radius_m)
    plastic_rate = float(equivalent_plastic_strain_rate_s)
    coefficient = float(plastic_accommodation_coefficient)
    if any(not math.isfinite(v) or v < 0.0 for v in (supply, radius, plastic_rate, coefficient)):
        raise ValueError("cavity growth inputs must be finite and nonnegative")
    if radius <= 0.0:
        raise ValueError("cavity radius must be positive")
    transport = supply / (2.0 * math.pi * radius)
    plastic = coefficient * radius * plastic_rate
    if transport <= 0.0 or plastic <= 0.0:
        combined = 0.0
    else:
        combined = 1.0 / (1.0/transport + 1.0/plastic)
    if transport < plastic:
        branch = "VACANCY_TRANSPORT_LIMITED"
    elif plastic < transport:
        branch = "PLASTIC_ACCOMMODATION_LIMITED"
    else:
        branch = "CO_LIMITED"
    return MappingProxyType({
        "Rdot_transport_m_s": transport,
        "Rdot_plastic_m_s": plastic,
        "Rdot_accommodation_m_s": plastic,
        "Rdot_m_s": combined,
        "limiting_branch": branch,
    })


def preview_site_inventory_event(
    previous: VacancyTransportStateV2, mesh: Any, registry: TransportSurfaceRegistry,
    *, row: VacancyTransportParameterRowV2, maximum_duration_s: float,
    accepted_equivalent_plastic_strain_rate_s: np.ndarray,
    accepted_dislocation_density_m2: np.ndarray,
    accepted_mean_stress_Pa: np.ndarray,
    time_tolerance_s: float = 1.0e-10, failure_injector=None,
    linear_solver_provider=None,
) -> tuple[float | None, VacancyTransportStateV2]:
    """Locate the earliest inventory-backed site hit on the accepted clock.

    Every trial is pure.  Therefore an injected failure or a rejected root
    leaves ``previous`` bit-for-bit unchanged and no losing clock is skipped.
    """
    maximum = float(maximum_duration_s)
    if maximum <= 0.0 or not math.isfinite(maximum):
        raise ValueError("event horizon must be finite and positive")
    kwargs = dict(
        row=row,
        accepted_equivalent_plastic_strain_rate_s=accepted_equivalent_plastic_strain_rate_s,
        accepted_dislocation_density_m2=accepted_dislocation_density_m2,
        accepted_mean_stress_Pa=accepted_mean_stress_Pa,
    )
    endpoint = solve_transport_interval(
        previous, mesh, registry, duration_s=maximum,
        failure_injector=failure_injector, **kwargs,
        linear_solver_provider=linear_solver_provider,
    )

    def eligibility(state: VacancyTransportStateV2) -> list[bool]:
        values: list[bool] = []
        for site in state.site_states:
            if (site.required_embryo_inventory_area_m2 <= 0.0
                    or site.hit_count >= site.required_hit_count):
                values.append(False)
                continue
            per_hit = site.required_embryo_inventory_area_m2 / site.required_hit_count
            required_inventory = per_hit*float(site.hit_count+1)
            values.append(
                site.reserved_inventory_area_m2 >= required_inventory
                and site.birth_action >= site.next_birth_threshold()
            )
        return values

    eligible = [i for i, value in enumerate(eligibility(endpoint)) if value]
    if not eligible:
        return None, endpoint
    low, high = 0.0, maximum
    event_state = endpoint
    while high-low > max(time_tolerance_s, 8.0*np.finfo(float).eps*maximum):
        mid = 0.5*(low+high)
        trial = solve_transport_interval(
            previous, mesh, registry, duration_s=mid,
            failure_injector=failure_injector, **kwargs,
            linear_solver_provider=linear_solver_provider,
        )
        if any(eligibility(trial)):
            high, event_state = mid, trial
        else:
            low = mid
    if failure_injector is not None:
        failure_injector("event_root", previous)
    return high, event_state


def advance_transport_to_time(
    previous: VacancyTransportStateV2, mesh: Any, registry: TransportSurfaceRegistry,
    *, row: VacancyTransportParameterRowV2, elapsed_time_s: float,
    accepted_equivalent_plastic_strain_rate_s: np.ndarray,
    accepted_dislocation_density_m2: np.ndarray,
    accepted_mean_stress_Pa: np.ndarray, failure_injector=None,
    linear_solver_provider=None,
) -> VacancyTransportStateV2:
    """Advance the accepted RT0/P0 owner without committing a site event."""
    elapsed = float(elapsed_time_s)
    if not math.isfinite(elapsed) or elapsed < 0.0:
        raise ValueError("transport advance time must be finite and nonnegative")
    if elapsed == 0.0:
        return previous
    return solve_transport_interval(
        previous, mesh, registry, row=row, duration_s=elapsed,
        accepted_equivalent_plastic_strain_rate_s=accepted_equivalent_plastic_strain_rate_s,
        accepted_dislocation_density_m2=accepted_dislocation_density_m2,
        accepted_mean_stress_Pa=accepted_mean_stress_Pa,
        failure_injector=failure_injector,
        linear_solver_provider=linear_solver_provider,
    )


def commit_site_inventory_hit(
    original: VacancyTransportStateV2, advanced: VacancyTransportStateV2,
    *, expected_site_id: str | None = None,
) -> VacancyTransportStateV2:
    """Consume exactly one already-qualified, inventory-backed site hit."""
    if (original.parameter_row_fingerprint != advanced.parameter_row_fingerprint
            or original.surface_registry_fingerprint != advanced.surface_registry_fingerprint):
        raise RuntimeError("stale vacancy-site preview: transport ownership changed")
    original_by_id = {site.site_id: site for site in original.site_states}
    candidates = []
    for index, site in enumerate(advanced.site_states):
        old = original_by_id.get(site.site_id)
        if old is None or site.hit_count != old.hit_count:
            raise RuntimeError("site preview changed hit ownership before commit")
        if site.required_embryo_inventory_area_m2 <= 0.0 or site.hit_count >= site.required_hit_count:
            continue
        required = (site.required_embryo_inventory_area_m2/site.required_hit_count
                    * float(site.hit_count+1))
        if (site.reserved_inventory_area_m2 + 1.0e-24 >= required
                and site.birth_action >= site.next_birth_threshold()):
            candidates.append(index)
    if expected_site_id is not None:
        candidates = [index for index in candidates
                      if advanced.site_states[index].site_id == expected_site_id]
    if not candidates:
        raise RuntimeError("vacancy-site hit is not eligible at commit")
    winner = min(candidates, key=lambda index: advanced.site_states[index].site_id)
    sites = list(advanced.site_states)
    site = sites[winner]
    sites[winner] = replace(site, hit_count=site.hit_count+1)
    return replace(advanced, site_states=tuple(sites))


def locate_site_inventory_event(
    previous: VacancyTransportStateV2, mesh: Any, registry: TransportSurfaceRegistry,
    *, row: VacancyTransportParameterRowV2, maximum_duration_s: float,
    accepted_equivalent_plastic_strain_rate_s: np.ndarray,
    accepted_dislocation_density_m2: np.ndarray,
    accepted_mean_stress_Pa: np.ndarray,
    time_tolerance_s: float = 1.0e-10, failure_injector=None,
    linear_solver_provider=None,
) -> tuple[float | None, VacancyTransportStateV2]:
    """Compatibility wrapper preserving the former locate-and-commit result."""
    time, advanced = preview_site_inventory_event(
        previous, mesh, registry, row=row, maximum_duration_s=maximum_duration_s,
        accepted_equivalent_plastic_strain_rate_s=accepted_equivalent_plastic_strain_rate_s,
        accepted_dislocation_density_m2=accepted_dislocation_density_m2,
        accepted_mean_stress_Pa=accepted_mean_stress_Pa,
        time_tolerance_s=time_tolerance_s, failure_injector=failure_injector,
        linear_solver_provider=linear_solver_provider,
    )
    if time is None:
        return None, advanced
    return time, commit_site_inventory_hit(previous, advanced)


def compete_on_common_transport_clock(
    previous: VacancyTransportStateV2, mesh: Any, registry: TransportSurfaceRegistry,
    *, row: VacancyTransportParameterRowV2, maximum_duration_s: float,
    accepted_equivalent_plastic_strain_rate_s: np.ndarray,
    accepted_dislocation_density_m2: np.ndarray,
    accepted_mean_stress_Pa: np.ndarray,
    external_event_times_s: Mapping[str, float | None] = MappingProxyType({}),
    time_tolerance_s: float = 1.0e-10, failure_injector=None,
    linear_solver_provider=None,
) -> Mapping[str, Any]:
    """Compete transport events with times localized by the existing schedulers.

    ``external_event_times_s`` is the narrow adapter for ordinary cleavage,
    emission, ligament, and downstream first-passage clocks.  Their ownership
    and RNG remain in the frozen fracture/void schedulers.  The returned
    elapsed time is the one interval through which those callers must advance
    every losing clock before committing the named event.
    """
    maximum = float(maximum_duration_s)
    candidates = []
    for name, value in external_event_times_s.items():
        if value is None:
            continue
        time = float(value)
        if not math.isfinite(time) or time < 0.0:
            raise ValueError("external event times must be finite and nonnegative")
        if time <= maximum:
            candidates.append((time, str(name)))
    site_time, site_state = locate_site_inventory_event(
        previous, mesh, registry, row=row, maximum_duration_s=maximum,
        accepted_equivalent_plastic_strain_rate_s=accepted_equivalent_plastic_strain_rate_s,
        accepted_dislocation_density_m2=accepted_dislocation_density_m2,
        accepted_mean_stress_Pa=accepted_mean_stress_Pa,
        time_tolerance_s=time_tolerance_s, failure_injector=failure_injector,
        linear_solver_provider=linear_solver_provider,
    )
    if site_time is not None:
        candidates.append((site_time, "VACANCY_BIRTH_HIT"))
    if not candidates:
        return MappingProxyType({
            "elapsed_time_s": maximum, "event": None, "transport_state": site_state,
            "tie_group": (), "all_losing_clocks_advance_s": maximum,
        })
    earliest = min(time for time, _ in candidates)
    tie = tuple(sorted(name for time, name in candidates
                       if abs(time-earliest) <= time_tolerance_s))
    if site_time is not None and abs(site_time-earliest) <= time_tolerance_s:
        accepted = site_state
    else:
        accepted = solve_transport_interval(
            previous, mesh, registry, row=row, duration_s=max(earliest, np.finfo(float).tiny),
            accepted_equivalent_plastic_strain_rate_s=accepted_equivalent_plastic_strain_rate_s,
            accepted_dislocation_density_m2=accepted_dislocation_density_m2,
            accepted_mean_stress_Pa=accepted_mean_stress_Pa,
            linear_solver_provider=linear_solver_provider,
        )
    return MappingProxyType({
        "elapsed_time_s": earliest,
        "event": tie[0] if len(tie) == 1 else "EXPLICIT_TIE_GROUP",
        "transport_state": accepted,
        "tie_group": tie,
        "all_losing_clocks_advance_s": earliest,
    })


def vacancy_transport_refinement_indicators(
    mesh: Any, registry: TransportSurfaceRegistry, *, vacancy_fraction_gp: np.ndarray,
    plastic_source_rate_s: np.ndarray, climb_exchange_rate_s: np.ndarray,
    duration_s: float, diffusivity_m2_s: float,
    site_states: Sequence[LocalizedCaptureSite] = (),
) -> Mapping[str, Any]:
    """Element indicators used in addition to the frozen mechanics criteria."""
    c = np.asarray(vacancy_fraction_gp, dtype=float)
    source = np.asarray(plastic_source_rate_s, dtype=float)
    climb = np.asarray(climb_exchange_rate_s, dtype=float)
    if any(value.shape != (mesh.ne,) for value in (c, source, climb)):
        raise ValueError("transport indicators require one value per element")
    centroids = np.asarray(mesh.nodes)[np.asarray(mesh.elems)].mean(axis=1)
    gradient = np.zeros(mesh.ne)
    residual = np.abs(source-climb)
    distance_to_sink = np.full(mesh.ne, math.inf)
    for face in registry.faces:
        if face.kind == SurfaceKind.INTACT_INTERIOR_INTERFACE:
            i, j = face.owner_elements
            distance = math.fsum(face.owner_distances_m)
            jump = abs(c[i]-c[j]) / distance
            gradient[i] = max(gradient[i], jump)
            gradient[j] = max(gradient[j], jump)
        elif face.kind in PERFECT_SINK_KINDS:
            point = np.asarray(face.center_xy_m)
            distance_to_sink = np.minimum(distance_to_sink, np.linalg.norm(centroids-point, axis=1))
    distance_to_site = np.full(mesh.ne, math.inf)
    site_support = math.inf
    for site in site_states:
        distance_to_site = np.minimum(
            distance_to_site,
            np.linalg.norm(centroids-np.asarray(site.center_xy_m), axis=1),
        )
        site_support = min(site_support, site.support_radius_m)
    diffusion_length = math.sqrt(max(float(diffusivity_m2_s)*float(duration_s), 0.0))
    local_size = np.sqrt(2.0*np.asarray(mesh.area_e))
    controlling = np.minimum(distance_to_sink, distance_to_site)
    if math.isfinite(site_support):
        controlling = np.minimum(controlling, site_support)
    controlling = np.minimum(controlling, diffusion_length)
    marked = np.flatnonzero(
        (local_size > np.maximum(controlling, 1.0e-30))
        | (gradient*local_size > 0.05*np.maximum(c, DEFAULT_C_MIN))
        | (residual*float(duration_s) > 0.05*np.maximum(c, DEFAULT_C_MIN))
    )
    return MappingProxyType({
        "vacancy_gradient_indicator": gradient,
        "transport_residual_indicator": residual,
        "plastic_source_indicator": np.abs(source),
        "climb_sink_indicator": np.abs(climb),
        "distance_to_nearest_sink_m": distance_to_sink,
        "distance_to_nearest_site_m": distance_to_site,
        "diffusion_length_m": diffusion_length,
        "marked_element_ids": tuple(map(int, marked)),
    })


def transfer_transport_state_overlap(
    previous: VacancyTransportStateV2, old_mesh: Any, new_mesh: Any,
    new_registry: TransportSurfaceRegistry, *, capture_relative_tolerance: float = 5.0e-3,
    domain_change_policy: str = "SAME_SOLID_DOMAIN",
    new_parent_mechanics_element_ids: np.ndarray | None = None, failure_injector=None,
) -> tuple[VacancyTransportStateV2, Mapping[str, Any]]:
    """Overlap-conservative P0 transfer with fail-closed local-site audits."""
    from .voiding_production_v5 import _triangle_overlap_area
    old_triangles = np.asarray(old_mesh.nodes)[np.asarray(old_mesh.elems)]
    if failure_injector is not None:
        failure_injector("remesh_transfer", previous)
    new_triangles = np.asarray(new_mesh.nodes)[np.asarray(new_mesh.elems)]
    old_area = np.asarray(old_mesh.area_e, dtype=float)
    new_area = np.asarray(new_mesh.area_e, dtype=float)
    old_c = np.asarray(previous.vacancy_fraction_gp, dtype=float)
    new_c = np.zeros(new_mesh.ne)
    old_lo = old_triangles.min(axis=1); old_hi = old_triangles.max(axis=1)
    old_centers = old_triangles.mean(axis=1)
    old_radii = np.max(np.linalg.norm(
        old_triangles-old_centers[:, None, :], axis=2), axis=1)
    old_tree = cKDTree(old_centers)
    maximum_old_radius = float(np.max(old_radii))
    old_overlap_coverage = np.zeros(old_mesh.ne, dtype=float)
    new_overlap_coverage = np.zeros(new_mesh.ne, dtype=float)
    for new_id, new_triangle in enumerate(new_triangles):
        lo = new_triangle.min(axis=0); hi = new_triangle.max(axis=0)
        center = new_triangle.mean(axis=0)
        radius = float(np.max(np.linalg.norm(new_triangle-center, axis=1)))
        candidates = np.asarray(old_tree.query_ball_point(
            center, radius+maximum_old_radius+1.0e-15), dtype=int)
        if len(candidates):
            candidates = candidates[
                np.all(old_hi[candidates] >= lo, axis=1)
                & np.all(old_lo[candidates] <= hi, axis=1)]
        overlap_integral = 0.0
        overlap_coverage = 0.0
        for old_id in candidates:
            overlap = _triangle_overlap_area(old_triangles[old_id], new_triangle)
            if overlap > 0.0:
                overlap_coverage += overlap
                overlap_integral += overlap*old_c[old_id]
                old_overlap_coverage[old_id] += overlap
        if overlap_coverage <= 0.0:
            raise RuntimeError("transport overlap transfer found an uncovered target element")
        # Clipped polygons meeting on old edges can over/under-cover by a few
        # ulps.  Normalize the local partition before forming the P0 average;
        # this preserves constants without a whole-domain concentration scale.
        new_c[new_id] = overlap_integral/overlap_coverage
        new_overlap_coverage[new_id] = overlap_coverage
    old_inventory = math.fsum(float(a*c) for a, c in zip(old_area, old_c))
    new_inventory = math.fsum(float(a*c) for a, c in zip(new_area, new_c))
    residual = old_inventory-new_inventory
    budget = conservation_roundoff_budget_m2(old_inventory, new_inventory)
    # Bound clipping error from the actual area operands.  Coverage defects
    # must first be consistent with the finite sequence of polygon operations;
    # only then may their concentration-weighted effect enter the conservative
    # correction budget.  A large overlap defect cannot authorize itself.
    domain_area = max(float(math.fsum(old_area)), float(math.fsum(new_area)))
    coverage_roundoff_budget = max(
        64.0*np.spacing(max(domain_area, np.finfo(float).tiny)),
        2048.0*np.finfo(float).eps*domain_area*(old_mesh.ne+new_mesh.ne),
    )
    old_coverage_defect = float(math.fsum(abs(old_area-old_overlap_coverage)))
    new_coverage_defect = float(math.fsum(abs(new_area-new_overlap_coverage)))
    total_coverage_defect = old_coverage_defect+new_coverage_defect
    old_coverage_excess = float(math.fsum(
        max(float(covered-area), 0.0)
        for area, covered in zip(old_area, old_overlap_coverage)))
    accepted_coverage_defect = (
        total_coverage_defect
        if domain_change_policy == "SAME_SOLID_DOMAIN"
        else new_coverage_defect+old_coverage_excess)
    if accepted_coverage_defect > coverage_roundoff_budget:
        raise RuntimeError("overlap geometry coverage exceeds operand-aware budget: "+repr({
            "coverage_defect_area_m2": accepted_coverage_defect,
            "coverage_roundoff_budget_area_m2": coverage_roundoff_budget,
            "old_coverage_defect_area_m2": old_coverage_defect,
            "old_coverage_excess_area_m2": old_coverage_excess,
            "new_coverage_defect_area_m2": new_coverage_defect,
        }))
    old_inventory_coverage_budget = math.fsum((
        float(abs(c)*(abs(a-covered) if domain_change_policy == "SAME_SOLID_DOMAIN"
                      else max(covered-a, 0.0)))
        for c, a, covered in zip(old_c, old_area, old_overlap_coverage)
    ))
    coverage_inventory_budget = old_inventory_coverage_budget+math.fsum((
        float(abs(c)*abs(a-covered))
        for c, a, covered in zip(new_c, new_area, new_overlap_coverage)
    ))
    transfer_budget = budget+coverage_inventory_budget
    removed_solid = 0.0
    correction_target = None
    correction_concentration = 0.0
    ledgers = dict(previous.ledgers)
    if domain_change_policy == "ACTIVE_CAVITY_ABSORPTION_ONE_WAY":
        if residual < -budget:
            raise RuntimeError("one-way cavity policy forbids domain growth/shrinkage")
        removed_solid = max(residual, 0.0)
        inventory = previous.vacancy_formation_volume_m3/previous.atomic_volume_m3*removed_solid
        ledgers["active_cavity_absorption_area_m2"] += inventory
        ledgers["active_cavity_inventory_area_m2"] += inventory
        residual = 0.0
    elif domain_change_policy != "SAME_SOLID_DOMAIN":
        raise ValueError("unknown vacancy domain-change policy")
    elif abs(residual) > transfer_budget:
        raise RuntimeError("overlap P0 transport transfer changed global inventory: "
            +repr({"residual_area_m2": residual,
                   "roundoff_budget_m2": budget,
                   "geometric_transfer_budget_m2": transfer_budget,
                   "old_inventory_area_m2": old_inventory,
                   "new_inventory_area_m2": new_inventory,
                   "old_elements": int(old_mesh.ne),
                   "new_elements": int(new_mesh.ne)}))
    if residual and domain_change_policy == "SAME_SOLID_DOMAIN":
        target = int(np.argmax(new_area))
        correction = residual/new_area[target]
        if abs(residual) > transfer_budget:
            raise RuntimeError("transport remesh correction exceeds last-bit budget")
        new_c[target] += correction
        correction_target = target
        correction_concentration = float(correction)
    local_audits = []
    for site in previous.site_states:
        old_w = _capture_weights(old_mesh, site)
        new_w = _capture_weights(new_mesh, site)
        old_local = math.fsum(float(a*w*c) for a, w, c in zip(old_area, old_w, old_c))
        new_local = math.fsum(float(a*w*c) for a, w, c in zip(new_area, new_w, new_c))
        relative = abs(new_local-old_local)/max(abs(old_local), DEFAULT_C_MIN)
        if relative > capture_relative_tolerance:
            raise RuntimeError("transport remesh changed active capture-region inventory")
        local_audits.append({"site_id": site.site_id, "relative_error": relative})
    transferred = replace(
        previous, vacancy_fraction_gp=new_c,
        surface_registry_fingerprint=new_registry.fingerprint,
        surface_registry=new_registry, ledgers=ledgers,
        transport_mesh=new_mesh,
        parent_mechanics_element_ids=(
            np.arange(new_mesh.ne, dtype=int) if new_parent_mechanics_element_ids is None
            else np.asarray(new_parent_mechanics_element_ids, dtype=int)
        ),
        last_solver_audit={},
    )
    return transferred, MappingProxyType({
        "method": "ELEMENT_OVERLAP_CONSERVATIVE_P0_NO_GLOBAL_RESCALING",
        "source_inventory_fraction_area": old_inventory,
        "target_inventory_fraction_area": math.fsum(float(a*c) for a, c in zip(new_area, new_c)),
        "roundoff_correction_fraction_area": residual,
        "roundoff_budget_fraction_area": budget,
        "coverage_roundoff_budget_area_m2": coverage_roundoff_budget,
        "coverage_defect_area_m2": accepted_coverage_defect,
        "physical_old_domain_nonoverlap_area_m2": (
            old_coverage_defect if
            domain_change_policy == "ACTIVE_CAVITY_ABSORPTION_ONE_WAY" else 0.0),
        "coverage_inventory_budget_fraction_area": coverage_inventory_budget,
        "transfer_acceptance_budget_fraction_area": transfer_budget,
        "correction_target_element_id": correction_target,
        "correction_concentration_fraction": correction_concentration,
        "removed_solid_fraction_area": removed_solid,
        "domain_change_policy": domain_change_policy,
        "active_capture_region_audits": tuple(local_audits),
        "flux_transferred": False,
        "surface_registry_rebuilt": True,
    })


def _registry_payload(registry: TransportSurfaceRegistry) -> dict[str, Any]:
    return {
        "schema": registry.schema,
        "fingerprint": registry.fingerprint,
        "crack_physical_length_m": registry.crack_physical_length_m,
        "blocked_interior_edges": [list(edge) for edge in registry.blocked_interior_edges],
        "faces": [{**asdict(face), "kind": face.kind.value} for face in registry.faces],
    }


def _state_payload(state: VacancyTransportStateV2) -> dict[str, Any]:
    return {
        "vacancy_fraction_gp": state.vacancy_fraction_gp.tolist(),
        "vacancy_temperature_K": state.vacancy_temperature_K,
        "parameter_row_id": state.parameter_row_id,
        "parameter_row_fingerprint": state.parameter_row_fingerprint,
        "parameter_row": (None if state.parameter_row is None else asdict(state.parameter_row)),
        "atomic_volume_m3": state.atomic_volume_m3,
        "vacancy_formation_volume_m3": state.vacancy_formation_volume_m3,
        "surface_registry_fingerprint": state.surface_registry_fingerprint,
        "surface_registry": (None if state.surface_registry is None
                             else _registry_payload(state.surface_registry)),
        "transport_mesh": (None if state.transport_mesh is None else {
            "nodes": np.asarray(state.transport_mesh.nodes).tolist(),
            "elems": np.asarray(state.transport_mesh.elems).tolist(),
        }),
        "parent_mechanics_element_ids": (
            None if state.parent_mechanics_element_ids is None
            else state.parent_mechanics_element_ids.tolist()
        ),
        "transport_time_s": state.transport_time_s,
        "ledgers": dict(state.ledgers),
        "site_states": [asdict(site) for site in state.site_states],
        "last_solver_audit": dict(state.last_solver_audit),
        "schema": state.schema,
    }


def _registry_from_payload(registry_data: Mapping[str, Any]) -> TransportSurfaceRegistry:
    faces = tuple(TransportFace(
        **{**face, "kind": SurfaceKind(face["kind"]),
           "owner_elements": tuple(face["owner_elements"]),
           "node_ids": None if face["node_ids"] is None else tuple(face["node_ids"]),
           "center_xy_m": tuple(face["center_xy_m"]),
           "owner_distances_m": tuple(face["owner_distances_m"])}
    ) for face in registry_data["faces"])
    return TransportSurfaceRegistry(
        faces=faces,
        blocked_interior_edges=tuple(tuple(edge) for edge in registry_data["blocked_interior_edges"]),
        crack_physical_length_m=registry_data["crack_physical_length_m"],
        schema=registry_data["schema"],
    )


def _state_from_payload(state_data: Mapping[str, Any]) -> VacancyTransportStateV2:
    values = dict(state_data)
    registry_data = values.get("surface_registry")
    values["surface_registry"] = (
        None if registry_data is None else _registry_from_payload(registry_data)
    )
    values["vacancy_fraction_gp"] = np.asarray(values["vacancy_fraction_gp"])
    if values.get("parameter_row") is not None:
        values["parameter_row"] = VacancyTransportParameterRowV2(**values["parameter_row"])
    mesh_data = values.get("transport_mesh")
    values["transport_mesh"] = (None if mesh_data is None else rebuild_tri_mesh(
        np.asarray(mesh_data["nodes"], float), np.asarray(mesh_data["elems"], int),
    ))
    if values.get("parent_mechanics_element_ids") is not None:
        values["parent_mechanics_element_ids"] = np.asarray(values["parent_mechanics_element_ids"], int)
    values["site_states"] = tuple(LocalizedCaptureSite(**site) for site in values["site_states"])
    return VacancyTransportStateV2(**values)


def write_transport_checkpoint(
    state: VacancyTransportStateV2, registry: TransportSurfaceRegistry, path: str | Any,
    *, failure_injector=None,
) -> Mapping[str, Any]:
    """Atomically publish a complete accepted V2 transport boundary."""
    import os
    import pickle
    from pathlib import Path
    if state.surface_registry_fingerprint != registry.fingerprint:
        raise RuntimeError("cannot checkpoint stale transport-surface ownership")
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = pickle.dumps({"state": _state_payload(state),
                            "registry": _registry_payload(registry)}, protocol=5)
    digest = hashlib.sha256(payload).hexdigest()
    payload_name = target.name+f".{digest}.state.pkl"
    manifest = {
        "schema": SCHEMA,
        "state_file": payload_name,
        "state_sha256": digest,
        "transport_time_s": state.transport_time_s,
        "parameter_row_id": state.parameter_row_id,
        "parameter_row_fingerprint": state.parameter_row_fingerprint,
        "vacancy_fraction_gp": state.vacancy_fraction_gp.tolist(),
        "vacancy_temperature_K": state.vacancy_temperature_K,
        "atomic_volume_m3": state.atomic_volume_m3,
        "vacancy_formation_volume_m3": state.vacancy_formation_volume_m3,
        "ledgers": dict(state.ledgers),
        "site_states": [asdict(site) for site in state.site_states],
        "surface_registry": _registry_payload(registry),
    }
    payload_path = target.with_name(payload_name)
    payload_tmp = payload_path.with_name(payload_path.name+".tmp")
    manifest_tmp = target.with_name(target.name+".tmp")
    payload_tmp.write_bytes(payload)
    manifest_tmp.write_text(json.dumps(manifest, sort_keys=True, indent=2, allow_nan=False)+"\n")
    if failure_injector is not None:
        failure_injector("checkpoint_publication", state)
    os.replace(payload_tmp, payload_path)
    os.replace(manifest_tmp, target)
    return MappingProxyType(manifest)


def restore_transport_checkpoint(path: str | Any) -> tuple[VacancyTransportStateV2, TransportSurfaceRegistry]:
    import pickle
    from pathlib import Path
    target = Path(path)
    manifest = json.loads(target.read_text())
    if manifest.get("schema") != SCHEMA:
        raise ValueError("unsupported vacancy transport checkpoint schema")
    payload = target.with_name(manifest["state_file"]).read_bytes()
    if hashlib.sha256(payload).hexdigest() != manifest.get("state_sha256"):
        raise ValueError("vacancy transport checkpoint payload hash mismatch")
    restored = pickle.loads(payload)
    state_data, registry_data = restored["state"], restored["registry"]
    registry = _registry_from_payload(registry_data)
    state = _state_from_payload(state_data)
    if state.surface_registry_fingerprint != registry.fingerprint:
        raise ValueError("vacancy transport checkpoint registry identity mismatch")
    canonical_registry = json.loads(json.dumps(
        _registry_payload(registry), sort_keys=True, allow_nan=False,
    ))
    if canonical_registry != manifest.get("surface_registry"):
        raise ValueError("vacancy transport checkpoint surface registry mismatch")
    if state.vacancy_fraction_gp.tolist() != manifest.get("vacancy_fraction_gp"):
        raise ValueError("vacancy transport checkpoint field mismatch")
    return state, registry


__all__ = [
    "DEFAULT_C_MAX", "DEFAULT_C_MIN", "KAPPA_V", "LOCAL_REACTION_V1_LEGACY",
    "ExactCutTransportMesh", "LocalizedCaptureSite", "MODEL_ID", "PERFECT_SINK_KINDS", "REFERENCE_UNCALIBRATED",
    "SCHEMA", "SurfaceKind", "TRANSPORT_LEDGER_KEYS", "TransportFace",
    "TransportSurfaceRegistry", "VacancyTransportBoundsError",
    "VacancyTransportParameterRowV2", "VacancyTransportStateV2",
    "VACANCY_ENERGY_STUDY_SPECS", "vacancy_energy_study_row",
    "atomic_volume_from_burgers_vector_m", "build_transport_surface_registry",
    "build_registry_from_accepted_topology",
    "build_exact_cut_transport_mesh", "build_exact_cut_transport_mesh_for_network",
    "accepted_elastoplastic_stress_3d", "accepted_plane_strain_mean_stress_Pa",
    "cavity_growth_rate_from_transport",
    "compete_on_common_transport_clock",
    "chemical_potential_J", "dilute_flux_1d", "equilibrium_vacancy_fraction",
    "initial_transport_state", "rebind_registry_after_accepted_topology",
    "heal_embryo_reservation", "promote_subgrid_inventory", "stabilize_reserved_embryo",
    "transfer_reserved_embryo_to_growth", "stabilize_growth_owned_inventory",
    "heal_growth_owned_inventory",
    "advance_transport_to_time", "commit_site_inventory_hit",
    "locate_site_inventory_event", "preview_site_inventory_event",
    "restore_transport_checkpoint", "solve_transport_interval",
    "thermodynamic_flux_1d", "transfer_transport_state_overlap", "transport_conservation",
    "vacancy_transport_refinement_indicators", "write_transport_checkpoint",
]
