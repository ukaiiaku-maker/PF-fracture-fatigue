"""Stateful explicit-cavity extension for the sharp-front fracture model.

V1 is deliberately disabled unless ``VoidingConfig.enabled`` is true.  A
resolved void is represented by removing solid triangles and retaining a closed
traction-free internal boundary; it is never represented by damage or a soft
material.  Kinetic parameters in this module are diagnostic qualification
inputs, not calibrated material data.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy.spatial import Delaunay

from .config import KB
from .mesh import TriMesh, rebuild_tri_mesh

SCHEMA_VERSION = "voiding-v1.0"
CALIBRATION_STATUS = "DIAGNOSTIC_IMPLEMENTATION_QUALIFICATION_ONLY"


class SiteStatus(str, Enum):
    AVAILABLE_SITE = "AVAILABLE_SITE"
    EMBRYO = "EMBRYO"
    HEALED_SITE = "HEALED_SITE"
    CONSUMED_SITE = "CONSUMED_SITE"


class CavityStatus(str, Enum):
    STABLE_SUBGRID_VOID = "STABLE_SUBGRID_VOID"
    RESOLVED_VOID = "RESOLVED_VOID"
    CONNECTED_VOID = "CONNECTED_VOID"
    DOWNSTREAM_FRONT_ACTIVE = "DOWNSTREAM_FRONT_ACTIVE"
    MERGED_OR_CONSUMED = "MERGED_OR_CONSUMED"


class SiteClass(str, Enum):
    PRESCRIBED_TEST_SITE = "PRESCRIBED_TEST_SITE"
    INTRAGRANULAR_DISLOCATION_STRUCTURE_SITE = "INTRAGRANULAR_DISLOCATION_STRUCTURE_SITE"
    GRAIN_BOUNDARY_SITE = "GRAIN_BOUNDARY_SITE"
    PARTICLE_SITE = "PARTICLE_SITE"
    TRIPLE_JUNCTION_SITE = "TRIPLE_JUNCTION_SITE"


@dataclass(frozen=True)
class GeometryConvention:
    bulk_constraint: str = "PLANE_STRAIN"
    out_of_plane_thickness_m: float = 1.0
    cavity_measure: str = "2D_AREA_PER_UNIT_THICKNESS"
    capillary_pressure_law: str = "gamma_over_R_cylindrical_through_thickness"


@dataclass(frozen=True)
class VoidingConfig:
    enabled: bool = False
    schema_version: str = SCHEMA_VERSION
    calibration_status: str = CALIBRATION_STATUS
    not_calibrated: bool = True
    geometry: GeometryConvention = field(default_factory=GeometryConvention)
    promotion_min_boundary_segments: int = 24
    promotion_max_h_over_R: float = 0.35
    promotion_min_ligament_layers: float = 4.0
    local_defect_provider: str = "SITE_LOCAL_REPLACEABLE_V1"


@dataclass
class FirstPassageState:
    accumulated_hazard: float = 0.0
    threshold: float = math.inf
    rng_state: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def seeded(cls, seed: int) -> "FirstPassageState":
        rng = np.random.default_rng(seed)
        threshold = float(-math.log(max(float(rng.random()), np.finfo(float).tiny)))
        return cls(0.0, threshold, copy.deepcopy(rng.bit_generator.state))


@dataclass
class VoidSite:
    void_or_site_id: str
    site_class: SiteClass
    position_m: Tuple[float, float]
    status: SiteStatus = SiteStatus.AVAILABLE_SITE
    statistical_weight_birth_only: float = 1.0
    required_hits: int = 1
    completion_lambda: float = 0.0
    birth: FirstPassageState = field(default_factory=FirstPassageState)
    stabilization: FirstPassageState = field(default_factory=FirstPassageState)
    healing: FirstPassageState = field(default_factory=FirstPassageState)
    defect_inventory: float = 0.0
    creation_transaction: int = 0
    last_accepted_transaction: int = 0
    material_configuration_hash: str = ""
    source_commit: str = ""
    schema_version: str = SCHEMA_VERSION


@dataclass
class Cavity:
    void_or_site_id: str
    parent_site_id: str
    site_class: SiteClass
    center_m: Tuple[float, float]
    status: CavityStatus = CavityStatus.STABLE_SUBGRID_VOID
    radius_m: float = 0.0
    area_per_unit_thickness_m2: float = 0.0
    perimeter_m: float = 0.0
    defect_inventory: float = 0.0
    shape_representation: str = "SUBGRID_CIRCLE"
    orientation_rad: float = 0.0
    surface_boundary_node_ids: List[int] = field(default_factory=list)
    connected_crack_component_id: Optional[str] = None
    downstream_candidate_state: Optional[Dict[str, Any]] = None
    creation_transaction: int = 0
    last_accepted_transaction: int = 0
    material_configuration_hash: str = ""
    source_commit: str = ""
    schema_version: str = SCHEMA_VERSION

    def set_radius(self, radius_m: float) -> None:
        if radius_m < 0:
            raise ValueError("void radius must be non-negative")
        self.radius_m = float(radius_m)
        self.area_per_unit_thickness_m2 = math.pi * self.radius_m**2
        self.perimeter_m = 2.0 * math.pi * self.radius_m


@dataclass
class LengthLedger:
    fractured_ligament_increment: float = 0.0
    free_void_span_increment: float = 0.0
    active_front_coordinate_increment: float = 0.0
    projected_fractured_length: float = 0.0
    projected_free_span: float = 0.0
    projected_front_advance: float = 0.0
    total_connected_free_surface_extent: float = 0.0


@dataclass
class VoidRegistry:
    config: VoidingConfig = field(default_factory=VoidingConfig)
    sites: Dict[str, VoidSite] = field(default_factory=dict)
    cavities: Dict[str, Cavity] = field(default_factory=dict)
    ledger: LengthLedger = field(default_factory=LengthLedger)
    transaction_id: int = 0
    event_history: List[Dict[str, Any]] = field(default_factory=list)

    def instantiate_site(self, site: VoidSite) -> None:
        if not self.config.enabled:
            raise RuntimeError("voiding disabled: no site may be instantiated")
        if site.void_or_site_id in self.sites:
            raise ValueError("duplicate site id")
        self.sites[site.void_or_site_id] = site

    def to_dict(self) -> Dict[str, Any]:
        def encode(x: Any) -> Any:
            if isinstance(x, Enum):
                return x.value
            if hasattr(x, "__dataclass_fields__"):
                return {k: encode(v) for k, v in asdict(x).items()}
            if isinstance(x, dict):
                return {str(k): encode(v) for k, v in x.items()}
            if isinstance(x, (list, tuple)):
                return [encode(v) for v in x]
            if isinstance(x, np.generic):
                return x.item()
            return x
        return encode(self)

    def fingerprint(self) -> str:
        raw = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(raw.encode()).hexdigest()


class VoidTransaction:
    """Copy-on-trial transaction restoring geometry, state, clocks and RNG."""
    def __init__(self, registry: VoidRegistry, geometry_owner: Optional[Any] = None):
        self.registry = registry
        self.geometry_owner = geometry_owner
        self._state = copy.deepcopy(registry)
        self._geometry = copy.deepcopy(geometry_owner)
        self.committed = False

    def commit(self) -> None:
        self.registry.transaction_id += 1
        self.committed = True

    def rollback(self) -> None:
        self.registry.__dict__.clear()
        self.registry.__dict__.update(copy.deepcopy(self._state.__dict__))
        if self.geometry_owner is not None:
            self.geometry_owner.__dict__.clear()
            self.geometry_owner.__dict__.update(copy.deepcopy(self._geometry.__dict__))

    def __enter__(self) -> "VoidTransaction":
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        if exc_type is not None or not self.committed:
            self.rollback()
        return False


def poisson_completion_probability(required_hits: int, completion_lambda: float) -> float:
    """P[N>=K] for a Poisson completion clock, evaluated stably for small K."""
    if required_hits < 1 or completion_lambda < 0:
        raise ValueError("required_hits >= 1 and completion_lambda >= 0")
    term = 1.0
    partial = term
    for j in range(1, required_hits):
        term *= completion_lambda / j
        partial += term
    return float(np.clip(1.0 - math.exp(-completion_lambda) * partial, 0.0, 1.0))


def tensorial_activation_work(stress_xy: np.ndarray, normal: Sequence[float],
                               tangent: Sequence[float], dilatation_volume_m3: float,
                               opening_volume_m3: float, shear_volume_m3: float) -> float:
    """Hydrostatic, normal-opening and signed shear activation work."""
    s = np.asarray(stress_xy, float).reshape(2, 2)
    n = np.asarray(normal, float); n /= np.linalg.norm(n)
    t = np.asarray(tangent, float); t /= np.linalg.norm(t)
    hydro = float(np.trace(s) / 3.0)
    normal_stress = float(n @ s @ n)
    shear = float(t @ s @ n)
    return (max(hydro, 0.0) * dilatation_volume_m3
            + max(normal_stress, 0.0) * opening_volume_m3
            + shear * shear_volume_m3)


def birth_intensity(site: VoidSite, temperature_K: float, attempt_rate_s: float,
                    barrier_J: float, activation_work_J: float) -> float:
    """Birth-only population intensity; site weight is forbidden elsewhere."""
    completion = poisson_completion_probability(site.required_hits, site.completion_lambda)
    exponent = np.clip(-(barrier_J - activation_work_J) / (KB * temperature_K), -745.0, 700.0)
    return float(site.statistical_weight_birth_only * attempt_rate_s * math.exp(exponent) * completion)


def series_limited_growth_rate(diffusion_rate_m_s: float, plastic_rate_m_s: float) -> float:
    if diffusion_rate_m_s <= 0.0 or plastic_rate_m_s <= 0.0:
        return min(diffusion_rate_m_s, plastic_rate_m_s)
    return 1.0 / (1.0 / diffusion_rate_m_s + 1.0 / plastic_rate_m_s)


def advance_site_lifecycle(registry: VoidRegistry, site_id: str, dt_s: float,
                           birth_rate_s: float, stabilization_rate_s: float,
                           healing_rate_s: float, initial_radius_m: float) -> Optional[Cavity]:
    """Advance distinct birth/stabilization/healing first passages transactionally."""
    if not registry.config.enabled:
        return None
    site = registry.sites[site_id]
    if site.status == SiteStatus.AVAILABLE_SITE:
        site.birth.accumulated_hazard += max(birth_rate_s, 0.0) * dt_s
        if site.birth.accumulated_hazard >= site.birth.threshold:
            site.status = SiteStatus.EMBRYO
            registry.event_history.append({"event": "EMBRYO_BIRTH", "site_id": site_id})
    if site.status == SiteStatus.EMBRYO:
        site.stabilization.accumulated_hazard += max(stabilization_rate_s, 0.0) * dt_s
        site.healing.accumulated_hazard += max(healing_rate_s, 0.0) * dt_s
        stable_fraction = site.stabilization.accumulated_hazard / max(site.stabilization.threshold, 1e-300)
        heal_fraction = site.healing.accumulated_hazard / max(site.healing.threshold, 1e-300)
        if max(stable_fraction, heal_fraction) >= 1.0:
            if heal_fraction > stable_fraction:
                site.status = SiteStatus.HEALED_SITE
                registry.event_history.append({"event": "EMBRYO_HEALING", "site_id": site_id})
                return None
            cavity_id = f"void:{site_id}"
            if cavity_id in registry.cavities:
                raise RuntimeError("stable void birth attempted more than once")
            cavity = Cavity(cavity_id, site_id, site.site_class, site.position_m,
                            defect_inventory=site.defect_inventory)
            cavity.set_radius(initial_radius_m)
            registry.cavities[cavity_id] = cavity
            site.status = SiteStatus.CONSUMED_SITE
            registry.event_history.append({"event": "EMBRYO_STABILIZATION", "site_id": site_id,
                                           "void_id": cavity_id})
            return cavity
    return None


@dataclass
class ExplicitHoleMesh:
    mesh: TriMesh
    cavity_boundary_nodes: np.ndarray
    cavity_boundary_edges: np.ndarray
    center_m: Tuple[float, float]
    radius_m: float
    area_m2: float
    perimeter_m: float
    topology_fingerprint: str
    minimum_quality: float
    maximum_aspect_ratio: float


def _triangle_quality(x: np.ndarray) -> Tuple[float, float]:
    lengths = np.linalg.norm(x[:, [1, 2, 0], :] - x[:, [0, 1, 2], :], axis=2)
    a = x[:, 1] - x[:, 0]
    b = x[:, 2] - x[:, 0]
    area = np.abs(a[:, 0]*b[:, 1] - a[:, 1]*b[:, 0]) / 2.0
    quality = 4.0 * math.sqrt(3.0) * area / np.maximum(np.sum(lengths**2, axis=1), 1e-300)
    aspect = np.max(lengths, axis=1) / np.maximum(np.min(lengths, axis=1), 1e-300)
    return float(np.min(quality)), float(np.max(aspect))


def make_explicit_circular_hole_mesh(width_m: float, height_m: float,
                                     center_m: Tuple[float, float], radius_m: float,
                                     target_h_m: float, boundary_segments: int = 48) -> ExplicitHoleMesh:
    """Delaunay plate mesh with a polygonal, closed, traction-free circular hole."""
    if radius_m <= 0 or target_h_m <= 0 or boundary_segments < 12:
        raise ValueError("invalid explicit-hole resolution")
    cx, cy = center_m
    if not (radius_m < cx < width_m-radius_m and -height_m/2+radius_m < cy < height_m/2-radius_m):
        raise ValueError("hole must be strictly internal")
    xs = np.arange(0.0, width_m + 0.5*target_h_m, target_h_m)
    ys = np.arange(-height_m/2, height_m/2 + 0.5*target_h_m, target_h_m)
    gx, gy = np.meshgrid(xs, ys)
    background = np.c_[gx.ravel(), gy.ravel()]
    radial = np.hypot(background[:, 0]-cx, background[:, 1]-cy)
    background = background[radial > radius_m + 0.35*target_h_m]
    theta = 2*math.pi*np.arange(boundary_segments)/boundary_segments
    ring = np.c_[cx + radius_m*np.cos(theta), cy + radius_m*np.sin(theta)]
    outer = np.array([[0, -height_m/2], [0, height_m/2],
                      [width_m, -height_m/2], [width_m, height_m/2]], float)
    nodes = np.vstack([ring, background, outer])
    key = np.round(nodes / max(target_h_m*1e-7, 1e-15)).astype(np.int64)
    _, idx = np.unique(key, axis=0, return_index=True)
    nodes = nodes[np.sort(idx)]
    elems = Delaunay(nodes).simplices
    tri_x = nodes[elems]
    cent = tri_x.mean(axis=1)
    vertices_outside = np.all(np.hypot(tri_x[:, :, 0]-cx, tri_x[:, :, 1]-cy) >= radius_m*(1-1e-10), axis=1)
    centroid_outside = np.hypot(cent[:, 0]-cx, cent[:, 1]-cy) >= radius_m
    in_plate = ((cent[:, 0] >= 0) & (cent[:, 0] <= width_m) &
                (cent[:, 1] >= -height_m/2) & (cent[:, 1] <= height_m/2))
    elems = elems[vertices_outside & centroid_outside & in_plate]
    mesh = rebuild_tri_mesh(nodes, elems, tip_centers=np.array(center_m))
    radii = np.hypot(nodes[:, 0]-cx, nodes[:, 1]-cy)
    boundary = np.where(np.isclose(radii, radius_m, rtol=1e-8, atol=target_h_m*1e-7))[0]
    angles = np.arctan2(nodes[boundary, 1]-cy, nodes[boundary, 0]-cx)
    boundary = boundary[np.argsort(angles)]
    edges = np.c_[boundary, np.roll(boundary, -1)]
    polygon = nodes[boundary]
    area = 0.5*abs(float(np.sum(polygon[:, 0]*np.roll(polygon[:, 1], -1)
                                 - polygon[:, 1]*np.roll(polygon[:, 0], -1))))
    perimeter = float(np.sum(np.linalg.norm(polygon-np.roll(polygon, -1, axis=0), axis=1)))
    quality, aspect = _triangle_quality(nodes[elems])
    fp = hashlib.sha256(nodes.tobytes()+elems.tobytes()+edges.tobytes()).hexdigest()
    return ExplicitHoleMesh(mesh, boundary, edges, center_m, radius_m, area,
                            perimeter, fp, quality, aspect)


def validate_explicit_hole(hole: ExplicitHoleMesh, wake_element_mask: Optional[np.ndarray] = None) -> Dict[str, Any]:
    cent = hole.mesh.nodes[hole.mesh.elems].mean(axis=1)
    dist = np.hypot(cent[:, 0]-hole.center_m[0], cent[:, 1]-hole.center_m[1])
    edge_degree: Dict[int, int] = {}
    for a, b in hole.cavity_boundary_edges:
        edge_degree[int(a)] = edge_degree.get(int(a), 0)+1
        edge_degree[int(b)] = edge_degree.get(int(b), 0)+1
    no_inside = bool(np.all(dist >= hole.radius_m*(1-1e-10)))
    closed = len(edge_degree) >= 12 and all(v == 2 for v in edge_degree.values())
    wake_overlap = False
    if wake_element_mask is not None:
        mask = np.asarray(wake_element_mask, bool)
        if len(mask) != hole.mesh.ne:
            raise ValueError("wake mask must align with retained solid elements")
        wake_overlap = bool(np.any(mask & (dist < hole.radius_m)))
    return {"no_triangles_inside": no_inside, "closed_boundary_components": int(closed),
            "no_wake_overlap": not wake_overlap, "traction_boundary_condition": "NATURAL_ZERO_TRACTION",
            "true_hole_no_void_material": True}


def ray_circle_first_intersection(origin: Sequence[float], direction: Sequence[float],
                                  center: Sequence[float], radius: float) -> Optional[np.ndarray]:
    o = np.asarray(origin, float); d = np.asarray(direction, float); d /= np.linalg.norm(d)
    c = np.asarray(center, float); q = o-c
    disc = float((q@d)**2 - (q@q-radius**2))
    if disc < 0:
        return None
    roots = [-(q@d)-math.sqrt(disc), -(q@d)+math.sqrt(disc)]
    positive = [v for v in roots if v > 1e-14]
    return None if not positive else o + min(positive)*d


def crack_to_void_ligament_candidate(tip: Sequence[float], direction: Sequence[float],
                                     cavity: Cavity, existing_barrier_id: str) -> Optional[Dict[str, Any]]:
    hit = ray_circle_first_intersection(tip, direction, cavity.center_m, cavity.radius_m)
    if hit is None:
        return None
    ligament = float(np.linalg.norm(hit-np.asarray(tip, float)))
    return {"candidate_class": "CRACK_TO_VOID_LIGAMENT", "start_m": list(map(float, tip)),
            "end_m": hit.tolist(), "fractured_ligament_length_m": ligament,
            "free_void_span_m": 0.0, "barrier_id": existing_barrier_id,
            "uses_existing_cleavage_barrier": True}


def connect_crack_to_void(registry: VoidRegistry, cavity_id: str, candidate: Dict[str, Any],
                          crack_component_id: str, verifier: Optional[Callable[[], None]] = None) -> None:
    """Atomically connect a ligament; any remesh/equilibrium/topology veto rolls back."""
    with VoidTransaction(registry) as tx:
        cavity = registry.cavities[cavity_id]
        if cavity.status != CavityStatus.RESOLVED_VOID:
            raise ValueError("only a resolved void can connect")
        length = float(candidate["fractured_ligament_length_m"])
        registry.ledger.fractured_ligament_increment += length
        registry.ledger.projected_fractured_length += length
        registry.ledger.active_front_coordinate_increment += length
        registry.ledger.projected_front_advance += length
        cavity.status = CavityStatus.CONNECTED_VOID
        cavity.connected_crack_component_id = crack_component_id
        if verifier is not None:
            verifier()
        registry.event_history.append({"event": "CRACK_TO_VOID_LIGAMENT", "void_id": cavity_id,
                                       "fractured_length_m": length})
        tx.commit()


def activate_downstream_front(registry: VoidRegistry, cavity_id: str, point_m: Sequence[float],
                              direction: Sequence[float], front_id: str,
                              renewed_tip_radius_m: float) -> Dict[str, Any]:
    """Separate cleavage event driven by direct smooth-boundary tensor traction."""
    cavity = registry.cavities[cavity_id]
    if cavity.status != CavityStatus.CONNECTED_VOID:
        raise ValueError("downstream nucleation requires a connected void")
    if renewed_tip_radius_m <= 0:
        raise ValueError("tip renewal radius must be positive")
    event = {"event": "DOWNSTREAM_CAVITY_SURFACE_NUCLEATION", "front_id": front_id,
             "parent_void_id": cavity_id, "point_m": list(map(float, point_m)),
             "direction": list(map(float, direction)), "driving": "DIRECT_LOCAL_BOUNDARY_TENSOR",
             "analytical_tip_amplification_used": False, "r_tip_m": float(renewed_tip_radius_m),
             "R_void_m": cavity.radius_m}
    cavity.status = CavityStatus.DOWNSTREAM_FRONT_ACTIVE
    registry.ledger.free_void_span_increment += 2*cavity.radius_m
    registry.ledger.projected_free_span += 2*cavity.radius_m
    registry.ledger.active_front_coordinate_increment += 2*cavity.radius_m
    registry.ledger.projected_front_advance += 2*cavity.radius_m
    registry.event_history.append(event)
    return event


def promotion_is_resolved(cavity: Cavity, local_h_m: float, ligament_m: float,
                          config: VoidingConfig, boundary_segments: int) -> bool:
    return (cavity.status == CavityStatus.STABLE_SUBGRID_VOID
            and boundary_segments >= config.promotion_min_boundary_segments
            and local_h_m/cavity.radius_m <= config.promotion_max_h_over_R
            and ligament_m/local_h_m >= config.promotion_min_ligament_layers)


def promote_cavity(cavity: Cavity, hole: ExplicitHoleMesh, transaction_id: int) -> None:
    """Representation-only promotion preserving ID, area, inventory, and lineage."""
    target_area = cavity.area_per_unit_thickness_m2
    area_error = abs(hole.area_m2-target_area)/max(target_area, 1e-300)
    chord_error_bound = 1.0-math.sin(math.pi/len(hole.cavity_boundary_nodes))/(math.pi/len(hole.cavity_boundary_nodes))
    if area_error > max(0.03, 2*chord_error_bound):
        raise ValueError("promotion area conservation gate failed")
    cavity.status = CavityStatus.RESOLVED_VOID
    cavity.shape_representation = "EXPLICIT_TRACTION_FREE_POLYGON"
    cavity.surface_boundary_node_ids = hole.cavity_boundary_nodes.astype(int).tolist()
    cavity.perimeter_m = hole.perimeter_m
    cavity.last_accepted_transaction = transaction_id


def empty_registry(config: Optional[VoidingConfig] = None) -> VoidRegistry:
    return VoidRegistry(config=config or VoidingConfig())
