"""Direct prescribed-geometry FEM states for v10.2.28 signed kernels.

This module constructs deterministic fixed-crack elastic states directly from a
mechanical configuration.  It does not import or advance fracture hazards,
source emission, moving-process-zone kinetics, or material parameter options.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from .config import ElasticProperties, GeometryConfig, MeshConfig
from .crystal import (
    bcc_cleavage_traces,
    bcc_slip_traces,
    cubic_plane_strain_D,
    zener_ratio,
)
from .kernel_configuration_v10227 import MechanicalKernelConfiguration
from .kernel_normalization_contract_v10228 import (
    DEFAULT_BURGERS_M,
    DEFAULT_KINETIC_PACKET_LENGTH_M,
    KernelNormalizationContract,
)
from .mesh import make_boundary_data, make_tri_mesh
from .physical_fem_snapshot_v10212 import SnapshotMetadata, save_snapshot
from .unit_slip_perturbation_v10212 import equilibrated_base_state

MODEL_ID = "v10.2.28_direct_prescribed_geometry_fem_states_v1"
ANCHOR_SCHEMA = "v10.2.28_prescribed_geometry_anchor_plan_v1"
MANIFEST_SCHEMA = "v10.2.28_prescribed_geometry_snapshot_manifest_v1"


@dataclass(frozen=True)
class PrescribedGeometryAnchor:
    state_id: str
    extension_m: float
    crack_tip_xy_m: tuple[float, float]
    crack_direction: tuple[float, float]


def _finite_positive(value: float, name: str) -> float:
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be positive and finite")
    return result


def _round_up(value: float, quantum: float) -> float:
    return math.ceil((float(value) - 1.0e-12 * float(quantum)) / float(quantum)) * float(quantum)


def prescribed_crack_direction(
    configuration: MechanicalKernelConfiguration,
) -> np.ndarray:
    """Return the deterministic straight-front direction for this configuration."""
    policy = str(
        dict(configuration.extra).get(
            "prescribed_crack_path_policy",
            "forward_100_cleavage_trace",
        )
    )
    if policy == "explicit_nominal_crack_angle":
        angle = math.radians(float(configuration.nominal_crack_angle_deg))
        direction = np.array([math.cos(angle), math.sin(angle)], dtype=float)
    elif policy == "forward_100_cleavage_trace":
        traces = bcc_cleavage_traces(float(configuration.theta_deg), include_110=False)
        if not traces:
            raise RuntimeError("no BCC {100} cleavage traces are available")
        global_forward = np.array([1.0, 0.0], dtype=float)
        candidates: list[np.ndarray] = []
        for trace in traces:
            tangent = np.asarray(trace["t"], dtype=float).reshape(2)
            if float(tangent @ global_forward) < 0.0:
                tangent = -tangent
            candidates.append(tangent)
        direction = max(candidates, key=lambda value: float(value @ global_forward))
    else:
        raise ValueError(f"unsupported prescribed crack-path policy: {policy!r}")
    norm = float(np.linalg.norm(direction))
    if not math.isfinite(norm) or norm <= 0.0:
        raise ValueError("prescribed crack direction must be finite and nonzero")
    direction = direction / norm
    if direction[0] <= 1.0e-12:
        raise ValueError("single-front prescribed geometry requires a globally forward crack")
    return direction


def plan_prescribed_geometry_anchors(
    configuration: MechanicalKernelConfiguration,
    required_max_extension_um: float,
) -> list[PrescribedGeometryAnchor]:
    """Plan exact geometric interpolation stations without stochastic advancement."""
    required_m = 1.0e-6 * _finite_positive(
        required_max_extension_um, "required_max_extension_um"
    )
    da = _finite_positive(configuration.da_phys_m, "configuration.da_phys_m")
    spacing = max(
        _round_up(configuration.atlas_anchor_spacing_m, da),
        da,
    )
    final_extension = _round_up(required_m, da)
    values = [0.0]
    current = spacing
    while current < final_extension - 1.0e-15:
        values.append(current)
        current += spacing
    if final_extension > values[-1] + 1.0e-15:
        values.append(final_extension)
    if len(values) < 2:
        values.append(da)

    direction = prescribed_crack_direction(configuration)
    origin = np.array([configuration.initial_crack_length_m, 0.0], dtype=float)
    anchors: list[PrescribedGeometryAnchor] = []
    for extension in values:
        tip = origin + float(extension) * direction
        anchors.append(
            PrescribedGeometryAnchor(
                state_id=f"E{int(round(1.0e6 * extension)):07d}",
                extension_m=float(extension),
                crack_tip_xy_m=(float(tip[0]), float(tip[1])),
                crack_direction=(float(direction[0]), float(direction[1])),
            )
        )
    return anchors


def plan_explicit_prescribed_geometry_anchors(
    configuration: MechanicalKernelConfiguration,
    extension_levels_um: tuple[float, ...],
) -> list[PrescribedGeometryAnchor]:
    """Plan an explicit append-only list of measured geometry stations."""
    if not extension_levels_um:
        raise ValueError("explicit prescribed-geometry levels cannot be empty")
    da = _finite_positive(configuration.da_phys_m, "configuration.da_phys_m")
    direction = prescribed_crack_direction(configuration)
    origin = np.array([configuration.initial_crack_length_m, 0.0], dtype=float)
    levels_m: list[float] = []
    for value_um in extension_levels_um:
        value_m = 1.0e-6 * float(value_um)
        if not math.isfinite(value_m) or value_m < 0.0:
            raise ValueError("explicit prescribed-geometry levels must be finite and nonnegative")
        quantum = round(value_m / da)
        if not math.isclose(value_m, quantum * da, rel_tol=0.0, abs_tol=1.0e-12 * da):
            raise ValueError("explicit prescribed-geometry levels must align with da_phys_m")
        levels_m.append(float(quantum * da))
    if any(right <= left for left, right in zip(levels_m, levels_m[1:])):
        raise ValueError("explicit prescribed-geometry levels must be strictly increasing")
    return [
        PrescribedGeometryAnchor(
            state_id=f"E{int(round(1.0e6 * extension)):07d}",
            extension_m=extension,
            crack_tip_xy_m=tuple(float(value) for value in origin + extension * direction),
            crack_direction=(float(direction[0]), float(direction[1])),
        )
        for extension in levels_m
    ]


def _isotropic_elastic_properties(
    configuration: MechanicalKernelConfiguration,
) -> tuple[ElasticProperties, np.ndarray, dict[str, float]]:
    """Map the current Zener-one cubic tensor exactly to the existing FEM material."""
    C11 = float(configuration.crystal_C11_Pa)
    C12 = float(configuration.crystal_C12_Pa)
    C44 = float(configuration.crystal_C44_Pa)
    zener = float(zener_ratio(C11, C12, C44))
    if not math.isclose(zener, 1.0, rel_tol=2.0e-10, abs_tol=2.0e-10):
        raise NotImplementedError(
            "the direct v10.2.28 provider currently supports the audited Zener-one "
            "elastic tensor only; a non-isotropic cubic tensor requires an anisotropic "
            "interaction-integral auxiliary solution"
        )
    lam = C12
    mu = C44
    E = mu * (3.0 * lam + 2.0 * mu) / (lam + mu)
    nu = lam / (2.0 * (lam + mu))
    burgers = float(dict(configuration.extra).get("burgers_m", DEFAULT_BURGERS_M))
    material = ElasticProperties(E=E, nu=nu, b=burgers)
    D = cubic_plane_strain_D(C11, C12, C44, theta_deg=configuration.theta_deg)
    return material, np.asarray(D, dtype=float), {
        "C11_Pa": C11,
        "C12_Pa": C12,
        "C44_Pa": C44,
        "zener_ratio": zener,
        "equivalent_E_Pa": E,
        "equivalent_nu": nu,
    }


def _orientation(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    return float(np.cross(b - a, c - a))


def _segments_intersect(
    p0: np.ndarray, p1: np.ndarray, q0: np.ndarray, q1: np.ndarray, tolerance: float
) -> bool:
    def sign(value: float) -> int:
        if value > tolerance:
            return 1
        if value < -tolerance:
            return -1
        return 0

    o1 = sign(_orientation(p0, p1, q0))
    o2 = sign(_orientation(p0, p1, q1))
    o3 = sign(_orientation(q0, q1, p0))
    o4 = sign(_orientation(q0, q1, p1))
    return o1 * o2 <= 0 and o3 * o4 <= 0


def _distance_to_segment(points: np.ndarray, start: np.ndarray, end: np.ndarray) -> np.ndarray:
    segment = end - start
    length2 = max(float(segment @ segment), 1.0e-300)
    parameter = np.clip(((points - start) @ segment) / length2, 0.0, 1.0)
    closest = start[None, :] + parameter[:, None] * segment[None, :]
    return np.linalg.norm(points - closest, axis=1)


def _prescribed_damage_field(
    mesh,
    boundary,
    *,
    start: np.ndarray,
    tip: np.ndarray,
    direction: np.ndarray,
    burgers_m: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Create a deterministic stiffness-killed straight crack ending at ``tip``."""
    damage = np.zeros(int(mesh.nn), dtype=float)
    damage[np.asarray(boundary.notch_nodes, dtype=int)] = 1.0
    extension = float(np.linalg.norm(tip - start))
    if extension <= 1.0e-18:
        return damage, {
            "extended_crack_elements": 0,
            "extended_crack_nodes": 0,
            "crack_band_half_width_m": 0.0,
            "ahead_of_tip_damaged_nodes": 0,
        }

    nodes = np.asarray(mesh.nodes, dtype=float)
    elems = np.asarray(mesh.elems, dtype=int)
    band = max(0.55 * float(mesh.hbar_tip), 4.0 * float(burgers_m), 1.0e-12)
    selected_elements: list[int] = []
    tolerance = max(1.0e-14, 1.0e-8 * band)
    for element_index, triangle_indices in enumerate(elems):
        triangle = nodes[triangle_indices]
        near = bool(np.any(_distance_to_segment(triangle, start, tip) <= band))
        crossed = False
        if not near:
            for edge in ((0, 1), (1, 2), (2, 0)):
                if _segments_intersect(
                    start, tip, triangle[edge[0]], triangle[edge[1]], tolerance
                ):
                    crossed = True
                    break
        if near or crossed:
            selected_elements.append(element_index)

    if not selected_elements:
        raise RuntimeError("prescribed crack segment has no FEM support")
    candidate_nodes = np.unique(elems[np.asarray(selected_elements, dtype=int)].ravel())
    projection = (nodes[candidate_nodes] - start[None, :]) @ direction
    transverse = np.abs(
        (nodes[candidate_nodes] - start[None, :])
        @ np.array([-direction[1], direction[0]], dtype=float)
    )
    keep = (
        (projection >= -tolerance)
        & (projection <= extension + tolerance)
        & (transverse <= max(2.5 * band, float(mesh.hbar_tip)))
    )
    extended_nodes = candidate_nodes[keep]
    damage[extended_nodes] = 1.0
    all_projection = (nodes - tip[None, :]) @ direction
    ahead = int(np.count_nonzero((damage >= 0.5) & (all_projection > tolerance)))
    if ahead:
        damage[(damage >= 0.5) & (all_projection > tolerance)] = 0.0
    return damage, {
        "extended_crack_elements": int(len(selected_elements)),
        "extended_crack_nodes": int(len(extended_nodes)),
        "crack_band_half_width_m": float(band),
        "ahead_of_tip_damaged_nodes": 0,
        "ahead_of_tip_candidates_removed": ahead,
        "crack_representation": "element_intersection_supported_nodal_stiffness_kill",
    }


def _channel_geometry(theta_deg: float) -> tuple[tuple[tuple[float, float], ...], tuple[tuple[float, float], ...]]:
    traces = bcc_slip_traces(float(theta_deg))
    if len(traces) != 2:
        raise RuntimeError(f"expected two reduced BCC slip traces; got {len(traces)}")
    directions = tuple(tuple(float(value) for value in trace["t"]) for trace in traces)
    normals = tuple(tuple(float(value) for value in trace["n"]) for trace in traces)
    return directions, normals


def _engine_configuration_payload(
    configuration: MechanicalKernelConfiguration,
    *,
    material: ElasticProperties,
) -> dict[str, Any]:
    packet_length = float(
        dict(configuration.extra).get(
            "kinetic_packet_length_m",
            DEFAULT_KINETIC_PACKET_LENGTH_M,
        )
    )
    normalization = KernelNormalizationContract(
        burgers_m=float(material.b),
        kinetic_packet_length_m=packet_length,
    ).validate()
    return {
        "schema": "v10.2.28_direct_kernel_engine_geometry_contract_v1",
        "b_m": float(material.b),
        "front_config": {
            "L_pz": float(configuration.process_zone_length_m),
            "da": float(configuration.da_phys_m),
        },
        "mpz_config": {
            "length_m": float(configuration.process_zone_length_m),
            "n_bins": int(configuration.process_zone_bins),
            "n_systems": 2,
        },
        "tip_config": {
            "packet_length_m": packet_length,
            "normalization_role": "unchanged production activation-to-line conversion",
        },
        "anisotropic_config": {
            "crystal_theta_deg": float(configuration.theta_deg),
        },
        "mechanical_configuration": configuration.canonical_payload(),
        "normalization_contract": normalization.audit_payload(),
        "material_parameter_option": None,
        "hazard_seed": None,
        "kinetics_advanced": False,
    }


def build_prescribed_geometry_snapshots(
    configuration: MechanicalKernelConfiguration,
    *,
    required_max_extension_um: float,
    outroot: str | Path,
    reference_opening_strain: float = 1.0e-5,
    explicit_extension_levels_um: tuple[float, ...] | None = None,
    mesh_seed_indices: tuple[int, ...] | None = None,
) -> dict[str, Any]:
    """Build all deterministic fixed-crack states required by one kernel family."""
    if configuration.branching_mode != "single_front" or configuration.maximum_fronts != 1:
        raise NotImplementedError("the v10.2.28 direct provider is single-front only")
    strain = _finite_positive(reference_opening_strain, "reference_opening_strain")
    root = Path(outroot).expanduser().resolve()
    if root.exists():
        raise FileExistsError(f"refusing to overwrite prescribed-geometry root: {root}")
    root.mkdir(parents=True)

    anchors = (
        plan_prescribed_geometry_anchors(configuration, required_max_extension_um)
        if explicit_extension_levels_um is None
        else plan_explicit_prescribed_geometry_anchors(
            configuration, explicit_extension_levels_um
        )
    )
    if explicit_extension_levels_um is not None and not math.isclose(
        1.0e6 * anchors[-1].extension_m,
        float(required_max_extension_um),
        rel_tol=0.0,
        abs_tol=1.0e-9,
    ):
        raise ValueError("required_max_extension_um must equal the final explicit level")
    if mesh_seed_indices is None:
        seed_indices = tuple(range(len(anchors)))
    else:
        seed_indices = tuple(int(value) for value in mesh_seed_indices)
        if len(seed_indices) != len(anchors):
            raise ValueError("mesh_seed_indices must match the explicit anchor count")
        if any(value < 0 for value in seed_indices) or len(set(seed_indices)) != len(seed_indices):
            raise ValueError("mesh_seed_indices must be unique nonnegative integers")
    material, D, elasticity = _isotropic_elastic_properties(configuration)
    geometry = GeometryConfig(
        Lx=float(configuration.specimen_length_x_m),
        Ly=float(configuration.specimen_length_y_m),
        a0=float(configuration.initial_crack_length_m),
        notch_half_thickness=float(configuration.notch_half_thickness_m),
    )
    mesh_cfg = MeshConfig(
        nx=int(configuration.mesh_nx),
        ny=int(configuration.mesh_ny),
        jitter=0.0,
        tip_h_fine=float(configuration.measurement_tip_h_fine_m),
        tip_ratio=float(configuration.measurement_tip_ratio),
    )
    active_dx = float(configuration.process_zone_length_m) / int(configuration.process_zone_bins)
    active_x = tuple((index + 0.5) * active_dx for index in range(int(configuration.process_zone_bins)))
    channel_directions, channel_normals = _channel_geometry(configuration.theta_deg)
    engine_config = _engine_configuration_payload(configuration, material=material)
    opening = 0.5 * strain * float(configuration.specimen_length_y_m)
    start = np.array([configuration.initial_crack_length_m, 0.0], dtype=float)

    state_rows: list[dict[str, Any]] = []
    for anchor, seed_index in zip(anchors, seed_indices):
        tip = np.asarray(anchor.crack_tip_xy_m, dtype=float)
        direction = np.asarray(anchor.crack_direction, dtype=float)
        mesh = make_tri_mesh(
            geometry,
            mesh_cfg,
            seed=1729 + seed_index,
            tip_center=tip,
        )
        boundary = make_boundary_data(mesh, geometry)
        damage, damage_audit = _prescribed_damage_field(
            mesh,
            boundary,
            start=start,
            tip=tip,
            direction=direction,
            burgers_m=material.b,
        )
        ep_gp = np.zeros((3, int(mesh.ne)), dtype=float)
        rho_gp = np.zeros(int(mesh.ne), dtype=float)
        baseline_u = np.zeros(int(mesh.ndof), dtype=float)
        base = equilibrated_base_state(
            mesh=mesh,
            boundary=boundary,
            baseline_u=baseline_u,
            baseline_ep_gp=ep_gp,
            rho_gp=rho_gp,
            d=damage,
            D=D,
            mat=material,
            Uy_top=opening,
            Uy_bot=-opening,
        )
        crack_path = (
            (0.0, 0.0),
            (float(configuration.initial_crack_length_m), 0.0),
            (float(tip[0]), float(tip[1])),
        )
        metadata = SnapshotMetadata(
            state_id=anchor.state_id,
            r_eff_over_r0=1.0,
            opening_strength_fraction=0.0,
            crack_extension_m=float(anchor.extension_m),
            temperature_K=float(
                configuration.temperature_K
                if configuration.temperature_dependent_mechanics
                else 300.0
            ),
            Uy_top_m=float(opening),
            Uy_bot_m=float(-opening),
            crack_tip_xy_m=anchor.crack_tip_xy_m,
            crack_direction=anchor.crack_direction,
            interaction_ell_m=float(configuration.interaction_length_m),
            exclude_radius_m=0.0,
            active_x_m=active_x,
            wake_x_m=(),
            channel_directions=channel_directions,
            channel_normals=channel_normals,
            material=asdict(material),
            engine_config=engine_config,
            fem_tip_geometry_blunted=False,
            r_eff_is_analytical_tip_state=False,
            cohesive_network_present=False,
            crack_path_xy_m=crack_path,
            displacement_state="post_dirichlet_equilibrium",
            active_kernel_supported=True,
            wake_kernel_supported=False,
        )
        state_root = root / anchor.state_id
        payload = save_snapshot(
            state_root,
            metadata=metadata,
            mesh=mesh,
            boundary=boundary,
            u=np.asarray(base["u"], dtype=float),
            ep_gp=ep_gp,
            rho_gp=rho_gp,
            d=damage,
            D=D,
        )
        payload.update(
            {
                "prescribed_geometry_model_id": MODEL_ID,
                "accepted_production_state_copied": False,
                "trajectory_state_cloned": False,
                "production_state_mutated": False,
                "plasticity_frozen": True,
                "kinetics_not_advanced": True,
                "hazard_clocks_not_advanced": True,
                "moving_process_zone_not_advanced": True,
                "fractional_moving_frame_not_called": True,
                "material_parameter_option_used": False,
                "hazard_seed_used": False,
                "prior_kernel_family_used": False,
                "fixed_crack_geometry": True,
                "direct_prescribed_crack_geometry": True,
                "measurement_mesh_hbar_tip_m": float(mesh.hbar_tip),
                "reference_opening_strain": strain,
                "base_reaction_top": float(base["reaction_top"]),
                "damage_construction": damage_audit,
                "elasticity": elasticity,
                "mechanical_configuration_fingerprint": configuration.fingerprint(),
            }
        )
        (state_root / "snapshot.json").write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n"
        )
        state_rows.append(
            {
                "state_id": anchor.state_id,
                "crack_extension_m": float(anchor.extension_m),
                "crack_tip_xy_m": list(anchor.crack_tip_xy_m),
                "crack_direction": list(anchor.crack_direction),
                "snapshot": str((state_root / "snapshot.json").resolve()),
                "mesh_nodes": int(mesh.nn),
                "mesh_elements": int(mesh.ne),
                "measurement_mesh_hbar_tip_m": float(mesh.hbar_tip),
                "damage_construction": damage_audit,
            }
        )

    anchor_plan = {
        "schema": ANCHOR_SCHEMA,
        "required_max_extension_um": float(required_max_extension_um),
        "atlas_anchor_spacing_m": float(configuration.atlas_anchor_spacing_m),
        "da_phys_m": float(configuration.da_phys_m),
        "anchors": state_rows,
    }
    if explicit_extension_levels_um is not None:
        anchor_plan["explicit_extension_levels_um"] = [
            float(value) for value in explicit_extension_levels_um
        ]
        anchor_plan["mesh_seed_indices"] = list(seed_indices)
    (root / "prescribed_geometry_anchor_plan.json").write_text(
        json.dumps(anchor_plan, indent=2, sort_keys=True) + "\n"
    )
    manifest = {
        "schema": MANIFEST_SCHEMA,
        "model_id": MODEL_ID,
        "mechanical_configuration": configuration.canonical_payload(),
        "mechanical_configuration_fingerprint": configuration.fingerprint(),
        "state_count": len(state_rows),
        "states": state_rows,
        "direct_prescribed_geometry": True,
        "material_parameter_option_required": False,
        "hazard_seed_required": False,
        "prior_kernel_family_required": False,
        "stochastic_trajectory_required": False,
        "fracture_hazard_advanced": False,
        "source_emission_advanced": False,
        "moving_process_zone_advanced": False,
        "production_physics_modified": False,
        "elasticity": elasticity,
    }
    (root / "kernel_capture_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    (root / "capture_complete.json").write_text(
        json.dumps(
            {
                "schema": "v10.2.28_direct_prescribed_geometry_complete_v1",
                "complete": True,
                "state_count": len(state_rows),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    return manifest


__all__ = [
    "MODEL_ID",
    "ANCHOR_SCHEMA",
    "MANIFEST_SCHEMA",
    "PrescribedGeometryAnchor",
    "prescribed_crack_direction",
    "plan_prescribed_geometry_anchors",
    "plan_explicit_prescribed_geometry_anchors",
    "build_prescribed_geometry_snapshots",
]
