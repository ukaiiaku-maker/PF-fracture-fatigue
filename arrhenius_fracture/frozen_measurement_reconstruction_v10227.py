"""Capture-only reconstruction of endpoint-resolved frozen FEM states.

This module is deliberately isolated from the production moving-tip solver. It
never receives a front engine and therefore cannot advance hazard clocks,
process-zone kinetics, source populations, or moving-frame advection. It clones
an already accepted trajectory state onto a separate graded measurement mesh,
reapplies the frozen sharp crack, and performs a fixed-plastic-state elastic
Dirichlet equilibrium solve for signed-kernel measurement.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Any, Iterable

import numpy as np
from scipy.spatial import cKDTree

from .config import GeometryConfig, MeshConfig
from .fem import assemble_mechanics, solve_dirichlet
from .mesh import make_boundary_data, make_tri_mesh

MODEL_ID = "v10.2.27_capture_only_endpoint_mesh_reconstruction_v3"


@dataclass(frozen=True)
class FrozenMeasurementMeshConfig:
    specimen_length_x_m: float
    specimen_length_y_m: float
    initial_crack_length_m: float
    notch_half_thickness_m: float
    mesh_nx: int
    mesh_ny: int
    tip_h_fine_m: float
    tip_ratio: float
    mesh_seed: int = 42
    kill_radius_floor_m: float = 0.0

    def validate(self) -> "FrozenMeasurementMeshConfig":
        for name in (
            "specimen_length_x_m",
            "specimen_length_y_m",
            "initial_crack_length_m",
            "notch_half_thickness_m",
            "tip_h_fine_m",
            "tip_ratio",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be positive and finite")
        floor = float(self.kill_radius_floor_m)
        if not math.isfinite(floor) or floor < 0.0:
            raise ValueError("kill_radius_floor_m must be nonnegative and finite")
        if int(self.mesh_nx) < 2 or int(self.mesh_ny) < 2:
            raise ValueError("measurement mesh_nx and mesh_ny must be at least two")
        if self.initial_crack_length_m >= self.specimen_length_x_m:
            raise ValueError("measurement initial crack must lie inside the specimen")
        if 2.0 * self.notch_half_thickness_m >= self.specimen_length_y_m:
            raise ValueError("measurement notch thickness must fit inside the specimen")
        return self

    def as_dict(self) -> dict[str, Any]:
        self.validate()
        return asdict(self)


def _unit_rows(path: Iterable[Iterable[float]]) -> tuple[np.ndarray, ...]:
    rows = []
    for row in path:
        value = np.asarray(tuple(row), dtype=float).reshape(2)
        if not np.all(np.isfinite(value)):
            raise ValueError("crack path contains a non-finite coordinate")
        rows.append(value)
    return tuple(rows)


def _path_length(path: tuple[np.ndarray, ...]) -> float:
    return float(
        sum(np.linalg.norm(right - left) for left, right in zip(path[:-1], path[1:]))
    )


def _resolve_crack_path(
    path: tuple[np.ndarray, ...],
    *,
    geometry: GeometryConfig,
    crack_tip_xy_m: np.ndarray,
    crack_extension_m: float,
    trajectory_hbar_tip_m: float,
) -> tuple[tuple[np.ndarray, ...], dict[str, Any]]:
    extension = float(crack_extension_m)
    if not math.isfinite(extension) or extension < 0.0:
        raise ValueError("crack_extension_m must be nonnegative and finite")
    initial_tip = np.array([geometry.a0, 0.0], dtype=float)
    direct_length = float(np.linalg.norm(crack_tip_xy_m - initial_tip))
    tolerance = max(
        2.0 * float(trajectory_hbar_tip_m),
        1.0e-9,
        1.0e-6 * max(extension, direct_length, 1.0e-12),
    )

    if path:
        if len(path) < 2:
            raise RuntimeError("serialized crack path must contain at least two points")
        start_gap = float(np.linalg.norm(path[0] - initial_tip))
        end_gap = float(np.linalg.norm(path[-1] - crack_tip_xy_m))
        length = _path_length(path)
        if start_gap > tolerance:
            raise RuntimeError(
                "serialized crack path is disconnected from the initial notch tip: "
                f"gap={start_gap:.9g} m, tolerance={tolerance:.9g} m"
            )
        if end_gap > tolerance:
            raise RuntimeError(
                "serialized crack path does not end at the accepted crack tip: "
                f"gap={end_gap:.9g} m, tolerance={tolerance:.9g} m"
            )
        if abs(length - extension) > tolerance:
            raise RuntimeError(
                "serialized crack-path length does not match cumulative extension: "
                f"path_length={length:.9g} m, extension={extension:.9g} m, "
                f"tolerance={tolerance:.9g} m"
            )
        return path, {
            "crack_path_source": "accepted_production_polyline",
            "accepted_production_polyline_available": True,
            "straight_single_front_path_synthesized": False,
            "crack_path_length_m": length,
            "direct_tip_displacement_m": direct_length,
            "crack_path_extension_consistency_tolerance_m": tolerance,
        }

    if extension <= tolerance and direct_length <= tolerance:
        return (), {
            "crack_path_source": "initial_notch_only",
            "accepted_production_polyline_available": False,
            "straight_single_front_path_synthesized": False,
            "crack_path_length_m": 0.0,
            "direct_tip_displacement_m": direct_length,
            "crack_path_extension_consistency_tolerance_m": tolerance,
        }

    # The current v10.2.27 campaign is single-front, nonbranching, and follows a
    # fixed straight 30-degree path, but the legacy observer does not serialize a
    # polyline. A straight segment is therefore admissible only when the direct
    # notch-tip displacement equals the cumulative path extension. A tortuous
    # state cannot satisfy this test and must provide an explicit polyline.
    if abs(direct_length - extension) > tolerance:
        raise RuntimeError(
            "nonzero capture lacks a production crack polyline and is not provably "
            "straight single-front growth: "
            f"direct_tip_displacement={direct_length:.9g} m, "
            f"cumulative_extension={extension:.9g} m, tolerance={tolerance:.9g} m"
        )
    resolved = (initial_tip, crack_tip_xy_m.copy())
    return resolved, {
        "crack_path_source": "verified_straight_single_front_tip_displacement",
        "accepted_production_polyline_available": False,
        "straight_single_front_path_synthesized": True,
        "crack_path_length_m": direct_length,
        "direct_tip_displacement_m": direct_length,
        "crack_path_extension_consistency_tolerance_m": tolerance,
    }


def _transfer_element_fields(old_mesh, new_mesh, ep_gp, rho_gp):
    old_centroids = old_mesh.nodes[old_mesh.elems].mean(axis=1)
    new_centroids = new_mesh.nodes[new_mesh.elems].mean(axis=1)
    _, parent = cKDTree(old_centroids).query(new_centroids)
    parent = np.asarray(parent, dtype=int)
    ep_new = np.ascontiguousarray(np.asarray(ep_gp, dtype=float)[:, parent])
    rho_new = np.ascontiguousarray(np.asarray(rho_gp, dtype=float)[parent])
    return ep_new, rho_new, parent


def _transfer_displacement(old_mesh, new_mesh, u):
    _, parent = cKDTree(old_mesh.nodes).query(new_mesh.nodes)
    old_u = np.asarray(u, dtype=float).reshape(-1, 2)
    return np.ascontiguousarray(old_u[np.asarray(parent, dtype=int)].reshape(-1))


def _reapply_frozen_sharp_crack(
    mesh,
    *,
    geometry: GeometryConfig,
    crack_tip_xy_m: np.ndarray,
    crack_path_xy_m: tuple[np.ndarray, ...],
    kill_radius_floor_m: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    damage = np.zeros(mesh.nn, dtype=float)
    x = mesh.nodes[:, 0]
    y = mesh.nodes[:, 1]
    damage[(x <= geometry.a0) & (np.abs(y) <= geometry.notch_half_thickness)] = 1.0

    path = crack_path_xy_m
    centroids = mesh.nodes[mesh.elems].mean(axis=1)
    element_radius = np.sqrt(np.maximum(mesh.area_e, 1.0e-30))
    kill_radius = max(float(mesh.hbar_tip), float(kill_radius_floor_m))
    killed_elements = np.zeros(mesh.ne, dtype=bool)
    for p0, p1 in zip(path[:-1], path[1:]):
        segment = p1 - p0
        length2 = float(segment @ segment)
        if length2 <= 1.0e-30:
            continue
        raw_fraction = ((centroids - p0[None, :]) @ segment) / length2
        fraction = np.clip(raw_fraction, 0.0, 1.0)
        projection = p0[None, :] + fraction[:, None] * segment[None, :]
        distance2 = np.sum((centroids - projection) ** 2, axis=1)
        radius = np.maximum(kill_radius, 0.7 * element_radius)
        within_segment = (raw_fraction >= 0.0) & (raw_fraction <= 1.0)
        selected = within_segment & (distance2 <= radius ** 2)
        killed_elements |= selected
        if np.any(selected):
            damage[mesh.elems[selected]] = 1.0

    ahead_killed = 0
    maximum_ahead_projection = 0.0
    if len(path) >= 2:
        last = path[-1] - path[-2]
        norm = float(np.linalg.norm(last))
        if norm > 1.0e-30:
            direction = last / norm
            ahead_projection = (centroids - crack_tip_xy_m[None, :]) @ direction
            threshold = max(0.25 * float(mesh.hbar_tip), 1.0e-12)
            ahead = killed_elements & (ahead_projection > threshold)
            ahead_killed = int(np.count_nonzero(ahead))
            maximum_ahead_projection = (
                float(np.max(ahead_projection[ahead])) if ahead_killed else 0.0
            )
            if ahead_killed:
                raise RuntimeError(
                    "capture-only crack reconstruction damaged elements ahead of the "
                    f"accepted tip: count={ahead_killed}, "
                    f"maximum_projection={maximum_ahead_projection:.9g} m"
                )

    return damage, {
        "crack_path_points": len(path),
        "killed_elements": int(np.count_nonzero(killed_elements)),
        "kill_radius_m": kill_radius,
        "kill_radius_floor_m": float(kill_radius_floor_m),
        "crack_damage_trace_width_policy": "one_local_measurement_element",
        "endpoint_caps_excluded": True,
        "ahead_of_tip_killed_elements": ahead_killed,
        "maximum_ahead_of_tip_killed_projection_m": maximum_ahead_projection,
        "initial_notch_reapplied": True,
        "accepted_or_verified_crack_path_reapplied": bool(len(path) >= 2),
    }


def reconstruct_frozen_measurement_state(
    *,
    source_mesh,
    source_boundary,
    source_u: np.ndarray,
    source_ep_gp: np.ndarray,
    source_rho_gp: np.ndarray,
    source_d: np.ndarray,
    D: np.ndarray,
    material,
    Uy_top_m: float,
    Uy_bot_m: float,
    crack_tip_xy_m: Iterable[float],
    crack_path_xy_m: Iterable[Iterable[float]],
    crack_extension_m: float,
    config: FrozenMeasurementMeshConfig,
) -> dict[str, Any]:
    """Clone one accepted trajectory state onto an endpoint-resolved mesh."""
    config = config.validate()
    tip = np.asarray(tuple(crack_tip_xy_m), dtype=float).reshape(2)
    if not np.all(np.isfinite(tip)):
        raise ValueError("crack tip must be finite")
    supplied_path = _unit_rows(crack_path_xy_m)
    source_damage = np.asarray(source_d, dtype=float)
    if source_damage.ndim != 1 or source_damage.size != source_mesh.nn:
        raise ValueError("source damage field is inconsistent with the trajectory mesh")

    geometry = GeometryConfig(
        Lx=float(config.specimen_length_x_m),
        Ly=float(config.specimen_length_y_m),
        a0=float(config.initial_crack_length_m),
        notch_half_thickness=float(config.notch_half_thickness_m),
    )
    path, path_audit = _resolve_crack_path(
        supplied_path,
        geometry=geometry,
        crack_tip_xy_m=tip,
        crack_extension_m=float(crack_extension_m),
        trajectory_hbar_tip_m=float(source_mesh.hbar_tip),
    )
    mesh_cfg = MeshConfig(
        nx=int(config.mesh_nx),
        ny=int(config.mesh_ny),
        tip_h_fine=float(config.tip_h_fine_m),
        tip_ratio=float(config.tip_ratio),
    )
    measurement_mesh = make_tri_mesh(
        geometry,
        mesh_cfg,
        seed=int(config.mesh_seed),
        tip_center=tip,
    )
    measurement_boundary = make_boundary_data(measurement_mesh, geometry)
    ep_gp, rho_gp, element_parent_map = _transfer_element_fields(
        source_mesh,
        measurement_mesh,
        source_ep_gp,
        source_rho_gp,
    )
    u_initial = _transfer_displacement(source_mesh, measurement_mesh, source_u)
    damage, crack_audit = _reapply_frozen_sharp_crack(
        measurement_mesh,
        geometry=geometry,
        crack_tip_xy_m=tip,
        crack_path_xy_m=path,
        kill_radius_floor_m=float(config.kill_radius_floor_m),
    )

    K, Rint, *_ = assemble_mechanics(
        measurement_mesh,
        u_initial,
        ep_gp,
        rho_gp,
        damage,
        np.asarray(D, dtype=float),
        material,
        cohesive_network=None,
    )
    u_equilibrium, reaction = solve_dirichlet(
        K,
        Rint,
        u_initial,
        measurement_boundary,
        float(Uy_top_m),
        float(Uy_bot_m),
    )
    K_check, R_check, *_ = assemble_mechanics(
        measurement_mesh,
        u_equilibrium,
        ep_gp,
        rho_gp,
        damage,
        np.asarray(D, dtype=float),
        material,
        cohesive_network=None,
    )
    _ = K_check

    audit = {
        "schema": MODEL_ID,
        "trajectory_state_cloned": True,
        "production_state_mutated": False,
        "plasticity_frozen": True,
        "kinetics_not_advanced": True,
        "hazard_clocks_not_advanced": True,
        "moving_process_zone_not_advanced": True,
        "fractional_moving_frame_not_called": True,
        "endpoint_mesh_reconstructed": True,
        "endpoint_mesh_re_equilibrated": True,
        "trajectory_mesh_hbar_tip_m": float(source_mesh.hbar_tip),
        "measurement_mesh_hbar_tip_m": float(measurement_mesh.hbar_tip),
        "trajectory_mesh_nodes": int(source_mesh.nn),
        "measurement_mesh_nodes": int(measurement_mesh.nn),
        "trajectory_mesh_elements": int(source_mesh.ne),
        "measurement_mesh_elements": int(measurement_mesh.ne),
        "transferred_parent_elements": int(len(element_parent_map)),
        "source_damage_field_interpolated": False,
        "source_damage_field_used_for_shape_validation_only": True,
        "measurement_damage_source": "initial_notch_plus_resolved_crack_path",
        "elastic_reaction_force": float(reaction),
        "postsolve_internal_force_norm": float(np.linalg.norm(R_check)),
        "measurement_mesh_config": config.as_dict(),
        **path_audit,
        **crack_audit,
    }
    return {
        "mesh": measurement_mesh,
        "boundary": measurement_boundary,
        "u": np.ascontiguousarray(u_equilibrium),
        "ep_gp": ep_gp,
        "rho_gp": rho_gp,
        "d": damage,
        "D": np.asarray(D, dtype=float).copy(),
        "crack_path_xy_m": tuple(tuple(float(value) for value in row) for row in path),
        "audit": audit,
    }


__all__ = [
    "MODEL_ID",
    "FrozenMeasurementMeshConfig",
    "reconstruct_frozen_measurement_state",
]
