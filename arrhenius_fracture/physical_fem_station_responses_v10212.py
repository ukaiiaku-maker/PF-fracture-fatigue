"""Generate physical signed responses only at FEM-resolved spatial stations.

The reduced MPZ grid may be much finer than the continuum mesh.  This module
therefore measures unit signed-slip responses at distinct, mesh-resolved
stations and records the complete reduced grid separately.  Projection from the
measured stations to that grid is performed later by the reviewed atlas builder;
sub-element response rows are never presented as direct FEM measurements.
"""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Iterable

import numpy as np

from .config import JIntegralConfig
from .physical_fem_snapshot_v10212 import (
    MODEL_ID as SNAPSHOT_MODEL_ID,
    RESPONSE_COLUMNS,
    SnapshotMetadata,
    load_snapshot,
)
from .unit_slip_perturbation_v10212 import (
    INTERACTION_INTEGRAL_MODEL_ID,
    SlipRibbonPerturbation,
    equilibrated_base_state,
    interaction_response,
)

MODEL_ID = "v10.2.13_fem_resolved_signed_spatial_station_responses"


def _unit(values) -> np.ndarray:
    vector = np.asarray(values, dtype=float).reshape(2)
    norm = float(np.linalg.norm(vector))
    if not math.isfinite(norm) or norm <= 0.0:
        raise ValueError("direction vector must be finite and nonzero")
    return vector / norm


def _element_damage(mesh, damage: np.ndarray) -> np.ndarray:
    value = np.asarray(damage, dtype=float).reshape(-1)
    if value.size == int(mesh.nn):
        return np.mean(value[np.asarray(mesh.elems, dtype=int)], axis=1)
    if value.size == int(mesh.ne):
        return value.copy()
    raise ValueError("damage field is incompatible with FEM mesh")


def _station_indices(coordinates: tuple[float, ...], minimum_spacing_m: float) -> list[int]:
    x = np.asarray(coordinates, dtype=float)
    if x.size == 0:
        return []
    if np.any(~np.isfinite(x)) or np.any(np.diff(x) < 0.0):
        raise ValueError("MPZ coordinates must be finite and nondecreasing")
    selected = [0]
    for index in range(1, x.size - 1):
        if float(x[index] - x[selected[-1]]) >= minimum_spacing_m:
            selected.append(index)
    if x.size > 1 and selected[-1] != x.size - 1:
        selected.append(x.size - 1)
    return selected


def _nearest_intact_centroid(
    *,
    mesh,
    damage: np.ndarray,
    nominal: np.ndarray,
    tip: np.ndarray,
    forward: np.ndarray,
    region: str,
    crack_normal: np.ndarray | None = None,
    side_sign: float = 0.0,
) -> np.ndarray:
    centroids = np.asarray(mesh.nodes, dtype=float)[np.asarray(mesh.elems, dtype=int)].mean(axis=1)
    de = _element_damage(mesh, damage)
    offset = centroids - tip[None, :]
    longitudinal = offset @ forward
    admissible = de < 0.05
    if region == "active":
        admissible &= longitudinal > 0.0
    elif region == "wake":
        admissible &= longitudinal < 0.0
        if crack_normal is not None and side_sign != 0.0:
            transverse = offset @ crack_normal
            admissible &= transverse * side_sign > 0.0
    else:
        raise ValueError(region)
    indices = np.flatnonzero(admissible)
    if indices.size == 0:
        raise ValueError(f"no intact FEM element is available for {region} perturbation")
    distances = np.linalg.norm(centroids[indices] - nominal[None, :], axis=1)
    return centroids[indices[int(np.argmin(distances))]].copy()


def _ribbon_geometry(
    *,
    region: str,
    system: int,
    x_m: float,
    width_m: float,
    mesh,
    damage: np.ndarray,
    tip: np.ndarray,
    forward: np.ndarray,
    slip_direction: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, dict]:
    crack_normal = np.asarray([-forward[1], forward[0]], dtype=float)
    slip = _unit(slip_direction)
    geometric_ray = slip if float(slip @ forward) >= 0.0 else -slip
    resolution = max(float(width_m), 2.0 * float(mesh.hbar_tip), 1.0e-12)
    requested_distance = max(float(x_m), 0.0)
    if region == "active":
        # The first reduced bin can be closer to the crack tip than the continuum
        # mesh can represent.  Use the nearest FEM-resolved terminal whose support
        # is separated from the source clipping window; the requested reduced-grid
        # coordinate remains recorded for the later reviewed projection.
        minimum_resolved_length = 4.0 * resolution
        distance = max(requested_distance, minimum_resolved_length)
        nominal_end = tip + distance * geometric_ray
        end = _nearest_intact_centroid(
            mesh=mesh,
            damage=damage,
            nominal=nominal_end,
            tip=tip,
            forward=forward,
            region="active",
        )
        start = tip.copy()
    else:
        minimum_resolved_length = resolution
        distance = max(requested_distance, minimum_resolved_length)
        face = tip - distance * forward
        side_projection = float(slip @ crack_normal)
        side_sign = 1.0 if side_projection >= 0.0 else -1.0
        nominal_start = face + resolution * side_sign * crack_normal
        start = _nearest_intact_centroid(
            mesh=mesh,
            damage=damage,
            nominal=nominal_start,
            tip=tip,
            forward=forward,
            region="wake",
            crack_normal=crack_normal,
            side_sign=side_sign,
        )
        nominal_end = start + resolution * geometric_ray
        end = _nearest_intact_centroid(
            mesh=mesh,
            damage=damage,
            nominal=nominal_end,
            tip=tip,
            forward=forward,
            region="wake",
            crack_normal=crack_normal,
            side_sign=side_sign,
        )
        if float(np.linalg.norm(end - start)) < 0.5 * width_m:
            nominal_end = start + 2.0 * resolution * geometric_ray
            end = _nearest_intact_centroid(
                mesh=mesh,
                damage=damage,
                nominal=nominal_end,
                tip=tip,
                forward=forward,
                region="wake",
                crack_normal=crack_normal,
                side_sign=side_sign,
            )
    if float(np.linalg.norm(end - start)) <= 0.0:
        raise ValueError(f"degenerate {region} ribbon geometry for system {system}")
    return start, end, {
        "region": region,
        "system": int(system),
        "requested_x_m": float(x_m),
        "requested_distance_m": requested_distance,
        "minimum_fem_resolved_ribbon_length_m": minimum_resolved_length,
        "fem_resolution_extension_applied": bool(distance > requested_distance + 1.0e-15),
        "start_xy_m": start.tolist(),
        "end_xy_m": end.tolist(),
        "actual_ribbon_length_m": float(np.linalg.norm(end - start)),
        "mesh_resolution_m": float(mesh.hbar_tip),
        "placement_resolution_m": resolution,
        "slip_direction_sign_preserved": True,
    }


def generate_station_responses(
    snapshot_root: str | Path,
    *,
    out_csv: str | Path,
    magnitudes: Iterable[float] = (0.25, 0.5),
    ribbon_width_m: float | None = None,
    minimum_station_spacing_m: float | None = None,
    interaction_cfg: JIntegralConfig | None = None,
) -> dict:
    data = load_snapshot(snapshot_root)
    meta: SnapshotMetadata = data["metadata"]
    out_csv = Path(out_csv)
    if out_csv.exists():
        raise FileExistsError(f"refusing to overwrite {out_csv}")
    values = sorted({float(value) for value in magnitudes})
    if len(values) < 2 or any(not math.isfinite(value) or value <= 0.0 for value in values):
        raise ValueError("at least two positive perturbation magnitudes are required")
    width = (
        max(2.0 * float(data["mesh"].hbar_tip), 10.0 * float(data["mat"].b))
        if ribbon_width_m is None
        else float(ribbon_width_m)
    )
    if not math.isfinite(width) or width <= 0.0:
        raise ValueError("ribbon width must be positive and finite")
    station_spacing = (
        max(2.0 * width, 2.0 * float(data["mesh"].hbar_tip))
        if minimum_station_spacing_m is None
        else float(minimum_station_spacing_m)
    )
    if not math.isfinite(station_spacing) or station_spacing <= 0.0:
        raise ValueError("minimum station spacing must be positive and finite")

    base = equilibrated_base_state(
        mesh=data["mesh"], boundary=data["boundary"], baseline_u=data["u"],
        baseline_ep_gp=data["ep_gp"], rho_gp=data["rho_gp"], d=data["d"],
        D=data["D"], mat=data["mat"], Uy_top=meta.Uy_top_m, Uy_bot=meta.Uy_bot_m,
    )
    tip = np.asarray(meta.crack_tip_xy_m, dtype=float)
    forward = _unit(meta.crack_direction)
    domain_length = float(np.ptp(data["mesh"].nodes[:, 0]) + np.ptp(data["mesh"].nodes[:, 1]))
    crack_segments = [(tip - max(domain_length, meta.interaction_ell_m) * forward, tip)]
    channel_directions = [_unit(row) for row in meta.channel_directions]
    channel_normals = [_unit(row) for row in meta.channel_normals]
    region_grids = {"active": tuple(meta.active_x_m), "wake": tuple(meta.wake_x_m)}
    station_map = {
        region: _station_indices(grid, station_spacing) for region, grid in region_grids.items()
    }
    rows = []
    placements = []
    for region, grid in region_grids.items():
        for system, (slip, normal) in enumerate(zip(channel_directions, channel_normals)):
            for bin_index in station_map[region]:
                x_m = float(grid[bin_index])
                start, end, placement = _ribbon_geometry(
                    region=region, system=system, x_m=x_m, width_m=width,
                    mesh=data["mesh"], damage=data["d"], tip=tip, forward=forward,
                    slip_direction=slip,
                )
                placements.append({**placement, "bin": int(bin_index)})
                for sign in (-1, 1):
                    for magnitude in values:
                        perturbation = SlipRibbonPerturbation(
                            system=system, region=region, bin_index=bin_index,
                            start_xy_m=start, end_xy_m=end, slip_direction=slip,
                            plane_normal=normal, width_m=width,
                            burgers_m=float(data["mat"].b),
                            signed_line_content=float(sign) * magnitude,
                        )
                        response = interaction_response(
                            mesh=data["mesh"], base_state=base,
                            baseline_ep_gp=data["ep_gp"], rho_gp=data["rho_gp"],
                            d=data["d"], D=data["D"], mat=data["mat"],
                            boundary=data["boundary"], Uy_top=meta.Uy_top_m,
                            Uy_bot=meta.Uy_bot_m, crack_tip=tip,
                            crack_direction=forward, interaction_ell_m=meta.interaction_ell_m,
                            perturbation=perturbation, interaction_cfg=interaction_cfg,
                            crack_segments=crack_segments,
                            exclude_radius_m=meta.exclude_radius_m,
                        )
                        audit = response["perturbation"]
                        rows.append({
                            "state_id": meta.state_id,
                            "r_eff_over_r0": meta.r_eff_over_r0,
                            "opening_strength_fraction": meta.opening_strength_fraction,
                            "crack_extension_m": meta.crack_extension_m,
                            "region": region, "system": system, "bin": bin_index,
                            "x_m": x_m, "burgers_sign": sign,
                            "delta_signed_line_content": float(sign) * magnitude,
                            "K_I_base_Pa_sqrt_m": response["K_I_base_Pa_sqrt_m"],
                            "K_I_perturbed_Pa_sqrt_m": response["K_I_perturbed_Pa_sqrt_m"],
                            "K_II_base_Pa_sqrt_m": response["K_II_base_Pa_sqrt_m"],
                            "K_II_perturbed_Pa_sqrt_m": response["K_II_perturbed_Pa_sqrt_m"],
                            "interaction_integral_schema": INTERACTION_INTEGRAL_MODEL_ID,
                            "ribbon_width_m": width,
                            "mesh_area_ratio": audit["mesh_area_ratio"],
                        })
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=RESPONSE_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    report = {
        "schema": MODEL_ID,
        "snapshot_schema": SNAPSHOT_MODEL_ID,
        "snapshot": str(Path(snapshot_root).resolve()),
        "responses": str(out_csv.resolve()),
        "state_id": meta.state_id,
        "response_rows": len(rows),
        "physical_fem_responses_generated": True,
        "responses_are_measured_stations_not_full_grid": True,
        "full_active_grid_x_m": list(meta.active_x_m),
        "full_wake_grid_x_m": list(meta.wake_x_m),
        "measured_station_indices": station_map,
        "minimum_station_spacing_m": station_spacing,
        "ribbon_width_m": width,
        "perturbation_magnitudes": values,
        "placements": placements,
        "interaction_integral_schema": INTERACTION_INTEGRAL_MODEL_ID,
        "fem_tip_geometry_blunted": meta.fem_tip_geometry_blunted,
        "r_eff_is_analytical_tip_state": meta.r_eff_is_analytical_tip_state,
        "production_parameterization_allowed": False,
    }
    out_csv.with_suffix(".audit.json").write_text(json.dumps(report, indent=2))
    return report


__all__ = ["MODEL_ID", "generate_station_responses"]
