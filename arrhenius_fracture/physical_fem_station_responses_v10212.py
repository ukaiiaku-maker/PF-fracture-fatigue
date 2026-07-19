"""Generate production-mesh active signed responses at FEM-resolved stations.

v10.2.14 deliberately limits the mechanically measured operator to the active
process zone.  The current scalar wake state does not retain the two-dimensional
position required to replay a persistent signed wake after crack deflection, so
wake rows are neither generated nor silently approximated.
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
    DEFAULT_MINIMUM_RESIDUAL_STIFFNESS_FRACTION,
    DEFAULT_STIFFNESS_KAPPA,
    INTERACTION_INTEGRAL_MODEL_ID,
    SlipRibbonPerturbation,
    element_residual_stiffness_fraction,
    equilibrated_base_state,
    interaction_response,
)

MODEL_ID = "v10.2.14_production_mesh_active_signed_spatial_station_responses"


def _unit(values) -> np.ndarray:
    vector = np.asarray(values, dtype=float).reshape(2)
    norm = float(np.linalg.norm(vector))
    if not math.isfinite(norm) or norm <= 0.0:
        raise ValueError("direction vector must be finite and nonzero")
    return vector / norm


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


def _nearest_supported_terminal(
    *,
    mesh,
    damage: np.ndarray,
    tip: np.ndarray,
    ray: np.ndarray,
    nominal: np.ndarray,
    minimum_distance_m: float,
    width_m: float,
    minimum_residual_stiffness_fraction: float,
    stiffness_kappa: float,
) -> tuple[np.ndarray, dict]:
    centroids = np.asarray(mesh.nodes, dtype=float)[
        np.asarray(mesh.elems, dtype=int)
    ].mean(axis=1)
    residual = element_residual_stiffness_fraction(
        mesh, damage, stiffness_kappa=stiffness_kappa
    )
    offset = centroids - tip[None, :]
    longitudinal = offset @ ray
    transverse_vector = offset - longitudinal[:, None] * ray[None, :]
    transverse = np.linalg.norm(transverse_vector, axis=1)
    h_tip = max(float(mesh.hbar_tip), 1.0e-15)
    corridor = max(float(width_m) + h_tip, 2.0 * h_tip)
    supported = (
        (residual >= float(minimum_residual_stiffness_fraction))
        & (longitudinal >= float(minimum_distance_m))
        & (transverse <= corridor)
    )
    indices = np.flatnonzero(supported)
    if indices.size == 0:
        raise ValueError(
            "no mechanically supported FEM terminal is available on the active "
            "slip ray"
        )
    geometric_error = np.linalg.norm(centroids[indices] - nominal[None, :], axis=1)
    order = np.lexsort((transverse[indices], geometric_error))
    index = int(indices[int(order[0])])
    return centroids[index].copy(), {
        "terminal_element": index,
        "terminal_longitudinal_m": float(longitudinal[index]),
        "terminal_transverse_m": float(transverse[index]),
        "terminal_residual_stiffness_fraction": float(residual[index]),
        "terminal_corridor_half_width_m": float(corridor),
    }


def _active_ribbon_geometry(
    *,
    system: int,
    x_m: float,
    width_m: float,
    mesh,
    damage: np.ndarray,
    tip: np.ndarray,
    forward: np.ndarray,
    slip_direction: np.ndarray,
    minimum_residual_stiffness_fraction: float,
    stiffness_kappa: float,
) -> tuple[np.ndarray, np.ndarray, dict]:
    slip = _unit(slip_direction)
    ray = slip if float(slip @ forward) >= 0.0 else -slip
    resolution = max(float(width_m), 2.0 * float(mesh.hbar_tip), 1.0e-12)
    requested_distance = max(float(x_m), 0.0)
    distance = max(requested_distance, 4.0 * resolution)
    start = np.asarray(tip, dtype=float).copy()
    nominal_end = start + distance * ray
    end, terminal = _nearest_supported_terminal(
        mesh=mesh,
        damage=damage,
        tip=start,
        ray=ray,
        nominal=nominal_end,
        minimum_distance_m=3.0 * resolution,
        width_m=width_m,
        minimum_residual_stiffness_fraction=(
            minimum_residual_stiffness_fraction
        ),
        stiffness_kappa=stiffness_kappa,
    )
    length = float(np.linalg.norm(end - start))
    if length < 2.0 * resolution:
        raise ValueError(
            f"active ribbon for system {system} is not mesh resolved: "
            f"length={length:.6g}, resolution={resolution:.6g}"
        )
    return start, end, {
        "region": "active",
        "system": int(system),
        "requested_x_m": requested_distance,
        "start_xy_m": start.tolist(),
        "end_xy_m": end.tolist(),
        "actual_ribbon_length_m": length,
        "mesh_resolution_m": float(mesh.hbar_tip),
        "placement_resolution_m": resolution,
        "source_is_physical_crack_surface_tip": True,
        "source_relocated_into_intact_material": False,
        "active_coordinate_semantics": "distance_along_system_slip_ray_from_current_tip",
        **terminal,
    }


def _snapshot_crack_segments(meta: SnapshotMetadata, mesh) -> list[tuple[np.ndarray, np.ndarray]]:
    path = tuple(getattr(meta, "crack_path_xy_m", ()) or ())
    if len(path) >= 2:
        points = [np.asarray(row, dtype=float).reshape(2) for row in path]
        return [(points[i], points[i + 1]) for i in range(len(points) - 1)]
    tip = np.asarray(meta.crack_tip_xy_m, dtype=float)
    forward = _unit(meta.crack_direction)
    domain_length = float(
        np.ptp(mesh.nodes[:, 0]) + np.ptp(mesh.nodes[:, 1])
    )
    return [(tip - max(domain_length, meta.interaction_ell_m) * forward, tip)]


def generate_station_responses(
    snapshot_root: str | Path,
    *,
    out_csv: str | Path,
    magnitudes: Iterable[float] = (0.25, 0.5),
    ribbon_width_m: float | None = None,
    minimum_station_spacing_m: float | None = None,
    interaction_cfg: JIntegralConfig | None = None,
    minimum_residual_stiffness_fraction: float = (
        DEFAULT_MINIMUM_RESIDUAL_STIFFNESS_FRACTION
    ),
    stiffness_kappa: float = DEFAULT_STIFFNESS_KAPPA,
) -> dict:
    data = load_snapshot(snapshot_root)
    meta: SnapshotMetadata = data["metadata"]
    out_csv = Path(out_csv)
    if out_csv.exists():
        raise FileExistsError(f"refusing to overwrite {out_csv}")
    values = sorted({float(value) for value in magnitudes})
    if len(values) < 2 or any(
        not math.isfinite(value) or value <= 0.0 for value in values
    ):
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
        mesh=data["mesh"],
        boundary=data["boundary"],
        baseline_u=data["u"],
        baseline_ep_gp=data["ep_gp"],
        rho_gp=data["rho_gp"],
        d=data["d"],
        D=data["D"],
        mat=data["mat"],
        Uy_top=meta.Uy_top_m,
        Uy_bot=meta.Uy_bot_m,
    )
    tip = np.asarray(meta.crack_tip_xy_m, dtype=float)
    forward = _unit(meta.crack_direction)
    crack_segments = _snapshot_crack_segments(meta, data["mesh"])
    channel_directions = [_unit(row) for row in meta.channel_directions]
    channel_normals = [_unit(row) for row in meta.channel_normals]
    active_grid = tuple(meta.active_x_m)
    active_stations = _station_indices(active_grid, station_spacing)

    rows = []
    placements = []
    for system, (slip, normal) in enumerate(
        zip(channel_directions, channel_normals)
    ):
        for bin_index in active_stations:
            x_m = float(active_grid[bin_index])
            start, end, placement = _active_ribbon_geometry(
                system=system,
                x_m=x_m,
                width_m=width,
                mesh=data["mesh"],
                damage=data["d"],
                tip=tip,
                forward=forward,
                slip_direction=slip,
                minimum_residual_stiffness_fraction=(
                    minimum_residual_stiffness_fraction
                ),
                stiffness_kappa=stiffness_kappa,
            )
            placements.append({**placement, "bin": int(bin_index)})
            for sign in (-1, 1):
                for magnitude in values:
                    perturbation = SlipRibbonPerturbation(
                        system=system,
                        region="active",
                        bin_index=bin_index,
                        start_xy_m=start,
                        end_xy_m=end,
                        slip_direction=slip,
                        plane_normal=normal,
                        width_m=width,
                        burgers_m=float(data["mat"].b),
                        signed_line_content=float(sign) * magnitude,
                    )
                    response = interaction_response(
                        mesh=data["mesh"],
                        base_state=base,
                        baseline_ep_gp=data["ep_gp"],
                        rho_gp=data["rho_gp"],
                        d=data["d"],
                        D=data["D"],
                        mat=data["mat"],
                        boundary=data["boundary"],
                        Uy_top=meta.Uy_top_m,
                        Uy_bot=meta.Uy_bot_m,
                        crack_tip=tip,
                        crack_direction=forward,
                        interaction_ell_m=meta.interaction_ell_m,
                        perturbation=perturbation,
                        interaction_cfg=interaction_cfg,
                        crack_segments=crack_segments,
                        exclude_radius_m=meta.exclude_radius_m,
                        minimum_residual_stiffness_fraction=(
                            minimum_residual_stiffness_fraction
                        ),
                        stiffness_kappa=stiffness_kappa,
                    )
                    audit = response["perturbation"]
                    rows.append(
                        {
                            "state_id": meta.state_id,
                            "r_eff_over_r0": meta.r_eff_over_r0,
                            "opening_strength_fraction": meta.opening_strength_fraction,
                            "crack_extension_m": meta.crack_extension_m,
                            "region": "active",
                            "system": system,
                            "bin": bin_index,
                            "x_m": x_m,
                            "burgers_sign": sign,
                            "delta_signed_line_content": float(sign) * magnitude,
                            "K_I_base_Pa_sqrt_m": response[
                                "K_I_base_Pa_sqrt_m"
                            ],
                            "K_I_perturbed_Pa_sqrt_m": response[
                                "K_I_perturbed_Pa_sqrt_m"
                            ],
                            "K_II_base_Pa_sqrt_m": response[
                                "K_II_base_Pa_sqrt_m"
                            ],
                            "K_II_perturbed_Pa_sqrt_m": response[
                                "K_II_perturbed_Pa_sqrt_m"
                            ],
                            "interaction_integral_schema": (
                                INTERACTION_INTEGRAL_MODEL_ID
                            ),
                            "ribbon_width_m": width,
                            "mesh_area_ratio": audit["mesh_area_ratio"],
                        }
                    )

    if not rows:
        raise ValueError("no active signed FEM response rows were generated")
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
        "measured_station_indices": {"active": active_stations, "wake": []},
        "minimum_station_spacing_m": station_spacing,
        "ribbon_width_m": width,
        "perturbation_magnitudes": values,
        "placements": placements,
        "interaction_integral_schema": INTERACTION_INTEGRAL_MODEL_ID,
        "minimum_residual_stiffness_fraction": float(
            minimum_residual_stiffness_fraction
        ),
        "stiffness_kappa": float(stiffness_kappa),
        "active_kernel_mechanically_measured": True,
        "wake_kernel_mechanically_measured": False,
        "wake_shielding_supported": False,
        "wake_rows_generated": 0,
        "wake_disable_reason": (
            "scalar wake bins do not preserve two-dimensional signed line positions "
            "after crack advance or deflection"
        ),
        "source_is_physical_crack_surface": True,
        "source_relocated_into_intact_material": False,
        "fem_tip_geometry_blunted": meta.fem_tip_geometry_blunted,
        "r_eff_is_analytical_tip_state": meta.r_eff_is_analytical_tip_state,
        "production_parameterization_allowed": False,
    }
    out_csv.with_suffix(".audit.json").write_text(json.dumps(report, indent=2))
    return report


__all__ = ["MODEL_ID", "generate_station_responses"]
