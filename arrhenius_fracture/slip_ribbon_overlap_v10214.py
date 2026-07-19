"""Exact triangle/finite-ribbon overlap for v10.2.14 signed perturbations.

The earlier production evaluator snapped every requested ribbon endpoint to a
nearby element centroid and selected whole elements by centroid distance.  On a
graded mesh this silently mapped many distinct MPZ bins to the same endpoint.
This module keeps the requested endpoint exactly and integrates the rectangular
ribbon over each triangular element by polygon clipping.  Element eigenstrain is
weighted by the exact overlap-area fraction, preserving the continuum ribbon
area without requiring centroids to lie on the slip ray.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .unit_slip_perturbation_v1026 import SlipRibbonPerturbation

MODEL_ID = "v10.2.14_exact_triangle_ribbon_overlap"


@dataclass(frozen=True)
class RibbonOverlapSupport:
    overlap_area_e_m2: np.ndarray
    terminal_overlap_area_e_m2: np.ndarray
    terminal_window_m: float


def _clip_half_plane(
    polygon: np.ndarray,
    *,
    axis: int,
    bound: float,
    keep_greater: bool,
    tolerance: float,
) -> np.ndarray:
    if polygon.shape[0] == 0:
        return polygon

    def signed_distance(point: np.ndarray) -> float:
        value = float(point[axis] - bound)
        return value if keep_greater else -value

    output: list[np.ndarray] = []
    previous = polygon[-1]
    previous_distance = signed_distance(previous)
    previous_inside = previous_distance >= -tolerance
    for current in polygon:
        current_distance = signed_distance(current)
        current_inside = current_distance >= -tolerance
        if current_inside != previous_inside:
            denominator = previous_distance - current_distance
            if abs(denominator) > 0.0:
                fraction = previous_distance / denominator
                output.append(previous + fraction * (current - previous))
        if current_inside:
            output.append(current)
        previous = current
        previous_distance = current_distance
        previous_inside = current_inside
    if not output:
        return np.empty((0, 2), dtype=float)
    return np.asarray(output, dtype=float)


def _clip_rectangle(
    polygon: np.ndarray,
    *,
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
    tolerance: float,
) -> np.ndarray:
    result = np.asarray(polygon, dtype=float)
    for axis, bound, keep_greater in (
        (0, x_min, True),
        (0, x_max, False),
        (1, y_min, True),
        (1, y_max, False),
    ):
        result = _clip_half_plane(
            result,
            axis=axis,
            bound=bound,
            keep_greater=keep_greater,
            tolerance=tolerance,
        )
        if result.shape[0] == 0:
            break
    return result


def _polygon_area(polygon: np.ndarray) -> float:
    if polygon.shape[0] < 3:
        return 0.0
    x = polygon[:, 0]
    y = polygon[:, 1]
    return 0.5 * abs(float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))


def _overlap_areas(
    mesh,
    *,
    start: np.ndarray,
    end: np.ndarray,
    width_m: float,
    terminal_window_m: float,
) -> RibbonOverlapSupport:
    tangent = np.asarray(end, dtype=float) - np.asarray(start, dtype=float)
    length = float(np.linalg.norm(tangent))
    if length <= 0.0:
        raise ValueError("slip-ribbon length must be positive")
    tangent /= length
    transverse = np.array([-tangent[1], tangent[0]], dtype=float)
    rotation = np.column_stack([tangent, transverse])
    nodes = np.asarray(mesh.nodes, dtype=float)
    elems = np.asarray(mesh.elems, dtype=int)
    local_nodes = (nodes - np.asarray(start, dtype=float)[None, :]) @ rotation
    half_width = 0.5 * float(width_m)
    terminal_start = max(0.0, length - float(terminal_window_m))
    tolerance = 1.0e-12 * max(length, float(width_m), 1.0)
    overlap = np.zeros(int(mesh.ne), dtype=float)
    terminal = np.zeros(int(mesh.ne), dtype=float)
    for element, conn in enumerate(elems):
        triangle = local_nodes[conn]
        clipped = _clip_rectangle(
            triangle,
            x_min=0.0,
            x_max=length,
            y_min=-half_width,
            y_max=half_width,
            tolerance=tolerance,
        )
        overlap[element] = _polygon_area(clipped)
        if overlap[element] > 0.0:
            terminal_clipped = _clip_rectangle(
                triangle,
                x_min=terminal_start,
                x_max=length,
                y_min=-half_width,
                y_max=half_width,
                tolerance=tolerance,
            )
            terminal[element] = _polygon_area(terminal_clipped)
    return RibbonOverlapSupport(
        overlap_area_e_m2=overlap,
        terminal_overlap_area_e_m2=terminal,
        terminal_window_m=float(terminal_window_m),
    )


def overlap_weighted_slip_ribbon_increment(
    mesh,
    perturbation: SlipRibbonPerturbation,
) -> tuple[np.ndarray, dict[str, Any], RibbonOverlapSupport]:
    """Return an area-conservative element eigenstrain increment.

    Each triangle receives the continuum ribbon eigenstrain multiplied by its
    exact ribbon-overlap fraction.  This avoids endpoint snapping and the
    whole-element area errors of centroid selection.
    """
    p = perturbation.validate()
    start = np.asarray(p.start_xy_m, dtype=float).reshape(2)
    end = np.asarray(p.end_xy_m, dtype=float).reshape(2)
    length = float(np.linalg.norm(end - start))
    h_tip = max(float(getattr(mesh, "hbar_tip", 0.0)), 0.0)
    terminal_window = min(
        length,
        max(float(p.width_m), 2.0 * h_tip, 1.0e-12),
    )
    support = _overlap_areas(
        mesh,
        start=start,
        end=end,
        width_m=float(p.width_m),
        terminal_window_m=terminal_window,
    )
    area_e = np.asarray(mesh.area_e, dtype=float)
    if area_e.shape != (int(mesh.ne),) or np.any(area_e <= 0.0):
        raise ValueError("mesh element areas must be positive")
    overlap = np.asarray(support.overlap_area_e_m2, dtype=float)
    selected = overlap > max(1.0e-30, 1.0e-14 * float(p.width_m) * length)
    if not np.any(selected):
        raise ValueError(
            "slip ribbon has no triangular overlap; verify the endpoint, width, and mesh"
        )
    fractions = np.clip(overlap / area_e, 0.0, 1.0)
    slip = np.asarray(p.slip_direction, dtype=float)
    slip /= np.linalg.norm(slip)
    normal = np.asarray(p.plane_normal, dtype=float)
    normal /= np.linalg.norm(normal)
    gamma = float(p.plastic_shear)
    voigt = gamma * np.array(
        [
            slip[0] * normal[0],
            slip[1] * normal[1],
            slip[0] * normal[1] + slip[1] * normal[0],
        ],
        dtype=float,
    )
    increment = voigt[:, None] * fractions[None, :]
    requested_area = length * float(p.width_m)
    represented_area = float(np.sum(overlap))
    terminal_requested_area = support.terminal_window_m * float(p.width_m)
    terminal_area = float(np.sum(support.terminal_overlap_area_e_m2))
    return increment, {
        **p.audit_payload(),
        "overlap_schema": MODEL_ID,
        "integration_scheme": "exact_triangle_rectangle_overlap_area_weighting",
        "endpoint_snapped_to_element_centroid": False,
        "requested_endpoint_used_exactly": True,
        "selected_elements": int(np.count_nonzero(selected)),
        "represented_area_m2": represented_area,
        "represented_ribbon_length_m": length,
        "requested_ribbon_area_m2": requested_area,
        "mesh_area_ratio": represented_area / max(requested_area, 1.0e-30),
        "terminal_window_m": float(support.terminal_window_m),
        "terminal_overlap_area_m2": terminal_area,
        "terminal_requested_area_m2": terminal_requested_area,
        "terminal_geometry_coverage_ratio": terminal_area
        / max(terminal_requested_area, 1.0e-30),
        "maximum_element_overlap_fraction": float(np.max(fractions)),
    }, support


__all__ = [
    "MODEL_ID",
    "RibbonOverlapSupport",
    "overlap_weighted_slip_ribbon_increment",
]
