"""Resolved-opening calibration for coarse moving-tip MPZ transactions.

The first accepted crack event is advanced through explicit geometry/plasticity
substeps.  The additive state left by those substeps, relative to pure moving-
frame advection of the same pre-event state, is stored as a protocol-local
operator and can be applied after later coarse crack increments.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


def _clip_polygon_axis(poly: np.ndarray, axis: int, bound: float, keep_ge: bool) -> np.ndarray:
    """Clip a convex polygon against one axis-aligned half plane."""
    if len(poly) == 0:
        return poly
    out = []
    prev = poly[-1]
    prev_in = (prev[axis] >= bound) if keep_ge else (prev[axis] <= bound)
    for cur in poly:
        cur_in = (cur[axis] >= bound) if keep_ge else (cur[axis] <= bound)
        if cur_in != prev_in:
            den = cur[axis] - prev[axis]
            frac = 0.0 if abs(den) < 1.0e-30 else (bound - prev[axis]) / den
            out.append(prev + frac * (cur - prev))
        if cur_in:
            out.append(cur)
        prev, prev_in = cur, cur_in
    return np.asarray(out, dtype=float).reshape(-1, 2)


def _triangle_rectangle_area(triangle: np.ndarray, xmin: float, xmax: float,
                             ymin: float, ymax: float) -> float:
    """Exact constant-element quadrature over an axis-aligned rectangle."""
    poly = np.asarray(triangle, dtype=float).reshape(-1, 2)
    poly = _clip_polygon_axis(poly, 0, xmin, True)
    poly = _clip_polygon_axis(poly, 0, xmax, False)
    poly = _clip_polygon_axis(poly, 1, ymin, True)
    poly = _clip_polygon_axis(poly, 1, ymax, False)
    if len(poly) < 3:
        return 0.0
    x, y = poly[:, 0], poly[:, 1]
    return 0.5 * abs(float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))


@dataclass
class IncrementalAdvectionOperator:
    distance_m: float
    active_retained_by_bin: np.ndarray
    wake_retained_by_bin: np.ndarray
    forward_passage_additive: float
    calibration_temperature_K: float
    calibration_Kmax_Pa_sqrt_m: float
    calibration_R: float
    microsteps: int
    event_duration_s: float
    inherited_count: float
    opening_generated_count: float
    admissible: bool = True
    total_shield_Pa_sqrt_m: float = 0.0
    shield_over_applied_Kmax: float = 0.0

    def apply(self, mpz: Any) -> dict[str, float]:
        if self.active_retained_by_bin.size != mpz.n_bins:
            raise ValueError("coarse advection operator active-grid mismatch")
        if self.wake_retained_by_bin.size != mpz.wake_n_bins:
            raise ValueError("coarse advection operator wake-grid mismatch")
        mpz.retained += self.active_retained_by_bin[None, :] / mpz.n_systems
        mpz.wake_retained += self.wake_retained_by_bin[None, :] / mpz.n_systems
        mpz.forward_passage_slip_count += max(self.forward_passage_additive, 0.0)
        active = float(np.sum(self.active_retained_by_bin))
        wake = float(np.sum(self.wake_retained_by_bin))
        mpz.bulk_inherited_retained_total += active + wake
        return {
            "coarse_operator_active_retained": active,
            "coarse_operator_wake_retained": wake,
            "coarse_operator_forward_passage": max(self.forward_passage_additive, 0.0),
        }

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema": "incremental-opening-advection-operator/1",
            "distance_m": self.distance_m,
            "active_retained_by_bin": self.active_retained_by_bin.tolist(),
            "wake_retained_by_bin": self.wake_retained_by_bin.tolist(),
            "forward_passage_additive": self.forward_passage_additive,
            "calibration_temperature_K": self.calibration_temperature_K,
            "calibration_Kmax_Pa_sqrt_m": self.calibration_Kmax_Pa_sqrt_m,
            "calibration_R": self.calibration_R,
            "microsteps": self.microsteps,
            "event_duration_s": self.event_duration_s,
            "inherited_count": self.inherited_count,
            "opening_generated_count": self.opening_generated_count,
            "admissible": self.admissible,
            "total_shield_Pa_sqrt_m": self.total_shield_Pa_sqrt_m,
            "shield_over_applied_Kmax": self.shield_over_applied_Kmax,
        }


def density_profile_in_active_zone(
    *, centers_x: np.ndarray, centers_y: np.ndarray, areas: np.ndarray,
    density_m2: np.ndarray, tip_x_m: float, length_m: float, n_bins: int,
    capture_width_m: float, element_vertices: np.ndarray | None = None,
) -> np.ndarray:
    """Area-integrate a nonnegative 2-D density into tip-relative MPZ bins."""
    rel = np.asarray(centers_x, float) - float(tip_x_m)
    half = 0.5 * max(float(capture_width_m), 0.0)
    mask = ((rel >= 0.0) & (rel < float(length_m))
            & (np.abs(np.asarray(centers_y, float)) <= half))
    profile = np.zeros(max(int(n_bins), 1), dtype=float)
    dx = float(length_m) / len(profile)
    if element_vertices is not None:
        triangles = np.asarray(element_vertices, dtype=float)
        rho = np.maximum(np.asarray(density_m2, float), 0.0)
        # Constant-element density is integrated using exact triangle/strip
        # overlap.  This avoids the full-element centroid jumps that otherwise
        # dominate a narrow moving capture window during mesh refinement.
        bbox = ((np.max(triangles[:, :, 0], axis=1) >= tip_x_m)
                & (np.min(triangles[:, :, 0], axis=1) <= tip_x_m + length_m)
                & (np.max(triangles[:, :, 1], axis=1) >= -half)
                & (np.min(triangles[:, :, 1], axis=1) <= half))
        for e in np.flatnonzero(bbox & (rho > 0.0)):
            lo = max(int(np.floor((np.min(triangles[e, :, 0]) - tip_x_m) / dx)), 0)
            hi = min(int(np.floor((np.max(triangles[e, :, 0]) - tip_x_m) / dx)), len(profile) - 1)
            for j in range(lo, hi + 1):
                overlap = _triangle_rectangle_area(
                    triangles[e], tip_x_m + j * dx, tip_x_m + (j + 1) * dx,
                    -half, half)
                profile[j] += rho[e] * overlap
        return profile
    if not np.any(mask):
        return profile
    indices = np.minimum((rel[mask] / dx).astype(int), len(profile) - 1)
    np.add.at(
        profile, indices,
        np.maximum(np.asarray(density_m2, float)[mask], 0.0)
        * np.asarray(areas, float)[mask],
    )
    return profile


def incoming_material_profile(
    *, centers_x: np.ndarray, centers_y: np.ndarray, areas: np.ndarray,
    excess_density_m2: np.ndarray, old_tip_x_m: float, new_tip_x_m: float,
    length_m: float, n_bins: int, capture_width_m: float,
    element_vertices: np.ndarray | None = None,
) -> np.ndarray:
    """Project only material newly entering the forward edge of the moving MPZ."""
    x = np.asarray(centers_x, float)
    old_rel = x - float(old_tip_x_m)
    new_rel = x - float(new_tip_x_m)
    half = 0.5 * max(float(capture_width_m), 0.0)
    mask = ((old_rel >= float(length_m)) & (new_rel < float(length_m))
            & (new_rel >= 0.0) & (np.abs(np.asarray(centers_y, float)) <= half))
    profile = np.zeros(max(int(n_bins), 1), dtype=float)
    dx = float(length_m) / len(profile)
    if element_vertices is not None:
        triangles = np.asarray(element_vertices, dtype=float)
        rho = np.maximum(np.asarray(excess_density_m2, float), 0.0)
        xmin = float(old_tip_x_m) + float(length_m)
        xmax = float(new_tip_x_m) + float(length_m)
        bbox = ((np.max(triangles[:, :, 0], axis=1) >= xmin)
                & (np.min(triangles[:, :, 0], axis=1) <= xmax)
                & (np.max(triangles[:, :, 1], axis=1) >= -half)
                & (np.min(triangles[:, :, 1], axis=1) <= half))
        for e in np.flatnonzero(bbox & (rho > 0.0)):
            overlap = _triangle_rectangle_area(triangles[e], xmin, xmax, -half, half)
            if overlap <= 0.0:
                continue
            # Newly encountered material enters through the forward-most bin.
            profile[-1] += rho[e] * overlap
        return profile
    if not np.any(mask):
        return profile
    indices = np.minimum((new_rel[mask] / dx).astype(int), len(profile) - 1)
    np.add.at(
        profile, indices,
        np.maximum(np.asarray(excess_density_m2, float)[mask], 0.0)
        * np.asarray(areas, float)[mask],
    )
    return profile


def operator_from_states(
    *, explicit: Any, pure_advection: Any, distance_m: float,
    temperature_K: float, Kmax_Pa_sqrt_m: float, R: float, microsteps: int,
    event_duration_s: float, inherited_count: float,
    opening_generated_count: float,
) -> IncrementalAdvectionOperator:
    active = np.maximum(
        np.sum(explicit.retained, axis=0) - np.sum(pure_advection.retained, axis=0),
        0.0,
    )
    wake = np.maximum(
        np.sum(explicit.wake_retained, axis=0)
        - np.sum(pure_advection.wake_retained, axis=0),
        0.0,
    )
    forward = max(
        float(explicit.forward_passage_slip_count)
        - float(pure_advection.forward_passage_slip_count),
        0.0,
    )
    return IncrementalAdvectionOperator(
        distance_m=float(distance_m),
        active_retained_by_bin=active,
        wake_retained_by_bin=wake,
        forward_passage_additive=forward,
        calibration_temperature_K=float(temperature_K),
        calibration_Kmax_Pa_sqrt_m=float(Kmax_Pa_sqrt_m),
        calibration_R=float(R), microsteps=int(microsteps),
        event_duration_s=float(event_duration_s),
        inherited_count=float(inherited_count),
        opening_generated_count=float(opening_generated_count),
    )
