"""Prospective circular-void crack-steering predictor and geometry checks.

The analytical expression is the circular-void (m=0) specialization of Eq. 4
in Misseroni, Movchan, Movchan & Bigoni (2015).  It is intentionally a
diagnostic predictor; it never selects or creates a production crack branch.
"""
from __future__ import annotations

import math
from typing import Iterable

import numpy as np


MODEL_ID = "misseroni-movchan-movchan-bigoni-2015-circular-plane-stress"


def circular_void_path_offset_m(
    crack_coordinate_m: float | np.ndarray,
    *,
    cavity_center_m: tuple[float, float],
    cavity_radius_m: float,
    poisson_ratio: float,
) -> float | np.ndarray:
    """Plane-stress asymptotic path H(l) for one circular traction-free void."""
    x0, y0 = map(float, cavity_center_m)
    radius = float(cavity_radius_m)
    nu = float(poisson_ratio)
    if not (math.isfinite(x0) and math.isfinite(y0) and abs(y0) > 0.0):
        raise ValueError("the asymptotic predictor requires a finite off-axis cavity")
    if not math.isfinite(radius) or radius <= 0.0:
        raise ValueError("cavity radius must be finite and positive")
    if not math.isfinite(nu) or not (-1.0 < nu < 0.5):
        raise ValueError("plane-stress Poisson ratio lies outside its admissible range")
    coordinate = np.asarray(crack_coordinate_m, dtype=float)
    if np.any(~np.isfinite(coordinate)):
        raise ValueError("crack coordinate must be finite")
    dx = x0 - coordinate
    t = dx / np.sqrt(y0*y0 + dx*dx)
    value = ((1.0-nu*nu)*radius*radius/(2.0*y0)
             * (2.0 - 2.0*t - t*t + t*t*t))
    return float(value) if value.ndim == 0 else value


def initial_steering_prediction(
    tip_x_m: float,
    *,
    cavity_center_m: tuple[float, float],
    cavity_radius_m: float,
    poisson_ratio: float,
) -> dict[str, float | str]:
    """Return the local asymptotic offset and tangent angle at a fixed tip."""
    radius = float(cavity_radius_m)
    step = max(radius*1.0e-5, 1.0e-15)
    minus = circular_void_path_offset_m(
        tip_x_m-step, cavity_center_m=cavity_center_m,
        cavity_radius_m=radius, poisson_ratio=poisson_ratio)
    plus = circular_void_path_offset_m(
        tip_x_m+step, cavity_center_m=cavity_center_m,
        cavity_radius_m=radius, poisson_ratio=poisson_ratio)
    slope = (plus-minus)/(2.0*step)
    offset = circular_void_path_offset_m(
        tip_x_m, cavity_center_m=cavity_center_m,
        cavity_radius_m=radius, poisson_ratio=poisson_ratio)
    return {
        "model": MODEL_ID,
        "offset_at_initial_tip_m": float(offset),
        "slope_at_initial_tip": float(slope),
        "tangent_angle_deg": math.degrees(math.atan(float(slope))),
        "steering_sign": "ABOVE" if slope > 0.0 else "BELOW" if slope < 0.0 else "STRAIGHT",
        "applicability": "INITIAL_STEERING_ONLY_SMALL_FAR_VOID_ASYMPTOTIC",
        "diameter_over_centerline_distance": 2.0*radius/abs(float(cavity_center_m[1])),
    }


def segment_circle_clearance_m(
    first_m: Iterable[float], second_m: Iterable[float],
    *, cavity_center_m: Iterable[float], cavity_radius_m: float,
) -> float:
    """Signed minimum segment-to-ideal-circle clearance (positive is a miss)."""
    first = np.asarray(tuple(first_m), dtype=float)
    second = np.asarray(tuple(second_m), dtype=float)
    center = np.asarray(tuple(cavity_center_m), dtype=float)
    delta = second-first
    fraction = float(np.clip((center-first)@delta/max(float(delta@delta), 1.0e-300), 0.0, 1.0))
    return float(np.linalg.norm(first+fraction*delta-center)-float(cavity_radius_m))
