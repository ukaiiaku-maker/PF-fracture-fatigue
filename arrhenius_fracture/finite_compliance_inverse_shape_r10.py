"""Finite-compliance inverse-shape observables for the additive R10 search.

This module evaluates the same cohesive potential used by the native elements.
It does not fit a Paris law.  Rates are elementary opening hazards from the
connected minimum to its next saddle under the condensed environmental spring.
"""
from __future__ import annotations

import math
from typing import Any, Iterable

import numpy as np
from scipy.optimize import brentq

from .cohesive_arrhenius import (
    BILINEAR, COMPACT_INDEPENDENT, SMOOTH_EXPONENTIAL,
    BilinearCohesiveLaw, CohesiveThermalLaw, CompactIndependentCohesiveLaw,
    EV_TO_J, cohesive_shape_response, mechanical_completion,
)
from .material_manifest import KB_EV_PER_K

DEFAULT_TIP_RADIUS_M = 1.0e-6


def shape_coordinates(shape: str, thermal: CohesiveThermalLaw,
                      temperature_K: float) -> dict[str, float]:
    """Return physical support, peak, work, and initial-stiffness coordinates."""
    gamma = thermal.gamma(temperature_K)
    peak = thermal.t_hat(temperature_K)
    if shape == SMOOTH_EXPONENTIAL:
        cutoff = float(mechanical_completion(shape, 0., thermal, temperature_K)["cutoff_m"])
        peak_d = thermal.delta0_m
        k0 = thermal.initial_stiffness(temperature_K)
    elif shape == BILINEAR:
        law = BilinearCohesiveLaw(thermal)
        cutoff = law.final_separation(temperature_K)
        peak_d = law.peak_separation(temperature_K)
        k0 = float(law.tangent(0., temperature_K))
    elif shape == COMPACT_INDEPENDENT:
        law = CompactIndependentCohesiveLaw(thermal)
        cutoff = law.final_separation(temperature_K)
        peak_d = law.peak_separation(temperature_K)
        k0 = law.initial_stiffness(temperature_K)
    else:
        raise ValueError(f"unsupported cohesive shape: {shape}")
    return {"Gamma_J_m2": gamma, "peak_traction_Pa": peak,
            "peak_separation_m": peak_d, "completion_cutoff_m": cutoff,
            "initial_stiffness_Pa_per_m": k0}


def _stationary(imposed_m: float, environment_stiffness_Pa_per_m: float,
                thermal: CohesiveThermalLaw, temperature_K: float,
                shape: str, points: int = 321) -> list[dict[str, float | str]]:
    """All finite-compliance stationary points, without branch relabeling."""
    imposed = float(imposed_m)
    kenv = float(environment_stiffness_Pa_per_m)
    coords = shape_coordinates(shape, thermal, temperature_K)
    cutoff = coords["completion_cutoff_m"]
    if shape == SMOOTH_EXPONENTIAL:
        span = max(cutoff, imposed + 8.0 * thermal.delta0_m)
        # Exponential spacing resolves both the contact basin and long tail.
        positive = np.geomspace(max(span * 1e-10, 1e-20), span, points - 1)
        grid = np.r_[0., positive]
    else:
        span = max(cutoff * 1.001, imposed + .05 * cutoff)
        grid = np.linspace(0., span, points)

    def residual(d):
        return float(cohesive_shape_response(shape, d, thermal, temperature_K)[0]) + kenv * (d-imposed)

    values = np.asarray(cohesive_shape_response(shape, grid, thermal, temperature_K)[0]) + kenv * (grid-imposed)
    roots: list[float] = []
    tolerance = 128. * np.finfo(float).eps * max(float(np.max(np.abs(values))), 1.)
    for index in range(len(grid)-1):
        left, right = float(grid[index]), float(grid[index+1])
        fl, fr = float(values[index]), float(values[index+1])
        if abs(fl) <= tolerance:
            roots.append(left)
        if fl * fr < 0.:
            roots.append(float(brentq(residual, left, right,
                                      xtol=max(span*1e-13, 1e-22), rtol=1e-12)))
    if abs(float(values[-1])) <= tolerance:
        roots.append(float(grid[-1]))
    roots.sort()
    unique: list[float] = []
    for root in roots:
        if not unique or abs(root-unique[-1]) > max(span*1e-9, 1e-20):
            unique.append(root)
    output = []
    for root in unique:
        _, tangent, work = cohesive_shape_response(shape, root, thermal, temperature_K)
        curvature = float(tangent) + kenv
        energy = float(work) + .5 * kenv * (imposed-root)**2
        output.append({"delta_m": root, "energy_J_m2": energy,
                       "curvature_Pa_per_m": curvature,
                       "kind": "MINIMUM" if curvature > 0. else "SADDLE"})
    return output


def connected_barrier(imposed_m: float, environment_stiffness_Pa_per_m: float,
                      thermal: CohesiveThermalLaw, temperature_K: float,
                      shape: str) -> dict[str, Any]:
    points = _stationary(imposed_m, environment_stiffness_Pa_per_m, thermal,
                         temperature_K, shape)
    minima = [point for point in points if point["kind"] == "MINIMUM"]
    if not minima:
        return {"state": "NO_CONNECTED_MINIMUM", "barrier_J_m2": None,
                "root_gap_m": None, "stationary": points}
    connected = minima[0]
    saddles = [point for point in points if point["kind"] == "SADDLE" and
               float(point["delta_m"]) > float(connected["delta_m"])]
    if not saddles:
        return {"state": "NO_ESCAPE_SADDLE", "barrier_J_m2": None,
                "root_gap_m": None, "connected": connected,
                "stationary": points}
    saddle = saddles[0]
    barrier = float(saddle["energy_J_m2"])-float(connected["energy_J_m2"])
    return {"state": "METASTABLE", "barrier_J_m2": max(barrier, 0.),
            "root_gap_m": float(saddle["delta_m"])-float(connected["delta_m"]),
            "connected": connected, "saddle": saddle, "stationary": points}


def branch_loss(shape: str, thermal: CohesiveThermalLaw, temperature_K: float,
                environment_stiffness_Pa_per_m: float) -> dict[str, float | str | None]:
    """First bonded-branch fold of the actual spring-plus-potential system."""
    kenv = float(environment_stiffness_Pa_per_m)
    coords = shape_coordinates(shape, thermal, temperature_K)
    peak_d, cutoff = coords["peak_separation_m"], coords["completion_cutoff_m"]
    if shape == BILINEAR:
        fold_d = peak_d
        traction = float(cohesive_shape_response(shape, fold_d, thermal, temperature_K)[0])
        imposed = fold_d + traction / kenv
        return {"state": "CORNER_BRANCH_LOSS", "fold_separation_m": fold_d,
                "fold_imposed_m": imposed}
    grid = np.linspace(peak_d*(1.+1e-10), cutoff*(1.-1e-10), 2049)
    tangent = np.asarray(cohesive_shape_response(shape, grid, thermal, temperature_K)[1]) + kenv
    crossings = np.flatnonzero(tangent[:-1]*tangent[1:] < 0.)
    for index in crossings:
        left, right = float(grid[index]), float(grid[index+1])
        root = float(brentq(lambda d: float(cohesive_shape_response(
            shape, d, thermal, temperature_K)[1])+kenv, left, right,
            xtol=max(cutoff*1e-14, 1e-24), rtol=1e-13))
        # The first + to - crossing is the bonded branch's upper fold.
        before = float(cohesive_shape_response(shape, left, thermal, temperature_K)[1])+kenv
        after = float(cohesive_shape_response(shape, right, thermal, temperature_K)[1])+kenv
        if before > 0. and after < 0.:
            traction = float(cohesive_shape_response(shape, root, thermal, temperature_K)[0])
            return {"state": "SMOOTH_BRANCH_LOSS", "fold_separation_m": root,
                    "fold_imposed_m": root+traction/kenv}
    return {"state": "NO_BONDED_FOLD", "fold_separation_m": None,
            "fold_imposed_m": None}


def imposed_from_K(K_MPa_sqrt_m: float, environment_stiffness_Pa_per_m: float,
                   tip_radius_m: float = DEFAULT_TIP_RADIUS_M) -> float:
    stress = float(K_MPa_sqrt_m)*1e6/math.sqrt(2.*math.pi*float(tip_radius_m))
    return stress/float(environment_stiffness_Pa_per_m)


def K_from_imposed(imposed_m: float, environment_stiffness_Pa_per_m: float,
                   tip_radius_m: float = DEFAULT_TIP_RADIUS_M) -> float:
    stress = float(environment_stiffness_Pa_per_m)*float(imposed_m)
    return stress*math.sqrt(2.*math.pi*float(tip_radius_m))/1e6


def opening_rate(K_MPa_sqrt_m: float, environment_stiffness_Pa_per_m: float,
                 thermal: CohesiveThermalLaw, temperature_K: float, shape: str,
                 activation_area_m2: float, attempt_frequency_s: float,
                 tip_radius_m: float = DEFAULT_TIP_RADIUS_M) -> dict[str, Any]:
    imposed = imposed_from_K(K_MPa_sqrt_m, environment_stiffness_Pa_per_m,
                            tip_radius_m)
    landscape = connected_barrier(imposed, environment_stiffness_Pa_per_m,
                                  thermal, temperature_K, shape)
    density = landscape.get("barrier_J_m2")
    if density is None:
        rate = 0.
        barrier_kBT = None
    else:
        barrier = float(density)*float(activation_area_m2)
        barrier_kBT = barrier/(KB_EV_PER_K*EV_TO_J*float(temperature_K))
        rate = float(attempt_frequency_s)*math.exp(max(-barrier_kBT, -745.))
    return {**landscape, "K_MPa_sqrt_m": float(K_MPa_sqrt_m),
            "imposed_m": imposed, "elementary_rate_s-1": rate,
            "barrier_kBT": barrier_kBT}


def rate_curve(K_values: Iterable[float], environment_stiffness_Pa_per_m: float,
               thermal: CohesiveThermalLaw, temperature_K: float, shape: str,
               activation_area_m2: float, attempt_frequency_s: float,
               frequency_Hz: float = 1000.) -> dict[str, Any]:
    rows = [opening_rate(K, environment_stiffness_Pa_per_m, thermal,
                         temperature_K, shape, activation_area_m2,
                         attempt_frequency_s) for K in K_values]
    K = np.asarray([row["K_MPa_sqrt_m"] for row in rows])
    rates = np.asarray([row["elementary_rate_s-1"] for row in rows])
    per_cycle = np.minimum(rates/max(float(frequency_Hz), 1e-300), 1.)
    positive = rates > 0.
    slopes = np.full(len(rows), np.nan)
    curvature = np.full(len(rows), np.nan)
    if np.count_nonzero(positive) >= 3:
        indices = np.flatnonzero(positive)
        x = np.log(K[indices]); y = np.log(rates[indices])
        slopes[indices] = np.gradient(y, x, edge_order=2)
        curvature[indices] = np.gradient(slopes[indices], x, edge_order=2)
    for index, row in enumerate(rows):
        row["events_per_cycle_peak_rate"] = float(per_cycle[index])
        row["local_log_slope"] = None if not np.isfinite(slopes[index]) else float(slopes[index])
        row["local_log_curvature"] = (None if not np.isfinite(curvature[index])
                                      else float(curvature[index]))
    moderate = positive & (slopes >= 2.) & (slopes <= 8.) & (per_cycle >= 1e-12) & (per_cycle <= 1e-2)
    best_width = 0.; best_bounds = None
    start = None
    for index, flag in enumerate(np.r_[moderate, False]):
        if flag and start is None:
            start = index
        elif not flag and start is not None:
            end = index-1
            width = math.log10(K[end]/K[start]) if end > start else 0.
            if width > best_width:
                best_width, best_bounds = width, (float(K[start]), float(K[end]))
            start = None
    return {"rows": rows, "moderate_connected_width_log10K": best_width,
            "moderate_connected_bounds_MPa_sqrt_m": best_bounds,
            "max_moderate_points": int(np.count_nonzero(moderate)),
            "peak_rate_approximation": True}


__all__ = ["branch_loss", "connected_barrier", "imposed_from_K",
           "K_from_imposed", "opening_rate", "rate_curve", "shape_coordinates"]
