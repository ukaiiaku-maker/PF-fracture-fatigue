"""Finite-compliance inverse-shape observables for the additive R10 search.

This module evaluates the same cohesive potential used by the native elements.
It does not fit a Paris law.  Rates are elementary opening hazards from the
connected minimum to its next saddle under the condensed environmental spring.
"""
from __future__ import annotations

import math
from typing import Any, Iterable

import numpy as np
from scipy.integrate import quad
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
        if shape == COMPACT_INDEPENDENT and thermal.compact_variant in {
                "rounded_plateau","two_scale_tail"}:
            rise=thermal.compact_rise_fraction*cutoff
            fall=(thermal.compact_fall_fraction if thermal.compact_variant=="rounded_plateau"
                  else thermal.compact_drop_fraction)*cutoff
            segment=max(points//4,33)
            grid=np.unique(np.r_[np.linspace(0.,rise,segment),
                np.linspace(rise,fall,2*segment),np.linspace(fall,cutoff,segment),
                np.linspace(cutoff,span,segment)])
        else:
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
                      shape: str, verify_quadrature: bool = False) -> dict[str, Any]:
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
    quadrature=None;relative_error=None
    if verify_quadrature:
        integrand=lambda separation:(float(cohesive_shape_response(
            shape,separation,thermal,temperature_K)[0])+
            float(environment_stiffness_Pa_per_m)*(separation-float(imposed_m)))
        quadrature=float(quad(integrand,float(connected["delta_m"]),
            float(saddle["delta_m"]),epsabs=max(abs(barrier)*1e-11,1e-14),
            epsrel=1e-10,limit=200)[0])
        relative_error=abs(quadrature-barrier)/max(abs(quadrature),abs(barrier),1e-300)
    return {"state": "METASTABLE", "barrier_J_m2": max(barrier, 0.),
            "root_gap_m": float(saddle["delta_m"])-float(connected["delta_m"]),
            "connected": connected, "saddle": saddle, "stationary": points,
            "barrier_quadrature_J_m2":quadrature,
            "barrier_quadrature_relative_error":relative_error}


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


def metastable_window(shape: str, thermal: CohesiveThermalLaw,
                      temperature_K: float,
                      environment_stiffness_Pa_per_m: float) -> dict[str, Any]:
    """Loads between separated-branch birth and bonded-branch loss."""
    kenv = float(environment_stiffness_Pa_per_m)
    coordinates = shape_coordinates(shape, thermal, temperature_K)
    peak_d = coordinates["peak_separation_m"]
    cutoff = coordinates["completion_cutoff_m"]
    if shape == BILINEAR:
        peak = thermal.t_hat(temperature_K)
        fold_P = peak + kenv*peak_d
        birth_P = kenv*cutoff
        state = "METASTABLE_WINDOW" if 0. < birth_P < fold_P else "NO_METASTABLE_WINDOW"
        return {"state": state, "birth_load_Pa": birth_P,
                "fold_load_Pa": fold_P, "birth_separation_m": cutoff,
                "fold_separation_m": peak_d,
                "load_ratio": fold_P/birth_P if birth_P > 0. else math.inf}
    grid = np.linspace(peak_d*(1.+1e-10), cutoff*(1.-1e-10), 4097)
    derivative = np.asarray(cohesive_shape_response(
        shape, grid, thermal, temperature_K)[1])+kenv
    indices = np.flatnonzero(derivative[:-1]*derivative[1:] < 0.)
    extrema = []
    for index in indices:
        left, right = float(grid[index]), float(grid[index+1])
        root = float(brentq(lambda d: float(cohesive_shape_response(
            shape, d, thermal, temperature_K)[1])+kenv, left, right,
            xtol=max(cutoff*1e-14, 1e-24), rtol=1e-13))
        traction = float(cohesive_shape_response(shape, root, thermal,
                                                  temperature_K)[0])
        load = traction+kenv*root
        kind = "MAXIMUM" if derivative[index] > 0. else "MINIMUM"
        extrema.append({"kind": kind, "separation_m": root, "load_Pa": load})
    maxima = [point for point in extrema if point["kind"] == "MAXIMUM"]
    minima = [point for point in extrema if point["kind"] == "MINIMUM"]
    if not maxima or not minima:
        return {"state": "NO_METASTABLE_WINDOW", "extrema": extrema,
                "birth_load_Pa": None, "fold_load_Pa": None,
                "load_ratio": None}
    # The connected bonded basin loses stability at the first maximum.  The
    # first following minimum is where its directly adjacent separated branch
    # is born; later extrema belong to secondary tail branches and must not
    # silently replace that accessibility window.
    fold = maxima[0]
    birth = next((point for point in minima
                  if point["separation_m"] > fold["separation_m"]), minima[0])
    valid = 0. < float(birth["load_Pa"]) < float(fold["load_Pa"])
    return {"state": "METASTABLE_WINDOW" if valid else "NO_METASTABLE_WINDOW",
            "birth_load_Pa": birth["load_Pa"], "fold_load_Pa": fold["load_Pa"],
            "birth_separation_m": birth["separation_m"],
            "fold_separation_m": fold["separation_m"],
            "load_ratio": float(fold["load_Pa"])/float(birth["load_Pa"]),
            "extrema": extrema}


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
        sensitivity_identity = None
    else:
        barrier = float(density)*float(activation_area_m2)
        barrier_kBT = barrier/(KB_EV_PER_K*EV_TO_J*float(temperature_K))
        rate = float(attempt_frequency_s)*math.exp(max(-barrier_kBT, -745.))
        generalized_load_Pa = float(environment_stiffness_Pa_per_m)*imposed
        sensitivity_identity = (float(activation_area_m2)*generalized_load_Pa*
            float(landscape["root_gap_m"])/(KB_EV_PER_K*EV_TO_J*float(temperature_K)))
    return {**landscape, "K_MPa_sqrt_m": float(K_MPa_sqrt_m),
            "imposed_m": imposed, "elementary_rate_s-1": rate,
            "barrier_kBT": barrier_kBT,
            "sensitivity_identity": sensitivity_identity}


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
        identity=row.get("sensitivity_identity")
        row["sensitivity_identity_relative_error"]=(None if identity is None or
            not np.isfinite(slopes[index]) else abs(float(slopes[index])-float(identity))/
            max(abs(float(slopes[index])),abs(float(identity)),1e-300))
    useful = positive & (slopes >= 2.) & (slopes <= 8.) & (per_cycle >= 1e-14) & (per_cycle <= 1e-1)
    strict = positive & (slopes >= 3.4) & (slopes <= 4.6) & (per_cycle >= 1e-14) & (per_cycle <= 1e-1)

    def connected_width(mask):
        best_K=0.;best_rate=0.;best_bounds=None;start=None
        for index, flag in enumerate(np.r_[mask, False]):
            if flag and start is None:start=index
            elif not flag and start is not None:
                end=index-1
                width_K=math.log10(K[end]/K[start]) if end>start else 0.
                width_rate=math.log10(rates[end]/rates[start]) if end>start else 0.
                if width_rate>best_rate:
                    best_K,best_rate,best_bounds=width_K,width_rate,(float(K[start]),float(K[end]))
                start=None
        return best_K,best_rate,best_bounds
    useful_K,useful_rate,useful_bounds=connected_width(useful)
    strict_K,strict_rate,strict_bounds=connected_width(strict)
    identity_errors=[row["sensitivity_identity_relative_error"] for row in rows[1:-1]
                     if row["sensitivity_identity_relative_error"] is not None]
    return {"rows": rows, "moderate_connected_width_log10K": useful_K,
            "moderate_connected_width_log10rate": useful_rate,
            "moderate_connected_bounds_MPa_sqrt_m": useful_bounds,
            "strict_connected_width_log10K": strict_K,
            "strict_connected_width_log10rate": strict_rate,
            "strict_connected_bounds_MPa_sqrt_m": strict_bounds,
            "strict_internal_reference_pass": bool(strict_rate >= 2.0),
            "sensitivity_identity_max_relative_error":max(identity_errors,default=None),
            "max_moderate_points": int(np.count_nonzero(useful)),
            "peak_rate_approximation": True}


def cyclic_rate(Kmax_MPa_sqrt_m: float, environment_stiffness_Pa_per_m: float,
                thermal: CohesiveThermalLaw, temperature_K: float, shape: str,
                activation_area_m2: float, attempt_frequency_s: float,
                R: float = .1, frequency_Hz: float = 1000.,
                increment_m: float = 1.e-6, phases: int = 128,
                tip_radius_m: float = DEFAULT_TIP_RADIUS_M,
                integration: str = "finite_interval_gauss") -> dict[str, Any]:
    """Frozen waveform integral with explicit accessibility-boundary support.

    ``midpoint`` is retained only to reproduce archived phase-grid references.
    Production selection uses Gauss integration on intervals split at the
    metastable birth/fold crossings, so a narrow peak-accessible interval
    cannot disappear between uniform phase samples.
    """
    count=max(int(phases),8)
    mean=.5*(1.+float(R));amplitude=.5*(1.-float(R))
    if integration == "midpoint":
        angles=(np.arange(count,dtype=float)+.5)*2.*math.pi/count
        weights=np.full(count,1./count)
    elif integration == "finite_interval_gauss":
        window=metastable_window(shape,thermal,temperature_K,
            environment_stiffness_Pa_per_m)
        boundaries=[0.,math.pi,2.*math.pi]
        if window.get("state") == "METASTABLE_WINDOW" and amplitude > 0.:
            for load_key in ("birth_load_Pa","fold_load_Pa"):
                imposed=float(window[load_key])/float(environment_stiffness_Pa_per_m)
                boundary_K=K_from_imposed(imposed,environment_stiffness_Pa_per_m,
                    tip_radius_m)
                cosine=(boundary_K/float(Kmax_MPa_sqrt_m)-mean)/amplitude
                if -1. < cosine < 1.:
                    theta=math.acos(cosine);boundaries.extend((theta,2.*math.pi-theta))
        boundaries=sorted(set(boundaries))
        nodes,base_weights=np.polynomial.legendre.leggauss(count)
        angle_parts=[];weight_parts=[]
        for left,right in zip(boundaries[:-1],boundaries[1:]):
            midpoint=.5*(left+right)
            Kmid=float(Kmax_MPa_sqrt_m)*(mean+amplitude*math.cos(midpoint))
            check=opening_rate(Kmid,environment_stiffness_Pa_per_m,thermal,
                temperature_K,shape,activation_area_m2,attempt_frequency_s,
                tip_radius_m)
            if check["state"] != "METASTABLE":
                continue
            angle_parts.append(midpoint+.5*(right-left)*nodes)
            weight_parts.append(.5*(right-left)*base_weights/(2.*math.pi))
        angles=(np.concatenate(angle_parts) if angle_parts else np.asarray([],float))
        weights=(np.concatenate(weight_parts) if weight_parts else np.asarray([],float))
    else:
        raise ValueError(f"unsupported cyclic integration method: {integration}")
    factors=mean+amplitude*np.cos(angles)
    evaluations=[opening_rate(float(Kmax_MPa_sqrt_m)*float(factor),
        environment_stiffness_Pa_per_m,thermal,temperature_K,shape,
        activation_area_m2,attempt_frequency_s,tip_radius_m) for factor in factors]
    rates=np.asarray([row["elementary_rate_s-1"] for row in evaluations])
    action=float(np.sum(weights*rates)/float(frequency_Hz))
    sensitivities=np.asarray([0. if row["sensitivity_identity"] is None
        else row["sensitivity_identity"] for row in evaluations])
    weighted_rate=weights*rates
    weighted=(float(np.sum(weighted_rate*sensitivities)/np.sum(weighted_rate))
              if float(np.sum(weighted_rate))>0. else None)
    return {"Kmax_MPa_sqrt_m":float(Kmax_MPa_sqrt_m),
        "opening_action_per_cycle":action,
        "ensemble_rate_m_per_cycle":float(increment_m)*action,
        "hazard_weighted_sensitivity_identity":weighted,
        "phase_count":len(evaluations),"requested_phase_count":count,
        "accessible_phase_count":sum(row["state"]=="METASTABLE" for row in evaluations),
        "accessible_phase_fraction":float(np.sum(weights)),
        "integration_method":integration,
        "peak_barrier_kBT":min((row["barrier_kBT"] for row in evaluations
            if row["barrier_kBT"] is not None),default=None)}


__all__ = ["branch_loss", "metastable_window", "connected_barrier", "imposed_from_K",
           "K_from_imposed", "opening_rate", "rate_curve", "cyclic_rate",
           "shape_coordinates"]
