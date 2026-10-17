"""Exponential cohesive energetics and the associated Arrhenius barrier.

This module is deliberately independent of the R10 EXP-floor implementation.
It provides the elementary, per-physical-patch separation landscape.  Site
multiplicity and the inherited cooperative completion law remain owners of the
calling layer and are never folded into :class:`ExponentialCohesiveBarrier`.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Any

import numpy as np
from scipy.special import lambertw
from scipy.optimize import brentq

from .material_manifest import KB_EV_PER_K

EV_TO_J = 1.602176634e-19
EULER = math.e

FAMILY = "D_COHESIVE_ARRHENIUS_EXP_PLASTICITY"
EXP_FLOOR_REFERENCE = "EXP_FLOOR_REFERENCE"
COHESIVE_BARRIER_ONLY = "COHESIVE_BARRIER_ONLY"
COHESIVE_KINETIC_CZM = "COHESIVE_KINETIC_CZM"
COHESIVE_RATE_INDEPENDENT_CONTROL = "COHESIVE_RATE_INDEPENDENT_CONTROL"
EXECUTION_IDENTITIES = frozenset({
    EXP_FLOOR_REFERENCE, COHESIVE_BARRIER_ONLY, COHESIVE_KINETIC_CZM,
    COHESIVE_RATE_INDEPENDENT_CONTROL,
})


@dataclass(frozen=True)
class CohesiveThermalLaw:
    """One separable cohesive free-energy scale with fixed ``delta0``."""

    Gamma_ref_J_m2: float
    t_hat_ref_Pa: float
    S_Gamma_J_m2_K: float = 0.0
    T_ref_K: float = 300.0

    def __post_init__(self) -> None:
        if not math.isfinite(self.Gamma_ref_J_m2) or self.Gamma_ref_J_m2 <= 0.0:
            raise ValueError("Gamma_ref_J_m2 must be finite and positive")
        if not math.isfinite(self.t_hat_ref_Pa) or self.t_hat_ref_Pa <= 0.0:
            raise ValueError("t_hat_ref_Pa must be finite and positive")
        if not math.isfinite(self.S_Gamma_J_m2_K):
            raise ValueError("S_Gamma_J_m2_K must be finite")

    @property
    def delta0_m(self) -> float:
        return self.Gamma_ref_J_m2 / (EULER * self.t_hat_ref_Pa)

    def gamma(self, temperature_K: float) -> float:
        value = self.Gamma_ref_J_m2 - (
            float(temperature_K) - self.T_ref_K
        ) * self.S_Gamma_J_m2_K
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError("cohesive Gamma(T) left its positive raw domain")
        return value

    def t_hat(self, temperature_K: float) -> float:
        return self.gamma(temperature_K) / (EULER * self.delta0_m)

    def initial_stiffness(self, temperature_K: float) -> float:
        return self.gamma(temperature_K) / self.delta0_m**2

    def validate_range(self, temperatures_K) -> "CohesiveThermalLaw":
        for temperature in temperatures_K:
            self.gamma(float(temperature))
        return self


def potential_J_m2(delta_m, thermal: CohesiveThermalLaw, temperature_K: float):
    delta = np.asarray(delta_m, dtype=float)
    x = np.maximum(delta, 0.0) / thermal.delta0_m
    # -expm1(-x)-x*exp(-x) is stable at small x.
    return thermal.gamma(temperature_K) * (-np.expm1(-x) - x * np.exp(-x))


def traction_Pa(delta_m, thermal: CohesiveThermalLaw, temperature_K: float):
    delta = np.asarray(delta_m, dtype=float)
    x = np.maximum(delta, 0.0) / thermal.delta0_m
    return thermal.gamma(temperature_K) / thermal.delta0_m * x * np.exp(-x)


def tangent_Pa_per_m(delta_m, thermal: CohesiveThermalLaw,
                     temperature_K: float):
    delta = np.asarray(delta_m, dtype=float)
    x = np.maximum(delta, 0.0) / thermal.delta0_m
    return (thermal.gamma(temperature_K) / thermal.delta0_m**2
            * (1.0 - x) * np.exp(-x))


def stationary_points(stress_Pa: float, thermal: CohesiveThermalLaw,
                      temperature_K: float) -> dict[str, Any]:
    """Return stationary states without inventing endpoints or contact roots."""
    stress = float(stress_Pa)
    peak = thermal.t_hat(temperature_K)
    if stress < 0.0:
        return {"state": "INADMISSIBLE_COMPRESSION", "s": stress / peak,
                "x_min": None, "x_saddle": None}
    if stress == 0.0:
        return {"state": "ZERO_TRACTION_LIMIT", "s": 0.0,
                "x_min": 0.0, "x_saddle": math.inf}
    s = stress / peak
    if s >= 1.0:
        return {"state": "BARRIERLESS_SPINODAL", "s": s,
                "x_min": 1.0, "x_saddle": 1.0}
    z = -s / EULER
    xm = float(-lambertw(z, 0).real)
    xs = float(-lambertw(z, -1).real)
    return {"state": "METASTABLE", "s": s, "x_min": xm, "x_saddle": xs}


def _dimensionless_barrier(s: float, xm: float, xs: float) -> float:
    eps = 1.0 - s
    if eps <= 1.0e-7:
        return 4.0 * math.sqrt(2.0) / (3.0 * EULER) * eps**1.5
    return max((1.0 + xm) * math.exp(-xm)
               - (1.0 + xs) * math.exp(-xs)
               - s / EULER * (xs - xm), 0.0)


def traction_barrier(stress_Pa: float, thermal: CohesiveThermalLaw,
                     temperature_K: float, activation_area_m2: float
                     ) -> dict[str, Any]:
    area = float(activation_area_m2)
    if not math.isfinite(area) or area <= 0.0:
        raise ValueError("activation_area_m2 must be finite and positive")
    roots = stationary_points(stress_Pa, thermal, temperature_K)
    gamma = thermal.gamma(temperature_K)
    state = roots["state"]
    if state == "INADMISSIBLE_COMPRESSION":
        barrier = math.nan; volume = math.nan
    elif state == "ZERO_TRACTION_LIMIT":
        barrier = area * gamma; volume = math.inf
    elif state == "BARRIERLESS_SPINODAL":
        barrier = 0.0; volume = 0.0
    else:
        xm = float(roots["x_min"]); xs = float(roots["x_saddle"])
        barrier = area * gamma * _dimensionless_barrier(roots["s"], xm, xs)
        volume = area * thermal.delta0_m * (xs - xm)
    return {**roots, "barrier_J": barrier, "barrier_eV": barrier / EV_TO_J,
            "activation_volume_m3": volume, "Gamma_J_m2": gamma,
            "t_hat_Pa": thermal.t_hat(temperature_K),
            "delta0_m": thermal.delta0_m}


def finite_compliance_landscape(imposed_delta_m: float,
                                environment_stiffness_Pa_per_m: float,
                                thermal: CohesiveThermalLaw,
                                temperature_K: float) -> dict[str, Any]:
    """Track stationary points of ``psi + k_env/2*(Delta-delta)^2``.

    This is a local condensation reference, not a fit coordinate.  A single
    stable root in a convex combined landscape is reported as
    ``NO_ESCAPE_SADDLE`` and is never converted into a zero barrier.
    """
    kenv = float(environment_stiffness_Pa_per_m)
    imposed = float(imposed_delta_m)
    if kenv <= 0.0 or imposed < 0.0:
        raise ValueError("finite compliance requires positive stiffness and nonnegative Delta")
    d0 = thermal.delta0_m
    gamma = thermal.gamma(temperature_K)
    kappa = kenv * d0**2 / gamma
    D = imposed / d0

    def residual(x):
        return x * math.exp(-x) + kappa * (x - D)

    xmax = max(40.0, D + 20.0)
    grid = np.unique(np.r_[np.linspace(0.0, min(xmax, 6.0), 1201),
                           np.geomspace(6.0, xmax, 600) if xmax > 6.0 else []])
    values = np.asarray([residual(float(x)) for x in grid])
    roots = []
    for a, b, fa, fb in zip(grid[:-1], grid[1:], values[:-1], values[1:]):
        if fa == 0.0: roots.append(float(a))
        if fa * fb < 0.0: roots.append(float(brentq(residual, a, b)))
    if values[-1] == 0.0: roots.append(float(grid[-1]))
    roots = sorted({round(x, 13) for x in roots})
    stationary = []
    for x in roots:
        curvature = gamma/d0**2 * ((1.0-x)*math.exp(-x) + kappa)
        energy = gamma*(-math.expm1(-x)-x*math.exp(-x)) + 0.5*kenv*(imposed-x*d0)**2
        stationary.append({"x": x, "delta_m": x*d0, "energy_J_m2": energy,
                           "curvature_Pa_per_m": curvature,
                           "kind": "MINIMUM" if curvature > 0.0 else "SADDLE"})
    minima = [p for p in stationary if p["kind"] == "MINIMUM"]
    saddles = [p for p in stationary if p["kind"] == "SADDLE"]
    if not minima:
        state = "BARRIERLESS_SPINODAL"
        barrier = 0.0
    elif not saddles:
        state = "NO_ESCAPE_SADDLE"
        barrier = None
    else:
        connected = minima[0]
        later = [p for p in saddles if p["x"] > connected["x"]]
        if not later:
            state = "NO_ESCAPE_SADDLE"; barrier = None
        else:
            state = "METASTABLE"; barrier = later[0]["energy_J_m2"]-connected["energy_J_m2"]
    return {"state": state, "barrier_J_m2": barrier, "stationary": stationary,
            "kappa": kappa, "imposed_delta_over_delta0": D,
            "environment_stiffness_Pa_per_m": kenv}


def bilinear_barrier_J(stress_ratio, Gamma_J_m2: float,
                       activation_area_m2: float):
    s = np.clip(np.asarray(stress_ratio, dtype=float), 0.0, 1.0)
    return activation_area_m2 * Gamma_J_m2 * (1.0 - s)**2


def morse_barrier_J(stress_ratio, Gamma_J_m2: float,
                    activation_area_m2: float):
    s = np.clip(np.asarray(stress_ratio, dtype=float), 0.0, 1.0)
    r = np.sqrt(1.0 - s)
    safe_r = np.minimum(r, 1.0 - np.finfo(float).eps)
    logterm = np.log1p(safe_r) - np.log1p(-safe_r)
    value = r - 0.5 * s * logterm
    value = np.where(s == 0.0, 1.0, value)
    return activation_area_m2 * Gamma_J_m2 * np.maximum(value, 0.0)


@dataclass(frozen=True)
class ExponentialCohesiveBarrier:
    """Elementary per-site barrier adapter consumed by the R10 engine."""

    thermal: CohesiveThermalLaw
    activation_area_m2: float
    attempt_frequency_s: float = 1.0e12
    execution_identity: str = COHESIVE_BARRIER_ONLY

    def __post_init__(self) -> None:
        if self.execution_identity not in {
            COHESIVE_BARRIER_ONLY, COHESIVE_KINETIC_CZM,
        }:
            raise ValueError("unsupported cohesive Arrhenius execution identity")
        if self.activation_area_m2 <= 0.0 or self.attempt_frequency_s <= 0.0:
            raise ValueError("activation area and attempt frequency must be positive")

    @property
    def sigc0_Pa(self) -> float:
        return self.thermal.t_hat(self.thermal.T_ref_K)

    @property
    def Tref_K(self) -> float:
        return self.thermal.T_ref_K

    @property
    def G00_eV(self) -> float:
        return self.activation_area_m2 * self.thermal.Gamma_ref_J_m2 / EV_TO_J

    def values_eV(self, stress_Pa, temperature_K: float):
        stress = np.asarray(stress_Pa, dtype=float)
        out = np.empty_like(stress)
        for index, value in np.ndenumerate(stress):
            result = traction_barrier(float(value), self.thermal, temperature_K,
                                      self.activation_area_m2)
            out[index] = result["barrier_eV"]
        return out

    def rate(self, stress_Pa, temperature_K: float):
        stress = np.asarray(stress_Pa, dtype=float)
        G = self.values_eV(stress, temperature_K)
        rates = self.attempt_frequency_s * np.exp(np.clip(
            -G / max(KB_EV_PER_K * float(temperature_K), 1.0e-30), -700.0, 0.0,
        ))
        # The positive-traction zero limit is mathematically finite, but exact
        # contact/compression is not an admissible irreversible opening event.
        return np.where(stress > 0.0, rates, 0.0)

    def activation_volume_m3(self, stress_Pa: float,
                             temperature_K: float) -> float:
        return float(traction_barrier(
            stress_Pa, self.thermal, temperature_K, self.activation_area_m2,
        )["activation_volume_m3"])

    def entropy_J_K(self, stress_Pa: float, temperature_K: float) -> float:
        h = max(1.0e-3, 1.0e-5 * max(abs(float(temperature_K)), 1.0))
        gp = traction_barrier(stress_Pa, self.thermal, temperature_K + h,
                              self.activation_area_m2)["barrier_J"]
        gm = traction_barrier(stress_Pa, self.thermal, temperature_K - h,
                              self.activation_area_m2)["barrier_J"]
        return -(gp - gm) / (2.0 * h)

    def as_dict(self) -> dict[str, Any]:
        return {"kind": "exponential_cohesive_arrhenius",
                "family": FAMILY, "thermal": asdict(self.thermal),
                "activation_area_m2": self.activation_area_m2,
                "attempt_frequency_s": self.attempt_frequency_s,
                "execution_identity": self.execution_identity}


@dataclass(frozen=True)
class RateIndependentCohesiveControl:
    """Explicitly non-Arrhenius control using the same cohesive envelope."""

    thermal: CohesiveThermalLaw
    execution_identity: str = COHESIVE_RATE_INDEPENDENT_CONTROL

    def should_rupture(self, separation_m: float, temperature_K: float) -> bool:
        # The control fails at the envelope maximum (x=1).  It is intentionally
        # not exposed through ``rate`` and cannot be mistaken for Arrhenius data.
        self.thermal.gamma(temperature_K)
        return float(separation_m) >= self.thermal.delta0_m

    def traction(self, separation_m, temperature_K: float):
        return traction_Pa(separation_m, self.thermal, temperature_K)

    def rate(self, *_args, **_kwargs):
        raise RuntimeError("rate-independent cohesive control has no Arrhenius rate")


__all__ = [
    "COHESIVE_BARRIER_ONLY", "COHESIVE_KINETIC_CZM",
    "COHESIVE_RATE_INDEPENDENT_CONTROL", "EXECUTION_IDENTITIES",
    "EXP_FLOOR_REFERENCE", "FAMILY", "CohesiveThermalLaw",
    "ExponentialCohesiveBarrier", "RateIndependentCohesiveControl",
    "bilinear_barrier_J", "morse_barrier_J",
    "potential_J_m2", "stationary_points", "tangent_Pa_per_m",
    "traction_Pa", "traction_barrier", "finite_compliance_landscape",
]
