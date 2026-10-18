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
from scipy.special import beta as beta_function, betainc, lambertw
from scipy.optimize import brentq

from .material_manifest import KB_EV_PER_K

EV_TO_J = 1.602176634e-19
EULER = math.e

FAMILY = "D_COHESIVE_ARRHENIUS_EXP_PLASTICITY"
EXP_FLOOR_REFERENCE = "EXP_FLOOR_REFERENCE"
COHESIVE_BARRIER_ONLY = "COHESIVE_BARRIER_ONLY"
COHESIVE_KINETIC_CZM = "COHESIVE_KINETIC_CZM"
COHESIVE_RATE_INDEPENDENT_CONTROL = "COHESIVE_RATE_INDEPENDENT_CONTROL"
COHESIVE_PEAK_CUTOFF_CONTROL = "COHESIVE_PEAK_CUTOFF_CONTROL"
COHESIVE_ATHERMAL_MECHANICAL = "COHESIVE_ATHERMAL_MECHANICAL"
COHESIVE_MIXED_MECHANICAL_THERMAL = "COHESIVE_MIXED_MECHANICAL_THERMAL"
EXECUTION_IDENTITIES = frozenset({
    EXP_FLOOR_REFERENCE, COHESIVE_BARRIER_ONLY, COHESIVE_KINETIC_CZM,
    COHESIVE_RATE_INDEPENDENT_CONTROL, COHESIVE_PEAK_CUTOFF_CONTROL,
    COHESIVE_ATHERMAL_MECHANICAL, COHESIVE_MIXED_MECHANICAL_THERMAL,
})

OPENING_ALTERNATIVES_FAMILY = "D_MECHANICAL_MIXED_CZM_EXP_PLASTICITY"
SMOOTH_EXPONENTIAL = "smooth_exponential"
BILINEAR = "bilinear"
COMPACT_INDEPENDENT = "compact_independent"


@dataclass(frozen=True)
class CohesiveThermalLaw:
    """One separable cohesive free-energy scale with fixed ``delta0``."""

    Gamma_ref_J_m2: float
    t_hat_ref_Pa: float
    S_Gamma_J_m2_K: float = 0.0
    T_ref_K: float = 300.0
    # Opt-in research coordinates for the compact, finite-support cohesive
    # potential.  They are inert for the canonical exponential/bilinear laws.
    compact_b: float = 2.0
    compact_c: float = 0.0
    compact_variant: str = "beta_polynomial"
    compact_rise_fraction: float = 0.1
    compact_fall_fraction: float = 0.85
    compact_plateau_drop: float = 0.05

    def __post_init__(self) -> None:
        if not math.isfinite(self.Gamma_ref_J_m2) or self.Gamma_ref_J_m2 <= 0.0:
            raise ValueError("Gamma_ref_J_m2 must be finite and positive")
        if not math.isfinite(self.t_hat_ref_Pa) or self.t_hat_ref_Pa <= 0.0:
            raise ValueError("t_hat_ref_Pa must be finite and positive")
        if not math.isfinite(self.S_Gamma_J_m2_K):
            raise ValueError("S_Gamma_J_m2_K must be finite")
        if not math.isfinite(self.compact_b) or self.compact_b < 1.0:
            raise ValueError("compact_b must be finite and at least one")
        if not math.isfinite(self.compact_c) or self.compact_c <= -1.0:
            raise ValueError("compact_c must be finite and greater than -1")
        if self.compact_variant not in {"beta_polynomial", "rounded_plateau"}:
            raise ValueError("unsupported compact_variant")
        if not (0.0 < self.compact_rise_fraction < self.compact_fall_fraction < 1.0):
            raise ValueError("compact rise/fall fractions must obey 0 < rise < fall < 1")
        if not (0.0 < self.compact_plateau_drop < 1.0):
            raise ValueError("compact_plateau_drop must lie between zero and one")

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
    if s > 1.0:
        return {"state": "ABOVE_SPINODAL_NO_STATIONARY", "s": s,
                "x_min": None, "x_saddle": None}
    if s == 1.0:
        return {"state": "COALESCED_SPINODAL", "s": s,
                "x_min": 1.0, "x_saddle": 1.0}
    eps = 1.0-s
    if eps <= 1.0e-5:
        # Corless branch-point expansion.  Evaluating W_0 and W_-1
        # independently loses their separation (and hence V*) near -1/e.
        p = math.sqrt(2.0*eps)
        p2=p*p; p3=p2*p; p4=p2*p2; p5=p4*p
        xm = 1.0-p+p2/3.0-11.0*p3/72.0+43.0*p4/540.0-769.0*p5/17280.0
        xs = 1.0+p+p2/3.0+11.0*p3/72.0+43.0*p4/540.0+769.0*p5/17280.0
    else:
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
    elif state in {"COALESCED_SPINODAL", "ABOVE_SPINODAL_NO_STATIONARY"}:
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
                                temperature_K: float,
                                connected_state: dict[str, Any] | None = None
                                ) -> dict[str, Any]:
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

    convex = kappa >= math.exp(-2.0)
    turning_points = []
    fold_displacements = []
    if convex:
        brackets = [(0.0, max(D+2.0, 4.0))]
    else:
        x1 = 1.0-float(lambertw(-EULER*kappa, 0).real)
        x2 = 1.0-float(lambertw(-EULER*kappa, -1).real)
        turning_points = [x1,x2]
        fold_displacements = [x1*x1/(x1-1.0),x2*x2/(x2-1.0)]
        brackets = [(0.0,x1),(x1,x2),(x2,max(D+2.0,x2+4.0))]
    roots=[]
    scale=max(1.0,kappa*max(D,1.0))
    zero_tol=128.0*np.finfo(float).eps*scale
    for left,right in brackets:
        fl=residual(left);fr=residual(right)
        if abs(fl)<=zero_tol: roots.append(left)
        if fl*fr < 0.0:
            roots.append(float(brentq(residual,left,right,xtol=1e-14,rtol=4*np.finfo(float).eps)))
        if abs(fr)<=zero_tol: roots.append(right)
    roots=sorted(roots)
    unique=[]
    for root in roots:
        if not unique or abs(root-unique[-1])>2e-12*max(1.0,abs(root)):
            unique.append(root)
    roots=unique
    stationary = []
    for x in roots:
        curvature = gamma/d0**2 * ((1.0-x)*math.exp(-x) + kappa)
        energy = gamma*(-math.expm1(-x)-x*math.exp(-x)) + 0.5*kenv*(imposed-x*d0)**2
        stationary.append({"x": x, "delta_m": x*d0, "energy_J_m2": energy,
                           "curvature_Pa_per_m": curvature,
                           "kind": ("FOLD" if abs(curvature) <= 1e-10*gamma/d0**2
                                    else "MINIMUM" if curvature > 0.0 else "SADDLE")})
    minima = [p for p in stationary if p["kind"] == "MINIMUM"]
    saddles = [p for p in stationary if p["kind"] == "SADDLE"]
    prior_branch=(connected_state or {}).get("branch","INTACT")
    intact=minima[0] if minima else None
    separated=minima[-1] if len(minima)>1 else None
    bonded_loss_D=(max(fold_displacements) if fold_displacements else None)
    intact_lost=bool(not convex and bonded_loss_D is not None and D>bonded_loss_D)
    if convex:
        connected=minima[0]
        state="CONVEX_CONNECTED";barrier=None;branch="CONNECTED"
    elif prior_branch == "SEPARATED":
        connected=separated or (minima[-1] if minima else None)
        state="SEPARATED_BRANCH";barrier=None;branch="SEPARATED"
    elif intact_lost:
        connected=minima[-1] if minima else None
        state="INTACT_BRANCH_LOST_TO_SEPARATED";barrier=0.0;branch="SEPARATED"
    elif intact is None:
        connected=None;state="NUMERICAL_ROOT_FAILURE";barrier=None;branch=prior_branch
    else:
        connected=intact;branch="INTACT"
        later=[p for p in saddles if p["x"]>connected["x"]]
        if later:
            state="METASTABLE";barrier=later[0]["energy_J_m2"]-connected["energy_J_m2"]
        else:
            state="NO_ESCAPE_SADDLE";barrier=None
    next_state={"branch":branch,"x":None if connected is None else connected["x"],
                "imposed_delta_over_delta0":D,"intact_branch_lost":intact_lost}
    return {"state": state, "barrier_J_m2": barrier, "stationary": stationary,
            "kappa": kappa, "imposed_delta_over_delta0": D,
            "environment_stiffness_Pa_per_m": kenv,"convex":convex,
            "turning_points_x":turning_points,"fold_displacements_over_delta0":fold_displacements,
            "bonded_branch_loss_D":bonded_loss_D,"connected":connected,
            "connected_state":next_state}


def bilinear_barrier_J(stress_ratio, Gamma_J_m2: float,
                       activation_area_m2: float):
    s = np.clip(np.asarray(stress_ratio, dtype=float), 0.0, 1.0)
    return activation_area_m2 * Gamma_J_m2 * (1.0 - s)**2


@dataclass(frozen=True)
class BilinearCohesiveLaw:
    """Bilinear law matched to an exponential law in work, peak and stiffness."""

    thermal: CohesiveThermalLaw

    @property
    def peak_separation_m(self) -> float:
        return self.thermal.Gamma_ref_J_m2 / (
            EULER**2 * self.thermal.t_hat_ref_Pa
        )

    def peak_separation(self, temperature_K: float) -> float:
        # Gamma/t_hat remains fixed under CohesiveThermalLaw's coupled scaling.
        self.thermal.gamma(temperature_K)
        return self.peak_separation_m

    def final_separation(self, temperature_K: float) -> float:
        return 2.0 * self.thermal.gamma(temperature_K) / self.thermal.t_hat(temperature_K)

    def traction(self, separation_m, temperature_K: float):
        d = np.maximum(np.asarray(separation_m, dtype=float), 0.0)
        dp = self.peak_separation(temperature_K)
        df = self.final_separation(temperature_K)
        peak = self.thermal.t_hat(temperature_K)
        return np.where(d <= dp, peak*d/dp,
                        np.where(d < df, peak*(df-d)/(df-dp), 0.0))

    def tangent(self, separation_m, temperature_K: float):
        d = np.maximum(np.asarray(separation_m, dtype=float), 0.0)
        dp = self.peak_separation(temperature_K)
        df = self.final_separation(temperature_K)
        peak = self.thermal.t_hat(temperature_K)
        return np.where(d < dp, peak/dp,
                        np.where(d < df, -peak/(df-dp), 0.0))

    def potential(self, separation_m, temperature_K: float):
        d = np.maximum(np.asarray(separation_m, dtype=float), 0.0)
        dp = self.peak_separation(temperature_K)
        df = self.final_separation(temperature_K)
        peak = self.thermal.t_hat(temperature_K)
        elastic = 0.5*peak*d*d/dp
        up = 0.5*peak*dp + peak*((d-dp) - 0.5*(d-dp)**2/(df-dp))
        return np.where(d <= dp, elastic,
                        np.where(d < df, up, self.thermal.gamma(temperature_K)))


@dataclass(frozen=True)
class CompactIndependentCohesiveLaw:
    """Finite-support single-potential research law.

    Its normalized traction is ``x (1-x)^b (1+c x)`` on ``0 <= x <= 1``.
    The support is chosen so that the same potential has exactly the declared
    work and peak traction.  ``b`` and ``c`` independently alter the initial
    stiffness and softening distribution without introducing a Paris law or a
    separate stress-dependent activation barrier.
    """

    thermal: CohesiveThermalLaw

    @property
    def b(self) -> float:
        return float(self.thermal.compact_b)

    @property
    def c(self) -> float:
        return float(self.thermal.compact_c)

    @property
    def peak_x(self) -> float:
        if self.thermal.compact_variant == "rounded_plateau":
            return float(self.thermal.compact_rise_fraction)
        b, c = self.b, self.c
        if abs(c) < 1.0e-14:
            return 1.0 / (b + 1.0)
        # g'(x)/(1-x)^(b-1) = 1 +(2c-b-1)x-c(b+2)x^2.
        roots = np.roots([-c * (b + 2.0), 2.0 * c - b - 1.0, 1.0])
        admissible = [float(r.real) for r in roots
                      if abs(float(r.imag)) < 1.0e-10 and 0.0 < r.real < 1.0]
        if len(admissible) != 1:
            raise ValueError("compact cohesive shape must have one interior peak")
        return admissible[0]

    @property
    def peak_shape(self) -> float:
        if self.thermal.compact_variant == "rounded_plateau":
            return 1.0
        x = self.peak_x
        return x * (1.0 - x) ** self.b * (1.0 + self.c * x)

    @property
    def shape_integral(self) -> float:
        if self.thermal.compact_variant == "rounded_plateau":
            rise = float(self.thermal.compact_rise_fraction)
            fall = float(self.thermal.compact_fall_fraction)
            drop = float(self.thermal.compact_plateau_drop)
            return (2.0*rise/3.0 + (fall-rise)*(1.0-drop/2.0)
                    + (1.0-drop)*(1.0-fall)/3.0)
        return float(beta_function(2.0, self.b + 1.0)
                     + self.c * beta_function(3.0, self.b + 1.0))

    def final_separation(self, temperature_K: float) -> float:
        gamma = self.thermal.gamma(temperature_K)
        peak = self.thermal.t_hat(temperature_K)
        return gamma * self.peak_shape / (peak * self.shape_integral)

    def peak_separation(self, temperature_K: float) -> float:
        return self.peak_x * self.final_separation(temperature_K)

    def initial_stiffness(self, temperature_K: float) -> float:
        derivative0 = (2.0/float(self.thermal.compact_rise_fraction)
                       if self.thermal.compact_variant == "rounded_plateau" else 1.0)
        return (self.thermal.t_hat(temperature_K) * derivative0 /
                (self.final_separation(temperature_K) * self.peak_shape))

    def traction(self, separation_m, temperature_K: float):
        d = np.maximum(np.asarray(separation_m, dtype=float), 0.0)
        dc = self.final_separation(temperature_K)
        x = np.clip(d / dc, 0.0, 1.0)
        if self.thermal.compact_variant == "rounded_plateau":
            rise=float(self.thermal.compact_rise_fraction);fall=float(self.thermal.compact_fall_fraction)
            drop=float(self.thermal.compact_plateau_drop)
            u=np.clip(x/rise,0.,1.);w=np.clip((x-rise)/(fall-rise),0.,1.)
            v=np.clip((x-fall)/(1.-fall),0.,1.)
            g=np.where(x<rise,2.*u-u*u,np.where(x<fall,
                1.-drop*(3.*w*w-2.*w*w*w),(1.-drop)*(1.-v)**2))
        else:
            g = x * np.maximum(1.0 - x, 0.0) ** self.b * (1.0 + self.c * x)
        value = self.thermal.t_hat(temperature_K) * g / self.peak_shape
        return np.where(d < dc, value, 0.0)

    def tangent(self, separation_m, temperature_K: float):
        d = np.maximum(np.asarray(separation_m, dtype=float), 0.0)
        dc = self.final_separation(temperature_K)
        x = np.clip(d / dc, 0.0, 1.0)
        if self.thermal.compact_variant == "rounded_plateau":
            rise=float(self.thermal.compact_rise_fraction);fall=float(self.thermal.compact_fall_fraction)
            drop=float(self.thermal.compact_plateau_drop)
            u=np.clip(x/rise,0.,1.);w=np.clip((x-rise)/(fall-rise),0.,1.)
            v=np.clip((x-fall)/(1.-fall),0.,1.)
            derivative=np.where(x<rise,2.*(1.-u)/rise,np.where(x<fall,
                -drop*(6.*w-6.*w*w)/(fall-rise),
                -2.*(1.-drop)*(1.-v)/(1.-fall)))
        else:
            polynomial = (1.0 + (2.0 * self.c - self.b - 1.0) * x
                          - self.c * (self.b + 2.0) * x * x)
            derivative = np.maximum(1.0 - x, 0.0) ** (self.b - 1.0) * polynomial
        value = self.thermal.t_hat(temperature_K) / (dc * self.peak_shape) * derivative
        return np.where(d < dc, value, 0.0)

    def potential(self, separation_m, temperature_K: float):
        d = np.maximum(np.asarray(separation_m, dtype=float), 0.0)
        dc = self.final_separation(temperature_K)
        x = np.clip(d / dc, 0.0, 1.0)
        if self.thermal.compact_variant == "rounded_plateau":
            rise=float(self.thermal.compact_rise_fraction);fall=float(self.thermal.compact_fall_fraction)
            drop=float(self.thermal.compact_plateau_drop)
            u=np.clip(x/rise,0.,1.);w=np.clip((x-rise)/(fall-rise),0.,1.)
            v=np.clip((x-fall)/(1.-fall),0.,1.)
            rise_area=rise*(u*u-u*u*u/3.)
            middle_base=2.*rise/3.
            middle=middle_base+(fall-rise)*(w-drop*(w*w*w-.5*w*w*w*w))
            fall_base=middle_base+(fall-rise)*(1.-drop/2.)
            tail=fall_base+(1.-drop)*(1.-fall)*(v-v*v+v*v*v/3.)
            partial=np.where(x<rise,rise_area,np.where(x<fall,middle,tail))
        else:
            partial = (beta_function(2.0, self.b + 1.0) *
                       betainc(2.0, self.b + 1.0, x)
                       + self.c * beta_function(3.0, self.b + 1.0) *
                       betainc(3.0, self.b + 1.0, x))
        value = self.thermal.gamma(temperature_K) * partial / self.shape_integral
        return np.where(d < dc, value, self.thermal.gamma(temperature_K))


def cohesive_shape_response(shape: str, separation_m,
                            thermal: CohesiveThermalLaw,
                            temperature_K: float):
    """Return traction, tangent and stored surface work for a declared shape."""
    if shape == SMOOTH_EXPONENTIAL:
        return (traction_Pa(separation_m, thermal, temperature_K),
                tangent_Pa_per_m(separation_m, thermal, temperature_K),
                potential_J_m2(separation_m, thermal, temperature_K))
    if shape == BILINEAR:
        law = BilinearCohesiveLaw(thermal)
        return (law.traction(separation_m, temperature_K),
                law.tangent(separation_m, temperature_K),
                law.potential(separation_m, temperature_K))
    if shape == COMPACT_INDEPENDENT:
        law = CompactIndependentCohesiveLaw(thermal)
        return (law.traction(separation_m, temperature_K),
                law.tangent(separation_m, temperature_K),
                law.potential(separation_m, temperature_K))
    raise ValueError(f"unsupported cohesive shape: {shape}")


def mechanical_completion(shape: str, separation_m: float,
                          thermal: CohesiveThermalLaw, temperature_K: float,
                          residual_work_fraction: float = 1.0e-6,
                          residual_traction_fraction: float = 1.0e-4) -> dict[str, Any]:
    """Geometry-owned completion test; it contains no kinetic event clock."""
    d = max(float(separation_m), 0.0)
    gamma = thermal.gamma(temperature_K)
    peak = thermal.t_hat(temperature_K)
    traction, _, work = cohesive_shape_response(shape, d, thermal, temperature_K)
    remaining = max(gamma-float(work), 0.0)
    if shape == BILINEAR:
        cutoff = BilinearCohesiveLaw(thermal).final_separation(temperature_K)
        complete = d >= cutoff
    elif shape == COMPACT_INDEPENDENT:
        cutoff = CompactIndependentCohesiveLaw(thermal).final_separation(temperature_K)
        complete = d >= cutoff
    else:
        cutoff = float(brentq(
            lambda x: (1.0+x)*math.exp(-x)-residual_work_fraction, 1.0, 100.0,
        ))*thermal.delta0_m
        complete = (remaining <= residual_work_fraction*gamma and
                    float(traction) <= residual_traction_fraction*peak)
    return {"complete": bool(complete), "cutoff_m": cutoff,
            "remaining_work_J_m2": remaining,
            "remaining_work_fraction": remaining/gamma,
            "traction_fraction": float(traction)/peak}


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
        if self.execution_identity == COHESIVE_KINETIC_CZM:
            raise RuntimeError(
                "COHESIVE_KINETIC_CZM requires rate_from_environment; "
                "the traction-controlled adapter is barrier-only"
            )
        return self.traction_controlled_rate(stress_Pa, temperature_K)

    def traction_controlled_rate(self, stress_Pa, temperature_K: float):
        stress = np.asarray(stress_Pa, dtype=float)
        G = self.values_eV(stress, temperature_K)
        rates = self.attempt_frequency_s * np.exp(np.clip(
            -G / max(KB_EV_PER_K * float(temperature_K), 1.0e-30), -700.0, 0.0,
        ))
        # The positive-traction zero limit is mathematically finite, but exact
        # contact/compression is not an admissible irreversible opening event.
        return np.where(stress > 0.0, rates, 0.0)

    def rate_from_environment(self, imposed_delta_m: float,
                              environment_stiffness_Pa_per_m: float,
                              temperature_K: float,
                              connected_state: dict[str, Any] | None = None
                              ) -> dict[str, Any]:
        landscape=finite_compliance_landscape(
            imposed_delta_m,environment_stiffness_Pa_per_m,self.thermal,
            temperature_K,connected_state,
        )
        barrier_density=landscape["barrier_J_m2"]
        if landscape["state"] in {"INTACT_BRANCH_LOST_TO_SEPARATED"}:
            barrier_J=0.0
            continuation="FINITE_ATTEMPT_RATE_SPINODAL_REGULARIZATION"
        elif barrier_density is None:
            barrier_J=math.inf
            continuation=None
        else:
            barrier_J=max(float(barrier_density),0.0)*self.activation_area_m2
            continuation=None
        rate=(0.0 if not math.isfinite(barrier_J) else self.attempt_frequency_s*math.exp(
            max(-barrier_J/max(KB_EV_PER_K*EV_TO_J*float(temperature_K),1e-300),-700.0)
        ))
        return {**landscape,"barrier_J":barrier_J,"barrier_eV":barrier_J/EV_TO_J,
                "elementary_rate_s-1":rate,"continuation":continuation,
                "consumer":"FINITE_COMPLIANCE_CONNECTED_COHESIVE_LANDSCAPE"}

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
class PeakCutoffCohesiveControl:
    """Diagnostic cutoff at the traction peak; work is only ``Gamma*(1-2/e)``."""

    thermal: CohesiveThermalLaw
    execution_identity: str = COHESIVE_PEAK_CUTOFF_CONTROL

    def should_rupture(self, separation_m: float, temperature_K: float) -> bool:
        # The control fails at the envelope maximum (x=1).  It is intentionally
        # not exposed through ``rate`` and cannot be mistaken for Arrhenius data.
        self.thermal.gamma(temperature_K)
        return float(separation_m) >= self.thermal.delta0_m

    def traction(self, separation_m, temperature_K: float):
        return traction_Pa(separation_m, self.thermal, temperature_K)

    def rate(self, *_args, **_kwargs):
        raise RuntimeError("rate-independent cohesive control has no Arrhenius rate")

    def as_dict(self):
        return {"kind":"peak_cutoff_diagnostic","execution_identity":self.execution_identity,
                "work_fraction_at_cutoff":1.0-2.0/EULER}


RateIndependentCohesiveControl = PeakCutoffCohesiveControl


@dataclass(frozen=True)
class MatchedWorkRateIndependentCohesiveControl:
    """Rate-independent envelope control with a bounded residual surface work."""

    thermal: CohesiveThermalLaw
    residual_work_fraction: float = 1.0e-6
    execution_identity: str = COHESIVE_RATE_INDEPENDENT_CONTROL

    def __post_init__(self):
        if not (0.0 < self.residual_work_fraction < 1.0):
            raise ValueError("residual_work_fraction must lie strictly between zero and one")

    @property
    def cutoff_x(self) -> float:
        # Remaining work/Gamma=(1+x)exp(-x).
        return float(brentq(lambda x:(1.0+x)*math.exp(-x)-self.residual_work_fraction,1.0,100.0))

    def should_rupture(self,separation_m:float,temperature_K:float) -> bool:
        self.thermal.gamma(temperature_K)
        return float(separation_m)>=self.cutoff_x*self.thermal.delta0_m

    def traction(self,separation_m,temperature_K:float):
        return traction_Pa(separation_m,self.thermal,temperature_K)

    def rate(self,*_args,**_kwargs):
        raise RuntimeError("matched-work rate-independent control has no Arrhenius rate")

    def as_dict(self):
        return {"kind":"matched_work_rate_independent_envelope",
                "execution_identity":self.execution_identity,"cutoff_x":self.cutoff_x,
                "residual_work_fraction":self.residual_work_fraction,
                "traversed_work_fraction":1.0-self.residual_work_fraction}


__all__ = [
    "COHESIVE_BARRIER_ONLY", "COHESIVE_KINETIC_CZM",
    "COHESIVE_RATE_INDEPENDENT_CONTROL", "COHESIVE_PEAK_CUTOFF_CONTROL",
    "COHESIVE_ATHERMAL_MECHANICAL", "COHESIVE_MIXED_MECHANICAL_THERMAL",
    "OPENING_ALTERNATIVES_FAMILY", "SMOOTH_EXPONENTIAL", "BILINEAR",
    "COMPACT_INDEPENDENT",
    "EXECUTION_IDENTITIES",
    "EXP_FLOOR_REFERENCE", "FAMILY", "CohesiveThermalLaw",
    "ExponentialCohesiveBarrier", "RateIndependentCohesiveControl",
    "PeakCutoffCohesiveControl", "MatchedWorkRateIndependentCohesiveControl",
    "bilinear_barrier_J", "morse_barrier_J",
    "BilinearCohesiveLaw", "CompactIndependentCohesiveLaw",
    "cohesive_shape_response", "mechanical_completion",
    "potential_J_m2", "stationary_points", "tangent_Pa_per_m",
    "traction_Pa", "traction_barrier", "finite_compliance_landscape",
]
