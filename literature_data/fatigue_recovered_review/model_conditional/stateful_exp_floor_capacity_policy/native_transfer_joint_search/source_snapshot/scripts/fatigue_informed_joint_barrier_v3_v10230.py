"""Additive V3 fatigue-informed thermodynamic opening surfaces.

The V2 registry remains immutable.  This module defines the separately named
smooth broad basis, solves the three dependent 300 K reference parameters, and
serializes complete V3 surfaces without teaching the legacy ``Basis`` class
about smoothing.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
from typing import Any, Mapping

import numpy as np
from scipy.integrate import cumulative_trapezoid
from scipy.optimize import least_squares

from arrhenius_fracture.material_manifest import KB_EV_PER_K
from scripts.thermodynamic_joint_barrier_v10230 import (
    Basis,
    EV_J,
    ThermodynamicSurface,
    cooperative_rate,
)
from arrhenius_fracture.inverse_fatigue_barrier_design_v10230 import waveform_fraction


T_REF_K = 300.0
RADIUS_M = 1.0e-6
STRESS_CAP_PA = 30.0e9
R_RATIO = 0.1
FREQUENCY_HZ = 1000.0
RAMP_RATE_MPA_SQRT_M_PER_S = 0.005
ACTION_THRESHOLD = math.log(2.0)
EVENT_LENGTH_M = 5.0e-6


def canonical_hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode()
    ).hexdigest()


@dataclass(frozen=True)
class SmoothBroadBasis:
    """Normalized even regularization of a fractional EXP stress basis."""

    sigma_b_Pa: float
    exponent: float
    smoothing_ratio: float = 1.0e-4

    def __post_init__(self) -> None:
        if self.sigma_b_Pa <= 0.0 or self.exponent <= 0.0 or self.smoothing_ratio <= 0.0:
            raise ValueError("smooth broad-basis parameters must be positive")

    @property
    def _denominator(self) -> float:
        eta = self.smoothing_ratio
        return (1.0 + eta * eta) ** (0.5 * self.exponent) - eta ** self.exponent

    def _z(self, stress_Pa):
        s = np.maximum(np.asarray(stress_Pa, dtype=float), 0.0)
        t = s / self.sigma_b_Pa
        eta = self.smoothing_ratio
        return ((t * t + eta * eta) ** (0.5 * self.exponent) - eta ** self.exponent) / self._denominator

    def value(self, stress_Pa):
        return np.exp(-self._z(stress_Pa))

    def derivative(self, stress_Pa):
        s = np.maximum(np.asarray(stress_Pa, dtype=float), 0.0)
        t = s / self.sigma_b_Pa
        eta = self.smoothing_ratio
        u = t * t + eta * eta
        dz = self.exponent * t * u ** (0.5 * self.exponent - 1.0) / (
            self.sigma_b_Pa * self._denominator
        )
        return -self.value(s) * dz

    def second_derivative(self, stress_Pa):
        s = np.maximum(np.asarray(stress_Pa, dtype=float), 0.0)
        t = s / self.sigma_b_Pa
        eta = self.smoothing_ratio
        u = t * t + eta * eta
        prefactor = self.exponent / (self.sigma_b_Pa**2 * self._denominator)
        dz = self.exponent * t * u ** (0.5 * self.exponent - 1.0) / (
            self.sigma_b_Pa * self._denominator
        )
        d2z = prefactor * (
            u ** (0.5 * self.exponent - 1.0)
            + (self.exponent - 2.0) * t * t * u ** (0.5 * self.exponent - 2.0)
        )
        return self.value(s) * (dz * dz - d2z)


def stress_from_K(K_MPa_sqrt_m):
    K = np.asarray(K_MPa_sqrt_m, dtype=float)
    return np.minimum(K * 1.0e6 / math.sqrt(2.0 * math.pi * RADIUS_M), STRESS_CAP_PA)


def cycle_action(surface, Kmax_MPa_sqrt_m: float, *, phase_steps: int = 2048) -> float:
    h = waveform_fraction(R_RATIO, int(phase_steps))
    stress = np.minimum(
        float(Kmax_MPa_sqrt_m) * 1.0e6 * h / math.sqrt(2.0 * math.pi * RADIUS_M),
        STRESS_CAP_PA,
    )
    rate = cooperative_rate(surface.raw_rate_s(stress, T_REF_K), surface.hits, surface.tau_s)
    return float(np.mean(rate) / FREQUENCY_HZ)


def local_cycle_slope(surface, Kmax_MPa_sqrt_m: float, *, perturbation: float = 2.0e-3,
                      phase_steps: int = 2048) -> float:
    lo = float(Kmax_MPa_sqrt_m) * math.exp(-perturbation)
    hi = float(Kmax_MPa_sqrt_m) * math.exp(perturbation)
    actions = [cycle_action(surface, x, phase_steps=phase_steps) for x in (lo, hi)]
    if min(actions) <= 0.0:
        return math.nan
    return float((math.log(actions[1]) - math.log(actions[0])) / (2.0 * perturbation))


def ramp_action(surface, K_end_MPa_sqrt_m: float, *, points: int = 4096) -> float:
    end = float(K_end_MPa_sqrt_m)
    K = np.linspace(0.0, end, int(points) + 1)
    rate = cooperative_rate(
        surface.raw_rate_s(stress_from_K(K), T_REF_K), surface.hits, surface.tau_s
    )
    return float(cumulative_trapezoid(rate, K, initial=0.0)[-1] / RAMP_RATE_MPA_SQRT_M_PER_S)


def first_passage_root(surface, *, maximum_K: float = 80.0, points: int = 16000) -> float:
    K = np.linspace(0.0, float(maximum_K), int(points) + 1)
    rate = cooperative_rate(
        surface.raw_rate_s(stress_from_K(K), T_REF_K), surface.hits, surface.tau_s
    )
    action = cumulative_trapezoid(rate, K, initial=0.0) / RAMP_RATE_MPA_SQRT_M_PER_S
    if action[-1] < ACTION_THRESHOLD:
        return math.nan
    index = int(np.searchsorted(action, ACTION_THRESHOLD))
    return float(np.interp(ACTION_THRESHOLD, action[index - 1:index + 1], K[index - 1:index + 1]))


@dataclass(frozen=True)
class KineticSurface(ThermodynamicSurface):
    """Thermodynamic surface carrying its exact renewal law."""

    hits: float = 3.273241435177624
    tau_s: float = 6.992153587194454e-7


def _reconstruct_entropy(source, broad, guard, active_stress_Pa: float,
                         high_stress_Pa: float = STRESS_CAP_PA) -> tuple[float, tuple[float, float]]:
    anchors = np.array([0.0, float(active_stress_Pa), float(high_stress_Pa)])
    matrix = np.column_stack((np.ones(3), broad.value(anchors), guard.value(anchors)))
    target = np.asarray(source.entropy_kB(anchors, T_REF_K), dtype=float)
    if abs(np.linalg.det(matrix)) < 1.0e-12:
        raise ValueError("singular V3 entropy-anchor reconstruction")
    coefficients = np.linalg.solve(matrix, target)
    return float(coefficients[0]), (float(coefficients[1]), float(coefficients[2]))


def make_surface(source, *, G_inf_eV: float, D_b_eV: float, sigma_b_Pa: float,
                 broad_exponent: float, sigma_g_Pa: float, guard_exponent: float,
                 smoothing_ratio: float, active_stress_Pa: float, hits: float, tau_s: float):
    broad = SmoothBroadBasis(sigma_b_Pa, broad_exponent, smoothing_ratio)
    guard = Basis(1.0, sigma_g_Pa, guard_exponent)
    source_zero = float(source.G_eV(0.0, T_REF_K))
    D_g_eV = source_zero - float(G_inf_eV) - float(D_b_eV)
    entropy_inf, entropy_amplitudes = _reconstruct_entropy(
        source, broad, guard, active_stress_Pa
    )
    return KineticSurface(
        process="cleavage",
        attempt_frequency_s=float(source.attempt_frequency_s),
        reference_floor_eV=float(G_inf_eV),
        reference_amplitudes_eV=(float(D_b_eV), float(D_g_eV)),
        bases=(broad, guard),
        entropy_infinity_kB=entropy_inf,
        entropy_amplitudes_kB=entropy_amplitudes,
        heat_capacity_infinity_kB=0.0,
        heat_capacity_amplitudes_kB=(0.0, 0.0),
        hits=float(hits),
        tau_s=float(tau_s),
    )


def solve_reference_anchors(source, *, sigma_b_Pa: float, broad_exponent: float,
                            guard_exponent: float, smoothing_ratio: float,
                            active_stress_Pa: float, reference_K: float,
                            source_F0_K: float, target_slope: float, hits: float,
                            tau_s: float, phase_steps: int = 2048) -> tuple[KineticSurface, dict[str, Any]]:
    if isinstance(source, KineticSurface):
        source_kinetic = source
    else:
        source_kinetic = KineticSurface(**source.__dict__, hits=hits, tau_s=tau_s)
    target_action = cycle_action(source_kinetic, reference_K, phase_steps=phase_steps)
    source_zero = float(source.G_eV(0.0, T_REF_K))

    def decode(x):
        floor_fraction = 1.0 / (1.0 + math.exp(-float(x[0])))
        broad_fraction = 1.0 / (1.0 + math.exp(-float(x[1])))
        floor = source_zero * floor_fraction
        broad_amplitude = (source_zero - floor) * broad_fraction
        return floor, broad_amplitude, math.exp(float(x[2]))

    def evaluate(x):
        floor, broad_amplitude, guard_sigma = decode(x)
        surface = make_surface(
            source, G_inf_eV=floor, D_b_eV=broad_amplitude,
            sigma_b_Pa=sigma_b_Pa, broad_exponent=broad_exponent,
            sigma_g_Pa=guard_sigma, guard_exponent=guard_exponent,
            smoothing_ratio=smoothing_ratio, active_stress_Pa=active_stress_Pa,
            hits=hits, tau_s=tau_s,
        )
        action = cycle_action(surface, reference_K, phase_steps=phase_steps)
        slope = local_cycle_slope(surface, reference_K, phase_steps=phase_steps)
        ramp = ramp_action(surface, source_F0_K)
        residual = np.array([
            math.log(max(action, 1.0e-300) / max(target_action, 1.0e-300)),
            (slope - target_slope) / max(abs(target_slope), 1.0),
            (ramp - ACTION_THRESHOLD) / ACTION_THRESHOLD,
        ])
        return surface, residual, action, slope, ramp

    starts = [
        (-1.0, 1.0, math.log(0.25e9)),
        (-2.0, 2.0, math.log(0.75e9)),
        (0.0, 0.0, math.log(1.5e9)),
        (1.0, -1.0, math.log(3.0e9)),
    ]
    best = None
    for start in starts:
        fit = least_squares(
            lambda x: evaluate(x)[1], start,
            bounds=([-14.0, -14.0, math.log(1.0e6)], [14.0, 14.0, math.log(50.0e9)]),
            max_nfev=800, xtol=1.0e-11, ftol=1.0e-11, gtol=1.0e-11,
        )
        surface, residual, action, slope, ramp = evaluate(fit.x)
        score = float(np.max(np.abs(residual)))
        item = (score, surface, residual, action, slope, ramp, fit)
        if best is None or score < best[0]:
            best = item
    assert best is not None
    score, surface, residual, action, slope, ramp, fit = best
    metrics = {
        "solver_success": bool(fit.success),
        "solver_message": str(fit.message),
        "solver_nfev": int(fit.nfev),
        "maximum_scaled_anchor_residual": score,
        "cycle_action_target": target_action,
        "cycle_action_solved": action,
        "local_slope_target": float(target_slope),
        "local_slope_solved": slope,
        "ramp_action_target": ACTION_THRESHOLD,
        "ramp_action_solved": ramp,
        "anchor_residual_vector": residual.tolist(),
    }
    return surface, metrics


def serialize_surface(surface: KineticSurface) -> dict[str, Any]:
    bases = []
    for basis in surface.bases:
        if isinstance(basis, SmoothBroadBasis):
            bases.append({"kind": "SMOOTH_NORMALIZED_BROAD_V1", **asdict(basis)})
        elif isinstance(basis, Basis):
            bases.append({"kind": "LEGACY_EXP", **asdict(basis)})
        else:
            raise TypeError(f"unsupported V3 basis {type(basis).__name__}")
    return {
        "schema": "v10.2.30_v3_thermodynamic_surface_v1",
        "process": surface.process,
        "attempt_frequency_s": surface.attempt_frequency_s,
        "reference_floor_eV": surface.reference_floor_eV,
        "reference_amplitudes_eV": list(surface.reference_amplitudes_eV),
        "bases": bases,
        "entropy_infinity_kB": surface.entropy_infinity_kB,
        "entropy_amplitudes_kB": list(surface.entropy_amplitudes_kB),
        "heat_capacity_infinity_kB": surface.heat_capacity_infinity_kB,
        "heat_capacity_amplitudes_kB": list(surface.heat_capacity_amplitudes_kB),
        "hits": surface.hits,
        "tau_s": surface.tau_s,
    }


def deserialize_surface(data: Mapping[str, Any]) -> KineticSurface:
    if data.get("schema") != "v10.2.30_v3_thermodynamic_surface_v1":
        raise ValueError("unknown V3 surface schema")
    bases = []
    for item in data["bases"]:
        item = dict(item)
        kind = item.pop("kind")
        bases.append(SmoothBroadBasis(**item) if kind == "SMOOTH_NORMALIZED_BROAD_V1" else Basis(**item))
    return KineticSurface(
        process=data["process"], attempt_frequency_s=float(data["attempt_frequency_s"]),
        reference_floor_eV=float(data["reference_floor_eV"]),
        reference_amplitudes_eV=tuple(data["reference_amplitudes_eV"]), bases=tuple(bases),
        entropy_infinity_kB=float(data["entropy_infinity_kB"]),
        entropy_amplitudes_kB=tuple(data["entropy_amplitudes_kB"]),
        heat_capacity_infinity_kB=float(data["heat_capacity_infinity_kB"]),
        heat_capacity_amplitudes_kB=tuple(data["heat_capacity_amplitudes_kB"]),
        hits=float(data["hits"]), tau_s=float(data["tau_s"]),
    )


def reference_window(surface, *, reference_K: float, target_slope: float,
                     load_min: float = 1.0, load_max: float = 50.0,
                     load_points: int = 241, phase_steps: int = 2048) -> dict[str, Any]:
    loads = np.geomspace(load_min, load_max, int(load_points))
    actions = np.array([cycle_action(surface, K, phase_steps=phase_steps) for K in loads])
    rates = EVENT_LENGTH_M * actions
    slopes = np.gradient(np.log(np.maximum(rates, 1.0e-300)), np.log(loads))
    mask = (slopes >= 0.9 * target_slope) & (slopes <= 1.1 * target_slope)
    reference_index = int(np.argmin(abs(np.log(loads / reference_K))))
    if not mask[reference_index]:
        lo = hi = reference_index
        width = 0.0
    else:
        lo = reference_index
        hi = reference_index
        while lo > 0 and mask[lo - 1]:
            lo -= 1
        while hi + 1 < len(mask) and mask[hi + 1]:
            hi += 1
        width = float(math.log10(rates[hi] / rates[lo])) if rates[lo] > 0.0 else 0.0
    return {
        "reference_index": reference_index,
        "window_index_low": lo,
        "window_index_high": hi,
        "window_K_low": float(loads[lo]),
        "window_K_high": float(loads[hi]),
        "window_rate_low_m_per_cycle": float(rates[lo]),
        "window_rate_high_m_per_cycle": float(rates[hi]),
        "window_rate_decades": width,
        "screening_success": bool(mask[reference_index] and width >= 2.0),
        "loads_MPa_sqrt_m": loads.tolist(),
        "cycle_actions": actions.tolist(),
        "analytical_event_length_rates_m_per_cycle": rates.tolist(),
        "local_slopes": slopes.tolist(),
        "slope_mask": mask.tolist(),
        "rate_semantics": "A0_FIXED_EVENT_LENGTH_INITIALIZER_NOT_DEVELOPED_GROWTH",
    }


def derivative_audit(surface: KineticSurface) -> dict[str, float]:
    broad = surface.bases[0]
    stress = np.r_[0.0, np.geomspace(1.0e3, STRESS_CAP_PA, 180)]
    h = np.maximum(1.0, 1.0e-5 * np.maximum(stress, 1.0))
    lo = np.maximum(stress - h, 0.0)
    hi = stress + h
    fd1 = (broad.value(hi) - broad.value(lo)) / (hi - lo)
    positive = stress > h
    fd2 = (broad.value(stress[positive] + h[positive]) - 2.0 * broad.value(stress[positive])
           + broad.value(stress[positive] - h[positive])) / h[positive] ** 2
    return {
        "broad_first_derivative_max_abs": float(np.max(abs(fd1 - broad.derivative(stress)))),
        "broad_second_derivative_max_abs": float(np.max(abs(fd2 - broad.second_derivative(stress[positive])))),
        "broad_first_derivative_at_zero": float(broad.derivative(0.0)),
        "broad_second_derivative_at_zero": float(broad.second_derivative(0.0)),
        "minimum_activation_volume_m3": float(np.min(surface.activation_volume_m3(stress, T_REF_K))),
        "maximum_dG_dsigma_eV_per_Pa": float(np.max(surface.dG_dsigma_eV_per_Pa(stress, T_REF_K))),
    }


__all__ = [
    "ACTION_THRESHOLD", "KineticSurface", "SmoothBroadBasis", "canonical_hash",
    "cycle_action", "derivative_audit", "deserialize_surface", "first_passage_root",
    "local_cycle_slope", "make_surface", "ramp_action", "reference_window",
    "serialize_surface", "solve_reference_anchors",
]
