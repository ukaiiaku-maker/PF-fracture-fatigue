"""Analysis-only thermodynamic barrier surfaces for the v10.2.30 joint search.

This module is deliberately not imported by a production entrypoint.  It
extends frozen 300 K EXP-floor surfaces with a low-stress cleavage guard and
thermodynamically linked entropy/heat-capacity fields.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Mapping

import numpy as np
from scipy.special import gammainc, gammaln

from .material_manifest import KB_EV_PER_K

EV_J = 1.602176634e-19
TR_K = 300.0


@dataclass(frozen=True)
class Basis:
    alpha: float
    sigma_Pa: float
    exponent: float

    def value(self, stress_Pa):
        s = np.maximum(np.asarray(stress_Pa, dtype=float), 0.0)
        return np.exp(-self.alpha * (s / self.sigma_Pa) ** self.exponent)

    def derivative(self, stress_Pa):
        s = np.maximum(np.asarray(stress_Pa, dtype=float), 0.0)
        q = self.value(s)
        out = np.zeros_like(s)
        positive = s > 0.0
        out[positive] = (-self.alpha * self.exponent
                         * (s[positive] / self.sigma_Pa) ** self.exponent
                         * q[positive] / s[positive])
        return out


@dataclass(frozen=True)
class ThermodynamicSurface:
    process: str
    attempt_frequency_s: float
    reference_floor_eV: float
    reference_amplitudes_eV: tuple[float, ...]
    bases: tuple[Basis, ...]
    entropy_infinity_kB: float
    entropy_amplitudes_kB: tuple[float, ...]
    heat_capacity_infinity_kB: float = 0.0
    heat_capacity_amplitudes_kB: tuple[float, ...] = ()

    def _basis_values(self, stress_Pa):
        return [b.value(stress_Pa) for b in self.bases]

    def _basis_derivatives(self, stress_Pa):
        return [b.derivative(stress_Pa) for b in self.bases]

    def reference_G_eV(self, stress_Pa):
        values = self._basis_values(stress_Pa)
        result = np.zeros_like(np.asarray(stress_Pa, dtype=float)) + self.reference_floor_eV
        for amplitude, q in zip(self.reference_amplitudes_eV, values):
            result += amplitude * q
        return result

    def entropy_kB(self, stress_Pa, temperature_K):
        values = self._basis_values(stress_Pa)
        result = np.zeros_like(np.asarray(stress_Pa, dtype=float)) + self.entropy_infinity_kB
        for amplitude, q in zip(self.entropy_amplitudes_kB, values):
            result += amplitude * q
        cp_inf = self.heat_capacity_infinity_kB
        cp_amp = self.heat_capacity_amplitudes_kB or (0.0,) * len(self.bases)
        cp = np.zeros_like(result) + cp_inf
        for amplitude, q in zip(cp_amp, values):
            cp += amplitude * q
        return result + cp * math.log(float(temperature_K) / TR_K)

    def G_eV(self, stress_Pa, temperature_K):
        T = float(temperature_K)
        s0 = self.entropy_kB(stress_Pa, TR_K)
        values = self._basis_values(stress_Pa)
        cp_amp = self.heat_capacity_amplitudes_kB or (0.0,) * len(self.bases)
        cp = np.zeros_like(np.asarray(stress_Pa, dtype=float)) + self.heat_capacity_infinity_kB
        for amplitude, q in zip(cp_amp, values):
            cp += amplitude * q
        thermal = (T - TR_K) - T * math.log(T / TR_K)
        return self.reference_G_eV(stress_Pa) - (T - TR_K) * KB_EV_PER_K * s0 + KB_EV_PER_K * cp * thermal

    def enthalpy_eV(self, stress_Pa, temperature_K):
        T = float(temperature_K)
        return self.G_eV(stress_Pa, T) + T * KB_EV_PER_K * self.entropy_kB(stress_Pa, T)

    def dG_dsigma_eV_per_Pa(self, stress_Pa, temperature_K):
        T = float(temperature_K)
        dq = self._basis_derivatives(stress_Pa)
        result = np.zeros_like(np.asarray(stress_Pa, dtype=float))
        for amplitude, derivative in zip(self.reference_amplitudes_eV, dq):
            result += amplitude * derivative
        for amplitude, derivative in zip(self.entropy_amplitudes_kB, dq):
            result -= (T - TR_K) * KB_EV_PER_K * amplitude * derivative
        thermal = (T - TR_K) - T * math.log(T / TR_K)
        cp_amp = self.heat_capacity_amplitudes_kB or (0.0,) * len(self.bases)
        for amplitude, derivative in zip(cp_amp, dq):
            result += KB_EV_PER_K * thermal * amplitude * derivative
        return result

    def activation_volume_m3(self, stress_Pa, temperature_K):
        return -self.dG_dsigma_eV_per_Pa(stress_Pa, temperature_K) * EV_J

    def dS_dsigma_J_per_K_Pa(self, stress_Pa, temperature_K):
        dq = self._basis_derivatives(stress_Pa)
        result = np.zeros_like(np.asarray(stress_Pa, dtype=float))
        for amplitude, derivative in zip(self.entropy_amplitudes_kB, dq):
            result += KB_EV_PER_K * amplitude * derivative
        cp_amp = self.heat_capacity_amplitudes_kB or (0.0,) * len(self.bases)
        factor = math.log(float(temperature_K) / TR_K)
        for amplitude, derivative in zip(cp_amp, dq):
            result += KB_EV_PER_K * factor * amplitude * derivative
        return result * EV_J

    def dV_dT_m3_per_K(self, stress_Pa, temperature_K):
        return self.dS_dsigma_J_per_K_Pa(stress_Pa, temperature_K)

    def raw_rate_s(self, stress_Pa, temperature_K):
        exponent = -self.G_eV(stress_Pa, temperature_K) / (KB_EV_PER_K * float(temperature_K))
        return self.attempt_frequency_s * np.exp(np.clip(exponent, -745.0, 700.0))


def exp_floor_reference(barrier):
    G0 = float(barrier.values_eV(0.0, TR_K))
    floor = min(barrier.floor_max_fraction * G0,
                max(barrier.floor_min_eV, barrier.floor_fraction * G0))
    return floor, G0 - floor, Basis(float(barrier.alpha), float(barrier.sigc0_Pa),
                                    float(barrier.exponent))


def derive_guard_sigma(stress_low_active_Pa: float, log10_q: float, exponent: float) -> float:
    return float(stress_low_active_Pa) / (-float(log10_q) * math.log(10.0)) ** (1.0 / float(exponent))


def solve_cleavage_entropy(base: Basis, guard: Basis, active_stress_Pa: float,
                           zero_kB: float, active_kB: float, infinity_kB: float):
    matrix = np.array([[1.0, 1.0],
                       [float(base.value(active_stress_Pa)), float(guard.value(active_stress_Pa))]])
    rhs = np.array([float(zero_kB) - float(infinity_kB),
                    float(active_kB) - float(infinity_kB)])
    if abs(np.linalg.det(matrix)) < 1.0e-10:
        raise ValueError("singular cleavage entropy coordinates")
    return tuple(np.linalg.solve(matrix, rhs))


def solve_emission_entropy(base: Basis, active_stress_Pa: float,
                           active_kB: float, infinity_kB: float):
    q = float(base.value(active_stress_Pa))
    if q < 1.0e-10:
        raise ValueError("emission active coordinate has vanishing basis")
    return (float(active_kB) - float(infinity_kB)) / q


def cooperative_rate(raw_rate_s, hits: float, tau_s: float):
    return gammainc(float(hits), np.minimum(np.asarray(raw_rate_s) * float(tau_s), 1.0e12)) / float(tau_s)


def cooperative_Q(raw_rate_s, hits: float, tau_s: float):
    x = np.asarray(raw_rate_s, dtype=float) * float(tau_s)
    out = np.zeros_like(x)
    middle = (x >= 1e-9) & (x <= 100.0)
    out[x < 1e-9] = hits
    if np.any(middle):
        xx = x[middle]; p = gammainc(hits, xx)
        out[middle] = np.exp(hits * np.log(xx) - xx - gammaln(hits) - np.log(np.maximum(p, 1e-300)))
    return out


def weighted_quantile(values, weights, quantile):
    order = np.argsort(values); x = np.asarray(values)[order]; w = np.asarray(weights)[order]
    if w.sum() <= 0: raise ValueError("zero action weight")
    return float(x[np.searchsorted(np.cumsum(w), float(quantile) * w.sum(), side="left")])


def candidate_surface_from_parameters(parent_manifest, parameters: Mapping[str, float],
                                      cleavage_low_stress_Pa: float,
                                      cleavage_active_stress_Pa: float,
                                      emission_active_stress_Pa: float):
    cfloor, camp, cbasis = exp_floor_reference(parent_manifest.cleavage)
    efloor, eamp, ebasis = exp_floor_reference(parent_manifest.emission)
    guard = Basis(1.0, derive_guard_sigma(cleavage_low_stress_Pa,
                                         parameters["guard_log10_q_low"],
                                         parameters["guard_exponent"]),
                  parameters["guard_exponent"])
    guard_amp = parameters["cleavage_zero_target_eV"] - (cfloor + camp)
    c_entropy = solve_cleavage_entropy(cbasis, guard, cleavage_active_stress_Pa,
                                       parameters["cleavage_entropy_zero_kB"],
                                       parameters["cleavage_entropy_active_kB"],
                                       parameters["cleavage_entropy_infinity_kB"])
    e_entropy = solve_emission_entropy(ebasis, emission_active_stress_Pa,
                                       parameters["emission_entropy_active_kB"],
                                       parameters["emission_entropy_infinity_kB"])
    c_cp = (parameters.get("cleavage_heat_capacity_base_kB", 0.0),
            parameters.get("cleavage_heat_capacity_guard_kB", 0.0))
    e_cp = (parameters.get("emission_heat_capacity_base_kB", 0.0),)
    cleavage = ThermodynamicSurface("cleavage", parent_manifest.cleavage.attempt_frequency_s,
        cfloor, (camp, guard_amp), (cbasis, guard), parameters["cleavage_entropy_infinity_kB"],
        c_entropy, 0.0, c_cp)
    emission = ThermodynamicSurface("emission", parent_manifest.emission.attempt_frequency_s,
        efloor, (eamp,), (ebasis,), parameters["emission_entropy_infinity_kB"],
        (e_entropy,), 0.0, e_cp)
    return cleavage, emission

