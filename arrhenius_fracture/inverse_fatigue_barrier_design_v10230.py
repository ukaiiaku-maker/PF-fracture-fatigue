"""Prospective inverse design tools for v10.2.30 opening barriers.

This analysis-only module maps declared crack-growth targets to exact opening
barriers and production-compatible bounded EXP-floor approximations.  It does
not fit archived fatigue curves and does not alter the production solver.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Callable, Mapping, Sequence

import numpy as np
from scipy.integrate import cumulative_trapezoid
from scipy.optimize import least_squares
from scipy.special import gammainc, gammaincinv, gammaln

from .material_manifest import ExpFloorBarrier, KB_EV_PER_K, MaterialManifest
from .persistent_site_source_v10221 import solve_backstress_limited_activations
from .stochastic_avalanche_tip import threshold_event_length_factor


MODEL_ID = "v10.2.30_inverse_fatigue_barrier_design_v1"
TARGET_INCOMPATIBLE = "TARGET_INCOMPATIBLE_WITH_COOPERATIVE_SATURATION"


@dataclass(frozen=True)
class RenewalControls:
    hits: float = 3.0
    tau_s: float = 1.0e-6
    event_length_m: float = 5.0e-6
    frequency_Hz: float = 1000.0
    temperature_K: float = 300.0
    radius_m: float = 1.0e-6
    n_phase: int = 4096
    stress_cap_Pa: float = 30.0e9


@dataclass(frozen=True)
class ExpFloorBounds:
    G00_eV: tuple[float, float] = (0.4, 4.0)
    sigc_GPa: tuple[float, float] = (0.2, 8.0)
    alpha: tuple[float, float] = (0.02, 5.0)
    exponent: tuple[float, float] = (0.25, 12.0)
    floor_fraction: tuple[float, float] = (0.001, 0.80)

    def lower(self) -> np.ndarray:
        return np.array([self.G00_eV[0], self.sigc_GPa[0], self.alpha[0],
                         self.exponent[0], self.floor_fraction[0]], dtype=float)

    def upper(self) -> np.ndarray:
        return np.array([self.G00_eV[1], self.sigc_GPa[1], self.alpha[1],
                         self.exponent[1], self.floor_fraction[1]], dtype=float)


@dataclass(frozen=True)
class B1Controls:
    n_phase: int = 48
    maximum_cycles: int = 1_000_000
    burn_events: int = 30
    sample_events: int = 80
    r0_m: float = 1.0e-6
    blunting_length_m: float = 0.5e-6
    burgers_m: float = 2.74e-10
    shear_modulus_Pa: float = 160.0e9
    taylor_stress_fraction: float = 1.0 / math.sqrt(3.0)
    forest_floor_m2: float = 5.0e12
    reference_density_m2: float = 5.0e12
    reference_front_width_m: float = 10.0e-6
    reference_source_area_m2: float = 25.0e-12
    mpz_length_m: float = 50.0e-6
    n_systems: int = 2
    n_bins: int = 80
    source_zone_length_m: float = 2.0e-6
    source_line_per_activation: float = 0.9124087591240877
    crystal_theta_deg: float = 30.0
    schmid_reference: float = 0.5
    stress_cap_Pa: float = 30.0e9
    event_length_m: float = 5.0e-6
    cleavage_hits: float = 3.0
    cleavage_tau_s: float = 1.0e-6
    hazard_seed: int = 1720
    avalanche_minimum_factor: float = 0.5
    avalanche_maximum_factor: float = 4.0


def phase_midpoints(n_phase: int) -> np.ndarray:
    n = max(int(n_phase), 32)
    return 2.0 * math.pi * (np.arange(n, dtype=float) + 0.5) / n


def waveform_fraction(R: float, n_phase: int = 4096,
                      tensile_clip: bool = True) -> np.ndarray:
    phi = phase_midpoints(n_phase)
    h = 0.5 * (1.0 + float(R)) + 0.5 * (1.0 - float(R)) * np.cos(phi)
    return np.maximum(h, 0.0) if tensile_clip else h


def waveform_factor_C(M: float, R: float, n_phase: int = 65536) -> float:
    if float(M) <= 0.0:
        raise ValueError("M must be positive")
    return float(np.mean(np.power(waveform_fraction(R, n_phase), float(M))))


def ideal_mode_I_drive_factors(crystal_theta_deg: float = 30.0,
                               schmid_reference: float = 0.5) -> np.ndarray:
    """Reduced-channel factors from the Williams mode-I tensor field.

    This is the analytical counterpart of the production finite-radius tensor
    probe and uses the same BCC trace definitions and Schmid reference.
    """
    from .crystal import bcc_slip_traces

    factors = []
    reference = max(abs(float(schmid_reference)), 1.0e-12)
    for trace in bcc_slip_traces(float(crystal_theta_deg)):
        t = np.asarray(trace["t"], dtype=float)
        n = np.asarray(trace["n"], dtype=float)
        theta = math.atan2(float(t[1]), float(t[0]))
        c, s = math.cos(theta / 2.0), math.sin(theta / 2.0)
        s3, c3 = math.sin(3.0 * theta / 2.0), math.cos(3.0 * theta / 2.0)
        tensor = np.array([
            [c * (1.0 - s * s3), c * s * c3],
            [c * s * c3, c * (1.0 + s * s3)],
        ])
        factors.append(abs(float(t @ tensor @ n)) / reference)
    return np.asarray(factors, dtype=float)


def cooperative_rate_from_raw(raw_rate_s: np.ndarray | float, hits: float,
                              tau_s: float) -> np.ndarray:
    x = np.maximum(np.asarray(raw_rate_s, dtype=float), 0.0) * float(tau_s)
    return np.asarray(gammainc(float(hits), x) / float(tau_s), dtype=float)


def cooperative_Q(hits: float, x: np.ndarray | float) -> np.ndarray:
    """Return d ln P(m,x) / d ln x with stable asymptotic limits."""
    m = float(hits)
    values = np.asarray(x, dtype=float)
    out = np.empty_like(values)
    small = values < 1.0e-7
    large = values > 100.0
    middle = ~(small | large)
    out[small] = m
    out[large] = 0.0
    if np.any(middle):
        xx = values[middle]
        P = gammainc(m, xx)
        log_num = m * np.log(xx) - xx - gammaln(m)
        out[middle] = np.exp(log_num - np.log(np.maximum(P, 1.0e-300)))
    return out


def exp_floor_drop_derivative_eV(barrier: ExpFloorBarrier,
                                 stress_Pa: np.ndarray | float,
                                 temperature_K: float) -> np.ndarray:
    """Return -dG/dln(sigma), in eV, for the current EXP-floor surface."""
    sigma = np.maximum(np.asarray(stress_Pa, dtype=float), 0.0)
    dT = float(temperature_K) - barrier.Tref_K
    G0 = max(barrier.G00_eV + barrier.gT_eV_per_K * dT, 1.0e-12)
    sigc = max(barrier.sigc0_Pa + barrier.sT_Pa_per_K * dT, 1.0)
    floor = min(barrier.floor_max_fraction * G0,
                max(barrier.floor_min_eV, barrier.floor_fraction * G0))
    n = max(barrier.exponent, 1.0e-9)
    u = max(barrier.alpha, 0.0) * np.power(sigma / sigc, n)
    return (G0 - floor) * n * u * np.exp(-u)


def cycle_growth_and_slope(barrier: ExpFloorBarrier, Kmax_MPa_sqrt_m: float,
                           R: float, controls: RenewalControls = RenewalControls(),
                           radius_m: float | None = None,
                           K_shield_Pa_sqrt_m: float = 0.0,
                           dlnsigma_dlnK: np.ndarray | float = 1.0) -> dict[str, float]:
    h = waveform_fraction(R, controls.n_phase)
    K = float(Kmax_MPa_sqrt_m) * 1.0e6 * h
    radius = controls.radius_m if radius_m is None else float(radius_m)
    sigma = np.minimum(
        np.maximum(K - float(K_shield_Pa_sqrt_m), 0.0)
        / math.sqrt(2.0 * math.pi * radius),
        controls.stress_cap_Pa,
    )
    raw = np.asarray(barrier.rate(sigma, controls.temperature_K), dtype=float)
    x = raw * controls.tau_s
    effective = cooperative_rate_from_raw(raw, controls.hits, controls.tau_s)
    action = float(np.mean(effective) / controls.frequency_Hz)
    growth = controls.event_length_m * action
    drop = exp_floor_drop_derivative_eV(barrier, sigma, controls.temperature_K)
    phase_slope = cooperative_Q(controls.hits, x) * drop / (
        KB_EV_PER_K * controls.temperature_K
    ) * np.asarray(dlnsigma_dlnK, dtype=float)
    denominator = float(np.sum(effective))
    slope = float(np.sum(effective * phase_slope) / denominator) if denominator > 0 else math.nan
    return {
        "da_dN": growth,
        "opening_action_per_cycle": action,
        "local_slope": slope,
        "peak_raw_rate_s": float(np.max(raw)),
        "peak_cooperative_rate_s": float(np.max(effective)),
        "cooperative_saturation_fraction": float(np.max(gammainc(controls.hits, x))),
        "weighted_Q": float(np.sum(effective * cooperative_Q(controls.hits, x)) /
                            max(denominator, 1.0e-300)),
    }


def centered_log_slope(function: Callable[[float], float], K: float,
                       relative_step: float = 1.0e-5) -> float:
    h = float(relative_step)
    lo, hi = float(K) * math.exp(-h), float(K) * math.exp(h)
    return (math.log(function(hi)) - math.log(function(lo))) / (2.0 * h)


def evolving_stress_log_derivative(Kmax_Pa_sqrt_m: float,
                                   K_shield_Pa_sqrt_m: float,
                                   dKshield_dlnK_Pa_sqrt_m: float,
                                   dlnradius_dlnK: float) -> float:
    denominator = float(Kmax_Pa_sqrt_m) - float(K_shield_Pa_sqrt_m)
    if denominator <= 0.0:
        raise ValueError("opening stress is clipped and not differentiable")
    return ((float(Kmax_Pa_sqrt_m) - float(dKshield_dlnK_Pa_sqrt_m)) /
            denominator - 0.5 * float(dlnradius_dlnK))


def exact_target_instantaneous_rate(stress_Pa: np.ndarray | float, *, M: float,
                                    R: float, K_ref_MPa_sqrt_m: float,
                                    g_ref_m_per_cycle: float,
                                    controls: RenewalControls) -> np.ndarray:
    sigma_ref = float(K_ref_MPa_sqrt_m) * 1.0e6 / math.sqrt(
        2.0 * math.pi * controls.radius_m
    )
    C = waveform_factor_C(M, R)
    prefactor = controls.frequency_Hz * float(g_ref_m_per_cycle) / (
        controls.event_length_m * C
    )
    sigma = np.maximum(np.asarray(stress_Pa, dtype=float), 0.0)
    return prefactor * np.power(sigma / sigma_ref, float(M))


def invert_cooperative_rate_to_barrier(rate_s: np.ndarray | float, *, hits: float,
                                       tau_s: float, attempt_frequency_s: float,
                                       temperature_K: float) -> np.ndarray:
    rate = np.asarray(rate_s, dtype=float)
    y = tau_s * rate
    if np.any(~np.isfinite(y)) or np.any(y <= 0.0) or np.any(y >= 1.0):
        raise ValueError(TARGET_INCOMPATIBLE)
    x = gammaincinv(float(hits), y)
    if np.any(~np.isfinite(x)) or np.any(x <= 0.0):
        raise ValueError(TARGET_INCOMPATIBLE)
    return KB_EV_PER_K * float(temperature_K) * np.log(
        float(attempt_frequency_s) * float(tau_s) / x
    )


def exact_power_law_target_barrier(stress_Pa: np.ndarray | float, *, M: float,
                                   R: float, K_ref_MPa_sqrt_m: float,
                                   g_ref_m_per_cycle: float,
                                   attempt_frequency_s: float,
                                   controls: RenewalControls) -> np.ndarray:
    rate = exact_target_instantaneous_rate(
        stress_Pa, M=M, R=R, K_ref_MPa_sqrt_m=K_ref_MPa_sqrt_m,
        g_ref_m_per_cycle=g_ref_m_per_cycle, controls=controls,
    )
    return invert_cooperative_rate_to_barrier(
        rate, hits=controls.hits, tau_s=controls.tau_s,
        attempt_frequency_s=attempt_frequency_s,
        temperature_K=controls.temperature_K,
    )


def rare_event_target_barrier(stress_Pa: np.ndarray | float, *, M: float,
                              sigma_ref_Pa: float, G_ref_eV: float,
                              controls: RenewalControls) -> np.ndarray:
    sigma = np.asarray(stress_Pa, dtype=float)
    if np.any(sigma <= 0.0):
        raise ValueError("rare-event target requires positive stress")
    return float(G_ref_eV) - (KB_EV_PER_K * controls.temperature_K * float(M) /
                             controls.hits) * np.log(sigma / float(sigma_ref_Pa))


def rare_event_reference_barrier(rate_s: float, *, controls: RenewalControls,
                                 attempt_frequency_s: float) -> float:
    raw = (math.gamma(controls.hits + 1.0) * float(rate_s) /
           controls.tau_s ** (controls.hits - 1.0)) ** (1.0 / controls.hits)
    return KB_EV_PER_K * controls.temperature_K * math.log(
        float(attempt_frequency_s) / raw
    )


def slope_profile(kind: str, s: np.ndarray, parameters: Mapping[str, object]) -> np.ndarray:
    name = str(kind).upper()
    x = np.asarray(s, dtype=float)
    if name == "CONSTANT_SLOPE":
        return np.full_like(x, float(parameters["M"]))
    if name == "PIECEWISE_CONSTANT_SLOPE":
        edges = np.asarray(parameters["edges"], dtype=float)
        values = np.asarray(parameters["values"], dtype=float)
        if len(values) != len(edges) + 1:
            raise ValueError("piecewise values must have len(edges)+1")
        return values[np.searchsorted(edges, x, side="right")]
    if name == "LOGISTIC_SLOPE_WINDOW":
        M = float(parameters["M_P"])
        son, soff = float(parameters["s_on"]), float(parameters["s_off"])
        won, woff = float(parameters["w_on"]), float(parameters["w_off"])
        L1 = 1.0 / (1.0 + np.exp(np.clip(-(x - son) / won, -700.0, 700.0)))
        L2 = 1.0 / (1.0 + np.exp(np.clip(-(x - soff) / woff, -700.0, 700.0)))
        return M * (L1 - L2)
    if name == "USER_TABULATED_SLOPE_PROFILE":
        nodes = np.asarray(parameters["s"], dtype=float)
        values = np.asarray(parameters["M"], dtype=float)
        if len(nodes) != len(values) or len(nodes) < 2 or np.any(np.diff(nodes) <= 0):
            raise ValueError("tabulated slope nodes must be strictly increasing")
        return np.interp(x, nodes, values, left=values[0], right=values[-1])
    raise ValueError(f"unknown target slope profile {kind!r}")


def integrate_slope_profile(kind: str, s: np.ndarray, parameters: Mapping[str, object],
                            log_rate_ref: float, s_ref: float = 0.0) -> np.ndarray:
    x = np.asarray(s, dtype=float)
    order = np.argsort(np.r_[x, float(s_ref)])
    combined = np.r_[x, float(s_ref)][order]
    slopes = slope_profile(kind, combined, parameters)
    integral = cumulative_trapezoid(slopes, combined, initial=0.0)
    ref_index = int(np.where(order == len(x))[0][0])
    values = float(log_rate_ref) + integral - integral[ref_index]
    inverse = np.empty_like(order)
    inverse[order] = np.arange(len(order))
    return values[inverse[:len(x)]]


def logistic_rare_event_barrier(s: np.ndarray | float, *, G0_eV: float,
                                M_P: float, s_on: float, s_off: float,
                                w_on: float, w_off: float,
                                controls: RenewalControls) -> np.ndarray:
    x = np.asarray(s, dtype=float)
    soft_on = w_on * np.logaddexp(0.0, (x - s_on) / w_on)
    soft_off = w_off * np.logaddexp(0.0, (x - s_off) / w_off)
    return G0_eV - KB_EV_PER_K * controls.temperature_K * M_P / controls.hits * (
        soft_on - soft_off
    )


def barrier_from_vector(values: Sequence[float], template: ExpFloorBarrier) -> ExpFloorBarrier:
    G00, sigc_GPa, alpha, exponent, floor_fraction = map(float, values)
    return ExpFloorBarrier(
        G00_eV=G00, gT_eV_per_K=template.gT_eV_per_K,
        sigc0_Pa=sigc_GPa * 1.0e9, sT_Pa_per_K=template.sT_Pa_per_K,
        alpha=alpha, exponent=exponent, floor_fraction=floor_fraction,
        floor_min_eV=template.floor_min_eV,
        floor_max_fraction=template.floor_max_fraction,
        Tref_K=template.Tref_K,
        attempt_frequency_s=template.attempt_frequency_s,
    )


def project_exp_floor(*, sigma_grid_Pa: np.ndarray, target_barrier_eV: np.ndarray,
                      target_rate_s: np.ndarray, K_grid_MPa_sqrt_m: np.ndarray,
                      target_growth: np.ndarray, target_slope: np.ndarray,
                      R: float, template: ExpFloorBarrier,
                      controls: RenewalControls, bounds: ExpFloorBounds,
                      weights: Mapping[str, float], start: Sequence[float] | None = None
                      ) -> dict[str, object]:
    sigma = np.asarray(sigma_grid_Pa, dtype=float)
    target_G = np.asarray(target_barrier_eV, dtype=float)
    target_log_rate = np.log(np.asarray(target_rate_s, dtype=float))
    target_growth = np.asarray(target_growth, dtype=float)
    target_slope = np.asarray(target_slope, dtype=float)
    target_drop = -np.gradient(target_G, np.log(sigma), edge_order=2)
    if start is None:
        start = [template.G00_eV, template.sigc0_Pa / 1e9, template.alpha,
                 template.exponent, template.floor_fraction]

    def residual(vector: np.ndarray) -> np.ndarray:
        barrier = barrier_from_vector(vector, template)
        G = barrier.values_eV(sigma, controls.temperature_K)
        drop = exp_floor_drop_derivative_eV(barrier, sigma, controls.temperature_K)
        rate = cooperative_rate_from_raw(
            barrier.rate(sigma, controls.temperature_K), controls.hits, controls.tau_s
        )
        evaluated = [cycle_growth_and_slope(barrier, K, R, controls)
                     for K in K_grid_MPa_sqrt_m]
        growth = np.array([x["da_dN"] for x in evaluated])
        slope = np.array([x["local_slope"] for x in evaluated])
        values = {
            "barrier": (G - target_G) / 0.1,
            "derivative": (drop - target_drop) / 0.1,
            "rate": np.log(rate) - target_log_rate,
            "growth": np.log(growth) - np.log(target_growth),
            "slope": slope - target_slope,
            "parameter": ((vector - np.asarray(start, dtype=float)) /
                          np.maximum(bounds.upper() - bounds.lower(), 1.0e-12)),
        }
        pieces = [math.sqrt(float(weights[name])) * values[name]
                  for name in values if float(weights.get(name, 0.0)) > 0.0]
        return np.concatenate(pieces)

    fit = least_squares(residual, np.clip(np.asarray(start, dtype=float), bounds.lower(), bounds.upper()),
                        bounds=(bounds.lower(), bounds.upper()), xtol=1e-11,
                        ftol=1e-11, gtol=1e-11, max_nfev=2500)
    barrier = barrier_from_vector(fit.x, template)
    G = barrier.values_eV(sigma, controls.temperature_K)
    drop = exp_floor_drop_derivative_eV(barrier, sigma, controls.temperature_K)
    rate = cooperative_rate_from_raw(barrier.rate(sigma, controls.temperature_K),
                                     controls.hits, controls.tau_s)
    evaluated = [cycle_growth_and_slope(barrier, K, R, controls)
                 for K in K_grid_MPa_sqrt_m]
    growth = np.array([x["da_dN"] for x in evaluated])
    slope = np.array([x["local_slope"] for x in evaluated])
    return {
        "barrier": barrier,
        "vector": fit.x,
        "success": bool(fit.success),
        "cost": float(fit.cost),
        "nfev": int(fit.nfev),
        "barrier_RMS_eV": float(np.sqrt(np.mean((G - target_G) ** 2))),
        "derivative_RMS_eV": float(np.sqrt(np.mean((drop - target_drop) ** 2))),
        "log_rate_RMS": float(np.sqrt(np.mean((np.log(rate) - target_log_rate) ** 2))),
        "log_growth_RMS": float(np.sqrt(np.mean((np.log(growth) - np.log(target_growth)) ** 2))),
        "slope_RMS": float(np.sqrt(np.mean((slope - target_slope) ** 2))),
        "minimum_saturation_margin": float(1.0 - max(x["cooperative_saturation_fraction"] for x in evaluated)),
        "growth": growth,
        "slope": slope,
    }


def b1_event_conditioned_emission(manifest: MaterialManifest, row: Mapping[str, object],
                                  Kmax_MPa_sqrt_m: float, R: float,
                                  frequency_Hz: float = 1000.0,
                                  temperature_K: float = 300.0,
                                  cleavage_barrier: ExpFloorBarrier | None = None,
                                  controls: B1Controls = B1Controls()) -> dict[str, float | int | bool | str]:
    """Expected event-conditioned one-cycle map with persistent source slip.

    The map resolves waveform phase, retains cleavage residual action, and
    translates the exponential source-linked blunting moment only at an event.
    PT populations are intentionally absent.  A unit renewal action and the
    prospective mean-preserving event length compute the expectation map.
    """
    barrier = manifest.cleavage if cleavage_barrier is None else cleavage_barrier
    h = waveform_fraction(R, controls.n_phase)
    K = float(Kmax_MPa_sqrt_m) * 1.0e6 * h
    dt = 1.0 / (float(frequency_Hz) * controls.n_phase)
    q = 0.0
    rho_back_m2 = np.zeros(controls.n_systems, dtype=float)
    residual_action = 0.0
    rng = np.random.default_rng(np.random.SeedSequence([controls.hazard_seed, 0]))
    threshold_action = max(float(rng.exponential(1.0)), 1.0e-12)
    cycles = 0
    events = 0
    sampled_events = 0
    sample_start_cycle = 0
    sampled_extension = 0.0
    radius_samples: list[float] = []
    backstress_samples: list[float] = []
    emitted_total = 0.0
    rho_site = float(row["rho_source0_m2"])
    Lq = controls.blunting_length_m
    dx = controls.mpz_length_m / controls.n_bins
    backstress_length = max(Lq, dx, controls.burgers_m)
    x_bins = np.arange(controls.n_bins, dtype=float) * dx
    weights = np.exp(-x_bins / backstress_length)
    norm = max(float(np.sum(weights)), 1.0e-30)
    nsrc = max(min(int(math.ceil(controls.source_zone_length_m / dx)),
                    controls.n_bins), 1)
    source_weight = float(np.sum(weights[:nsrc])) / nsrc
    strip_width = max(Lq, dx)
    rho_per_activation = (controls.source_line_per_activation * source_weight /
                          (norm * dx * strip_width))
    q_per_activation = controls.source_line_per_activation * source_weight
    max_events = controls.burn_events + controls.sample_events
    drive_factors = ideal_mode_I_drive_factors(
        controls.crystal_theta_deg, controls.schmid_reference
    )
    if len(drive_factors) != controls.n_systems:
        raise ValueError("B1 requires one ideal mode-I drive factor per reduced system")

    def emission_rate(stress_Pa: float) -> float:
        surface = manifest.emission
        dT = float(temperature_K) - surface.Tref_K
        G0 = max(surface.G00_eV + surface.gT_eV_per_K * dT, 1.0e-12)
        sigc = max(surface.sigc0_Pa + surface.sT_Pa_per_K * dT, 1.0)
        floor = min(surface.floor_max_fraction * G0,
                    max(surface.floor_min_eV, surface.floor_fraction * G0))
        u = max(surface.alpha, 0.0) * (max(stress_Pa, 0.0) / sigc) ** max(
            surface.exponent, 1.0e-9
        )
        G = floor + (G0 - floor) * math.exp(-u)
        return surface.attempt_frequency_s * math.exp(
            max(-G / (KB_EV_PER_K * float(temperature_K)), -700.0)
        )

    while cycles < controls.maximum_cycles and events < max_events:
        cycles += 1
        for K_phase in K:
            radius = controls.r0_m + manifest.c_blunt * controls.burgers_m * max(q, 0.0)
            sigma_open = min(max(float(K_phase), 0.0) /
                             math.sqrt(2.0 * math.pi * radius), controls.stress_cap_Pa)
            backstress_prefactor = (controls.shear_modulus_Pa * controls.burgers_m /
                                    controls.taylor_stress_fraction)
            sigma_back_by_system = backstress_prefactor * np.sqrt(
                np.maximum(rho_back_m2, 0.0)
            )
            sigma_back = float(np.mean(sigma_back_by_system))
            rho_width = max(controls.forest_floor_m2 + float(np.sum(rho_back_m2)),
                            controls.reference_density_m2)
            width = controls.reference_front_width_m * math.sqrt(
                controls.reference_density_m2 / rho_width
            )
            width = min(max(width, Lq), controls.mpz_length_m)
            arc = controls.reference_source_area_m2 / (
                controls.r0_m * controls.reference_front_width_m
            )
            multiplicity = rho_site * arc * radius * width
            activations = np.array([
                solve_backstress_limited_activations(
                    multiplicity=multiplicity,
                    dt_s=dt,
                    drive_stress_Pa=float(drive_factors[system]) * sigma_open,
                    rho_initial_m2=float(rho_back_m2[system]),
                    rho_increment_per_activation_m2=rho_per_activation,
                    backstress_prefactor_Pa_sqrt_m2=backstress_prefactor,
                    rate_function=emission_rate,
                )
                for system in range(controls.n_systems)
            ])
            rho_back_m2 += rho_per_activation * activations
            emitted = controls.source_line_per_activation * float(np.sum(activations))
            q += q_per_activation * float(np.sum(activations))
            emitted_total += emitted
            radius = controls.r0_m + manifest.c_blunt * controls.burgers_m * max(q, 0.0)
            sigma_c = min(max(float(K_phase), 0.0) /
                          math.sqrt(2.0 * math.pi * radius), controls.stress_cap_Pa)
            raw = float(barrier.rate(sigma_c, temperature_K))
            effective = float(cooperative_rate_from_raw(raw, controls.cleavage_hits,
                                                        controls.cleavage_tau_s))
            residual_action += effective * dt
            if residual_action >= threshold_action:
                event_factor = threshold_event_length_factor(
                    threshold_action,
                    minimum_factor=controls.avalanche_minimum_factor,
                    maximum_factor=controls.avalanche_maximum_factor,
                )
                event_length = controls.event_length_m * event_factor
                residual_action = 0.0
                events += 1
                q *= math.exp(-event_length / Lq)
                rho_back_m2 *= math.exp(-event_length / backstress_length)
                if events == controls.burn_events:
                    sample_start_cycle = cycles
                elif events > controls.burn_events:
                    sampled_events += 1
                    sampled_extension += event_length
                    radius_samples.append(controls.r0_m + manifest.c_blunt *
                                          controls.burgers_m * max(q, 0.0))
                    backstress_samples.append(sigma_back)
                threshold_action = max(float(rng.exponential(1.0)), 1.0e-12)
                if events >= max_events:
                    break
            if events >= max_events:
                break
    measured_cycles = max(cycles - sample_start_cycle, 0)
    converged = sampled_events >= controls.sample_events and measured_cycles > 0
    return {
        "model_id": "B1_EVENT_CONDITIONED_EMISSION_v1",
        "da_dN": sampled_extension / measured_cycles if converged else math.nan,
        "event_rate_per_cycle": sampled_events / measured_cycles if converged else math.nan,
        "mean_event_length_m": (sampled_extension / sampled_events
                                if sampled_events else math.nan),
        "cycles": cycles,
        "events": events,
        "sample_events": sampled_events,
        "cleavage_residual_action": residual_action,
        "cleavage_threshold_action": threshold_action,
        "net_source_linked_slip": q,
        "emission_backstress_density_m2": float(np.mean(rho_back_m2)),
        "r_eff_m": float(np.mean(radius_samples)) if radius_samples else math.nan,
        "backstress_Pa": float(np.mean(backstress_samples)) if backstress_samples else math.nan,
        "emitted_total": emitted_total,
        "fixed_point_converged": converged,
        "event_length_semantics": "prospective_mean_of_mean_preserving_threshold_scaled_proposal",
        "energy_gate_assumption": "full_mean_proposal_admitted_in_validated_developed_domain",
        "PT_coordinates": "held_at_A_NATIVE_not_in_B1_state",
    }


def exp_floor_rare_event_max_slope(barrier: ExpFloorBarrier,
                                   temperature_K: float, hits: float) -> float:
    G0 = max(barrier.G00_eV + barrier.gT_eV_per_K *
             (float(temperature_K) - barrier.Tref_K), 1.0e-12)
    floor = min(barrier.floor_max_fraction * G0,
                max(barrier.floor_min_eV, barrier.floor_fraction * G0))
    return float(hits) * (G0 - floor) * barrier.exponent / (
        math.e * KB_EV_PER_K * float(temperature_K)
    )


__all__ = [
    "B1Controls", "ExpFloorBounds", "MODEL_ID", "RenewalControls",
    "TARGET_INCOMPATIBLE", "b1_event_conditioned_emission",
    "centered_log_slope", "cooperative_Q", "cooperative_rate_from_raw",
    "cycle_growth_and_slope", "evolving_stress_log_derivative",
    "exact_power_law_target_barrier", "exact_target_instantaneous_rate",
    "exp_floor_drop_derivative_eV", "exp_floor_rare_event_max_slope",
    "integrate_slope_profile", "invert_cooperative_rate_to_barrier",
    "logistic_rare_event_barrier", "phase_midpoints", "project_exp_floor",
    "rare_event_reference_barrier", "rare_event_target_barrier",
    "slope_profile", "waveform_factor_C", "waveform_fraction",
]
