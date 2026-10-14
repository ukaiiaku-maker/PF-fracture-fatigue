"""Finite-domain bilinear analogues of native EXP-floor opening barriers."""
from __future__ import annotations

import math
from typing import Iterable

import numpy as np
from scipy.optimize import least_squares

from .material_manifest import KB_EV_PER_K


TRAIN_DECAY_FRACTIONS = (0.95, 0.80, 0.60, 0.40, 0.20, 0.10)
HELD_OUT_DECAY_FRACTIONS = (0.90, 0.70, 0.50, 0.30, 0.15)
FIT_TEMPERATURES_K = (300.0, 550.0, 800.0, 1050.0, 1300.0)


def stress_at_decay_fraction(surface, temperature_K: float, fraction: float) -> float:
    fraction = float(fraction)
    if not 0.0 < fraction < 1.0:
        raise ValueError("barrier decay fraction must lie strictly between zero and one")
    diagnostic = surface.diagnostic(0.0, temperature_K)
    return float(diagnostic["executed_sigc_Pa"]) * (
        -math.log(fraction) / float(surface.alpha)
    ) ** (1.0 / float(surface.exponent))


def bilinear_barrier_and_volume(
    stress_Pa: np.ndarray, B0_eV: float, peak_traction_Pa: float
) -> tuple[np.ndarray, np.ndarray]:
    q = np.asarray(stress_Pa, dtype=float)
    B0 = float(B0_eV)
    tc = float(peak_traction_Pa)
    if B0 <= 0.0 or tc <= 0.0 or np.any(q < 0.0) or np.any(q >= tc):
        raise ValueError("bilinear activated barrier requires B0>0 and 0<=q<tc")
    remaining = 1.0 - q / tc
    return B0 * remaining * remaining, 2.0 * B0 / tc * remaining


def _native_values(surface, temperature_K: float, stress_Pa: np.ndarray):
    barriers = []
    volumes = []
    for stress in stress_Pa:
        diagnostic = surface.diagnostic(float(stress), temperature_K)
        derivative = diagnostic["partial_G_partial_sigma_eV_Pa"]
        if derivative is None:
            raise ValueError("native derivative is undefined inside the fit domain")
        barriers.append(float(diagnostic["barrier_eV"]))
        volumes.append(-float(derivative))
    return np.asarray(barriers), np.asarray(volumes)


def fit_temperature_slice(
    surface, temperature_K: float, *, beta: float = 1.0,
    fractions: Iterable[float] = TRAIN_DECAY_FRACTIONS,
) -> dict[str, object]:
    """Fit identifiable ``B0=A*Gamma`` and ``tc`` on a frozen native window."""
    T_K = float(temperature_K)
    q = np.asarray([
        stress_at_decay_fraction(surface, T_K, fraction) for fraction in fractions
    ])
    native_G, native_V = _native_values(surface, T_K, q)
    q_span = float(np.max(q) - np.min(q))
    kT = KB_EV_PER_K * T_K
    weights = np.full(q.size, 1.0 / q.size)
    lower = np.log([1.0e-8, np.max(q) * (1.0 + 1.0e-9)])
    upper = np.log([1.0e3, max(np.max(q) * 1.0e4, 1.0e12)])

    def residual(log_parameters):
        B0, tc = np.exp(log_parameters)
        fitted_G, fitted_V = bilinear_barrier_and_volume(q, B0, tc)
        barrier = np.sqrt(weights) * (fitted_G - native_G) / kT
        volume = np.sqrt(float(beta) * weights) * q_span * (
            fitted_V - native_V
        ) / kT
        return np.concatenate((barrier, volume))

    G0 = float(surface.diagnostic(0.0, T_K)["executed_G0_eV"])
    starts = [
        (G0 * B_scale, np.max(q) * tc_scale)
        for B_scale in (0.5, 1.0, 2.0)
        for tc_scale in (1.05, 1.5, 3.0, 10.0)
    ]
    solutions = []
    for B_start, tc_start in starts:
        start = np.log([B_start, tc_start])
        answer = least_squares(
            residual, np.clip(start, lower, upper), bounds=(lower, upper),
            xtol=1.0e-13, ftol=1.0e-13, gtol=1.0e-13, max_nfev=5000,
        )
        solutions.append(answer)
    best = min(solutions, key=lambda answer: float(np.dot(answer.fun, answer.fun)))
    B0, tc = np.exp(best.x)

    def diagnostics(fit_fractions):
        q_eval = np.asarray([
            stress_at_decay_fraction(surface, T_K, fraction)
            for fraction in fit_fractions
        ])
        native_barrier, native_volume = _native_values(surface, T_K, q_eval)
        fit_barrier, fit_volume = bilinear_barrier_and_volume(q_eval, B0, tc)
        barrier_error = fit_barrier - native_barrier
        log_rate_error = -barrier_error / kT
        volume_error = fit_volume - native_volume
        return {
            "stress_Pa": q_eval.tolist(),
            "barrier_error_eV": barrier_error.tolist(),
            "log_elementary_rate_error": log_rate_error.tolist(),
            "activation_volume_error_eV_Pa": volume_error.tolist(),
            "barrier_error_rms_eV": float(np.sqrt(np.mean(barrier_error ** 2))),
            "barrier_error_max_abs_eV": float(np.max(np.abs(barrier_error))),
            "log_rate_error_rms": float(np.sqrt(np.mean(log_rate_error ** 2))),
            "log_rate_error_max_abs": float(np.max(np.abs(log_rate_error))),
            "activation_volume_error_rms_eV_Pa": float(np.sqrt(np.mean(volume_error ** 2))),
            "activation_volume_error_max_abs_eV_Pa": float(np.max(np.abs(volume_error))),
        }

    objective_values = sorted(float(np.dot(item.fun, item.fun)) for item in solutions)
    return {
        "temperature_K": T_K,
        "B0_A_times_Gamma_eV": float(B0),
        "peak_traction_Pa": float(tc),
        "fit_window_stress_Pa": [float(np.min(q)), float(np.max(q))],
        "q_span_Pa": q_span,
        "beta": float(beta),
        "training_decay_fractions": list(fractions),
        "held_out_decay_fractions": list(HELD_OUT_DECAY_FRACTIONS),
        "objective": objective_values[0],
        "multistart_count": len(solutions),
        "second_best_objective": objective_values[1],
        "training": diagnostics(fractions),
        "held_out": diagnostics(HELD_OUT_DECAY_FRACTIONS),
        "activation_area_m2": None,
        "separation_work_J_m2": None,
        "final_separation_m": None,
        "physical_support_status": "UNIDENTIFIED_REQUIRES_AUTHORITATIVE_ACTIVATION_AREA",
    }


def fit_temperature_law(surface, temperatures_K=FIT_TEMPERATURES_K) -> dict[str, object]:
    slices = [fit_temperature_slice(surface, T_K) for T_K in temperatures_K]
    reference = min(slices, key=lambda item: abs(item["temperature_K"] - surface.Tref_K))
    knots = np.asarray([item["temperature_K"] for item in slices], dtype=float)
    log_B0 = np.log([item["B0_A_times_Gamma_eV"] for item in slices])
    log_tc = np.log([item["peak_traction_Pa"] for item in slices])

    def parameters(T_K: float) -> tuple[float, float]:
        if not knots[0] <= T_K <= knots[-1]:
            raise ValueError("bilinear temperature law does not extrapolate")
        return (
            float(np.exp(np.interp(T_K, knots, log_B0))),
            float(np.exp(np.interp(T_K, knots, log_tc))),
        )

    held_temperature_rows = []
    for T_K in 0.5 * (knots[:-1] + knots[1:]):
        q = np.asarray([
            stress_at_decay_fraction(surface, T_K, fraction)
            for fraction in HELD_OUT_DECAY_FRACTIONS
        ])
        B0, tc = parameters(float(T_K))
        native_G, native_V = _native_values(surface, float(T_K), q)
        fitted_G, fitted_V = bilinear_barrier_and_volume(q, B0, tc)
        h = 1.0e-3
        B_minus, tc_minus = parameters(float(T_K - h))
        B_plus, tc_plus = parameters(float(T_K + h))
        fitted_minus = bilinear_barrier_and_volume(q, B_minus, tc_minus)[0]
        fitted_plus = bilinear_barrier_and_volume(q, B_plus, tc_plus)[0]
        fitted_dT = (fitted_plus - fitted_minus) / (2.0 * h)
        native_dT = np.asarray([
            surface.diagnostic(float(stress), float(T_K))[
                "partial_G_partial_T_fixed_stress_state_eV_K"
            ]
            for stress in q
        ], dtype=float)
        kT = KB_EV_PER_K * float(T_K)
        barrier_error = fitted_G - native_G
        volume_error = fitted_V - native_V
        thermal_error = fitted_dT - native_dT
        held_temperature_rows.append({
            "temperature_K": float(T_K),
            "stress_Pa": q.tolist(),
            "B0_A_times_Gamma_eV": B0,
            "peak_traction_Pa": tc,
            "barrier_error_rms_eV": float(np.sqrt(np.mean(barrier_error ** 2))),
            "barrier_error_max_abs_eV": float(np.max(np.abs(barrier_error))),
            "log_rate_error_rms": float(np.sqrt(np.mean((barrier_error / kT) ** 2))),
            "log_rate_error_max_abs": float(np.max(np.abs(barrier_error / kT))),
            "activation_volume_error_rms_eV_Pa": float(np.sqrt(np.mean(volume_error ** 2))),
            "activation_volume_error_max_abs_eV_Pa": float(np.max(np.abs(volume_error))),
            "thermal_slope_error_rms_eV_K": float(np.sqrt(np.mean(thermal_error ** 2))),
            "thermal_slope_error_max_abs_eV_K": float(np.max(np.abs(thermal_error))),
            "thermal_derivative_convention": "fixed physical traction; centered difference of log-linear fit at held-out interval midpoint versus native analytic derivative",
        })
    return {
        "schema": "bilinear-opening-analogue/1",
        "temperature_law": {
            "kind": "piecewise_linear_interpolation_in_log_positive_parameters",
            "extrapolation": "forbidden",
            "knots_K": knots.tolist(),
            "log_B0_eV": log_B0.tolist(),
            "log_peak_traction_Pa": log_tc.tolist(),
        },
        "slices": slices,
        "held_out_temperature_diagnostics": held_temperature_rows,
        "reference_temperature_frozen_control": {
            "temperature_K": reference["temperature_K"],
            "B0_A_times_Gamma_eV": reference["B0_A_times_Gamma_eV"],
            "peak_traction_Pa": reference["peak_traction_Pa"],
        },
        "outside_domain_behavior": "not fitted; EXP-floor retains a positive asymptote while the unmodified bilinear barrier reaches zero at peak traction and is mechanically unstable beyond it",
    }


__all__ = [
    "FIT_TEMPERATURES_K", "HELD_OUT_DECAY_FRACTIONS", "TRAIN_DECAY_FRACTIONS",
    "bilinear_barrier_and_volume", "fit_temperature_law",
    "fit_temperature_slice", "stress_at_decay_fraction",
]
