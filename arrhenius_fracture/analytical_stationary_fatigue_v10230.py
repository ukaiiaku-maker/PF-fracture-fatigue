"""Parameter-free analytical reductions of the v10.2.30 fatigue model.

This module is analysis-only.  It evaluates the production EXP-floor and
cooperative-renewal equations by cycle quadrature and supplies three transparent
levels: A0 (opening only), A1 (persistent emission/blunting balance), and A2
(A1 plus the phase-averaged Peierls/Taylor mobile-retained balance).

The spatial production state is intentionally reduced to cycle-averaged moments.
Consequently A1/A2 are steady-state closures to be tested, not alternate
production constitutive laws.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Mapping

import numpy as np
from scipy.special import gammainc

from .material_manifest import ExpFloorBarrier, MaterialManifest


MODEL_ID = "v10.2.30_parameter_free_stationary_analytical_hierarchy_v1"


@dataclass(frozen=True)
class AnalyticalControls:
    n_phase: int = 4096
    r0_m: float = 1.0e-6
    sigma_cap_Pa: float = 30.0e9
    burgers_m: float = 2.74e-10
    shear_modulus_Pa: float = 160.0e9
    taylor_stress_fraction: float = 1.0 / math.sqrt(3.0)
    forest_floor_m2: float = 5.0e12
    blunting_length_m: float = 0.5e-6
    mpz_length_m: float = 50.0e-6
    reference_density_m2: float = 5.0e12
    reference_front_width_m: float = 10.0e-6
    reference_source_area_m2: float = 25.0e-12
    source_zone_length_m: float = 2.0e-6
    n_systems: int = 2
    emission_drive_factor: float = 1.0
    cleavage_hits: float = 3.0
    cleavage_tau_s: float = 1.0e-6
    mean_event_length_m: float = 5.0e-6
    jump_fraction: float = 1.0
    max_iterations: int = 400
    relative_tolerance: float = 1.0e-9
    damping: float = 0.25


def _phase(n: int) -> np.ndarray:
    count = max(int(n), 32)
    return 2.0 * math.pi * (np.arange(count, dtype=float) + 0.5) / count


def waveform_K(
    Kmax_Pa_sqrt_m: float, R: float, n_phase: int, closure_clip: bool = True
) -> np.ndarray:
    phase = _phase(n_phase)
    Kmin = float(R) * float(Kmax_Pa_sqrt_m)
    K = 0.5 * (float(Kmax_Pa_sqrt_m) + Kmin) + 0.5 * (
        float(Kmax_Pa_sqrt_m) - Kmin
    ) * np.cos(phase)
    return np.maximum(K, 0.0) if closure_clip else K


def opening_rate_s(
    manifest: MaterialManifest,
    stress_Pa: np.ndarray,
    temperature_K: float,
    controls: AnalyticalControls,
) -> tuple[np.ndarray, np.ndarray]:
    raw = np.asarray(manifest.cleavage.rate(stress_Pa, temperature_K), dtype=float)
    m = max(float(controls.cleavage_hits), 1.0)
    if m > 1.0 + 1.0e-12:
        x = np.minimum(np.maximum(raw, 0.0) * controls.cleavage_tau_s, 1.0e12)
        effective = gammainc(m, x) / controls.cleavage_tau_s
    else:
        effective = raw
    return np.asarray(effective, dtype=float), raw


def cycle_opening(
    manifest: MaterialManifest,
    Kmax_MPa_sqrt_m: float,
    R: float,
    temperature_K: float,
    frequency_Hz: float,
    radius_m: float,
    controls: AnalyticalControls,
    K_shield_Pa_sqrt_m: float = 0.0,
) -> dict[str, float]:
    K = waveform_K(Kmax_MPa_sqrt_m * 1.0e6, R, controls.n_phase)
    Keff = np.maximum(K - float(K_shield_Pa_sqrt_m), 0.0)
    sigma = np.minimum(
        Keff / math.sqrt(2.0 * math.pi * max(float(radius_m), 1.0e-30)),
        controls.sigma_cap_Pa,
    )
    effective, raw = opening_rate_s(manifest, sigma, temperature_K, controls)
    mu = float(np.mean(effective) / float(frequency_Hz))
    return {
        "mu_open": mu,
        "q_open": mu,
        "mu_open_raw": float(np.mean(raw) / float(frequency_Hz)),
        "lambda_open_peak_s": float(np.max(effective)),
        "sigma_open_peak_Pa": float(np.max(sigma)),
        "da_dN": controls.mean_event_length_m * mu,
    }


def _transport_surface_average(
    surface: ExpFloorBarrier,
    stress: np.ndarray,
    temperature_K: float,
) -> float:
    return float(np.mean(np.asarray(surface.rate(stress, temperature_K), dtype=float)))


def _state_rates(
    manifest: MaterialManifest,
    rho_site0_m2: float,
    K: np.ndarray,
    temperature_K: float,
    frequency_Hz: float,
    radius_m: float,
    slip_count: float,
    growth_m_per_cycle: float,
    controls: AnalyticalControls,
) -> dict[str, float]:
    # The reduced density and residence length are the exact moments used by the
    # production local backstress/blunting kernels.  No coefficient is fit here.
    Lq = max(controls.blunting_length_m, controls.burgers_m)
    rho_state = max(
        controls.forest_floor_m2 + max(slip_count, 0.0) / (Lq * Lq), 1.0
    )
    sigma_open = np.minimum(
        np.maximum(K, 0.0)
        / math.sqrt(2.0 * math.pi * max(radius_m, controls.r0_m)),
        controls.sigma_cap_Pa,
    )
    sigma_back = (
        controls.shear_modulus_Pa
        * controls.burgers_m
        * math.sqrt(max(rho_state - controls.forest_floor_m2, 0.0))
        / max(abs(controls.taylor_stress_fraction), 1.0e-6)
    )
    sigma_emit = np.maximum(
        controls.emission_drive_factor * sigma_open - sigma_back, 0.0
    )
    emit_site = np.asarray(manifest.emission.rate(sigma_emit, temperature_K), dtype=float)

    width = controls.reference_front_width_m * math.sqrt(
        controls.reference_density_m2 / max(rho_state, controls.reference_density_m2)
    )
    width = min(max(width, Lq), controls.mpz_length_m)
    arc = controls.reference_source_area_m2 / (
        controls.r0_m * controls.reference_front_width_m
    )
    # Persistent-site density remains a frozen-row input outside MaterialManifest.
    multiplicity = max(float(rho_site0_m2), 0.0)
    multiplicity *= arc * radius_m * width
    emit_per_cycle = (
        controls.n_systems * multiplicity * float(np.mean(emit_site)) / frequency_Hz
    )

    peierls_surface = manifest.peierls.as_surface(manifest.emission)
    taylor_surface = manifest.taylor.as_surface(manifest.emission)
    tau_p = controls.taylor_stress_fraction * sigma_open
    k_peierls = _transport_surface_average(
        peierls_surface, tau_p, temperature_K
    )
    spacing = 1.0 / (2.0 * math.sqrt(rho_state))
    phi = spacing / controls.burgers_m
    tau_t = controls.taylor_stress_fraction * sigma_open * phi
    t_single = np.asarray(taylor_surface.rate(tau_t, temperature_K), dtype=float)
    m_eff = 1.0 + max(manifest.taylor_corr_scale, 0.0) * max(
        math.sqrt(rho_state / max(manifest.taylor_corr_rho_c_m2, 1.0)) - 1.0,
        0.0,
    )
    k_taylor = float(np.mean(gammainc(max(m_eff, 1.0), np.minimum(t_single, 1.0e12))))
    jump = controls.jump_fraction * spacing
    velocity = jump * k_peierls
    k_enc = max(manifest.encounter_efficiency, 0.0) * velocity * math.sqrt(rho_state)
    k_escape = velocity / max(controls.mpz_length_m, 1.0e-30)
    k_adv = frequency_Hz * max(growth_m_per_cycle, 0.0) / max(
        controls.mpz_length_m, 1.0e-30
    )
    return {
        "rho_state_m2": rho_state,
        "sigma_back_Pa": sigma_back,
        "mu_emit": max(emit_per_cycle, 0.0),
        "lambda_emit_peak_s": float(np.max(emit_site)),
        "multiplicity_per_system": multiplicity,
        "peierls_rate_s": max(k_peierls, 0.0),
        "taylor_completion_rate_s": max(k_taylor, 0.0),
        "encounter_rate_s": max(k_enc, 0.0),
        "escape_rate_s": max(k_escape, 0.0),
        "advance_loss_rate_s": max(k_adv, 0.0),
        "taylor_m_eff": m_eff,
    }


def solve_hierarchy(
    manifest: MaterialManifest,
    row: Mapping[str, str | float],
    Kmax_MPa_sqrt_m: float,
    R: float,
    temperature_K: float,
    frequency_Hz: float,
    controls: AnalyticalControls = AnalyticalControls(),
) -> dict[str, float | bool | int | str]:
    rho_site0_m2 = float(row["rho_source0_m2"])
    a0 = cycle_opening(
        manifest, Kmax_MPa_sqrt_m, R, temperature_K, frequency_Hz,
        controls.r0_m, controls,
    )
    K = waveform_K(Kmax_MPa_sqrt_m * 1.0e6, R, controls.n_phase)

    def balance(value: float) -> tuple[float, dict[str, float], dict[str, float]]:
        radius = controls.r0_m + manifest.c_blunt * controls.burgers_m * max(value, 0.0)
        opening = cycle_opening(
            manifest, Kmax_MPa_sqrt_m, R, temperature_K, frequency_Hz,
            radius, controls,
        )
        rates = _state_rates(
            manifest, rho_site0_m2, K, temperature_K, frequency_Hz, radius, value,
            float(opening["da_dN"]), controls,
        )
        loss_fraction = max(float(opening["da_dN"]) / controls.blunting_length_m, 1.0e-300)
        target = max(rates["mu_emit"] / loss_fraction, 0.0)
        return value - target, opening, rates

    f0, _, _ = balance(0.0)
    bracket_high = 1.0e-12
    fhigh = f0
    iteration = 0
    while iteration < controls.max_iterations:
        iteration += 1
        fhigh, _, _ = balance(bracket_high)
        if math.isfinite(fhigh) and fhigh >= 0.0:
            break
        bracket_high *= 10.0
    converged = math.isfinite(fhigh) and f0 <= 0.0 and fhigh >= 0.0
    if converged:
        lo, hi = 0.0, bracket_high
        for _ in range(controls.max_iterations):
            mid = 0.5 * (lo + hi)
            fmid, _, _ = balance(mid)
            if fmid >= 0.0:
                hi = mid
            else:
                lo = mid
            if hi - lo <= controls.relative_tolerance * max(1.0, hi):
                break
        slip = 0.5 * (lo + hi)
        residual_value, a1, state = balance(slip)
        last = abs(residual_value) / max(1.0, abs(slip))
    else:
        slip = math.nan
        a1 = {name: math.nan for name in a0}
        state = {
            name: math.nan for name in (
                "mu_emit", "sigma_back_Pa", "peierls_rate_s",
                "taylor_completion_rate_s", "encounter_rate_s",
                "escape_rate_s", "advance_loss_rate_s",
            )
        }

    radius = (
        controls.r0_m + manifest.c_blunt * controls.burgers_m * max(slip, 0.0)
        if math.isfinite(slip) else math.nan
    )

    R_emit_s = state["mu_emit"] * frequency_Hz
    A = state["encounter_rate_s"] + state["escape_rate_s"] + state["advance_loss_rate_s"]
    B = state["taylor_completion_rate_s"] + manifest.retained_recovery_rate_s + state["advance_loss_rate_s"]
    determinant = max(A * B - state["taylor_completion_rate_s"] * state["encounter_rate_s"], 1.0e-300)
    mobile = R_emit_s * B / determinant
    retained = R_emit_s * state["encounter_rate_s"] / determinant
    # The two reduced signed channels are symmetry paired in the stationary mean;
    # their retained K projections cancel.  Nonzero archived shielding is retained
    # as a validation residual rather than inferred from da/dN.
    K_shield = 0.0
    a2 = (
        cycle_opening(
            manifest, Kmax_MPa_sqrt_m, R, temperature_K, frequency_Hz,
            radius, controls, K_shield,
        ) if converged else {name: math.nan for name in a0}
    )
    return {
        "model_id": MODEL_ID,
        "A0_da_dN": a0["da_dN"],
        "A1_da_dN": a1["da_dN"],
        "A2_da_dN": a2["da_dN"],
        "A0_mu_open": a0["mu_open"],
        "A1_mu_open": a1["mu_open"],
        "A2_mu_open": a2["mu_open"],
        "mu_open_raw": a2["mu_open_raw"],
        "mu_emit": state["mu_emit"],
        "stationary_net_source_slip": slip,
        "stationary_mobile": mobile,
        "stationary_retained": retained,
        "r_eff_m": radius,
        "sigma_back_Pa": state["sigma_back_Pa"],
        "K_shield_Pa_sqrt_m": K_shield,
        "peierls_rate_s": state["peierls_rate_s"],
        "taylor_completion_rate_s": state["taylor_completion_rate_s"],
        "encounter_rate_s": state["encounter_rate_s"],
        "escape_rate_s": state["escape_rate_s"],
        "retained_fraction": retained / max(mobile + retained, 1.0e-300),
        "physical_return_fraction": 0.0,
        "fixed_point_converged": converged,
        "fixed_point_iterations": iteration,
        "fixed_point_residual": last,
        "event_length_mean_m": controls.mean_event_length_m,
        "event_length_semantics": "mean_preserving_threshold_correlated_proposal",
    }


def controls_dict(controls: AnalyticalControls) -> dict[str, float | int]:
    return asdict(controls)


__all__ = [
    "MODEL_ID", "AnalyticalControls", "controls_dict", "cycle_opening",
    "opening_rate_s", "solve_hierarchy", "waveform_K",
]
