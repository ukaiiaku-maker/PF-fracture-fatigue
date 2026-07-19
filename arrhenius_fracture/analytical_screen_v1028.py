"""Fast mechanics-informed screening before state-resolved 1-D calibration.

This module is deliberately a screening model, not a replacement constitutive
law. It uses the production material barriers, the serialized local tip geometry,
the authorized state-resolved signed drive family, and the authorized signed
shielding family. It integrates no-feedback cleavage and emission hazards under a
K ramp, then reports linearized source-bin shielding and transport/retention
indicators. Every promoted candidate must still pass the exact shared 1-D engine.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from scipy.special import gammainc

from .material_manifest import MaterialManifest
from .signed_kernel_family_v1026 import StateResolvedSignedShieldingKernelFamily
from .state_resolved_drive_family_v1027 import StateResolvedSignedDriveFamily

MODEL_ID = "v10.2.8_mechanics_informed_analytical_screen"
DEFAULT_ANALYTICAL_TEMPERATURES_K = tuple(float(T) for T in range(300, 1201, 100))
DBTT_FIRST_PASSAGE_TEMPERATURES_K = (300.0, 700.0, 900.0, 1200.0)
WEAKT_FIRST_PASSAGE_TEMPERATURES_K = (300.0, 700.0, 1200.0)


@dataclass(frozen=True)
class AnalyticalMechanics:
    r0_m: float
    sigma_cap_Pa: float
    cleavage_hits: float
    cleavage_tau_s: float
    source_bin_count: int

    @classmethod
    def from_engine_config(cls, path_or_payload: str | Path | dict[str, Any]):
        payload = (
            json.loads(Path(path_or_payload).read_text())
            if isinstance(path_or_payload, (str, Path))
            else dict(path_or_payload)
        )
        front = dict(payload["front_config"])
        mpz = dict(payload["mpz_config"])
        result = cls(
            r0_m=float(front["r0"]),
            sigma_cap_Pa=float(front["sigma_cap"]),
            cleavage_hits=float(front["m_hits"]),
            cleavage_tau_s=float(front["tau_c"]),
            source_bin_count=int(mpz["source_bin_count"]),
        )
        if result.r0_m <= 0.0 or result.sigma_cap_Pa <= 0.0:
            raise ValueError("analytical screen requires positive r0 and sigma_cap")
        if result.cleavage_hits <= 0.0 or result.cleavage_tau_s <= 0.0:
            raise ValueError("invalid cleavage renewal configuration")
        if result.source_bin_count < 1:
            raise ValueError("source_bin_count must be positive")
        return result


@dataclass(frozen=True)
class AnalyticalControl:
    Kdot_MPa_sqrt_m_s: float = 0.005
    Kmax_MPa_sqrt_m: float = 80.0
    dK_MPa_sqrt_m: float = 0.05
    temperatures_K: tuple[float, ...] = DEFAULT_ANALYTICAL_TEMPERATURES_K

    def validate(self) -> "AnalyticalControl":
        if self.Kdot_MPa_sqrt_m_s <= 0.0:
            raise ValueError("Kdot must be positive")
        if self.Kmax_MPa_sqrt_m <= 0.0 or self.dK_MPa_sqrt_m <= 0.0:
            raise ValueError("Kmax and dK must be positive")
        temperatures = tuple(float(T) for T in self.temperatures_K)
        if not temperatures or any(T <= 0.0 for T in temperatures):
            raise ValueError("analytical temperatures must be positive")
        if any(b <= a for a, b in zip(temperatures, temperatures[1:])):
            raise ValueError("analytical temperatures must be strictly increasing")
        object.__setattr__(self, "temperatures_K", temperatures)
        return self


def _local_state(K_MPa_sqrt_m: float, mechanics: AnalyticalMechanics) -> tuple[float, dict[str, float]]:
    K_Pa_sqrt_m = max(float(K_MPa_sqrt_m), 0.0) * 1.0e6
    sigma_uncapped = K_Pa_sqrt_m / math.sqrt(2.0 * math.pi * mechanics.r0_m)
    sigma_local = min(max(sigma_uncapped, 0.0), mechanics.sigma_cap_Pa)
    return sigma_local, {
        "r_eff_over_r0": 1.0,
        "opening_strength_fraction": sigma_local / mechanics.sigma_cap_Pa,
        "crack_extension_m": 0.0,
    }


def _effective_cleavage_rate(manifest: MaterialManifest, sigma_Pa: float, T_K: float,
                             mechanics: AnalyticalMechanics) -> float:
    raw = float(np.asarray(manifest.cleavage.rate(sigma_Pa, T_K)))
    m = max(mechanics.cleavage_hits, 1.0)
    if m <= 1.0 + 1.0e-12:
        return max(raw, 0.0)
    argument = min(max(raw * mechanics.cleavage_tau_s, 0.0), 1.0e12)
    return float(gammainc(m, argument) / mechanics.cleavage_tau_s)


def _integrate_temperature(
    manifest: MaterialManifest,
    T_K: float,
    mechanics: AnalyticalMechanics,
    control: AnalyticalControl,
    kernel: StateResolvedSignedShieldingKernelFamily,
    drive: StateResolvedSignedDriveFamily,
) -> dict[str, Any]:
    dt_per_MPa = 1.0 / control.Kdot_MPa_sqrt_m_s
    n_systems = drive.n_systems
    site_capacity = np.full(n_systems, float(manifest.source_sites_per_system))
    cumulative_site_hazard = np.zeros(n_systems, dtype=float)
    cumulative_cleavage = 0.0
    cumulative_emission_first = 0.0
    K_cleave = math.nan
    K_emit = math.nan
    K_left = 0.0
    last_factors = np.zeros(n_systems)
    last_sigma = 0.0

    while K_left < control.Kmax_MPa_sqrt_m - 1.0e-15:
        dK = min(control.dK_MPa_sqrt_m, control.Kmax_MPa_sqrt_m - K_left)
        K_mid = K_left + 0.5 * dK
        sigma_local, coordinates = _local_state(K_mid, mechanics)
        signed_factors = drive.resolve(**coordinates)
        rates_site = np.asarray(
            manifest.emission.rate(np.abs(signed_factors) * sigma_local, T_K),
            dtype=float,
        )
        rates_site = np.maximum(rates_site, 0.0)
        rates_site[np.sign(signed_factors) == 0.0] = 0.0
        dt = dK * dt_per_MPa
        previous_c = cumulative_cleavage
        previous_e = cumulative_emission_first
        cumulative_cleavage += _effective_cleavage_rate(
            manifest, sigma_local, T_K, mechanics
        ) * dt
        cumulative_site_hazard += rates_site * dt
        cumulative_emission_first += float(np.sum(site_capacity * rates_site)) * dt
        if not math.isfinite(K_cleave) and cumulative_cleavage >= 1.0:
            fraction = (1.0 - previous_c) / max(cumulative_cleavage - previous_c, 1.0e-300)
            K_cleave = K_left + min(max(fraction, 0.0), 1.0) * dK
        if not math.isfinite(K_emit) and cumulative_emission_first >= 1.0:
            fraction = (1.0 - previous_e) / max(cumulative_emission_first - previous_e, 1.0e-300)
            K_emit = K_left + min(max(fraction, 0.0), 1.0) * dK
        last_factors = signed_factors
        last_sigma = sigma_local
        K_left += dK
        if math.isfinite(K_cleave) and K_left >= K_cleave:
            break

    expected_activations = site_capacity * (1.0 - np.exp(-cumulative_site_hazard))
    line_content = expected_activations * kernel.activation_to_line_content
    evaluation_K = K_cleave if math.isfinite(K_cleave) else control.Kmax_MPa_sqrt_m
    sigma_eval, coordinates_eval = _local_state(evaluation_K, mechanics)
    signed_factors_eval = drive.resolve(**coordinates_eval)
    active_kernel, _ = kernel.resolve(**coordinates_eval)
    nsrc = min(max(mechanics.source_bin_count, 1), active_kernel.shape[1])
    source_operator = np.mean(active_kernel[:, :nsrc], axis=1)
    signed_line = np.sign(signed_factors_eval) * line_content
    Kshield_linear = float(np.sum(source_operator * signed_line)) / 1.0e6

    peierls_surface = manifest.peierls.as_surface(manifest.emission)
    taylor_surface = manifest.taylor.as_surface(manifest.emission)
    resolved_stress = np.abs(signed_factors_eval) * sigma_eval
    escape_rate = np.asarray(peierls_surface.rate(resolved_stress, T_K), dtype=float)
    trap_rate = manifest.encounter_efficiency * np.asarray(
        taylor_surface.rate(resolved_stress, T_K), dtype=float
    )
    recovery_rate = max(float(manifest.retained_recovery_rate_s), 0.0)
    denominator = escape_rate + trap_rate + recovery_rate
    retained_fraction = np.divide(
        trap_rate,
        denominator,
        out=np.zeros_like(trap_rate),
        where=denominator > 0.0,
    )
    emission_advantage = (
        (K_cleave - K_emit) / K_cleave
        if math.isfinite(K_cleave) and math.isfinite(K_emit) and K_cleave > 0.0
        else math.nan
    )
    return {
        "temperature_K": float(T_K),
        "K_cleave_no_plastic_MPa_sqrt_m": K_cleave,
        "K_first_emission_MPa_sqrt_m": K_emit,
        "emission_advantage_fraction": emission_advantage,
        "expected_source_activations": float(np.sum(expected_activations)),
        "expected_signed_line_content": float(np.sum(signed_line)),
        "linearized_source_bin_Kshield_MPa_sqrt_m": Kshield_linear,
        "mean_retained_fraction_indicator": float(np.mean(retained_fraction)),
        "mean_escape_rate_s": float(np.mean(escape_rate)),
        "mean_trap_rate_s": float(np.mean(trap_rate)),
        "source_capacity_total": float(np.sum(site_capacity)),
        "opening_strength_fraction_at_cleavage": float(
            coordinates_eval["opening_strength_fraction"]
        ),
        "drive_signed_factors_at_cleavage": signed_factors_eval.tolist(),
        "last_sigma_local_Pa": float(last_sigma),
        "last_drive_signed_factors": last_factors.tolist(),
    }


def analytical_screen(
    manifest: MaterialManifest,
    mechanics: AnalyticalMechanics,
    control: AnalyticalControl,
    kernel: StateResolvedSignedShieldingKernelFamily,
    drive: StateResolvedSignedDriveFamily,
    *,
    target_class: str,
) -> dict[str, Any]:
    control = control.validate()
    drive.validate_against_kernel_family(kernel)
    details = [
        _integrate_temperature(manifest, T, mechanics, control, kernel, drive)
        for T in control.temperatures_K
    ]
    cleavage = np.asarray(
        [row["K_cleave_no_plastic_MPa_sqrt_m"] for row in details], dtype=float
    )
    advantage = np.asarray(
        [row["emission_advantage_fraction"] for row in details], dtype=float
    )
    shielding = np.asarray(
        [row["linearized_source_bin_Kshield_MPa_sqrt_m"] for row in details],
        dtype=float,
    )
    finite = bool(np.all(np.isfinite(cleavage)))
    target = str(target_class).strip().lower()
    if target == "dbtt":
        low_adv = float(advantage[0]) if np.isfinite(advantage[0]) else -1.0
        high_adv = float(advantage[-1]) if np.isfinite(advantage[-1]) else -1.0
        transition_fraction = float(np.mean(np.diff(advantage) >= -0.02)) if np.all(np.isfinite(advantage)) else 0.0
        strict = bool(
            finite
            and 5.0 <= cleavage[0] <= 30.0
            and low_adv <= 0.10
            and high_adv >= 0.05
            and transition_fraction >= 0.70
            and shielding[-1] >= 0.5
        )
        objective = (
            10.0 * max(0.0, low_adv - 0.10) ** 2
            + 20.0 * max(0.0, 0.05 - high_adv) ** 2
            + 10.0 * max(0.0, 0.70 - transition_fraction) ** 2
            + 2.0 * max(0.0, 0.5 - shielding[-1]) ** 2
        )
        summary = {
            "analytical_pass": strict,
            "analytical_objective": float(objective + (0.0 if finite else 1.0e6)),
            "low_emission_advantage_fraction": low_adv,
            "high_emission_advantage_fraction": high_adv,
            "emission_advantage_monotonic_fraction": transition_fraction,
            "high_linearized_Kshield_MPa_sqrt_m": float(shielding[-1]),
        }
    elif target in {"weakt", "weak_t", "fcc_like"}:
        span = float(np.max(cleavage) / np.min(cleavage)) if finite and np.min(cleavage) > 0.0 else math.nan
        shield_fraction = np.abs(shielding) / np.maximum(cleavage, 1.0e-12)
        advantage_span = (
            float(np.max(advantage) - np.min(advantage))
            if np.all(np.isfinite(advantage))
            else math.nan
        )
        strict = bool(
            finite
            and span <= 1.25
            and math.isfinite(advantage_span)
            and advantage_span <= 0.30
            and float(np.mean(shield_fraction)) >= 0.02
            and float(np.mean(shield_fraction)) <= 0.35
        )
        objective = (
            20.0 * max(0.0, (span if math.isfinite(span) else 10.0) - 1.25) ** 2
            + 10.0 * max(0.0, (advantage_span if math.isfinite(advantage_span) else 10.0) - 0.30) ** 2
            + 10.0 * max(0.0, 0.02 - float(np.mean(shield_fraction))) ** 2
            + 5.0 * max(0.0, float(np.mean(shield_fraction)) - 0.35) ** 2
        )
        summary = {
            "analytical_pass": strict,
            "analytical_objective": float(objective + (0.0 if finite else 1.0e6)),
            "cleavage_temperature_span_ratio": span,
            "emission_advantage_span": advantage_span,
            "mean_linearized_shield_fraction": float(np.mean(shield_fraction)),
        }
    else:
        raise ValueError(f"unsupported analytical target class {target_class!r}")
    return {
        "schema": MODEL_ID,
        "candidate_id": manifest.candidate_id,
        "target_class": target_class,
        "temperatures_K": list(control.temperatures_K),
        "screen_is_nonbinding": True,
        "screen_omits_spatial_transport_feedback": True,
        "details": details,
        **summary,
    }


__all__ = [
    "MODEL_ID",
    "DEFAULT_ANALYTICAL_TEMPERATURES_K",
    "DBTT_FIRST_PASSAGE_TEMPERATURES_K",
    "WEAKT_FIRST_PASSAGE_TEMPERATURES_K",
    "AnalyticalMechanics",
    "AnalyticalControl",
    "analytical_screen",
]
