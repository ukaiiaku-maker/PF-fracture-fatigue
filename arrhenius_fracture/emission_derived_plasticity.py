"""Emission-derived Peierls--Taylor kinetics for the full-field FEM update.

The moving process-zone model and the surrounding FEM field use the same
candidate-specific Arrhenius surfaces.  This module contains no fitted bulk
parameters: a v10.2.19 entry installs the exact emission, Peierls and Taylor
surfaces from the selected v9.11.1 registry row on ``DislocationConfig``.

Peierls glide and correlated Taylor release are sequential kinetic bottlenecks.
The Taylor single-hit clock is converted to a multi-hit completion rate over the
existing correlation renewal time, then the two rates are combined by their
series waiting time.  The resulting velocity enters the Orowan relation.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import numpy as np
from scipy.special import gammainc

from .material_manifest import KB_EV_PER_K, MaterialManifest


@dataclass(frozen=True)
class ExpFloorRateSurface:
    G00_eV: float
    gT_eV_per_K: float
    sigc0_Pa: float
    sT_Pa_per_K: float
    alpha: float
    exponent: float
    floor_fraction: float
    floor_min_eV: float
    floor_max_fraction: float
    Tref_K: float
    attempt_frequency_s: float

    def barrier_eV(self, stress_Pa: np.ndarray | float, temperature_K: float) -> np.ndarray:
        sigma = np.maximum(np.asarray(stress_Pa, dtype=float), 0.0)
        dT = float(temperature_K) - float(self.Tref_K)
        G0 = max(float(self.G00_eV) + float(self.gT_eV_per_K) * dT, 1.0e-12)
        sigc = max(float(self.sigc0_Pa) + float(self.sT_Pa_per_K) * dT, 1.0)
        raw_floor = max(float(self.floor_min_eV), float(self.floor_fraction) * G0)
        floor = min(float(self.floor_max_fraction) * G0, raw_floor)
        x = np.maximum(sigma / sigc, 0.0)
        return np.maximum(
            floor + (G0 - floor) * np.exp(
                -max(float(self.alpha), 0.0)
                * np.power(x, max(float(self.exponent), 1.0e-12))
            ),
            0.0,
        )

    def rate_s(self, stress_Pa: np.ndarray | float, temperature_K: float) -> np.ndarray:
        G = self.barrier_eV(stress_Pa, temperature_K)
        denom = max(KB_EV_PER_K * float(temperature_K), 1.0e-30)
        return max(float(self.attempt_frequency_s), 0.0) * np.exp(
            np.clip(-G / denom, -700.0, 0.0)
        )


@dataclass(frozen=True)
class EmissionDerivedPeierlsTaylorConfig:
    peierls: ExpFloorRateSurface
    taylor: ExpFloorRateSurface
    peierls_stress_fraction: float
    taylor_stress_fraction: float
    taylor_corr_rho_c_m2: float
    taylor_corr_scale: float
    taylor_renewal_time_s: float
    taylor_m_cap: float
    taylor_phi_max: float
    mobile_fraction: float
    mobile_saturation_density_m2: float
    mobile_density_floor_m2: float
    jump_fraction: float
    jump_length_min_m: float
    equivalent_strain_factor: float
    exact_manifest_mapping: bool
    candidate_id: str

    def validate(self) -> "EmissionDerivedPeierlsTaylorConfig":
        positive = {
            "taylor_corr_rho_c_m2": self.taylor_corr_rho_c_m2,
            "taylor_renewal_time_s": self.taylor_renewal_time_s,
            "taylor_phi_max": self.taylor_phi_max,
            "mobile_saturation_density_m2": self.mobile_saturation_density_m2,
            "mobile_density_floor_m2": self.mobile_density_floor_m2,
            "jump_length_min_m": self.jump_length_min_m,
        }
        for name, value in positive.items():
            if not math.isfinite(float(value)) or float(value) <= 0.0:
                raise ValueError(f"{name} must be finite and positive; got {value!r}")
        if not math.isfinite(float(self.taylor_m_cap)) and not math.isinf(float(self.taylor_m_cap)):
            raise ValueError("taylor_m_cap must be finite or infinity")
        if float(self.mobile_fraction) < 0.0:
            raise ValueError("mobile_fraction must be nonnegative")
        if float(self.jump_fraction) < 0.0:
            raise ValueError("jump_fraction must be nonnegative")
        if float(self.equivalent_strain_factor) <= 0.0:
            raise ValueError("equivalent_strain_factor must be positive")
        return self


def _surface_from_prefix(disl_cfg: Any, prefix: str) -> ExpFloorRateSurface:
    return ExpFloorRateSurface(
        G00_eV=float(getattr(disl_cfg, f"{prefix}_G00_eV")),
        gT_eV_per_K=float(getattr(disl_cfg, f"{prefix}_gT_eV_per_K")),
        sigc0_Pa=float(getattr(disl_cfg, f"{prefix}_sigc0_Pa")),
        sT_Pa_per_K=float(getattr(disl_cfg, f"{prefix}_sT_Pa_per_K")),
        alpha=float(getattr(disl_cfg, f"{prefix}_exp_a")),
        exponent=float(getattr(disl_cfg, f"{prefix}_exp_n")),
        floor_fraction=float(getattr(disl_cfg, f"{prefix}_floor_frac")),
        floor_min_eV=float(getattr(disl_cfg, f"{prefix}_floor_min_eV")),
        floor_max_fraction=float(getattr(disl_cfg, f"{prefix}_floor_max_frac")),
        Tref_K=float(getattr(disl_cfg, f"{prefix}_Tref_K")),
        attempt_frequency_s=float(getattr(disl_cfg, f"{prefix}_nu0_s")),
    )


def _fallback_surface(disl_cfg: Any, branch: str) -> ExpFloorRateSurface:
    parent_G = max(float(getattr(disl_cfg, "pt_emit_G00_eV", 1.0)), 1.0e-12)
    energy_ratio = float(getattr(disl_cfg, f"pt_{branch}_energy_ratio", 1.0))
    entropy_ratio = float(getattr(disl_cfg, f"pt_{branch}_entropy_ratio", energy_ratio))
    stress_ratio = float(getattr(disl_cfg, f"pt_{branch}_stress_ratio", 1.0))
    G00 = max(parent_G * energy_ratio, 1.0e-12)
    floor_min_parent = float(getattr(disl_cfg, "pt_emit_floor_min_eV", 1.0e-4))
    return ExpFloorRateSurface(
        G00_eV=G00,
        gT_eV_per_K=float(getattr(disl_cfg, "pt_emit_gT_eV_per_K", 0.0)) * entropy_ratio,
        sigc0_Pa=max(float(getattr(disl_cfg, "pt_emit_sigc0_Pa", 1.0)) * stress_ratio, 1.0),
        sT_Pa_per_K=float(getattr(disl_cfg, "pt_emit_sT_Pa_per_K", 0.0)) * stress_ratio,
        alpha=float(getattr(disl_cfg, "pt_emit_exp_a", 1.0)),
        exponent=float(getattr(disl_cfg, "pt_emit_exp_n", 1.0)),
        floor_fraction=float(getattr(disl_cfg, "pt_emit_floor_frac", 0.02)),
        floor_min_eV=floor_min_parent * G00 / parent_G,
        floor_max_fraction=float(getattr(disl_cfg, "pt_emit_floor_max_frac", 0.95)),
        Tref_K=float(getattr(disl_cfg, "pt_emit_Tref_K", 481.33)),
        attempt_frequency_s=float(getattr(disl_cfg, f"pt_{branch}_nu0_s", 1.0e12)),
    )


def config_from_dislocation_config(disl_cfg: Any) -> EmissionDerivedPeierlsTaylorConfig:
    exact = bool(getattr(disl_cfg, "pt_exact_manifest_mapping", False))
    if exact:
        peierls = _surface_from_prefix(disl_cfg, "pt_exact_peierls")
        taylor = _surface_from_prefix(disl_cfg, "pt_exact_taylor")
    else:
        peierls = _fallback_surface(disl_cfg, "peierls")
        taylor = _fallback_surface(disl_cfg, "taylor")
    return EmissionDerivedPeierlsTaylorConfig(
        peierls=peierls,
        taylor=taylor,
        peierls_stress_fraction=float(getattr(disl_cfg, "pt_peierls_stress_fraction", 1.0 / math.sqrt(3.0))),
        taylor_stress_fraction=float(getattr(disl_cfg, "pt_taylor_stress_fraction", 1.0 / math.sqrt(3.0))),
        taylor_corr_rho_c_m2=float(getattr(disl_cfg, "pt_taylor_corr_rho_c", 1.0e14)),
        taylor_corr_scale=float(getattr(disl_cfg, "pt_exact_taylor_corr_scale", getattr(disl_cfg, "pt_taylor_m_scale", 1.0))),
        taylor_renewal_time_s=float(getattr(disl_cfg, "pt_taylor_renewal_time_s", 1.0e-9)),
        taylor_m_cap=float(getattr(disl_cfg, "pt_taylor_m_cap", float("inf"))),
        taylor_phi_max=float(getattr(disl_cfg, "pt_taylor_phi_max", 20.0)),
        mobile_fraction=float(getattr(disl_cfg, "pt_mobile_fraction", 0.01)),
        mobile_saturation_density_m2=float(getattr(disl_cfg, "pt_mobile_saturation_density_m2", 1.0e14)),
        mobile_density_floor_m2=float(getattr(disl_cfg, "pt_mobile_density_floor_m2", 1.0e6)),
        jump_fraction=float(getattr(disl_cfg, "pt_jump_fraction", 1.0)),
        jump_length_min_m=float(getattr(disl_cfg, "pt_jump_length_min_m", 2.5e-10)),
        equivalent_strain_factor=float(getattr(disl_cfg, "pt_equivalent_strain_factor", 1.0 / math.sqrt(3.0))),
        exact_manifest_mapping=exact,
        candidate_id=str(getattr(disl_cfg, "pt_exact_candidate_id", "fallback_scaled_parent")),
    ).validate()


def _install_surface(disl_cfg: Any, prefix: str, surface: Any) -> None:
    setattr(disl_cfg, f"{prefix}_G00_eV", float(surface.G00_eV))
    setattr(disl_cfg, f"{prefix}_gT_eV_per_K", float(surface.gT_eV_per_K))
    setattr(disl_cfg, f"{prefix}_sigc0_Pa", float(surface.sigc0_Pa))
    setattr(disl_cfg, f"{prefix}_sT_Pa_per_K", float(surface.sT_Pa_per_K))
    setattr(disl_cfg, f"{prefix}_exp_a", float(surface.alpha))
    setattr(disl_cfg, f"{prefix}_exp_n", float(surface.exponent))
    setattr(disl_cfg, f"{prefix}_floor_frac", float(surface.floor_fraction))
    setattr(disl_cfg, f"{prefix}_floor_min_eV", float(surface.floor_min_eV))
    setattr(disl_cfg, f"{prefix}_floor_max_frac", float(surface.floor_max_fraction))
    setattr(disl_cfg, f"{prefix}_Tref_K", float(surface.Tref_K))
    setattr(disl_cfg, f"{prefix}_nu0_s", float(surface.attempt_frequency_s))


def install_manifest_bulk_kinetics(
    disl_cfg: Any,
    manifest: MaterialManifest,
    registry_row: dict[str, str],
) -> dict[str, Any]:
    """Install one exact registry row on the surrounding-field kinetic law."""
    p_surface = manifest.peierls.as_surface(manifest.emission)
    t_surface = manifest.taylor.as_surface(manifest.emission)
    _install_surface(disl_cfg, "pt_exact_peierls", p_surface)
    _install_surface(disl_cfg, "pt_exact_taylor", t_surface)
    disl_cfg.pt_exact_manifest_mapping = True
    disl_cfg.pt_exact_candidate_id = str(manifest.candidate_id)
    disl_cfg.bulk_kinetics_model = "emission_derived_peierls_taylor_multihit"
    disl_cfg.pt_peierls_stress_fraction = float(registry_row["peierls_stress_fraction"])
    disl_cfg.pt_taylor_stress_fraction = float(registry_row["taylor_stress_fraction"])
    disl_cfg.pt_taylor_corr_rho_c = float(manifest.taylor_corr_rho_c_m2)
    disl_cfg.pt_exact_taylor_corr_scale = float(manifest.taylor_corr_scale)
    disl_cfg.pt_taylor_m_scale = float(manifest.taylor_corr_scale)
    disl_cfg.pt_peierls_nu0_s = float(manifest.peierls.attempt_frequency_s)
    disl_cfg.pt_taylor_nu0_s = float(manifest.taylor.attempt_frequency_s)
    disl_cfg.thermo_consistency_mode = "time_cone"
    disl_cfg.plastic_update_mode = "explicit_rate"
    disl_cfg.enable_plasticity = True
    return {
        "candidate_id": str(manifest.candidate_id),
        "exact_manifest_mapping": True,
        "bulk_kinetics_model": str(disl_cfg.bulk_kinetics_model),
        "peierls_surface": p_surface.__dict__,
        "taylor_surface": t_surface.__dict__,
        "peierls_stress_fraction": float(disl_cfg.pt_peierls_stress_fraction),
        "taylor_stress_fraction": float(disl_cfg.pt_taylor_stress_fraction),
        "taylor_corr_rho_c_m2": float(disl_cfg.pt_taylor_corr_rho_c),
        "taylor_corr_scale": float(disl_cfg.pt_exact_taylor_corr_scale),
        "taylor_renewal_time_s": float(disl_cfg.pt_taylor_renewal_time_s),
        "mobile_fraction": float(disl_cfg.pt_mobile_fraction),
        "bulk_mult_frac": float(disl_cfg.bulk_mult_frac),
        "thermo_consistency_mode": str(disl_cfg.thermo_consistency_mode),
    }


class EmissionDerivedPeierlsTaylorModel:
    def __init__(self, config: EmissionDerivedPeierlsTaylorConfig):
        self.config = config.validate()

    def rates(
        self,
        equivalent_stress_Pa: np.ndarray | float,
        forest_density_m2: np.ndarray | float,
        temperature_K: float,
        burgers_m: float,
    ) -> dict[str, np.ndarray]:
        cfg = self.config
        seq = np.maximum(np.asarray(equivalent_stress_Pa, dtype=float), 0.0)
        rho = np.maximum(np.asarray(forest_density_m2, dtype=float), 1.0)
        seq, rho = np.broadcast_arrays(seq, rho)
        b = max(abs(float(burgers_m)), 1.0e-30)

        spacing = 1.0 / (2.0 * np.sqrt(rho))
        phi = np.minimum(spacing / b, max(float(cfg.taylor_phi_max), 1.0))
        tau_p = max(float(cfg.peierls_stress_fraction), 0.0) * seq
        tau_t = max(float(cfg.taylor_stress_fraction), 0.0) * seq * phi

        Gp = cfg.peierls.barrier_eV(tau_p, temperature_K)
        Gt = cfg.taylor.barrier_eV(tau_t, temperature_K)
        lambda_p = cfg.peierls.rate_s(tau_p, temperature_K)
        lambda_t1 = cfg.taylor.rate_s(tau_t, temperature_K)

        ratio = np.sqrt(rho / max(float(cfg.taylor_corr_rho_c_m2), 1.0))
        m_eff = 1.0 + max(float(cfg.taylor_corr_scale), 0.0) * np.maximum(ratio - 1.0, 0.0)
        if math.isfinite(float(cfg.taylor_m_cap)):
            m_eff = np.minimum(m_eff, max(float(cfg.taylor_m_cap), 1.0))
        m_eff = np.maximum(m_eff, 1.0)

        tc = max(float(cfg.taylor_renewal_time_s), 1.0e-30)
        hits = np.clip(lambda_t1 * tc, 0.0, 1.0e12)
        lambda_t = gammainc(m_eff, hits) / tc

        denom = lambda_p + lambda_t
        lambda_series = np.divide(
            lambda_p * lambda_t,
            denom,
            out=np.zeros_like(denom),
            where=denom > 0.0,
        )

        rho_mobile = max(float(cfg.mobile_fraction), 0.0) * rho / (
            1.0 + rho / max(float(cfg.mobile_saturation_density_m2), 1.0)
        )
        rho_mobile = np.maximum(rho_mobile, float(cfg.mobile_density_floor_m2))
        rho_mobile = np.minimum(rho_mobile, rho)
        jump = np.maximum(max(float(cfg.jump_fraction), 0.0) * spacing, float(cfg.jump_length_min_m))
        velocity = jump * lambda_series
        equivalent_rate = max(float(cfg.equivalent_strain_factor), 0.0) * rho_mobile * b * velocity

        arrays = {
            "peierls_rate_s": lambda_p,
            "taylor_single_hit_rate_s": lambda_t1,
            "taylor_completion_rate_s": lambda_t,
            "series_rate_s": lambda_series,
            "taylor_m_eff": m_eff,
            "rho_mobile_m2": rho_mobile,
            "G_peierls_eV": Gp,
            "G_taylor_eV": Gt,
            "jump_length_m": jump,
            "velocity_m_per_s": velocity,
            "equivalent_plastic_rate_s": equivalent_rate,
        }
        for name, value in arrays.items():
            if not np.all(np.isfinite(value)) or np.any(value < 0.0):
                raise FloatingPointError(f"non-finite or negative {name} in emission-derived bulk kinetics")
        return arrays


__all__ = [
    "ExpFloorRateSurface",
    "EmissionDerivedPeierlsTaylorConfig",
    "EmissionDerivedPeierlsTaylorModel",
    "config_from_dislocation_config",
    "install_manifest_bulk_kinetics",
]
