"""FEM adapter for the existing emission-derived Peierls/Taylor kinetics.

This module does not introduce a second constitutive law.  It applies the same
EXP-floor descendants, correlated Taylor Poisson completion, and series
bottleneck already used by :mod:`arrhenius_fracture.unified_mpz` to P0 FEM
stress and forest-density fields.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
from scipy.special import gammainc

from .material_manifest import ExpFloorBarrier


@dataclass(frozen=True)
class EmissionDerivedPeierlsTaylorConfig:
    parent: ExpFloorBarrier
    peierls_energy_ratio: float
    peierls_entropy_ratio: float
    peierls_stress_ratio: float
    peierls_attempt_frequency_s: float
    taylor_energy_ratio: float
    taylor_entropy_ratio: float
    taylor_stress_ratio: float
    taylor_attempt_frequency_s: float
    taylor_corr_rho_c_m2: float
    taylor_renewal_time_s: float
    taylor_m_exponent: float
    taylor_m_scale: float
    taylor_m_cap: float
    mobile_fraction: float
    mobile_saturation_density_m2: float
    mobile_density_floor_m2: float
    jump_fraction: float
    jump_length_min_m: float
    equivalent_strain_factor: float
    taylor_phi_max: float
    peierls_surface: ExpFloorBarrier | None = None
    taylor_surface: ExpFloorBarrier | None = None


def _descendant(parent, energy_ratio, entropy_ratio, stress_ratio, attempt):
    energy = max(float(energy_ratio), 0.0)
    entropy = max(float(entropy_ratio), 0.0)
    stress = max(float(stress_ratio), 1.0e-30)
    return ExpFloorBarrier(
        G00_eV=energy*parent.G00_eV,
        gT_eV_per_K=entropy*parent.gT_eV_per_K,
        sigc0_Pa=stress*parent.sigc0_Pa,
        sT_Pa_per_K=stress*parent.sT_Pa_per_K,
        alpha=parent.alpha, exponent=parent.exponent,
        floor_fraction=parent.floor_fraction,
        floor_min_eV=energy*parent.floor_min_eV,
        floor_max_fraction=parent.floor_max_fraction,
        Tref_K=parent.Tref_K, attempt_frequency_s=float(attempt),
    )


def config_from_dislocation_config(config) -> EmissionDerivedPeierlsTaylorConfig:
    parent = ExpFloorBarrier(
        G00_eV=float(config.pt_emit_G00_eV),
        gT_eV_per_K=float(config.pt_emit_gT_eV_per_K),
        sigc0_Pa=float(config.pt_emit_sigc0_Pa),
        sT_Pa_per_K=float(config.pt_emit_sT_Pa_per_K),
        alpha=float(config.pt_emit_exp_a), exponent=float(config.pt_emit_exp_n),
        floor_fraction=float(config.pt_emit_floor_frac),
        floor_min_eV=float(config.pt_emit_floor_min_eV),
        floor_max_fraction=float(config.pt_emit_floor_max_frac),
        Tref_K=float(config.pt_emit_Tref_K), attempt_frequency_s=1.0e11,
    )
    manifest = getattr(config, "_emission_derived_material_manifest", None)
    return EmissionDerivedPeierlsTaylorConfig(
        parent=parent,
        peierls_energy_ratio=float(config.pt_peierls_energy_ratio),
        peierls_entropy_ratio=float(config.pt_peierls_entropy_ratio),
        peierls_stress_ratio=float(config.pt_peierls_stress_ratio),
        peierls_attempt_frequency_s=float(config.pt_peierls_nu0_s),
        taylor_energy_ratio=float(config.pt_taylor_energy_ratio),
        taylor_entropy_ratio=float(config.pt_taylor_entropy_ratio),
        taylor_stress_ratio=float(config.pt_taylor_stress_ratio),
        taylor_attempt_frequency_s=float(config.pt_taylor_nu0_s),
        taylor_corr_rho_c_m2=float(config.pt_taylor_corr_rho_c),
        taylor_renewal_time_s=float(config.pt_taylor_renewal_time_s),
        taylor_m_exponent=float(config.pt_taylor_m_exponent),
        taylor_m_scale=float(config.pt_taylor_m_scale),
        taylor_m_cap=float(config.pt_taylor_m_cap),
        mobile_fraction=float(config.pt_mobile_fraction),
        mobile_saturation_density_m2=float(config.pt_mobile_saturation_density_m2),
        mobile_density_floor_m2=float(config.pt_mobile_density_floor_m2),
        jump_fraction=float(config.pt_jump_fraction),
        jump_length_min_m=float(config.pt_jump_length_min_m),
        equivalent_strain_factor=float(config.pt_equivalent_strain_factor),
        taylor_phi_max=float(config.pt_taylor_phi_max),
        peierls_surface=(None if manifest is None else manifest.peierls.as_surface(manifest.emission)),
        taylor_surface=(None if manifest is None else manifest.taylor.as_surface(manifest.emission)),
    )


def bind_dislocation_config_to_material_manifest(config, manifest):
    """Bind the immutable selected material row to the ordinary FEM adapter."""
    config._emission_derived_material_manifest = manifest
    return config


class EmissionDerivedPeierlsTaylorModel:
    def __init__(self, config: EmissionDerivedPeierlsTaylorConfig):
        self.config = config
        self.peierls = config.peierls_surface or _descendant(
            config.parent, config.peierls_energy_ratio,
            config.peierls_entropy_ratio, config.peierls_stress_ratio,
            config.peierls_attempt_frequency_s,
        )
        self.taylor = config.taylor_surface or _descendant(
            config.parent, config.taylor_energy_ratio,
            config.taylor_entropy_ratio, config.taylor_stress_ratio,
            config.taylor_attempt_frequency_s,
        )

    def rates(self, equivalent_stress_Pa, forest_density_m2, temperature_K, burgers_vector_m):
        cfg = self.config
        stress = np.maximum(np.asarray(equivalent_stress_Pa, float), 0.0)
        rho = np.maximum(np.asarray(forest_density_m2, float), 1.0)
        spacing = 1.0/(2.0*np.sqrt(rho))
        peierls_stress = cfg.peierls_stress_ratio*stress
        phi = np.minimum(spacing/max(abs(float(burgers_vector_m)), 1.0e-30),
                         max(cfg.taylor_phi_max, 1.0))
        taylor_stress = cfg.taylor_stress_ratio*stress*phi
        # v10.4.1 detailed balance.  The reverse barrier is symmetric about the
        # zero-stress barrier: G_rev=2*G(0)-G_fwd.  This adds no fitted constant.
        kT = max(8.617333262145e-5*float(temperature_K), 1.0e-30)
        peierls_forward_barrier = np.asarray(
            self.peierls.values_eV(peierls_stress, temperature_K), float)
        peierls_zero_barrier = np.asarray(
            self.peierls.values_eV(np.zeros_like(stress), temperature_K), float)
        peierls_reverse_barrier = np.maximum(
            2.0*peierls_zero_barrier-peierls_forward_barrier,
            peierls_zero_barrier,
        )
        peierls_forward = self.peierls.attempt_frequency_s*np.exp(
            np.clip(-peierls_forward_barrier/kT, -745.0, 0.0))
        peierls_reverse = self.peierls.attempt_frequency_s*np.exp(
            np.clip(-peierls_reverse_barrier/kT, -745.0, 0.0))
        peierls_rate = np.maximum(peierls_forward-peierls_reverse, 0.0)
        peierls_rate = np.where(stress == 0.0, 0.0, peierls_rate)
        taylor_forward_barrier = np.asarray(
            self.taylor.values_eV(taylor_stress, temperature_K), float)
        taylor_zero_barrier = np.asarray(
            self.taylor.values_eV(np.zeros_like(stress), temperature_K), float)
        taylor_reverse_barrier = np.maximum(
            2.0*taylor_zero_barrier-taylor_forward_barrier,
            taylor_zero_barrier,
        )
        taylor_forward = self.taylor.attempt_frequency_s*np.exp(
            np.clip(-taylor_forward_barrier/kT, -745.0, 0.0))
        taylor_reverse = self.taylor.attempt_frequency_s*np.exp(
            np.clip(-taylor_reverse_barrier/kT, -745.0, 0.0))
        taylor_single = np.maximum(taylor_forward-taylor_reverse, 0.0)
        taylor_single = np.where(stress == 0.0, 0.0, taylor_single)
        density_ratio = np.sqrt(rho/max(cfg.taylor_corr_rho_c_m2, 1.0))
        m_eff = 1.0 + max(cfg.taylor_m_scale, 0.0)*np.maximum(
            density_ratio-1.0, 0.0,
        )**max(cfg.taylor_m_exponent, 1.0e-12)
        if math.isfinite(cfg.taylor_m_cap):
            m_eff = np.minimum(m_eff, max(cfg.taylor_m_cap, 1.0))
        renewal = max(cfg.taylor_renewal_time_s, 1.0e-30)
        taylor_completion = gammainc(
            np.maximum(m_eff, 1.0), np.minimum(taylor_single*renewal, 1.0e12),
        )/renewal
        series = np.divide(
            peierls_rate*taylor_completion, peierls_rate+taylor_completion,
            out=np.zeros_like(peierls_rate), where=(peierls_rate+taylor_completion) > 0.0,
        )
        rho_mobile = np.clip(
            cfg.mobile_fraction*rho, cfg.mobile_density_floor_m2,
            cfg.mobile_saturation_density_m2,
        )
        jump = np.maximum(cfg.jump_fraction*spacing, cfg.jump_length_min_m)
        equivalent_rate = (max(cfg.equivalent_strain_factor, 0.0)*rho_mobile
                           *abs(float(burgers_vector_m))*jump*series)
        return {
            "equivalent_plastic_rate_s": equivalent_rate,
            "peierls_forward_rate_s": peierls_forward,
            "peierls_reverse_rate_s": peierls_reverse,
            "peierls_rate_s": peierls_rate,
            "taylor_forward_single_hit_rate_s": taylor_forward,
            "taylor_reverse_single_hit_rate_s": taylor_reverse,
            "taylor_single_hit_rate_s": taylor_single,
            "taylor_completion_rate_s": taylor_completion,
            "series_rate_s": series,
            "taylor_m_eff": m_eff,
            "rho_mobile_m2": rho_mobile,
            "G_peierls_eV": np.asarray(self.peierls.values_eV(
                peierls_stress, temperature_K), float),
            "G_taylor_eV": np.asarray(self.taylor.values_eV(
                taylor_stress, temperature_K), float),
        }


__all__ = [
    "EmissionDerivedPeierlsTaylorConfig", "EmissionDerivedPeierlsTaylorModel",
    "bind_dislocation_config_to_material_manifest", "config_from_dislocation_config",
]
