"""Emission-derived Arrhenius Peierls--Taylor bulk kinetics.

This module supplies the constitutive component referenced by ``plasticity.py``
for the production ``emission_derived_peierls_taylor_multihit`` path. Peierls
and Taylor transport surfaces are transferred from the selected material row,
combined as sequential bottlenecks, and converted to equivalent plastic strain
rate with an Orowan carrier relation. Thermodynamic admissibility of the final
strain increment remains the responsibility of ``plasticity.update_plasticity``.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import numpy as np
from scipy.special import gammainc

KB_EV_PER_K = 8.617333262145e-5
MODEL_ID = "v10.4.0_emission_derived_peierls_taylor_multihit"


def _finite(value: Any, default: float) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return float(default)
    return result if math.isfinite(result) else float(default)


def _positive(value: Any, default: float, floor: float = 1.0e-30) -> float:
    return max(_finite(value, default), floor)


@dataclass(frozen=True)
class ExpFloorSurface:
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

    def barrier_eV(self, stress_Pa: np.ndarray, T_K: float) -> np.ndarray:
        stress = np.maximum(np.asarray(stress_Pa, dtype=float), 0.0)
        dT = float(T_K) - self.Tref_K
        G0 = max(self.G00_eV + self.gT_eV_per_K * dT, 1.0e-12)
        sigc = max(self.sigc0_Pa + self.sT_Pa_per_K * dT, 1.0)
        raw_floor = max(self.floor_min_eV, self.floor_fraction * G0)
        floor = min(self.floor_max_fraction * G0, raw_floor)
        x = np.maximum(stress / sigc, 0.0)
        return np.maximum(
            floor
            + (G0 - floor)
            * np.exp(-max(self.alpha, 0.0) * np.power(x, max(self.exponent, 1.0e-12))),
            0.0,
        )

    def rate_s(self, stress_Pa: np.ndarray, T_K: float) -> np.ndarray:
        barrier = self.barrier_eV(stress_Pa, T_K)
        exponent = -barrier / max(KB_EV_PER_K * float(T_K), 1.0e-30)
        return self.attempt_frequency_s * np.exp(np.clip(exponent, -745.0, 0.0))


@dataclass(frozen=True)
class EmissionDerivedPeierlsTaylorConfig:
    peierls: ExpFloorSurface
    taylor: ExpFloorSurface
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
    taylor_phi_max: float


def _surface_from_config(
    cfg: Any,
    prefix: str,
    *,
    parent: ExpFloorSurface,
    default_nu0: float,
) -> ExpFloorSurface:
    direct_G = getattr(cfg, f"pt_{prefix}_G00_eV", None)
    direct_gT = getattr(cfg, f"pt_{prefix}_gT_eV_per_K", None)
    energy_ratio = _finite(getattr(cfg, f"pt_{prefix}_energy_ratio", 1.0), 1.0)
    entropy_ratio = _finite(getattr(cfg, f"pt_{prefix}_entropy_ratio", 1.0), 1.0)
    stress_ratio = _positive(getattr(cfg, f"pt_{prefix}_stress_ratio", 1.0), 1.0)
    G00 = _positive(direct_G, parent.G00_eV * energy_ratio)
    gT = _finite(direct_gT, parent.gT_eV_per_K * entropy_ratio)
    scale = G00 / max(parent.G00_eV, 1.0e-30)
    return ExpFloorSurface(
        G00_eV=G00,
        gT_eV_per_K=gT,
        sigc0_Pa=stress_ratio * parent.sigc0_Pa,
        sT_Pa_per_K=stress_ratio * parent.sT_Pa_per_K,
        alpha=_positive(
            getattr(cfg, f"pt_{prefix}_exp_a", parent.alpha), parent.alpha
        ),
        exponent=_positive(
            getattr(cfg, f"pt_{prefix}_exp_n", parent.exponent), parent.exponent
        ),
        floor_fraction=_finite(
            getattr(cfg, f"pt_{prefix}_floor_frac", parent.floor_fraction),
            parent.floor_fraction,
        ),
        floor_min_eV=_positive(
            getattr(cfg, f"pt_{prefix}_floor_min_eV", parent.floor_min_eV * scale),
            parent.floor_min_eV * scale,
        ),
        floor_max_fraction=min(
            max(
                _finite(
                    getattr(
                        cfg,
                        f"pt_{prefix}_floor_max_frac",
                        parent.floor_max_fraction,
                    ),
                    parent.floor_max_fraction,
                ),
                1.0e-12,
            ),
            1.0,
        ),
        Tref_K=_positive(
            getattr(cfg, f"pt_{prefix}_Tref_K", parent.Tref_K), parent.Tref_K
        ),
        attempt_frequency_s=_positive(
            getattr(cfg, f"pt_{prefix}_nu0_s", default_nu0), default_nu0
        ),
    )


def config_from_dislocation_config(cfg: Any) -> EmissionDerivedPeierlsTaylorConfig:
    parent = ExpFloorSurface(
        G00_eV=_positive(getattr(cfg, "pt_emit_G00_eV", 1.0), 1.0),
        gT_eV_per_K=_finite(getattr(cfg, "pt_emit_gT_eV_per_K", 0.0), 0.0),
        sigc0_Pa=_positive(getattr(cfg, "pt_emit_sigc0_Pa", 2.0e9), 2.0e9),
        sT_Pa_per_K=_finite(getattr(cfg, "pt_emit_sT_Pa_per_K", 0.0), 0.0),
        alpha=_positive(getattr(cfg, "pt_emit_exp_a", 0.1), 0.1),
        exponent=_positive(getattr(cfg, "pt_emit_exp_n", 1.0), 1.0),
        floor_fraction=max(
            _finite(getattr(cfg, "pt_emit_floor_frac", 0.02), 0.02), 0.0
        ),
        floor_min_eV=_positive(
            getattr(cfg, "pt_emit_floor_min_eV", 1.0e-4), 1.0e-4
        ),
        floor_max_fraction=min(
            max(
                _finite(getattr(cfg, "pt_emit_floor_max_frac", 0.95), 0.95),
                1.0e-12,
            ),
            1.0,
        ),
        Tref_K=_positive(getattr(cfg, "pt_emit_Tref_K", 481.33), 481.33),
        attempt_frequency_s=1.0,
    )
    peierls = _surface_from_config(
        cfg, "peierls", parent=parent, default_nu0=1.0e12
    )
    taylor = _surface_from_config(
        cfg, "taylor", parent=parent, default_nu0=1.0e11
    )
    cap = _finite(getattr(cfg, "pt_taylor_m_cap", float("inf")), float("inf"))
    return EmissionDerivedPeierlsTaylorConfig(
        peierls=peierls,
        taylor=taylor,
        taylor_corr_rho_c_m2=_positive(
            getattr(cfg, "pt_taylor_corr_rho_c", 1.0e14), 1.0e14
        ),
        taylor_renewal_time_s=_positive(
            getattr(cfg, "pt_taylor_renewal_time_s", 1.0e-9), 1.0e-9
        ),
        taylor_m_exponent=_positive(
            getattr(cfg, "pt_taylor_m_exponent", 1.0), 1.0
        ),
        taylor_m_scale=max(
            _finite(getattr(cfg, "pt_taylor_m_scale", 1.0), 1.0), 0.0
        ),
        taylor_m_cap=cap,
        mobile_fraction=max(
            _finite(getattr(cfg, "pt_mobile_fraction", 0.01), 0.01), 0.0
        ),
        mobile_saturation_density_m2=_positive(
            getattr(cfg, "pt_mobile_saturation_density_m2", 1.0e14), 1.0e14
        ),
        mobile_density_floor_m2=_positive(
            getattr(cfg, "pt_mobile_density_floor_m2", 1.0e6), 1.0e6
        ),
        jump_fraction=max(
            _finite(getattr(cfg, "pt_jump_fraction", 1.0), 1.0), 0.0
        ),
        jump_length_min_m=_positive(
            getattr(cfg, "pt_jump_length_min_m", 2.5e-10), 2.5e-10
        ),
        taylor_phi_max=_positive(
            getattr(cfg, "pt_taylor_phi_max", 20.0), 20.0
        ),
    )


class EmissionDerivedPeierlsTaylorModel:
    def __init__(self, cfg: EmissionDerivedPeierlsTaylorConfig):
        self.cfg = cfg

    def rates(
        self,
        equivalent_stress_Pa: np.ndarray,
        forest_density_m2: np.ndarray,
        T_K: float,
        b_m: float,
    ) -> dict[str, np.ndarray]:
        stress = np.maximum(np.asarray(equivalent_stress_Pa, dtype=float), 0.0)
        rho = np.maximum(
            np.asarray(forest_density_m2, dtype=float),
            self.cfg.mobile_density_floor_m2,
        )
        b = max(float(b_m), 1.0e-30)

        peierls_rate = self.cfg.peierls.rate_s(stress, T_K)
        spacing = 1.0 / (2.0 * np.sqrt(rho))
        phi = np.minimum(spacing / b, self.cfg.taylor_phi_max)
        taylor_local_stress = stress * phi
        taylor_single = self.cfg.taylor.rate_s(taylor_local_stress, T_K)

        density_ratio = np.maximum(rho / self.cfg.taylor_corr_rho_c_m2, 0.0)
        order = 1.0 + self.cfg.taylor_m_scale * np.power(
            density_ratio, self.cfg.taylor_m_exponent
        )
        if math.isfinite(self.cfg.taylor_m_cap):
            order = np.minimum(order, max(self.cfg.taylor_m_cap, 1.0))
        order = np.maximum(order, 1.0)

        window = self.cfg.taylor_renewal_time_s
        argument = np.clip(taylor_single * window, 0.0, 1.0e12)
        taylor_completion = gammainc(order, argument) / window

        tiny = 1.0e-300
        series_rate = np.where(
            (peierls_rate > 0.0) & (taylor_completion > 0.0),
            1.0
            / (
                1.0 / np.maximum(peierls_rate, tiny)
                + 1.0 / np.maximum(taylor_completion, tiny)
            ),
            0.0,
        )

        mobile_density = np.minimum(
            np.maximum(self.cfg.mobile_fraction * rho, 0.0),
            self.cfg.mobile_saturation_density_m2,
        )
        jump_length = np.maximum(
            self.cfg.jump_fraction * spacing,
            self.cfg.jump_length_min_m,
        )
        equivalent_rate = mobile_density * b * jump_length * series_rate
        equivalent_rate = np.where(np.isfinite(equivalent_rate), equivalent_rate, 0.0)

        return {
            "equivalent_plastic_rate_s": equivalent_rate,
            "peierls_rate_s": peierls_rate,
            "taylor_single_hit_rate_s": taylor_single,
            "taylor_completion_rate_s": taylor_completion,
            "series_rate_s": series_rate,
            "taylor_hit_order": order,
            "taylor_phi": phi,
            "mobile_density_m2": mobile_density,
            "jump_length_m": jump_length,
            "peierls_barrier_eV": self.cfg.peierls.barrier_eV(stress, T_K),
            "taylor_barrier_eV": self.cfg.taylor.barrier_eV(
                taylor_local_stress, T_K
            ),
            "model_id": np.full(stress.shape, 1.0),
        }


__all__ = [
    "EmissionDerivedPeierlsTaylorConfig",
    "EmissionDerivedPeierlsTaylorModel",
    "ExpFloorSurface",
    "MODEL_ID",
    "config_from_dislocation_config",
]
