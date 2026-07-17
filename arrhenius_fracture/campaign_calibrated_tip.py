"""Campaign-calibrated continuum tip-source budget.

This model uses the promoted zero-dimensional tuning campaign as the reference
closure instead of introducing an unconstrained time-recycling source law.
Distributed plasticity remains the spatial Arrhenius Peierls--Taylor continuum.
The crack-tip source capacity is a bounded continuum measure per
crystallographic system:

    dS_s/dt = -lambda_emit,s S_s

and it recovers only as new crack-tip material/geometry is exposed:

    dS_s = (S0_s - S_s) [1 - exp(-da / L_refresh)].

The promoted source_sites_per_system fixes S0, source_refresh_length_m fixes the
reference recovery length, c_blunt keeps its original role, and
max_K_shield_MPa_sqrt_m bounds active cleavage shielding as in the campaign.
Two dimensionless calibration scales are exposed: one for the local Taylor back
stress and one for the source-refresh length.
"""
from __future__ import annotations

import math
from types import MethodType
from typing import Any

import numpy as np

from .fractional_moving_frame import fractional_moving_frame_advance
from .separated_source_tip import SeparatedSourceKineticTipEngine
from .kinetic_tip_cell import KineticMovingTipFrontEngine


SOURCE_MODEL = "campaign_calibrated_tip_budget"


def _campaign_local_density_m2(state) -> np.ndarray:
    """Near-tip mobile-plus-retained density on the fixed promoted MPZ scale.

    The averaging length is not allowed to expand with the blunted radius. That
    previous feedback diluted the density precisely when a large emitted cloud
    should have increased the emission back stress. The existing MPZ
    blunting-length/grid scale supplies the local averaging volume and adds no
    new fitted length.
    """
    length = max(
        float(state.cfg.blunting_length_m),
        float(state.dx),
        abs(float(getattr(state, "_campaign_b", 0.0))),
        1.0e-12,
    )
    weights = np.exp(-np.asarray(state.x, dtype=float) / length)
    norm = max(float(np.sum(weights)), 1.0e-30)
    count = (
        np.maximum(np.asarray(state.mobile, dtype=float), 0.0)
        + np.maximum(np.asarray(state.retained, dtype=float), 0.0)
    )
    near_count = np.sum(count * weights[None, :], axis=1) / norm
    width = max(float(state.cfg.blunting_length_m), float(state.dx), 1.0e-12)
    return np.maximum(near_count / max(float(state.dx) * width, 1.0e-30), 0.0)


def _campaign_backstress(state) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rho = _campaign_local_density_m2(state)
    G = max(float(getattr(state, "_campaign_G_Pa", 0.0)), 0.0)
    b = abs(float(getattr(state, "_campaign_b", 0.0)))
    scale = max(float(getattr(state, "_campaign_backstress_scale", 1.0)), 0.0)
    tau = scale * G * b * np.sqrt(np.maximum(rho, 0.0))
    resolved = max(abs(float(state.cfg.taylor_stress_fraction)), 1.0e-6)
    sigma = tau / resolved
    return rho, tau, sigma


def _campaign_emit(
    self,
    dt: float,
    stress_Pa: float,
    T_K: float,
    system_weights: np.ndarray | None = None,
) -> float:
    """Consume a bounded continuum tip-source budget.

    No Peierls time-clearing term replenishes source capacity while the crack is
    stationary. Peierls/Taylor kinetics still transport and store the emitted
    distributed population, which supplies the evolving local back stress.
    """
    dt = max(float(dt), 0.0)
    if dt <= 0.0:
        return 0.0

    weights = np.ones(self.n_systems, dtype=float)
    if system_weights is not None:
        raw = np.maximum(np.asarray(system_weights, dtype=float).reshape(-1), 0.0)
        if raw.size < self.n_systems:
            raw = np.pad(raw, (0, self.n_systems - raw.size), mode="edge")
        raw = raw[: self.n_systems]
        weights = raw / np.max(raw) if np.max(raw) > 0.0 else np.zeros_like(raw)

    rho, tau_back, sigma_back = _campaign_backstress(self)
    sigma_open = max(float(stress_Pa), 0.0)
    sigma_eff = np.maximum(sigma_open - sigma_back, 0.0)
    lam_site = np.maximum(
        np.asarray(
            [self.emission_rate_per_site(float(sig), T_K) for sig in sigma_eff],
            dtype=float,
        ),
        0.0,
    )

    available0 = np.maximum(np.asarray(self.available_sites, dtype=float), 0.0)
    probability = 1.0 - np.exp(-np.minimum(lam_site * dt, 700.0))
    emitted_system = np.minimum(available0 * probability * weights, available0)
    self.available_sites = np.maximum(available0 - emitted_system, 0.0)

    nsrc = max(min(int(self.cfg.source_bin_count), self.n_bins), 1)
    self.mobile[:, :nsrc] += emitted_system[:, None] / nsrc
    self.accumulated_slip[:, :nsrc] += emitted_system[:, None] / nsrc
    emitted = float(np.sum(emitted_system))
    self.emitted_total += emitted

    reference = np.maximum(np.asarray(self.site_capacity, dtype=float), 0.0)
    activity = np.divide(
        self.available_sites,
        reference,
        out=np.zeros_like(self.available_sites),
        where=reference > 0.0,
    )
    self.tip_source_activity = np.clip(activity, 0.0, 1.0)
    self.continuum_source_last_clear_rate_s = 0.0
    self.continuum_source_last_effective_multiplicity = float(np.sum(self.available_sites))
    self.continuum_source_last_emission_rate_s = float(np.sum(lam_site * available0 * weights))
    self.continuum_source_last_aggregate_hazard_s = self.continuum_source_last_emission_rate_s
    self.continuum_source_last_throughput_bound = float(np.sum(available0))
    self.continuum_source_last_rho_back_m2 = float(np.mean(rho))
    self.continuum_source_last_tau_back_Pa = float(np.mean(tau_back))
    self.continuum_source_last_sigma_back_Pa = float(np.mean(sigma_back))
    self.continuum_source_last_sigma_emit_effective_Pa = float(np.mean(sigma_eff))
    self.continuum_source_last_sigma_emit_effective_min_Pa = float(np.min(sigma_eff))
    self.campaign_source_budget_remaining_total = float(np.sum(self.available_sites))
    self.campaign_source_budget_consumed_total = float(np.sum(reference - self.available_sites))
    return emitted


def _campaign_advance(self, distance_m: float) -> dict[str, float]:
    before = np.asarray(self.available_sites, dtype=float).copy()
    result = fractional_moving_frame_advance(self, distance_m)

    # The generic fractional mover retains the legacy linear source refresh.
    # Undo that bookkeeping and apply only the campaign exponential recovery
    # below so source capacity is not replenished twice.
    self.available_sites = before.copy()

    d = max(float(distance_m), 0.0)
    refresh_scale = max(float(getattr(self, "_campaign_refresh_scale", 1.0)), 1.0e-12)
    reference_length = max(float(self.manifest.source_refresh_length_m), float(self.dx), 1.0e-12)
    effective_length = reference_length * refresh_scale
    fraction = 1.0 - math.exp(-min(d / effective_length, 700.0))
    self.available_sites += (self.site_capacity - self.available_sites) * fraction
    self.available_sites = np.clip(self.available_sites, 0.0, self.site_capacity)
    self.tip_source_activity = np.divide(
        self.available_sites,
        self.site_capacity,
        out=np.zeros_like(self.available_sites),
        where=self.site_capacity > 0.0,
    )
    refreshed = max(float(np.sum(self.available_sites - before)), 0.0)
    result["source_sites_refreshed"] = refreshed
    result["campaign_source_refresh_fraction"] = fraction
    result["campaign_source_refresh_length_m"] = effective_length
    self.campaign_source_last_refresh_fraction = fraction
    self.campaign_source_last_refresh_length_m = effective_length
    self.campaign_source_budget_remaining_total = float(np.sum(self.available_sites))
    self.campaign_source_budget_consumed_total = float(np.sum(self.site_capacity - self.available_sites))
    return result


def install_campaign_calibrated_source(
    state,
    b: float,
    G_Pa: float,
    backstress_scale: float = 1.0,
    refresh_scale: float = 1.0,
) -> None:
    reference = max(float(state.manifest.source_sites_per_system), 0.0)
    state.source_model = SOURCE_MODEL
    state.reference_source_multiplicity = reference
    state.site_capacity = np.full(state.n_systems, reference, dtype=float)
    state.available_sites = state.site_capacity.copy()
    state.tip_source_activity = np.ones(state.n_systems, dtype=float)
    state._campaign_b = float(b)
    state._campaign_G_Pa = max(float(G_Pa), 0.0)
    state._campaign_backstress_scale = max(float(backstress_scale), 0.0)
    state._campaign_refresh_scale = max(float(refresh_scale), 1.0e-12)
    state._campaign_max_K_shield_Pa_sqrt_m = max(
        float(state.manifest.max_K_shield_MPa_sqrt_m), 0.0
    ) * 1.0e6
    state.continuum_source_last_clear_rate_s = 0.0
    state.continuum_source_last_effective_multiplicity = float(np.sum(state.available_sites))
    state.continuum_source_last_emission_rate_s = 0.0
    state.continuum_source_last_aggregate_hazard_s = 0.0
    state.continuum_source_last_throughput_bound = float(np.sum(state.available_sites))
    state.continuum_source_last_rho_back_m2 = 0.0
    state.continuum_source_last_tau_back_Pa = 0.0
    state.continuum_source_last_sigma_back_Pa = 0.0
    state.continuum_source_last_sigma_emit_effective_Pa = 0.0
    state.continuum_source_last_sigma_emit_effective_min_Pa = 0.0
    state.campaign_source_last_refresh_fraction = 0.0
    state.campaign_source_last_refresh_length_m = max(
        float(state.manifest.source_refresh_length_m) * state._campaign_refresh_scale,
        float(state.dx),
    )
    state.campaign_source_budget_remaining_total = float(np.sum(state.available_sites))
    state.campaign_source_budget_consumed_total = 0.0
    state._emit = MethodType(_campaign_emit, state)
    state.advance = MethodType(_campaign_advance, state)


class CampaignCalibratedTipEngine(SeparatedSourceKineticTipEngine):
    """Separated-stress moving-tip engine tied to the promoted campaign budget."""

    campaign_calibrated_source_active = True
    _campaign_backstress_scale_default = 1.0
    _campaign_refresh_scale_default = 1.0

    @classmethod
    def configure_campaign(cls, backstress_scale: float = 1.0, refresh_scale: float = 1.0) -> None:
        cls._campaign_backstress_scale_default = max(float(backstress_scale), 0.0)
        cls._campaign_refresh_scale_default = max(float(refresh_scale), 1.0e-12)

    @classmethod
    def audit_payload(cls) -> dict[str, Any]:
        payload = super().audit_payload()
        payload["campaign_calibration"] = {
            "backstress_scale": cls._campaign_backstress_scale_default,
            "refresh_length_scale": cls._campaign_refresh_scale_default,
            "source_budget_from_manifest": True,
            "shielding_cap_from_manifest": True,
            "temporal_source_recycling": False,
        }
        return payload

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        install_campaign_calibrated_source(
            self.mpz,
            self.b,
            self.G,
            self._campaign_backstress_scale_default,
            self._campaign_refresh_scale_default,
        )

    def _active_shielding_raw_uncapped(self) -> float:
        return float(KineticMovingTipFrontEngine._active_shielding_signed(self))

    def _active_shielding_signed(self) -> float:
        raw = self._active_shielding_raw_uncapped()
        cap = max(float(self.manifest.max_K_shield_MPa_sqrt_m), 0.0) * 1.0e6
        if cap <= 0.0:
            return raw
        return float(np.clip(raw, -cap, cap))

    def _campaign_diagnostics(self) -> dict[str, Any]:
        raw = self._active_shielding_raw_uncapped()
        effective = self._active_shielding_signed()
        return {
            "campaign_source_model": SOURCE_MODEL,
            "campaign_backstress_scale": self._campaign_backstress_scale_default,
            "campaign_refresh_length_scale": self._campaign_refresh_scale_default,
            "campaign_source_budget_total": float(np.sum(self.mpz.site_capacity)),
            "campaign_source_budget_remaining": float(np.sum(self.mpz.available_sites)),
            "campaign_source_budget_consumed": float(np.sum(self.mpz.site_capacity - self.mpz.available_sites)),
            "campaign_source_refresh_length_m": float(self.mpz.campaign_source_last_refresh_length_m),
            "campaign_source_refresh_fraction": float(self.mpz.campaign_source_last_refresh_fraction),
            "campaign_active_K_shield_raw_Pa_sqrt_m": raw,
            "campaign_active_K_shield_effective_Pa_sqrt_m": effective,
            "campaign_active_K_shield_cap_Pa_sqrt_m": max(float(self.manifest.max_K_shield_MPa_sqrt_m), 0.0) * 1.0e6,
            "campaign_temporal_source_recycling": False,
        }

    def step(self, K, T, dt):
        result = super().step(K, T, dt)
        diag = self._campaign_diagnostics()
        result.update(diag)
        result["mpz_active_K_shield_Pa_sqrt_m"] = diag["campaign_active_K_shield_effective_Pa_sqrt_m"]
        result["mpz_total_K_shield_Pa_sqrt_m"] = diag["campaign_active_K_shield_effective_Pa_sqrt_m"] + float(result.get("mpz_wake_K_shield_Pa_sqrt_m", 0.0))
        result["mpz_K_shield_Pa_sqrt_m"] = result["mpz_total_K_shield_Pa_sqrt_m"]
        if type(self)._audit_records:
            type(self)._audit_records[-1].update(diag)
        return result

    def cycle_step_waveform(self, *args, **kwargs):
        result = super().cycle_step_waveform(*args, **kwargs)
        result.update(self._campaign_diagnostics())
        return result
