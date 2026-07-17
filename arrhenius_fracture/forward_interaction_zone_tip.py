"""Spatial source-capacity memory in the forward crack-tip interaction zone.

The promoted campaign source count remains the temperature-independent total
virgin source content per crystallographic system.  Unlike the scalar campaign
refresh law, that content is distributed over a finite material interval ahead
of the crack tip.  In the moving frame, available source capacity advects toward
``xi=0`` with crack growth and virgin capacity enters only through the far-edge
inflow boundary.  Local Arrhenius emission therefore depends on the stress and
thermal exposure accumulated while material traverses the interaction zone.

Wake state and wake shielding are not part of this closure.  Mobile and retained
populations already represented by the active MPZ remain the only contributors
to local Taylor back stress, active shielding, and blunting.
"""
from __future__ import annotations

import math
from types import MethodType
from typing import Any

import numpy as np

from .campaign_calibrated_tip import (
    CampaignCalibratedTipEngine,
    _campaign_backstress,
)
from .fractional_moving_frame import fractional_moving_frame_advance
from .unified_mpz import UnifiedMPZState


SOURCE_MODEL = "forward_interaction_zone_source_field"
FORWARD_SCHEMA = "v10.1.8_forward_interaction_zone"


def _sync_scalar_source_views(state) -> None:
    """Maintain legacy vector views used by inherited diagnostics."""
    state.site_capacity = np.sum(state.forward_source_capacity_field, axis=1)
    state.available_sites = np.sum(state.forward_source_available_field, axis=1)
    state.tip_source_activity = np.divide(
        state.available_sites,
        state.site_capacity,
        out=np.zeros_like(state.available_sites),
        where=state.site_capacity > 0.0,
    )
    state.campaign_source_budget_remaining_total = float(np.sum(state.available_sites))
    state.campaign_source_budget_consumed_total = float(
        np.sum(state.site_capacity - state.available_sites)
    )


def _shift_source_field_with_virgin_inflow(
    field: np.ndarray,
    distance_m: float,
    dx: float,
    n_interaction_bins: int,
    virgin_count_per_bin: np.ndarray,
) -> tuple[np.ndarray, float, float]:
    """Conservatively translate available source capacity toward the tip.

    Counts leaving the ``xi=0`` boundary are reported as available-source
    outflow.  Material entering beyond the far boundary carries the virgin
    count per bin.  A spatially uniform virgin field is therefore invariant
    under arbitrary fractional translations.
    """
    source = np.maximum(np.asarray(field, dtype=float), 0.0)
    n_systems, n_bins = source.shape
    n_int = max(min(int(n_interaction_bins), n_bins), 1)
    virgin = np.maximum(np.asarray(virgin_count_per_bin, dtype=float).reshape(-1), 0.0)
    if virgin.size < n_systems:
        virgin = np.pad(virgin, (0, n_systems - virgin.size), mode="edge")
    virgin = virgin[:n_systems]

    shift = max(float(distance_m), 0.0) / max(float(dx), 1.0e-30)
    if shift <= 0.0:
        return source.copy(), 0.0, 0.0
    if shift >= n_int:
        out = np.zeros_like(source)
        out[:, :n_int] = virgin[:, None]
        return out, float(np.sum(source[:, :n_int])), float(np.sum(out[:, :n_int]))

    whole = int(math.floor(shift))
    frac = shift - whole
    out = np.zeros_like(source)
    retained_old = 0.0
    virgin_inflow = 0.0

    for j in range(n_int):
        for source_index, weight in ((j + whole, 1.0 - frac), (j + whole + 1, frac)):
            if weight <= 0.0:
                continue
            if source_index < n_int:
                contribution = source[:, source_index] * weight
                retained_old += float(np.sum(contribution))
            else:
                contribution = virgin * weight
                virgin_inflow += float(np.sum(contribution))
            out[:, j] += contribution

    old_total = float(np.sum(source[:, :n_int]))
    available_outflow = max(old_total - retained_old, 0.0)
    return out, available_outflow, max(virgin_inflow, 0.0)


def _forward_transport_rates(self, stress_profile, rho, T_K, b):
    """Apply one temperature-independent scale to Taylor encounter/trapping."""
    rates = UnifiedMPZState._transport_rates(self, stress_profile, rho, T_K, b)
    scale = max(float(getattr(self, "_forward_retention_scale", 1.0)), 0.0)
    rates["encounter"] = np.asarray(rates["encounter"], dtype=float) * scale
    rates["forward_retention_scale"] = np.full_like(
        np.asarray(rates["encounter"], dtype=float), scale
    )
    return rates


def _forward_emit(
    self,
    dt: float,
    stress_Pa: float,
    T_K: float,
    system_weights: np.ndarray | None = None,
) -> float:
    """Emit locally from the spatial virgin-source field."""
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
    stress_profile = np.maximum(self.local_stress_profile_Pa(sigma_open), 0.0)
    sigma_eff = np.maximum(stress_profile[None, :] - sigma_back[:, None], 0.0)

    lam_site = np.zeros_like(sigma_eff)
    for system in range(self.n_systems):
        lam_site[system, :] = np.asarray(
            self.manifest.emission.rate(sigma_eff[system, :], T_K), dtype=float
        )
    lam_site = np.maximum(lam_site, 0.0)

    available0 = np.maximum(
        np.asarray(self.forward_source_available_field, dtype=float), 0.0
    )
    probability = 1.0 - np.exp(-np.minimum(lam_site * dt, 700.0))
    emitted_field = np.minimum(
        available0 * probability * weights[:, None], available0
    )
    self.forward_source_available_field = np.maximum(
        available0 - emitted_field, 0.0
    )

    self.mobile += emitted_field
    self.accumulated_slip += emitted_field
    emitted = float(np.sum(emitted_field))
    self.emitted_total += emitted
    self.forward_source_cumulative_consumed += emitted
    _sync_scalar_source_views(self)

    active_weights = emitted_field
    emitted_sum = float(np.sum(active_weights))
    if emitted_sum > 0.0:
        self.forward_source_last_emission_centroid_m = float(
            np.sum(active_weights * self.x[None, :]) / emitted_sum
        )
    self.continuum_source_last_clear_rate_s = 0.0
    self.continuum_source_last_effective_multiplicity = float(
        np.sum(self.forward_source_available_field)
    )
    self.continuum_source_last_emission_rate_s = float(
        np.sum(lam_site * available0 * weights[:, None])
    )
    self.continuum_source_last_aggregate_hazard_s = (
        self.continuum_source_last_emission_rate_s
    )
    self.continuum_source_last_throughput_bound = float(np.sum(available0))
    self.continuum_source_last_rho_back_m2 = float(np.mean(rho))
    self.continuum_source_last_tau_back_Pa = float(np.mean(tau_back))
    self.continuum_source_last_sigma_back_Pa = float(np.mean(sigma_back))
    self.continuum_source_last_sigma_emit_effective_Pa = float(np.mean(sigma_eff))
    self.continuum_source_last_sigma_emit_effective_min_Pa = float(np.min(sigma_eff))
    return emitted


def _forward_advance(self, distance_m: float) -> dict[str, float]:
    """Advance active plastic fields and the forward source field together."""
    d = max(float(distance_m), 0.0)
    source_before = np.asarray(self.forward_source_available_field, dtype=float).copy()

    # Retain the validated continuous moving-frame transport for mobile,
    # retained, and slip fields.  Its scalar source refresh is discarded below.
    result = fractional_moving_frame_advance(self, d)

    shifted, available_outflow, virgin_inflow = _shift_source_field_with_virgin_inflow(
        source_before,
        d,
        self.dx,
        self.forward_interaction_n_bins,
        self.forward_source_virgin_count_per_bin,
    )
    self.forward_source_available_field = np.minimum(
        np.maximum(shifted, 0.0), self.forward_source_capacity_field
    )
    _sync_scalar_source_views(self)

    self.forward_source_cumulative_inflow += virgin_inflow
    self.forward_source_cumulative_available_outflow += available_outflow
    self.forward_source_last_inflow = virgin_inflow
    self.forward_source_last_available_outflow = available_outflow
    self.campaign_source_last_refresh_fraction = min(
        d / max(self.forward_interaction_length_m, self.dx), 1.0
    )
    self.campaign_source_last_refresh_length_m = self.forward_interaction_length_m

    result["source_sites_refreshed"] = virgin_inflow
    result["source_sites_available_outflow"] = available_outflow
    result["forward_source_virgin_inflow"] = virgin_inflow
    result["forward_source_available_outflow"] = available_outflow
    result["forward_interaction_zone_length_m"] = self.forward_interaction_length_m
    result["forward_interaction_zone_n_bins"] = float(
        self.forward_interaction_n_bins
    )
    return result


def install_forward_interaction_zone_source(
    state,
    interaction_length_scale: float = 1.0,
    retention_scale: float = 1.0,
) -> None:
    """Replace scalar refresh with a spatial forward source-capacity field."""
    scale = max(float(interaction_length_scale), 1.0e-12)
    base_length = max(float(state.manifest.source_refresh_length_m), float(state.dx))
    requested_length = min(base_length * scale, float(state.length_m))
    n_int = max(min(int(round(requested_length / state.dx)), state.n_bins), 1)
    interaction_length = n_int * float(state.dx)

    total_per_system = max(float(state.manifest.source_sites_per_system), 0.0)
    virgin_per_bin = np.full(
        state.n_systems, total_per_system / float(n_int), dtype=float
    )
    capacity = np.zeros((state.n_systems, state.n_bins), dtype=float)
    capacity[:, :n_int] = virgin_per_bin[:, None]

    state.source_model = SOURCE_MODEL
    state.forward_interaction_length_scale = scale
    state.forward_interaction_length_m = interaction_length
    state.forward_interaction_n_bins = n_int
    state.forward_retention_scale = max(float(retention_scale), 0.0)
    state._forward_retention_scale = state.forward_retention_scale
    state.forward_source_virgin_count_per_bin = virgin_per_bin
    state.forward_source_capacity_field = capacity
    state.forward_source_available_field = capacity.copy()
    state.forward_source_cumulative_consumed = 0.0
    state.forward_source_cumulative_inflow = 0.0
    state.forward_source_cumulative_available_outflow = 0.0
    state.forward_source_last_inflow = 0.0
    state.forward_source_last_available_outflow = 0.0
    state.forward_source_last_emission_centroid_m = 0.0

    _sync_scalar_source_views(state)
    state.campaign_source_last_refresh_fraction = 0.0
    state.campaign_source_last_refresh_length_m = interaction_length
    state._transport_rates = MethodType(_forward_transport_rates, state)
    state._emit = MethodType(_forward_emit, state)
    state.advance = MethodType(_forward_advance, state)


class ForwardInteractionZoneTipEngine(CampaignCalibratedTipEngine):
    """Campaign engine with moving spatial source capacity ahead of the tip."""

    forward_interaction_zone_active = True
    _interaction_length_scale_default = 1.0
    _retention_scale_default = 1.0

    @classmethod
    def configure_forward_zone(
        cls,
        interaction_length_scale: float = 1.0,
        retention_scale: float = 1.0,
    ) -> None:
        cls._interaction_length_scale_default = max(
            float(interaction_length_scale), 1.0e-12
        )
        cls._retention_scale_default = max(float(retention_scale), 0.0)

    @classmethod
    def audit_payload(cls) -> dict[str, Any]:
        payload = super().audit_payload()
        payload["forward_interaction_zone"] = {
            "schema": FORWARD_SCHEMA,
            "interaction_length_scale": cls._interaction_length_scale_default,
            "retention_scale": cls._retention_scale_default,
            "total_virgin_source_content_from_manifest": True,
            "spatial_source_capacity_field": True,
            "far_boundary_virgin_inflow": True,
            "scalar_uniform_source_refresh": False,
            "wake_primary_toughening_state": False,
        }
        return payload

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        install_forward_interaction_zone_source(
            self.mpz,
            self._interaction_length_scale_default,
            self._retention_scale_default,
        )

    def _forward_diagnostics(self) -> dict[str, Any]:
        available = np.asarray(
            self.mpz.forward_source_available_field, dtype=float
        )
        capacity = np.asarray(
            self.mpz.forward_source_capacity_field, dtype=float
        )
        available_total = float(np.sum(available))
        consumed_local = float(np.sum(capacity - available))
        if available_total > 0.0:
            centroid = float(
                np.sum(available * self.mpz.x[None, :]) / available_total
            )
        else:
            centroid = 0.0
        depleted = np.maximum(capacity - available, 0.0)
        depleted_total = float(np.sum(depleted))
        depletion_centroid = (
            float(np.sum(depleted * self.mpz.x[None, :]) / depleted_total)
            if depleted_total > 0.0
            else 0.0
        )
        return {
            "forward_interaction_schema": FORWARD_SCHEMA,
            "forward_interaction_length_scale": self._interaction_length_scale_default,
            "forward_interaction_length_m": self.mpz.forward_interaction_length_m,
            "forward_interaction_n_bins": self.mpz.forward_interaction_n_bins,
            "forward_retention_scale": self._retention_scale_default,
            "forward_source_capacity_total": float(np.sum(capacity)),
            "forward_source_available_total": available_total,
            "forward_source_consumed_local": consumed_local,
            "forward_source_available_fraction": (
                available_total / float(np.sum(capacity))
                if float(np.sum(capacity)) > 0.0
                else 0.0
            ),
            "forward_source_available_centroid_m": centroid,
            "forward_source_depletion_centroid_m": depletion_centroid,
            "forward_source_last_emission_centroid_m": self.mpz.forward_source_last_emission_centroid_m,
            "forward_source_cumulative_consumed": self.mpz.forward_source_cumulative_consumed,
            "forward_source_cumulative_inflow": self.mpz.forward_source_cumulative_inflow,
            "forward_source_cumulative_available_outflow": self.mpz.forward_source_cumulative_available_outflow,
            "forward_source_last_inflow": self.mpz.forward_source_last_inflow,
            "forward_source_last_available_outflow": self.mpz.forward_source_last_available_outflow,
            "forward_active_mobile_count": float(self.mpz.mobile_count),
            "forward_active_retained_count": float(self.mpz.retained_count),
            "forward_active_total_count": float(
                self.mpz.mobile_count + self.mpz.retained_count
            ),
        }

    def step(self, K, T, dt):
        result = super().step(K, T, dt)
        diag = self._forward_diagnostics()
        result.update(diag)
        if type(self)._audit_records:
            type(self)._audit_records[-1].update(diag)
        return result

    def cycle_step_waveform(self, *args, **kwargs):
        result = super().cycle_step_waveform(*args, **kwargs)
        result.update(self._forward_diagnostics())
        return result
