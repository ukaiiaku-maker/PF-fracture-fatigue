"""Minimal non-depleting source law for the v10.1 moving-tip cell.

Distributed plasticity remains a continuum Peierls--Taylor transport/storage
problem.  No spatial or discrete distributed-source inventory is introduced.
The only source state is one dimensionless activity fraction for each
crystallographic tip channel.  It is exhausted by the aggregate Arrhenius
nucleation hazard and recovers when emitted material clears the tip or when
crack advance renews the local geometry over the current blunted tip radius.

The manifest ``source_sites_per_system`` is retained only as the already
calibrated low-rate hazard multiplicity.  It is placed inside the aggregate
channel hazard; it is not interpreted as that many independently recycling
channels and is never consumed as a finite population.
"""
from __future__ import annotations

import math
from types import MethodType
from typing import Any

import numpy as np

from .fractional_moving_frame import fractional_moving_frame_advance
from .kinetic_tip_cell import KineticMovingTipFrontEngine


SOURCE_MODEL = "minimal_continuum_aggregate_tip_channel"


def _source_hardening_activity(state) -> np.ndarray:
    """Return a parameter-free Taylor crowding factor per tip system.

    Mobile and retained dislocations in the source-zone bins both reduce tip
    cycling.  The promoted Taylor correlation density supplies the only scale,
    so no additional hardening or source-density parameter is introduced.
    """
    nsrc = max(min(int(state.cfg.source_bin_count), state.n_bins), 1)
    width = max(float(state.cfg.blunting_length_m), state.dx, 1.0e-12)
    near_tip = (
        np.maximum(state.mobile[:, :nsrc], 0.0)
        + np.maximum(state.retained[:, :nsrc], 0.0)
    )
    line_count = np.sum(near_tip, axis=1)
    excess_rho = line_count / max(nsrc * state.dx * width, 1.0e-30)
    rho_c = max(float(state.manifest.taylor_corr_rho_c_m2), 1.0)
    return 1.0 / (1.0 + np.sqrt(np.maximum(excess_rho, 0.0) / rho_c))


def _continuum_emit(self, dt: float, stress_Pa: float, T_K: float,
                    system_weights: np.ndarray | None = None) -> float:
    """Cycle one aggregate tip channel per crystallographic system.

    For system ``s`` the aggregate nucleation hazard is

        Lambda_s = M_ref * lambda_site * h_s * w_s,

    where ``M_ref`` is the promoted legacy multiplicity and ``h_s`` is the
    current Taylor crowding factor.  The activity satisfies

        dq_s/dt = k_clear (1-q_s) - Lambda_s q_s.

    This equation is integrated exactly.  Crucially, the high-hazard emission
    throughput saturates at ``k_clear`` per crystallographic channel, not at
    ``M_ref * k_clear``.  The earlier v10.1.1 placement of ``M_ref`` outside the
    occupancy equation created the observed runaway shielding state.
    """
    dt = max(float(dt), 0.0)
    if dt <= 0.0:
        return 0.0

    nsrc = max(min(int(self.cfg.source_bin_count), self.n_bins), 1)
    stress_profile = self.local_stress_profile_Pa(stress_Pa)
    rho = self.local_forest_density_m2(False)
    rates = self._transport_rates(stress_profile, rho, T_K, self._continuum_b)

    velocity_clear = float(np.mean(np.maximum(rates["velocity"][:nsrc], 0.0)))
    radius = max(
        float(getattr(self, "_continuum_tip_radius_m", self.cfg.blunting_length_m)),
        abs(float(self._continuum_b)),
        1.0e-12,
    )
    clear_rate = velocity_clear / radius

    hardening = _source_hardening_activity(self)
    lam_site = max(self.emission_rate_per_site(stress_Pa, T_K), 0.0)
    weights = np.ones(self.n_systems, dtype=float)
    if system_weights is not None:
        raw = np.maximum(np.asarray(system_weights, dtype=float).reshape(-1), 0.0)
        if raw.size < self.n_systems:
            raw = np.pad(raw, (0, self.n_systems - raw.size), mode="edge")
        raw = raw[: self.n_systems]
        if np.max(raw) > 0.0:
            weights = raw / np.max(raw)
        else:
            weights[:] = 0.0

    reference = max(float(self.reference_source_multiplicity), 0.0)
    aggregate_hazard = reference * lam_site * hardening * weights
    activity0 = np.clip(np.asarray(self.tip_source_activity, dtype=float), 0.0, 1.0)
    total_rate = aggregate_hazard + clear_rate
    qeq = np.divide(
        clear_rate,
        total_rate,
        out=np.ones_like(total_rate),
        where=total_rate > 0.0,
    )
    decay = np.exp(-np.minimum(total_rate * dt, 700.0))
    integral_q = qeq * dt + np.divide(
        (activity0 - qeq) * (1.0 - decay),
        total_rate,
        out=activity0 * dt,
        where=total_rate > 0.0,
    )
    activity1 = np.clip(qeq + (activity0 - qeq) * decay, 0.0, 1.0)

    emitted_system = np.maximum(aggregate_hazard * integral_q, 0.0)
    # One initially active aggregate channel may fire once, after which repeated
    # events are limited by physical clearing.  This fail-fast invariant catches
    # any future reintroduction of parallel-channel multiplication.
    throughput_bound = np.maximum(activity0 + clear_rate * dt, 0.0)
    tolerance = 1.0e-10 * np.maximum(1.0, throughput_bound)
    if np.any(emitted_system > throughput_bound + tolerance):
        raise RuntimeError(
            "aggregate tip-source throughput exceeded activity-plus-clearing bound"
        )

    self.tip_source_activity = activity1
    self.available_sites = reference * activity1 * hardening

    self.mobile[:, :nsrc] += emitted_system[:, None] / nsrc
    self.accumulated_slip[:, :nsrc] += emitted_system[:, None] / nsrc
    emitted = float(np.sum(emitted_system))
    self.emitted_total += emitted

    self.continuum_source_last_clear_rate_s = clear_rate
    self.continuum_source_last_hardening = float(np.mean(hardening))
    self.continuum_source_last_effective_multiplicity = float(np.sum(self.available_sites))
    self.continuum_source_last_emission_rate_s = float(
        np.sum(aggregate_hazard * activity0)
    )
    self.continuum_source_last_aggregate_hazard_s = float(np.sum(aggregate_hazard))
    self.continuum_source_last_throughput_bound = float(np.sum(throughput_bound))
    return emitted


def _continuum_advance(self, distance_m: float) -> dict[str, float]:
    """Fractionally advect the MPZ and renew tip activity by geometry change."""
    before_sites = np.asarray(self.available_sites, dtype=float).copy()
    activity0 = np.asarray(self.tip_source_activity, dtype=float).copy()
    result = fractional_moving_frame_advance(self, distance_m)

    d = max(float(distance_m), 0.0)
    radius = max(
        float(getattr(self, "_continuum_tip_radius_m", self.cfg.blunting_length_m)),
        abs(float(self._continuum_b)),
        1.0e-12,
    )
    fraction = 1.0 - math.exp(-min(d / radius, 700.0))
    self.tip_source_activity = np.clip(
        activity0 + (1.0 - activity0) * fraction,
        0.0,
        1.0,
    )
    hardening = _source_hardening_activity(self)
    self.available_sites = (
        max(float(self.reference_source_multiplicity), 0.0)
        * self.tip_source_activity
        * hardening
    )
    refreshed = max(float(np.sum(self.available_sites - before_sites)), 0.0)
    result["source_sites_refreshed"] = refreshed
    result["tip_source_activity_recovered_geometry"] = float(
        np.sum(self.tip_source_activity - activity0)
    )
    result["tip_source_geometry_fraction"] = fraction
    return result


def install_minimal_continuum_source(state, b: float) -> None:
    """Install the aggregate-channel law on one unified MPZ state instance."""
    reference = max(float(state.manifest.source_sites_per_system), 0.0)
    state.source_model = SOURCE_MODEL
    state.reference_source_multiplicity = reference
    state.tip_source_activity = np.ones(state.n_systems, dtype=float)
    state._continuum_b = float(b)
    state._continuum_tip_radius_m = max(float(state.cfg.blunting_length_m), abs(float(b)))
    state.site_capacity = np.full(state.n_systems, reference, dtype=float)
    state.available_sites = state.site_capacity.copy()
    state.continuum_source_last_clear_rate_s = 0.0
    state.continuum_source_last_hardening = 1.0
    state.continuum_source_last_effective_multiplicity = float(np.sum(state.available_sites))
    state.continuum_source_last_emission_rate_s = 0.0
    state.continuum_source_last_aggregate_hazard_s = 0.0
    state.continuum_source_last_throughput_bound = float(state.n_systems)
    state._emit = MethodType(_continuum_emit, state)
    state.advance = MethodType(_continuum_advance, state)


def source_diagnostics(state) -> dict[str, Any]:
    return {
        "tip_source_model": str(getattr(state, "source_model", "legacy_finite_sites")),
        "tip_source_activity_mean": float(np.mean(getattr(state, "tip_source_activity", np.ones(state.n_systems)))),
        "tip_source_activity_min": float(np.min(getattr(state, "tip_source_activity", np.ones(state.n_systems)))),
        "tip_source_reference_multiplicity_per_system": float(getattr(state, "reference_source_multiplicity", 0.0)),
        "tip_source_effective_multiplicity_total": float(getattr(state, "continuum_source_last_effective_multiplicity", np.sum(state.available_sites))),
        "tip_source_hardening_factor": float(getattr(state, "continuum_source_last_hardening", 1.0)),
        "tip_source_clear_rate_s": float(getattr(state, "continuum_source_last_clear_rate_s", 0.0)),
        "tip_source_emission_rate_s": float(getattr(state, "continuum_source_last_emission_rate_s", 0.0)),
        "tip_source_aggregate_hazard_s": float(getattr(state, "continuum_source_last_aggregate_hazard_s", 0.0)),
        "tip_source_throughput_bound_step": float(getattr(state, "continuum_source_last_throughput_bound", 0.0)),
    }


class ContinuumSourceKineticTipEngine(KineticMovingTipFrontEngine):
    """Moving-tip engine with continuum Peierls--Taylor distributed plasticity."""

    minimal_continuum_source_active = True

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        install_minimal_continuum_source(self.mpz, self.b)

    def _plastic_half_step(self, dt: float, T: float, stress: float) -> dict[str, float]:
        self.mpz._continuum_tip_radius_m = self.r_eff()
        result = super()._plastic_half_step(dt, T, stress)
        result.update(source_diagnostics(self.mpz))
        return result

    def step(self, K, T, dt):
        result = super().step(K, T, dt)
        diag = source_diagnostics(self.mpz)
        result.update(diag)
        if type(self)._audit_records:
            type(self)._audit_records[-1].update(diag)
        return result

    def cycle_step_waveform(self, *args, **kwargs):
        result = super().cycle_step_waveform(*args, **kwargs)
        result.update(source_diagnostics(self.mpz))
        return result
