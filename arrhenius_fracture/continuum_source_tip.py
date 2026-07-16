"""Minimal continuum source law with local emission back stress.

Distributed plasticity remains a continuum Peierls--Taylor transport/storage
problem.  No spatial or discrete distributed-source inventory is introduced.
The only source state is one dimensionless activity fraction for each
crystallographic tip channel.  It is exhausted by the aggregate Arrhenius
nucleation hazard and recovers when emitted material clears the tip or when
crack advance renews the local geometry over the current blunted tip radius.

The manifest ``source_sites_per_system`` is retained only as the calibrated
low-rate hazard multiplicity.  It is placed inside one aggregate channel
hazard per crystallographic system and is never consumed as a finite
population.

Emission is driven by the tip stress minus a local Taylor back stress computed
from the evolving mobile-plus-retained density ahead of the crack tip.  The
existing resolved-stress fraction supplies the normal/shear projection, so no
new fitted back-stress coefficient or density scale is introduced.
"""
from __future__ import annotations

import math
from types import MethodType
from typing import Any

import numpy as np

from .fractional_moving_frame import fractional_moving_frame_advance
from .kinetic_tip_cell import KineticMovingTipFrontEngine


SOURCE_MODEL = "minimal_continuum_local_emission_backstress"


def _local_tip_density_m2(state) -> np.ndarray:
    """Return exponentially weighted mobile-plus-retained density per system.

    The averaging length is the current blunted tip radius, bounded below by
    the existing MPZ blunting length and grid spacing.  Counts are converted to
    line density using the same local strip width used by the MPZ forest law.
    Only explicitly evolved dislocations contribute; the numerical background
    forest-density floor does not create an initial emission back stress.
    """
    radius = max(
        float(getattr(state, "_continuum_tip_radius_m", state.cfg.blunting_length_m)),
        float(state.cfg.blunting_length_m),
        float(state.dx),
        abs(float(getattr(state, "_continuum_b", 0.0))),
        1.0e-12,
    )
    width = max(float(state.cfg.blunting_length_m), float(state.dx), 1.0e-12)
    weights = np.exp(-np.asarray(state.x, dtype=float) / radius)
    weight_sum = max(float(np.sum(weights)), 1.0e-30)
    count = (
        np.maximum(np.asarray(state.mobile, dtype=float), 0.0)
        + np.maximum(np.asarray(state.retained, dtype=float), 0.0)
    )
    weighted_count_per_bin = np.sum(count * weights[None, :], axis=1) / weight_sum
    return np.maximum(
        weighted_count_per_bin / max(float(state.dx) * width, 1.0e-30),
        0.0,
    )


def _local_emission_backstress_Pa(
    state,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return local density, Taylor shear, and equivalent emission back stress.

    tau_back = G b sqrt(rho_tip)
    sigma_back = tau_back / m

    where ``m`` is the existing resolved-stress fraction.  This is the minimal
    opposing-stress closure requested for crack-tip emission and introduces no
    additional fitted coefficient.
    """
    rho = _local_tip_density_m2(state)
    G = max(float(getattr(state, "_continuum_G_Pa", 0.0)), 0.0)
    b = abs(float(getattr(state, "_continuum_b", 0.0)))
    tau_back = G * b * np.sqrt(np.maximum(rho, 0.0))
    resolved_fraction = max(
        abs(float(getattr(state.cfg, "taylor_stress_fraction", 1.0))),
        1.0e-6,
    )
    sigma_back = tau_back / resolved_fraction
    return rho, tau_back, sigma_back


def _continuum_emit(
    self,
    dt: float,
    stress_Pa: float,
    T_K: float,
    system_weights: np.ndarray | None = None,
) -> float:
    """Cycle one aggregate tip channel per crystallographic system.

    The local dislocation cloud first reduces the stress entering the promoted
    Arrhenius emission barrier:

        sigma_eff,s = max(sigma_tip - sigma_back,s, 0)

        Lambda_s = M_ref * lambda_emit(sigma_eff,s, T) * w_s

        dq_s/dt = k_clear (1-q_s) - Lambda_s q_s

    The occupancy equation is integrated exactly.  At high nucleation hazard
    the repeated-event throughput saturates at the Peierls clearing rate per
    crystallographic channel, while the evolving Taylor back stress can shut
    nucleation down before that limit is reached.
    """
    dt = max(float(dt), 0.0)
    if dt <= 0.0:
        return 0.0

    nsrc = max(min(int(self.cfg.source_bin_count), self.n_bins), 1)
    stress_profile = self.local_stress_profile_Pa(stress_Pa)
    rho_transport = self.local_forest_density_m2(False)
    rates = self._transport_rates(
        stress_profile, rho_transport, T_K, self._continuum_b
    )

    velocity_clear = float(np.mean(np.maximum(rates["velocity"][:nsrc], 0.0)))
    radius = max(
        float(getattr(self, "_continuum_tip_radius_m", self.cfg.blunting_length_m)),
        abs(float(self._continuum_b)),
        1.0e-12,
    )
    clear_rate = velocity_clear / radius

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

    rho_back, tau_back, sigma_back = _local_emission_backstress_Pa(self)
    sigma_tip = max(float(stress_Pa), 0.0)
    sigma_effective = np.maximum(sigma_tip - sigma_back, 0.0)
    lam_site = np.maximum(
        np.asarray(self.manifest.emission.rate(sigma_effective, T_K), dtype=float),
        0.0,
    )

    reference = max(float(self.reference_source_multiplicity), 0.0)
    aggregate_hazard = reference * lam_site * weights
    activity0 = np.clip(
        np.asarray(self.tip_source_activity, dtype=float), 0.0, 1.0
    )
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
    throughput_bound = np.maximum(activity0 + clear_rate * dt, 0.0)
    tolerance = 1.0e-10 * np.maximum(1.0, throughput_bound)
    if np.any(emitted_system > throughput_bound + tolerance):
        raise RuntimeError(
            "aggregate tip-source throughput exceeded activity-plus-clearing bound"
        )

    self.tip_source_activity = activity1
    self.available_sites = reference * activity1

    self.mobile[:, :nsrc] += emitted_system[:, None] / nsrc
    self.accumulated_slip[:, :nsrc] += emitted_system[:, None] / nsrc
    emitted = float(np.sum(emitted_system))
    self.emitted_total += emitted

    self.continuum_source_last_clear_rate_s = clear_rate
    self.continuum_source_last_effective_multiplicity = float(
        np.sum(self.available_sites)
    )
    self.continuum_source_last_emission_rate_s = float(
        np.sum(aggregate_hazard * activity0)
    )
    self.continuum_source_last_aggregate_hazard_s = float(
        np.sum(aggregate_hazard)
    )
    self.continuum_source_last_throughput_bound = float(
        np.sum(throughput_bound)
    )
    self.continuum_source_last_rho_back_m2 = float(np.mean(rho_back))
    self.continuum_source_last_tau_back_Pa = float(np.mean(tau_back))
    self.continuum_source_last_sigma_back_Pa = float(np.mean(sigma_back))
    self.continuum_source_last_sigma_emit_effective_Pa = float(
        np.mean(sigma_effective)
    )
    self.continuum_source_last_sigma_emit_effective_min_Pa = float(
        np.min(sigma_effective)
    )
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
    self.available_sites = (
        max(float(self.reference_source_multiplicity), 0.0)
        * self.tip_source_activity
    )
    refreshed = max(float(np.sum(self.available_sites - before_sites)), 0.0)
    result["source_sites_refreshed"] = refreshed
    result["tip_source_activity_recovered_geometry"] = float(
        np.sum(self.tip_source_activity - activity0)
    )
    result["tip_source_geometry_fraction"] = fraction
    return result


def install_minimal_continuum_source(
    state, b: float, G_Pa: float | None = None
) -> None:
    """Install the aggregate-channel law on one unified MPZ state instance."""
    reference = max(float(state.manifest.source_sites_per_system), 0.0)
    state.source_model = SOURCE_MODEL
    state.reference_source_multiplicity = reference
    state.tip_source_activity = np.ones(state.n_systems, dtype=float)
    state._continuum_b = float(b)
    state._continuum_G_Pa = max(float(G_Pa or 0.0), 0.0)
    state._continuum_tip_radius_m = max(
        float(state.cfg.blunting_length_m), abs(float(b))
    )
    state.site_capacity = np.full(state.n_systems, reference, dtype=float)
    state.available_sites = state.site_capacity.copy()
    state.continuum_source_last_clear_rate_s = 0.0
    state.continuum_source_last_effective_multiplicity = float(
        np.sum(state.available_sites)
    )
    state.continuum_source_last_emission_rate_s = 0.0
    state.continuum_source_last_aggregate_hazard_s = 0.0
    state.continuum_source_last_throughput_bound = float(state.n_systems)
    state.continuum_source_last_rho_back_m2 = 0.0
    state.continuum_source_last_tau_back_Pa = 0.0
    state.continuum_source_last_sigma_back_Pa = 0.0
    state.continuum_source_last_sigma_emit_effective_Pa = 0.0
    state.continuum_source_last_sigma_emit_effective_min_Pa = 0.0
    state._emit = MethodType(_continuum_emit, state)
    state.advance = MethodType(_continuum_advance, state)


def source_diagnostics(state) -> dict[str, Any]:
    return {
        "tip_source_model": str(
            getattr(state, "source_model", "legacy_finite_sites")
        ),
        "tip_source_activity_mean": float(
            np.mean(
                getattr(
                    state,
                    "tip_source_activity",
                    np.ones(state.n_systems),
                )
            )
        ),
        "tip_source_activity_min": float(
            np.min(
                getattr(
                    state,
                    "tip_source_activity",
                    np.ones(state.n_systems),
                )
            )
        ),
        "tip_source_reference_multiplicity_per_system": float(
            getattr(state, "reference_source_multiplicity", 0.0)
        ),
        "tip_source_effective_multiplicity_total": float(
            getattr(
                state,
                "continuum_source_last_effective_multiplicity",
                np.sum(state.available_sites),
            )
        ),
        "tip_source_clear_rate_s": float(
            getattr(state, "continuum_source_last_clear_rate_s", 0.0)
        ),
        "tip_source_emission_rate_s": float(
            getattr(state, "continuum_source_last_emission_rate_s", 0.0)
        ),
        "tip_source_aggregate_hazard_s": float(
            getattr(state, "continuum_source_last_aggregate_hazard_s", 0.0)
        ),
        "tip_source_throughput_bound_step": float(
            getattr(state, "continuum_source_last_throughput_bound", 0.0)
        ),
        "tip_source_local_density_m2": float(
            getattr(state, "continuum_source_last_rho_back_m2", 0.0)
        ),
        "tip_source_backstress_shear_Pa": float(
            getattr(state, "continuum_source_last_tau_back_Pa", 0.0)
        ),
        "tip_source_backstress_equivalent_Pa": float(
            getattr(state, "continuum_source_last_sigma_back_Pa", 0.0)
        ),
        "tip_source_effective_emission_stress_Pa": float(
            getattr(
                state,
                "continuum_source_last_sigma_emit_effective_Pa",
                0.0,
            )
        ),
        "tip_source_effective_emission_stress_min_Pa": float(
            getattr(
                state,
                "continuum_source_last_sigma_emit_effective_min_Pa",
                0.0,
            )
        ),
    }


class ContinuumSourceKineticTipEngine(KineticMovingTipFrontEngine):
    """Moving-tip engine with continuum Peierls--Taylor distributed plasticity."""

    minimal_continuum_source_active = True

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        install_minimal_continuum_source(self.mpz, self.b, self.G)

    def _plastic_half_step(
        self, dt: float, T: float, stress: float
    ) -> dict[str, float]:
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
