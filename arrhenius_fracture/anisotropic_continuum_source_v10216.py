"""Anisotropic extension of the accepted continuum tip-source law.

This module changes no material barrier, transport operator, shielding kernel,
FEM/J calculation, or crack-geometry transaction.  It only prevents the later
anisotropic-emission installer from replacing the accepted v10.1.3 continuum
source lifecycle with a finite, depletable source inventory.

``source_sites_per_system`` remains a low-rate Arrhenius hazard multiplicity.
The only source state is the dimensionless channel activity ``q_s``.  Activity
is reduced by emission and restored continuously by Peierls clearing; the
existing continuum crack-advance method restores activity by geometric renewal
over the current effective tip radius.
"""
from __future__ import annotations

import math
from types import MethodType
from typing import Any

import numpy as np

from . import anisotropic_emission_v10174 as _anisotropic
from .continuum_source_tip import (
    SOURCE_MODEL as CONTINUUM_SOURCE_MODEL,
    _local_emission_backstress_Pa,
    install_minimal_continuum_source,
)

MODEL_ID = "v10.2.16_anisotropic_continuum_source_parameter_transfer"
SOURCE_MODEL = "minimal_continuum_local_emission_backstress_anisotropic"
_ORIGINAL_ANISOTROPIC_INSTALL = _anisotropic.install_anisotropic_campaign_emission


def _anisotropic_continuum_emit(
    self,
    dt: float,
    stress_Pa: float,
    T_K: float,
    system_weights: np.ndarray | None = None,
) -> float:
    """Evolve one aggregate continuum source channel per slip system exactly.

    The directional factor enters the effective stress before the Arrhenius
    barrier.  No second post-hazard weight is permitted.  The multiplicity
    ``M_ref`` is never decremented::

        sigma_eff,s = max(f_s sigma_tip - sigma_back,s, 0)
        Lambda_s = M_ref lambda_s(sigma_eff,s, T)
        dq_s/dt = k_clear (1-q_s) - Lambda_s q_s

    Emitted activity is inserted into the existing spatial mobile/slip fields;
    all subsequent Peierls--Taylor transport, trapping, release, recovery, and
    crack-advance renewal remain the inherited production implementation.
    """
    dt = max(float(dt), 0.0)
    if dt <= 0.0:
        return 0.0
    if system_weights is not None:
        supplied = np.asarray(system_weights, dtype=float)
        if supplied.size and not np.allclose(supplied, 1.0):
            raise RuntimeError(
                "post-hazard system_weights are forbidden in anisotropic continuum emission"
            )

    nsrc = max(min(int(self.cfg.source_bin_count), self.n_bins), 1)
    stress_profile = self.local_stress_profile_Pa(stress_Pa)
    rho_transport = self.local_forest_density_m2(False)
    transport = self._transport_rates(
        stress_profile,
        rho_transport,
        T_K,
        self._continuum_b,
    )
    velocity_clear = float(
        np.mean(np.maximum(np.asarray(transport["velocity"][:nsrc]), 0.0))
    )
    radius = max(
        float(getattr(self, "_continuum_tip_radius_m", self.cfg.blunting_length_m)),
        abs(float(self._continuum_b)),
        1.0e-12,
    )
    clear_rate = velocity_clear / radius

    factors = _anisotropic._drive_factors_for_state(self)
    rho_back, tau_back, sigma_back = _local_emission_backstress_Pa(self)
    sigma_opening = max(float(stress_Pa), 0.0)
    sigma_emit = np.maximum(factors * sigma_opening - sigma_back, 0.0)
    rates_per_site = np.maximum(
        np.asarray(
            [
                self.emission_rate_per_site(float(value), T_K)
                for value in sigma_emit
            ],
            dtype=float,
        ),
        0.0,
    )

    reference = max(float(self.reference_source_multiplicity), 0.0)
    aggregate_hazard = reference * rates_per_site
    activity0 = np.clip(
        np.asarray(self.tip_source_activity, dtype=float),
        0.0,
        1.0,
    )
    total_rate = aggregate_hazard + clear_rate
    equilibrium = np.divide(
        clear_rate,
        total_rate,
        out=np.ones_like(total_rate),
        where=total_rate > 0.0,
    )
    decay = np.exp(-np.minimum(total_rate * dt, 700.0))
    integrated_activity = equilibrium * dt + np.divide(
        (activity0 - equilibrium) * (1.0 - decay),
        total_rate,
        out=activity0 * dt,
        where=total_rate > 0.0,
    )
    activity1 = np.clip(
        equilibrium + (activity0 - equilibrium) * decay,
        0.0,
        1.0,
    )

    emitted_by_system = np.maximum(aggregate_hazard * integrated_activity, 0.0)
    throughput_bound = np.maximum(activity0 + clear_rate * dt, 0.0)
    tolerance = 1.0e-10 * np.maximum(1.0, throughput_bound)
    if np.any(emitted_by_system > throughput_bound + tolerance):
        raise RuntimeError(
            "aggregate continuum source throughput exceeded activity-plus-clearing bound"
        )

    self.tip_source_activity = activity1
    # Compatibility projection only.  This is M_ref*q, not a finite inventory.
    self.site_capacity = np.full(self.n_systems, reference, dtype=float)
    self.available_sites = reference * activity1
    self.mobile[:, :nsrc] += emitted_by_system[:, None] / nsrc
    self.accumulated_slip[:, :nsrc] += emitted_by_system[:, None] / nsrc
    emitted = float(np.sum(emitted_by_system))
    self.emitted_total += emitted

    probability = 1.0 - np.exp(-np.minimum(aggregate_hazard * dt, 700.0))
    self.continuum_source_last_clear_rate_s = float(clear_rate)
    self.continuum_source_last_effective_multiplicity = float(
        np.sum(reference * activity1)
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
    self.continuum_source_last_sigma_emit_effective_Pa = float(np.mean(sigma_emit))
    self.continuum_source_last_sigma_emit_effective_min_Pa = float(np.min(sigma_emit))

    # Legacy budget diagnostics are retained as explicit compatibility fields,
    # but the reference multiplicity is constant and never consumed.
    self.campaign_source_budget_remaining_total = float(
        self.n_systems * reference
    )
    self.campaign_source_budget_consumed_total = 0.0
    self.anisotropic_last_drive_factors = factors.copy()
    self.anisotropic_last_sigma_opening_Pa = float(sigma_opening)
    self.anisotropic_last_rho_back_by_system_m2 = rho_back.copy()
    self.anisotropic_last_tau_back_by_system_Pa = tau_back.copy()
    self.anisotropic_last_sigma_back_by_system_Pa = sigma_back.copy()
    self.anisotropic_last_sigma_emit_by_system_Pa = sigma_emit.copy()
    self.anisotropic_last_lambda_emit_by_system_s = rates_per_site.copy()
    self.anisotropic_last_probability_by_system = probability.copy()
    self.anisotropic_last_dN_emit_by_system = emitted_by_system.copy()
    return emitted


def install_anisotropic_continuum_emission(
    state: Any,
    config: _anisotropic.AnisotropicEmissionConfig,
) -> None:
    """Install anisotropic driving while preserving the accepted source lifecycle."""
    b = float(getattr(state, "_campaign_b", getattr(state, "_continuum_b", 0.0)))
    G = float(getattr(state, "_campaign_G_Pa", getattr(state, "_continuum_G_Pa", 0.0)))
    install_minimal_continuum_source(state, b=b, G_Pa=G)

    # Reuse the audited tensor-drive and transport initialization, then replace
    # only its finite-site emission method with the continuum activity equation.
    _ORIGINAL_ANISOTROPIC_INSTALL(state, config)
    state._emit = MethodType(_anisotropic_continuum_emit, state)
    state.source_model = SOURCE_MODEL
    state.continuum_source_reference_model = CONTINUUM_SOURCE_MODEL
    state.continuum_source_finite_inventory = False
    state.continuum_source_multiplicity_consumed = False
    state.continuum_source_available_sites_semantics = "derived_M_ref_times_activity_proxy"


def audit_payload() -> dict[str, Any]:
    return {
        "model_id": MODEL_ID,
        "source_model": SOURCE_MODEL,
        "reference_source_model": CONTINUUM_SOURCE_MODEL,
        "finite_distributed_source_inventory": False,
        "source_sites_per_system_role": "low_rate_arrhenius_hazard_multiplicity",
        "source_multiplicity_consumed": False,
        "source_state": "dimensionless_activity_per_crystallographic_tip_channel",
        "activity_loss": "aggregate_arrhenius_emission_hazard",
        "activity_recovery_stationary": "peierls_clearing_over_current_tip_radius",
        "activity_recovery_crack_advance": "geometric_renewal_over_current_tip_radius",
        "local_backstress": "evolving_mobile_plus_retained_density",
        "anisotropic_factor_location": "effective_stress_before_arrhenius_barrier",
        "post_hazard_directional_weighting": False,
        "material_barriers_changed": False,
        "transport_operator_changed": False,
        "shielding_law_changed": False,
        "crack_geometry_changed": False,
    }


__all__ = [
    "MODEL_ID",
    "SOURCE_MODEL",
    "_anisotropic_continuum_emit",
    "install_anisotropic_continuum_emission",
    "audit_payload",
]
