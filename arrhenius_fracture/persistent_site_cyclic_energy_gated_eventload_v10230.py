"""Event-load-consistent transactional engine for v10.2.30.

The cyclic cleavage hazard remains phase-resolved and state coupled. The separate
energetic admissibility and event-reward diagnostics are evaluated at the outer
geometry-event load, Kmax, because the rapid sharp-wake transaction is committed
at that same load. No independent fracture criterion or resistance is added.
"""
from __future__ import annotations

import copy
from typing import Any

from .persistent_site_cyclic_energy_gated_v10230 import (
    HazardEnergyGatedPersistentSiteCyclicTipEngine,
    MODEL_ID as TRANSACTION_MODEL_ID,
)
from .stochastic_hazard_tip import StochasticHazardDiagnosticTipEngine
from . import stochastic_avalanche_tip as _avalanche_tip
from .hazard_energy_event_gate_v10230 import continuum_gate_diagnostics


MODEL_ID = "v10.2.30_event_load_consistent_transactional_energy_gate"


class EventLoadConsistentHazardEnergyGatedPersistentSiteCyclicTipEngine(
    HazardEnergyGatedPersistentSiteCyclicTipEngine
):
    """Evaluate the extension gate and its barrier at the geometry event load."""

    hazard_energy_gate_event_load_consistent_v10230 = True

    def _integrate_coupled(
        self,
        K: float,
        T: float,
        dt: float,
        stress_override: float | None = None,
        lambda_override: float | None = None,
    ) -> dict[str, Any]:
        self._synchronize_driver_checkpoint_length()
        proposal = float(self.avalanche_event_advance_m)
        proposal_factor = float(self.avalanche_event_length_factor)

        # The coupled hazard quadrature may use a phase-averaged stress override,
        # but the rapid geometry transaction occurs at Kmax. Evaluate the energetic
        # initiation diagnostic from the current state at that same event load.
        continuum = continuum_gate_diagnostics(self, K, T, stress_override_Pa=None)
        continuum["hazard_integration_stress_override_Pa"] = (
            None if stress_override is None else float(stress_override)
        )
        continuum["energy_gate_event_load_consistent"] = True
        self.energy_gate_last_continuum = dict(continuum)

        effective_lambda = lambda_override
        if not bool(continuum["energy_gate_continuum_open"]):
            effective_lambda = 0.0

        base_da = float(self.f.da)
        rng_state_before = copy.deepcopy(self._hazard_rng.bit_generator.state)
        threshold_before = float(self.hazard_threshold_action)
        action_before = float(self.hazard_action_current)
        event_index_before = int(self.hazard_event_index)
        history_len_before = len(self.hazard_threshold_history)
        n_adv_before = int(self.n_adv)

        self.f.da = 0.0
        try:
            result = StochasticHazardDiagnosticTipEngine._integrate_coupled(
                self,
                K,
                T,
                dt,
                stress_override=stress_override,
                lambda_override=effective_lambda,
            )
        finally:
            self.f.da = base_da

        fired = bool(result.get("fired", False))
        if fired:
            completed_threshold = float(
                result.get("hazard_threshold_completed_action", threshold_before)
            )
            completed_action = float(
                result.get("hazard_action_completed", completed_threshold)
            )
            event_sigma = max(float(self.sigma_tip(float(K))), 0.0)
            _, _, event_barrier_J = self.lambda_cleave(event_sigma, float(T))
            event_barrier_J = max(float(event_barrier_J), 0.0)
            result["Gc_J"] = event_barrier_J
            result["energy_gate_event_sigma_tip_Pa"] = event_sigma
            result["energy_gate_event_barrier_J"] = event_barrier_J
            descriptor = {
                "event_advance_m": proposal,
                "event_length_factor": proposal_factor,
                "threshold_action": completed_threshold,
                "hazard_seed": int(self.hazard_cfg.seed),
                "hazard_event_index": int(self.hazard_event_index - 1),
                "geometry_subsegment_fraction": float(
                    self.avalanche_cfg.geometry_subsegment_fraction
                ),
                "energy_gate_engine_id": int(self._engine_id),
                "event_K_Pa_sqrt_m": max(float(K), 0.0),
                "event_temperature_K": float(T),
                "event_sigma_tip_Pa": event_sigma,
                "hazard_barrier_J": event_barrier_J,
                "hazard_cooperative_hits": float(self.f.m_hits),
                "hazard_burgers_vector_m": float(self.b),
                "energy_gate_continuum": dict(continuum),
                "energy_gate_event_load_consistent": True,
            }
            self._energy_gate_pending = {
                "descriptor": descriptor,
                "rng_state_before": rng_state_before,
                "threshold_before": threshold_before,
                "action_before": action_before,
                "event_index_before": event_index_before,
                "history_len_before": history_len_before,
                "n_adv_before": n_adv_before,
                "proposal_m": proposal,
                "proposal_factor": proposal_factor,
            }
            if not self._energy_gate_provisional:
                _avalanche_tip._PENDING_GEOMETRY_EVENTS.append(descriptor)
            self._set_current_event_length()

        result.update(
            {
                "hazard_energy_gate_model_id": MODEL_ID,
                "hazard_energy_gate_transaction_model_id": TRANSACTION_MODEL_ID,
                "hazard_energy_gate_continuum_open": bool(
                    continuum["energy_gate_continuum_open"]
                ),
                "hazard_energy_gate_continuum": dict(continuum),
                "energy_gate_event_load_consistent": True,
                "stochastic_event_proposed_advance_m": proposal if fired else 0.0,
                "stochastic_event_proposed_factor": proposal_factor if fired else 0.0,
                "avalanche_event_advance_m": 0.0,
                "avalanche_event_length_factor": 0.0,
                "avalanche_current_event_advance_m": float(
                    self.avalanche_event_advance_m
                ),
                "avalanche_current_event_length_factor": float(
                    self.avalanche_event_length_factor
                ),
                "kinetic_micro_advance_step_m": 0.0,
                "kinetic_checkpoint_progress_m": float(self.B)
                * float(self.avalanche_event_advance_m),
                "transactional_event_translation_pending": bool(
                    fired and not self._energy_gate_provisional
                ),
            }
        )
        return result


__all__ = [
    "EventLoadConsistentHazardEnergyGatedPersistentSiteCyclicTipEngine",
    "MODEL_ID",
]
