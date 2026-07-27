"""Event-load-consistent transactional engine for v10.2.30.

The cyclic cleavage hazard remains phase-resolved and state coupled. The separate
energetic admissibility and event-reward diagnostics are evaluated at the outer
geometry-event load, Kmax, because the rapid sharp-wake transaction is committed
at that same load. No independent fracture criterion or resistance is added.
"""
from __future__ import annotations

import copy
import math
from typing import Any

import numpy as np

from .fatigue_v1 import CycleHazardResult
from .persistent_site_cyclic_v10229 import _numeric
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
    """Evaluate selector, hazard gate, and reward at the geometry event load."""

    hazard_energy_gate_event_load_consistent_v10230 = True

    def preview_cycle_waveform(
        self, controller, waveform, T_K: float
    ) -> CycleHazardResult:
        """Use the same stationary-tip state policy for block selection and commit."""

        phase = np.asarray(controller._phases(), dtype=float)
        if phase.size < 1:
            raise ValueError("cyclic preview requires at least one waveform phase")
        Kvals = np.asarray(waveform.K_phase(phase), dtype=float)
        dt_phase = float(waveform.period_s) / float(phase.size)
        trial = copy.deepcopy(self)

        mobile0 = float(trial.mpz.mobile_count)
        retained0 = float(trial.mpz.retained_count)
        emitted = trapped = released = escaped = 0.0
        peierls_progress = taylor_progress = 0.0
        cleavage_action = 0.0
        sig_samples: list[float] = []
        emit_weighted_sig = 0.0
        emit_weight = 0.0
        elapsed_s = 0.0

        for kval in Kvals:
            K = max(float(kval), 0.0)
            half = 0.5 * dt_phase

            sig0 = float(trial.sigma_tip(K))
            first = trial._plastic_half_step(half, T_K, sig0)
            sig_mid = float(trial.sigma_tip(K))
            lam_mid, _raw_mid, _Gc_mid = trial.lambda_cleave(sig_mid, T_K)
            lam_mid = (
                max(float(lam_mid), 0.0) if math.isfinite(lam_mid) else 0.0
            )
            gate = continuum_gate_diagnostics(
                trial,
                float(waveform.Kmax),
                T_K,
                stress_override_Pa=None,
            )
            if not bool(gate["energy_gate_continuum_open"]):
                lam_mid = 0.0

            remaining_action = max(1.0 - float(trial.B), 0.0)
            dB_unclipped = lam_mid * dt_phase
            dB = min(dB_unclipped, remaining_action)
            trial.B += dB

            sig1 = float(trial.sigma_tip(K))
            second = trial._plastic_half_step(half, T_K, sig1)
            trial.t += dt_phase
            elapsed_s += dt_phase
            cleavage_action += dB_unclipped
            sig_samples.extend((sig0, sig_mid, sig1))

            for update in (first, second):
                emitted += _numeric(update, "dN_emit")
                trapped += _numeric(update, "dN_trapped")
                released += _numeric(update, "dN_released")
                escaped += _numeric(update, "dN_escaped")
                peierls_progress += half * max(
                    _numeric(update, "peierls_rate_s"), 0.0
                )
                taylor_progress += half * max(
                    _numeric(update, "taylor_completion_rate_s"), 0.0
                )
                weight = max(_numeric(update, "dN_emit"), 0.0)
                sigma_eff = float(
                    getattr(
                        trial.mpz,
                        "continuum_source_last_sigma_emit_effective_Pa",
                        sig_mid,
                    )
                )
                emit_weighted_sig += weight * max(sigma_eff, 0.0)
                emit_weight += weight

            if trial.B >= 1.0 - 1.0e-12:
                break

        elapsed_cycles = max(
            elapsed_s * float(waveform.frequency_Hz), 1.0e-300
        )
        scale = 1.0 / elapsed_cycles
        mobile_delta = abs(float(trial.mpz.mobile_count) - mobile0)
        retained_delta = abs(float(trial.mpz.retained_count) - retained0)

        emitted *= scale
        trapped *= scale
        released *= scale
        escaped *= scale
        peierls_progress *= scale
        taylor_progress *= scale
        cleavage_action *= scale
        mobile_delta *= scale
        retained_delta *= scale

        stored_per_cycle = max(retained_delta, trapped)
        mobile_per_cycle = max(mobile_delta, emitted + released - trapped)
        avg_sigma = float(np.mean(sig_samples)) if sig_samples else 0.0
        max_sigma = float(np.max(sig_samples)) if sig_samples else 0.0
        avg_emit_sigma = (
            emit_weighted_sig / emit_weight if emit_weight > 0.0 else avg_sigma
        )
        storage_fraction = (
            min(max(stored_per_cycle / emitted, 0.0), 1.0)
            if emitted > 0.0
            else 0.0
        )

        return CycleHazardResult(
            mu_emit=max(emitted, 0.0),
            mu_peierls=max(peierls_progress, 0.0),
            mu_taylor=max(taylor_progress, 0.0),
            mu_escape=max(escaped, 0.0),
            mu_cleave=max(cleavage_action, 0.0),
            store_per_cycle=max(stored_per_cycle, 0.0),
            mobile_per_cycle=max(mobile_per_cycle, 0.0),
            escape_per_cycle=max(escaped, 0.0),
            peierls_per_cycle=max(peierls_progress, 0.0),
            taylor_per_cycle=max(taylor_progress, 0.0),
            avg_sigma_tip=avg_sigma,
            max_sigma_tip=max_sigma,
            avg_sigma_emit_eff=max(float(avg_emit_sigma), 0.0),
            storage_fraction=storage_fraction,
        )

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
                "stationary_tip_selector_commit_parity": True,
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
