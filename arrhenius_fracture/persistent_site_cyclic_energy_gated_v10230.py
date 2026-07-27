"""Transactional persistent-site cyclic engine for v10.2.30.

Cleavage first passage is unchanged. During the waiting cycles the persistent-site
source/mobile/retained state evolves at a stationary geometric tip. A completed
renewal creates a pending stochastic event proposal. The moving-frame state and
sharp-wake geometry are translated together only after the hazard-derived energy
gate approves the committed event distance.
"""
from __future__ import annotations

import copy
from typing import Any

from .persistent_site_cyclic_coupled_audited_v10229 import (
    AuditedCoupledPersistentSiteCyclicTipEngine,
)
from .stochastic_hazard_tip import StochasticHazardDiagnosticTipEngine
from . import stochastic_avalanche_tip as _avalanche_tip
from .hazard_energy_event_gate_v10230 import (
    attach_pending_event_info,
    continuum_gate_diagnostics,
    register_engine,
)


MODEL_ID = "v10.2.30_transactional_persistent_site_energy_gated_cyclic"


class HazardEnergyGatedPersistentSiteCyclicTipEngine(
    AuditedCoupledPersistentSiteCyclicTipEngine
):
    """State-coupled fatigue with atomic energy-gated event translation."""

    hazard_energy_gated_v10230 = True

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._energy_gate_provisional = False
        self._energy_gate_pending: dict[str, Any] | None = None
        self.energy_gate_last_continuum: dict[str, Any] = {}
        self.energy_gate_committed_event_count = 0
        self.energy_gate_committed_path_m = 0.0
        register_engine(self)

    def __deepcopy__(self, memo):
        cls = type(self)
        result = cls.__new__(cls)
        memo[id(self)] = result
        for key, value in self.__dict__.items():
            setattr(result, key, copy.deepcopy(value, memo))
        result._energy_gate_provisional = True
        result._energy_gate_pending = None
        return result

    def _integrate_coupled(
        self,
        K: float,
        T: float,
        dt: float,
        stress_override: float | None = None,
        lambda_override: float | None = None,
    ) -> dict[str, Any]:
        """Integrate hazard/plasticity without translating the crack before commit."""

        self._synchronize_driver_checkpoint_length()
        proposal = float(self.avalanche_event_advance_m)
        proposal_factor = float(self.avalanche_event_length_factor)
        continuum = continuum_gate_diagnostics(
            self,
            K,
            T,
            stress_override_Pa=stress_override,
        )
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
            barrier_J = max(float(result.get("Gc_J", 0.0)), 0.0)
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
                "event_sigma_tip_Pa": float(result.get("sigma_tip", 0.0)),
                "hazard_barrier_J": barrier_J,
                "hazard_cooperative_hits": float(self.f.m_hits),
                "hazard_burgers_vector_m": float(self.b),
                "energy_gate_continuum": dict(continuum),
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
                "hazard_energy_gate_continuum_open": bool(
                    continuum["energy_gate_continuum_open"]
                ),
                "hazard_energy_gate_continuum": dict(continuum),
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

    def cycle_step_waveform(
        self,
        controller,
        waveform,
        T_K: float,
        requested_cycles=None,
        force_cycles=None,
    ):
        result = super().cycle_step_waveform(
            controller,
            waveform,
            T_K,
            requested_cycles=requested_cycles,
            force_cycles=force_cycles,
        )
        result["hazard_energy_gate_model_id"] = MODEL_ID
        result["hazard_energy_gate_continuum"] = dict(
            self.energy_gate_last_continuum
        )
        result["hazard_energy_gate_continuum_open"] = bool(
            self.energy_gate_last_continuum.get(
                "energy_gate_continuum_open", False
            )
        )
        result["transactional_event_translation_pending"] = bool(
            result.get("fired", False) and not self._energy_gate_provisional
        )
        if result.get("fired", False) and not self._energy_gate_provisional:
            attach_pending_event_info(self._engine_id, result)
        return result

    def commit_energy_gated_event(
        self,
        committed_length_m: float,
        gate: dict[str, Any],
        result_ref: dict[str, Any] | None,
    ) -> None:
        """Commit moving-frame translation after the geometry gate succeeds."""

        length = max(float(committed_length_m), 0.0)
        if length <= 0.0:
            raise ValueError("committed energy-gated event length must be positive")
        pending = self._energy_gate_pending
        if pending is None:
            raise RuntimeError("no pending transactional event exists")

        advance = self.mpz.advance(length)
        self.micro_advance_total_m += length
        self.a_adv += length
        self.checkpoint_advance_total_m += length
        self.avalanche_last_completed_advance_m = length
        base = max(float(self.avalanche_base_checkpoint_m), 1.0e-300)
        self.avalanche_last_completed_factor = length / base
        self.avalanche_event_length_history.append(length)
        self.energy_gate_committed_event_count += 1
        self.energy_gate_committed_path_m += length

        info = result_ref if isinstance(result_ref, dict) else {}
        dt_used = max(
            float(
                info.get(
                    "kinetic_dt_consumed_s",
                    info.get("dt_consumed", 0.0),
                )
            ),
            0.0,
        )
        info.update(
            {
                "hazard_energy_gate_model_id": MODEL_ID,
                "transactional_event_translation_pending": False,
                "stochastic_event_proposed_advance_m": float(
                    pending["proposal_m"]
                ),
                "stochastic_event_proposed_factor": float(
                    pending["proposal_factor"]
                ),
                "energy_admissible_event_length_m": float(
                    gate.get("energy_admissible_event_length_m", length)
                ),
                "avalanche_event_advance_m": length,
                "avalanche_event_length_factor": length / base,
                "kinetic_micro_advance_step_m": length,
                "kinetic_micro_advance_total_m": float(
                    self.micro_advance_total_m
                ),
                "kinetic_checkpoint_committed_total_m": float(
                    self.checkpoint_advance_total_m
                ),
                "v_crack": length / dt_used if dt_used > 0.0 else 0.0,
                "N_em": float(self.N_em),
                "N_em_retained": float(self.N_em),
                "N_em_shed_to_wake": float(
                    advance.get("wake_mobile", 0.0)
                    + advance.get("wake_retained", 0.0)
                ),
                **{
                    key: value
                    for key, value in advance.items()
                    if isinstance(value, (int, float))
                },
                **{
                    key: value
                    for key, value in gate.items()
                    if key not in {"equilibrated_displacement", "trial_rows"}
                },
            }
        )

        records = getattr(type(self), "_audit_records", None)
        if isinstance(records, list):
            for record in reversed(records):
                if int(record.get("engine_id", -1)) == int(self._engine_id):
                    record.update(
                        {
                            "hazard_energy_gate_model_id": MODEL_ID,
                            "stochastic_event_proposed_advance_m": float(
                                pending["proposal_m"]
                            ),
                            "energy_gated_event_advance_m": length,
                            "energy_gate_arrest_reason": str(
                                gate.get("arrest_reason", "unknown")
                            ),
                            "hazard_resistance_J_per_m2": float(
                                gate.get("hazard_resistance_J_per_m2", 0.0)
                            ),
                            "orientation_gamma_relative": float(
                                gate.get("orientation_gamma_relative", 1.0)
                            ),
                            "athermal_Gc_used": False,
                        }
                    )
                    break
        self._energy_gate_pending = None

    def restore_geometry_veto(self, n_restore: int = 1) -> None:
        """Restore a completed first passage without undoing accepted plastic time."""

        pending = self._energy_gate_pending
        if pending is None:
            self.B += float(max(int(n_restore), 1))
            return
        self._hazard_rng.bit_generator.state = copy.deepcopy(
            pending["rng_state_before"]
        )
        self.hazard_threshold_action = float(pending["threshold_before"])
        self.hazard_action_current = float(
            max(pending["action_before"], pending["threshold_before"])
        )
        self.B = 1.0
        self.hazard_event_index = int(pending["event_index_before"])
        while len(self.hazard_threshold_history) > int(
            pending["history_len_before"]
        ):
            self.hazard_threshold_history.pop()
        self.n_adv = int(pending["n_adv_before"])
        self._set_current_event_length()
        self._energy_gate_pending = None


__all__ = [
    "HazardEnergyGatedPersistentSiteCyclicTipEngine",
    "MODEL_ID",
]
