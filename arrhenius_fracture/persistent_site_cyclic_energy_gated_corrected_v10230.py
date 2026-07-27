"""Corrections to the transactional v10.2.30 persistent-site fatigue engine.

The continuum K^2/E' comparison is retained only as a diagnostic.  It does not
suppress or rescale the cleavage hazard.  A completed first-passage event uses
the active cleavage barrier reevaluated at Kmax after the accepted waiting-cycle
state evolution.  If the full anisotropic FEM energy trial admits no positive
geometry increment, the first-passage attempt is consumed as a nonpropagating
attempt and the next stochastic threshold remains active.
"""
from __future__ import annotations

import copy
from typing import Any

from . import hazard_energy_event_gate_v10230 as _gate
from . import persistent_site_cyclic_energy_gated_v10230 as _base
from . import stochastic_avalanche_tip as _avalanche_tip


MODEL_ID = "v10.2.30_corrected_transactional_energy_gated_cyclic"


class CorrectedHazardEnergyGatedPersistentSiteCyclicTipEngine(
    _base.HazardEnergyGatedPersistentSiteCyclicTipEngine
):
    """Preserve first-passage kinetics and gate only the committed event reward."""

    hazard_energy_gate_continuum_affects_hazard = False
    event_load_policy = "cycle_maximum_Kmax"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.energy_gate_zero_length_attempt_count = 0

    def _integrate_coupled(
        self,
        K: float,
        T: float,
        dt: float,
        stress_override: float | None = None,
        lambda_override: float | None = None,
    ) -> dict[str, Any]:
        diagnostic: dict[str, Any] = {}
        original_continuum = _base.continuum_gate_diagnostics

        def diagnostic_only(*args, **kwargs):
            nonlocal diagnostic
            diagnostic = dict(_gate.continuum_gate_diagnostics(*args, **kwargs))
            admitted = dict(diagnostic)
            admitted["energy_gate_continuum_open_diagnostic"] = bool(
                diagnostic.get("energy_gate_continuum_open", False)
            )
            admitted["energy_gate_continuum_open"] = True
            admitted["energy_gate_continuum_affects_hazard"] = False
            return admitted

        _base.continuum_gate_diagnostics = diagnostic_only
        try:
            result = super()._integrate_coupled(
                K,
                T,
                dt,
                stress_override=stress_override,
                lambda_override=lambda_override,
            )
        finally:
            _base.continuum_gate_diagnostics = original_continuum

        self.energy_gate_last_continuum = dict(diagnostic)
        result["hazard_energy_gate_continuum"] = dict(diagnostic)
        result["hazard_energy_gate_continuum_open"] = bool(
            diagnostic.get("energy_gate_continuum_open", False)
        )
        result["hazard_energy_gate_continuum_affects_hazard"] = False
        result["hazard_energy_gate_event_load_policy"] = self.event_load_policy

        if bool(result.get("fired", False)):
            event_K = max(float(K), 0.0)
            event_sigma = max(float(self.sigma_tip(event_K)), 0.0)
            _, _, event_barrier_J = self.lambda_cleave(event_sigma, float(T))
            pending = self._energy_gate_pending
            if pending is not None:
                descriptor = pending["descriptor"]
                descriptor.update(
                    {
                        "event_K_Pa_sqrt_m": event_K,
                        "event_sigma_tip_Pa": event_sigma,
                        "hazard_barrier_J": max(float(event_barrier_J), 0.0),
                        "event_load_policy": self.event_load_policy,
                        "hazard_energy_gate_continuum_affects_hazard": False,
                    }
                )
                for queued in reversed(_avalanche_tip._PENDING_GEOMETRY_EVENTS):
                    if int(queued.get("energy_gate_engine_id", -1)) == int(
                        self._engine_id
                    ):
                        queued.update(descriptor)
                        break
            result.update(
                {
                    "event_K_Pa_sqrt_m": event_K,
                    "event_sigma_tip_Pa": event_sigma,
                    "hazard_barrier_J": max(float(event_barrier_J), 0.0),
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
        result["hazard_energy_gate_continuum_affects_hazard"] = False
        result["hazard_energy_gate_event_load_policy"] = self.event_load_policy
        result["energy_gate_zero_length_attempt_count"] = int(
            self.energy_gate_zero_length_attempt_count
        )
        return result

    def _latest_zero_length_gate(self) -> dict[str, Any] | None:
        backend = getattr(_gate, "_LAST_BACKEND", None)
        log = getattr(backend, "advance_log", None)
        if not isinstance(log, list) or not log:
            return None
        row = log[-1]
        if bool(row.get("inserted", True)):
            return None
        if str(row.get("arrest_reason", "")) != "no_energy_admissible_increment":
            return None
        if float(row.get("committed_event_length_m", 0.0)) > 0.0:
            return None
        return dict(row)

    def _consume_zero_length_attempt(self, gate: dict[str, Any]) -> None:
        pending = self._energy_gate_pending
        if pending is None:
            return
        self.n_adv = int(pending.get("n_adv_before", self.n_adv))
        self.energy_gate_zero_length_attempt_count += 1
        info = pending["descriptor"].get("energy_gate_result_ref")
        if isinstance(info, dict):
            info.update(
                {
                    "fired": False,
                    "n_fire": 0,
                    "transactional_event_translation_pending": False,
                    "hazard_energy_gate_attempt_consumed": True,
                    "hazard_energy_gate_zero_length_attempt": True,
                    "stochastic_event_proposed_advance_m": float(
                        pending.get("proposal_m", 0.0)
                    ),
                    "energy_admissible_event_length_m": 0.0,
                    "avalanche_event_advance_m": 0.0,
                    "kinetic_micro_advance_step_m": 0.0,
                    "energy_gate_zero_length_attempt_count": int(
                        self.energy_gate_zero_length_attempt_count
                    ),
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
                            "fired": False,
                            "hazard_energy_gate_attempt_consumed": True,
                            "hazard_energy_gate_zero_length_attempt": True,
                            "energy_gated_event_advance_m": 0.0,
                            "energy_gate_arrest_reason": str(
                                gate.get("arrest_reason", "unknown")
                            ),
                            "energy_gate_zero_length_attempt_count": int(
                                self.energy_gate_zero_length_attempt_count
                            ),
                            "athermal_Gc_used": False,
                        }
                    )
                    break
        self._energy_gate_pending = None

    def restore_geometry_veto(self, n_restore: int = 1) -> None:
        gate = self._latest_zero_length_gate()
        if self._energy_gate_pending is not None and gate is not None:
            self._consume_zero_length_attempt(gate)
            return
        super().restore_geometry_veto(n_restore)


__all__ = [
    "CorrectedHazardEnergyGatedPersistentSiteCyclicTipEngine",
    "MODEL_ID",
]
