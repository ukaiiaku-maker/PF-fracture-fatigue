"""Audit adapter for state-coupled v10.2.29 persistent-site fatigue."""
from __future__ import annotations

from .persistent_site_cyclic_audited_v10229 import (
    _add_persistent_fields,
    _cycle_block_audit_fields,
)
from .persistent_site_cyclic_coupled_v10229 import (
    CoupledPersistentSiteCyclicTipEngine,
)


_COUPLED_KEYS = (
    "coupled_hazard_model_id",
    "coupled_hazard_phase_resolved",
    "coupled_hazard_frozen_within_outer_block",
    "coupled_hazard_accepted_segments",
    "coupled_hazard_rejected_splits",
    "coupled_hazard_maximum_depth",
    "coupled_hazard_lambda_evaluations",
    "coupled_hazard_lambda_start_s",
    "coupled_hazard_lambda_end_s",
    "coupled_hazard_lambda_min_s",
    "coupled_hazard_lambda_max_s",
    "coupled_hazard_log_lambda_span_decades",
    "coupled_hazard_sigma_start_Pa",
    "coupled_hazard_sigma_end_Pa",
    "coupled_hazard_transient_cycles",
    "coupled_hazard_stationary_tail_cycles",
    "coupled_hazard_event_localized",
    "coupled_hazard_config",
    "coupled_hazard_segments",
)


def _coupled_fields(result: dict) -> dict:
    return {key: result[key] for key in _COUPLED_KEYS if key in result}


class AuditedCoupledPersistentSiteCyclicTipEngine(
    CoupledPersistentSiteCyclicTipEngine
):
    """Expose persistent-source and evolving-cleavage audit fields."""

    def step(self, K, T, dt):
        return _add_persistent_fields(self, super().step(K, T, dt))

    def cycle_step_waveform(
        self, controller, waveform, T_K: float, requested_cycles=None, force_cycles=None
    ):
        result = _add_persistent_fields(
            self,
            super().cycle_step_waveform(
                controller,
                waveform,
                T_K,
                requested_cycles=requested_cycles,
                force_cycles=force_cycles,
            ),
        )
        type(self)._audit_records.append(
            {
                "engine_id": int(getattr(self, "_engine_id", -1)),
                "loading_mode": "cyclic",
                "time_s": float(result.get("time_s", self.t)),
                "temperature_K": float(T_K),
                "cycles_requested": float(result.get("cycles_requested", 0.0)),
                "cycles_consumed": float(result.get("cycles_consumed", 0.0)),
                "cycles_unused": float(result.get("cycles_unused", 0.0)),
                "event_localized": bool(result.get("cycle_event_localized", False)),
                "fired": bool(result.get("fired", False)),
                "B": float(result.get("B", self.B)),
                "physical_hazard_action_block": float(
                    result.get("physical_hazard_action_block", 0.0)
                ),
                "persistent_sigma_back_Pa": float(result["sigma_back"]),
                "persistent_aggregate_emission_hazard_s": float(
                    result["persistent_site_aggregate_hazard_s"]
                ),
                "persistent_site_multiplicity_per_system": float(
                    result["persistent_site_multiplicity_per_system"]
                ),
                "persistent_site_front_width_m": float(
                    result["persistent_site_front_width_m"]
                ),
                "persistent_site_source_area_m2": float(
                    result["persistent_site_source_area_m2"]
                ),
                "persistent_tip_radius_m": float(result["persistent_tip_radius_m"]),
                "persistent_source_inventory_active": False,
                "persistent_source_refresh_active": False,
                "explicit_recovery_active": False,
                "engine_native_cycle_predictor": True,
                "state_coupled_cleavage_hazard": True,
                **_cycle_block_audit_fields(controller, self, result),
                **_coupled_fields(result),
            }
        )
        return result


__all__ = ["AuditedCoupledPersistentSiteCyclicTipEngine"]
