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
    "coupled_hazard_forward_marcher",
    "coupled_hazard_partition_robust_state_control",
    "coupled_hazard_recursive_bisection",
    "coupled_hazard_two_half_step_state_committed",
    "coupled_hazard_third_commit_integration",
    "coupled_hazard_accepted_segments",
    "coupled_hazard_rejected_splits",
    "coupled_hazard_trial_integrations",
    "coupled_hazard_maximum_depth",
    "coupled_hazard_lambda_evaluations",
    "coupled_hazard_work_budget_exhausted",
    "coupled_hazard_zero_progress",
    "coupled_hazard_partial_return",
    "coupled_hazard_cycles_requested",
    "coupled_hazard_cycles_consumed",
    "coupled_hazard_lambda_start_s",
    "coupled_hazard_lambda_end_s",
    "coupled_hazard_lambda_min_s",
    "coupled_hazard_lambda_max_s",
    "coupled_hazard_lambda_start_per_s",
    "coupled_hazard_lambda_end_per_s",
    "coupled_hazard_lambda_min_per_s",
    "coupled_hazard_lambda_max_per_s",
    "coupled_hazard_log_lambda_span_decades",
    "coupled_hazard_sigma_start_Pa",
    "coupled_hazard_sigma_end_Pa",
    "coupled_hazard_shield_start_Pa_sqrt_m",
    "coupled_hazard_shield_end_Pa_sqrt_m",
    "coupled_hazard_mobile_start",
    "coupled_hazard_mobile_end",
    "coupled_hazard_retained_start",
    "coupled_hazard_retained_end",
    "coupled_hazard_backstress_start_Pa",
    "coupled_hazard_backstress_end_Pa",
    "coupled_hazard_next_segment_cycles",
    "coupled_hazard_transient_cycles",
    "coupled_hazard_stationary_tail_cycles",
    "coupled_hazard_event_localized",
    "coupled_hazard_wall_seconds",
    "coupled_hazard_config",
    "coupled_hazard_segments",
)

_PRE_STATE_KEYS = (
    "state_N_em_pre",
    "state_mobile_count_pre",
    "state_retained_count_pre",
    "state_emitted_total_pre",
    "state_escaped_total_pre",
    "state_micro_advance_total_m_pre",
    "state_active_K_shield_signed_Pa_sqrt_m_pre",
    "state_wake_K_shield_signed_Pa_sqrt_m_pre",
    "state_sigma_back_Pa_pre",
)


def _coupled_fields(result: dict) -> dict:
    keys = _COUPLED_KEYS + _PRE_STATE_KEYS
    return {key: result[key] for key in keys if key in result}


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
                "B_pre": float(result.get("B_pre", self.B)),
                "B": float(result.get("B", self.B)),
                "dB_block": float(result.get("dB_block", result.get("dB", 0.0))),
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


__all__ = ["AuditedCoupledPersistentSiteCyclicTipEngine", "_coupled_fields"]
