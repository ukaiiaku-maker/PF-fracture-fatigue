"""Audit adapter for state-coupled v10.2.29 persistent-site fatigue."""
from __future__ import annotations

import os

from .persistent_site_cyclic_audited_v10229 import (
    _add_persistent_fields,
    _cycle_block_audit_fields,
)
from .persistent_site_cyclic_coupled_v10229 import (
    CoupledPersistentSiteCyclicTipEngine,
)
from .persistent_site_high_cycle_state_v10230 import serialize_active_state


_COUPLED_KEYS = (
    "coupled_hazard_model_id",
    "coupled_hazard_phase_resolved",
    "coupled_hazard_frozen_within_outer_block",
    "coupled_hazard_forward_marcher",
    "coupled_hazard_partition_robust_state_control",
    "coupled_hazard_high_cycle_engine",
    "coupled_hazard_phase_resolved_poincare_map",
    "coupled_hazard_periodic_solver",
    "coupled_hazard_stationary_first_passage",
    "coupled_hazard_slow_projective",
    "coupled_hazard_event_restart",
    "coupled_hazard_rate_separated_ledgers",
    "coupled_hazard_positivity_preserving_coordinates",
    "coupled_hazard_live_checkpointing",
    "coupled_hazard_growth_objective",
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
    "coupled_hazard_mode_operations",
    "coupled_hazard_modes",
    "coupled_hazard_geometry_preserved_before_event",
    "coupled_hazard_ledger_delta",
    "coupled_hazard_stochastic_threshold_preserved_until_event",
    "coupled_hazard_wall_seconds",
    "coupled_hazard_config",
    "coupled_hazard_segments",
    "cycle_integration_mode",
    "explicit_cycle_phase_start",
    "explicit_cycle_phase_end",
    "explicit_cycle_index",
    "explicit_cycle_event_count",
    "explicit_same_cycle_post_event_continuation",
    "explicit_multi_cycle_projection",
    "explicit_phase_records",
    "high_cycle_cache_invalidated",
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


def _active_state_snapshot(engine) -> dict:
    enabled = os.environ.get("V10230_SAVE_ACTIVE_STATE_SNAPSHOT", "0").strip().lower()
    if enabled not in {"1", "true", "yes", "on"}:
        return {}
    snapshot = serialize_active_state(engine)
    return {
        "coupled_hazard_active_state_snapshot": {
            "model_id": "v10.2.30_complete_high_cycle_active_state_v1",
            "vector": snapshot.vector.tolist(),
            "fields": [
                {
                    "owner": field.owner,
                    "name": field.name,
                    "shape": list(field.shape),
                    "start": int(field.start),
                    "stop": int(field.stop),
                    "floor": float(field.floor),
                }
                for field in snapshot.fields
            ],
            "diagnostics": dict(snapshot.diagnostics),
            "geometry_signature": list(snapshot.geometry_signature),
        }
    }


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
                **_active_state_snapshot(self),
            }
        )
        return result


__all__ = [
    "AuditedCoupledPersistentSiteCyclicTipEngine",
    "_active_state_snapshot",
    "_coupled_fields",
]
