"""Explicit physical-cycle integration for v10.2.32 sharp-front LCF.

This module changes only the time-integration regime.  It advances the existing
persistent-site/cleavage engine through the established midpoint waveform
quadrature, stops at the first stochastic event, and retains the unused phase
for the outer geometry transaction.  The next outer solve therefore resumes at
the same physical-cycle phase rather than restarting at the waveform origin.
"""
from __future__ import annotations

import copy
import math
from typing import Any

import numpy as np

from .fatigue_v1 import CycleHazardResult
from .persistent_site_cyclic_energy_gated_corrected_v10230 import (
    CorrectedHazardEnergyGatedPersistentSiteCyclicTipEngine,
)
from .persistent_site_high_cycle_engine_v10230 import invalidate_high_cycle_cache


MODEL_ID = "v10.2.32_explicit_physical_cycle_same_phase_continuation_v1"


def _numeric(mapping: dict[str, Any], key: str) -> float:
    value = mapping.get(key, 0.0)
    return float(value) if isinstance(value, (int, float, np.integer, np.floating)) else 0.0


def _sum_numeric(target: dict[str, float], source: dict[str, Any]) -> None:
    for key, value in source.items():
        if isinstance(value, (int, float, np.integer, np.floating)):
            target[key] = target.get(key, 0.0) + float(value)


def _phase_position(engine) -> float:
    phase = float(getattr(engine, "explicit_cycle_phase", 0.0))
    if not math.isfinite(phase):
        raise FloatingPointError("explicit-cycle phase is non-finite")
    phase %= 1.0
    return 0.0 if phase >= 1.0 - 1.0e-14 else phase


def remaining_cycle_fraction(engine) -> float:
    """Return the unconsumed fraction of the current physical cycle."""
    phase = _phase_position(engine)
    return max(1.0 - phase, 0.0)


def select_explicit_cycle_block(
    controller,
    prediction,
    user_block_cycles: float | None,
    linear_diagnostic: dict[str, Any],
) -> dict[str, Any]:
    """Select one unprojected physical-cycle remainder.

    The exact phase marcher is authoritative and may return earlier at first
    passage.  No raw state target or high-cycle representation participates.
    """
    engine = getattr(prediction, "_v10229_vhcf_engine", None)
    if engine is None or not bool(getattr(engine, "explicit_cycle_v1032", False)):
        return dict(linear_diagnostic)
    selected = remaining_cycle_fraction(engine)
    audit = {
        "schema": MODEL_ID,
        "search_strategy": "one_explicit_physical_cycle_remainder",
        "cycles": selected,
        "limiter": "explicit_cycle_boundary",
        "phase_start": _phase_position(engine),
        "phase_end": 1.0,
        "private_trial_evaluations": 0,
        "multi_cycle_projection": False,
        "committer_authoritative": True,
        "partial_return_allowed": True,
    }
    engine._v10229_last_vhcf_block_audit = audit
    return {
        "cycles": selected,
        "limiter": "explicit_cycle_boundary",
        "unlimited_cycles": selected,
        "candidate_limits": {"explicit_cycle_remainder": selected},
    }


def _state_diagnostic(engine, *, phase: float, K: float, result: dict[str, Any]) -> dict[str, Any]:
    mpz = engine.mpz
    geometry = engine.source_geometry() if hasattr(engine, "source_geometry") else {}
    shielding = float(engine._active_shielding_signed()) if hasattr(engine, "_active_shielding_signed") else 0.0
    sigma_back = float(engine.sigma_back()) if hasattr(engine, "sigma_back") else 0.0
    dt = max(float(result.get("dt_consumed", 0.0)), 0.0)
    plastic = result.get("plastic", {})
    return {
        "cycle_index": int(getattr(engine, "explicit_cycle_index", 0)),
        "cycle_phase": float(phase),
        "applied_K_Pa_sqrt_m": float(K),
        "local_tip_stress_Pa": float(result.get("sigma_tip", engine.sigma_tip(K))),
        "cleavage_hazard_rate_s": float(result.get("lambda_c", 0.0)),
        "physical_hazard_action": float(getattr(engine, "hazard_action_current", 0.0)),
        "threshold_action": float(getattr(engine, "hazard_threshold_action", 1.0)),
        "B": float(getattr(engine, "B", 0.0)),
        "mobile_count": float(getattr(mpz, "mobile_count", 0.0)),
        "retained_count": float(getattr(mpz, "retained_count", 0.0)),
        "shielding_Pa_sqrt_m": shielding,
        "backstress_Pa": sigma_back,
        "tip_radius_m": float(engine.r_eff()) if hasattr(engine, "r_eff") else 0.0,
        "front_width_m": float(geometry.get("front_width_m", 0.0)),
        "emission_rate_s": _numeric(plastic, "dN_emit") / max(dt, 1.0e-300),
        "transport_escape_rate_s": _numeric(plastic, "dN_escaped") / max(dt, 1.0e-300),
        "phase_dt_s": dt,
        "fired": bool(result.get("fired", False)),
    }


def advance_explicit_cycle_remainder(
    engine,
    controller,
    waveform,
    temperature_K: float,
    requested_cycles: float,
) -> dict[str, Any]:
    """Advance phase bins transactionally until a cycle boundary or event."""
    n_phase = max(int(controller.cfg.n_phase), 8)
    frequency = max(float(waveform.frequency_Hz), 1.0e-300)
    period = float(waveform.period_s)
    remaining = min(max(float(requested_cycles), 0.0), remaining_cycle_fraction(engine))
    phase_start = _phase_position(engine)
    consumed = 0.0
    dB = dH = 0.0
    plastic: dict[str, float] = {}
    advance: dict[str, float] = {}
    phase_records: list[dict[str, Any]] = []
    microsteps = 0
    fired = False
    last: dict[str, Any] = {
        "lambda_c": 0.0, "lambda_c_raw": 0.0, "sigma_tip": engine.sigma_tip(float(waveform.Kmax)),
        "Gc_J": 0.0, "dt_consumed": 0.0, "dt_unused": 0.0,
    }
    while remaining > 1.0e-15 and not fired:
        phase = _phase_position(engine)
        index = min(int(math.floor(phase * n_phase + 1.0e-12)), n_phase - 1)
        boundary = (index + 1) / n_phase
        segment_cycles = min(remaining, max(boundary - phase, 0.0))
        if segment_cycles <= 1.0e-15:
            engine.explicit_cycle_phase = 0.0 if index + 1 >= n_phase else boundary
            if index + 1 >= n_phase:
                engine.explicit_cycle_index = int(getattr(engine, "explicit_cycle_index", 0)) + 1
            continue
        radians = (index + 0.5) * (2.0 * math.pi / n_phase)
        K_phase = float(np.asarray(waveform.K_phase(np.asarray([radians])))[0])
        setattr(engine, "_energy_gate_event_K_override", float(waveform.Kmax))
        try:
            current = engine._integrate_coupled(
                K_phase,
                float(temperature_K),
                segment_cycles * period,
            )
        finally:
            if hasattr(engine, "_energy_gate_event_K_override"):
                delattr(engine, "_energy_gate_event_K_override")
        used = max(float(current.get("dt_consumed", 0.0)), 0.0) * frequency
        if used <= 0.0 and segment_cycles > 0.0:
            raise RuntimeError("explicit-cycle phase marcher made zero physical progress")
        used = min(used, segment_cycles)
        consumed += used
        remaining = max(remaining - used, 0.0)
        new_phase = phase + used
        if new_phase >= 1.0 - 1.0e-13:
            engine.explicit_cycle_phase = 0.0
            engine.explicit_cycle_index = int(getattr(engine, "explicit_cycle_index", 0)) + 1
        else:
            engine.explicit_cycle_phase = new_phase
        dB += float(current.get("dB", 0.0))
        dH += float(current.get("physical_hazard_action_step", 0.0))
        _sum_numeric(plastic, current.get("plastic", {}))
        _sum_numeric(advance, current.get("advance", {}))
        microsteps += int(current.get("microsteps", 0))
        fired = bool(current.get("fired", False))
        phase_records.append(_state_diagnostic(
            engine, phase=_phase_position(engine), K=K_phase, result=current
        ))
        last = current
        if fired:
            invalidate_high_cycle_cache(engine, "explicit_cycle_first_passage_event")
            break
        if used < segment_cycles - 1.0e-13:
            raise RuntimeError("explicit-cycle segment returned partially without first passage")
    return {
        **last,
        "fired": fired,
        "n_fire": 1 if fired else 0,
        "dB": dB,
        "physical_hazard_action_step": dH,
        "plastic": plastic,
        "advance": advance,
        "microsteps": microsteps,
        "dt_consumed": consumed * period,
        "dt_unused": max(float(requested_cycles) - consumed, 0.0) * period,
        "explicit_cycle_phase_start": phase_start,
        "explicit_cycle_phase_end": _phase_position(engine),
        "explicit_cycle_index": int(getattr(engine, "explicit_cycle_index", 0)),
        "explicit_phase_records": phase_records,
    }


class ExplicitCycleHazardEnergyGatedPersistentSiteCyclicTipEngine(
    CorrectedHazardEnergyGatedPersistentSiteCyclicTipEngine
):
    """The production v10.2.30 engine with an explicit cycle clock."""

    explicit_cycle_v1032 = True

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.explicit_cycle_phase = 0.0
        self.explicit_cycle_index = 0
        self.explicit_cycle_event_count = 0

    def preview_cycle_waveform(self, controller, waveform, T_K: float) -> CycleHazardResult:
        # The inherited preview is private, RNG-neutral, and uses the same phase
        # quadrature.  It is diagnostic only; explicit block selection ignores
        # its tangent limits and commits no more than one physical cycle.
        return super().preview_cycle_waveform(controller, waveform, T_K)

    def cycle_step_waveform(
        self, controller, waveform, T_K: float, requested_cycles=None, force_cycles=None
    ):
        pred = self.preview_cycle_waveform(controller, waveform, T_K)
        requested = force_cycles if force_cycles is not None else requested_cycles
        if requested is None:
            requested = remaining_cycle_fraction(self)
        cycles_requested = min(max(float(requested), 0.0), remaining_cycle_fraction(self))
        B_pre = float(self.B)
        mobile_pre = float(self.mpz.mobile_count)
        retained_pre = float(self.mpz.retained_count)
        escaped_pre = float(self.mpz.escaped_total)
        emitted_pre = float(self.mpz.emitted_total)
        N_pre = float(self.N_em)
        coupled = advance_explicit_cycle_remainder(
            self, controller, waveform, T_K, cycles_requested
        )
        consumed = max(float(coupled["dt_consumed"]), 0.0) * float(waveform.frequency_Hz)
        if coupled["fired"]:
            self.explicit_cycle_event_count += 1
        diagnostics = self.mpz.diagnostics(self.G, self.nu, self.b, self.f.r0)
        active_signed = float(self._active_shielding_signed())
        wake_signed = float(self._wake_shielding_signed())
        plastic = coupled.get("plastic", {})
        advance = coupled.get("advance", {})
        result = {
            "cycles": consumed,
            "cycles_requested": cycles_requested,
            "cycles_consumed": consumed,
            "cycles_unused": max(cycles_requested - consumed, 0.0),
            "cycle_event_localized": bool(coupled["fired"]),
            "cycle_limiter": "explicit_first_passage" if coupled["fired"] else "explicit_cycle_boundary",
            "cycle_unlimited": cycles_requested,
            "cycle_candidate_limits": {"explicit_cycle_remainder": cycles_requested},
            "time_s": self.t,
            "Kmax_Pa_sqrt_m": float(waveform.Kmax),
            "Kmin_Pa_sqrt_m": float(waveform.R * waveform.Kmax),
            "DeltaK_Pa_sqrt_m": float(waveform.DeltaK),
            "R": float(waveform.R), "frequency_Hz": float(waveform.frequency_Hz),
            "T_K": float(T_K),
            "mu_emit": float(pred.mu_emit), "mu_peierls": float(pred.mu_peierls),
            "mu_taylor": float(pred.mu_taylor), "mu_escape": float(pred.mu_escape),
            "mu_cleave_pred": float(pred.mu_cleave),
            "store_per_cycle": float(pred.store_per_cycle),
            "mobile_per_cycle": float(pred.mobile_per_cycle),
            "escape_per_cycle": float(pred.escape_per_cycle),
            "peierls_per_cycle": float(pred.peierls_per_cycle),
            "taylor_per_cycle": float(pred.taylor_per_cycle),
            "storage_fraction": float(pred.storage_fraction),
            "lambda_e": float(pred.mu_emit * waveform.frequency_Hz),
            "lambda_c": float(coupled.get("lambda_c", 0.0)),
            "lambda_c_raw": float(coupled.get("lambda_c_raw", 0.0)),
            "sigma_tip": float(coupled.get("sigma_tip", self.sigma_tip(float(waveform.Kmax)))),
            "sigma_back": float(self.sigma_back()),
            "r_eff": float(self.r_eff()),
            "B_pre": B_pre, "B": float(self.B), "N_em": float(self.N_em),
            "N_em_pre_renewal": N_pre, "N_em_retained": float(self.N_em),
            "N_em_shed_to_wake": float(advance.get("wake_mobile", 0.0) + advance.get("wake_retained", 0.0)),
            "dN_emit_block": max(float(self.mpz.emitted_total) - emitted_pre, 0.0),
            "dN_store_block": abs(float(self.mpz.retained_count) - retained_pre),
            "dN_mobile_block": abs(float(self.mpz.mobile_count) - mobile_pre),
            "dN_escape_block": max(float(self.mpz.escaped_total) - escaped_pre, 0.0),
            "dN_peierls_block": _numeric(plastic, "dN_peierls"),
            "dN_taylor_block": _numeric(plastic, "dN_taylor"),
            "dB_block": float(coupled.get("dB", 0.0)),
            "physical_hazard_action_block": float(coupled.get("physical_hazard_action_step", 0.0)),
            "fired": bool(coupled["fired"]), "n_fire": int(coupled["n_fire"]),
            "v_crack": 0.0,
            "kinetic_tip_cell_active": True,
            "kinetic_micro_advance_step_m": float(coupled.get("da", 0.0)),
            "kinetic_micro_advance_total_m": float(self.micro_advance_total_m),
            "kinetic_active_K_shield_signed_Pa_sqrt_m": active_signed,
            "kinetic_wake_K_shield_signed_Pa_sqrt_m": wake_signed,
            "kinetic_internal_substeps": int(coupled.get("microsteps", 0)),
            "kinetic_dt_requested_s": cycles_requested * float(waveform.period_s),
            "kinetic_dt_consumed_s": float(coupled["dt_consumed"]),
            "kinetic_dt_unused_s": float(coupled["dt_unused"]),
            "persistent_site_cyclic_model_id": MODEL_ID,
            "persistent_site_engine_native_predictor": True,
            "legacy_fatigue_barrier_predictor_used": False,
            "cycle_integration_mode": "explicit",
            "explicit_cycle_phase_start": float(coupled["explicit_cycle_phase_start"]),
            "explicit_cycle_phase_end": float(coupled["explicit_cycle_phase_end"]),
            "explicit_cycle_index": int(coupled["explicit_cycle_index"]),
            "explicit_cycle_event_count": int(self.explicit_cycle_event_count),
            "explicit_same_cycle_post_event_continuation": True,
            "explicit_multi_cycle_projection": False,
            "explicit_phase_records": coupled["explicit_phase_records"],
            "high_cycle_cache_invalidated": bool(coupled["fired"]),
            "hazard_threshold_completed_action": float(coupled.get("hazard_threshold_completed_action", 0.0)),
            "hazard_action_completed": float(coupled.get("hazard_action_completed", 0.0)),
            "hazard_threshold_next_action": float(coupled.get("hazard_threshold_next_action", self.hazard_threshold_action)),
            **plastic, **advance, **diagnostics,
        }
        # Reuse the two production wrappers to attach transactional energy-gate
        # diagnostics and the pending result reference without re-integrating.
        result["hazard_energy_gate_model_id"] = getattr(self, "hazard_energy_gate_model_id", "v10.2.30")
        result["hazard_energy_gate_continuum"] = dict(self.energy_gate_last_continuum)
        result["hazard_energy_gate_continuum_open"] = bool(self.energy_gate_last_continuum.get("energy_gate_continuum_open", False))
        result["hazard_energy_gate_continuum_affects_hazard"] = False
        result["hazard_energy_gate_event_load_policy"] = self.event_load_policy
        result["transactional_event_translation_pending"] = bool(result["fired"] and not self._energy_gate_provisional)
        result["energy_gate_zero_length_attempt_count"] = int(self.energy_gate_zero_length_attempt_count)
        result.update(self._hazard_diagnostics())
        from .persistent_site_cyclic_audited_v10229 import (
            _add_persistent_fields, _cycle_block_audit_fields,
        )
        from .persistent_site_cyclic_coupled_audited_v10229 import (
            _active_state_snapshot, _coupled_fields,
        )
        result = _add_persistent_fields(self, result)
        if result["fired"] and not self._energy_gate_provisional:
            from .hazard_energy_event_gate_v10230 import attach_pending_event_info
            attach_pending_event_info(self._engine_id, result)
        type(self)._audit_records.append({
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
            "dB_block": float(result.get("dB_block", 0.0)),
            "physical_hazard_action_block": float(result.get("physical_hazard_action_block", 0.0)),
            "persistent_sigma_back_Pa": float(result["sigma_back"]),
            "persistent_aggregate_emission_hazard_s": float(result["persistent_site_aggregate_hazard_s"]),
            "persistent_site_multiplicity_per_system": float(result["persistent_site_multiplicity_per_system"]),
            "persistent_site_front_width_m": float(result["persistent_site_front_width_m"]),
            "persistent_site_source_area_m2": float(result["persistent_site_source_area_m2"]),
            "persistent_tip_radius_m": float(result["persistent_tip_radius_m"]),
            "persistent_source_inventory_active": False,
            "persistent_source_refresh_active": False,
            "explicit_recovery_active": False,
            "engine_native_cycle_predictor": True,
            "state_coupled_cleavage_hazard": True,
            **_cycle_block_audit_fields(controller, self, result),
            **_coupled_fields(result),
            **_active_state_snapshot(self),
        })
        return result


__all__ = [
    "ExplicitCycleHazardEnergyGatedPersistentSiteCyclicTipEngine",
    "MODEL_ID",
    "advance_explicit_cycle_remainder",
    "remaining_cycle_fraction",
    "select_explicit_cycle_block",
]
