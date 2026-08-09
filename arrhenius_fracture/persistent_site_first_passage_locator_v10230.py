"""Transactional cycle-number first-passage localization for v10.2.30.

This operator changes numerical orchestration only.  Candidate states are private
copies, the production stochastic state is immutable until the final
phase-resolved commit, and crack geometry is never advanced here.
"""
from __future__ import annotations

import copy
import math
from typing import Any

import numpy as np

from .persistent_site_high_cycle_state_v10230 import (
    capture_stochastic_state,
    geometry_signature,
)
from .persistent_site_poincare_v10230 import one_cycle_map


MODEL_ID = "v10.2.30_transactional_cycle_number_first_passage_locator_v1"


def _sum(target: dict[str, float], source: dict[str, Any]) -> None:
    for key, value in source.items():
        if isinstance(value, (int, float, np.integer, np.floating)):
            target[key] = target.get(key, 0.0) + float(value)


def _private_action(engine, controller, waveform, temperature_K: float, cycles: float):
    """Evaluate H(cycles) without permitting a stochastic or geometry event."""
    from . import stochastic_avalanche_tip as avalanche

    work = copy.deepcopy(engine)
    stochastic = capture_stochastic_state(engine)
    geometry = geometry_signature(engine)
    pending_before = list(avalanche._PENDING_GEOMETRY_EVENTS)
    phases = np.asarray(controller._phases(), dtype=float)
    K_values = np.asarray(waveform.K_phase(phases), dtype=float)
    period = float(waveform.period_s)
    total_phases = max(float(cycles), 0.0) * len(phases)
    action = 0.0
    evaluations = 0
    try:
        phase_index = 0
        while phase_index < math.ceil(total_phases):
            phase_fraction = min(total_phases - phase_index, 1.0)
            result = work._integrate_coupled(
                max(float(K_values[phase_index % len(phases)]), 0.0),
                float(temperature_K), period * phase_fraction / len(phases),
            )
            action += max(float(result.get("physical_hazard_action_step", 0.0)), 0.0)
            evaluations += 1
            if bool(result.get("fired", False)):
                break
            phase_index += 1
    finally:
        avalanche._PENDING_GEOMETRY_EVENTS.clear()
        avalanche._PENDING_GEOMETRY_EVENTS.extend(pending_before)
    if capture_stochastic_state(engine) != stochastic:
        raise RuntimeError("first-passage candidate changed live stochastic state")
    if geometry_signature(engine) != geometry:
        raise RuntimeError("first-passage candidate changed live geometry")
    return action, evaluations


def _commit_phase_resolved(
    engine, controller, waveform, temperature_K: float, cycles: float,
    threshold: float,
):
    phases = np.asarray(controller._phases(), dtype=float)
    K_values = np.asarray(waveform.K_phase(phases), dtype=float)
    remaining = max(float(cycles), 0.0) * len(phases)
    period = float(waveform.period_s)
    result: dict[str, Any] = {}
    plastic: dict[str, float] = {}
    advance: dict[str, float] = {}
    action = consumed_s = da = packet = variance = 0.0
    microsteps = 0
    phase_index = 0
    while phase_index < math.ceil(remaining):
        K_value = K_values[phase_index % len(phases)]
        phase_fraction = min(max(remaining - phase_index, 0.0), 1.0)
        if phase_fraction <= 0.0:
            break
        step = engine._integrate_coupled(
            max(float(K_value), 0.0), float(temperature_K),
            period * phase_fraction / len(phases),
        )
        result = dict(step)
        action += max(float(step.get("physical_hazard_action_step", 0.0)), 0.0)
        consumed_s += max(float(step.get("dt_consumed", 0.0)), 0.0)
        da += max(float(step.get("da", 0.0)), 0.0)
        packet += max(float(step.get("packet_mean", 0.0)), 0.0)
        variance += max(float(step.get("packet_variance_m2", 0.0)), 0.0)
        microsteps += int(step.get("microsteps", 0))
        _sum(plastic, dict(step.get("plastic", {})))
        _sum(advance, dict(step.get("advance", {})))
        if bool(step.get("fired", False)):
            break
        phase_index += 1
    result.update({
        "fired": bool(result.get("fired", False)),
        "n_fire": int(bool(result.get("fired", False))),
        "dB": action / max(float(threshold), 1.0e-300),
        "physical_hazard_action_step": action,
        "da": da,
        "dt_consumed": consumed_s,
        "dt_unused": max(cycles * period - consumed_s, 0.0),
        "packet_mean": packet,
        "packet_variance_m2": variance,
        "plastic": plastic,
        "advance": advance,
        "microsteps": microsteps,
    })
    return result


def localize_first_passage(
    engine, controller, waveform, temperature_K: float, cycles_available: float,
) -> dict[str, Any]:
    """Bracket in cycle number, safeguard the root, then commit one event."""
    stochastic_before = capture_stochastic_state(engine)
    geometry_before = geometry_signature(engine)
    threshold = float(stochastic_before["hazard_threshold_action"])
    current = float(stochastic_before["hazard_action_current"])
    remaining_action = max(threshold - current, 0.0)
    cycle = one_cycle_map(engine, controller, waveform, temperature_K)
    rate = max(float(cycle.hazard_action_per_cycle), 0.0)
    if rate <= 0.0 or cycles_available <= 0.0:
        return {"fired": False, "cycles": 0.0, "dH": 0.0,
                "evaluations": len(cycle.phase_hazard_action), "failure_reason": "zero_hazard_rate"}

    estimate = min(max(remaining_action / rate, 0.0), float(cycles_available))
    low, f_low = 0.0, -remaining_action
    high = min(max(1.0, math.ceil(estimate)), float(cycles_available))
    action_high, evaluations = _private_action(
        engine, controller, waveform, temperature_K, high)
    f_high = action_high - remaining_action
    while f_high < 0.0 and high < cycles_available:
        low, f_low = high, f_high
        high = min(max(2.0 * high, high + 1.0), float(cycles_available))
        action_high, used = _private_action(
            engine, controller, waveform, temperature_K, high)
        evaluations += used
        f_high = action_high - remaining_action
    if f_high < 0.0:
        bounded = _commit_phase_resolved(
            engine, controller, waveform, temperature_K,
            float(cycles_available), threshold,
        )
        consumed_bounded = max(
            float(bounded.get("dt_consumed", 0.0)), 0.0
        ) * float(waveform.frequency_Hz)
        bounded.update({
            "coupled_hazard_cycles_consumed": consumed_bounded,
            "coupled_hazard_partial_return": False,
            "coupled_hazard_event_localized": False,
            "coupled_hazard_first_passage_locator": True,
            "coupled_hazard_locator_model_id": MODEL_ID,
            "coupled_hazard_locator_failure_reason": "no_bracket_within_horizon",
            "coupled_hazard_locator_trial_evaluations": evaluations,
        })
        return bounded

    # Safeguarded integer-cycle bisection.  Keeping both endpoints at waveform
    # boundaries is essential: the only fractional advance is the final exact
    # waveform cycle below.
    iterations = 0
    while high - low > 1.0:
        iterations += 1
        candidate = float(math.floor(0.5 * (low + high)))
        if candidate <= low:
            candidate = low + 1.0
        action, used = _private_action(
            engine, controller, waveform, temperature_K, candidate)
        evaluations += used
        value = action - remaining_action
        if value >= 0.0:
            high, f_high = candidate, value
        else:
            low, f_low = candidate, value

    # The low endpoint is committed as exact whole cycles without stochastic
    # consumption, then the final waveform cycle consumes exactly one passage.
    committed_cycles = 0.0
    plastic: dict[str, float] = {}
    whole_low = int(math.floor(low))
    for _ in range(whole_low):
        mapped = one_cycle_map(engine, controller, waveform, temperature_K)
        from .persistent_site_high_cycle_engine_v10230_v2 import _commit_exact_cycle, _cache
        _commit_exact_cycle(engine, mapped, _cache(engine))
        committed_cycles += 1.0
        _sum(plastic, mapped.plastic_totals)
    final_span = high - committed_cycles
    final = _commit_phase_resolved(
        engine, controller, waveform, temperature_K, final_span, threshold)
    committed_cycles += max(float(final.get("dt_consumed", 0.0)), 0.0) * float(waveform.frequency_Hz)
    if not final.get("fired", False):
        raise RuntimeError("bracketed first passage did not fire in final waveform cycle")
    geometry_after = geometry_signature(engine)
    # The stochastic engine provisionally increments n_adv when it constructs
    # the pending event descriptor.  Physical crack and MPZ distances remain
    # unchanged until the checked outer energy-gate transaction.
    if geometry_after[1:] != geometry_before[1:]:
        raise RuntimeError("first-passage locator advanced physical geometry before energy gate")
    _sum(plastic, dict(final.get("plastic", {})))
    final.update({
        "coupled_hazard_cycles_consumed": committed_cycles,
        "coupled_hazard_partial_return": False,
        "coupled_hazard_event_localized": True,
        "coupled_hazard_first_passage_locator": True,
        "coupled_hazard_locator_model_id": MODEL_ID,
        "coupled_hazard_locator_bracket_low": low,
        "coupled_hazard_locator_bracket_high": high,
        "coupled_hazard_locator_iterations": iterations,
        "coupled_hazard_locator_trial_evaluations": evaluations,
        "plastic": plastic,
    })
    return final


__all__ = ["MODEL_ID", "localize_first_passage"]
