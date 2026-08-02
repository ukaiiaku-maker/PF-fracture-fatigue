"""Bounded forward state-coupled VHCF integration for v10.2.30.

This module changes numerical orchestration only.  The existing persistent-site
constitutive update, Arrhenius cleavage rate, stochastic first-passage threshold,
and transactional crack-event machinery remain authoritative.

The inherited v10.2.29 integrator recursively bisected an entire requested block
and returned only after the complete recursion finished.  A VHCF-sized proposal
could therefore create an effectively unbounded depth-first calculation with no
partial progress.  This implementation instead marches monotonically forward.
Each proposed segment is evaluated by one full step and two complete half steps;
the two-half-step endpoint is the accepted state.  Hard work budgets return honest
partial progress rather than silently running for hours.
"""
from __future__ import annotations

import copy
import math
import os
import time
from typing import Any

from . import persistent_site_coupled_hazard_v10229 as _legacy
from . import stochastic_avalanche_tip as _avalanche_tip


MODEL_ID = "v10.2.30_bounded_forward_state_coupled_hazard_v1"


def _finite(value: Any, default: float = 0.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return float(default)
    return number if math.isfinite(number) else float(default)


def _positive(value: Any, default: float = 0.0) -> float:
    return max(_finite(value, default), 0.0)


def _env_float(name: str, default: float, minimum: float = 0.0) -> float:
    value = _finite(os.environ.get(name, default), default)
    return max(value, minimum)


def _env_int(name: str, default: int, minimum: int = 1) -> int:
    try:
        value = int(os.environ.get(name, default))
    except (TypeError, ValueError):
        value = int(default)
    return max(value, minimum)


def forward_config(controller) -> dict[str, float | int]:
    minimum = max(
        _positive(getattr(controller.cfg, "min_block_cycles", 1.0e-6), 1.0e-6),
        _env_float("V10230_FORWARD_MIN_CYCLES", 1.0e-6, 1.0e-18),
    )
    return {
        "minimum_cycles": minimum,
        "initial_cycles": _env_float("V10230_FORWARD_INITIAL_CYCLES", 1.0e-2, minimum),
        "growth_factor": _env_float("V10230_FORWARD_GROWTH_FACTOR", 2.0, 1.0),
        "shrink_factor": min(
            _env_float("V10230_FORWARD_SHRINK_FACTOR", 0.5, 1.0e-6), 0.95
        ),
        "clock_relative_tol": _env_float(
            "V10230_FORWARD_CLOCK_REL_TOL", 1.0e-3, 1.0e-12
        ),
        "shield_relative_tol": _env_float(
            "V10230_FORWARD_SHIELD_REL_TOL", 1.0e-3, 1.0e-12
        ),
        "sigma_relative_tol": _env_float(
            "V10230_FORWARD_SIGMA_REL_TOL", 1.0e-3, 1.0e-12
        ),
        "radius_relative_tol": _env_float(
            "V10230_FORWARD_RADIUS_REL_TOL", 1.0e-3, 1.0e-12
        ),
        "log_lambda_tol_decades": _env_float(
            "V10230_FORWARD_LOG_LAMBDA_TOL_DECADES", 0.01, 1.0e-12
        ),
        "event_localization_cycles": _env_float(
            "V10230_FORWARD_EVENT_LOCALIZATION_CYCLES", 1.0e-6, minimum
        ),
        "maximum_accepted_segments": _env_int(
            "V10230_FORWARD_MAX_ACCEPTED_SEGMENTS", 256, 1
        ),
        "maximum_trial_integrations": _env_int(
            "V10230_FORWARD_MAX_TRIAL_INTEGRATIONS", 1024, 3
        ),
        "heartbeat_segments": _env_int(
            "V10230_FORWARD_HEARTBEAT_SEGMENTS", 16, 1
        ),
    }


def _sum_numeric(target: dict[str, float], source: dict[str, Any]) -> None:
    for key, value in source.items():
        if isinstance(value, (int, float)):
            target[key] = target.get(key, 0.0) + float(value)


def _combine_results(first: dict[str, Any], second: dict[str, Any] | None) -> dict[str, Any]:
    if second is None:
        return dict(first)
    result = dict(second)
    for key in (
        "dB",
        "physical_hazard_action_step",
        "da",
        "dt_consumed",
        "packet_mean",
        "packet_variance_m2",
        "microsteps",
    ):
        result[key] = _finite(first.get(key, 0.0)) + _finite(second.get(key, 0.0))
    result["fired"] = bool(first.get("fired", False) or second.get("fired", False))
    result["n_fire"] = int(bool(result["fired"]))
    result["dt_unused"] = 0.0
    plastic: dict[str, float] = {}
    advance: dict[str, float] = {}
    _sum_numeric(plastic, dict(first.get("plastic", {})))
    _sum_numeric(plastic, dict(second.get("plastic", {})))
    _sum_numeric(advance, dict(first.get("advance", {})))
    _sum_numeric(advance, dict(second.get("advance", {})))
    result["plastic"] = plastic
    result["advance"] = advance
    return result


def _endpoint(engine, controller, waveform, temperature_K: float) -> dict[str, float]:
    stats = _legacy._phase_statistics(engine, controller, waveform, temperature_K)
    return {
        "B": _finite(getattr(engine, "B", 0.0)),
        "lambda_per_s": _positive(stats["lambda_avg_s"]),
        "sigma_Pa": _positive(stats["sigma_avg_Pa"]),
        "K_shield_Pa_sqrt_m": _finite(stats["K_shield_Pa_sqrt_m"]),
        "r_eff_m": _positive(stats["r_eff_m"], 1.0e-30),
    }


def _log_difference(a: float, b: float) -> float:
    high = max(_positive(a), _positive(b))
    if high <= 0.0:
        return 0.0
    low = max(min(_positive(a), _positive(b)), 1.0e-300)
    return abs(math.log10(high / low))


def _relative_difference(a: float, b: float, floor: float = 1.0) -> float:
    return abs(float(a) - float(b)) / max(abs(float(a)), abs(float(b)), floor)


def _error_metrics(
    full_endpoint: dict[str, float],
    half_endpoint: dict[str, float],
    full_result: dict[str, Any],
    half_result: dict[str, Any],
    waveform,
    config: dict[str, float | int],
) -> dict[str, float]:
    applied = max(abs(float(waveform.Kmax)), 1.0)
    effective = max(
        abs(applied - half_endpoint["K_shield_Pa_sqrt_m"]),
        1.0e-6 * applied,
        1.0,
    )
    clock_scale = max(
        abs(_finite(full_result.get("dB", 0.0))),
        abs(_finite(half_result.get("dB", 0.0))),
        1.0e-15,
    )
    metrics = {
        "clock_relative_error": abs(
            _finite(full_result.get("dB", 0.0))
            - _finite(half_result.get("dB", 0.0))
        )
        / clock_scale,
        "shield_relative_error": abs(
            full_endpoint["K_shield_Pa_sqrt_m"]
            - half_endpoint["K_shield_Pa_sqrt_m"]
        )
        / effective,
        "sigma_relative_error": _relative_difference(
            full_endpoint["sigma_Pa"], half_endpoint["sigma_Pa"], 1.0
        ),
        "radius_relative_error": _relative_difference(
            full_endpoint["r_eff_m"], half_endpoint["r_eff_m"], 1.0e-30
        ),
        "log_lambda_error_decades": _log_difference(
            full_endpoint["lambda_per_s"], half_endpoint["lambda_per_s"]
        ),
    }
    ratios = {
        "clock": metrics["clock_relative_error"]
        / float(config["clock_relative_tol"]),
        "shield": metrics["shield_relative_error"]
        / float(config["shield_relative_tol"]),
        "sigma": metrics["sigma_relative_error"]
        / float(config["sigma_relative_tol"]),
        "radius": metrics["radius_relative_error"]
        / float(config["radius_relative_tol"]),
        "lambda": metrics["log_lambda_error_decades"]
        / float(config["log_lambda_tol_decades"]),
    }
    metrics["maximum_error_ratio"] = max(ratios.values(), default=0.0)
    metrics["limiting_error"] = max(ratios, key=ratios.get) if ratios else "none"
    return metrics


def _constant_segment(engine, waveform, temperature_K, cycles, stats):
    return _legacy._commit_constant_segment(
        engine,
        waveform,
        temperature_K,
        cycles,
        stats["sigma_avg_Pa"],
        stats["lambda_avg_s"],
    )


def _evaluate_segment(engine, controller, waveform, temperature_K: float, cycles: float):
    start_stats = _legacy._phase_statistics(
        engine, controller, waveform, temperature_K
    )

    full = copy.deepcopy(engine)
    full_result = _constant_segment(full, waveform, temperature_K, cycles, start_stats)
    full_endpoint = _endpoint(full, controller, waveform, temperature_K)

    half = copy.deepcopy(engine)
    half_cycles = 0.5 * cycles
    first_result = _constant_segment(
        half, waveform, temperature_K, half_cycles, start_stats
    )
    if bool(first_result.get("fired", False)):
        half_result = dict(first_result)
        half_endpoint = _endpoint(half, controller, waveform, temperature_K)
        return full, full_result, full_endpoint, half, half_result, half_endpoint, 2

    mid_stats = _legacy._phase_statistics(
        half, controller, waveform, temperature_K
    )
    second_result = _constant_segment(
        half,
        waveform,
        temperature_K,
        cycles - half_cycles,
        mid_stats,
    )
    half_result = _combine_results(first_result, second_result)
    half_endpoint = _endpoint(half, controller, waveform, temperature_K)
    return full, full_result, full_endpoint, half, half_result, half_endpoint, 3


def _adopt_state(target, source, *, fired: bool) -> None:
    engine_id = getattr(target, "_engine_id", None)
    had_provisional = hasattr(target, "_energy_gate_provisional")
    source_state = source.__dict__
    source.__dict__ = {}
    target.__dict__.clear()
    target.__dict__.update(source_state)
    if engine_id is not None:
        target._engine_id = engine_id
    if had_provisional:
        target._energy_gate_provisional = False
    if fired:
        pending = getattr(target, "_energy_gate_pending", None)
        descriptor = pending.get("descriptor") if isinstance(pending, dict) else None
        if isinstance(descriptor, dict):
            descriptor["energy_gate_engine_id"] = int(engine_id)
            if not any(item is descriptor for item in _avalanche_tip._PENDING_GEOMETRY_EVENTS):
                _avalanche_tip._PENDING_GEOMETRY_EVENTS.append(descriptor)


def integrate_state_coupled_waveform(
    engine,
    controller,
    waveform,
    temperature_K: float,
    cycles_requested: float,
) -> dict[str, Any]:
    """Advance monotonically with bounded work and honest partial return."""
    requested = max(float(cycles_requested), 0.0)
    frequency = max(float(waveform.frequency_Hz), 0.0)
    period = float(waveform.period_s)
    config = forward_config(controller)
    if requested <= 0.0 or frequency <= 0.0 or period <= 0.0:
        return {
            "fired": False,
            "n_fire": 0,
            "dB": 0.0,
            "physical_hazard_action_step": 0.0,
            "da": 0.0,
            "dt_consumed": 0.0,
            "dt_unused": requested * max(period, 0.0),
            "plastic": {},
            "advance": {},
            "microsteps": 0,
            "coupled_hazard_model_id": MODEL_ID,
            "coupled_hazard_work_budget_exhausted": False,
        }

    totals: dict[str, float] = {}
    wake_totals: dict[str, float] = {}
    consumed_cycles = 0.0
    dB_total = 0.0
    dH_total = 0.0
    da_total = 0.0
    packet_mean = 0.0
    packet_variance = 0.0
    microsteps = 0
    accepted_segments = 0
    rejected_segments = 0
    trial_integrations = 0
    segment_audit: list[dict[str, Any]] = []
    last_result: dict[str, Any] = {}
    fired = False
    work_budget_exhausted = False
    start_wall = time.monotonic()

    initial_endpoint = _endpoint(engine, controller, waveform, temperature_K)
    lambda_history = [initial_endpoint["lambda_per_s"]]
    sigma_history = [initial_endpoint["sigma_Pa"]]
    shield_history = [initial_endpoint["K_shield_Pa_sqrt_m"]]

    segment = min(requested, float(config["initial_cycles"]))
    segment = max(min(segment, requested), float(config["minimum_cycles"]))

    while consumed_cycles < requested and not fired:
        if (
            accepted_segments >= int(config["maximum_accepted_segments"])
            or trial_integrations + 3 > int(config["maximum_trial_integrations"])
        ):
            work_budget_exhausted = True
            break

        remaining = requested - consumed_cycles
        segment = min(segment, remaining)
        if segment <= 0.0:
            break

        (
            _full,
            full_result,
            full_endpoint,
            half,
            half_result,
            half_endpoint,
            evaluations,
        ) = _evaluate_segment(
            engine, controller, waveform, temperature_K, segment
        )
        trial_integrations += evaluations
        errors = _error_metrics(
            full_endpoint,
            half_endpoint,
            full_result,
            half_result,
            waveform,
            config,
        )
        trial_fired = bool(
            full_result.get("fired", False) or half_result.get("fired", False)
        )
        minimum_reached = segment <= float(config["minimum_cycles"])
        event_localized = (
            trial_fired
            and segment <= float(config["event_localization_cycles"])
        )
        accept = (
            (not trial_fired and errors["maximum_error_ratio"] <= 1.0)
            or event_localized
            or minimum_reached
        )

        if not accept:
            rejected_segments += 1
            segment *= float(config["shrink_factor"])
            segment = max(segment, float(config["minimum_cycles"]))
            continue

        _adopt_state(engine, half, fired=bool(half_result.get("fired", False)))
        actual_cycles = _positive(half_result.get("dt_consumed", 0.0)) * frequency
        consumed_cycles += actual_cycles
        dB_total += _positive(half_result.get("dB", 0.0))
        dH_total += _positive(
            half_result.get("physical_hazard_action_step", 0.0)
        )
        da_total += _positive(half_result.get("da", 0.0))
        packet_mean += _positive(half_result.get("packet_mean", 0.0))
        packet_variance += _positive(
            half_result.get("packet_variance_m2", 0.0)
        )
        microsteps += int(half_result.get("microsteps", 0))
        _sum_numeric(totals, dict(half_result.get("plastic", {})))
        _sum_numeric(wake_totals, dict(half_result.get("advance", {})))
        accepted_segments += 1
        fired = bool(half_result.get("fired", False))
        last_result = dict(half_result)

        lambda_history.append(half_endpoint["lambda_per_s"])
        sigma_history.append(half_endpoint["sigma_Pa"])
        shield_history.append(half_endpoint["K_shield_Pa_sqrt_m"])
        segment_audit.append(
            {
                "cycles_proposed": float(segment),
                "cycles_consumed": float(actual_cycles),
                "cumulative_cycles": float(consumed_cycles),
                "accepted": True,
                "fired": bool(fired),
                **errors,
            }
        )

        if (
            accepted_segments % int(config["heartbeat_segments"]) == 0
            and consumed_cycles < requested
        ):
            print(
                "  v10.2.30 forward VHCF heartbeat: "
                f"consumed={consumed_cycles:.9g}/{requested:.9g} cycles "
                f"accepted={accepted_segments} rejected={rejected_segments} "
                f"trials={trial_integrations} next={segment:.9g}",
                flush=True,
            )

        if fired or actual_cycles + 1.0e-15 < segment:
            break
        if errors["maximum_error_ratio"] <= 0.125:
            segment *= float(config["growth_factor"])

    final_endpoint = _endpoint(engine, controller, waveform, temperature_K)
    combined = dict(last_result)
    combined.update(
        {
            "fired": bool(fired),
            "n_fire": 1 if fired else 0,
            "v_crack": da_total / (consumed_cycles * period)
            if consumed_cycles > 0.0
            else 0.0,
            "dB": float(dB_total),
            "physical_hazard_action_step": float(dH_total),
            "da": float(da_total),
            "dt_consumed": float(consumed_cycles * period),
            "dt_unused": max((requested - consumed_cycles) * period, 0.0),
            "packet_mean": float(packet_mean),
            "packet_variance_m2": float(packet_variance),
            "lambda_c": float(final_endpoint["lambda_per_s"]),
            "lambda_c_raw": float(final_endpoint["lambda_per_s"]),
            "sigma_tip": float(final_endpoint["sigma_Pa"]),
            "plastic": totals,
            "advance": wake_totals,
            "microsteps": int(microsteps),
            "coupled_hazard_model_id": MODEL_ID,
            "coupled_hazard_phase_resolved": True,
            "coupled_hazard_frozen_within_outer_block": False,
            "coupled_hazard_forward_marcher": True,
            "coupled_hazard_recursive_bisection": False,
            "coupled_hazard_two_half_step_state_committed": True,
            "coupled_hazard_third_commit_integration": False,
            "coupled_hazard_accepted_segments": int(accepted_segments),
            "coupled_hazard_rejected_splits": int(rejected_segments),
            "coupled_hazard_trial_integrations": int(trial_integrations),
            "coupled_hazard_work_budget_exhausted": bool(work_budget_exhausted),
            "coupled_hazard_partial_return": bool(
                not fired and consumed_cycles + 1.0e-15 < requested
            ),
            "coupled_hazard_cycles_requested": float(requested),
            "coupled_hazard_cycles_consumed": float(consumed_cycles),
            "coupled_hazard_lambda_start_s": float(lambda_history[0]),
            "coupled_hazard_lambda_end_s": float(lambda_history[-1]),
            "coupled_hazard_lambda_start_per_s": float(lambda_history[0]),
            "coupled_hazard_lambda_end_per_s": float(lambda_history[-1]),
            "coupled_hazard_lambda_min_per_s": float(
                min(lambda_history, default=0.0)
            ),
            "coupled_hazard_lambda_max_per_s": float(
                max(lambda_history, default=0.0)
            ),
            "coupled_hazard_log_lambda_span_decades": float(
                _legacy._log_span(lambda_history)
            ),
            "coupled_hazard_sigma_start_Pa": float(sigma_history[0]),
            "coupled_hazard_sigma_end_Pa": float(sigma_history[-1]),
            "coupled_hazard_shield_start_Pa_sqrt_m": float(shield_history[0]),
            "coupled_hazard_shield_end_Pa_sqrt_m": float(shield_history[-1]),
            "coupled_hazard_event_localized": bool(
                fired and consumed_cycles + 1.0e-15 < requested
            ),
            "coupled_hazard_wall_seconds": float(time.monotonic() - start_wall),
            "coupled_hazard_config": dict(config),
            "coupled_hazard_segments": segment_audit,
        }
    )
    return combined


__all__ = [
    "MODEL_ID",
    "forward_config",
    "integrate_state_coupled_waveform",
]
