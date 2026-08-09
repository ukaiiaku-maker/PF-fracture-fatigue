"""Production high-cycle state machine for v10.2.30 fatigue.

The constitutive trajectory is advanced by exactly three admissible operators:

* an exact phase-resolved one-cycle Poincare map for transient evolution;
* a validated projective map for a slowly drifting active state;
* exact first-passage propagation after a periodic state is certified.

A rejected periodic or projective solve never sends the calculation back to a
sub-cycle adaptive marcher. It commits a bounded burst of exact physical cycles
and retries acceleration from the evolved state. The sub-cycle marcher is used
only inside the first-passage guard where an event must be localized within a
cycle. No Paris law, fitted growth law, extra threshold draw, or athermal fracture
criterion is introduced.
"""
from __future__ import annotations

import math
import os
import time
from typing import Any

from . import persistent_site_forward_robust_v10230 as _transient
from .persistent_site_high_cycle_propagation_v10230 import (
    projective_config,
    propagate_projective_cycles,
    propagate_stationary_cycles,
)
from .persistent_site_high_cycle_state_v10230 import (
    capture_ledgers,
    capture_stochastic_state,
    geometry_signature,
    ledger_delta,
    residual_metrics,
)
from .persistent_site_periodic_solver_v10230 import solve_periodic_state
from .persistent_site_poincare_v10230 import PoincareResult, one_cycle_map
from .persistent_site_first_passage_locator_v10230 import localize_first_passage


MODEL_ID = "v10.2.30_production_high_cycle_state_machine_v2"


def _env_float(name: str, default: float, minimum: float = 0.0) -> float:
    try:
        value = float(os.environ.get(name, default))
    except (TypeError, ValueError):
        value = float(default)
    if not math.isfinite(value):
        value = float(default)
    return max(value, minimum)


def _env_int(name: str, default: int, minimum: int = 1) -> int:
    try:
        value = int(os.environ.get(name, default))
    except (TypeError, ValueError):
        value = int(default)
    return max(value, minimum)


def high_cycle_config() -> dict[str, float | int]:
    return {
        "stationary_relative_tolerance": _env_float(
            "V10230_HIGH_CYCLE_STATIONARY_REL_TOL", 1.0e-7, 1.0e-14
        ),
        "stationary_diagnostic_tolerance": _env_float(
            "V10230_HIGH_CYCLE_STATIONARY_DIAGNOSTIC_TOL", 1.0e-5, 1.0e-14
        ),
        "stationary_admission_distance": _env_float(
            "V10230_HIGH_CYCLE_STATIONARY_ADMISSION_DISTANCE", 1.0e-6, 1.0e-14
        ),
        "maximum_mode_operations": _env_int(
            "V10230_HIGH_CYCLE_MAX_MODE_OPERATIONS", 256, 1
        ),
        "heartbeat_operations": _env_int(
            "V10230_HIGH_CYCLE_HEARTBEAT_OPERATIONS", 4, 1
        ),
        "exact_burst_initial_cycles": _env_int(
            "V10230_HIGH_CYCLE_EXACT_BURST_INITIAL", 8, 1
        ),
        "exact_burst_maximum_cycles": _env_int(
            "V10230_HIGH_CYCLE_EXACT_BURST_MAX", 256, 1
        ),
        "exact_burst_growth_factor": _env_float(
            "V10230_HIGH_CYCLE_EXACT_BURST_GROWTH", 2.0, 1.0
        ),
        "periodic_retry_initial_cycles": _env_float(
            "V10230_HIGH_CYCLE_PERIODIC_RETRY_INITIAL", 32.0, 1.0
        ),
        "periodic_retry_maximum_cycles": _env_float(
            "V10230_HIGH_CYCLE_PERIODIC_RETRY_MAX", 4096.0, 1.0
        ),
        "projective_retry_initial_cycles": _env_float(
            "V10230_HIGH_CYCLE_PROJECTIVE_RETRY_INITIAL", 8.0, 1.0
        ),
        "projective_retry_maximum_cycles": _env_float(
            "V10230_HIGH_CYCLE_PROJECTIVE_RETRY_MAX", 4096.0, 1.0
        ),
        "projective_growth_factor": _env_float(
            "V10230_PROJECTIVE_GROWTH_FACTOR", 8.0, 1.0
        ),
        "maximum_projective_cycles": _env_float(
            "V10230_PROJECTIVE_MAX_CYCLES", 1.0e9, 1.0
        ),
        "event_guard_cycles": _env_float(
            "V10230_HIGH_CYCLE_EVENT_GUARD_CYCLES", 2.0, 1.0
        ),
    }


def _print_heartbeat(
    consumed: float,
    requested: float,
    operations: int,
    modes: list[dict[str, Any]],
    cache: dict[str, Any],
    config: dict[str, float | int],
) -> None:
    if operations % int(config["heartbeat_operations"]) != 0 or not modes:
        return
    last = modes[-1]
    print(
        "  v10.2.30 high-cycle heartbeat: "
        f"consumed={consumed:.9g}/{requested:.9g} "
        f"mode={last.get('mode', 'unknown')} operations={operations} "
        f"next_exact={cache.get('exact_burst_next_cycles', 0)} "
        f"projective_retry={float(cache.get('projective_retry_cycles', 0.0)):.9g} "
        f"detail={last.get('failure_reason', last.get('projective_failure', 'none'))}",
        flush=True,
    )


def _sum_numeric(target: dict[str, float], source: dict[str, Any]) -> None:
    for key, value in source.items():
        if isinstance(value, (int, float)):
            target[key] = target.get(key, 0.0) + float(value)


def _initial_projective_cycles() -> float:
    config = projective_config()
    return max(
        float(config["minimum_project_cycles"]),
        float(config["initial_factor"]) * float(config["burst_cycles"]),
    )


def _cache(engine) -> dict[str, Any]:
    signature = geometry_signature(engine)
    config = high_cycle_config()
    cache = getattr(engine, "_v10230_high_cycle_cache", None)
    if not isinstance(cache, dict) or cache.get("geometry_signature") != signature:
        cache = {
            "geometry_signature": signature,
            "projective_next_cycles": _initial_projective_cycles(),
            "projective_attempts": 0,
            "projective_failure_count": 0,
            "projective_retry_cycles": 0.0,
            "exact_cycles_since_projective": math.inf,
            "periodic_attempts": 0,
            "periodic_failure_count": 0,
            "periodic_retry_cycles": 0.0,
            "exact_cycles_since_periodic": math.inf,
            "last_periodic_converged": False,
            "exact_burst_next_cycles": int(config["exact_burst_initial_cycles"]),
        }
        engine._v10230_high_cycle_cache = cache
    return cache


def invalidate_high_cycle_cache(engine, reason: str) -> None:
    engine._v10230_high_cycle_cache = {}
    cache = _cache(engine)
    cache["invalidated_reason"] = str(reason)


def _projective_with_requested_scale(
    engine,
    controller,
    waveform,
    temperature_K: float,
    cycles_requested: float,
    requested_project_cycles: float,
):
    burst = max(int(projective_config()["burst_cycles"]), 1)
    factor = max(float(requested_project_cycles) / float(burst), 1.0)
    previous = os.environ.get("V10230_PROJECTIVE_INITIAL_FACTOR")
    os.environ["V10230_PROJECTIVE_INITIAL_FACTOR"] = f"{factor:.17g}"
    try:
        return propagate_projective_cycles(
            engine, controller, waveform, temperature_K, cycles_requested
        )
    finally:
        if previous is None:
            os.environ.pop("V10230_PROJECTIVE_INITIAL_FACTOR", None)
        else:
            os.environ["V10230_PROJECTIVE_INITIAL_FACTOR"] = previous


def _update_hazard(engine, action: float) -> float:
    added = max(float(action), 0.0)
    threshold = max(float(getattr(engine, "hazard_threshold_action", 1.0)), 1.0e-300)
    current = max(float(getattr(engine, "hazard_action_current", 0.0)), 0.0)
    new_action = min(current + added, threshold)
    engine.hazard_action_current = new_action
    engine.B = min(new_action / threshold, 1.0)
    engine.hazard_last_progress_rate_s = 0.0
    return (new_action - current) / threshold


def _adopt_cycle_engine(target, source, cache: dict[str, Any]) -> None:
    """Adopt a private exact-cycle endpoint without replacing target identity."""
    engine_id = getattr(target, "_engine_id", None)
    source_state = source.__dict__
    source.__dict__ = {}
    target.__dict__.clear()
    target.__dict__.update(source_state)
    if engine_id is not None:
        target._engine_id = engine_id
    if hasattr(target, "_energy_gate_provisional"):
        target._energy_gate_provisional = False
    if hasattr(target, "_energy_gate_pending"):
        target._energy_gate_pending = None
    target._v10230_high_cycle_cache = cache


def _commit_exact_cycle(
    engine,
    cycle: PoincareResult,
    cache: dict[str, Any],
) -> float:
    geometry_before = geometry_signature(engine)
    stochastic_before = capture_stochastic_state(engine)
    _adopt_cycle_engine(engine, cycle.engine, cache)
    if geometry_signature(engine) != geometry_before:
        raise RuntimeError("exact Poincare-cycle commit changed crack geometry")
    if capture_stochastic_state(engine) != stochastic_before:
        raise RuntimeError("exact Poincare-cycle commit changed threshold or RNG state")
    return _update_hazard(engine, cycle.hazard_action_per_cycle)


def _cycles_to_event(engine, action_per_cycle: float) -> float:
    action = max(float(action_per_cycle), 0.0)
    if action <= 0.0:
        return math.inf
    threshold = max(float(getattr(engine, "hazard_threshold_action", 1.0)), 1.0e-300)
    current = max(float(getattr(engine, "hazard_action_current", 0.0)), 0.0)
    return max(threshold - current, 0.0) / action


def _advance_exact_cycle_burst(
    engine,
    controller,
    waveform,
    temperature_K: float,
    cycles_available: float,
    burst_cycles: int,
    cache: dict[str, Any],
    *,
    first_cycle: PoincareResult | None = None,
) -> dict[str, Any]:
    maximum = min(max(int(math.floor(cycles_available)), 0), max(int(burst_cycles), 0))
    guard = float(high_cycle_config()["event_guard_cycles"])
    consumed = 0.0
    dB = 0.0
    dH = 0.0
    plastic: dict[str, float] = {}
    event_near = False
    last_cycle: PoincareResult | None = None

    for index in range(maximum):
        cycle = (
            first_cycle
            if index == 0 and first_cycle is not None
            else one_cycle_map(engine, controller, waveform, temperature_K)
        )
        if _cycles_to_event(engine, cycle.hazard_action_per_cycle) <= guard:
            event_near = True
            last_cycle = cycle
            break
        normalized = _commit_exact_cycle(engine, cycle, cache)
        consumed += 1.0
        dB += normalized
        dH += max(float(cycle.hazard_action_per_cycle), 0.0)
        _sum_numeric(plastic, cycle.plastic_totals)
        last_cycle = cycle

    return {
        "cycles": consumed,
        "dB": dB,
        "dH": dH,
        "plastic": plastic,
        "event_near": event_near,
        "last_cycle": last_cycle,
    }


def _accumulate_transient(
    result: dict[str, Any],
    plastic: dict[str, float],
    advance: dict[str, float],
) -> tuple[float, float, float, float, float, float, int, bool]:
    cycles = max(float(result.get("coupled_hazard_cycles_consumed", 0.0)), 0.0)
    dB = max(float(result.get("dB", 0.0)), 0.0)
    dH = max(float(result.get("physical_hazard_action_step", 0.0)), 0.0)
    da = max(float(result.get("da", 0.0)), 0.0)
    packet_mean = max(float(result.get("packet_mean", 0.0)), 0.0)
    packet_variance = max(float(result.get("packet_variance_m2", 0.0)), 0.0)
    microsteps = int(result.get("microsteps", 0))
    _sum_numeric(plastic, dict(result.get("plastic", {})))
    _sum_numeric(advance, dict(result.get("advance", {})))
    return (
        cycles,
        dB,
        dH,
        da,
        packet_mean,
        packet_variance,
        microsteps,
        bool(result.get("fired", False)),
    )


def integrate_state_coupled_waveform(
    engine,
    controller,
    waveform,
    temperature_K: float,
    cycles_requested: float,
) -> dict[str, Any]:
    requested = max(float(cycles_requested), 0.0)
    period = max(float(waveform.period_s), 0.0)
    config = high_cycle_config()
    if requested <= 0.0 or period <= 0.0:
        return _transient.integrate_state_coupled_waveform(
            engine, controller, waveform, temperature_K, requested
        )
    if getattr(engine, "_energy_gate_pending", None) is not None:
        raise RuntimeError("high-cycle propagation cannot start with a pending crack event")

    cache = _cache(engine)
    initial_geometry = geometry_signature(engine)
    initial_ledgers = capture_ledgers(engine)
    initial_stochastic = capture_stochastic_state(engine)
    consumed = 0.0
    dB_total = 0.0
    dH_total = 0.0
    da_total = 0.0
    packet_mean_total = 0.0
    packet_variance_total = 0.0
    microsteps = 0
    plastic: dict[str, float] = {}
    advance: dict[str, float] = {}
    fired = False
    work_budget_exhausted = False
    modes: list[dict[str, Any]] = []
    operations = 0
    start_wall = time.monotonic()
    last_result: dict[str, Any] = {}

    while consumed < requested and not fired:
        operations += 1
        if operations > int(config["maximum_mode_operations"]):
            work_budget_exhausted = True
            break
        remaining = requested - consumed
        cycle = one_cycle_map(engine, controller, waveform, temperature_K)
        current_residual = residual_metrics(
            cycle.state_start,
            cycle.state_end,
            relative_tolerance=float(config["stationary_relative_tolerance"]),
            diagnostic_tolerance=float(config["stationary_diagnostic_tolerance"]),
        )

        periodic = None
        periodic_due = bool(
            int(cache.get("periodic_attempts", 0)) == 0
            or current_residual.converged
            or float(cache.get("exact_cycles_since_periodic", 0.0))
            >= float(cache.get("periodic_retry_cycles", 0.0))
        )
        if periodic_due:
            periodic = solve_periodic_state(
                engine,
                controller,
                waveform,
                temperature_K,
                initial_state=cycle.state_start,
            )
            cache["periodic_attempts"] = int(cache.get("periodic_attempts", 0)) + 1
            cache["last_periodic_converged"] = bool(periodic.converged)
            cache["exact_cycles_since_periodic"] = 0.0
            if periodic.converged:
                cache["periodic_failure_count"] = 0
                cache["periodic_retry_cycles"] = float(
                    config["periodic_retry_initial_cycles"]
                )
            else:
                failures = int(cache.get("periodic_failure_count", 0)) + 1
                cache["periodic_failure_count"] = failures
                cache["periodic_retry_cycles"] = min(
                    float(config["periodic_retry_maximum_cycles"]),
                    float(config["periodic_retry_initial_cycles"])
                    * (2.0 ** min(failures - 1, 20)),
                )
            modes.append(
                {
                    "mode": "periodic_search",
                    "cycles": 0.0,
                    "converged": periodic.converged,
                    "iterations": periodic.iterations,
                    "map_evaluations": periodic.map_evaluations,
                    "current_map_residual": current_residual.maximum_relative,
                    "fixed_point_residual": periodic.residual.maximum_relative,
                    "distance_from_current": periodic.distance_from_initial,
                    "failure_reason": periodic.failure_reason,
                    "next_retry_cycles": cache["periodic_retry_cycles"],
                }
            )

        stationary_admissible = bool(
            periodic is not None
            and current_residual.converged
            and periodic.converged
            and periodic.distance_from_initial
            <= float(config["stationary_admission_distance"])
        )
        if stationary_admissible:
            stationary = propagate_stationary_cycles(engine, waveform, cycle, remaining)
            consumed += stationary.cycles_consumed
            dB_total += stationary.normalized_clock_added
            dH_total += stationary.hazard_action_added
            modes.append(
                {
                    "mode": "stationary_tail",
                    "cycles": stationary.cycles_consumed,
                    "event_within_guard": stationary.event_within_guard,
                    "state_residual": current_residual.maximum_relative,
                    "fixed_point_distance": periodic.distance_from_initial,
                    "hazard_action_per_cycle": cycle.hazard_action_per_cycle,
                }
            )
            if consumed >= requested:
                break
            _print_heartbeat(consumed, requested, operations, modes, cache, config)
            remaining = requested - consumed
            local = _transient.integrate_state_coupled_waveform(
                engine,
                controller,
                waveform,
                temperature_K,
                min(remaining, float(config["event_guard_cycles"])),
            )
            (
                local_cycles,
                local_dB,
                local_dH,
                local_da,
                local_packet_mean,
                local_packet_variance,
                local_microsteps,
                fired,
            ) = _accumulate_transient(local, plastic, advance)
            consumed += local_cycles
            dB_total += local_dB
            dH_total += local_dH
            da_total += local_da
            packet_mean_total += local_packet_mean
            packet_variance_total += local_packet_variance
            microsteps += local_microsteps
            last_result = dict(local)
            modes.append(
                {
                    "mode": "event_guard_transient",
                    "cycles": local_cycles,
                    "fired": fired,
                    "partial": bool(local.get("coupled_hazard_partial_return", False)),
                }
            )
            if fired:
                invalidate_high_cycle_cache(engine, "first_passage_event")
                break
            if local_cycles <= 0.0:
                work_budget_exhausted = True
                break
            continue

        projective_due = bool(
            int(cache.get("projective_attempts", 0)) == 0
            or float(cache.get("exact_cycles_since_projective", 0.0))
            >= float(cache.get("projective_retry_cycles", 0.0))
        )
        projected = None
        if projective_due:
            requested_project = min(
                float(cache.get("projective_next_cycles", 1.0)),
                float(config["maximum_projective_cycles"]),
                remaining,
            )
            projected = _projective_with_requested_scale(
                engine,
                controller,
                waveform,
                temperature_K,
                remaining,
                requested_project,
            )
            cache["projective_attempts"] = int(cache.get("projective_attempts", 0)) + 1
            cache["exact_cycles_since_projective"] = 0.0
            if projected.accepted and projected.cycles_consumed > 0.0:
                consumed += projected.cycles_consumed
                dB_total += projected.normalized_clock_added
                dH_total += projected.hazard_action_added
                cache["projective_failure_count"] = 0
                cache["projective_retry_cycles"] = float(
                    config["projective_retry_initial_cycles"]
                )
                cache["projective_next_cycles"] = min(
                    max(
                        projected.projected_cycles
                        * float(config["projective_growth_factor"]),
                        requested_project,
                    ),
                    float(config["maximum_projective_cycles"]),
                )
                modes.append(
                    {
                        "mode": "slow_projective",
                        "cycles": projected.cycles_consumed,
                        "burst_cycles": projected.burst_cycles,
                        "projected_cycles": projected.projected_cycles,
                        "drift_error": projected.drift_relative_error,
                        "hazard_error": projected.hazard_relative_error,
                        "attempts": projected.attempts,
                    }
                )
                _print_heartbeat(consumed, requested, operations, modes, cache, config)
                continue

            failures = int(cache.get("projective_failure_count", 0)) + 1
            cache["projective_failure_count"] = failures
            cache["projective_retry_cycles"] = min(
                float(config["projective_retry_maximum_cycles"]),
                float(config["projective_retry_initial_cycles"])
                * (2.0 ** min(failures - 1, 20)),
            )
            cache["projective_next_cycles"] = max(
                float(projective_config()["minimum_project_cycles"]),
                0.5 * requested_project,
            )
            modes.append(
                {
                    "mode": "projective_rejected",
                    "cycles": 0.0,
                    "failure_reason": projected.failure_reason,
                    "drift_error": projected.drift_relative_error,
                    "hazard_error": projected.hazard_relative_error,
                    "attempts": projected.attempts,
                    "next_retry_cycles": cache["projective_retry_cycles"],
                }
            )

        exact_target = min(
            int(cache.get("exact_burst_next_cycles", 1)),
            int(config["exact_burst_maximum_cycles"]),
            max(int(math.floor(remaining)), 0),
        )
        exact = _advance_exact_cycle_burst(
            engine,
            controller,
            waveform,
            temperature_K,
            remaining,
            exact_target,
            cache,
            first_cycle=cycle,
        )
        exact_cycles = float(exact["cycles"])
        if exact_cycles > 0.0:
            consumed += exact_cycles
            dB_total += float(exact["dB"])
            dH_total += float(exact["dH"])
            _sum_numeric(plastic, dict(exact["plastic"]))
            cache["exact_cycles_since_periodic"] = float(
                cache.get("exact_cycles_since_periodic", 0.0)
            ) + exact_cycles
            cache["exact_cycles_since_projective"] = float(
                cache.get("exact_cycles_since_projective", 0.0)
            ) + exact_cycles
            cache["exact_burst_next_cycles"] = min(
                int(config["exact_burst_maximum_cycles"]),
                max(
                    int(config["exact_burst_initial_cycles"]),
                    int(
                        math.ceil(
                            exact_target * float(config["exact_burst_growth_factor"])
                        )
                    ),
                ),
            )
            modes.append(
                {
                    "mode": "exact_cycle_burst",
                    "cycles": exact_cycles,
                    "requested_burst_cycles": exact_target,
                    "event_within_guard": bool(exact["event_near"]),
                    "projective_failure": (
                        projected.failure_reason if projected is not None else None
                    ),
                    "next_burst_cycles": cache["exact_burst_next_cycles"],
                }
            )
            _print_heartbeat(consumed, requested, operations, modes, cache, config)
            continue

        local_request = min(remaining, float(config["event_guard_cycles"]))
        local = localize_first_passage(
            engine, controller, waveform, temperature_K, local_request
        )
        (
            local_cycles,
            local_dB,
            local_dH,
            local_da,
            local_packet_mean,
            local_packet_variance,
            local_microsteps,
            fired,
        ) = _accumulate_transient(local, plastic, advance)
        consumed += local_cycles
        dB_total += local_dB
        dH_total += local_dH
        da_total += local_da
        packet_mean_total += local_packet_mean
        packet_variance_total += local_packet_variance
        microsteps += local_microsteps
        last_result = dict(local)
        modes.append(
            {
                "mode": "first_passage_cycle_locator",
                "cycles": local_cycles,
                "fired": fired,
                "partial": bool(local.get("coupled_hazard_partial_return", False)),
                "bracket_low": local.get("coupled_hazard_locator_bracket_low"),
                "bracket_high": local.get("coupled_hazard_locator_bracket_high"),
                "trial_evaluations": local.get("coupled_hazard_locator_trial_evaluations"),
            }
        )
        if fired:
            invalidate_high_cycle_cache(engine, "first_passage_event")
            break
        if local_cycles <= 0.0:
            work_budget_exhausted = True
            break
        _print_heartbeat(consumed, requested, operations, modes, cache, config)

    if fired:
        final_lambda = max(float(last_result.get("lambda_c", 0.0)), 0.0)
        final_sigma = max(float(last_result.get("sigma_tip", 0.0)), 0.0)
    else:
        final_cycle = one_cycle_map(engine, controller, waveform, temperature_K)
        final_lambda = float(final_cycle.hazard_action_per_cycle / period)
        final_sigma = float(
            final_cycle.state_start.diagnostics.get("sigma_tip_Pa", 0.0)
        )
    final_ledgers = capture_ledgers(engine)
    result = dict(last_result)
    result.update(
        {
            "fired": bool(fired),
            "n_fire": 1 if fired else 0,
            "v_crack": da_total / (consumed * period) if consumed > 0.0 else 0.0,
            "dB": float(dB_total),
            "physical_hazard_action_step": float(dH_total),
            "da": float(da_total),
            "dt_consumed": float(consumed * period),
            "dt_unused": max((requested - consumed) * period, 0.0),
            "packet_mean": float(packet_mean_total),
            "packet_variance_m2": float(packet_variance_total),
            "lambda_c": final_lambda,
            "lambda_c_raw": final_lambda,
            "sigma_tip": final_sigma,
            "plastic": plastic,
            "advance": advance,
            "microsteps": int(microsteps),
            "coupled_hazard_model_id": MODEL_ID,
            "coupled_hazard_high_cycle_engine": True,
            "coupled_hazard_phase_resolved_poincare_map": True,
            "coupled_hazard_periodic_solver": True,
            "coupled_hazard_stationary_first_passage": True,
            "coupled_hazard_slow_projective": True,
            "coupled_hazard_exact_cycle_transient": True,
            "coupled_hazard_subcycle_marcher_event_only": True,
            "coupled_hazard_event_restart": True,
            "coupled_hazard_cycles_requested": float(requested),
            "coupled_hazard_cycles_consumed": float(consumed),
            "coupled_hazard_work_budget_exhausted": bool(work_budget_exhausted),
            "coupled_hazard_partial_return": bool(
                not fired and consumed + 1.0e-12 < requested
            ),
            "coupled_hazard_event_localized": bool(fired and consumed < requested),
            "coupled_hazard_mode_operations": int(operations),
            "coupled_hazard_modes": modes,
            "coupled_hazard_wall_seconds": float(time.monotonic() - start_wall),
            "coupled_hazard_geometry_preserved_before_event": bool(
                fired or geometry_signature(engine) == initial_geometry
            ),
            "coupled_hazard_ledger_delta": ledger_delta(
                initial_ledgers, final_ledgers
            ),
            "coupled_hazard_stochastic_threshold_preserved_until_event": bool(
                fired
                or capture_stochastic_state(engine)["hazard_threshold_action"]
                == initial_stochastic["hazard_threshold_action"]
            ),
            "coupled_hazard_config": dict(config),
            "coupled_hazard_cache": dict(cache),
        }
    )
    return result


__all__ = [
    "MODEL_ID",
    "high_cycle_config",
    "integrate_state_coupled_waveform",
    "invalidate_high_cycle_cache",
]
