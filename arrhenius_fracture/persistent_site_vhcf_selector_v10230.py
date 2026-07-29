"""Performance-safe nonlinear VHCF block selection for v10.2.30.

The accepted cycle block is still determined by exact persistent-site trial commits
and the existing physical increment targets.  The numerical search no longer
starts by trial-integrating the full user horizon.  It starts from the existing
one-cycle tangent estimate, expands geometrically until it brackets the first
violated target, and then refines that bracket in log-cycle space.
"""
from __future__ import annotations

import math
import os
from typing import Any

from . import persistent_site_vhcf_selector_v10229 as _base


MODEL_ID = "v10.2.30_persistent_site_geometric_vhcf_block_v1"
attach_prediction_context = _base.attach_prediction_context
_block_trial = _base._block_trial
_targets = _base._targets
_ratio = _base._ratio


def _env_float(name: str, default: float, minimum: float) -> float:
    try:
        value = float(os.environ.get(name, default))
    except (TypeError, ValueError):
        value = float(default)
    if not math.isfinite(value):
        value = float(default)
    return max(value, minimum)


def _env_int(name: str, default: int, minimum: int) -> int:
    try:
        value = int(os.environ.get(name, default))
    except (TypeError, ValueError):
        value = int(default)
    return max(value, minimum)


def _search_config() -> dict[str, float | int]:
    return {
        "growth_factor": _env_float("V10230_VHCF_GROWTH_FACTOR", 16.0, 1.000001),
        "maximum_growth_trials": _env_int(
            "V10230_VHCF_MAX_GROWTH_TRIALS", 32, 1
        ),
        "maximum_bisection_trials": _env_int(
            "V10230_VHCF_MAX_BISECTION_TRIALS", 48, 1
        ),
        "relative_cycle_tolerance": _env_float(
            "V10230_VHCF_RELATIVE_CYCLE_TOL", 1.0e-8, 1.0e-14
        ),
    }


def _bounded(value: float, lower: float, upper: float) -> float:
    return min(max(float(value), float(lower)), float(upper))


def _same_cycle(a: float, b: float) -> bool:
    return math.isclose(float(a), float(b), rel_tol=1.0e-14, abs_tol=1.0e-18)


def _audit_and_return(
    *,
    engine,
    selected: float,
    limiter: str,
    selected_trial: dict[str, Any],
    cap: float,
    linear_cycles: float,
    targets: dict[str, float],
    cache: dict[float, dict[str, Any]],
    trial_order: list[float],
    config: dict[str, float | int],
    bracket_low: float | None,
    bracket_high: float | None,
) -> dict[str, Any]:
    cap_key = next((key for key in cache if _same_cycle(key, cap)), None)
    cap_trial = cache.get(cap_key) if cap_key is not None else None
    selected_ratio, selected_limiter = _ratio(selected_trial, targets)
    audit = {
        "schema": MODEL_ID,
        "search_strategy": "linear_seed_geometric_bracket_log_bisection",
        "cycles": float(selected),
        "limiter": str(limiter),
        "cap_cycles": float(cap),
        "linear_cycles": float(linear_cycles),
        "targets": dict(targets),
        "selected_metrics": dict(selected_trial.get("metrics", {})),
        "selected_ratio": float(selected_ratio),
        "selected_physical_limiter": str(selected_limiter),
        "cap_evaluated": cap_trial is not None,
        "cap_metrics": (
            dict(cap_trial.get("metrics", {})) if cap_trial is not None else None
        ),
        "largest_trial_cycles": float(max(cache, default=0.0)),
        "trial_evaluations": len(cache),
        "trial_cycles": [float(value) for value in trial_order],
        "first_trial_cycles": float(trial_order[0] if trial_order else 0.0),
        "full_cap_was_first_trial": bool(
            trial_order and _same_cycle(trial_order[0], cap) and not _same_cycle(linear_cycles, cap)
        ),
        "bracket_low_cycles": None if bracket_low is None else float(bracket_low),
        "bracket_high_cycles": None if bracket_high is None else float(bracket_high),
        "search_config": dict(config),
    }
    engine._v10229_last_vhcf_block_audit = audit
    return {
        "cycles": float(selected),
        "limiter": str(limiter),
        "unlimited_cycles": float(selected),
        "candidate_limits": {
            "nonlinear_cap_cycles": float(cap),
            "nonlinear_selected_cycles": float(selected),
            "linear_selected_cycles": float(linear_cycles),
            "nonlinear_largest_trial_cycles": float(max(cache, default=0.0)),
        },
    }


def select_nonlinear_block(
    controller,
    prediction,
    user_block_cycles: float | None,
    linear_diagnostic: dict[str, Any],
) -> dict[str, Any]:
    """Return the largest exact block satisfying the existing physical targets."""
    engine = getattr(prediction, "_v10229_vhcf_engine", None)
    waveform = getattr(prediction, "_v10229_vhcf_waveform", None)
    temperature_K = getattr(prediction, "_v10229_vhcf_temperature_K", None)
    if engine is None or waveform is None or temperature_K is None:
        return dict(linear_diagnostic)
    if not bool(getattr(engine, "persistent_site_cyclic_v10229", False)):
        return dict(linear_diagnostic)

    cfg = controller.cfg
    requested = float(
        user_block_cycles
        if user_block_cycles is not None
        else getattr(cfg, "block_cycles", 1.0)
    )
    max_block = float(getattr(cfg, "max_block_cycles", requested))
    min_block = max(float(getattr(cfg, "min_block_cycles", 0.0)), 0.0)
    mode = str(
        getattr(cfg, "cycle_block_mode", "requested_cap") or "requested_cap"
    ).lower()
    cap = (
        max_block
        if mode in ("hazard", "hazard_limited", "rate", "auto")
        else min(requested, max_block)
    )
    if not math.isfinite(cap) or cap <= 0.0:
        return dict(linear_diagnostic)
    cap = max(cap, min_block)

    targets = _targets(controller, float(getattr(engine, "B", 0.0)))
    config = _search_config()
    cache: dict[float, dict[str, Any]] = {}
    trial_order: list[float] = []

    def evaluate(value: float) -> tuple[float, dict[str, Any], float, str]:
        cycles = _bounded(value, min_block, cap)
        key = next((item for item in cache if _same_cycle(item, cycles)), None)
        if key is None:
            key = float(cycles)
            cache[key] = _block_trial(
                engine, waveform, float(temperature_K), prediction, key
            )
            trial_order.append(key)
        trial = cache[key]
        ratio, physical_limiter = _ratio(trial, targets)
        return key, trial, float(ratio), str(physical_limiter)

    linear_cycles = _bounded(
        float(linear_diagnostic.get("cycles", min_block)), min_block, cap
    )
    low_cycles, low_trial, low_ratio, _ = evaluate(linear_cycles)

    if low_ratio > 1.0 + 1.0e-10:
        if _same_cycle(low_cycles, min_block):
            return _audit_and_return(
                engine=engine,
                selected=min_block,
                limiter="min_block_cycles",
                selected_trial=low_trial,
                cap=cap,
                linear_cycles=linear_cycles,
                targets=targets,
                cache=cache,
                trial_order=trial_order,
                config=config,
                bracket_low=None,
                bracket_high=low_cycles,
            )
        min_cycles, min_trial, min_ratio, _ = evaluate(min_block)
        if min_ratio > 1.0 + 1.0e-10:
            return _audit_and_return(
                engine=engine,
                selected=min_block,
                limiter="min_block_cycles",
                selected_trial=min_trial,
                cap=cap,
                linear_cycles=linear_cycles,
                targets=targets,
                cache=cache,
                trial_order=trial_order,
                config=config,
                bracket_low=None,
                bracket_high=low_cycles,
            )
        bracket_low = min_cycles
        bracket_low_trial = min_trial
        bracket_high = low_cycles
    else:
        if _same_cycle(low_cycles, cap):
            selected = (
                float(low_trial.get("cycles_consumed", cap))
                if bool(low_trial.get("fired", False))
                else cap
            )
            limiter = (
                "nonlinear_event_localized"
                if bool(low_trial.get("fired", False))
                else "max_block_cycles"
            )
            return _audit_and_return(
                engine=engine,
                selected=selected,
                limiter=limiter,
                selected_trial=low_trial,
                cap=cap,
                linear_cycles=linear_cycles,
                targets=targets,
                cache=cache,
                trial_order=trial_order,
                config=config,
                bracket_low=low_cycles,
                bracket_high=None,
            )

        bracket_low = low_cycles
        bracket_low_trial = low_trial
        bracket_high = None
        growth = float(config["growth_factor"])
        for _ in range(int(config["maximum_growth_trials"])):
            positive_step = max(min_block, max(bracket_low, 1.0) * 1.0e-12, 1.0e-18)
            proposed = min(cap, max(bracket_low * growth, bracket_low + positive_step))
            high_cycles, high_trial, high_ratio, _ = evaluate(proposed)
            if high_ratio <= 1.0 + 1.0e-10:
                bracket_low = high_cycles
                bracket_low_trial = high_trial
                if _same_cycle(high_cycles, cap):
                    selected = (
                        float(high_trial.get("cycles_consumed", cap))
                        if bool(high_trial.get("fired", False))
                        else cap
                    )
                    limiter = (
                        "nonlinear_event_localized"
                        if bool(high_trial.get("fired", False))
                        else "max_block_cycles"
                    )
                    return _audit_and_return(
                        engine=engine,
                        selected=selected,
                        limiter=limiter,
                        selected_trial=high_trial,
                        cap=cap,
                        linear_cycles=linear_cycles,
                        targets=targets,
                        cache=cache,
                        trial_order=trial_order,
                        config=config,
                        bracket_low=bracket_low,
                        bracket_high=None,
                    )
                continue
            bracket_high = high_cycles
            break
        if bracket_high is None:
            raise RuntimeError(
                "v10.2.30 VHCF geometric search exhausted before reaching or bracketing "
                f"cap={cap:g} cycles"
            )

    assert bracket_high is not None
    tolerance = float(config["relative_cycle_tolerance"])
    for _ in range(int(config["maximum_bisection_trials"])):
        width = bracket_high - bracket_low
        if width <= tolerance * max(bracket_high, 1.0):
            break
        if bracket_low > 0.0:
            mid = math.sqrt(bracket_low * bracket_high)
        else:
            mid = 0.5 * (bracket_low + bracket_high)
        mid_cycles, mid_trial, mid_ratio, _ = evaluate(mid)
        if mid_ratio <= 1.0 + 1.0e-10:
            bracket_low = mid_cycles
            bracket_low_trial = mid_trial
        else:
            bracket_high = mid_cycles

    selected = float(bracket_low)
    _selected_ratio, physical_limiter = _ratio(bracket_low_trial, targets)
    return _audit_and_return(
        engine=engine,
        selected=selected,
        limiter=f"nonlinear_{physical_limiter}",
        selected_trial=bracket_low_trial,
        cap=cap,
        linear_cycles=linear_cycles,
        targets=targets,
        cache=cache,
        trial_order=trial_order,
        config=config,
        bracket_low=bracket_low,
        bracket_high=bracket_high,
    )


__all__ = [
    "MODEL_ID",
    "attach_prediction_context",
    "select_nonlinear_block",
]
