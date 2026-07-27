"""Nonlinear VHCF block selection for the v10.2.29 persistent-site engine.

The existing fatigue controller computes a one-cycle tangent prediction.  That
prediction is appropriate in high-activity conditions, but it can be excessively
conservative over long cycle blocks because persistent emission is committed by
an implicit Taylor-backstress-limited solve.  This module evaluates candidate
blocks on a private engine clone using the exact existing commit path and selects
the largest block satisfying the configured physical state-increment targets.
"""
from __future__ import annotations

import copy
import math
from typing import Any

import numpy as np


MODEL_ID = "v10.2.29_persistent_site_nonlinear_vhcf_block_v1"


def _numeric(mapping: dict[str, Any], key: str) -> float:
    value = mapping.get(key, 0.0)
    if isinstance(value, (int, float, np.integer, np.floating)):
        return float(value)
    return 0.0


def _finite_positive(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) and number > 0.0 else None


def attach_prediction_context(prediction, engine, waveform, temperature_K: float):
    """Attach transient routing metadata to an engine-native prediction."""
    setattr(prediction, "_v10229_vhcf_engine", engine)
    setattr(prediction, "_v10229_vhcf_waveform", waveform)
    setattr(prediction, "_v10229_vhcf_temperature_K", float(temperature_K))
    return prediction


def _targets(controller, B_pre: float) -> dict[str, float]:
    cfg = controller.cfg
    out: dict[str, float] = {}
    target_dB = _finite_positive(getattr(cfg, "target_dB", None))
    if target_dB is not None:
        out["cleavage_clock"] = min(target_dB, max(1.0 - float(B_pre), 0.0))
    for name, attr in (
        ("stored_pz", "target_dN_store"),
        ("emitted_pz", "target_dN_emit"),
        ("mobile_pz", "target_dN_mobile"),
        ("escape_pz", "target_dN_escape"),
    ):
        value = _finite_positive(getattr(cfg, attr, None))
        if value is not None:
            out[name] = value
    return out


def _block_trial(engine, waveform, temperature_K: float, prediction, cycles: float) -> dict:
    """Commit one candidate block on a private clone and return actual increments."""
    requested = max(float(cycles), 0.0)
    trial = copy.deepcopy(engine)
    mobile_pre = float(trial.mpz.mobile_count)
    retained_pre = float(trial.mpz.retained_count)
    escaped_pre = float(trial.mpz.escaped_total)
    emitted_pre = float(trial.mpz.emitted_total)

    coupled = trial._integrate_coupled(
        float(waveform.Kmax),
        float(temperature_K),
        requested * float(waveform.period_s),
        stress_override=float(prediction.avg_sigma_tip),
        lambda_override=float(prediction.mu_cleave) * float(waveform.frequency_Hz),
    )
    consumed = max(float(coupled.get("dt_consumed", 0.0)), 0.0) * max(
        float(waveform.frequency_Hz), 0.0
    )
    plastic = dict(coupled.get("plastic", {}))
    emitted = max(float(trial.mpz.emitted_total) - emitted_pre, 0.0)
    escaped = max(float(trial.mpz.escaped_total) - escaped_pre, 0.0)
    retained_delta = abs(float(trial.mpz.retained_count) - retained_pre)
    mobile_delta = abs(float(trial.mpz.mobile_count) - mobile_pre)
    trapped = max(_numeric(plastic, "dN_trapped"), 0.0)
    released = max(_numeric(plastic, "dN_released"), 0.0)
    escaped_plastic = max(_numeric(plastic, "dN_escaped"), 0.0)

    return {
        "cycles_requested": requested,
        "cycles_consumed": consumed,
        "fired": bool(coupled.get("fired", False)),
        "metrics": {
            "cleavage_clock": max(float(coupled.get("dB", 0.0)), 0.0),
            "emitted_pz": emitted,
            "stored_pz": max(retained_delta, trapped),
            "mobile_pz": max(
                mobile_delta,
                max(emitted + released - trapped - escaped_plastic, 0.0),
            ),
            "escape_pz": max(escaped, escaped_plastic),
        },
        "internal_steps": int(coupled.get("microsteps", 0)),
    }


def _ratio(trial: dict, targets: dict[str, float]) -> tuple[float, str]:
    ratios = {
        name: max(float(trial["metrics"].get(name, 0.0)), 0.0) / target
        for name, target in targets.items()
        if target > 0.0
    }
    if not ratios:
        return 0.0, "max_block_cycles"
    limiter = max(ratios, key=ratios.get)
    return float(ratios[limiter]), str(limiter)


def select_nonlinear_block(
    controller,
    prediction,
    user_block_cycles: float | None,
    linear_diagnostic: dict[str, Any],
) -> dict[str, Any]:
    """Return the largest exact block satisfying current physical targets."""
    engine = getattr(prediction, "_v10229_vhcf_engine", None)
    waveform = getattr(prediction, "_v10229_vhcf_waveform", None)
    temperature_K = getattr(prediction, "_v10229_vhcf_temperature_K", None)
    if engine is None or waveform is None or temperature_K is None:
        return dict(linear_diagnostic)
    if not bool(getattr(engine, "persistent_site_cyclic_v10229", False)):
        return dict(linear_diagnostic)

    cfg = controller.cfg
    req = float(
        user_block_cycles
        if user_block_cycles is not None
        else getattr(cfg, "block_cycles", 1.0)
    )
    max_block = float(getattr(cfg, "max_block_cycles", req))
    min_block = max(float(getattr(cfg, "min_block_cycles", 0.0)), 0.0)
    mode = str(getattr(cfg, "cycle_block_mode", "requested_cap") or "requested_cap").lower()
    cap = max_block if mode in ("hazard", "hazard_limited", "rate", "auto") else min(req, max_block)
    if not math.isfinite(cap) or cap <= 0.0:
        return dict(linear_diagnostic)
    cap = max(cap, min_block)
    targets = _targets(controller, float(getattr(engine, "B", 0.0)))
    cache: dict[float, dict] = {}

    def evaluate(value: float) -> dict:
        key = float(value)
        if key not in cache:
            cache[key] = _block_trial(
                engine, waveform, float(temperature_K), prediction, key
            )
        return cache[key]

    cap_trial = evaluate(cap)
    cap_ratio, cap_limiter = _ratio(cap_trial, targets)
    if cap_ratio <= 1.0 + 1.0e-10:
        selected = float(cap_trial["cycles_consumed"]) if cap_trial["fired"] else cap
        limiter = "nonlinear_event_localized" if cap_trial["fired"] else "max_block_cycles"
        selected_trial = cap_trial
    else:
        linear_cycles = max(
            min(float(linear_diagnostic.get("cycles", min_block)), cap),
            min_block,
        )
        low_trial = evaluate(linear_cycles)
        low_ratio, _ = _ratio(low_trial, targets)
        if low_ratio <= 1.0 + 1.0e-10:
            lo = linear_cycles
            hi = cap
            lo_trial = low_trial
        else:
            lo = min_block
            hi = linear_cycles
            lo_trial = evaluate(lo)
            lo_ratio, _ = _ratio(lo_trial, targets)
            if lo_ratio > 1.0 + 1.0e-10:
                selected = min_block
                selected_trial = lo_trial
                limiter = "min_block_cycles"
                audit = {
                    "schema": MODEL_ID,
                    "cycles": selected,
                    "limiter": limiter,
                    "cap_cycles": cap,
                    "linear_cycles": linear_cycles,
                    "targets": targets,
                    "cap_metrics": cap_trial["metrics"],
                    "selected_metrics": selected_trial["metrics"],
                    "trial_evaluations": len(cache),
                }
                engine._v10229_last_vhcf_block_audit = audit
                return {
                    "cycles": selected,
                    "limiter": limiter,
                    "unlimited_cycles": selected,
                    "candidate_limits": {
                        "nonlinear_cap_cycles": cap,
                        "nonlinear_selected_cycles": selected,
                        "linear_selected_cycles": linear_cycles,
                    },
                }

        for _ in range(52):
            if hi - lo <= 1.0e-8 * max(hi, 1.0):
                break
            mid = 0.5 * (lo + hi)
            mid_trial = evaluate(mid)
            mid_ratio, _ = _ratio(mid_trial, targets)
            if mid_ratio <= 1.0 + 1.0e-10:
                lo = mid
                lo_trial = mid_trial
            else:
                hi = mid
        selected = float(lo)
        selected_trial = lo_trial
        _selected_ratio, physical_limiter = _ratio(selected_trial, targets)
        limiter = f"nonlinear_{physical_limiter}"

    audit = {
        "schema": MODEL_ID,
        "cycles": float(selected),
        "limiter": str(limiter),
        "cap_cycles": float(cap),
        "linear_cycles": float(linear_diagnostic.get("cycles", selected)),
        "targets": dict(targets),
        "cap_metrics": dict(cap_trial["metrics"]),
        "selected_metrics": dict(selected_trial["metrics"]),
        "cap_limiter": cap_limiter,
        "cap_ratio": float(cap_ratio),
        "trial_evaluations": len(cache),
    }
    engine._v10229_last_vhcf_block_audit = audit
    return {
        "cycles": float(selected),
        "limiter": str(limiter),
        "unlimited_cycles": float(selected),
        "candidate_limits": {
            "nonlinear_cap_cycles": float(cap),
            "nonlinear_selected_cycles": float(selected),
            "linear_selected_cycles": float(linear_diagnostic.get("cycles", selected)),
        },
    }


__all__ = [
    "MODEL_ID",
    "attach_prediction_context",
    "select_nonlinear_block",
]
