"""State-coupled v10.2.30 VHCF selector with performance-safe bracketing."""
from __future__ import annotations

import copy
from typing import Any

from . import persistent_site_vhcf_selector_v10230 as _base
from .persistent_site_coupled_hazard_v10229 import integrate_state_coupled_waveform


MODEL_ID = "v10.2.30_state_coupled_geometric_vhcf_selector_v1"
attach_prediction_context = _base.attach_prediction_context


def _coupled_block_trial(
    controller,
    engine,
    waveform,
    temperature_K: float,
    prediction,
    cycles: float,
) -> dict[str, Any]:
    requested = max(float(cycles), 0.0)
    trial = copy.deepcopy(engine)
    mobile_pre = float(trial.mpz.mobile_count)
    retained_pre = float(trial.mpz.retained_count)
    escaped_pre = float(trial.mpz.escaped_total)
    emitted_pre = float(trial.mpz.emitted_total)

    coupled = integrate_state_coupled_waveform(
        trial,
        controller,
        waveform,
        float(temperature_K),
        requested,
    )
    consumed = max(float(coupled.get("dt_consumed", 0.0)), 0.0) * max(
        float(waveform.frequency_Hz), 0.0
    )
    plastic = dict(coupled.get("plastic", {}))
    emitted = max(float(trial.mpz.emitted_total) - emitted_pre, 0.0)
    escaped = max(float(trial.mpz.escaped_total) - escaped_pre, 0.0)
    retained_delta = abs(float(trial.mpz.retained_count) - retained_pre)
    mobile_delta = abs(float(trial.mpz.mobile_count) - mobile_pre)
    trapped = max(float(plastic.get("dN_trapped", 0.0)), 0.0)
    released = max(float(plastic.get("dN_released", 0.0)), 0.0)
    escaped_plastic = max(float(plastic.get("dN_escaped", 0.0)), 0.0)

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
        "coupled_hazard_accepted_segments": int(
            coupled.get("coupled_hazard_accepted_segments", 0)
        ),
        "coupled_hazard_rejected_splits": int(
            coupled.get("coupled_hazard_rejected_splits", 0)
        ),
        "coupled_hazard_log_lambda_span_decades": float(
            coupled.get("coupled_hazard_log_lambda_span_decades", 0.0)
        ),
    }


def select_nonlinear_block(
    controller,
    prediction,
    user_block_cycles: float | None,
    linear_diagnostic: dict[str, Any],
) -> dict[str, Any]:
    """Use exact state-coupled trials with the v10.2.30 geometric search."""
    original = _base._block_trial

    def routed(engine, waveform, temperature_K, pred, cycles):
        return _coupled_block_trial(
            controller,
            engine,
            waveform,
            temperature_K,
            pred,
            cycles,
        )

    _base._block_trial = routed
    try:
        result = _base.select_nonlinear_block(
            controller,
            prediction,
            user_block_cycles,
            linear_diagnostic,
        )
    finally:
        _base._block_trial = original

    engine = getattr(prediction, "_v10229_vhcf_engine", None)
    if engine is not None:
        audit = dict(getattr(engine, "_v10229_last_vhcf_block_audit", {}))
        audit["selector_schema"] = MODEL_ID
        audit["state_coupled_cleavage_hazard"] = True
        audit["full_horizon_first_trial_avoided"] = not bool(
            audit.get("full_cap_was_first_trial", False)
        )
        engine._v10229_last_vhcf_block_audit = audit
    return result


__all__ = [
    "MODEL_ID",
    "attach_prediction_context",
    "select_nonlinear_block",
]
