"""Production high-cycle engine for event-to-event fatigue growth.

The v2 event-localization state machine is retained. Slow waiting intervals use
rate-separated, positivity-preserving chained DMD. Atomic live checkpoints are
written at integrator entry/return, after validated projective segments, and
through throttled exact-cycle recovery.
"""
from __future__ import annotations

import os
import types

from . import persistent_site_high_cycle_engine_v10230_v2 as _base
from .persistent_site_high_cycle_checkpoint_v10230 import maybe_write_checkpoint
from .persistent_site_high_cycle_dmd_v10230_v5 import (
    MODEL_ID as DMD_MODEL_ID,
    chained_dmd_config,
    propagate_chained_dmd_cycles,
)


MODEL_ID = (
    "v10.2.30_production_event_to_event_high_cycle_v5_"
    "rate_separated_positive_state_dmd"
)


def _projective_with_requested_scale(
    engine,
    controller,
    waveform,
    temperature_K: float,
    cycles_requested: float,
    requested_project_cycles: float,
):
    previous = os.environ.get("V10230_DMD_CHAIN_MAX_SEGMENTS")
    if previous is None:
        os.environ["V10230_DMD_CHAIN_MAX_SEGMENTS"] = "128"
    try:
        return propagate_chained_dmd_cycles(
            engine,
            controller,
            waveform,
            temperature_K,
            cycles_requested,
            requested_project_cycles=requested_project_cycles,
        )
    finally:
        if previous is None:
            os.environ.pop("V10230_DMD_CHAIN_MAX_SEGMENTS", None)
        else:
            os.environ["V10230_DMD_CHAIN_MAX_SEGMENTS"] = previous


def _commit_exact_cycle_with_checkpoint(engine, cycle, cache):
    normalized = _base._commit_exact_cycle(engine, cycle, cache)
    maybe_write_checkpoint(
        engine,
        reason="exact_cycle_progress",
        metadata={"model_id": MODEL_ID},
    )
    return normalized


def _bind_exact_burst():
    base = _base._advance_exact_cycle_burst
    namespace = dict(base.__globals__)
    namespace["_commit_exact_cycle"] = _commit_exact_cycle_with_checkpoint
    function = types.FunctionType(
        base.__code__,
        namespace,
        name=base.__name__,
        argdefs=base.__defaults__,
        closure=base.__closure__,
    )
    function.__kwdefaults__ = base.__kwdefaults__
    function.__annotations__ = dict(base.__annotations__)
    function.__doc__ = base.__doc__
    function.__module__ = __name__
    return function


_bound_exact_burst = _bind_exact_burst()


def _bind_integrator():
    base = _base.integrate_state_coupled_waveform
    namespace = dict(base.__globals__)
    namespace["MODEL_ID"] = MODEL_ID
    namespace["_projective_with_requested_scale"] = _projective_with_requested_scale
    namespace["_advance_exact_cycle_burst"] = _bound_exact_burst
    function = types.FunctionType(
        base.__code__,
        namespace,
        name=base.__name__,
        argdefs=base.__defaults__,
        closure=base.__closure__,
    )
    function.__kwdefaults__ = base.__kwdefaults__
    function.__annotations__ = dict(base.__annotations__)
    function.__doc__ = base.__doc__
    function.__module__ = __name__
    return function


_bound_integrator = _bind_integrator()


def integrate_state_coupled_waveform(
    engine,
    controller,
    waveform,
    temperature_K: float,
    cycles_requested: float,
):
    maybe_write_checkpoint(
        engine,
        waveform=waveform,
        temperature_K=temperature_K,
        reason="integrator_entry",
        metadata={
            "model_id": MODEL_ID,
            "cycles_requested": float(cycles_requested),
        },
    )
    result = _bound_integrator(
        engine, controller, waveform, temperature_K, cycles_requested
    )
    fired = bool(result.get("fired", False))
    maybe_write_checkpoint(
        engine,
        waveform=waveform,
        temperature_K=temperature_K,
        reason="first_passage" if fired else "integrator_return",
        metadata={
            "model_id": MODEL_ID,
            "cycles_requested": float(cycles_requested),
            "cycles_consumed": float(
                result.get("coupled_hazard_cycles_consumed", 0.0)
            ),
            "fired": fired,
            "partial_return": bool(
                result.get("coupled_hazard_partial_return", False)
            ),
            "mode_operations": int(
                result.get("coupled_hazard_mode_operations", 0)
            ),
        },
    )
    result["coupled_hazard_rate_separated_ledgers"] = True
    result["coupled_hazard_positivity_preserving_coordinates"] = True
    result["coupled_hazard_live_checkpointing"] = True
    result["coupled_hazard_growth_objective"] = "event_to_event_100um"
    return result


high_cycle_config = _base.high_cycle_config
invalidate_high_cycle_cache = _base.invalidate_high_cycle_cache


__all__ = [
    "DMD_MODEL_ID",
    "MODEL_ID",
    "chained_dmd_config",
    "high_cycle_config",
    "integrate_state_coupled_waveform",
    "invalidate_high_cycle_cache",
]
