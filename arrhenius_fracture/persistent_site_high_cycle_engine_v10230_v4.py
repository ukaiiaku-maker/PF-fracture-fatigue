"""Production v10.2.30 high-cycle state machine with chained affine DMD.

The proven v2 event/periodic/exact-cycle state-machine control is retained, but
one slow-projective operation now advances through a sequence of independently
validated affine-DMD segments.  Successful segments continue immediately toward
the requested physical horizon; an exact-cycle fallback is entered only when the
chained propagator reports a genuine validation or event-guard stop.
"""
from __future__ import annotations

import types

from . import persistent_site_high_cycle_engine_v10230_v2 as _base
from .persistent_site_high_cycle_dmd_v10230_v3 import (
    MODEL_ID as DMD_MODEL_ID,
    chained_dmd_config,
    propagate_chained_dmd_cycles,
)


MODEL_ID = "v10.2.30_production_high_cycle_state_machine_v4_chained_affine_dmd"


def _chained_dmd_with_requested_scale(
    engine,
    controller,
    waveform,
    temperature_K: float,
    cycles_requested: float,
    requested_project_cycles: float,
):
    return propagate_chained_dmd_cycles(
        engine,
        controller,
        waveform,
        temperature_K,
        cycles_requested,
        requested_project_cycles=requested_project_cycles,
    )


def _bind_integrator():
    base = _base.integrate_state_coupled_waveform
    namespace = dict(base.__globals__)
    namespace["MODEL_ID"] = MODEL_ID
    namespace["_projective_with_requested_scale"] = (
        _chained_dmd_with_requested_scale
    )
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


integrate_state_coupled_waveform = _bind_integrator()
high_cycle_config = _base.high_cycle_config
invalidate_high_cycle_cache = _base.invalidate_high_cycle_cache


__all__ = [
    "MODEL_ID",
    "DMD_MODEL_ID",
    "chained_dmd_config",
    "high_cycle_config",
    "integrate_state_coupled_waveform",
    "invalidate_high_cycle_cache",
]
