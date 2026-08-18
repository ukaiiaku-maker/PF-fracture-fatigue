"""Explicit-cycle wrappers for the v5 finite-tip shielding audit."""
from __future__ import annotations

import os
from pathlib import Path
import sys

_SCRIPTS = Path(__file__).resolve().parent
_DEFAULT_V914 = Path(
    os.environ.get(
        "V914_ROOT",
        "/Volumes/Data/Data/Nanopillar_calculation/Arrhenius_FEM_CZM_MPZ_v9_14_cyclic_fatigue_knee_search",
    )
)
for _path in (str(_SCRIPTS), str(_DEFAULT_V914)):
    while _path in sys.path:
        sys.path.remove(_path)
sys.path.insert(0, str(_DEFAULT_V914))
sys.path.insert(0, str(_SCRIPTS))

import v914_minimal_reversible_explicit as _base_explicit
import v914_minimal_reversible_explicit_v3 as _v3_explicit
from v914_finite_tip_shielding_state_v5 import (
    FLOOR_MODEL_ID,
    SHIFT_MODEL_ID,
    FiniteTipFloorReversibleState,
    FiniteTipShiftReversibleState,
)


def _run_with_state(state_cls, model_id, *args, **kwargs):
    """Run the authoritative explicit integrator with one v5 state class.

    The base explicit integrator intentionally resolves its state class and
    model ID through module globals.  Rebind them transactionally so a floor
    audit and a shift audit cannot leak into one another in the same process.
    """
    old_state = _base_explicit.MinimalReversibleEmergentGNDState
    old_id = _base_explicit.REVERSIBLE_STATE_MODEL_ID
    old_diag = _base_explicit._diagnostic
    try:
        _base_explicit.MinimalReversibleEmergentGNDState = state_cls
        _base_explicit.REVERSIBLE_STATE_MODEL_ID = model_id
        _base_explicit._diagnostic = _v3_explicit._diagnostic
        return _base_explicit.run_minimal_reversible_explicit(*args, **kwargs)
    finally:
        _base_explicit.MinimalReversibleEmergentGNDState = old_state
        _base_explicit.REVERSIBLE_STATE_MODEL_ID = old_id
        _base_explicit._diagnostic = old_diag


def run_finite_tip_floor_explicit(*args, **kwargs):
    return _run_with_state(
        FiniteTipFloorReversibleState,
        FLOOR_MODEL_ID,
        *args,
        **kwargs,
    )


def run_finite_tip_shift_explicit(*args, **kwargs):
    return _run_with_state(
        FiniteTipShiftReversibleState,
        SHIFT_MODEL_ID,
        *args,
        **kwargs,
    )


__all__ = [
    "run_finite_tip_floor_explicit",
    "run_finite_tip_shift_explicit",
]
