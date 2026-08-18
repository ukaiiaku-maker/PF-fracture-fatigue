"""v4 explicit-cycle wrapper for fail-closed physical surface return.

The exact cycle, hazard, stochastic threshold, event-length, and crack-advance
integration remain unchanged.  This wrapper rebinds the reversible state class
to v4, where only true reverse-driven return of the emitted population can
cancel the emission-linked blunting ledger.
"""
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

import v914_minimal_reversible_explicit as _v2
import v914_minimal_reversible_explicit_v3 as _v3
from v914_minimal_reversible_state_v4 import (
    MODEL_ID,
    MinimalReversibleEmergentGNDState,
)


_v2.MinimalReversibleEmergentGNDState = MinimalReversibleEmergentGNDState
_v2.REVERSIBLE_STATE_MODEL_ID = MODEL_ID
_v2._diagnostic = _v3._diagnostic

MODE = _v2.MODE
SCHEMA = _v2.SCHEMA
run_minimal_reversible_explicit = _v2.run_minimal_reversible_explicit

__all__ = ["MODE", "SCHEMA", "run_minimal_reversible_explicit"]
