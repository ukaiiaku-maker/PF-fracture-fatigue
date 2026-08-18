"""v4 explicit-cycle wrapper for fail-closed physical surface return.

The exact cycle, hazard, stochastic threshold, event-length, and crack-advance
integration remain unchanged.  This wrapper rebinds the reversible state class
to v4, where only true reverse-driven return of the emitted population can
cancel the emission-linked blunting ledger.
"""
from __future__ import annotations

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
