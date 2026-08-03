"""Stable import path for the production v10.2.30 high-cycle engine.

The active implementation is the v3 state machine with validated affine-DMD
cycle-map propagation.  The v2 exact-cycle engine remains available as a
reference implementation and regression baseline.
"""
from .persistent_site_high_cycle_engine_v10230_v3 import (
    DMD_MODEL_ID,
    MODEL_ID,
    high_cycle_config,
    integrate_state_coupled_waveform,
    invalidate_high_cycle_cache,
)

__all__ = [
    "DMD_MODEL_ID",
    "MODEL_ID",
    "high_cycle_config",
    "integrate_state_coupled_waveform",
    "invalidate_high_cycle_cache",
]
