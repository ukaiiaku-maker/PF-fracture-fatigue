"""Stable import path for the production v10.2.30 high-cycle engine.

The active implementation is the v4 state machine with chained, independently
validated affine-DMD cycle-map segments.  The v2 exact-cycle engine and v3
single-segment DMD engine remain available as regression baselines.
"""
from .persistent_site_high_cycle_engine_v10230_v4 import (
    DMD_MODEL_ID,
    MODEL_ID,
    chained_dmd_config,
    high_cycle_config,
    integrate_state_coupled_waveform,
    invalidate_high_cycle_cache,
)

__all__ = [
    "DMD_MODEL_ID",
    "MODEL_ID",
    "chained_dmd_config",
    "high_cycle_config",
    "integrate_state_coupled_waveform",
    "invalidate_high_cycle_cache",
]
