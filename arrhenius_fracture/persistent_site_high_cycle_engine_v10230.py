"""Compatibility entry for the production v10.2.30 high-cycle engine.

The implementation lives in ``persistent_site_high_cycle_engine_v10230_v2``.
Keeping this module as the stable import path preserves the audited fatigue entry
while replacing ordinary sub-cycle fallback with exact committed Poincare-cycle
bursts.
"""
from .persistent_site_high_cycle_engine_v10230_v2 import (
    MODEL_ID,
    high_cycle_config,
    integrate_state_coupled_waveform,
    invalidate_high_cycle_cache,
)

__all__ = [
    "MODEL_ID",
    "high_cycle_config",
    "integrate_state_coupled_waveform",
    "invalidate_high_cycle_cache",
]
