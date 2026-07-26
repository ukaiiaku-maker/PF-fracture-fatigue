"""Delegate legacy controller prediction to an engine-native cyclic preview."""
from __future__ import annotations

from .fatigue_v1 import FatigueCycleHazardController

_ORIGINAL_INTEGRATE_ONE_CYCLE = None


def install_engine_native_cycle_preview() -> None:
    global _ORIGINAL_INTEGRATE_ONE_CYCLE
    if _ORIGINAL_INTEGRATE_ONE_CYCLE is not None:
        return
    original = FatigueCycleHazardController.integrate_one_cycle

    def delegated(self, front, waveform, T_K):
        preview = getattr(front, "preview_cycle_waveform", None)
        if preview is not None:
            return preview(self, waveform, T_K)
        return original(self, front, waveform, T_K)

    _ORIGINAL_INTEGRATE_ONE_CYCLE = original
    FatigueCycleHazardController.integrate_one_cycle = delegated


def restore_engine_native_cycle_preview() -> None:
    global _ORIGINAL_INTEGRATE_ONE_CYCLE
    if _ORIGINAL_INTEGRATE_ONE_CYCLE is not None:
        FatigueCycleHazardController.integrate_one_cycle = _ORIGINAL_INTEGRATE_ONE_CYCLE
        _ORIGINAL_INTEGRATE_ONE_CYCLE = None


__all__ = ["install_engine_native_cycle_preview", "restore_engine_native_cycle_preview"]
