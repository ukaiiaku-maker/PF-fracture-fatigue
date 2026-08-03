"""Delegate fatigue prediction and VHCF block sizing to persistent-site engines."""
from __future__ import annotations

from .fatigue_v1 import FatigueCycleHazardController
from .persistent_site_vhcf_coupled_selector_v10229 import (
    attach_prediction_context,
    select_nonlinear_block,
)

_ORIGINAL_INTEGRATE_ONE_CYCLE = None
_ORIGINAL_CHOOSE_BLOCK_DIAGNOSTIC = None
_ORIGINAL_CYCLE_STEP_FRONT = None
_ACTIVE_STEP_CONTEXT = None


def install_engine_native_cycle_preview() -> None:
    global _ORIGINAL_INTEGRATE_ONE_CYCLE
    global _ORIGINAL_CHOOSE_BLOCK_DIAGNOSTIC
    global _ORIGINAL_CYCLE_STEP_FRONT
    if _ORIGINAL_INTEGRATE_ONE_CYCLE is not None:
        return

    original_integrate = FatigueCycleHazardController.integrate_one_cycle
    original_choose = FatigueCycleHazardController.choose_block_cycles_diagnostic
    original_step = FatigueCycleHazardController.cycle_step_front

    def delegated_integrate(self, front, waveform, T_K):
        preview = getattr(front, "preview_cycle_waveform", None)
        if preview is not None:
            prediction = preview(self, waveform, T_K)
            if bool(getattr(front, "persistent_site_cyclic_v10229", False)):
                return attach_prediction_context(
                    prediction, front, waveform, T_K
                )
            return prediction
        return original_integrate(self, front, waveform, T_K)

    def delegated_choose(self, pred, user_block_cycles=None):
        linear = original_choose(self, pred, user_block_cycles)
        if getattr(pred, "_v10229_vhcf_engine", None) is None:
            context = _ACTIVE_STEP_CONTEXT
            if context is not None:
                front, waveform, temperature = context
                attach_prediction_context(pred, front, waveform, temperature)
        return select_nonlinear_block(self, pred, user_block_cycles, linear)

    def delegated_step(
        self,
        front,
        waveform,
        T_K,
        requested_cycles=None,
        force_cycles=None,
    ):
        global _ACTIVE_STEP_CONTEXT
        if (
            bool(getattr(front, "persistent_site_cyclic_v10229", False))
            and force_cycles is not None
            and callable(getattr(front, "cycle_step_waveform", None))
        ):
            old_context = _ACTIVE_STEP_CONTEXT
            cap = max(float(force_cycles), 0.0)
            try:
                _ACTIVE_STEP_CONTEXT = (front, waveform, float(T_K))
                # A globally selected shared cycle block is already a decision.
                # Commit it directly through the engine's explicit force path.
                # Do not mutate cfg.max_block_cycles and do not re-enter selection.
                return front.cycle_step_waveform(
                    self,
                    waveform,
                    T_K,
                    requested_cycles=cap,
                    force_cycles=cap,
                )
            finally:
                _ACTIVE_STEP_CONTEXT = old_context
        return original_step(
            self,
            front,
            waveform,
            T_K,
            requested_cycles=requested_cycles,
            force_cycles=force_cycles,
        )

    _ORIGINAL_INTEGRATE_ONE_CYCLE = original_integrate
    _ORIGINAL_CHOOSE_BLOCK_DIAGNOSTIC = original_choose
    _ORIGINAL_CYCLE_STEP_FRONT = original_step
    FatigueCycleHazardController.integrate_one_cycle = delegated_integrate
    FatigueCycleHazardController.choose_block_cycles_diagnostic = delegated_choose
    FatigueCycleHazardController.cycle_step_front = delegated_step


def restore_engine_native_cycle_preview() -> None:
    global _ORIGINAL_INTEGRATE_ONE_CYCLE
    global _ORIGINAL_CHOOSE_BLOCK_DIAGNOSTIC
    global _ORIGINAL_CYCLE_STEP_FRONT
    global _ACTIVE_STEP_CONTEXT
    if _ORIGINAL_INTEGRATE_ONE_CYCLE is not None:
        FatigueCycleHazardController.integrate_one_cycle = (
            _ORIGINAL_INTEGRATE_ONE_CYCLE
        )
        _ORIGINAL_INTEGRATE_ONE_CYCLE = None
    if _ORIGINAL_CHOOSE_BLOCK_DIAGNOSTIC is not None:
        FatigueCycleHazardController.choose_block_cycles_diagnostic = (
            _ORIGINAL_CHOOSE_BLOCK_DIAGNOSTIC
        )
        _ORIGINAL_CHOOSE_BLOCK_DIAGNOSTIC = None
    if _ORIGINAL_CYCLE_STEP_FRONT is not None:
        FatigueCycleHazardController.cycle_step_front = _ORIGINAL_CYCLE_STEP_FRONT
        _ORIGINAL_CYCLE_STEP_FRONT = None
    _ACTIVE_STEP_CONTEXT = None


__all__ = ["install_engine_native_cycle_preview", "restore_engine_native_cycle_preview"]
