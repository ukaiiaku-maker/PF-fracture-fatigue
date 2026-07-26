from types import SimpleNamespace

from arrhenius_fracture.fatigue_controller_delegate_v10229 import (
    install_engine_native_cycle_preview,
    restore_engine_native_cycle_preview,
)
from arrhenius_fracture.fatigue_v1 import FatigueCycleHazardController


class FakeFront:
    def preview_cycle_waveform(self, controller, waveform, T_K):
        return (controller, waveform, T_K, "engine-native")


def test_controller_delegates_prediction_to_front_engine():
    original = FatigueCycleHazardController.integrate_one_cycle
    install_engine_native_cycle_preview()
    try:
        controller = object.__new__(FatigueCycleHazardController)
        waveform = SimpleNamespace()
        out = controller.integrate_one_cycle(FakeFront(), waveform, 300.0)
        assert out == (controller, waveform, 300.0, "engine-native")
    finally:
        restore_engine_native_cycle_preview()
    assert FatigueCycleHazardController.integrate_one_cycle is original
