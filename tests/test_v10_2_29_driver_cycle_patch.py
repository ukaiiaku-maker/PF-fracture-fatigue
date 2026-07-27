from arrhenius_fracture import sharp_front
from arrhenius_fracture.fatigue_driver_cycle_accounting_v10229 import (
    install_consumed_cycle_accounting,
    restore_consumed_cycle_accounting,
)


def test_consumed_cycle_patch_installs_and_restores_exactly():
    original = sharp_front.run_2d
    install_consumed_cycle_accounting()
    try:
        assert sharp_front.run_2d is not original
    finally:
        restore_consumed_cycle_accounting()
    assert sharp_front.run_2d is original
