from __future__ import annotations

import pytest

from arrhenius_fracture.crack_rebonding_kinetics_v10230 import CrackRebondingControls, FeedbackMode
from arrhenius_fracture.crack_rebonding_v10230 import install_crack_rebonding


def test_hazard_only_is_the_only_implemented_mode():
    cfg = CrackRebondingControls(enabled=True, feedback_mode=FeedbackMode.HAZARD_ONLY_REBOND_SHIELD)
    cfg.validate()  # does not raise


@pytest.mark.parametrize(
    "mode",
    [FeedbackMode.COMMON_POSITIVE_LOCAL_K_REDUCTION, FeedbackMode.HAZARD_AND_ENERGY_GATE_COUPLED],
)
def test_other_modes_raise_not_implemented_at_config_resolution(mode):
    cfg = CrackRebondingControls(enabled=True, feedback_mode=mode)
    with pytest.raises(NotImplementedError):
        cfg.validate()


@pytest.mark.parametrize(
    "mode",
    [FeedbackMode.COMMON_POSITIVE_LOCAL_K_REDUCTION, FeedbackMode.HAZARD_AND_ENERGY_GATE_COUPLED],
)
def test_install_never_constructs_an_engine_override_for_unimplemented_modes(mode):
    class FakeEngine:
        pass

    engine = FakeEngine()
    cfg = CrackRebondingControls(enabled=True, feedback_mode=mode)
    with pytest.raises(NotImplementedError):
        install_crack_rebonding(engine, cfg)
    # Since install_crack_rebonding calls cfg.validate() first, no engine
    # attribute should have been touched before the raise.
    assert not hasattr(engine, "_rebonding_state")
    assert not hasattr(engine, "sigma_opening_tip")
