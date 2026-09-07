from __future__ import annotations

import pytest

from arrhenius_fracture.crack_rebonding_kinetics_v10230 import CrackRebondingControls, FeedbackMode
from arrhenius_fracture.crack_rebonding_v10230 import install_crack_rebonding


class FakeEngine:
    pass


def test_install_noop_when_disabled():
    engine = FakeEngine()
    install_crack_rebonding(engine, CrackRebondingControls(enabled=False))
    assert not hasattr(engine, "_rebonding_state")


def test_install_allocates_state_when_enabled():
    engine = FakeEngine()
    cfg = CrackRebondingControls(enabled=True)
    install_crack_rebonding(engine, cfg)
    assert getattr(engine, "_rebonding_state", None) is not None
    assert engine.rebonding_acceleration_qualified is False


def test_double_install_same_config_is_noop():
    engine = FakeEngine()
    cfg = CrackRebondingControls(enabled=True)
    install_crack_rebonding(engine, cfg)
    state_before = engine._rebonding_state
    install_crack_rebonding(engine, cfg)
    assert engine._rebonding_state is state_before


def test_double_install_different_config_raises():
    engine = FakeEngine()
    install_crack_rebonding(engine, CrackRebondingControls(enabled=True, chemistry_factor=1.0))
    with pytest.raises(RuntimeError):
        install_crack_rebonding(engine, CrackRebondingControls(enabled=True, chemistry_factor=0.5))
