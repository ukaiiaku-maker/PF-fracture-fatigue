from __future__ import annotations

import pytest

from arrhenius_fracture import sharp_front_v10_2_30_energy_gated_fatigue as entry


def test_enabled_rebonding_raises_before_any_monkeypatch_is_applied(monkeypatch):
    monkeypatch.setenv("V10230_ENERGY_GATE_ENABLED", "1")
    monkeypatch.setenv("V10230_CRACK_REBONDING_ENABLED", "1")
    monkeypatch.setenv("V10230_CRACK_REBONDING_MODEL_LEVEL", "CLEAN_REVERSIBLE_REBOND")

    original_engine = entry._v10229.AuditedCoupledPersistentSiteCyclicTipEngine
    original_build_shared_engine = entry._shared_state.build_shared_engine
    original_coupled_commit = entry._coupled_commit.integrate_state_coupled_waveform

    v10229_called = {"value": False}
    monkeypatch.setattr(entry._v10229, "main", lambda args: v10229_called.__setitem__("value", True))

    with pytest.raises(RuntimeError, match="rebonding_acceleration_qualified"):
        entry.main(["--fatigue-cycles", "--out", "/tmp/does-not-matter"])

    assert v10229_called["value"] is False
    # Nothing should have been monkeypatched -- the raise happens before any
    # assignment, so every module attribute must be exactly what it was.
    assert entry._v10229.AuditedCoupledPersistentSiteCyclicTipEngine is original_engine
    assert entry._shared_state.build_shared_engine is original_build_shared_engine
    assert entry._coupled_commit.integrate_state_coupled_waveform is original_coupled_commit


def test_disabled_rebonding_does_not_raise_the_acceleration_gate(monkeypatch):
    monkeypatch.setenv("V10230_ENERGY_GATE_ENABLED", "1")
    monkeypatch.delenv("V10230_CRACK_REBONDING_ENABLED", raising=False)

    monkeypatch.setattr(entry._v10229, "main", lambda args: "ok")

    result = entry.main(["--fatigue-cycles", "--out", ""])
    assert result == "ok"


def test_rebond_off_model_level_does_not_raise_even_if_enabled(monkeypatch):
    monkeypatch.setenv("V10230_ENERGY_GATE_ENABLED", "1")
    monkeypatch.setenv("V10230_CRACK_REBONDING_ENABLED", "1")
    monkeypatch.setenv("V10230_CRACK_REBONDING_MODEL_LEVEL", "REBOND_OFF")

    monkeypatch.setattr(entry._v10229, "main", lambda args: "ok")

    result = entry.main(["--fatigue-cycles", "--out", ""])
    assert result == "ok"
