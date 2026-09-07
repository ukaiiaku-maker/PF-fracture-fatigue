"""S8D (round-3 follow-up): the V10230_FATIGUE_INTEGRATOR_MODE selector.

Exploration confirmed the real CLI's --fatigue-cycles mode unconditionally
monkeypatches persistent_site_cyclic_coupled_v10229.integrate_state_coupled_
waveform (the name CoupledPersistentSiteCyclicTipEngine.cycle_step_waveform
actually calls) to the accelerated DMD/Poincare engine -- this is *why* the
Gate-S0 fail-closed check raises unconditionally whenever rebonding is
enabled: for a real CLI run, the confirmed rebonding injection point
(persistent_site_coupled_hazard_v10229.py) is never reached at all once
that monkeypatch is applied. This file tests the additive, default-
preserving selector that opts out of just that one swap.
"""
from __future__ import annotations

import pytest

from arrhenius_fracture import sharp_front_v10_2_30_energy_gated_fatigue as entry
from arrhenius_fracture.crack_rebonding_kinetics_v10230 import (
    fatigue_integrator_mode_from_environment,
)


def test_default_mode_is_accelerated():
    assert fatigue_integrator_mode_from_environment({}) == "accelerated"


def test_explicit_mode_selected_via_environment():
    assert fatigue_integrator_mode_from_environment(
        {"V10230_FATIGUE_INTEGRATOR_MODE": "explicit"}
    ) == "explicit"


def test_unknown_mode_raises():
    with pytest.raises(ValueError, match="Unknown V10230_FATIGUE_INTEGRATOR_MODE"):
        fatigue_integrator_mode_from_environment({"V10230_FATIGUE_INTEGRATOR_MODE": "bogus"})


def test_default_env_still_raises_the_acceleration_gate_when_rebonding_enabled(monkeypatch):
    """Regression-pins the pre-existing behavior in
    test_v10_2_30_crack_rebonding_acceleration_gate.py -- not setting
    V10230_FATIGUE_INTEGRATOR_MODE at all must be byte-identical to before
    this selector existed."""
    monkeypatch.setenv("V10230_ENERGY_GATE_ENABLED", "1")
    monkeypatch.setenv("V10230_CRACK_REBONDING_ENABLED", "1")
    monkeypatch.setenv("V10230_CRACK_REBONDING_MODEL_LEVEL", "CLEAN_REVERSIBLE_REBOND")
    monkeypatch.delenv("V10230_FATIGUE_INTEGRATOR_MODE", raising=False)

    original_coupled_commit = entry._coupled_commit.integrate_state_coupled_waveform
    monkeypatch.setattr(entry._v10229, "main", lambda args: "unreachable")

    with pytest.raises(RuntimeError, match="rebonding_acceleration_qualified"):
        entry.main(["--fatigue-cycles", "--out", "/tmp/does-not-matter"])

    assert entry._coupled_commit.integrate_state_coupled_waveform is original_coupled_commit


def test_explicit_mode_with_rebonding_enabled_does_not_raise_and_skips_the_accelerated_swap(monkeypatch):
    monkeypatch.setenv("V10230_ENERGY_GATE_ENABLED", "1")
    monkeypatch.setenv("V10230_CRACK_REBONDING_ENABLED", "1")
    monkeypatch.setenv("V10230_CRACK_REBONDING_MODEL_LEVEL", "CLEAN_REVERSIBLE_REBOND")
    monkeypatch.setenv("V10230_FATIGUE_INTEGRATOR_MODE", "explicit")

    original_coupled_commit = entry._coupled_commit.integrate_state_coupled_waveform
    observed_during_call = {}

    def _fake_v10229_main(args):
        # While main() is running, the swap must have been skipped: the name
        # must still point at the real, explicit, rebonding-compatible
        # integrator, not the accelerated one.
        observed_during_call["coupled_commit"] = entry._coupled_commit.integrate_state_coupled_waveform
        return "ok"

    monkeypatch.setattr(entry._v10229, "main", _fake_v10229_main)

    result = entry.main(["--fatigue-cycles", "--out", ""])

    assert result == "ok"
    assert observed_during_call["coupled_commit"] is original_coupled_commit
    assert observed_during_call["coupled_commit"] is not entry._high_cycle.integrate_state_coupled_waveform
    # Restored (idempotently) after the call either way.
    assert entry._coupled_commit.integrate_state_coupled_waveform is original_coupled_commit


def test_explicit_mode_still_installs_every_other_monkeypatch(monkeypatch):
    """Opting out of the accelerated-integrator swap must not skip any of
    the other, unrelated monkeypatches -- confirms this is a narrowly
    scoped, additive change, not a broader bypass."""
    monkeypatch.setenv("V10230_ENERGY_GATE_ENABLED", "1")
    monkeypatch.setenv("V10230_CRACK_REBONDING_ENABLED", "1")
    monkeypatch.setenv("V10230_CRACK_REBONDING_MODEL_LEVEL", "CLEAN_REVERSIBLE_REBOND")
    monkeypatch.setenv("V10230_FATIGUE_INTEGRATOR_MODE", "explicit")

    from arrhenius_fracture.persistent_site_cyclic_energy_gated_corrected_v10230 import (
        CorrectedHazardEnergyGatedPersistentSiteCyclicTipEngine,
    )

    observed = {}

    def _fake_v10229_main(args):
        observed["engine_class"] = entry._v10229.AuditedCoupledPersistentSiteCyclicTipEngine
        observed["build_shared_engine"] = entry._shared_state.build_shared_engine
        return "ok"

    monkeypatch.setattr(entry._v10229, "main", _fake_v10229_main)

    result = entry.main(["--fatigue-cycles", "--out", ""])

    assert result == "ok"
    assert observed["engine_class"] is CorrectedHazardEnergyGatedPersistentSiteCyclicTipEngine
    assert observed["build_shared_engine"] is not None  # rebound to the rebonding-installing wrapper
