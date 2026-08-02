import copy
from types import SimpleNamespace

import numpy as np

from arrhenius_fracture import persistent_site_vhcf_selector_v10230 as selector
from arrhenius_fracture.persistent_site_cyclic_energy_gated_corrected_v10230 import (
    CorrectedHazardEnergyGatedPersistentSiteCyclicTipEngine,
)
from arrhenius_fracture import persistent_site_trial_clone_v10230 as trial_clone


def _controller(cap=1.0e9):
    return SimpleNamespace(
        cfg=SimpleNamespace(
            block_cycles=cap,
            max_block_cycles=cap,
            min_block_cycles=1.0e-6,
            target_dB=0.1,
            target_dN_store=10.0,
            target_dN_emit=10.0,
            target_dN_mobile=10.0,
            target_dN_escape=10.0,
            cycle_block_mode="hazard_limited",
        )
    )


def _prediction(engine):
    pred = SimpleNamespace()
    return selector.attach_prediction_context(
        pred,
        engine,
        SimpleNamespace(),
        300.0,
    )


def _trial(cycles, emitted):
    return {
        "cycles_requested": float(cycles),
        "cycles_consumed": float(cycles),
        "fired": False,
        "metrics": {
            "cleavage_clock": 0.0,
            "emitted_pz": float(emitted),
            "stored_pz": 0.0,
            "mobile_pz": float(emitted),
            "escape_pz": 0.0,
        },
        "internal_steps": 1,
    }


def test_selector_brackets_from_linear_seed_instead_of_full_horizon(monkeypatch):
    cap = 1.0e9
    engine = SimpleNamespace(persistent_site_cyclic_v10229=True, B=0.0)
    calls = []

    def fake_trial(_engine, _wave, _temperature, _prediction, cycles):
        calls.append(float(cycles))
        return _trial(cycles, emitted=float(cycles) / 1000.0)

    monkeypatch.setattr(selector, "_block_trial", fake_trial)
    result = selector.select_nonlinear_block(
        _controller(cap),
        _prediction(engine),
        cap,
        {"cycles": 100.0, "limiter": "emitted_pz"},
    )

    assert calls[0] == 100.0
    assert cap not in calls
    assert np.isclose(result["cycles"], 1.0e4, rtol=2.0e-6)
    audit = engine._v10229_last_vhcf_block_audit
    assert audit["search_strategy"] == "linear_seed_geometric_bracket_log_bisection"
    assert audit["full_cap_was_first_trial"] is False
    assert audit["cap_evaluated"] is False


def test_selector_reaches_cap_for_self_blocking_stationary_state(monkeypatch):
    cap = 1.0e9
    engine = SimpleNamespace(persistent_site_cyclic_v10229=True, B=0.0)
    calls = []

    def fake_trial(_engine, _wave, _temperature, _prediction, cycles):
        calls.append(float(cycles))
        emitted = 8.0 * (1.0 - np.exp(-float(cycles) / 1.0e6))
        return _trial(cycles, emitted=emitted)

    monkeypatch.setattr(selector, "_block_trial", fake_trial)
    result = selector.select_nonlinear_block(
        _controller(cap),
        _prediction(engine),
        cap,
        {"cycles": 10.0, "limiter": "emitted_pz"},
    )

    assert result["cycles"] == cap
    assert result["limiter"] == "max_block_cycles"
    assert calls[0] == 10.0
    assert calls[-1] == cap
    assert len(calls) < 20


def test_fast_trial_clone_preserves_rng_state_and_independence():
    engine = object.__new__(CorrectedHazardEnergyGatedPersistentSiteCyclicTipEngine)
    engine._hazard_rng = np.random.default_rng(1720)
    engine._energy_gate_provisional = False
    engine._energy_gate_pending = {"pending": True}
    engine.scalar = 3.0

    cloned = trial_clone.fast_trial_deepcopy(engine, {})

    assert cloned is not engine
    assert cloned._hazard_rng is not engine._hazard_rng
    assert cloned._energy_gate_provisional is True
    assert cloned._energy_gate_pending is None
    assert np.array_equal(
        cloned._hazard_rng.random(8),
        engine._hazard_rng.random(8),
    )


def test_fast_trial_clone_install_and_restore():
    original = CorrectedHazardEnergyGatedPersistentSiteCyclicTipEngine.__deepcopy__
    trial_clone.install_fast_trial_clone()
    try:
        assert (
            CorrectedHazardEnergyGatedPersistentSiteCyclicTipEngine.__deepcopy__
            is trial_clone.fast_trial_deepcopy
        )
    finally:
        trial_clone.restore_fast_trial_clone()
    assert CorrectedHazardEnergyGatedPersistentSiteCyclicTipEngine.__deepcopy__ is original


def test_entry_installs_high_cycle_engine_and_fast_clone(monkeypatch, tmp_path):
    from arrhenius_fracture import fatigue_controller_delegate_v10229 as delegate
    from arrhenius_fracture import persistent_site_cyclic_coupled_v10229 as coupled_commit
    from arrhenius_fracture import persistent_site_high_cycle_engine_v10230 as high_cycle
    from arrhenius_fracture import persistent_site_forward_selector_v10230 as forward_selector
    from arrhenius_fracture import sharp_front_v10_2_29_fatigue_audited as v10229
    from arrhenius_fracture import sharp_front_v10_2_30_energy_gated_fatigue as entry

    observed = {}
    original_attach = delegate.attach_prediction_context
    original_select = delegate.select_nonlinear_block
    original_commit = coupled_commit.integrate_state_coupled_waveform
    original_deepcopy = CorrectedHazardEnergyGatedPersistentSiteCyclicTipEngine.__deepcopy__

    def fake_main(args):
        observed["attach"] = delegate.attach_prediction_context
        observed["select"] = delegate.select_nonlinear_block
        observed["commit"] = coupled_commit.integrate_state_coupled_waveform
        observed["deepcopy"] = (
            CorrectedHazardEnergyGatedPersistentSiteCyclicTipEngine.__deepcopy__
        )
        return "ok"

    monkeypatch.setattr(v10229, "main", fake_main)
    monkeypatch.setattr(entry, "write_last_energy_gate_diagnostics", lambda out: None)
    monkeypatch.setattr(entry, "_write_audit", lambda args: None)

    result = entry.main(["--fatigue-cycles", "--out", str(tmp_path)])
    assert result == "ok"
    assert observed["attach"] is forward_selector.attach_prediction_context
    assert observed["select"] is forward_selector.select_nonlinear_block
    assert observed["commit"] is high_cycle.integrate_state_coupled_waveform
    assert observed["deepcopy"] is trial_clone.fast_trial_deepcopy
    assert delegate.attach_prediction_context is original_attach
    assert delegate.select_nonlinear_block is original_select
    assert coupled_commit.integrate_state_coupled_waveform is original_commit
    assert CorrectedHazardEnergyGatedPersistentSiteCyclicTipEngine.__deepcopy__ is original_deepcopy


def test_room_temperature_wrapper_enforces_300K_four_class_scope():
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    text = (root / "scripts" / "run_v10_2_30_300K_four_class_fatigue.sh").read_text()
    assert '"${TEMPERATURE_K}" != "300"' in text
    assert "export TEMPERATURE_K=300" in text
    assert "run_v10_2_30_four_class_three_deltaK_energy_gate_qualification.sh" in text
