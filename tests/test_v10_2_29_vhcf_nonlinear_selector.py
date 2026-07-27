from types import SimpleNamespace

import numpy as np

from arrhenius_fracture import fatigue_controller_delegate_v10229 as delegate
from arrhenius_fracture.fatigue_v1 import (
    CycleHazardResult,
    FatigueControllerConfig,
    FatigueCycleHazardController,
)
from arrhenius_fracture import persistent_site_vhcf_selector_v10229 as selector


def prediction():
    return CycleHazardResult(
        mu_emit=1.0,
        mu_peierls=0.0,
        mu_taylor=0.0,
        mu_escape=0.0,
        mu_cleave=0.0,
        store_per_cycle=0.0,
        mobile_per_cycle=1.0,
        escape_per_cycle=0.0,
        peierls_per_cycle=0.0,
        taylor_per_cycle=0.0,
        avg_sigma_tip=1.0,
        max_sigma_tip=1.0,
        avg_sigma_emit_eff=1.0,
        storage_fraction=0.0,
    )


def controller(max_block=1.0e12):
    out = object.__new__(FatigueCycleHazardController)
    out.cfg = FatigueControllerConfig(
        block_cycles=max_block,
        max_block_cycles=max_block,
        min_block_cycles=1.0e-6,
        target_dB=0.1,
        target_dN_store=10.0,
        cycle_block_mode="hazard_limited",
        target_dN_emit=10.0,
        target_dN_mobile=10.0,
        target_dN_escape=10.0,
    )
    return out


def test_nonlinear_selector_accepts_horizon_after_source_self_blocks(monkeypatch):
    engine = SimpleNamespace(persistent_site_cyclic_v10229=True, B=0.0)
    wave = SimpleNamespace()
    pred = selector.attach_prediction_context(prediction(), engine, wave, 300.0)

    def fake_trial(_engine, _wave, _T, _pred, cycles):
        emitted = 8.0 * (1.0 - np.exp(-float(cycles) / 1.0e6))
        return {
            "cycles_requested": float(cycles),
            "cycles_consumed": float(cycles),
            "fired": False,
            "metrics": {
                "cleavage_clock": 0.0,
                "emitted_pz": emitted,
                "stored_pz": 0.0,
                "mobile_pz": emitted,
                "escape_pz": 0.0,
            },
            "internal_steps": 1,
        }

    monkeypatch.setattr(selector, "_block_trial", fake_trial)
    diag = selector.select_nonlinear_block(
        controller(), pred, 1.0e12, {"cycles": 10.0, "limiter": "emitted_pz"}
    )
    assert diag["cycles"] == 1.0e12
    assert diag["limiter"] == "max_block_cycles"
    assert engine._v10229_last_vhcf_block_audit["trial_evaluations"] == 1


def test_nonlinear_selector_bisects_actual_emission_increment(monkeypatch):
    engine = SimpleNamespace(persistent_site_cyclic_v10229=True, B=0.0)
    pred = selector.attach_prediction_context(
        prediction(), engine, SimpleNamespace(), 300.0
    )

    def fake_trial(_engine, _wave, _T, _pred, cycles):
        emitted = 20.0 * (1.0 - np.exp(-float(cycles) / 1.0e6))
        return {
            "cycles_requested": float(cycles),
            "cycles_consumed": float(cycles),
            "fired": False,
            "metrics": {
                "cleavage_clock": 0.0,
                "emitted_pz": emitted,
                "stored_pz": 0.0,
                "mobile_pz": emitted,
                "escape_pz": 0.0,
            },
            "internal_steps": 1,
        }

    monkeypatch.setattr(selector, "_block_trial", fake_trial)
    diag = selector.select_nonlinear_block(
        controller(), pred, 1.0e12, {"cycles": 10.0, "limiter": "emitted_pz"}
    )
    expected = np.log(2.0) * 1.0e6
    assert np.isclose(diag["cycles"], expected, rtol=2.0e-6)
    assert diag["limiter"] == "nonlinear_emitted_pz"
    assert diag["cycles"] > 1.0e4


def test_delegate_reenters_selector_under_global_cycle_cap(monkeypatch):
    class Front:
        persistent_site_cyclic_v10229 = True

        def preview_cycle_waveform(self, ctrl, waveform, temperature):
            return prediction()

        def cycle_step_waveform(
            self, ctrl, waveform, temperature, requested_cycles=None, force_cycles=None
        ):
            assert force_cycles is None
            assert requested_cycles == 123.0
            assert ctrl.cfg.max_block_cycles == 123.0
            pred = self.preview_cycle_waveform(ctrl, waveform, temperature)
            return ctrl.choose_block_cycles_diagnostic(pred, requested_cycles)

    monkeypatch.setattr(
        delegate,
        "select_nonlinear_block",
        lambda ctrl, pred, requested, linear: {
            "cycles": requested,
            "limiter": "nonlinear_test",
            "unlimited_cycles": requested,
            "candidate_limits": {"nonlinear_selected_cycles": requested},
        },
    )
    ctrl = controller(max_block=1.0e6)
    original_max = ctrl.cfg.max_block_cycles
    delegate.install_engine_native_cycle_preview()
    try:
        out = ctrl.cycle_step_front(
            Front(), SimpleNamespace(), 300.0,
            requested_cycles=123.0, force_cycles=123.0,
        )
    finally:
        delegate.restore_engine_native_cycle_preview()
    assert out["limiter"] == "nonlinear_test"
    assert ctrl.cfg.max_block_cycles == original_max
