from pathlib import Path
from types import SimpleNamespace

import pytest

from arrhenius_fracture import active_state_block_control_v10230 as patch
from arrhenius_fracture import persistent_site_coupled_hazard_v10229 as coupled
from arrhenius_fracture import persistent_site_vhcf_coupled_selector_v10230 as routed
from arrhenius_fracture import persistent_site_vhcf_selector_v10230 as selector


ROOT = Path(__file__).resolve().parents[1]


def _controller():
    return SimpleNamespace(
        cfg=SimpleNamespace(
            target_dB=0.1,
            target_dN_emit=0.1,
            target_dN_mobile=0.2,
            target_dN_store=0.3,
            target_dN_escape=0.4,
        )
    )


def test_active_targets_exclude_cumulative_flux_ledgers():
    controller = _controller()
    assert patch.active_state_targets(controller) == {
        "mobile_count": pytest.approx(0.2),
        "retained_count": pytest.approx(0.3),
    }
    assert patch.active_block_targets(controller, 0.25) == {
        "cleavage_clock": pytest.approx(0.1),
        "mobile_pz": pytest.approx(0.2),
        "stored_pz": pytest.approx(0.3),
    }


def test_flux_throughput_is_reported_but_does_not_replace_active_state(monkeypatch):
    engine = SimpleNamespace(
        mpz=SimpleNamespace(
            mobile_count=5.0,
            retained_count=7.0,
            emitted_total=11.0,
            escaped_total=13.0,
        )
    )
    waveform = SimpleNamespace(frequency_Hz=1000.0)

    def fake_integrate(trial, controller, waveform, temperature, cycles):
        trial.mpz.mobile_count += 0.02
        trial.mpz.retained_count += 0.03
        trial.mpz.emitted_total += 100.0
        trial.mpz.escaped_total += 99.0
        return {
            "dt_consumed": cycles / waveform.frequency_Hz,
            "dB": 1.0e-8,
            "plastic": {
                "dN_trapped": 40.0,
                "dN_released": 39.0,
                "dN_escaped": 99.0,
            },
            "microsteps": 2,
        }

    monkeypatch.setattr(coupled, "integrate_state_coupled_waveform", fake_integrate)
    result = patch.active_coupled_block_trial(
        _controller(),
        engine,
        waveform,
        300.0,
        SimpleNamespace(),
        1.0e6,
    )

    assert result["metrics"]["mobile_pz"] == pytest.approx(0.02)
    assert result["metrics"]["stored_pz"] == pytest.approx(0.03)
    assert result["metrics"]["emitted_pz"] == pytest.approx(100.0)
    assert result["metrics"]["escape_pz"] == pytest.approx(99.0)
    assert result["cumulative_flux_ledgers_are_block_limiters"] is False


def test_install_and_restore_routes_all_three_control_points():
    patch.restore_active_state_block_control()
    original_state = coupled._state_targets
    original_targets = selector._targets
    original_trial = routed._coupled_block_trial
    try:
        patch.install_active_state_block_control()
        assert coupled._state_targets is patch.active_state_targets
        assert selector._targets is patch.active_block_targets
        assert routed._coupled_block_trial is patch.active_coupled_block_trial
        audit = patch.audit_payload()
        assert audit["installed"] is True
        assert audit["persistent_source_physics_changed"] is False
    finally:
        patch.restore_active_state_block_control()
    assert coupled._state_targets is original_state
    assert selector._targets is original_targets
    assert routed._coupled_block_trial is original_trial


def test_active_state_launcher_uses_runtime_patch_and_practical_cycle_tolerance():
    text = (
        ROOT / "scripts" / "run_v10_2_30_300K_four_class_fatigue_active_state.sh"
    ).read_text()
    assert "V10230_ACTIVE_STATE_BLOCK_CONTROL=1" in text
    assert "V10230_VHCF_RELATIVE_CYCLE_TOL" in text
    assert "1e-4" in text
    assert "run_v10_2_30_300K_four_class_fatigue.sh" in text
