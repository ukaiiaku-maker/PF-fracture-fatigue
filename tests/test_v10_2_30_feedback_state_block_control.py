from pathlib import Path
from types import SimpleNamespace

import pytest

from arrhenius_fracture import active_state_block_control_v10230 as active
from arrhenius_fracture import feedback_state_block_control_v10230 as feedback
from arrhenius_fracture import persistent_site_coupled_hazard_v10229 as coupled
from arrhenius_fracture import persistent_site_vhcf_selector_v10230 as selector


ROOT = Path(__file__).resolve().parents[1]


def _controller():
    return SimpleNamespace(
        cfg=SimpleNamespace(
            min_block_cycles=1.0e-6,
            target_dB=0.1,
            target_dN_emit=0.1,
            target_dN_mobile=0.1,
            target_dN_store=0.1,
            target_dN_escape=0.1,
        )
    )


def test_feedback_targets_do_not_use_raw_population_counts():
    controller = _controller()
    assert feedback.feedback_state_targets(controller) == {}
    assert feedback.feedback_block_targets(controller, 0.25) == {
        "cleavage_clock": pytest.approx(0.1),
    }


def test_feedback_clock_target_respects_remaining_first_passage_action():
    controller = _controller()
    assert feedback.feedback_block_targets(controller, 0.96) == {
        "cleavage_clock": pytest.approx(0.04),
    }


def test_install_routes_inner_and_outer_controls_after_active_patch():
    feedback.restore_feedback_state_block_control()
    active.restore_active_state_block_control()
    original_state = coupled._state_targets
    original_targets = selector._targets
    try:
        feedback.install_feedback_state_block_control()
        assert coupled._state_targets is feedback.feedback_state_targets
        assert selector._targets is feedback.feedback_block_targets
        audit = feedback.audit_payload()
        assert audit["installed"] is True
        assert audit["raw_population_counts_are_block_limiters"] is False
        assert audit["mobile_retained_state_evolved"] is True
        assert audit["persistent_source_physics_changed"] is False
    finally:
        feedback.restore_feedback_state_block_control()
        active.restore_active_state_block_control()
    assert coupled._state_targets is original_state
    assert selector._targets is original_targets


def test_zero_absolute_dB_cutoff_keeps_low_hazard_feedback_checks_active(monkeypatch):
    monkeypatch.setenv("V10229_COUPLED_HAZARD_ABS_DB_TOL", "0")
    config = coupled.coupled_hazard_config(_controller())
    assert config["absolute_dB_tol"] == pytest.approx(0.0)
    assert config["sigma_relative_tol"] > 0.0
    assert config["log_lambda_tol_decades"] > 0.0


def test_feedback_launcher_installs_patch_and_disables_absolute_bypass():
    text = (
        ROOT
        / "scripts"
        / "run_v10_2_30_300K_four_class_fatigue_feedback_state.sh"
    ).read_text()
    assert "V10230_FEEDBACK_STATE_BLOCK_CONTROL=1" in text
    assert "V10229_COUPLED_HAZARD_ABS_DB_TOL=0" in text
    assert "V10230_VHCF_RELATIVE_CYCLE_TOL" in text
    assert "run_v10_2_30_300K_four_class_fatigue.sh" in text
