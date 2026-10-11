from arrhenius_fracture.resolved_stochastic_fatigue_r4 import (
    execute_resolved_stochastic_fatigue_r4,
)


def test_phase_exact_stochastic_fatigue_restart(tmp_path):
    result = execute_resolved_stochastic_fatigue_r4(tmp_path / "fatigue")
    assert result["passed"]
    assert len(result["events"]) >= result["minimum_target_events"] == 4
    assert result["restart"]["checkpoint_owned_state_exact"]
    assert result["restart"]["continuation_complete_state_phase_and_events_exact"]
    assert result["remaining_phase_continued_after_event"]
    assert result["terminal_extension_m"] > 0.0
