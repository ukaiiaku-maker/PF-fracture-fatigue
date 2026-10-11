import math

from arrhenius_fracture.r2_autonomous_reduced_fatigue import run_autonomous_reduced_fatigue


def test_autonomous_fatigue_events_restarts_and_returns(tmp_path):
    result = run_autonomous_reduced_fatigue(tmp_path / "autonomous")
    assert result["event_count"] == 1
    assert result["event_cycle"] == 9
    assert math.isclose(result["event_advance_m"], 5.0e-6, rel_tol=1.0e-15)
    assert result["checkpoint_exact"] is True
    assert result["restart_continuation_exact"] is True
    assert result["post_event_return_reload"]["negative_local_drive"] is True
    assert result["post_event_return_reload"]["physical_returned"] > 0.0
    assert result["future_spatial_schedule_consumed"] is False
    assert result["spatial_comparison"]["same_external_loading_and_temperature"] is False
