import pytest

from arrhenius_fracture.fatigue_convergence_r5 import (
    execute_fatigue_convergence_r5,
)


def test_phase_and_native_resolution_are_varied_separately(tmp_path):
    result = execute_fatigue_convergence_r5(
        tmp_path / "fatigue.json", phase_counts=(12, 24),
        native_action_limits=(0.02, 0.01), maximum_cycles=60,
    )
    assert result["passed"]
    assert result["separation"]["dimensions_varied_separately"]
    assert all(row["events"] for row in result["phase_drive_refinement"])
    assert result["qualification"]["status"] == "REFINEMENT_EXECUTED"
    assert result["qualification"]["accuracy_qualified"] is False
    for arm in (
        result["phase_drive_refinement"]
        + result["native_state_step_refinement_at_24_phases"]
        + result["outer_source_refresh_refinement_at_24_phases"]
    ):
        elapsed = [row["physical_elapsed_cycles"] for row in arm["rows"]]
        assert elapsed == sorted(elapsed)
        assert arm["fractional_elapsed_cycles"] == pytest.approx(3.0)
