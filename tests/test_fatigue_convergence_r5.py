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
