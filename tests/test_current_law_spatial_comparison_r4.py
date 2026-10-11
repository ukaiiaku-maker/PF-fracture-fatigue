from arrhenius_fracture.current_law_spatial_comparison_r4 import (
    execute_current_law_comparison,
)


def test_fresh_spatial_current_law_comparison(tmp_path):
    result = execute_current_law_comparison(
        "integration_evidence/r4_same_engine_coupled_spatial/current_law_spatial",
        tmp_path / "comparison.json",
    )
    assert result["passed"]
    assert result["spatial_summary"]["n_geometry_events"] >= 2
    assert result["qualifications"]["positive_unload_reload_executed"]
    assert not result["qualifications"]["spatial_sigma_cap_active_any_row"]
    assert result["prescribed_drive_replay"]["restart"]["bitwise_complete_state_equal"]
