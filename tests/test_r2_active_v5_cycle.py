from arrhenius_fracture.r2_active_v5_cycle import run_active_v5_cycle


def test_active_v5_positive_r_cycle_changes_fields_and_restarts_exactly(tmp_path):
    result = run_active_v5_cycle(tmp_path / "active-v5")
    assert result["active_cavity"] is True
    assert result["load_factors"] == [1.0, 0.2, 1.0]
    assert result["field_feedback_exercised"] is True
    assert result["midpoint_restart_exact"] is True
    assert result["restart_continuation_exact"] is True
    assert result["production_eligible_material_evidence"] is False
