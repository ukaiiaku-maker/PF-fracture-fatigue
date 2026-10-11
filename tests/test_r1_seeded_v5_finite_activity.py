from arrhenius_fracture.r1_seeded_v5_finite_activity import (
    run_seeded_finite_activity,
)


def test_seeded_fixture_has_native_finite_activity_conservation_and_restart(tmp_path):
    result = run_seeded_finite_activity(tmp_path / "seeded-v5")
    assert result["production_eligible_material_evidence"] is False
    assert result["natural_nucleation_claim"] is False
    assert result["rate_multiplier"] == 1.0
    assert result["terminal_cavity_radius_m"] == 3.0e-6
    assert result["terminal_cavity_geometry_generation"] > 0
    assert result["vacancy_ledgers"]["vacancy_consumed_by_void_area_m2"] > 0.0
    assert abs(result["vacancy_conservation"]["vacancy_conservation_residual_m2"]) <= 1.0e-22
    assert result["restart_exact"] is True
