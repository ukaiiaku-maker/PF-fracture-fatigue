from arrhenius_fracture.piecewise_equilibrated_boundary_stress_v2_contract import (
    ACCEPTANCE_LIMITS, OPERATOR_ID, contract_record,
)


def test_piecewise_v2_contract_is_result_free_and_does_not_impose_boundary_traction():
    record = contract_record()
    assert record["operator"] == OPERATOR_ID
    assert record["prospectively_frozen_before_central_D_E_evaluation"] is True
    assert record["constraints"]["zero_cavity_traction_imposed"] is False
    assert record["central_levels"] == ["D", "E"]
    assert record["additional_production_mesh_level"] == "PROHIBITED"
    assert "result" not in record


def test_piecewise_v2_limits_preserve_five_percent_boundary_gate():
    assert ACCEPTANCE_LIMITS["normalized_equilibrated_boundary_traction"] == 0.05
    assert ACCEPTANCE_LIMITS["traction_mesh_relative_change"] == 0.05
