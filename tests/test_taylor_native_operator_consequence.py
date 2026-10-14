from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "build_taylor_native_operator_consequence.py"


def test_native_operator_consequence_uses_accepted_spatial_state_and_exact_operator():
    text = SCRIPT.read_text()
    assert "refinement_step46_checkpoint" in text
    assert "joint_transport_exchange(" in text
    assert "historical_legacy_implicit_1s(" in text
    assert '"mobile_source_rate": None' in text
    assert '"source_exclusion_reason"' in text


def test_native_operator_consequence_has_strict_state_and_conservation_gates():
    text = SCRIPT.read_text()
    assert "FAIL_NO_ACTIVE_CONSEQUENCE" in text
    assert "FAIL_CURRENT_CONSERVATION" in text
    assert "FAIL_HISTORICAL_CONSERVATION" in text
    assert "FAIL_CURRENT_SOLVER_PATH" in text
    assert "FAIL_HISTORICAL_SOLVER_PATH" in text
    assert "FAIL_CURRENT_NUMERICAL_CORRECTION" in text
    assert "FAIL_HISTORICAL_NUMERICAL_CORRECTION" in text
    assert "failed_predecessors" in text


def test_native_operator_consequence_bounds_activity_without_changing_rates():
    text = SCRIPT.read_text()
    assert "TARGET_MAX_ACTIVITY = 1.0e6" in text
    assert "TARGET_MAX_ACTIVITY / maximum_activity_rate" in text
    assert '"diagnostic_interval_policy"' in text
    assert "current_activity_rate" in text
    assert "historical_activity_rate" in text
