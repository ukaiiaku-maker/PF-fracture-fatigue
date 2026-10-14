from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "run_taylor_completion_native_consequence.py"


def test_consequence_runner_is_diagnostic_only_and_uses_exact_historical_callable():
    text = SCRIPT.read_text()
    assert "historical_legacy_implicit_1s" in text
    assert "DIAGNOSTIC_ONLY_STATE_TRANSFER_NOT_PRODUCTION_SELECTABLE" in text
    assert "production_environment_selector\": None" in text
    assert "MPZ_TAYLOR_RELEASE_MODE" not in text
    assert "restart_from=SOURCE_CHECKPOINT" in text


def test_consequence_runner_executes_arms_in_isolated_processes():
    text = SCRIPT.read_text()
    assert "subprocess.run(" in text
    assert '"current_native"' in text
    assert '"historical_completion_diagnostic"' in text
    assert '"active_vector"' in text
