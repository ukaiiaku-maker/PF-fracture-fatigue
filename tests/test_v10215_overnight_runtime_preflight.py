from pathlib import Path


def test_overnight_runs_existing_2d_directly_and_preflight_is_opt_in():
    text = Path("scripts/run_v10_2_15_stage3_overnight.sh").read_text()
    assert "RUNTIME_PREFLIGHT=${RUNTIME_PREFLIGHT:-0}" in text
    assert "PHASE=runtime_preflight" in text
    assert "MODE=smoke" in text
    assert "STEPS_SMOKE=1" in text
    assert "RUNTIME_PREFLIGHT_PASSED" in text
    assert "PHASE=campaign" in text
    assert "signed atlas not used" in text
    assert "build_v10_2_14" not in text
