from pathlib import Path


def test_overnight_preflights_all_four_runtime_grids_before_full_campaign():
    text = Path("scripts/run_v10_2_15_stage3_overnight.sh").read_text()
    assert "PHASE=runtime_preflight" in text
    assert "MODE=smoke" in text
    assert "STEPS_SMOKE=1" in text
    assert "RUNTIME_PREFLIGHT_PASSED" in text
    preflight = text.index("PHASE=runtime_preflight")
    campaign = text.index("PHASE=campaign")
    assert preflight < campaign
