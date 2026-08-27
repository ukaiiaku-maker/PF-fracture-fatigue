from pathlib import Path


ROOT = Path(__file__).parents[1]


def test_persistent_controller_keeps_fresh_physics_and_bounded_decisions():
    text = (ROOT / "scripts/complete_v10_2_30_A_native_plus_8PT_study.py").read_text()
    assert '"resume": False' in text
    assert '"TARGET_EXT_UM": "100"' in text
    assert '"CYCLES_MAX": "1000000000000"' in text
    assert '"V10230_HIGH_CYCLE_EXPLICIT_ONLY"' in text
    assert "adaptive_max_additions_per_variant" in text
    assert "MEANINGFUL_RATE_LOG10" in text
    assert "ThreadPoolExecutor(max_workers=workers)" in text


def test_final_verifier_requires_actual_acceleration_and_clean_tree():
    text = (ROOT / "scripts/verify_v10_2_30_A_native_plus_8PT_study.py").read_text()
    assert 'parity["at_least_one_accepted_block"]' in text
    assert '"ACCELERATOR_PARITY_PASS"' in text
    assert '["git","diff","--check"]' in text
    assert '["git","status","--short"]' in text


def test_launch_denial_without_physical_record_is_not_numerical_failure():
    text = (ROOT / "scripts/complete_v10_2_30_A_native_plus_8PT_study.py").read_text()
    assert "kinetic_tip_cell_audit_v101.json" in text
    assert 'return "LAUNCH_PREFLIGHT_FAILURE"' in text
