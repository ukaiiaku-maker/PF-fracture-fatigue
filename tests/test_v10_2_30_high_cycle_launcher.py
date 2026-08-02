from pathlib import Path


def _text() -> str:
    root = Path(__file__).resolve().parents[1]
    return (root / "scripts" / "run_v10_2_30_weakt_0p55_high_cycle_1e12.sh").read_text()


def test_launcher_requests_native_1e12_horizon():
    text = _text()
    assert "CYCLES_MAX=${CYCLES_MAX:-1e12}" in text
    assert 'V10230_FORWARD_OUTER_PROPOSAL_CYCLES="$CYCLES_MAX"' in text
    assert '--cycles-max "$CYCLES_MAX"' in text
    assert '--block-cycles "$CYCLES_MAX"' in text
    assert '--max-block-cycles "$CYCLES_MAX"' in text


def test_launcher_enables_production_high_cycle_components():
    text = _text()
    assert "persistent_site_high_cycle_engine_v10230" in text
    assert "persistent_site_poincare_v10230" in text
    assert "persistent_site_periodic_solver_v10230" in text
    assert "persistent_site_high_cycle_propagation_v10230" in text
    assert "V10230_PROJECTIVE_MAX_CYCLES" in text
    assert "V10230_HIGH_CYCLE_STATIONARY_ADMISSION_DISTANCE" in text


def test_launcher_does_not_enable_experimental_runtime_overlays():
    text = _text()
    assert "unset V10230_ACTIVE_STATE_BLOCK_CONTROL" in text
    assert "unset V10230_FEEDBACK_STATE_BLOCK_CONTROL" in text
    assert "Gc0_athermal" not in text
