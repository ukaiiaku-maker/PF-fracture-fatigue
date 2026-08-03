from pathlib import Path


def _root() -> Path:
    return Path(__file__).resolve().parents[1]


def _text() -> str:
    return (_root() / "scripts" / "run_v10_2_30_weakt_0p55_high_cycle_1e12.sh").read_text()


def _generic_text() -> str:
    return (_root() / "scripts" / "run_v10_2_30_weakt_high_cycle_1e12.sh").read_text()


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


def test_generic_launcher_temp_script_cannot_dirty_worktree():
    text = _generic_text()
    assert 'mktemp "${TMPDIR:-/tmp}/v10_2_30_generic_weakt.XXXXXX"' in text
    assert '$ROOT/scripts/.v10_2_30_generic_weakt' not in text
    assert "requires the XXXXXX template at the end" in text
    assert 'rm -f -- "$GENERATED_LAUNCHER"' in text


def test_generic_launcher_binds_generated_copy_to_repository_root():
    text = _generic_text()
    assert '"$RUN_LABEL" "$ROOT" <<\'PY\'' in text
    assert "import shlex" in text
    assert "repo_root = sys.argv[4]" in text
    assert 'root_line = \'ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)\'' in text
    assert 'text.replace(root_line, f"ROOT={shlex.quote(repo_root)}", 1)' in text
    assert "low-level launcher ROOT contract changed" in text


def test_generic_launcher_stops_before_postprocessing_missing_output():
    text = _generic_text()
    guard = 'if [[ ! -d "$OUTROOT" ]]; then'
    postprocess = 'manifest_path = root / "high_cycle_run_manifest.json"'
    assert guard in text
    assert postprocess in text
    assert text.index(guard) < text.index(postprocess)


def test_generic_launcher_records_stochastic_first_passage_threshold():
    text = _generic_text()
    assert "HAZARD_SEED=${HAZARD_SEED:-2001726}" in text
    assert 'threshold_is_stochastic": True' in text
    assert 'unit_exponential_in_cumulative_hazard_action' in text
    assert 'independent_draw_per_first_passage_interval' in text
    assert 'current_interval_physical_hazard_action' in text
    assert 'current_interval_sampled_threshold' in text
