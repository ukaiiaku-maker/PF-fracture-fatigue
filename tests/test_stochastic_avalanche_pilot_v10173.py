from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import numpy as np

import arrhenius_fracture
from arrhenius_fracture import stochastic_avalanche_tip as avalanche_tip
from arrhenius_fracture.crack_backend import CrackAdvanceResult
from arrhenius_fracture.stochastic_avalanche_backend import AvalancheSubsegmentBackend
from arrhenius_fracture.stochastic_avalanche_tip import (
    clipped_exponential_mean,
    threshold_event_length_factor,
)


ROOT = Path(__file__).resolve().parents[1]
ENGINE = ROOT / "arrhenius_fracture" / "stochastic_avalanche_tip.py"
BACKEND = ROOT / "arrhenius_fracture" / "stochastic_avalanche_backend.py"
ENTRY = ROOT / "arrhenius_fracture" / "sharp_front_v10_1_7_3.py"
RUNNER = ROOT / "scripts" / "run_v10_1_7_3_stochastic_avalanche_pilot.sh"
ANALYZER = ROOT / "scripts" / "analyze_v10_1_7_3_stochastic_avalanche_pilot.py"


def _analyzer_module():
    spec = spec_from_file_location("v10173_analyzer", ANALYZER)
    assert spec is not None and spec.loader is not None
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_version_is_v10173():
    assert arrhenius_fracture.__version__ == "10.1.7.3"


def test_clipped_exponential_mean_matches_sampling():
    a, b = 0.5, 4.0
    rng = np.random.default_rng(9317)
    sample = np.clip(rng.exponential(1.0, size=500000), a, b)
    exact = clipped_exponential_mean(a, b)
    assert abs(float(np.mean(sample)) - exact) < 0.005


def test_threshold_scaled_lengths_are_bounded_and_mean_preserving():
    a, b = 0.5, 4.0
    rng = np.random.default_rng(7721)
    thresholds = rng.exponential(1.0, size=200000)
    factors = np.asarray([
        threshold_event_length_factor(x, "threshold_scaled", a, b)
        for x in thresholds
    ])
    normalization = clipped_exponential_mean(a, b)
    assert np.min(factors) >= a / normalization - 1.0e-14
    assert np.max(factors) <= b / normalization + 1.0e-14
    assert abs(float(np.mean(factors)) - 1.0) < 0.01


def test_deterministic_and_fixed_modes_recover_exact_unit_length():
    assert threshold_event_length_factor(
        0.01, "threshold_scaled", 0.5, 4.0, deterministic_threshold=True
    ) == 1.0
    assert threshold_event_length_factor(3.0, "fixed", 0.5, 4.0) == 1.0


def test_engine_correlates_waiting_threshold_and_event_reward_without_K_noise():
    text = ENGINE.read_text()
    assert "event_advance_m" in text
    assert "threshold_event_length_factor" in text
    assert "mean checkpoint length" in text
    assert '"avalanche_noise_added_to_K": False' in text
    assert '"avalanche_noise_added_to_barriers": False' in text
    assert "self.f.da = event_length" in text
    assert "_synchronize_driver_checkpoint_length" in text


def test_backend_uses_one_checked_commit_and_preserves_sharp_wake_identity():
    text = BACKEND.read_text()
    assert 'name = "sharp_wake"' in text
    assert 'diagnostic_name = "stochastic_avalanche_event"' in text
    assert '"geometry_realization": "single_checked_outer_commit"' in text
    assert '"realized_geometry_commits": 1' in text
    assert '"tip_following_remeshing_preserved": True' in text
    assert '"driver_endpoint_synchronized": True' in text
    assert "driver_endpoint[...] = endpoint" in text
    assert "write_last_avalanche_backend_diagnostics" in text
    assert "event_length_mismatch" in text
    assert "for index in range(n_segments)" not in text
    assert "Repeated calls without a FEM solve" in text


def test_backend_synchronizes_driver_endpoint_to_variable_event_length(monkeypatch):
    monkeypatch.setenv("CLEAVAGE_HAZARD_MODE", "exponential")
    monkeypatch.setenv("CLEAVAGE_EVENT_LENGTH_MODE", "threshold_scaled")

    class DummySharpWake:
        name = "sharp_wake"

        def __init__(self):
            self.cohesive_network = None
            self.advance_log = []

        def advance(self, **kwargs):
            p0 = np.asarray(kwargs["p0"], dtype=float)
            p1 = np.asarray(kwargs["p1"], dtype=float)
            moved = float(np.linalg.norm(p1 - p0))
            self.advance_log.append({
                "x0": float(p0[0]),
                "y0": float(p0[1]),
                "x1": float(p1[0]),
                "y1": float(p1[1]),
            })
            return CrackAdvanceResult(
                mesh=kwargs["mesh"],
                boundary=kwargs["boundary"],
                damage=np.asarray(kwargs["damage"], dtype=float).copy(),
                displacement=np.asarray(kwargs["displacement"], dtype=float).copy(),
                moved=moved,
                inserted=True,
                reason="ok",
            )

    avalanche_tip.clear_pending_geometry_events()
    avalanche_tip._PENDING_GEOMETRY_EVENTS.append({
        "event_advance_m": 8.0e-6,
        "event_length_factor": 1.6,
        "threshold_action": 1.6,
        "hazard_seed": 7,
        "hazard_event_index": 0,
        "geometry_subsegment_fraction": 0.1,
    })
    try:
        driver_endpoint = np.array([5.0e-6, 0.0])
        backend = AvalancheSubsegmentBackend(DummySharpWake())
        result = backend.advance(
            mesh=object(),
            boundary=object(),
            damage=np.zeros(1),
            displacement=np.zeros(1),
            p0=np.zeros(2),
            p1=driver_endpoint,
            direction=np.array([1.0, 0.0]),
            front_id=0,
            kill_r=1.0e-6,
        )
        assert result.inserted
        assert np.isclose(result.moved, 8.0e-6)
        assert np.allclose(driver_endpoint, [8.0e-6, 0.0])
        assert backend.advance_log[-1]["driver_endpoint_synchronized"] is True
        assert backend.advance_log[-1]["driver_endpoint_sync_error_m"] <= 1.0e-15
    finally:
        avalanche_tip.clear_pending_geometry_events()


def test_entry_patches_builder_writes_diagnostics_and_records_event_semantics():
    text = ENTRY.read_text()
    assert "StochasticAvalancheDiagnosticTipEngine" in text
    assert "CLEAVAGE_EVENT_LENGTH_MODE" in text
    assert "CLEAVAGE_EVENT_MIN_FACTOR" in text
    assert "CLEAVAGE_EVENT_MAX_FACTOR" in text
    assert "CLEAVAGE_EVENT_SUBSEGMENT_FRACTION" in text
    assert "from . import crack_backend as _crack_backend_module" in text
    assert "original_builder = _crack_backend_module.build_crack_backend" in text
    assert "_crack_backend_module.build_crack_backend = _builder" in text
    assert "_sharp_front_base.build_crack_backend" not in text
    assert "write_last_avalanche_backend_diagnostics" in text
    assert "_rewrite_summary_event_semantics" in text
    assert '"n_geometry_events"' in text
    assert '"n_equivalent_checkpoints_exact"' in text
    assert '"n_advances_semantics": "rounded_path_length_over_nominal_checkpoint"' in text
    assert '"geometry_subsegments_re_equilibrated": False' in text
    assert '"backend_semantic_identity_preserved": True' in text
    assert '"tip_following_remeshing_preserved": True' in text
    assert '"noise_added_to_K": False' in text


def test_runner_includes_controls_live_reporting_and_consistency_validation():
    text = RUNNER.read_text()
    assert 'SEEDS=${SEEDS:-"1 2"}' in text
    assert 'TARGET_EXT_UM=${TARGET_EXT_UM:-200}' in text
    assert 'EVENT_MIN_FACTOR=${EVENT_MIN_FACTOR:-0.5}' in text
    assert 'EVENT_MAX_FACTOR=${EVENT_MAX_FACTOR:-4.0}' in text
    assert 'EVENT_SUBSEGMENT_FRACTION=${EVENT_SUBSEGMENT_FRACTION:-0.1}' in text
    assert 'HEARTBEAT_SECONDS=${HEARTBEAT_SECONDS:-60}' in text
    assert "PYTHONUNBUFFERED=1" in text
    assert '"$PYTHON_BIN" -u -m' in text
    assert "HEARTBEAT case=" in text
    assert "CASE COMPLETE" in text
    assert "CAMPAIGN COMPLETE" in text
    assert "fixed_original" in text
    assert "segmented_deterministic" in text
    assert "stochastic_avalanche" in text
    assert "--crack-backend sharp_wake" in text
    assert 'header.index("crack_extension_m")' in text
    assert "projected_extension_m" in text
    assert "geometry_path_m" in text
    assert 'event.get("driver_endpoint_synchronized") is True' in text
    assert "len(geometry) == len(fired)" in text
    assert "equivalent_exact = geometry_path_m / da_m" in text
    assert 'int(row.get("n_advances", -1)) == equivalent_rounded' in text
    assert '"n_geometry_events": int(len(geometry))' in text
    assert "No FEM rerun is needed" in text
    assert 'int(row.get("n_advances", 0)) == len(geometry)' not in text
    assert "equivalent_checkpoints=" in text
    assert "VALIDATION " in text


def test_analyzer_separates_geometry_wrapper_bias_from_stochastic_decorrelation():
    text = ANALYZER.read_text()
    assert "segmented_control_normalized_rms_percent_of_fixed_range" in text
    assert "mean_detrended_seed_correlation_to_segmented_deterministic" in text
    assert "event_length_mean_within_20_percent" in text
    assert "geometry_waveform_decorrelated" in text
    assert "stochastic_avalanche_R_curve_ensemble.png" in text
    module = _analyzer_module()
    a = np.array([1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 1.0])
    assert module._detrended_corr(a, a) > 0.99
