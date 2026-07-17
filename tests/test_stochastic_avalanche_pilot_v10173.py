from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import numpy as np

import arrhenius_fracture
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
    a, b = 0.2, 4.0
    rng = np.random.default_rng(9317)
    sample = np.clip(rng.exponential(1.0, size=500000), a, b)
    exact = clipped_exponential_mean(a, b)
    assert abs(float(np.mean(sample)) - exact) < 0.005


def test_threshold_scaled_lengths_are_bounded_and_mean_preserving():
    a, b = 0.2, 4.0
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
        0.01, "threshold_scaled", 0.2, 4.0, deterministic_threshold=True
    ) == 1.0
    assert threshold_event_length_factor(3.0, "fixed", 0.2, 4.0) == 1.0


def test_engine_correlates_waiting_threshold_and_event_reward_without_K_noise():
    text = ENGINE.read_text()
    assert "event_advance_m" in text
    assert "threshold_event_length_factor" in text
    assert "mean checkpoint length" in text
    assert '"avalanche_noise_added_to_K": False' in text
    assert '"avalanche_noise_added_to_barriers": False' in text
    assert "self.f.da = event_length" in text


def test_backend_realizes_each_event_as_ten_equal_geometry_subsegments():
    text = BACKEND.read_text()
    assert "n_segments = max(int(math.ceil(1.0 / fraction)), 1)" in text
    assert "segment_length = total_length / n_segments" in text
    assert "mechanics_re_equilibrated_between_subsegments" in text
    assert 'name = "stochastic_avalanche_segmented"' in text


def test_entry_patches_crack_backend_defining_module_and_records_limits():
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
    assert '"geometry_subsegments_re_equilibrated": False' in text
    assert '"noise_added_to_K": False' in text


def test_runner_includes_fixed_segmented_and_stochastic_controls():
    text = RUNNER.read_text()
    assert 'SEEDS=${SEEDS:-"1 2"}' in text
    assert 'TARGET_EXT_UM=${TARGET_EXT_UM:-200}' in text
    assert 'EVENT_MIN_FACTOR=${EVENT_MIN_FACTOR:-0.2}' in text
    assert 'EVENT_MAX_FACTOR=${EVENT_MAX_FACTOR:-4.0}' in text
    assert 'EVENT_SUBSEGMENT_FRACTION=${EVENT_SUBSEGMENT_FRACTION:-0.1}' in text
    assert "fixed_original" in text
    assert "segmented_deterministic" in text
    assert "stochastic_avalanche" in text
    assert "--crack-backend sharp_wake" in text


def test_analyzer_separates_segmentation_bias_from_stochastic_decorrelation():
    text = ANALYZER.read_text()
    assert "segmented_control_normalized_rms_percent_of_fixed_range" in text
    assert "mean_detrended_seed_correlation_to_segmented_deterministic" in text
    assert "event_length_mean_within_20_percent" in text
    assert "geometry_waveform_decorrelated" in text
    assert "stochastic_avalanche_R_curve_ensemble.png" in text
    module = _analyzer_module()
    a = np.array([1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 1.0])
    assert module._detrended_corr(a, a) > 0.99
