from pathlib import Path

import numpy as np

import arrhenius_fracture
from arrhenius_fracture.stochastic_hazard_tip import (
    draw_hazard_threshold,
    normalized_progress_rate,
)


ROOT = Path(__file__).resolve().parents[1]
ENGINE = ROOT / "arrhenius_fracture" / "stochastic_hazard_tip.py"
ENTRY = ROOT / "arrhenius_fracture" / "sharp_front_v10_1_7_2.py"
RUNNER = ROOT / "scripts" / "run_v10_1_7_2_stochastic_hazard_pilot.sh"
ANALYZER = ROOT / "scripts" / "analyze_v10_1_7_2_stochastic_hazard_pilot.py"


def test_version_is_v10172():
    assert arrhenius_fracture.__version__ == "10.1.7.3"


def test_deterministic_threshold_is_exactly_one():
    rng = np.random.default_rng(123)
    assert draw_hazard_threshold("deterministic", rng) == 1.0
    assert normalized_progress_rate(5.0, 1.0) == 5.0


def test_exponential_thresholds_are_reproducible_and_unit_mean():
    rng1 = np.random.default_rng(9821)
    rng2 = np.random.default_rng(9821)
    a = np.asarray([draw_hazard_threshold("exponential", rng1) for _ in range(50000)])
    b = np.asarray([draw_hazard_threshold("exponential", rng2) for _ in range(50000)])
    assert np.array_equal(a, b)
    assert np.all(a > 0.0)
    assert abs(float(np.mean(a)) - 1.0) < 0.02


def test_progress_normalization_reaches_fixed_checkpoint_at_sampled_action():
    threshold = 0.37
    lam = 2.5
    waiting_time = threshold / lam
    progress = normalized_progress_rate(lam, threshold) * waiting_time
    assert abs(progress - 1.0) < 1.0e-14


def test_engine_randomizes_hazard_threshold_not_K_or_barriers():
    text = ENGINE.read_text()
    assert "rng.exponential(1.0)" in text
    assert "noise is added to K" in text
    assert "noise_added_to_K\": False" in text
    assert "noise_added_to_barriers\": False" in text
    assert "lambda_c / Xi" in text
    assert "dH = dB * threshold" in text
    assert "da = float(self.f.da)" in text


def test_entry_routes_only_the_versioned_process_to_stochastic_engine():
    text = ENTRY.read_text()
    assert "StochasticHazardDiagnosticTipEngine" in text
    assert "CLEAVAGE_HAZARD_MODE" in text
    assert "CLEAVAGE_HAZARD_SEED" in text
    assert '"constitutive_change_from_v10_1_7_1": False' in text
    assert '"noise_added_to_K": False' in text


def test_runner_is_dbtt_700K_ten_seed_200um_pilot():
    text = RUNNER.read_text()
    assert 'CLASS=${CLASS:-DBTT}' in text
    assert 'TEMP_K=${TEMP_K:-700}' in text
    assert 'SEEDS=${SEEDS:-"1 2 3 4 5 6 7 8 9 10"}' in text
    assert 'TARGET_EXT_UM=${TARGET_EXT_UM:-200}' in text
    assert "run_case deterministic 0" in text
    assert "CLEAVAGE_HAZARD_MODE=\"$mode\"" in text
    assert "--max-fronts 1" in text
    assert "--no-wake-shielding" in text


def test_analyzer_requires_mean_preservation_and_nonzero_scatter():
    text = ANALYZER.read_text()
    assert "ensemble_mean_within_5_percent" in text
    assert "visible_nonzero_band" in text
    assert "seed_paths_not_identical" in text
    assert "stochastic_hazard_R_curve_ensemble.png" in text
    assert "stochastic_hazard_threshold_distribution.png" in text
