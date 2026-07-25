from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "plot_v10_2_27_paper_four_class_K_vs_temperature.py"
SPEC = importlib.util.spec_from_file_location("v10227_k_vs_t", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_extension_weighted_average_constant_curve():
    pre = np.array([0.0, 250.0, 500.0, 750.0])
    post = np.array([250.0, 500.0, 750.0, 1000.0])
    resistance = np.array([12.0, 12.0, 12.0, 12.0])
    value = MODULE.extension_weighted_average(
        pre, post, resistance, 1000.0, 1000.0
    )
    assert np.isclose(value, 12.0)


def test_extension_weighted_average_piecewise_constant_curve():
    pre = np.array([0.0, 500.0])
    post = np.array([500.0, 1000.0])
    resistance = np.array([10.0, 20.0])
    value = MODULE.extension_weighted_average(
        pre, post, resistance, 1000.0, 1000.0
    )
    assert np.isclose(value, 15.0)


def test_average_rejects_incomplete_target():
    pre = np.array([0.0])
    post = np.array([500.0])
    resistance = np.array([10.0])
    value = MODULE.extension_weighted_average(
        pre, post, resistance, 1000.0, 500.0
    )
    assert np.isnan(value)


def test_validated_runner_invokes_temperature_postprocessor():
    runner = (
        ROOT / "scripts" / "run_v10_2_27_paper_four_class_30deg_long_rcurves_validated.sh"
    ).read_text()
    assert "plot_v10_2_27_paper_four_class_K_vs_temperature.py" in runner
