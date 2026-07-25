from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "plot_v10_2_27_paper_four_class_J_vs_temperature.py"
SPEC = importlib.util.spec_from_file_location("v10227_j_vs_t", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_kj_to_j_plane_strain_conversion_is_elastic_configurational_only():
    effective_modulus = 1.0e9
    values = MODULE.kj_to_j_kj_m2(np.array([1.0e6, 2.0e6]), effective_modulus)
    assert np.allclose(values, [1.0, 4.0])


def test_extension_weighted_average_piecewise_constant_j_curve():
    pre = np.array([0.0, 500.0])
    post = np.array([500.0, 1000.0])
    j_values = np.array([1.0, 4.0])
    value = MODULE.extension_weighted_average(
        pre, post, j_values, 1000.0, 1000.0
    )
    assert np.isclose(value, 2.5)


def test_j_average_rejects_incomplete_target():
    pre = np.array([0.0])
    post = np.array([500.0])
    j_values = np.array([1.0])
    value = MODULE.extension_weighted_average(
        pre, post, j_values, 1000.0, 500.0
    )
    assert np.isnan(value)


def test_validated_runner_does_not_auto_invoke_elastic_j_conversion():
    runner = (
        ROOT / "scripts" / "run_v10_2_27_paper_four_class_30deg_long_rcurves_validated.sh"
    ).read_text()
    assert "plot_v10_2_27_paper_four_class_J_vs_temperature.py" not in runner
    assert "plot_v10_2_27_paper_four_class_K_vs_temperature.py" in runner
