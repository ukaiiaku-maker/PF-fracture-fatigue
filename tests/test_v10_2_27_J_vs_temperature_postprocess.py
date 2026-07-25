from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np

from arrhenius_fracture.energy_ledger_output_v10227 import augment_steps_table


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = (
    ROOT
    / "scripts"
    / "plot_v10_2_27_paper_four_class_J_energy_vs_temperature.py"
)
SPEC = importlib.util.spec_from_file_location("v10227_j_energy_vs_t", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_reconstruct_external_work_uses_accepted_trapezoidal_history():
    data = np.zeros(
        2,
        dtype=[("Uapp_m", float), ("Ftop_N", float)],
    )
    data["Uapp_m"] = [1.0, 2.0]
    data["Ftop_N"] = [2.0, 4.0]
    assert np.allclose(MODULE.reconstruct_external_work(data), [1.0, 4.0])


def test_work_per_crack_area_uses_first_target_crossing():
    extension_um = np.array([0.0, 500.0, 1000.0, 1000.0])
    cumulative_work = np.array([0.0, 1.0, 2.0, 3.0])
    value = MODULE.work_per_crack_area_kj_m2(
        extension_um, cumulative_work, 1000.0
    )
    assert np.isclose(value, 2.0)


def test_extension_weighted_average_piecewise_constant_direct_j():
    pre = np.array([0.0, 500.0])
    post = np.array([500.0, 1000.0])
    values = np.array([1.0, 4.0])
    result = MODULE.extension_weighted_average(
        pre, post, values, 1000.0, 1000.0
    )
    assert np.isclose(result, 2.5)


def test_energy_output_augmentation_persists_direct_j_and_ledgers():
    steps = np.zeros((2, 15), dtype=float)
    steps[:, 0] = [1.0, 2.0]
    steps[:, 3] = [1.0e6, 2.0e6]
    hist = {
        "W_ext": [1.0, 4.0],
        "U_el": [0.4, 1.0],
        "W_p": [0.1, 0.5],
        "W_emit": [0.2, 1.0],
    }
    fronts = np.zeros((2, 19), dtype=float)
    fronts[:, 0] = [1.0, 2.0]
    fronts[:, 1] = 0.0
    fronts[:, 16] = [900.0, 1800.0]
    fronts[:, 17] = [1000.0, 2000.0]

    augmented, header, audit = augment_steps_table(
        steps,
        "step,Uapp_m,Ftop_N,KJ_Pa_sqrtm,c4,c5,c6,c7,c8,c9,c10,c11,c12,c13,c14",
        hist,
        fronts,
        1.0e9,
    )

    assert augmented.shape == (2, 22)
    assert "J_effective_direct_J_per_m2" in header
    assert np.allclose(augmented[:, 15], [1000.0, 2000.0])
    assert np.allclose(augmented[:, 16], [900.0, 1800.0])
    assert np.allclose(augmented[:, 17], [1.0, 4.0])
    assert np.allclose(augmented[:, 18], [0.4, 1.0])
    assert np.allclose(augmented[:, 19], [0.1, 0.5])
    assert np.allclose(augmented[:, 20], [0.2, 1.0])
    assert np.allclose(augmented[:, 21], [0.3, 1.5])
    assert audit["adaptive_rejected_trials_excluded"] is True


def test_legacy_full_field_without_persisted_bulk_ledger_is_not_guessed(tmp_path):
    (tmp_path / "v10_1_driver_modes.json").write_text(
        json.dumps({"bulk_plasticity_mode": "full_field"})
    )
    data = np.zeros(
        2,
        dtype=[
            ("Uapp_m", float),
            ("Ftop_N", float),
            ("W_emit_J_per_m", float),
        ],
    )
    ledgers, audit = MODULE._energy_ledgers(tmp_path, data)
    assert audit["bulk_plastic_work_available"] is False
    assert np.all(np.isnan(ledgers["W_bulk"]))


def test_validated_runner_invokes_direct_j_energy_postprocessor():
    runner = (
        ROOT
        / "scripts"
        / "run_v10_2_27_paper_four_class_30deg_long_rcurves_validated.sh"
    ).read_text()
    assert "plot_v10_2_27_paper_four_class_K_vs_temperature.py" in runner
    assert "plot_v10_2_27_paper_four_class_J_energy_vs_temperature.py" in runner
    assert "--youngs-modulus-pa" not in runner
    assert "--poisson-ratio" not in runner
