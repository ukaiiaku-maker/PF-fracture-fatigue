from __future__ import annotations

import csv
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "plot_v10_2_28_four_class_KJ_vs_temperature.py"
SPEC = importlib.util.spec_from_file_location("plot_v10228_KJ_temperature", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_locked_tungsten_cubic_tensor_gives_expected_isotropic_moduli():
    E, nu = MODULE._isotropic_moduli_from_cubic(523.0e9, 203.0e9, 160.0e9)
    assert E / 1.0e9 == pytest.approx(409.4770642201835)
    assert nu == pytest.approx(0.2796143250688705)
    assert E / (1.0 - nu * nu) / 1.0e9 == pytest.approx(444.2073170731707)


def test_non_zener_one_tensor_requires_explicit_effective_modulus():
    with pytest.raises(ValueError, match="not isotropic"):
        MODULE._isotropic_moduli_from_cubic(523.0e9, 203.0e9, 150.0e9)


def test_tail_average_is_weighted_by_propagation_overlap():
    pre = np.array([0.0, 5.0, 15.0])
    post = np.array([5.0, 15.0, 25.0])
    values = np.array([10.0, 20.0, 30.0])
    mean, covered, count = MODULE._weighted_tail_response(
        pre,
        post,
        values,
        tail_start_um=10.0,
        tail_end_um=25.0,
    )
    assert covered == pytest.approx(15.0)
    assert count == 2
    assert mean == pytest.approx((5.0 * 20.0 + 10.0 * 30.0) / 15.0)


def test_default_tail_policy_uses_at_least_length_and_fraction():
    assert MODULE._tail_width_um(
        1000.0,
        tail_length_um=200.0,
        tail_fraction=0.20,
        policy="maximum",
    ) == pytest.approx(200.0)
    assert MODULE._tail_width_um(
        1100.0,
        tail_length_um=200.0,
        tail_fraction=0.20,
        policy="maximum",
    ) == pytest.approx(220.0)


def test_plotter_writes_exact_eventwise_K_and_J_table(tmp_path: Path):
    outroot = tmp_path / "campaign"
    option = MODULE.OPTION_ORDER[0]
    case_root = outroot / option / "T300K_th30_seed3621"
    case_root.mkdir(parents=True)
    (case_root / "COMPLETE").write_text("\n")

    rows = np.array(
        [
            [1.0, 10.0e6, 10.0e-6, 10.0e-6, 1.0],
            [2.0, 20.0e6, 20.0e-6, 10.0e-6, 1.0],
            [3.0, 30.0e6, 30.0e-6, 10.0e-6, 1.0],
        ]
    )
    np.savetxt(
        case_root / "steps_0300K.csv",
        rows,
        delimiter=",",
        header="step,KJ_Pa_sqrtm,crack_extension_m,da_block_m,n_fire",
        comments="",
    )
    (case_root / "v10_2_27_paper_four_class_parameter_transfer.json").write_text(
        json.dumps(
            {
                "selected_candidate": "synthetic_candidate",
                "source_material_class": "peak",
            }
        )
    )
    (outroot / "v10_2_27_campaign_acceptance.json").write_text(
        json.dumps(
            {
                "records": [
                    {
                        "option": option,
                        "temperature_K": 300.0,
                        "theta_deg": 30.0,
                        "seed": 3621,
                        "case_root": str(case_root),
                        "complete": True,
                    }
                ]
            }
        )
    )

    rc = MODULE.main(
        [
            "--outroot",
            str(outroot),
            "--Eprime-GPa",
            "100",
            "--tail-length-um",
            "20",
            "--tail-fraction",
            "0",
            "--tail-policy",
            "length",
            "--minimum-tail-coverage-fraction",
            "1",
            "--no-plots",
        ]
    )
    assert rc == 0

    csv_path = (
        outroot
        / "temperature_response"
        / "v10_2_28_four_class_KJ_temperature_response.csv"
    )
    with csv_path.open(newline="") as stream:
        output = list(csv.DictReader(stream))
    assert len(output) == 1
    row = output[0]
    assert float(row["initial_K_MPa_sqrt_m"]) == pytest.approx(10.0)
    assert float(row["tail_average_K_MPa_sqrt_m"]) == pytest.approx(25.0)
    assert float(row["initial_J_kJ_m2"]) == pytest.approx(1.0)
    # Eventwise mean J over equal 10 um intervals: (20^2 + 30^2)/2 / 100.
    assert float(row["tail_average_J_kJ_m2"]) == pytest.approx(6.5)
    assert float(row["tail_covered_extension_um"]) == pytest.approx(20.0)
    assert int(row["tail_event_row_count"]) == 2


def test_plotter_does_not_use_two_checkpoint_tail_approximation():
    source = SCRIPT.read_text()
    assert "K_750" not in source
    assert "K_1000" not in source
    assert "fracture-event overlap length in crack-extension space" in source
