import json
from pathlib import Path

from scripts.compare_v10_2_30_weakt_partition_equivalence import compare


def _write(root: Path, records):
    root.mkdir()
    (root / "kinetic_tip_cell_audit_v101.json").write_text(
        json.dumps({"records": records})
    )


def _record(cycles, *, B=1.0e-6, mobile=10.0, backstress=2.0e8):
    return {
        "loading_mode": "cyclic",
        "cycles_consumed": cycles,
        "fired": False,
        "B": B,
        "state_mobile_count": mobile,
        "state_retained_count": 0.0,
        "state_emitted_total": mobile,
        "state_escaped_total": 0.0,
        "persistent_sigma_back_Pa": backstress,
        "persistent_tip_radius_m": 1.0e-6,
        "coupled_hazard_lambda_end_per_s": 1.0e-18,
        "coupled_hazard_sigma_end_Pa": 2.0e9,
        "coupled_hazard_shield_end_Pa_sqrt_m": 0.0,
    }


def test_partition_equivalence_passes_for_matching_endpoints(tmp_path):
    reference = tmp_path / "reference"
    partitioned = tmp_path / "partitioned"
    _write(reference, [_record(1.0e6)])
    rows = [_record(1.0e5, B=(index + 1) * 1.0e-7) for index in range(10)]
    rows[-1] = _record(1.0e5)
    _write(partitioned, rows)

    result = compare(
        reference,
        partitioned,
        state_relative_tol=1.0e-3,
        clock_relative_tol=1.0e-3,
        lambda_log_tol_decades=0.01,
        shield_absolute_tol=1.0e-6,
    )

    assert result["passed"] is True
    assert result["reference"]["record_count"] == 1
    assert result["partitioned"]["record_count"] == 10
    assert result["partitioned"]["cumulative_cycles"] == 1.0e6
    assert (partitioned / "v10_2_30_weakt_partition_equivalence.json").is_file()


def test_partition_equivalence_fails_for_state_drift(tmp_path):
    reference = tmp_path / "reference"
    partitioned = tmp_path / "partitioned"
    _write(reference, [_record(1.0e6)])
    _write(partitioned, [_record(1.0e6, mobile=12.0)])

    result = compare(
        reference,
        partitioned,
        state_relative_tol=1.0e-3,
        clock_relative_tol=1.0e-3,
        lambda_log_tol_decades=0.01,
        shield_absolute_tol=1.0e-6,
    )

    assert result["passed"] is False
    assert result["comparisons"]["mobile_count"]["passed"] is False
