from __future__ import annotations

import importlib.util
import math
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "build_v10_2_27_extended_active_only_atlas_finite_metadata.py"
SPEC = importlib.util.spec_from_file_location("finite_atlas_metadata_v10228", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def _endpoint_payload() -> dict:
    return {
        "spatial_cross_validation_not_required_for_two_endpoint_active_curves": True,
        "exact_endpoint_projection_assessment": {"ready": True},
        "real_atlas_authorization_gates": {
            "exact_endpoint_piecewise_linear_projection_ready": True
        },
        "input_states": [
            {
                "maximum_relative_load_variation": math.inf,
                "maximum_within_load_relative_spread": float("nan"),
                "coefficient": 1.25,
            }
        ],
        "measured_station_projection": {
            "maximum_relative_spatial_cross_validation_error": math.inf,
            "projection_checks": [
                {
                    "cross_validation_available": False,
                    "full_grid_count": 80,
                    "measured_bins": [0, 79],
                    "maximum_relative_cross_validation_error": math.inf,
                    "mode_I_leave_one_out": {
                        "available": False,
                        "maximum_relative_error": math.inf,
                        "reason": "at least three measured stations are required",
                    },
                    "mode_II_leave_one_out": {
                        "available": False,
                        "maximum_relative_error": math.inf,
                        "reason": "at least three measured stations are required",
                    },
                }
            ],
        },
    }


def test_only_explicit_unavailable_review_diagnostics_become_json_null():
    payload = _endpoint_payload()
    MODULE._validate_exact_endpoint_unavailable_cross_validation(payload)
    sanitized, replacements = MODULE._sanitize(payload)

    row = sanitized["input_states"][0]
    assert row["maximum_relative_load_variation"] is None
    assert row["maximum_within_load_relative_spread"] is None
    assert row["coefficient"] == pytest.approx(1.25)

    projection = sanitized["measured_station_projection"]
    assert projection["maximum_relative_spatial_cross_validation_error"] is None
    check = projection["projection_checks"][0]
    assert check["maximum_relative_cross_validation_error"] is None
    assert check["mode_I_leave_one_out"]["maximum_relative_error"] is None
    assert check["mode_II_leave_one_out"]["maximum_relative_error"] is None
    assert check["cross_validation_available"] is False
    assert replacements == 6


def test_unavailable_spatial_diagnostic_requires_passed_endpoint_gate():
    payload = _endpoint_payload()
    payload["exact_endpoint_projection_assessment"]["ready"] = False
    with pytest.raises(ValueError, match="passed exact-endpoint"):
        MODULE._validate_exact_endpoint_unavailable_cross_validation(payload)


def test_available_or_nonendpoint_cross_validation_cannot_be_sanitized():
    payload = _endpoint_payload()
    payload["measured_station_projection"]["projection_checks"][0][
        "cross_validation_available"
    ] = True
    with pytest.raises(ValueError, match="audited unavailable two-endpoint"):
        MODULE._validate_exact_endpoint_unavailable_cross_validation(payload)


def test_nonfinite_kernel_or_geometry_value_remains_fatal():
    with pytest.raises(ValueError, match="outside permitted review diagnostics"):
        MODULE._sanitize({"active_I": math.inf})


def test_generic_maximum_relative_error_outside_projection_path_is_fatal():
    with pytest.raises(ValueError, match="outside permitted review diagnostics"):
        MODULE._sanitize({"maximum_relative_error": math.inf})


def test_direct_builder_routes_extended_assembler_through_metadata_wrapper():
    source = (
        ROOT / "scripts" / "build_v10_2_28_prescribed_geometry_kernel_numpy2.py"
    ).read_text()
    assert "build_v10_2_27_extended_active_only_atlas_finite_metadata.py" in source
    assert "subprocess.run = _run_with_finite_atlas_metadata" in source
