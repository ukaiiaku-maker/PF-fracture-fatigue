import csv
import json

from arrhenius_fracture.physical_fem_capture_trace_v10212 import (
    PhysicalFEMCapture,
)


def test_reachable_state_trace_keeps_observed_radius_separate_from_kernel_coordinate(tmp_path):
    capture = PhysicalFEMCapture([], tmp_path / "capture")
    capture.coordinate_trace = [
        {
            "trace_index": 0,
            "temperature_K": 700.0,
            "K_applied_Pa_sqrt_m": 20.0e6,
            "observed_analytical_r_eff_over_r0": 3.25,
            "kernel_radius_compatibility_coordinate": 1.0,
            "opening_strength_fraction": 0.6,
            "crack_extension_m": 5.0e-6,
            "mechanics_serial": 10,
            "drive_serial": 10,
        }
    ]
    path = capture._write_trace()
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert rows[0]["observed_analytical_r_eff_over_r0"] == "3.25"
    assert rows[0]["kernel_radius_compatibility_coordinate"] == "1.0"
    audit = json.loads((path.parent / "reachable_physical_state_trace.json").read_text())
    assert audit["records"] == 1
    assert audit["kernel_radius_axis_policy"] == "disabled_constant_compatibility"
    assert audit["active_kernel_design_coordinates"] == [
        "opening_strength_fraction",
        "crack_extension_m",
    ]
