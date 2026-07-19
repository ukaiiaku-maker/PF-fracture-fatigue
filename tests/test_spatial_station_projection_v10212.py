import csv
import json

import pytest

from arrhenius_fracture.checked_spatial_station_projection_v10212 import (
    REQUIRED_INTERACTION_SCHEMA,
    STATION_SCHEMA,
    expand_station_response_files,
)
from arrhenius_fracture.physical_fem_snapshot_v10212 import RESPONSE_COLUMNS


def _write_station_file(path, measured_bins, schema=REQUIRED_INTERACTION_SCHEMA):
    full_grid = [0.0, 1.0, 2.0]
    rows = []
    for bin_index in measured_bins:
        x_m = full_grid[bin_index]
        H_I = 2.0 + 3.0 * x_m
        H_II = -1.0 + 0.5 * x_m
        for sign in (-1, 1):
            for magnitude in (0.25, 0.5):
                content = sign * magnitude
                rows.append(
                    {
                        "state_id": "S0",
                        "r_eff_over_r0": 1.0,
                        "opening_strength_fraction": 0.5,
                        "crack_extension_m": 0.0,
                        "region": "active",
                        "system": 0,
                        "bin": bin_index,
                        "x_m": x_m,
                        "burgers_sign": sign,
                        "delta_signed_line_content": content,
                        "K_I_base_Pa_sqrt_m": 10.0,
                        "K_I_perturbed_Pa_sqrt_m": 10.0 - H_I * content,
                        "K_II_base_Pa_sqrt_m": 1.0,
                        "K_II_perturbed_Pa_sqrt_m": 1.0 - H_II * content,
                        "interaction_integral_schema": schema,
                        "ribbon_width_m": 0.1,
                        "mesh_area_ratio": 1.0,
                    }
                )
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=RESPONSE_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    path.with_suffix(".audit.json").write_text(
        json.dumps(
            {
                "schema": STATION_SCHEMA,
                "state_id": "S0",
                "physical_fem_responses_generated": True,
                "responses_are_measured_stations_not_full_grid": True,
                "full_active_grid_x_m": full_grid,
                "full_wake_grid_x_m": [],
                "measured_station_indices": {"active": measured_bins, "wake": []},
                "ribbon_width_m": 0.1,
                "r_eff_is_analytical_tip_state": True,
                "production_parameterization_allowed": False,
            }
        )
    )


def test_projection_recovers_linear_curve_and_passes_cross_validation(tmp_path):
    path = tmp_path / "stations.csv"
    _write_station_file(path, [0, 1, 2])
    expanded, physical, report = expand_station_response_files([path])
    assert len(physical) == 1
    assert report["spatial_cross_validation_passed"] is True
    assert report["subelement_rows_claimed_as_direct_fem"] is False
    assert report["projected_schema_matches_measured_schema"] is True
    assert report["interaction_integral_schema"] == REQUIRED_INTERACTION_SCHEMA
    assert len(expanded) == 3 * 2 * 2
    middle = [
        row
        for row in expanded
        if row["bin"] == 1
        and row["burgers_sign"] == 1
        and row["delta_signed_line_content"] == 0.5
    ][0]
    assert middle["K_I_perturbed_Pa_sqrt_m"] == 10.0 - 5.0 * 0.5


def test_two_station_curve_remains_unauthorized(tmp_path):
    path = tmp_path / "stations.csv"
    _write_station_file(path, [0, 2])
    _expanded, _physical, report = expand_station_response_files([path])
    assert report["all_curves_have_leave_one_out_validation"] is False
    assert report["spatial_cross_validation_passed"] is False


def test_wrong_interaction_integral_schema_is_rejected(tmp_path):
    path = tmp_path / "stations.csv"
    _write_station_file(path, [0, 1, 2], schema="unreviewed_interaction_integral")
    with pytest.raises(ValueError, match="must use exactly"):
        expand_station_response_files([path])
