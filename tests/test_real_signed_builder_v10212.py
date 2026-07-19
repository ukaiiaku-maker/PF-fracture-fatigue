import csv
import json
from pathlib import Path
import subprocess
import sys

from arrhenius_fracture.interaction_integral_v1029 import MODEL_ID as II_SCHEMA
from arrhenius_fracture.mechanics_normalization_v10212 import MODEL_ID as NORM_SCHEMA
from arrhenius_fracture.physical_fem_snapshot_v10212 import RESPONSE_COLUMNS
from arrhenius_fracture.physical_fem_station_responses_v10212 import (
    MODEL_ID as STATION_SCHEMA,
)
from arrhenius_fracture.signed_kernel_family_v10212 import (
    RealSigned2DShieldingKernelFamily,
)

ROOT = Path(__file__).resolve().parents[1]


def _write_state(path: Path, *, state_id: str, opening: float, extension: float, observed_r: float):
    grid = [0.0, 1.0, 2.0]
    rows = []
    for bin_index, x_m in enumerate(grid):
        H_I = 2.0e6 + 1.0e6 * x_m
        H_II = 0.0
        for sign in (-1, 1):
            for magnitude in (0.25, 0.5):
                content = sign * magnitude
                rows.append(
                    {
                        "state_id": state_id,
                        "r_eff_over_r0": observed_r,
                        "opening_strength_fraction": opening,
                        "crack_extension_m": extension,
                        "region": "active",
                        "system": 0,
                        "bin": bin_index,
                        "x_m": x_m,
                        "burgers_sign": sign,
                        "delta_signed_line_content": content,
                        "K_I_base_Pa_sqrt_m": 10.0e6,
                        "K_I_perturbed_Pa_sqrt_m": 10.0e6 - H_I * content,
                        "K_II_base_Pa_sqrt_m": 0.0,
                        "K_II_perturbed_Pa_sqrt_m": -H_II * content,
                        "interaction_integral_schema": II_SCHEMA,
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
                "state_id": state_id,
                "physical_fem_responses_generated": True,
                "responses_are_measured_stations_not_full_grid": True,
                "full_active_grid_x_m": grid,
                "full_wake_grid_x_m": [],
                "measured_station_indices": {"active": [0, 1, 2], "wake": []},
                "minimum_station_spacing_m": 1.0,
                "ribbon_width_m": 0.1,
                "interaction_integral_schema": II_SCHEMA,
                "fem_tip_geometry_blunted": False,
                "r_eff_is_analytical_tip_state": True,
                "production_parameterization_allowed": False,
            }
        )
    )


def test_review_builder_emits_artifact_consumable_by_production_loader(tmp_path):
    responses = []
    index = 0
    for opening in (0.0, 0.5, 1.0):
        for extension in (0.0, 1.0e-5):
            path = tmp_path / f"S{index:02d}.csv"
            _write_state(
                path,
                state_id=f"S{index:02d}",
                opening=opening,
                extension=extension,
                observed_r=1.0 + 0.2 * index,
            )
            responses.append(path)
            index += 1

    normalization = tmp_path / "normalization.json"
    normalization.write_text(
        json.dumps(
            {
                "schema": NORM_SCHEMA,
                "normalization_source": "process_zone_geometry_and_line_spacing",
                "activation_to_line_content_by_system": [1.0],
                "source_capacity_bounds_per_system": [[1.0, 10.0]],
                "fitted_to_toughness_or_fatigue": False,
                "shielding_attenuation_factor_fitted": False,
            }
        )
    )
    out = tmp_path / "atlas.json"
    command = [
        sys.executable,
        "scripts/build_v10_2_12_real_signed_atlas.py",
        "--normalization",
        str(normalization),
        "--out",
        str(out),
    ]
    for path in responses:
        command.extend(["--responses", str(path)])
    completed = subprocess.run(
        command,
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr + completed.stdout
    payload = json.loads(out.read_text())
    assert payload["kernel_radius_axis_policy"] == "disabled_constant_compatibility"
    assert payload["production_parameterization_allowed"] is False
    assert {state["r_eff_over_r0"] for state in payload["states"]} == {1.0}

    family = RealSigned2DShieldingKernelFamily.from_json(out)
    active, wake = family.resolve(
        r_eff_over_r0=1000.0,
        opening_strength_fraction=0.5,
        crack_extension_m=1.0e-5,
    )
    assert active.shape == (1, 3)
    assert wake.shape == (1, 0)
    assert family.audit_payload()["last_observed_analytical_r_eff_over_r0"] == 1000.0
