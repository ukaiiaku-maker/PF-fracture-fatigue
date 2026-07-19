import csv
import json
from pathlib import Path
import subprocess
import sys

from arrhenius_fracture.interaction_integral_v1029 import MODEL_ID
from arrhenius_fracture.signed_kernel_family_v1029 import (
    StateResolvedSignedShieldingKernelFamily,
)


def _write_inputs(root: Path, *, omit_corner=False):
    root.mkdir(parents=True, exist_ok=True)
    responses = root / "responses.csv"
    normalization = root / "normalization.json"
    fields = [
        "interaction_integral_schema",
        "state_id",
        "r_eff_over_r0",
        "opening_strength_fraction",
        "crack_extension_m",
        "region",
        "system",
        "bin",
        "x_m",
        "burgers_sign",
        "delta_signed_line_content",
        "K_I_base_Pa_sqrt_m",
        "K_I_perturbed_Pa_sqrt_m",
        "K_II_base_Pa_sqrt_m",
        "K_II_perturbed_Pa_sqrt_m",
    ]
    rows = []
    for r in (1.0, 1.5):
        for opening in (0.0, 0.5, 1.0):
            for extension in (0.0, 5.0e-6):
                if omit_corner and r == 1.5 and opening == 1.0 and extension == 5.0e-6:
                    continue
                state_id = f"r{r}_o{opening}_a{extension}"
                # Opening-independent response deliberately validates boundary
                # stationarity for this synthetic software contract.
                H_I = 2.0e5 * r * (1.0 + extension / 5.0e-6)
                H_II = -0.1 * H_I
                for sign in (-1, 1):
                    for magnitude in (1.0, 2.0):
                        content = sign * magnitude
                        rows.append(
                            {
                                "interaction_integral_schema": MODEL_ID,
                                "state_id": state_id,
                                "r_eff_over_r0": r,
                                "opening_strength_fraction": opening,
                                "crack_extension_m": extension,
                                "region": "active",
                                "system": 0,
                                "bin": 0,
                                "x_m": 0.5,
                                "burgers_sign": sign,
                                "delta_signed_line_content": content,
                                "K_I_base_Pa_sqrt_m": 1.0e7,
                                "K_I_perturbed_Pa_sqrt_m": 1.0e7 - H_I * content,
                                "K_II_base_Pa_sqrt_m": 0.0,
                                "K_II_perturbed_Pa_sqrt_m": -H_II * content,
                            }
                        )
    with responses.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    normalization.write_text(
        json.dumps(
            {
                "normalization_source": "2d_unit_slip_to_line_content",
                "activation_to_line_content_by_system": [0.1],
                "source_capacity_bounds_per_system": [[1.0, 100.0]],
                "fitted_to_toughness_or_fatigue": False,
            }
        )
    )
    return responses, normalization


def test_v1029_builder_requires_complete_grid_and_validates_boundaries(tmp_path):
    root = Path(__file__).resolve().parents[1]
    script = root / "scripts" / "build_v10_2_9_state_resolved_kernel_family.py"
    responses, normalization = _write_inputs(tmp_path / "good")
    out = tmp_path / "family.json"
    completed = subprocess.run(
        [
            sys.executable,
            str(script),
            "--responses",
            str(responses),
            "--normalization",
            str(normalization),
            "--out",
            str(out),
        ],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr + completed.stdout
    payload = json.loads(out.read_text())
    assert payload["complete_cartesian_state_grid"]
    assert payload["opening_boundary_policy"]["policy"] == "validated_boundary_saturation"
    assert not payload["production_parameterization_allowed"]
    family = StateResolvedSignedShieldingKernelFamily.from_json(out)
    assert len(family.states) == 12

    bad_responses, bad_normalization = _write_inputs(
        tmp_path / "bad", omit_corner=True
    )
    failed = subprocess.run(
        [
            sys.executable,
            str(script),
            "--responses",
            str(bad_responses),
            "--normalization",
            str(bad_normalization),
            "--out",
            str(tmp_path / "bad.json"),
        ],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    )
    assert failed.returncode != 0
    assert "complete Cartesian grid" in (failed.stderr + failed.stdout)
