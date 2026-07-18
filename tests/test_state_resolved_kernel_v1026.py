import csv
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from arrhenius_fracture.signed_kernel_family_v1026 import (
    KernelState,
    StateResolvedSignedShieldingKernelFamily,
)


def _family(fixed=False):
    states = []
    for index, r in enumerate((1.0, 2.0)):
        states.append(
            KernelState(
                state_id=f"s{index}",
                coordinates=np.array([r, 0.5, 0.0]),
                active_I=np.array([[r], [-2.0 * r]]),
                wake_I=np.array([[0.5 * r], [-r]]),
                active_II=np.array([[0.1 * r], [0.2 * r]]),
                wake_II=np.array([[0.05 * r], [0.1 * r]]),
                metadata={},
            )
        )
    return StateResolvedSignedShieldingKernelFamily(
        states=states,
        active_x_m=np.array([0.5]),
        wake_x_m=np.array([0.5]),
        activation_to_line_content=np.array([0.1, 0.2]),
        source_capacity_bounds=np.array([[1.0, 100.0], [1.0, 100.0]]),
        fixed_kernel_assessment={
            "fixed_kernel_accepted": fixed,
            "reference_state_id": "s0",
        },
        interpolation={
            "method": "fixed_reference" if fixed else "inverse_distance",
            "neighbors": 2,
            "power": 2.0,
            "envelope_relative_tolerance": 1.0e-10,
        },
        metadata={"production_parameterization_allowed": False},
        source_path="test-family.json",
    )


def test_family_interpolates_inside_envelope_and_retains_mode_II():
    family = _family()
    active, wake = family.resolve(
        r_eff_over_r0=1.5,
        opening_strength_fraction=0.5,
        crack_extension_m=0.0,
    )
    assert active[:, 0] == pytest.approx([1.5, -3.0])
    assert wake[:, 0] == pytest.approx([0.75, -1.5])
    assert family.active_kernel_II[:, 0] == pytest.approx([0.15, 0.30])
    assert family.audit_payload()["last_state_ids"] == ["s0", "s1"]


def test_family_rejects_extrapolation():
    family = _family()
    with pytest.raises(RuntimeError, match="outside the validated"):
        family.resolve(
            r_eff_over_r0=2.1,
            opening_strength_fraction=0.5,
            crack_extension_m=0.0,
        )


def _write_builder_inputs(tmp_path: Path, two_magnitudes=True):
    tmp_path.mkdir(parents=True, exist_ok=True)
    response = tmp_path / "responses.csv"
    normalization = tmp_path / "normalization.json"
    fields = [
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
    magnitudes = (1.0, 2.0) if two_magnitudes else (1.0,)
    for ir, r in enumerate((1.0, 2.0)):
        for io, opening in enumerate((0.2, 0.6, 1.0)):
            for ia, extension in enumerate((0.0, 5.0e-6)):
                state_id = f"r{ir}_o{io}_a{ia}"
                for region in ("active", "wake"):
                    for system in (0, 1):
                        H_I = (1.0 if region == "active" else 0.5) * (
                            system + 1
                        )
                        H_II = 0.1 * H_I
                        for sign in (-1, 1):
                            for magnitude in magnitudes:
                                content = sign * magnitude
                                K_I_base = 10.0
                                K_II_base = 2.0
                                rows.append(
                                    {
                                        "state_id": state_id,
                                        "r_eff_over_r0": r,
                                        "opening_strength_fraction": opening,
                                        "crack_extension_m": extension,
                                        "region": region,
                                        "system": system,
                                        "bin": 0,
                                        "x_m": 0.5,
                                        "burgers_sign": sign,
                                        "delta_signed_line_content": content,
                                        "K_I_base_Pa_sqrt_m": K_I_base,
                                        "K_I_perturbed_Pa_sqrt_m": K_I_base
                                        - H_I * content,
                                        "K_II_base_Pa_sqrt_m": K_II_base,
                                        "K_II_perturbed_Pa_sqrt_m": K_II_base
                                        - H_II * content,
                                    }
                                )
    with response.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    normalization.write_text(
        json.dumps(
            {
                "normalization_source": "2d_unit_slip_to_line_content",
                "activation_to_line_content_by_system": [0.1, 0.1],
                "source_capacity_bounds_per_system": [[1.0, 100.0], [1.0, 100.0]],
                "fitted_to_toughness_or_fatigue": False,
            }
        )
    )
    return response, normalization


def test_builder_requires_two_magnitudes_and_builds_loadable_family(tmp_path):
    root = Path(__file__).resolve().parents[1]
    script = root / "scripts" / "build_v10_2_6_state_resolved_kernel_family.py"
    response, normalization = _write_builder_inputs(tmp_path, two_magnitudes=True)
    output = tmp_path / "family.json"
    completed = subprocess.run(
        [
            sys.executable,
            str(script),
            "--responses",
            str(response),
            "--normalization",
            str(normalization),
            "--out",
            str(output),
        ],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr + completed.stdout
    family = StateResolvedSignedShieldingKernelFamily.from_json(output)
    assert len(family.states) == 12
    assert family.fixed_kernel_assessment["fixed_kernel_accepted"] is True
    assert family.metadata["production_parameterization_allowed"] is False

    bad_response, bad_normalization = _write_builder_inputs(
        tmp_path / "bad", two_magnitudes=False
    )
    bad_output = tmp_path / "bad-family.json"
    failed = subprocess.run(
        [
            sys.executable,
            str(script),
            "--responses",
            str(bad_response),
            "--normalization",
            str(bad_normalization),
            "--out",
            str(bad_output),
        ],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    )
    assert failed.returncode != 0
    assert "at least two perturbation magnitudes" in (failed.stderr + failed.stdout)
