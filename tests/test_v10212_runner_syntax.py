from pathlib import Path
import ast
import subprocess


def test_v10212_python_files_parse():
    paths = (
        "arrhenius_fracture/checked_spatial_station_projection_v10212.py",
        "arrhenius_fracture/mechanics_normalization_v10212.py",
        "arrhenius_fracture/physical_fem_capture_v10212.py",
        "arrhenius_fracture/physical_fem_capture_trace_v10212.py",
        "arrhenius_fracture/physical_fem_snapshot_v10212.py",
        "arrhenius_fracture/physical_fem_station_responses_v10212.py",
        "arrhenius_fracture/sharp_front_v10_2_12.py",
        "arrhenius_fracture/sharp_front_v10_2_12_capture.py",
        "arrhenius_fracture/signed_kernel_family_v10212.py",
        "arrhenius_fracture/spatial_station_projection_v10212.py",
        "arrhenius_fracture/state_resolved_signed_engine_v10212.py",
        "arrhenius_fracture/unit_slip_perturbation_v10212.py",
        "scripts/build_v10_2_12_compatibility_radius_family.py",
        "scripts/build_v10_2_12_mechanics_normalization.py",
        "scripts/build_v10_2_12_real_signed_atlas.py",
        "scripts/evaluate_v10_2_12_signed_snapshot.py",
    )
    for path in paths:
        ast.parse(Path(path).read_text(), filename=path)


def test_v10212_shell_runner_has_valid_syntax():
    path = "scripts/run_v10_2_12_real_signed_atlas.sh"
    result = subprocess.run(
        ["bash", "-n", path],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
