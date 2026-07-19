from pathlib import Path
import os
import subprocess
import sys


def test_v10214_runner_shell_syntax():
    root = Path(__file__).resolve().parents[1]
    completed = subprocess.run(
        ["bash", "-n", str(root / "scripts" / "run_v10_2_14_active_signed_atlas.sh")],
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr


def test_v10214_direct_cli_imports_without_editable_install():
    root = Path(__file__).resolve().parents[1]
    for script, arguments in (
        ("evaluate_v10_2_14_active_load_invariance.py", ["--help"]),
        ("preflight_v10_2_14_production_geometry.py", []),
    ):
        completed = subprocess.run(
            [sys.executable, "-I", str(root / "scripts" / script), *arguments],
            cwd=root,
            text=True,
            capture_output=True,
            check=False,
        )
        assert completed.returncode == 0, completed.stderr


def test_v10214_runner_rejects_wrong_conda_environment():
    root = Path(__file__).resolve().parents[1]
    environment = dict(os.environ)
    environment.update(
        {
            "CONDA_DEFAULT_ENV": "base",
            "MODE": "load-invariance",
            "SNAPSHOT": "/tmp/not-used",
            "OUTROOT": "/tmp/not-used",
        }
    )
    completed = subprocess.run(
        ["bash", str(root / "scripts" / "run_v10_2_14_active_signed_atlas.sh")],
        cwd=root,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 2
    assert "activate conda environment" in completed.stderr
    assert "Current environment: 'base'" in completed.stderr
