from pathlib import Path
import ast
import subprocess


def test_v10210_python_runners_parse():
    for name in (
        "scripts/run_v10_2_10_staged_parameterization.py",
        "scripts/run_v10_2_10_physical_artifact_workflow.py",
    ):
        ast.parse(Path(name).read_text())


def test_v10210_shell_runners_have_valid_syntax():
    for name in (
        "scripts/run_v10_2_10_staged_parameterization.sh",
        "scripts/run_v10_2_10_physical_artifact_workflow.sh",
    ):
        result = subprocess.run(
            ["bash", "-n", name],
            capture_output=True,
            text=True,
            check=False,
        )
        assert result.returncode == 0, result.stderr
