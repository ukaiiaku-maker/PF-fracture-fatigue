from pathlib import Path
import subprocess


def test_v10214_runner_shell_syntax():
    root = Path(__file__).resolve().parents[1]
    completed = subprocess.run(
        ["bash", "-n", str(root / "scripts" / "run_v10_2_14_active_signed_atlas.sh")],
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
