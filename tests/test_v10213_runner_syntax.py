from pathlib import Path
import subprocess


def test_v10213_runner_has_valid_bash_syntax():
    script = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "run_v10_2_13_extension_only_signed_atlas.sh"
    )
    completed = subprocess.run(
        ["bash", "-n", str(script)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
