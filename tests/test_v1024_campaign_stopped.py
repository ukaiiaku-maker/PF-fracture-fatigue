from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]


def test_shell_campaign_runner_is_hard_stopped():
    result = subprocess.run(
        ["bash", str(ROOT / "scripts/run_v10_2_4_2d_calibrated_campaign.sh")],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 64
    assert "STOPPED" in result.stderr
    assert "signed 2-D unit-response" in result.stderr


def test_python_campaign_runner_is_hard_stopped():
    result = subprocess.run(
        [sys.executable, str(ROOT / "scripts/run_v10_2_4_reduced_campaign.py")],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode != 0
    assert "STOPPED" in result.stderr
    assert "fitted attenuation" in result.stderr
