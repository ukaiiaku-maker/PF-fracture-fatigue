from pathlib import Path
import subprocess


def test_v10216_runner_wraps_latest_stage3_launcher():
    text = Path("scripts/run_v10_2_16_stage3_monotonic_temperature_sweep.sh").read_text()
    assert "run_v10_2_15_stage3_monotonic_temperature_sweep.sh" in text
    assert "arrhenius_fracture.sharp_front_v10_2_16" in text
    assert "TARGET_EXT_UM=${TARGET_EXT_UM:-100}" in text
    assert "PYTHONUNBUFFERED=1" in text
    assert "PIPESTATUS[0]" in text
    assert "tee \"$log\"" in text


def test_v10216_runner_has_valid_bash_syntax():
    path = "scripts/run_v10_2_16_stage3_monotonic_temperature_sweep.sh"
    completed = subprocess.run(["bash", "-n", path], capture_output=True, text=True)
    assert completed.returncode == 0, completed.stderr
