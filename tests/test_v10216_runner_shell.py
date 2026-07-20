from pathlib import Path


def test_v10216_runner_uses_continuum_entry_and_100um_default():
    path = Path("scripts/run_v10_2_16_stage3_monotonic_temperature_sweep.sh")
    text = path.read_text()
    assert "arrhenius_fracture.sharp_front_v10_2_16" in text
    assert "TARGET_EXT_UM=${TARGET_EXT_UM:-100}" in text
    assert "--tip-source-model continuum" in text
    assert "PYTHONUNBUFFERED=1" in text
    assert "START:" in text
    assert "COMPLETE:" in text
    assert "CAMPAIGN COMPLETE:" in text
    assert "sharp_front_v10_2_15" not in text


def test_v10216_runner_has_valid_bash_syntax():
    import subprocess

    path = "scripts/run_v10_2_16_stage3_monotonic_temperature_sweep.sh"
    completed = subprocess.run(["bash", "-n", path], capture_output=True, text=True)
    assert completed.returncode == 0, completed.stderr
