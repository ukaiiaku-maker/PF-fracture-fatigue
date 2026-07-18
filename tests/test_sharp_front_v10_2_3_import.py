import ast
from pathlib import Path


def test_v10_2_3_entry_point_is_syntactically_valid():
    path = Path("arrhenius_fracture/sharp_front_v10_2_3.py")
    ast.parse(path.read_text())


def test_v10_2_3_monotonic_runner_uses_full_option_names():
    text = Path("scripts/run_v10_2_3_temperature_fracture_smoke.sh").read_text()
    assert "--crystal-theta-deg" in text
    assert "--target-crack-extension-um" in text
    assert "--theta-deg" not in text
    assert "--target-ext-um" not in text
