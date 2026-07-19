"""Prevent reintroduction of the removed variational fracture subsystem."""
from pathlib import Path


def test_legacy_variational_fracture_modules_are_absent():
    package = Path(__file__).resolve().parents[1] / "arrhenius_fracture"
    forbidden = {
        "a" + "t1.py",
        "a" + "t2_overlay.py",
        "phase_" + "field.py",
    }
    assert forbidden.isdisjoint({path.name for path in package.iterdir()})
