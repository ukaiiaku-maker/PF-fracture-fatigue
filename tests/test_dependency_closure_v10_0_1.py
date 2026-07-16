from __future__ import annotations

import importlib
from pathlib import Path


MODULES = (
    "arrhenius_fracture.config",
    "arrhenius_fracture.materials",
    "arrhenius_fracture.mesh",
    "arrhenius_fracture.fem",
    "arrhenius_fracture.j_integral",
    "arrhenius_fracture.crystal",
    "arrhenius_fracture.plasticity",
    "arrhenius_fracture.fatigue_v1",
    "arrhenius_fracture.crack_backend",
    "arrhenius_fracture.coalescence",
    "arrhenius_fracture.sharp_front",
    "arrhenius_fracture.sharp_front_v10_1",
    "arrhenius_fracture.material_manifest",
    "arrhenius_fracture.unified_mpz",
    "arrhenius_fracture.unified_front",
)


def test_production_imports_resolve_inside_current_checkout():
    root = Path(__file__).resolve().parents[1]
    package_root = (root / "arrhenius_fracture").resolve()
    outside = {}
    for name in MODULES:
        module = importlib.import_module(name)
        path = Path(module.__file__).resolve()
        if package_root not in path.parents:
            outside[name] = str(path)
    assert not outside, (
        "mixed editable installations detected; production modules resolved "
        f"outside {package_root}: {outside}"
    )


def test_sharp_backend_contains_no_cohesive_dependency():
    module = importlib.import_module("arrhenius_fracture.crack_backend")
    text = Path(module.__file__).read_text()
    assert "from .cohesive" not in text
    assert not hasattr(module, "EdgeSplitCZMBackend")
    assert not hasattr(module, "AdaptiveCZMBackend")
