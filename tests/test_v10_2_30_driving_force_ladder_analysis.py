from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "analyze_v10_2_30_driving_force_ladder.py"


def load_module():
    spec = importlib.util.spec_from_file_location("ladder_analysis", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_fraction_uses_existing_case_name_contract():
    module = load_module()
    assert module.fraction("/x/peak_f0p975_seed1720/output") == ("peak", .975)
    assert module.fraction("/x/dbtt_f1p100_seed1001723/output") == ("dbtt", 1.1)


def test_regime_marks_only_sub_thousand_cycle_target_as_near_monotonic():
    module = load_module()
    assert module.regime({"status": "completed", "cycles_to_target": 208.0}) == \
        "NEAR_MONOTONIC_CYCLIC_FAILURE"
    assert module.regime({"status": "completed", "cycles_to_target": 22348.0}) == "LCF"
    assert module.regime({"status": "censored", "cycles_to_target": None}) == "VHCF"
