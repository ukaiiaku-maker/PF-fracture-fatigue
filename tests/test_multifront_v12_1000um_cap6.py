from __future__ import annotations

import ast
from pathlib import Path

from scripts.build_pf_branching_theta40_append_only_kernel_v12_1000um import (
    ALL_LEVELS_UM, NEW_LEVELS_UM, OLD_LEVELS_UM,
)
from scripts.run_pf_current_source_multifront_v12_1000um_cap6 import command


ROOT = Path(__file__).resolve().parents[1]


def test_authorized_append_only_family_levels_are_exact():
    assert OLD_LEVELS_UM == (
        0.0, 200.0, 400.0, 415.0, 420.0, 425.0, 600.0, 745.0,
    )
    assert NEW_LEVELS_UM == (800.0, 1000.0, 1250.0, 1600.0)
    assert ALL_LEVELS_UM == OLD_LEVELS_UM + NEW_LEVELS_UM


def test_family_builder_cannot_advance_stochastic_or_production_state():
    source = (
        ROOT / "scripts"
        / "build_pf_branching_theta40_append_only_kernel_v12_1000um.py"
    )
    tree = ast.parse(source.read_text())
    imports = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imports.append(node.module or "")
    forbidden = ("hazard", "stochastic", "sharp_front", "moving_pz")
    assert not [name for name in imports if any(word in name for word in forbidden)]


def test_physical_runner_has_cap6_without_cumulative_birth_limit():
    source = (
        ROOT / "arrhenius_fracture"
        / "sharp_front_current_source_multifront_physical_v12.py"
    ).read_text()
    assert "maximum > 6" in source
    assert '"mechanistic", maximum_fronts, None' in source
    assert "V12_TARGET_MAXIMUM_FORWARD_REACH_UM" in source
    assert 'target_forward is not None' in source


def test_authorized_launch_command_uses_absolute_forward_reach_and_no_birth_stop(tmp_path):
    values = command(tmp_path / "out", tmp_path / "checkpoint", tmp_path / "family")
    assert values[values.index("--maximum-fronts") + 1] == "6"
    assert values[values.index("--v12-target-maximum-forward-reach-um") + 1] == "1000"
    assert "--v12-stop-after-total-births" not in values
    assert values[values.index("--bulk-plasticity-mode") + 1] == "tip_only"
    assert "--no-wake-shielding" in values
