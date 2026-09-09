from __future__ import annotations

import ast
import json
from pathlib import Path

from scripts.build_pf_branching_theta40_append_only_kernel_v5_3 import (
    ALL_LEVELS_UM,
    NEW_LEVELS_UM,
    NEW_STATE_IDS,
    OLD_LEVELS_UM,
    OLD_STATE_IDS,
    _canonicalize_family_provenance,
)

ROOT = Path(__file__).resolve().parents[1]


def _family() -> dict:
    return {
        "states": [
            {"state_id": state_id, "crack_extension_m": 1.0e-6 * level}
            for state_id, level in zip(OLD_STATE_IDS + NEW_STATE_IDS, ALL_LEVELS_UM)
        ],
        "campaign_promotion": {"automatic_from_completed_mechanics": True},
    }


def _write_inputs(root: Path) -> None:
    for state_id in OLD_STATE_IDS + NEW_STATE_IDS:
        state = root / "load_invariance" / state_id
        state.mkdir(parents=True)
        response = state / "active_station_responses_load_1.csv"
        response.write_text("state_id,value\n%s,1.0\n" % state_id)
        response.with_suffix(".audit.json").write_text(
            json.dumps(
                {
                    "responses": str(response.resolve()),
                    "snapshot": str((root / "transient" / "snapshot_1").resolve()),
                    "passed": True,
                }
            )
        )
        (state / "frozen_geometry_load_invariance.json").write_text(
            json.dumps(
                {
                    "generated_load_cases": [{"responses": str(response.resolve())}],
                    "checks": {
                        "maximum_relative_load_variation": 1.0e-12,
                        "maximum_within_load_relative_spread": 2.0e-12,
                    },
                }
            )
        )


def test_append_only_levels_are_exactly_the_authorized_set():
    assert OLD_LEVELS_UM == (0.0, 200.0, 400.0, 415.0)
    assert NEW_LEVELS_UM == (420.0, 425.0, 600.0, 745.0)
    assert ALL_LEVELS_UM == (0.0, 200.0, 400.0, 415.0, 420.0, 425.0, 600.0, 745.0)


def test_builder_has_no_stochastic_or_production_evolution_imports():
    path = ROOT / "scripts" / "build_pf_branching_theta40_append_only_kernel_v5_3.py"
    tree = ast.parse(path.read_text())
    imported = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imported.append(node.module or "")
    forbidden = ("hazard", "stochastic", "sharp_front", "moving_pz", "source_emission")
    assert not [name for name in imported if any(word in name for word in forbidden)]


def test_family_provenance_canonicalization_is_run_root_independent(tmp_path: Path):
    first_root = tmp_path / "run_A"
    second_root = tmp_path / "run_B"
    _write_inputs(first_root)
    _write_inputs(second_root)
    first = _family()
    second = _family()
    _canonicalize_family_provenance(first, loads=first_root / "load_invariance")
    _canonicalize_family_provenance(second, loads=second_root / "load_invariance")
    assert first == second
    assert "/run_A/" not in json.dumps(first)
    assert "/run_B/" not in json.dumps(second)
