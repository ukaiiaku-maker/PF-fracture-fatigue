from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "build_v10_2_27_extended_active_only_atlas.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("v10227_extended_builder", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_state(root: Path, state_id: str, extension_m: float):
    state = root / state_id
    state.mkdir(parents=True)
    response = state / "active_station_responses_load_1.csv"
    response.write_text("header\n")
    response.with_suffix(".audit.json").write_text("{}\n")
    report = state / "frozen_geometry_load_invariance.json"
    report.write_text(
        json.dumps(
            {
                "schema": "v10.2.14_active_frozen_geometry_load_invariance",
                "parent_state_id": state_id,
                "cumulative_crack_path_extension_m": extension_m,
                "load_invariance_passed": True,
                "active_kernel_mechanically_measured": True,
                "wake_kernel_mechanically_measured": False,
                "wake_shielding_supported": False,
                "generated_load_cases": [
                    {"load_scale": 1.0, "responses": str(response.resolve())}
                ],
                "checks": {
                    "maximum_relative_load_variation": 1.0e-9,
                    "maximum_within_load_relative_spread": 2.0e-8,
                },
            }
        )
        + "\n"
    )
    return response, report


def test_arbitrary_six_state_family_is_sorted_and_audited(tmp_path):
    module = _load_module()
    states = [
        ("E500", 5.0e-4),
        ("E000", 0.0),
        ("E1200", 1.2e-3),
        ("E200", 2.0e-4),
        ("E1000", 1.0e-3),
        ("E800", 8.0e-4),
    ]
    pairs = [_write_state(tmp_path, *state) for state in states]
    rows = module.load_state_rows(
        [pair[0] for pair in pairs],
        [pair[1] for pair in pairs],
    )
    assert [row["state_id"] for row in rows] == [
        "E000",
        "E200",
        "E500",
        "E800",
        "E1000",
        "E1200",
    ]
    assert rows[-1]["cumulative_crack_path_extension_m"] == pytest.approx(
        1.2e-3
    )
    assert all(len(row["response_sha256"]) == 64 for row in rows)
    source = SCRIPT.read_text()
    assert "REQUIRED_STATES" not in source
    assert "hardcoded_E000_E200_E500_E800_state_set_used" in source


def test_duplicate_extension_fails_closed(tmp_path):
    module = _load_module()
    first = _write_state(tmp_path, "E1000A", 1.0e-3)
    second = _write_state(tmp_path, "E1000B", 1.0e-3)
    with pytest.raises(ValueError, match="duplicate cumulative"):
        module.load_state_rows(
            [first[0], second[0]],
            [first[1], second[1]],
        )


def test_response_report_mismatch_fails_closed(tmp_path):
    module = _load_module()
    first = _write_state(tmp_path, "E000", 0.0)
    second = _write_state(tmp_path, "E1200", 1.2e-3)
    with pytest.raises(ValueError, match="response/report mismatch"):
        module.load_state_rows(
            [second[0], first[0]],
            [first[1], second[1]],
        )
