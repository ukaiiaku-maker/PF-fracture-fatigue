from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from arrhenius_fracture.sharp_front_current_source_branching import (
    CANDIDATE,
    OPTION,
    REGISTRY,
    SELECTION,
)
from arrhenius_fracture.sharp_front_current_source_branching_audited import (
    CLAIM_LABEL,
    validate,
)
from scripts.run_pf_current_source_branching_capability_pair import command


def _base_args(maximum_fronts: int) -> list[str]:
    return [
        "--current-source-branching-capability",
        "--maximum-fronts",
        str(maximum_fronts),
        "--mode",
        "2d",
        "--crack-backend",
        "sharp_wake",
        "--bulk-plasticity-mode",
        "tip_only",
    ]


@pytest.mark.parametrize("maximum_fronts", [1, 2])
def test_current_source_entry_accepts_only_matched_pair(maximum_fronts: int) -> None:
    assert validate(_base_args(maximum_fronts)) == maximum_fronts


@pytest.mark.parametrize(
    "args, message",
    [
        ([], "capability flag"),
        (_base_args(3), "maximum-fronts"),
        (_base_args(2) + ["--branch-probability", "0.5"], "forbidden"),
        (
            [
                "--current-source-branching-capability",
                "--maximum-fronts",
                "2",
                "--mode",
                "2d",
                "--crack-backend",
                "cohesive",
                "--bulk-plasticity-mode",
                "tip_only",
            ],
            "sharp_wake",
        ),
        (_base_args(2) + ["--crack-backend", "sharp_wake"], "duplicate launch"),
    ],
)
def test_current_source_entry_fails_closed(args: list[str], message: str) -> None:
    with pytest.raises(SystemExit, match=message):
        validate(args)


def test_matched_commands_differ_only_in_front_cap_and_output(tmp_path: Path) -> None:
    family = tmp_path / "family.json"
    control = command(tmp_path / "control", family, 1, 40.0)
    enabled = command(tmp_path / "enabled", family, 2, 40.0)
    normalized_control = list(control)
    normalized_enabled = list(enabled)
    for normalized in (normalized_control, normalized_enabled):
        normalized[normalized.index("--maximum-fronts") + 1] = "<FRONTS>"
        normalized[normalized.index("--out") + 1] = "<OUT>"
    assert normalized_control == normalized_enabled
    assert "--target-crack-extension-um" in control
    assert control[control.index("--target-crack-extension-um") + 1] == "300"
    assert control[control.index("--parameter-option") + 1] == OPTION


def test_current_transfer_inputs_are_frozen() -> None:
    assert CANDIDATE == "oneD_v2_focused_weak_T_0016"
    assert hashlib.sha256(REGISTRY.read_bytes()).hexdigest() == (
        "43d4973b97156430d177f29eee8bbd631872db2a3ef0c3293c27604b00c63ef4"
    )
    assert hashlib.sha256(SELECTION.read_bytes()).hexdigest() == (
        "ab902b986f55dcd7993dc0d2d3f262885bc9ecdca14197d664df18e4fb9e0acd"
    )
    assert CLAIM_LABEL == "CAPABILITY_DEMONSTRATION_NOT_VALIDATED_BRANCHING_PHYSICS"
