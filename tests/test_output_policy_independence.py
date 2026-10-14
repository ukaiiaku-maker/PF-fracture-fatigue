from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.qualify_output_policy_independence import PHYSICS_ARTIFACTS, compare_runs


def _fixture(root: Path, *, full: bool, payload: bytes = b"same") -> None:
    root.mkdir()
    args = {
        "material": "ceramic",
        "steps": 6,
        "out": str(root),
        "print_every": 1 if full else 20,
        "save_snapshots": 4 if full else 0,
        "snapshot_cols": 2 if full else 4,
        "no_mpz_state_output": not full,
        "no_plots": not full,
    }
    (root / "run_args.json").write_text(json.dumps(args))
    for name in PHYSICS_ARTIFACTS:
        (root / name).write_bytes(payload)
    if full:
        (root / "field_snapshots_700K.png").write_bytes(b"render")


def test_output_policy_comparison_requires_identical_calculated_state(tmp_path: Path):
    sparse, full = tmp_path / "sparse", tmp_path / "full"
    _fixture(sparse, full=False)
    _fixture(full, full=True)
    report = compare_runs(sparse, full)
    assert report["decision"] == "PASS"
    assert report["physics_arguments_identical"] is True
    assert report["full_only_rendered_outputs"] == ["field_snapshots_700K.png"]

    (full / PHYSICS_ARTIFACTS[0]).write_bytes(b"changed")
    assert compare_runs(sparse, full)["decision"] == "FAIL_STATE_DEPENDS_ON_OUTPUT_POLICY"


def test_output_policy_comparison_rejects_physics_changes(tmp_path: Path):
    sparse, full = tmp_path / "sparse", tmp_path / "full"
    _fixture(sparse, full=False)
    _fixture(full, full=True)
    args = json.loads((full / "run_args.json").read_text())
    args["steps"] = 7
    (full / "run_args.json").write_text(json.dumps(args))
    with pytest.raises(ValueError, match="arguments differ"):
        compare_runs(sparse, full)
