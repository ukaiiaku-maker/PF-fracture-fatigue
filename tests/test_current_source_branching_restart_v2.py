from __future__ import annotations

import json

import pytest

from arrhenius_fracture.branch_checkpoint_v11 import write_branch_checkpoint
from arrhenius_fracture.branching_qualification_v2 import CLAIM_LABEL
from scripts.run_pf_current_source_branching_restart import (
    build_restart_command, restart_environment,
)
from tests.test_v11_provider_transition_restart import checkpoint_at


def restart_fixture(tmp_path):
    case = tmp_path / "source_case"
    checkpoint = case / "checkpoint/latest.json"
    write_branch_checkpoint(checkpoint_at("after_A12", tmp_path), checkpoint)
    family = tmp_path / "family/family.json"; family.parent.mkdir()
    family.write_text("{}")
    (family.parent / "mechanical_configuration.json").write_text("{}")
    (case / "pf_current_source_branching_model_audit.json").write_text(json.dumps({
        "claim_label": CLAIM_LABEL,
        "argv": [
            "--current-source-branching-capability", "--maximum-fronts", "2",
            "--mode", "2d", "--crack-backend", "sharp_wake",
            "--bulk-plasticity-mode", "tip_only", "--target-crack-extension-um", "300",
            "--signed-kernel-family", str(family),
            "--out", str(case),
        ],
    }))
    return checkpoint


def test_restart_is_a_fresh_output_fork_with_strictly_larger_target(tmp_path) -> None:
    checkpoint = restart_fixture(tmp_path)
    destination = tmp_path / "continuation"
    command, plan = build_restart_command(checkpoint, destination, 1000.0)
    assert "--v11-restart-checkpoint" in command
    assert command[command.index("--out") + 1] == str(destination.resolve())
    assert command[command.index("--target-crack-extension-um") + 1] == "1000"
    assert plan["source_provider_cache_immutable"] is True
    assert plan["destination_provider_cache_rebound"] is True
    environment = restart_environment(command)
    assert environment["SIGNED_KERNEL_FAMILY_JSON"].endswith("family/family.json")
    assert environment["MECHANICAL_CONFIG"].endswith("family/mechanical_configuration.json")


def test_restart_rejects_existing_destination_and_nonincreasing_target(tmp_path) -> None:
    checkpoint = restart_fixture(tmp_path)
    with pytest.raises(ValueError, match="must exceed"):
        build_restart_command(checkpoint, tmp_path / "continuation", 1.0)
    destination = tmp_path / "exists"; destination.mkdir()
    with pytest.raises(ValueError, match="fresh path"):
        build_restart_command(checkpoint, destination, 1000.0)
