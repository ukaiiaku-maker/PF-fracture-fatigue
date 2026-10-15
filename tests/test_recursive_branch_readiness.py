import json

import pytest

from arrhenius_fracture.process_state_ownership_v11 import (
    ProcessStateOwner, ProcessStateOwnerRegistry,
)
from scripts.run_recursive_branch_readiness import (
    native_checkpoint_evidence, qualification_fixture,
)


def test_second_generation_owner_adapter_and_checkpoint_are_exact():
    result = qualification_fixture()
    assert result["status"] == "PARTIAL_ADAPTER_QUALIFIED_NATIVE_BIRTH_PENDING"
    assert result["native_birth_demonstrated"] is False
    assert result["checkpoint_exact"] is True
    assert result["fixture_classification"].startswith("INJECTED_")
    assert result["active_tips"] == [
        "daughter-2", "granddaughter-1", "granddaughter-2"
    ]


def test_unresolved_daughter_cannot_recursively_branch_again():
    registry = ProcessStateOwnerRegistry(
        {"root": "root"},
        {"root": ProcessStateOwner(
            "root", "single_tip_engine", process_engine_id="root"
        )},
    ).branch("root", "cluster", ("left", "right"))
    with pytest.raises(ValueError, match="parent_process_zone_still_unresolved"):
        registry.branch("left", "illegal", ("a", "b"))


def test_checkpoint_decoder_rejects_unknown_schema():
    payload = qualification_fixture()["owner_registry"]
    payload = json.loads(json.dumps(payload))
    payload["schema"] = "unknown"
    with pytest.raises(ValueError, match="unsupported"):
        ProcessStateOwnerRegistry.from_dict(payload)


@pytest.mark.external_evidence
def test_native_first_birth_daughters_and_restart_are_classified_truthfully():
    root = __import__("pathlib").Path(__file__).resolve().parents[1]
    evidence = root / "integration_evidence/targeted_completion"
    run = evidence / "theta40_v12_qualified_restarted"
    result = native_checkpoint_evidence(
        run / "checkpoint/latest.v12.pkl",
        run / "accepted_intervals/transactions/interval-00002184",
    )
    assert result["native_first_birth_demonstrated"] is True
    assert result["mechanically_evolved_daughters_demonstrated"] is True
    assert result["second_generation_birth_demonstrated"] is False
    assert result["positive_rate_active_fronts"]
    assert result["recursive_eligible_front_ids"] == []
    assert {row["reason"] for row in result["unresolved_owner_gates"]} == {
        "parent_process_zone_still_unresolved"
    }
    assert "--maximum-fronts" in result["next_native_command"]
    assert result["next_native_command"][
        result["next_native_command"].index("--maximum-fronts") + 1
    ] == "4"
