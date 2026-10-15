import json

import pytest

from arrhenius_fracture.process_state_ownership_v11 import (
    ProcessStateOwner, ProcessStateOwnerRegistry,
)
from scripts.run_recursive_branch_readiness import qualification_fixture


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
