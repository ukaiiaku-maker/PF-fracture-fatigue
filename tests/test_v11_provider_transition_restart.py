from dataclasses import replace
import json

import numpy as np
import pytest

from arrhenius_fracture.branch_checkpoint_v11 import (
    ProductionBranchCheckpoint, restore_branch_checkpoint, write_branch_checkpoint,
)
from arrhenius_fracture.crack_network_v11 import ROOT_BRANCH_ID
from arrhenius_fracture.live_topology_kernel_v11 import PROVIDER_ID
from arrhenius_fracture.live_topology_runtime_v11 import LiveTopologyRuntime
from tests.test_topology_transaction_v11 import fem_state


def checkpoint_at(label, tmp_path):
    state = fem_state()
    runtime = LiveTopologyRuntime(str(tmp_path / "cache"))
    if label not in {"before_provider_transition", "immediately_before_branch_birth"}:
        runtime = replace(runtime, routing=replace(
            runtime.routing, active_mechanics_provider=PROVIDER_ID,
            transition_step=3, transition_state_hash="accepted-state",
            topology_fingerprint="topology-sha",
        ))
    return ProductionBranchCheckpoint(
        state=state, physical_time_s=3.5, accepted_load=0.75,
        mesh_identity="mesh-sha", boundary_condition_state={"opening_m": 1e-7},
        provider_runtime=runtime, provider_cache_identity="cache-sha",
        topology_fingerprint="topology-sha",
        front_competitions={ROOT_BRANCH_ID: state.competition}, branch_clusters=(),
        projected_extension_m=2e-6, physical_extension_m=1.5e-6,
        handoff_guard_diagnostics={"restart_location": label},
        termination_reason=(
            "branch_cluster_independent_tip_handoff_required"
            if label == "independent_tip_handoff" else None
        ),
    )


@pytest.mark.parametrize("location", [
    "before_provider_transition", "immediately_after_provider_transition",
    "immediately_before_branch_birth", "immediately_after_branch_birth",
    "after_A1", "after_A12", "after_rejected_A12", "coalescence",
    "independent_tip_handoff",
])
def test_production_checkpoint_roundtrip_at_required_restart_locations(tmp_path, location):
    original = checkpoint_at(location, tmp_path)
    path = tmp_path / location / "checkpoint.json"
    manifest = write_branch_checkpoint(original, path)
    restored = restore_branch_checkpoint(path)
    assert restored.manifest_fields() == original.manifest_fields()
    assert restored.state.crack_network == original.state.crack_network
    np.testing.assert_array_equal(restored.state.displacement, original.state.displacement)
    np.testing.assert_array_equal(restored.state.damage, original.state.damage)
    assert restored.state.rng_state == original.state.rng_state
    assert restored.front_competitions == original.front_competitions
    assert manifest["handoff_guard_diagnostics"]["restart_location"] == location


def test_checkpoint_write_is_atomic_and_detects_corruption(tmp_path):
    path = tmp_path / "checkpoint.json"
    write_branch_checkpoint(checkpoint_at("after_A1", tmp_path), path)
    manifest = json.loads(path.read_text())
    assert not list(tmp_path.glob("*.tmp"))
    state_path = path.with_name(manifest["state_file"])
    state_path.write_bytes(state_path.read_bytes() + b"corrupt")
    with pytest.raises(ValueError, match="hash mismatch"):
        restore_branch_checkpoint(path)


def test_provider_lock_remains_irreversible_across_restart(tmp_path):
    path = tmp_path / "checkpoint.json"
    write_branch_checkpoint(checkpoint_at("after_A12", tmp_path), path)
    restored = restore_branch_checkpoint(path)
    assert restored.provider_runtime.routing.active_mechanics_provider == PROVIDER_ID
    with pytest.raises(RuntimeError, match="already locked"):
        restored.provider_runtime.transition(
            step=4, state_hash="next", legacy_result={}, request=None,
            protected_state=None,
        )
