from dataclasses import replace
import json
import pickle

import pytest

from arrhenius_fracture.crack_network_v11 import CrackNetworkState
from arrhenius_fracture.general_multifront_v12 import (
    FrontRuntimeState, MultiFrontRuntimeState, ProcessEngineState,
    ResourcePolicy, TopologyProposal, canonical_hash, commit_selected_proposal,
)
from arrhenius_fracture.multifront_checkpoint_v12 import (
    checkpoint_bytes, checkpoint_field_hashes, load_multifront_checkpoint,
    write_multifront_checkpoint,
)


def state_with_fronts(count):
    network = CrackNetworkState.one_tip(((0.0, 0.0),), initial_orientation_rad=0.0)
    front = network.active_tip_ids[0]
    runtime = MultiFrontRuntimeState.one_front(
        network,
        FrontRuntimeState(
            front, {"action": 0.25, "threshold": 1.5, "ordinal": 4},
            ("a", "b"), {"seed": 91, "rng": [3, 2, 1]}, interval_count=7,
        ),
        ProcessEngineState(
            "engine:root", "family:qualified",
            {"mobile": 3.0}, {"retained": 2.0}, {"system:+": 5.0},
            update_count=7, event_renewal_count=3,
            local_process_coordinate_m=10e-6,
            opaque_state_fingerprint="exact",
            mutable_state={"B": 0.5, "N_em": 4.0},
            rng_state={"bit_generator": "source-fixture", "position": 17},
        ),
        resource_policy=ResourcePolicy(
            "mechanistic", None, None, "experimental",
            "v11_correlated_proposal_compatibility",
        ), accepted_state_id="accepted", stress_field_state_id="stress",
    )
    runtime = replace(
        runtime,
        compatibility_provenance={
            "family_identity": "v5.3:745um:exact-topology",
            "exact_prefix_policy": "v5_4_2_exact_prefix_immutable",
        },
        output_counters={"front_rows": 8, "owner_rows": 7, "checkpoints": 3},
    )
    for index in range(count - 1):
        parent = runtime.active_front_ids[index % len(runtime.active_front_ids)]
        tip = runtime.crack_network.branch(parent).tip
        proposal = TopologyProposal(
            f"p:{index}", "two_arm", parent, runtime.owner_by_front[parent],
            ("a", "b"), index + 1.0,
            ((tip[0] + 1e-6, tip[1] - 1e-8), (tip[0] + 1e-6, tip[1] + 1e-8)),
            member_event_ids=(f"a#{index}", f"b#{index}"),
            member_event_ordinals=(index + 1, index + 1),
        )
        runtime = commit_selected_proposal(runtime, proposal)
    return runtime


@pytest.mark.parametrize("count", (1, 2, 8))
def test_checkpoint_explicit_pre_post_hashes_and_zero_time_restore(tmp_path, count):
    runtime = state_with_fronts(count)
    reference = {
        "kind": "immutable_accepted_fem_state",
        "mesh_sha256": "a" * 64,
        "state_sha256": "b" * 64,
    }
    before = checkpoint_field_hashes(runtime, accepted_fem_state_reference=reference)
    rng_before = canonical_hash({
        key: value.lineage_rng_state for key, value in runtime.front_runtimes.items()
    })
    target = tmp_path / f"n{count}.json"
    write_multifront_checkpoint(
        runtime, target, accepted_fem_state_reference=reference,
    )
    restored = load_multifront_checkpoint(target)
    after = checkpoint_field_hashes(
        restored.runtime,
        accepted_fem_state_reference=restored.accepted_fem_state_reference,
    )
    assert before == after == restored.field_hashes
    assert restored.runtime.scheduler == runtime.scheduler
    assert restored.runtime.output_counters == runtime.output_counters
    assert canonical_hash({
        key: value.lineage_rng_state
        for key, value in restored.runtime.front_runtimes.items()
    }) == rng_before
    assert len({id(item) for item in restored.runtime.process_engines.values()}) == len(
        restored.runtime.process_engines
    )
    assert restored.runtime.total_conserved_ledgers() == runtime.total_conserved_ledgers()
    assert restored.runtime.total_signed_system_ledgers() == runtime.total_signed_system_ledgers()


def test_checkpoint_bytes_are_order_invariant_and_deterministic():
    runtime = state_with_fronts(8)
    reference = {"z": 2, "a": 1}
    reordered = replace(
        runtime,
        front_runtimes=dict(reversed(tuple(runtime.front_runtimes.items()))),
        owner_by_front=dict(reversed(tuple(runtime.owner_by_front.items()))),
        process_regions=dict(reversed(tuple(runtime.process_regions.items()))),
        process_engines=dict(reversed(tuple(runtime.process_engines.items()))),
        junctions=dict(reversed(tuple(runtime.junctions.items()))),
    )
    first = checkpoint_bytes(runtime, accepted_fem_state_reference=reference)
    second = checkpoint_bytes(
        reordered, accepted_fem_state_reference=dict(reversed(tuple(reference.items()))),
    )
    assert first == second == checkpoint_bytes(
        runtime, accepted_fem_state_reference=reference,
    )
    payload = json.loads(first)
    assert set(payload["field_hashes"]["mutable_engines"]) == set(runtime.process_engines)
    assert set(payload["field_hashes"]["front_rng"]) == set(runtime.active_front_ids)


def test_pickle_and_json_roundtrips_preserve_no_engine_aliasing():
    runtime = state_with_fronts(8)
    pickled = pickle.loads(pickle.dumps(runtime, protocol=5))
    json_roundtrip = MultiFrontRuntimeState.from_dict(runtime.to_dict())
    assert pickled.to_dict() == json_roundtrip.to_dict() == runtime.to_dict()
    assert len({id(item) for item in pickled.process_engines.values()}) == len(
        pickled.process_engines
    )
