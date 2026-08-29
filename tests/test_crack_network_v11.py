import copy
import json
import math

import pytest
import numpy as np

from arrhenius_fracture.crack_network_v11 import (
    CrackBranchState,
    CrackNetworkState,
    ROOT_BRANCH_ID,
)
from arrhenius_fracture.stochastic_hazard_tip import draw_hazard_threshold


def test_one_tip_compatibility_container_is_branching_disabled_by_default():
    network = CrackNetworkState.one_tip([(0.0, 0.0), (3.0, 4.0), (4.0, 4.0)])

    assert network.branching_enabled is False
    assert network.active_tip_ids == (ROOT_BRANCH_ID,)
    assert network.total_physical_crack_length_m == pytest.approx(6.0)
    assert network.primary_projected_extension_m == pytest.approx(4.0)
    assert network.branch(ROOT_BRANCH_ID).tip == (4.0, 4.0)
    assert network.branch(ROOT_BRANCH_ID).orientation_history_rad == pytest.approx(
        (math.atan2(4.0, 3.0), 0.0)
    )


def test_canonical_serialization_round_trip_and_input_copy():
    state = {"shielding": {"signed_K": -2.5}, "event_index": 7}
    path = [[0.0, 0.0], [1.0, 0.0]]
    network = CrackNetworkState.one_tip(path, local_state=state)
    path[1][0] = 99.0
    state["event_index"] = 99

    encoded = network.to_json()
    restored = CrackNetworkState.from_dict(json.loads(encoded))

    assert restored.to_json() == encoded
    assert restored.branch(ROOT_BRANCH_ID).tip == (1.0, 0.0)
    assert restored.branch(ROOT_BRANCH_ID).local_state["event_index"] == 7


def test_constructor_sorts_branch_ids_and_validates_parent_generation():
    root = CrackBranchState(
        ROOT_BRANCH_ID, None, 0, 0, ((0.0, 0.0), (1.0, 0.0)), (0.0,), status="arrested"
    )
    child = CrackBranchState(
        "b00000001", ROOT_BRANCH_ID, 1, 2,
        ((1.0, 0.0), (2.0, 1.0)), (math.pi / 4.0,),
    )
    network = CrackNetworkState(
        branches=(child, root), branching_enabled=True, geometry_generation=1
    )
    assert tuple(branch.branch_id for branch in network.branches) == (
        ROOT_BRANCH_ID, "b00000001"
    )


@pytest.mark.parametrize(
    "mutation, message",
    [
        (lambda data: data.update(schema="wrong"), "unsupported"),
        (lambda data: data.update(active_tip_ids=[]), "active-tip"),
        (lambda data: data.update(total_physical_crack_length_m=9.0), "accounting"),
        (lambda data: data["branches"][0].update(parent_branch_id="missing"), "root"),
    ],
)
def test_malformed_serialized_network_fails_closed(mutation, message):
    payload = CrackNetworkState.one_tip([(0.0, 0.0), (1.0, 0.0)]).to_dict()
    payload = copy.deepcopy(payload)
    mutation(payload)
    with pytest.raises(ValueError, match=message):
        CrackNetworkState.from_dict(payload)


def test_disabled_network_rejects_extra_branch():
    root = CrackBranchState(
        ROOT_BRANCH_ID, None, 0, 0, ((0.0, 0.0),), (0.0,)
    )
    child = CrackBranchState(
        "b00000001", ROOT_BRANCH_ID, 1, 1, ((0.0, 0.0),), (0.5,)
    )
    with pytest.raises(ValueError, match="exactly one"):
        CrackNetworkState(branches=(root, child), branching_enabled=False)


def test_one_tip_container_construction_does_not_consume_hazard_rng():
    def trace(with_container):
        rng = np.random.default_rng(np.random.SeedSequence([1720, 1]))
        first = draw_hazard_threshold("exponential", rng)
        if with_container:
            CrackNetworkState.one_tip([(0.0, 0.0), (5.0e-6, 0.0)])
        second = draw_hazard_threshold("exponential", rng)
        return first, second, copy.deepcopy(rng.bit_generator.state)

    assert trace(with_container=True) == trace(with_container=False)
