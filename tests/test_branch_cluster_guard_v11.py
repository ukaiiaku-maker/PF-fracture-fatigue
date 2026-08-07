import math

from arrhenius_fracture.branch_cluster_v11 import create_unresolved_branch_cluster
from arrhenius_fracture.branch_cluster_guard_v11 import (
    TERMINATION_STATUS,
    evaluate_unresolved_cluster_guard,
)
from arrhenius_fracture.crack_network_v11 import CrackNetworkState, ROOT_BRANCH_ID
from arrhenius_fracture.topology_transaction_v11 import TopologyArm, extend_network_arm, mark_coalesced


def cluster_network(length=0.0, angle=math.pi / 4):
    network = CrackNetworkState.one_tip(((0.0, 0.0), (1.0, 0.0)))
    network, cluster = create_unresolved_branch_cluster(
        network, parent_branch_id=ROOT_BRANCH_ID, candidate_ids=("c1", "c2"),
        event_index=1, shared_process_state={"single_update_count": 0},
        conserved_ledgers={name: 0.0 for name in (
            "retained", "mobile", "escaped", "recovered", "stored_energy",
            "emission_work", "unconsumed_action",
        )},
    )
    if length:
        for branch_id, sign, candidate in zip(cluster.arm_branch_ids, (-1, 1), ("c1", "c2")):
            end = (1.0 + length * math.cos(angle), sign * length * math.sin(angle))
            network = extend_network_arm(
                network, TopologyArm(candidate, branch_id, (1.0, 0.0), end, length, 0.0)
            )
    return network, cluster


def test_below_guard_continues_shared_cluster():
    network, cluster = cluster_network(length=4.9)
    diagnostic = evaluate_unresolved_cluster_guard(
        network, cluster, branch_handoff_length_m=5.0,
        local_J_contour_radius_m=2.0,
        independently_valid_local_J=(True, True),
    )
    assert not diagnostic.handoff_required
    assert diagnostic.termination_status is None
    assert cluster.unresolved


def test_all_existing_scale_conditions_trigger_exact_controlled_stop():
    network, cluster = cluster_network(length=8.0)
    diagnostic = evaluate_unresolved_cluster_guard(
        network, cluster, branch_handoff_length_m=5.0,
        local_J_contour_radius_m=2.0,
        independently_valid_local_J=(True, True),
    )
    assert diagnostic.handoff_required
    assert diagnostic.termination_status == TERMINATION_STATUS
    assert diagnostic.local_contours_overlap is False
    assert diagnostic.to_dict()["guard_scale_contract"]["fitted_parameters"] is False
    assert cluster.unresolved


def test_invalid_contour_or_overlap_keeps_cluster_unresolved():
    network, cluster = cluster_network(length=8.0, angle=0.1)
    diagnostic = evaluate_unresolved_cluster_guard(
        network, cluster, branch_handoff_length_m=1.0,
        local_J_contour_radius_m=2.0,
        independently_valid_local_J=(True, False),
    )
    assert diagnostic.local_contours_overlap
    assert not diagnostic.handoff_required


def test_coalesced_arm_can_never_request_two_tip_handoff():
    network, cluster = cluster_network(length=8.0)
    first, second = cluster.arm_branch_ids
    network = mark_coalesced(network, second, first)
    diagnostic = evaluate_unresolved_cluster_guard(
        network, cluster, branch_handoff_length_m=1.0,
        local_J_contour_radius_m=0.1,
        independently_valid_local_J=(True, True),
    )
    assert diagnostic.independently_valid_local_J == (False, False)
    assert not diagnostic.handoff_required
