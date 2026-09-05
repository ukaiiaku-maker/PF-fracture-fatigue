from __future__ import annotations

from dataclasses import replace
import copy
import pickle
import random
from types import SimpleNamespace

import pytest
import numpy as np

from arrhenius_fracture.crack_network_v11 import CrackBranchState, CrackNetworkState
from arrhenius_fracture.tip_directional_observation_v11 import (
    apply_post_interval_event_renewal,
    candidate_tip_owners,
    marginal_trial_origin,
    observations_from_provider,
    observations_from_provider_by_front_candidate,
    require_same_tip_coupling,
    require_observation_state_contract,
    require_uncontaminated_replay_checkpoint,
    realized_topology_arm_lengths,
    select_shared_process_renewal_distance,
    select_controlling_observation,
    selected_event_owner,
    serialize_directional_observation,
)


C010 = "(010)[10-1]"
C100 = "(100)[01-1]"


def two_tip_network(*, reverse_active=False, rename=False):
    root = CrackBranchState(
        "root", None, 0, 0, ((0.0, 0.0), (1.0, 0.0)), (0.0,),
        status="terminated",
    )
    left_id, right_id = ("branch-1", "branch-2") if rename else ("zz-branch", "aa-branch")
    left = CrackBranchState(
        left_id, "root", 1, 1, ((1.0, 0.0), (2.0, 1.0)), (0.785,),
        local_state={"candidate_id": C010},
    )
    right = CrackBranchState(
        right_id, "root", 1, 1, ((1.0, 0.0), (3.0, -1.0)), (-0.464,),
        local_state={"candidate_id": C100},
    )
    return CrackNetworkState(
        (root, right, left) if reverse_active else (root, left, right),
        primary_branch_id="root", branching_enabled=True
    )


def provider_tips(reverse=False):
    rows = [
        {
            "tip_xy_m": [2.0, 1.0],
            "directional": [{
                "candidate_id": C010, "J_local_signed_J_per_m2": 10.0,
                "K_directional_Pa_sqrt_m": 100.0, "local_J_valid": True,
            }],
        },
        {
            "tip_xy_m": [3.0, -1.0],
            "directional": [{
                "candidate_id": C100, "J_local_signed_J_per_m2": 40.0,
                "K_directional_Pa_sqrt_m": 200.0, "local_J_valid": False,
                "local_J_invalid_reason": "contour",
            }],
        },
    ]
    return tuple(reversed(rows)) if reverse else tuple(rows)


def observations(reverse=False, *, network=None):
    result = observations_from_provider(
        two_tip_network() if network is None else network,
        provider_tips(reverse), (C010, C100),
        accepted_state_id="pre", topology_fingerprint="top",
    )
    return tuple(
        item.with_kinetics(
            kinetic_J_J_per_m2=max(item.signed_J_J_per_m2, 0.0),
            marginal_J_J_per_m2=(44.0 if item.candidate_id == C100 else None),
            directional_K_Pa_sqrt_m=item.directional_K_Pa_sqrt_m,
            rate_per_s=(2.0 if item.candidate_id == C100 else 1.0),
        )
        for item in result
    )


def test_candidate_to_tip_bijection_and_actual_rate_tip_serialization():
    network = two_tip_network()
    owners = candidate_tip_owners(network, (C010, C100))
    assert owners == {C010: "zz-branch", C100: "aa-branch"}
    rows = observations()
    control = select_controlling_observation(rows)
    serialized = [
        serialize_directional_observation(
            item, controlling_scalar_K_tip_id=control.tip_id,
            tensor_probe_tip_id=control.tip_id, selected_event_tip_id="zz-branch",
            process_owner_id="cluster", pre_event_state_id="pre",
            post_event_state_id="post",
        )
        for item in rows
    ]
    assert {item["tip_id"] for item in serialized} == {"aa-branch", "zz-branch"}
    assert all(item["controlling_scalar_K_tip_id"] == item["tensor_probe_tip_id"] for item in serialized)
    required = {
        "accepted_state_id", "stress_field_state_id",
        "pre_event_topology_fingerprint", "tip_id", "candidate_id", "tip_xy_m",
        "signed_local_J_J_per_m2", "kinetic_J_J_per_m2",
        "marginal_J_J_per_m2", "directional_K_Pa_sqrt_m",
        "directional_rate_per_s", "local_contour_valid",
        "local_contour_invalid_reason",
    }
    assert required <= set(serialized[0])


def test_ownership_and_control_are_invariant_to_tip_enumeration_and_branch_lexical_order():
    forward = observations(False)
    reverse = observations(True, network=two_tip_network(reverse_active=True))
    assert forward == reverse
    assert select_controlling_observation(forward).candidate_id == C100
    assert select_controlling_observation(reverse).tip_id == "aa-branch"
    renamed = observations(network=two_tip_network(rename=True))
    assert select_controlling_observation(renamed).candidate_id == C100
    assert select_controlling_observation(renamed).directional_K_Pa_sqrt_m == 200.0


def test_invalid_contour_marginal_trial_uses_candidate_owner_not_first_tip():
    network = two_tip_network()
    invalid = replace(
        next(item for item in observations() if item.candidate_id == C010),
        local_contour_valid=False,
        local_contour_invalid_reason="contour",
    )
    assert invalid.local_contour_valid is False
    tip_id, tip_xy = marginal_trial_origin(network, invalid)
    assert network.active_tip_ids[0] == "aa-branch"
    assert tip_id == "zz-branch"
    assert tip_xy == (2.0, 1.0)


def test_ambiguous_or_missing_multi_tip_ownership_fails_closed():
    network = two_tip_network()
    bad = replace(
        network,
        branches=tuple(
            replace(branch, local_state={"candidate_id": C010})
            if branch.branch_id == "aa-branch" else branch
            for branch in network.branches
        ),
    )
    with pytest.raises(RuntimeError, match="duplicate tip owners"):
        candidate_tip_owners(bad, (C010, C100))
    with pytest.raises(RuntimeError, match="physical owners"):
        observations_from_provider(
            network, tuple(reversed(provider_tips())), (C010, C100),
            accepted_state_id="pre", topology_fingerprint="top",
            coordinate_tolerance_m=10.0,
        )


def test_independent_fronts_can_each_observe_the_full_candidate_inventory():
    network = two_tip_network()
    tips = []
    for tip_id in network.active_tip_ids:
        branch = network.branch(tip_id)
        tips.append({
            "tip_xy_m": list(branch.tip),
            "directional": [
                {
                    "candidate_id": candidate_id,
                    "J_local_signed_J_per_m2": value,
                    "K_directional_Pa_sqrt_m": value * 10.0,
                    "local_J_valid": True,
                }
                for candidate_id, value in ((C010, 10.0), (C100, 20.0))
            ],
        })
    rows = observations_from_provider_by_front_candidate(
        network, tips,
        {tip_id: (C010, C100) for tip_id in network.active_tip_ids},
        accepted_state_id="accepted", stress_field_state_id="stress",
        topology_fingerprint="topology",
    )
    assert {(row.tip_id, row.candidate_id) for row in rows} == {
        (tip_id, candidate_id)
        for tip_id in network.active_tip_ids
        for candidate_id in (C010, C100)
    }


def test_same_tip_scalar_tensor_and_event_identity_are_independent():
    rows = observations()
    control = select_controlling_observation(rows)
    require_same_tip_coupling(control.tip_id, "aa-branch")
    with pytest.raises(RuntimeError, match="mixed scalar-K"):
        require_same_tip_coupling(control.tip_id, "zz-branch")
    proposal = SimpleNamespace(member_candidate_ids=(C010,))
    candidate, tip = selected_event_owner(proposal, rows)
    assert (candidate, tip) == (C010, "zz-branch")
    assert tip != control.tip_id


def test_observation_requires_same_accepted_stress_and_pre_event_topology_state():
    control = select_controlling_observation(observations())
    require_observation_state_contract(
        control, accepted_state_id="pre", stress_field_state_id="pre",
        pre_event_topology_fingerprint="top",
    )
    with pytest.raises(RuntimeError, match="stress-field identity mismatch"):
        require_observation_state_contract(
            control, accepted_state_id="pre", stress_field_state_id="other",
            pre_event_topology_fingerprint="top",
        )
    with pytest.raises(RuntimeError, match="pre-event topology mismatch"):
        require_observation_state_contract(
            control, accepted_state_id="pre", stress_field_state_id="pre",
            pre_event_topology_fingerprint="post",
        )


class ProcessState:
    def __init__(self):
        self.calls = []
        self.mobile = np.array([[4.0], [6.0]])
        self.retained = np.array([[1.0], [2.0]])
        self.wake_mobile = np.array([[0.0], [0.0]])
        self.wake_retained = np.array([[0.0], [0.0]])

    def advance(self, distance):
        self.calls.append(distance)
        self.wake_mobile += self.mobile
        self.wake_retained += self.retained
        self.mobile[:] = 0.0
        self.retained[:] = 0.0
        return {"wake_mobile": 3.0}


def test_event_renewal_occurs_once_after_interval_and_refuses_double_advance():
    state = ProcessState()
    info = {"da": 0.0, "interval_evolved": True}
    result = apply_post_interval_event_renewal(
        state, info, event_selected=True, event_distance_m=5.0e-6
    )
    assert state.calls == [5.0e-6]
    assert result == {"wake_mobile": 3.0}
    assert info["event_moving_frame_renewal_count"] == 1
    audit = info["event_moving_frame_conservation"]
    assert audit["active_mobile_change_by_system"] == [-4.0, -6.0]
    assert audit["wake_mobile_change_by_system"] == [4.0, 6.0]
    assert audit["active_plus_wake_plus_sinks_residual"] == 0.0
    with pytest.raises(RuntimeError, match="already applied"):
        apply_post_interval_event_renewal(
            state, info, event_selected=True, event_distance_m=5.0e-6
        )
    with pytest.raises(RuntimeError, match="double-apply"):
        apply_post_interval_event_renewal(
            ProcessState(), {"da": 1.0e-9}, event_selected=True,
            event_distance_m=5.0e-6,
        )


def test_non_event_has_no_renewal_and_event_cannot_precede_interval_evolution():
    state = ProcessState()
    info = {"da": 0.0, "interval_evolved": True}
    assert apply_post_interval_event_renewal(
        state, info, event_selected=False, event_distance_m=5.0e-6
    ) == {}
    assert state.calls == []
    assert info["event_moving_frame_renewal_count"] == 0
    with pytest.raises(RuntimeError, match="completed pre-event interval"):
        apply_post_interval_event_renewal(
            ProcessState(), {"da": 0.0}, event_selected=True,
            event_distance_m=5.0e-6,
        )


def test_v4_replay_source_refuses_first_event_and_all_later_checkpoints():
    require_uncontaminated_replay_checkpoint(1, first_selected_event_step=287)
    for step in (287, 288, 369, 2182):
        with pytest.raises(RuntimeError, match="defect-affected process-state history"):
            require_uncontaminated_replay_checkpoint(
                step, first_selected_event_step=287,
            )


def test_realized_one_arm_renewal_uses_accepted_geometry_not_nominal_length():
    pre = CrackNetworkState.one_tip(((0.0, 0.0), (1.0, 0.0)))
    root = pre.branch(pre.active_tip_ids[0])
    post = replace(pre, branches=(replace(
        root, path=root.path + ((1.3, 0.4),),
        orientation_history_rad=root.orientation_history_rad + (0.9272952180016123,),
    ),))
    lengths = realized_topology_arm_lengths(pre, post, (C010,))
    assert lengths == pytest.approx((0.5,))
    assert select_shared_process_renewal_distance(lengths) == pytest.approx(0.5)


def test_two_arm_shared_renewal_uses_max_tip_advance_not_total_crack_length():
    pre = CrackNetworkState.one_tip(((0.0, 0.0), (1.0, 0.0)))
    post = two_tip_network()
    lengths = realized_topology_arm_lengths(pre, post, (C010, C100))
    assert lengths == pytest.approx((2.0 ** 0.5, 5.0 ** 0.5))
    assert select_shared_process_renewal_distance(lengths) == pytest.approx(5.0 ** 0.5)
    assert select_shared_process_renewal_distance(lengths) != pytest.approx(sum(lengths))


def test_diagnostic_observation_build_is_state_and_rng_pure():
    network = two_tip_network()
    before = pickle.dumps(network, protocol=5)
    provider = provider_tips()
    provider_before = copy.deepcopy(provider)
    rng = random.Random(3621)
    rng_before = rng.getstate()
    observations_from_provider(
        network, provider, (C010, C100), accepted_state_id="pre",
        topology_fingerprint="top",
    )
    assert pickle.dumps(network, protocol=5) == before
    assert provider == provider_before
    assert rng.getstate() == rng_before
