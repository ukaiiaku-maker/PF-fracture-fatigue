from dataclasses import replace

from arrhenius_fracture.branch_cluster_v11 import create_unresolved_branch_cluster
from arrhenius_fracture.crack_network_v11 import CrackNetworkState, ROOT_BRANCH_ID
from arrhenius_fracture.directional_competition_v11 import tungsten_cleavage_candidates
from arrhenius_fracture.resolved_tip_state_v11 import resolve_unresolved_cluster
from arrhenius_fracture.topology_transaction_v11 import TopologyArm, extend_network_arm


def _cluster():
    candidates = tungsten_cleavage_candidates(theta_deg=45.0)
    network = replace(
        CrackNetworkState.one_tip(((0.0, 0.0), (5.0e-4, 0.0))),
        branching_enabled=True,
    )
    network, cluster = create_unresolved_branch_cluster(
        network, parent_branch_id=ROOT_BRANCH_ID,
        candidate_ids=tuple(item.candidate_id for item in candidates), event_index=3,
        shared_process_state={"B": 0.72, "N_em": 588.5, "W_emit": 0.59},
        conserved_ledgers={
            "retained": 588.5, "mobile": 2.0, "escaped": 1.0,
            "recovered": 3.0, "stored_energy": 4.0, "emission_work": 0.59,
            "unconsumed_action": 0.72,
        },
    )
    by_id = {item.candidate_id: item for item in candidates}
    for branch_id in cluster.arm_branch_ids:
        branch = network.branch(branch_id)
        candidate = by_id[branch.local_state["candidate_id"]]
        network = extend_network_arm(network, TopologyArm(
            candidate.candidate_id, branch_id, branch.tip,
            (branch.tip[0] + 5e-6 * candidate.direction_xy[0], branch.tip[1] + 5e-6 * candidate.direction_xy[1]),
            5e-6, 0.0,
        ))
    return candidates, network, cluster


def _fresh(branch_id):
    return (
        {"historical_state_imported": False, "B": 0.0, "N_em": 0.0, "W_emit": 0.0},
        {"historical_state_imported": False, "source_inventory": 0.0, "branch_id": branch_id},
    )


def test_handoff_retains_all_history_at_junction_and_creates_fresh_tips():
    candidates, network, cluster = _cluster()
    result = resolve_unresolved_cluster(
        network, cluster, candidates=candidates, global_hazard_seed=3621,
        fresh_tip_factory=_fresh,
    )
    assert not result.cluster.unresolved
    assert result.reservoir.historical_process_state == cluster.shared_process_state
    assert result.reservoir.historical_ledgers == cluster.conserved_ledgers
    assert set(result.tips) == set(cluster.arm_branch_ids)
    assert len({tip.rng_identity for tip in result.tips.values()}) == 2
    for tip in result.tips.values():
        assert tip.process_state["B"] == tip.process_state["N_em"] == tip.process_state["W_emit"] == 0.0
        assert tip.process_state["historical_state_imported"] is False
        assert tip.competition.consumed_event_ids == ()
        assert len(tip.competition.candidates) == len(candidates)
    assert sum(tip.process_state["N_em"] for tip in result.tips.values()) == 0.0


def test_handoff_is_deterministic_under_arm_enumeration():
    candidates, network, cluster = _cluster()
    left = resolve_unresolved_cluster(network, cluster, candidates=candidates, global_hazard_seed=3621, fresh_tip_factory=_fresh)
    right = resolve_unresolved_cluster(network, replace(cluster, arm_branch_ids=tuple(reversed(cluster.arm_branch_ids))), candidates=tuple(reversed(candidates)), global_hazard_seed=3621, fresh_tip_factory=_fresh)
    assert left.network.to_json() == right.network.to_json()
    assert left.reservoir == right.reservoir
    assert left.tips == right.tips


def test_handoff_rejects_any_factory_that_imports_historical_state():
    candidates, network, cluster = _cluster()
    def invalid(_):
        return ({"historical_state_imported": True}, {"historical_state_imported": False})
    try:
        resolve_unresolved_cluster(network, cluster, candidates=candidates, global_hazard_seed=3621, fresh_tip_factory=invalid)
    except ValueError as error:
        assert "reject historical import" in str(error)
    else:
        raise AssertionError("historical child-state import was accepted")
