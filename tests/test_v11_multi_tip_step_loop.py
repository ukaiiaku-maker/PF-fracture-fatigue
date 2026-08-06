from dataclasses import replace

from arrhenius_fracture.directional_competition_v11 import (
    DirectionalCompetitionState, DirectionalRate, accept_reservation,
    reserve_action, tungsten_cleavage_candidates,
)
from arrhenius_fracture.multi_tip_step_loop_v11 import advance_multi_tip_step
from arrhenius_fracture.production_step_loop_v11 import AcceptedStepContext
from arrhenius_fracture.topology_transaction_v11 import TopologyTrialResult
from tests.test_topology_transaction_v11 import fem_state


def test_two_tip_clocks_advance_independently_and_only_selected_events_consume():
    state = fem_state()
    root = state.crack_network.branch(state.crack_network.active_tip_ids[0])
    from arrhenius_fracture.crack_network_v11 import CrackBranchState
    children = (
        CrackBranchState("left", root.branch_id, 1, 1, (root.tip,), (0.2,)),
        CrackBranchState("right", root.branch_id, 1, 1, (root.tip,), (-0.2,)),
    )
    network = replace(state.crack_network, branches=(replace(root, status="terminated"),) + children, branching_enabled=True)
    state = replace(state, crack_network=network)
    candidates = tungsten_cleavage_candidates(theta_deg=45.0)
    competitions = {
        tip: DirectionalCompetitionState.initialize(candidates, global_hazard_seed=100 + index)
        for index, tip in enumerate(network.active_tip_ids)
    }
    trial_origins = []
    def trial(current, tip, proposal):
        trial_origins.append((tip, id(current.crack_network)))
        reserved = reserve_action(current.competition, proposal, event_rewards_m=(1.0,) * len(proposal.member_event_ids))
        consumed = accept_reservation(reserved, proposal.action_id)
        return TopologyTrialResult(True, replace(current, competition=consumed, event_counters={"tip": tip}), proposal.action_id, 2.0, 1.0, 1.0, None)
    result = advance_multi_tip_step(
        state, competitions, AcceptedStepContext(1, 0.0, 2.0, "multi"),
        correlation_interval_s=1e-6,
        solve_accepted=lambda current, context: current,
        evaluate_rates=lambda current, context: {
            tip: tuple(DirectionalRate(c.candidate_id, 0.6 if tip == "left" else 0.55, 1.0, 1.0, 1.0, c.gamma_rel) for c in candidates)
            for tip in network.active_tip_ids
        },
        trial_action=trial,
        update_process_states=lambda current, context, tip, proposal: current,
    )
    assert set(result.competitions) == {"left", "right"}
    assert result.selected_tip_id in {"left", "right"}
    assert result.selected_proposal is not None
    selected = result.competitions[result.selected_tip_id]
    other = result.competitions[({"left", "right"} - {result.selected_tip_id}).pop()]
    assert selected.consumed_event_ids
    assert not other.consumed_event_ids
    assert len({origin for _, origin in trial_origins}) == 1
