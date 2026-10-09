"""Global accepted-step transaction over independent directional tip clocks."""
from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
from typing import Callable, Mapping, Sequence

from .directional_competition_v11 import (
    CompetingActionProposal, DirectionalCompetitionState, DirectionalRate,
    construct_action_proposals,
)
from .production_step_loop_v11 import (
    AcceptedStepContext, ActionTrialDiagnostic, DirectionalStepRefinementRequired,
    _commit_interval, _logmean_rate,
)
from .topology_transaction_v11 import LiveFEMTopologyState, TopologyTrialResult


@dataclass(frozen=True)
class TipActionTrial:
    tip_id: str
    diagnostic: ActionTrialDiagnostic


@dataclass(frozen=True)
class MultiTipStepResult:
    state: LiveFEMTopologyState
    competitions: Mapping[str, DirectionalCompetitionState]
    rates_by_tip: Mapping[str, tuple[DirectionalRate, ...]]
    trials: tuple[TipActionTrial, ...]
    selected_tip_id: str | None
    selected_proposal: CompetingActionProposal | None


Trial = Callable[[LiveFEMTopologyState, str, CompetingActionProposal], TopologyTrialResult]


def _global_choice(
    trials: Sequence[TipActionTrial], *, global_seed: int, event_index: int,
) -> TipActionTrial | None:
    admissible = [item for item in trials if item.diagnostic.result.accepted]
    if not admissible:
        return None
    two = [item for item in admissible if item.diagnostic.proposal.action_type == "two_arm"]
    pool = two or admissible
    earliest = min(min(item.diagnostic.proposal.completion_times_s) for item in pool)
    tied = [item for item in pool if min(item.diagnostic.proposal.completion_times_s) <= earliest + 1e-12]
    return min(tied, key=lambda item: hashlib.sha256(
        f"{global_seed}|{event_index}|{item.tip_id}|{item.diagnostic.proposal.action_id}".encode()
    ).digest())


def advance_multi_tip_step(
    accepted: LiveFEMTopologyState,
    competitions: Mapping[str, DirectionalCompetitionState],
    context: AcceptedStepContext,
    *,
    correlation_interval_s: float,
    solve_accepted: Callable[[LiveFEMTopologyState, AcceptedStepContext], LiveFEMTopologyState],
    evaluate_rates: Callable[[LiveFEMTopologyState, AcceptedStepContext], Mapping[str, Sequence[DirectionalRate]]],
    trial_action: Trial,
    update_process_states: Callable[[LiveFEMTopologyState, AcceptedStepContext, str | None, CompetingActionProposal | None], LiveFEMTopologyState],
    maximum_directional_action_increment: float | None = None,
) -> MultiTipStepResult:
    active = set(accepted.crack_network.active_tip_ids)
    if set(competitions) != active:
        raise ValueError("multi-tip competitions must map one-to-one to active tips")
    solved = solve_accepted(accepted, context)
    raw_rates = evaluate_rates(solved, context)
    if set(raw_rates) != active:
        raise ValueError("multi-tip rates must map one-to-one to active tips")
    rates_by_tip = {tip: tuple(sorted(raw_rates[tip], key=lambda item: item.candidate_id)) for tip in sorted(active)}
    if maximum_directional_action_increment is not None:
        target = float(maximum_directional_action_increment)
        predicted = max(
            _logmean_rate(hazard.previous_rate_per_s, rate.lambda_per_s) * context.duration_s
            for tip, competition in competitions.items()
            for hazard in competition.hazard_states
            for rate in rates_by_tip[tip]
            if rate.candidate_id == hazard.candidate_id
        )
        if predicted > target:
            raise DirectionalStepRefinementRequired(predicted, target)
    committed = {
        tip: _commit_interval(competition, rates_by_tip[tip], context)
        for tip, competition in competitions.items()
    }
    trials = []
    for tip in sorted(active):
        interval_state = replace(solved, competition=committed[tip])
        rates_by_candidate = {
            item.candidate_id: item for item in rates_by_tip[tip]
        }
        for proposal in construct_action_proposals(
            committed[tip].hazard_states, correlation_interval_s=correlation_interval_s
        ):
            # Completion of a directional first-passage clock is kinetic
            # history, not an irrevocable authorization to extend a crack.
            # Revalidate every pending member against the current, fully
            # equilibrated crack-network mechanics before constructing a
            # topology trial.  The event remains pending when this veto fires.
            current_drives = tuple(
                rates_by_candidate[candidate_id].signed_J_J_per_m2
                for candidate_id in proposal.member_candidate_ids
            )
            if any(value <= 0.0 for value in current_drives):
                result = TopologyTrialResult(
                    False, interval_state, proposal.action_id,
                    0.0, 0.0, 0.0,
                    "current_signed_directional_J_nonpositive",
                )
            else:
                result = trial_action(interval_state, tip, proposal)
            if not result.accepted and result.state is not interval_state:
                raise RuntimeError("a rejected multi-tip trial mutated accepted state")
            trials.append(TipActionTrial(tip, ActionTrialDiagnostic(proposal, result)))
    seed = min(item.global_hazard_seed for item in committed.values())
    index = sum(item.competition_event_index for item in committed.values())
    selected = _global_choice(trials, global_seed=seed, event_index=index)
    selected_tip = selected.tip_id if selected else None
    proposal = selected.diagnostic.proposal if selected else None
    state = selected.diagnostic.result.state if selected else solved
    if selected:
        committed[selected.tip_id] = state.competition
    state = update_process_states(state, context, selected_tip, proposal)
    marked = tuple(
        replace(item, diagnostic=replace(
            item.diagnostic,
            selected=(selected is not None and item.tip_id == selected.tip_id and item.diagnostic.proposal.action_id == selected.diagnostic.proposal.action_id),
        ))
        for item in trials
    )
    return MultiTipStepResult(state, committed, rates_by_tip, marked, selected_tip, proposal)


__all__ = ["MultiTipStepResult", "TipActionTrial", "advance_multi_tip_step"]
