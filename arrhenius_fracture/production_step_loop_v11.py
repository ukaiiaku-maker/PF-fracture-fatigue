"""Accepted-state controller for bounded v11 mechanistic branching.

The controller is deliberately independent of mesh construction and command-line
policy.  Production adapters supply mechanics, rates, topology arms, and shared
process-state evolution; this module enforces their ordering and transaction
semantics.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
import math
from typing import Callable, Mapping, Sequence

from .directional_competition_v11 import (
    CompetingActionProposal,
    DirectionalCompetitionState,
    DirectionalRate,
    commit_directional_interval,
    construct_action_proposals,
    preview_directional_interval,
    select_temporal_or_degenerate_proposal,
)
from .topology_transaction_v11 import (
    LiveFEMTopologyState,
    TopologyArm,
    TopologyTrialResult,
)


MODEL_ID = "v11.monotonic_tip_only_mechanistic_branching_step_loop/1"


class DirectionalStepRefinementRequired(RuntimeError):
    def __init__(self, predicted_increment: float, target_increment: float):
        self.predicted_increment = float(predicted_increment)
        self.target_increment = float(target_increment)
        super().__init__(
            f"directional action increment {predicted_increment:.6g} exceeds "
            f"adaptive target {target_increment:.6g}"
        )


@dataclass(frozen=True)
class AcceptedStepContext:
    step: int
    physical_time_s: float
    duration_s: float
    accepted_state_id: str

    def __post_init__(self) -> None:
        if self.step < 0 or self.physical_time_s < 0.0 or self.duration_s < 0.0:
            raise ValueError("accepted step coordinates must be nonnegative")
        if not self.accepted_state_id:
            raise ValueError("accepted_state_id is required")


@dataclass(frozen=True)
class ActionTrialDiagnostic:
    proposal: CompetingActionProposal
    result: TopologyTrialResult
    selected: bool = False


@dataclass(frozen=True)
class AcceptedStepResult:
    state: LiveFEMTopologyState
    rates: tuple[DirectionalRate, ...]
    proposals: tuple[CompetingActionProposal, ...]
    trials: tuple[ActionTrialDiagnostic, ...]
    selected_action_id: str | None
    shared_state_update_count: int


SolveAccepted = Callable[[LiveFEMTopologyState, AcceptedStepContext], LiveFEMTopologyState]
EvaluateRates = Callable[[LiveFEMTopologyState, AcceptedStepContext], Sequence[DirectionalRate]]
TrialAction = Callable[[LiveFEMTopologyState, CompetingActionProposal], TopologyTrialResult]
UpdateShared = Callable[
    [LiveFEMTopologyState, AcceptedStepContext, CompetingActionProposal | None],
    LiveFEMTopologyState,
]


def _commit_interval(
    state: DirectionalCompetitionState,
    rates: Sequence[DirectionalRate],
    context: AcceptedStepContext,
) -> DirectionalCompetitionState:
    by_id = {rate.candidate_id: rate for rate in rates}
    expected = {candidate.candidate_id for candidate in state.candidates}
    if set(by_id) != expected:
        raise ValueError("directional rates must map one-to-one to candidates")
    hazards = tuple(
        replace(
            commit_directional_interval(
                hazard,
                preview_directional_interval(
                    hazard,
                    lambda_per_s=_logmean_rate(
                        hazard.previous_rate_per_s,
                        by_id[hazard.candidate_id].lambda_per_s,
                    ),
                    start_time_s=context.physical_time_s,
                    duration_s=context.duration_s,
                ),
            ),
            previous_rate_per_s=by_id[hazard.candidate_id].lambda_per_s,
        )
        for hazard in state.hazard_states
    )
    return replace(state, hazard_states=hazards)


def _logmean_rate(previous: float | None, current: float) -> float:
    """Match the production engine's accepted linear-load hazard quadrature."""
    now = max(float(current), 0.0)
    if previous is None:
        return now
    old = max(float(previous), 0.0)
    lo, hi = sorted((old, now))
    if lo <= 0.0:
        return 0.5 * hi
    if abs(hi - lo) <= 1.0e-12 * hi:
        return hi
    return (hi - lo) / math.log(hi / lo)


def _select(
    diagnostics: Sequence[ActionTrialDiagnostic], state: DirectionalCompetitionState
) -> ActionTrialDiagnostic | None:
    admissible = tuple(item for item in diagnostics if item.result.accepted)
    if not admissible:
        return None
    two_arm = tuple(item for item in admissible if item.proposal.action_type == "two_arm")
    pool = two_arm or tuple(item for item in admissible if item.proposal.action_type == "one_arm")
    chosen = select_temporal_or_degenerate_proposal(
        (item.proposal for item in pool),
        global_hazard_seed=state.global_hazard_seed,
        competition_event_index=state.competition_event_index,
    )
    return next(item for item in pool if item.proposal.action_id == chosen.action_id)


def advance_accepted_step(
    accepted: LiveFEMTopologyState,
    context: AcceptedStepContext,
    *,
    correlation_interval_s: float,
    solve_accepted: SolveAccepted,
    evaluate_directional_rates: EvaluateRates,
    trial_action: TrialAction,
    update_shared_state_once: UpdateShared,
    maximum_directional_action_increment: float | None = None,
) -> AcceptedStepResult:
    """Advance one interval while enforcing the production transaction order.

    Every topology trial receives the identical post-interval accepted object.
    A trial's returned state is ignored until selection, so A1/A2/A12 can never
    form a sequential trial chain.  The shared process state callback is invoked
    exactly once for both event and no-event intervals.
    """
    solved = solve_accepted(accepted, context)
    rates = tuple(sorted(
        evaluate_directional_rates(solved, context), key=lambda item: item.candidate_id
    ))
    if maximum_directional_action_increment is not None:
        target = float(maximum_directional_action_increment)
        if target <= 0.0:
            raise ValueError("maximum directional action increment must be positive")
        previous = {item.candidate_id: item.previous_rate_per_s for item in solved.competition.hazard_states}
        predicted = max(
            (_logmean_rate(previous[item.candidate_id], item.lambda_per_s) * context.duration_s for item in rates),
            default=0.0,
        )
        if predicted > target:
            raise DirectionalStepRefinementRequired(predicted, target)
    competition = _commit_interval(solved.competition, rates, context)
    interval_state = replace(solved, competition=competition)
    proposals = construct_action_proposals(
        competition.hazard_states, correlation_interval_s=correlation_interval_s
    )

    diagnostics = tuple(
        ActionTrialDiagnostic(proposal, trial_action(interval_state, proposal))
        for proposal in proposals
    )
    for item in diagnostics:
        if item.result.accepted and item.result.state is interval_state:
            raise RuntimeError("an accepted topology trial did not return copied committed state")
        if not item.result.accepted and item.result.state is not interval_state:
            raise RuntimeError("a rejected topology trial mutated the accepted state")

    selected = _select(diagnostics, competition)
    selected_proposal = selected.proposal if selected is not None else None
    selected_state = selected.result.state if selected is not None else interval_state
    if selected is not None and not set(selected.proposal.member_event_ids).issubset(
        selected_state.competition.consumed_event_ids
    ):
        raise RuntimeError("selected topology action did not consume exactly its reserved events")
    updated = update_shared_state_once(selected_state, context, selected_proposal)
    marked = tuple(
        replace(item, selected=(selected is not None and item.proposal.action_id == selected.proposal.action_id))
        for item in diagnostics
    )
    return AcceptedStepResult(
        state=updated,
        rates=rates,
        proposals=proposals,
        trials=marked,
        selected_action_id=(selected.proposal.action_id if selected is not None else None),
        shared_state_update_count=1,
    )


__all__ = [
    "AcceptedStepContext", "AcceptedStepResult", "ActionTrialDiagnostic",
    "DirectionalStepRefinementRequired", "MODEL_ID",
    "advance_accepted_step",
]
