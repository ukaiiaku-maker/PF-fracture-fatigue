"""Use the existing exact pair geometry/energy trial for a conditional mark.

The legacy trial API requires a two-member reservation ledger. V13 supplies a
private topology-reservation ledger for that API, never a second authoritative
cleavage clock. The returned state retains the parent's accepted clock/RNG and
process bookkeeping. Geometry, energy acceptance, and wake operations are the
unchanged callbacks used by the existing topology transaction.
"""
from dataclasses import dataclass, replace
import math

from .directional_competition_v11 import (
    CompletedDirectionalEvent, DirectionalCompetitionState, DirectionalHazardState,
    construct_action_proposals,
)
from .topology_transaction_v11 import execute_topology_trial


@dataclass(frozen=True)
class MarkedPairTrialResult:
    accepted: bool
    state: object
    rejection_reason: str | None
    energy_release_J_per_m: float
    hazard_dissipation_J_per_m: float
    energy_margin_J_per_m: float
    reservation_namespace: str


def trial_conditional_pair(*, pre_event_state, baseline_single_state, parent, mark,
                           arms, apply_trial_geometry, equilibrate_fixed_load,
                           network_geometry_already_realized=False):
    if mark.companion_id is None or mark.companion_id == parent.primary_candidate_id:
        raise ValueError("pair requires a distinct selected companion")
    if parent.primary_event_id not in baseline_single_state.competition.consumed_event_ids:
        raise ValueError("primary cleavage has not been accepted by the canonical parent")
    ids = {parent.primary_candidate_id, mark.companion_id}
    arms = tuple(arms)
    if len(arms) != 2 or {a.candidate_id for a in arms} != ids:
        raise ValueError("exact pair arms must contain the immutable primary and selected companion")
    primary_arm = next(a for a in arms if a.candidate_id == parent.primary_candidate_id)
    baseline_tips = {baseline_single_state.crack_network.branch(key).tip
                     for key in baseline_single_state.crack_network.active_tip_ids}
    if tuple(primary_arm.end_xy_m) not in baseline_tips:
        raise ValueError("pair must preserve the exact accepted primary endpoint")
    inventory = pre_event_state.competition.candidates
    if not ids.issubset({c.candidate_id for c in inventory}):
        raise ValueError("companion is outside the existing crystallographic inventory")
    hazards = []
    for candidate in inventory:
        if candidate.candidate_id in ids:
            ordinal = parent.primary_event_ordinal if candidate.candidate_id == parent.primary_candidate_id else 1
            event = CompletedDirectionalEvent(candidate.candidate_id, ordinal, parent.accepted_time_s, 0.0, 1.0, 1.0)
            hazards.append(DirectionalHazardState(candidate.candidate_id, action=1.0,
                completed_event_count=ordinal, current_threshold_action=2.0, pending_events=(event,),
                last_completion_time_s=parent.accepted_time_s))
        else:
            hazards.append(DirectionalHazardState(candidate.candidate_id))
    ledger = DirectionalCompetitionState(inventory, tuple(hazards), global_hazard_seed=0)
    proposal = next(p for p in construct_action_proposals(ledger.hazard_states, correlation_interval_s=0.0)
                    if p.action_type == "two_arm" and set(p.member_candidate_ids) == ids)
    isolated = replace(pre_event_state.isolated_copy(), competition=ledger)
    trial = execute_topology_trial(isolated, proposal, arms,
        apply_trial_geometry=apply_trial_geometry, equilibrate_fixed_load=equilibrate_fixed_load,
        network_geometry_already_realized=network_geometry_already_realized)
    state = baseline_single_state
    if trial.accepted:
        if not all(math.isfinite(value) for value in (trial.energy_release_J_per_m,
                trial.hazard_dissipation_J_per_m, trial.energy_margin_J_per_m)):
            raise ValueError("nonfinite exact pair energy")
        ledgers = dict(baseline_single_state.energy_ledgers)
        for key in ("topology_release_J_per_m", "hazard_dissipation_J_per_m"):
            ledgers[key] = trial.state.energy_ledgers[key]
        state = replace(trial.state,
            competition=baseline_single_state.competition,
            rng_state=baseline_single_state.rng_state,
            tip_process_state=baseline_single_state.tip_process_state,
            event_counters=baseline_single_state.event_counters,
            energy_ledgers=ledgers,
        )
    return MarkedPairTrialResult(trial.accepted, state, trial.rejection_reason,
        trial.energy_release_J_per_m, trial.hazard_dissipation_J_per_m,
        trial.energy_margin_J_per_m, "v13.branch-mark-private-topology-reservation-not-baseline-clock")
