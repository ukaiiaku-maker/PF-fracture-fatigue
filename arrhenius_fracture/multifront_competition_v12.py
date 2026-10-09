"""Two-phase V12 directional competition over accepted intervals.

Preview is pure.  Finalization is the sole owner of elapsed hazard action and
selected-event consumption.  It delegates every hazard calculation to the
qualified V11 directional-competition implementation.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Mapping

from .directional_competition_v11 import (
    DirectionalCompetitionState, commit_directional_interval,
    competition_state_from_dict, competition_state_to_dict,
    construct_action_proposals, consume_completed_events,
    preview_directional_interval,
)
from .general_multifront_v12 import (
    FrontRuntimeState, MultiFrontRuntimeState, TopologyProposal, canonical_hash,
    form_correlated_topology_proposals,
)


SCHEMA = "v12.two-phase-directional-competition/1"


def _rng_contract(state: DirectionalCompetitionState) -> dict[str, Any]:
    return {
        "global_hazard_seed": state.global_hazard_seed,
        "competition_event_index": state.competition_event_index,
        "thresholds": {
            item.candidate_id: {
                "threshold_seed": item.threshold_seed,
                "threshold_process": item.threshold_process,
                "current_threshold_action": item.current_threshold_action,
                "completed_event_count": item.completed_event_count,
            }
            for item in state.hazard_states
        },
    }


@dataclass(frozen=True)
class FrontCompetitionPreview:
    front_id: str
    accepted_competition_sha256: str
    previewed_competition: DirectionalCompetitionState
    action_increments: Mapping[str, float]
    completed_event_ids: tuple[str, ...]
    completion_times_s: Mapping[str, float]
    proposals: tuple[TopologyProposal, ...]
    rng_sha256_before: str
    rng_sha256_after: str


@dataclass(frozen=True)
class CompetitionIntervalPreview:
    accepted_registry_sha256: str
    duration_s: float
    fronts: Mapping[str, FrontCompetitionPreview]
    proposals: tuple[TopologyProposal, ...]


@dataclass(frozen=True)
class CompetitionFinalization:
    runtime: MultiFrontRuntimeState
    selected_event_ids: tuple[str, ...]
    unselected_pending_event_ids: tuple[str, ...]
    disposition_by_event_id: Mapping[str, str]
    competition_hashes_before: Mapping[str, str]
    competition_hashes_after: Mapping[str, str]
    rng_hashes_before: Mapping[str, str]
    rng_hashes_after: Mapping[str, str]


def preview_competitions(
    runtime: MultiFrontRuntimeState,
    rates_per_s: Mapping[str, Mapping[str, float]],
    endpoints_m: Mapping[str, Mapping[str, tuple[float, float]]],
    *, start_time_s: float, duration_s: float, correlation_interval_s: float,
) -> CompetitionIntervalPreview:
    """Preview all clocks without mutating any accepted V12 object."""
    before = runtime.registry_fingerprint
    fronts: dict[str, FrontCompetitionPreview] = {}
    all_proposals = []
    for front_id, front in runtime.front_runtimes.items():
        competition = competition_state_from_dict(front.competition_state)
        active_ids = set(front.mechanically_active_candidate_ids)
        if set(rates_per_s.get(front_id, {})) != active_ids:
            raise ValueError(
                "preview rates must cover every mechanically active candidate exactly once"
            )
        if set(endpoints_m.get(front_id, {})) != active_ids:
            raise ValueError(
                "preview endpoints must cover every mechanically active candidate exactly once"
            )
        previews = []
        hazards = []
        increments = {}
        for hazard in competition.hazard_states:
            if hazard.candidate_id not in active_ids:
                # A dormant clock is accepted state.  It receives neither an
                # invented zero rate nor elapsed action until mechanics makes
                # that candidate active again.
                hazards.append(hazard)
                continue
            item = preview_directional_interval(
                hazard, lambda_per_s=float(rates_per_s[front_id][hazard.candidate_id]),
                start_time_s=float(start_time_s), duration_s=float(duration_s),
                maximum_completed_events=1,
            )
            if item.completed_events:
                first = item.completed_events[0]
                item = replace(
                    item,
                    end_action=first.action_after,
                    duration_s=max(
                        0.0, first.completion_time_s - item.start_time_s
                    ),
                )
            previews.append(item)
            hazards.append(commit_directional_interval(hazard, item))
            increments[hazard.candidate_id] = item.end_action - item.start_action
        previewed = replace(competition, hazard_states=tuple(hazards))
        actions = construct_action_proposals(
            tuple(
                hazard for hazard in previewed.hazard_states
                if hazard.candidate_id in active_ids
            ),
            correlation_interval_s=correlation_interval_s,
        )
        proposals = form_correlated_topology_proposals(
            front_id=front_id, owner_id=runtime.owner_by_front[front_id],
            action_proposals=actions,
            end_points_by_candidate=endpoints_m.get(front_id, {}),
        )
        interval_end_s = float(start_time_s) + float(duration_s)
        eligible_proposals = []
        for proposal in proposals:
            if proposal.action_type != "one_arm" or correlation_interval_s <= 0.0:
                eligible_proposals.append(proposal)
                continue
            correlation_ready_s = (
                float(proposal.completion_time_s)
                + float(correlation_interval_s)
            )
            if correlation_ready_s <= interval_end_s + 1.0e-12:
                eligible_proposals.append(replace(
                    proposal, completion_time_s=correlation_ready_s,
                ))
        proposals = tuple(eligible_proposals)
        completed = tuple(
            event for item in previews for event in item.completed_events
        )
        rng_before = canonical_hash(_rng_contract(competition))
        # Preview computes deterministic replacement thresholds but consumes no
        # mutable random stream.  Its before/after RNG ownership is identical.
        rng_after = canonical_hash(_rng_contract(competition))
        front_preview = FrontCompetitionPreview(
            front_id=front_id,
            accepted_competition_sha256=canonical_hash(front.competition_state),
            previewed_competition=previewed,
            action_increments=dict(sorted(increments.items())),
            completed_event_ids=tuple(sorted(event.event_id for event in completed)),
            completion_times_s={
                event.event_id: event.completion_time_s for event in completed
            },
            proposals=proposals,
            rng_sha256_before=rng_before, rng_sha256_after=rng_after,
        )
        fronts[front_id] = front_preview
        all_proposals.extend(proposals)
    if runtime.registry_fingerprint != before:
        raise RuntimeError("directional competition preview mutated accepted runtime")
    return CompetitionIntervalPreview(
        accepted_registry_sha256=before, duration_s=float(duration_s),
        fronts=dict(sorted(fronts.items())),
        proposals=tuple(sorted(all_proposals, key=lambda item: item.selection_key)),
    )


def finalize_competitions(
    runtime: MultiFrontRuntimeState, preview: CompetitionIntervalPreview,
    *, selected_proposal: TopologyProposal | None,
    accepted: bool, resource_limit_stop: bool = False,
) -> CompetitionFinalization:
    """Commit all elapsed clocks and consume only an accepted selected event."""
    if preview.accepted_registry_sha256 != runtime.registry_fingerprint:
        raise ValueError("competition preview is stale relative to accepted runtime")
    before_hashes = {
        key: canonical_hash(value.competition_state)
        for key, value in runtime.front_runtimes.items()
    }
    rng_before = {
        key: canonical_hash(_rng_contract(competition_state_from_dict(value.competition_state)))
        for key, value in runtime.front_runtimes.items()
    }
    if resource_limit_stop or (selected_proposal is not None and not accepted):
        return CompetitionFinalization(
            runtime=runtime, selected_event_ids=(),
            unselected_pending_event_ids=(), disposition_by_event_id={},
            competition_hashes_before=before_hashes,
            competition_hashes_after=before_hashes,
            rng_hashes_before=rng_before, rng_hashes_after=rng_before,
        )
    fronts = dict(runtime.front_runtimes)
    selected_ids = tuple(
        () if selected_proposal is None else selected_proposal.member_event_ids
    )
    dispositions: dict[str, str] = {}
    pending = []
    for front_id, item in preview.fronts.items():
        competition = item.previewed_competition
        local_selected = selected_ids if (
            selected_proposal is not None and selected_proposal.front_id == front_id
        ) else ()
        if local_selected:
            competition = consume_completed_events(competition, local_selected)
            competition = replace(
                competition,
                competition_event_index=competition.competition_event_index + 1,
            )
        for event in item.completed_event_ids:
            disposition = "selected_consumed" if event in local_selected else "unselected_pending"
            dispositions[event] = disposition
            if disposition == "unselected_pending":
                pending.append(event)
        fronts[front_id] = FrontRuntimeState(
            front_id=front_id,
            competition_state=competition_state_to_dict(competition),
            candidate_ids=fronts[front_id].candidate_ids,
            lineage_rng_state={
                **dict(fronts[front_id].lineage_rng_state),
                "competition_event_index": competition.competition_event_index,
                "threshold_rng_contract_sha256": canonical_hash(_rng_contract(competition)),
            },
            interval_count=fronts[front_id].interval_count + 1,
            mechanically_active_candidate_ids=fronts[
                front_id
            ].mechanically_active_candidate_ids,
        )
    finalized = replace(runtime, front_runtimes=fronts)
    after_hashes = {
        key: canonical_hash(value.competition_state)
        for key, value in finalized.front_runtimes.items()
    }
    rng_after = {
        key: canonical_hash(_rng_contract(competition_state_from_dict(value.competition_state)))
        for key, value in finalized.front_runtimes.items()
    }
    return CompetitionFinalization(
        runtime=finalized, selected_event_ids=tuple(sorted(selected_ids)),
        unselected_pending_event_ids=tuple(sorted(pending)),
        disposition_by_event_id=dict(sorted(dispositions.items())),
        competition_hashes_before=before_hashes,
        competition_hashes_after=after_hashes,
        rng_hashes_before=rng_before, rng_hashes_after=rng_after,
    )


def daughter_competitions_from_v11_lineage(
    parent: FrontRuntimeState, daughter_front_ids: tuple[str, str],
    *, assigned_candidate_by_daughter: Mapping[str, str] | None = None,
) -> dict[str, FrontRuntimeState]:
    """Apply V11's post-action rule: both active tips receive exact state value.

    V11 stores ``front_competitions={tip: state.competition for tip in active}``.
    V12 clones that accepted value so daughters never alias mutable objects.
    """
    if len(daughter_front_ids) != 2 or len(set(daughter_front_ids)) != 2:
        raise ValueError("binary lineage requires exactly two distinct daughters")
    competition_state_from_dict(parent.competition_state)  # fail closed on partial state
    assigned = dict(assigned_candidate_by_daughter or {})
    if assigned and set(assigned) != set(daughter_front_ids):
        raise ValueError("assigned daughter-candidate map must cover both daughters")
    if not set(assigned.values()).issubset(parent.candidate_ids):
        raise ValueError("assigned daughter candidate is outside the competition")
    parent_rng_identity = canonical_hash({
        "front_id": parent.front_id,
        "lineage_rng_state": dict(parent.lineage_rng_state),
        "competition": parent.competition_state,
    })
    result = {
        child: FrontRuntimeState(
            front_id=child,
            competition_state=parent.competition_state,
            candidate_ids=parent.candidate_ids,
            lineage_rng_state={
                **dict(parent.lineage_rng_state),
                "v11_lineage_parent_front_id": parent.front_id,
                "v11_lineage_rule": "accepted_competition_value_for_every_active_tip",
                "daughter_front_id": child,
                "parent_rng_identity": parent_rng_identity,
                # V11 gives both active tips the same accepted competition
                # value.  V12 makes future lineage streams independently
                # addressable without drawing from either stream here.
                "daughter_rng_stream_identity": canonical_hash({
                    "parent_rng_identity": parent_rng_identity,
                    "daughter_front_id": child,
                    "rule": "v11-value-clone-v12-independent-address",
                }),
            },
            interval_count=parent.interval_count,
            mechanically_active_candidate_ids=(
                parent.candidate_ids if not assigned else (assigned[child],)
            ),
        )
        for child in sorted(daughter_front_ids)
    }
    identities = {
        value.lineage_rng_state["daughter_rng_stream_identity"]
        for value in result.values()
    }
    if len(identities) != 2:
        raise RuntimeError("daughter RNG stream identities are not independent")
    return result


__all__ = [
    "SCHEMA", "CompetitionFinalization", "CompetitionIntervalPreview",
    "FrontCompetitionPreview", "daughter_competitions_from_v11_lineage",
    "finalize_competitions", "preview_competitions",
]
