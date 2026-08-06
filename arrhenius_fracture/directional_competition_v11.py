"""Pure analytical directional-cleavage competition for v11.

Nothing in this module mutates crack geometry or the production driver.  It
previews rates through an existing cleavage engine and evolves only explicitly
provided analytical state.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass, replace
import hashlib
import json
import math
import pickle
from typing import Any, Iterable, Mapping, Sequence


SCHEMA = "v11.directional-competition/1"
IDENTITY_DIGITS = 14
VECTOR_TOLERANCE = 5.0e-13
TIME_TOLERANCE_S = 1.0e-12

Point = tuple[float, float]


def _finite(value: float, name: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _canonical_zero(value: float) -> float:
    value = round(float(value), IDENTITY_DIGITS)
    return 0.0 if value == 0.0 else value


def _unit_vector(value: Iterable[float], name: str) -> Point:
    vector = tuple(_finite(component, name) for component in value)
    if len(vector) != 2:
        raise ValueError(f"{name} must contain two components")
    norm = math.hypot(*vector)
    if norm <= 0.0:
        raise ValueError(f"{name} must be nonzero")
    result = (_canonical_zero(vector[0] / norm), _canonical_zero(vector[1] / norm))
    renorm = math.hypot(*result)
    return (_canonical_zero(result[0] / renorm), _canonical_zero(result[1] / renorm))


def canonical_candidate_id(
    *,
    plane_family: str,
    plane_variant: str,
    direction_xy: Iterable[float],
    orientation_convention: str,
) -> str:
    direction = _unit_vector(direction_xy, "direction_xy")
    identity = {
        "direction_xy": [format(value, f".{IDENTITY_DIGITS}g") for value in direction],
        "orientation_convention": str(orientation_convention).strip(),
        "plane_family": str(plane_family).strip(),
        "plane_variant": str(plane_variant).strip(),
    }
    if not all((identity["plane_family"], identity["plane_variant"], identity["orientation_convention"])):
        raise ValueError("candidate identity fields must not be empty")
    digest = hashlib.sha256(
        json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()[:20]
    return f"cleave:{identity['plane_family']}:{identity['plane_variant']}:{digest}"


@dataclass(frozen=True)
class CleavageCandidate:
    candidate_id: str
    plane_family: str
    plane_variant: str
    direction_xy: Point
    normal_xy: Point
    angle_rad: float
    gamma_rel: float
    orientation_convention: str

    @classmethod
    def create(
        cls,
        *,
        plane_family: str,
        plane_variant: str,
        direction_xy: Iterable[float],
        normal_xy: Iterable[float],
        gamma_rel: float,
        orientation_convention: str,
    ) -> "CleavageCandidate":
        direction = _unit_vector(direction_xy, "direction_xy")
        normal = _unit_vector(normal_xy, "normal_xy")
        angle = math.atan2(direction[1], direction[0])
        return cls(
            candidate_id=canonical_candidate_id(
                plane_family=plane_family,
                plane_variant=plane_variant,
                direction_xy=direction,
                orientation_convention=orientation_convention,
            ),
            plane_family=str(plane_family).strip(),
            plane_variant=str(plane_variant).strip(),
            direction_xy=direction,
            normal_xy=normal,
            angle_rad=angle,
            gamma_rel=_finite(gamma_rel, "gamma_rel"),
            orientation_convention=str(orientation_convention).strip(),
        )

    def __post_init__(self) -> None:
        direction = _unit_vector(self.direction_xy, "direction_xy")
        normal = _unit_vector(self.normal_xy, "normal_xy")
        if direction != self.direction_xy or normal != self.normal_xy:
            raise ValueError("candidate vectors must be canonically normalized")
        if abs(direction[0] * normal[0] + direction[1] * normal[1]) > VECTOR_TOLERANCE:
            raise ValueError("candidate direction and normal must be orthogonal")
        if not self.plane_family or not self.plane_variant or not self.orientation_convention:
            raise ValueError("candidate physical identity fields must not be empty")
        if not math.isfinite(self.angle_rad):
            raise ValueError("candidate angle must be finite")
        if not math.isclose(
            self.angle_rad, math.atan2(direction[1], direction[0]),
            rel_tol=0.0, abs_tol=VECTOR_TOLERANCE,
        ):
            raise ValueError("candidate angle is inconsistent with direction")
        if not math.isfinite(self.gamma_rel) or self.gamma_rel <= 0.0:
            raise ValueError("candidate gamma_rel must be positive and finite")
        expected = canonical_candidate_id(
            plane_family=self.plane_family,
            plane_variant=self.plane_variant,
            direction_xy=direction,
            orientation_convention=self.orientation_convention,
        )
        if self.candidate_id != expected:
            raise ValueError("candidate_id is inconsistent with physical identity")

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "plane_family": self.plane_family,
            "plane_variant": self.plane_variant,
            "direction_xy": list(self.direction_xy),
            "normal_xy": list(self.normal_xy),
            "angle_rad": self.angle_rad,
            "gamma_rel": self.gamma_rel,
            "orientation_convention": self.orientation_convention,
        }


def canonical_candidate_inventory(
    candidates: Iterable[CleavageCandidate],
) -> tuple[CleavageCandidate, ...]:
    result = tuple(sorted(candidates, key=lambda candidate: candidate.candidate_id))
    ids = tuple(candidate.candidate_id for candidate in result)
    if len(ids) != len(set(ids)):
        raise ValueError("candidate IDs must be unique")
    return result


def tungsten_cleavage_candidates(
    *,
    theta_deg: float,
    forward_xy: Iterable[float] = (1.0, 0.0),
    require_positive_forward: bool = True,
    include_110: bool = False,
    gamma_110_rel: float = 1.3,
) -> tuple[CleavageCandidate, ...]:
    """Canonical adapter around the existing tungsten cleavage traces."""
    from .crystal import bcc_cleavage_traces

    theta = _finite(theta_deg, "theta_deg")
    forward = _unit_vector(forward_xy, "forward_xy")
    convention = f"bcc-[001]-section:theta_deg={format(_canonical_zero(theta), '.14g')}"
    result = []
    for trace in bcc_cleavage_traces(theta, include_110, gamma_110_rel):
        direction = _unit_vector(trace["t"], "trace direction")
        normal = _unit_vector(trace["n"], "trace normal")
        if direction[0] * forward[0] + direction[1] * forward[1] < 0.0:
            direction = (_canonical_zero(-direction[0]), _canonical_zero(-direction[1]))
            normal = (_canonical_zero(-normal[0]), _canonical_zero(-normal[1]))
        projection = direction[0] * forward[0] + direction[1] * forward[1]
        if require_positive_forward and projection <= 0.0:
            continue
        result.append(
            CleavageCandidate.create(
                plane_family=str(trace["family"]),
                plane_variant=str(trace["name"]),
                direction_xy=direction,
                normal_xy=normal,
                gamma_rel=float(trace["gamma_rel"]),
                orientation_convention=convention,
            )
        )
    return canonical_candidate_inventory(result)


@dataclass(frozen=True)
class DirectionalRate:
    candidate_id: str
    lambda_per_s: float
    signed_J_J_per_m2: float
    positive_J_J_per_m2: float
    K_directional_Pa_sqrt_m: float
    gamma_rel: float

    def __post_init__(self) -> None:
        for name in (
            "lambda_per_s", "signed_J_J_per_m2", "positive_J_J_per_m2",
            "K_directional_Pa_sqrt_m", "gamma_rel",
        ):
            _finite(getattr(self, name), name)
        if min(self.lambda_per_s, self.positive_J_J_per_m2, self.K_directional_Pa_sqrt_m) < 0.0:
            raise ValueError("directional rate and positive drive fields must be nonnegative")
        if self.gamma_rel <= 0.0:
            raise ValueError("gamma_rel must be positive")


def directional_drive(
    candidate: CleavageCandidate,
    *,
    signed_J_J_per_m2: float,
    Eprime_Pa: float,
) -> DirectionalRate:
    signed = _finite(signed_J_J_per_m2, "signed_J_J_per_m2")
    Eprime = _finite(Eprime_Pa, "Eprime_Pa")
    if Eprime <= 0.0:
        raise ValueError("Eprime_Pa must be positive")
    positive = max(signed, 0.0)
    K = math.sqrt(Eprime * positive)
    return DirectionalRate(
        candidate_id=candidate.candidate_id,
        lambda_per_s=0.0,
        signed_J_J_per_m2=signed,
        positive_J_J_per_m2=positive,
        K_directional_Pa_sqrt_m=K,
        gamma_rel=candidate.gamma_rel,
    )


def preview_production_cleavage_rate(
    engine: Any,
    candidate: CleavageCandidate,
    *,
    signed_J_J_per_m2: float,
    Eprime_Pa: float,
    temperature_K: float,
) -> DirectionalRate:
    """Preview the existing engine hazard without changing engine state."""
    drive = directional_drive(
        candidate, signed_J_J_per_m2=signed_J_J_per_m2, Eprime_Pa=Eprime_Pa
    )
    before = pickle.dumps(engine.__dict__, protocol=5)
    if drive.K_directional_Pa_sqrt_m <= 0.0:
        rate = 0.0
    else:
        effective_K = drive.K_directional_Pa_sqrt_m / math.sqrt(candidate.gamma_rel)
        stress = engine.sigma_tip(effective_K)
        rate, _, _ = engine.lambda_cleave(stress, _finite(temperature_K, "temperature_K"))
        rate = max(_finite(rate, "lambda_per_s"), 0.0)
    after = pickle.dumps(engine.__dict__, protocol=5)
    if before != after:
        raise RuntimeError("production cleavage rate preview mutated engine state")
    return replace(drive, lambda_per_s=rate)


@dataclass(frozen=True)
class CompletedDirectionalEvent:
    candidate_id: str
    event_ordinal: int
    completion_time_s: float
    action_before: float
    action_increment: float
    action_after: float

    @property
    def event_id(self) -> str:
        return f"{self.candidate_id}#event:{self.event_ordinal:016d}"

    def __post_init__(self) -> None:
        if self.event_ordinal <= 0:
            raise ValueError("event ordinal must be positive")
        for name in ("completion_time_s", "action_before", "action_increment", "action_after"):
            _finite(getattr(self, name), name)
        if self.action_increment < 0.0 or self.action_after < self.action_before:
            raise ValueError("event action fields are inconsistent")


@dataclass(frozen=True)
class DirectionalHazardState:
    candidate_id: str
    action: float = 0.0
    previous_rate_per_s: float | None = None
    completed_event_count: int = 0
    residual_action: float = 0.0
    last_completion_time_s: float | None = None
    pending_events: tuple[CompletedDirectionalEvent, ...] = ()

    @classmethod
    def from_action(
        cls, candidate_id: str, action: float, *, previous_rate_per_s: float | None = None
    ) -> "DirectionalHazardState":
        value = _finite(action, "action")
        if value < 0.0:
            raise ValueError("action must be nonnegative")
        completed = int(math.floor(value + 1.0e-14))
        return cls(
            candidate_id=candidate_id,
            action=value,
            previous_rate_per_s=previous_rate_per_s,
            completed_event_count=completed,
            residual_action=value - completed,
        )

    def __post_init__(self) -> None:
        action = _finite(self.action, "action")
        residual = _finite(self.residual_action, "residual_action")
        if action < 0.0 or self.completed_event_count < 0:
            raise ValueError("directional action/count must be nonnegative")
        if not (0.0 <= residual < 1.0 + 1.0e-12):
            raise ValueError("residual action must lie in [0, 1)")
        expected = action - math.floor(action + 1.0e-14)
        if expected < 0.0 and abs(expected) < 1.0e-12:
            expected = 0.0
        if not math.isclose(residual, expected, rel_tol=0.0, abs_tol=1.0e-12):
            raise ValueError("residual action is inconsistent with accumulated action")
        if self.previous_rate_per_s is not None and _finite(self.previous_rate_per_s, "previous_rate") < 0.0:
            raise ValueError("previous rate must be nonnegative")
        if self.last_completion_time_s is not None:
            _finite(self.last_completion_time_s, "last_completion_time_s")
        if any(event.candidate_id != self.candidate_id for event in self.pending_events):
            raise ValueError("pending event belongs to another candidate")
        event_ids = tuple(event.event_id for event in self.pending_events)
        if len(event_ids) != len(set(event_ids)):
            raise ValueError("duplicate pending directional event")


@dataclass(frozen=True)
class DirectionalIntervalPreview:
    candidate_id: str
    start_action: float
    end_action: float
    rate_per_s: float
    start_time_s: float
    duration_s: float
    completed_events: tuple[CompletedDirectionalEvent, ...]


@dataclass(frozen=True)
class CompetingActionProposal:
    action_id: str
    member_candidate_ids: tuple[str, ...]
    member_event_ids: tuple[str, ...]
    member_event_ordinals: tuple[int, ...]
    completion_times_s: tuple[float, ...]
    action_type: str

    def __post_init__(self) -> None:
        size = len(self.member_candidate_ids)
        if size not in (1, 2):
            raise ValueError("directional action must have one or two members")
        if not (
            len(self.member_event_ids) == size
            and len(self.member_event_ordinals) == size
            and len(self.completion_times_s) == size
        ):
            raise ValueError("proposal member fields have inconsistent lengths")
        if tuple(sorted(self.member_candidate_ids)) != self.member_candidate_ids:
            raise ValueError("proposal candidates must be canonically ordered")
        if len(set(self.member_candidate_ids)) != size or len(set(self.member_event_ids)) != size:
            raise ValueError("proposal members must be distinct")
        if self.action_type != ("one_arm" if size == 1 else "two_arm"):
            raise ValueError("proposal action_type is inconsistent with members")
        if any(ordinal <= 0 for ordinal in self.member_event_ordinals):
            raise ValueError("proposal event ordinals must be positive")
        if any(not math.isfinite(time) for time in self.completion_times_s):
            raise ValueError("proposal completion times must be finite")
        if self.action_id != _action_id(self.member_event_ids):
            raise ValueError("action_id is inconsistent with member events")


def _action_id(event_ids: Sequence[str]) -> str:
    members = tuple(sorted(str(value) for value in event_ids))
    digest = hashlib.sha256(
        json.dumps(members, separators=(",", ":")).encode()
    ).hexdigest()[:24]
    return f"action:{len(members)}:{digest}"


def events_are_correlated(
    first: CompletedDirectionalEvent,
    second: CompletedDirectionalEvent,
    *,
    correlation_interval_s: float,
) -> bool:
    interval = _finite(correlation_interval_s, "correlation_interval_s")
    if interval < 0.0:
        raise ValueError("correlation interval must be nonnegative")
    if first.candidate_id == second.candidate_id:
        return False
    return abs(first.completion_time_s - second.completion_time_s) <= interval


def construct_action_proposals(
    hazard_states: Iterable[DirectionalHazardState],
    *,
    correlation_interval_s: float,
) -> tuple[CompetingActionProposal, ...]:
    states = tuple(sorted(hazard_states, key=lambda state: state.candidate_id))
    if len({state.candidate_id for state in states}) != len(states):
        raise ValueError("duplicate directional hazard state")
    events = tuple(
        sorted(
            (event for state in states for event in state.pending_events),
            key=lambda event: (event.candidate_id, event.event_ordinal),
        )
    )
    proposals = []

    def create(members: Sequence[CompletedDirectionalEvent]):
        ordered = tuple(sorted(members, key=lambda event: event.candidate_id))
        event_ids = tuple(event.event_id for event in ordered)
        proposals.append(
            CompetingActionProposal(
                action_id=_action_id(event_ids),
                member_candidate_ids=tuple(event.candidate_id for event in ordered),
                member_event_ids=event_ids,
                member_event_ordinals=tuple(event.event_ordinal for event in ordered),
                completion_times_s=tuple(event.completion_time_s for event in ordered),
                action_type="one_arm" if len(ordered) == 1 else "two_arm",
            )
        )

    for event in events:
        create((event,))
    for index, first in enumerate(events):
        for second in events[index + 1:]:
            if events_are_correlated(
                first, second, correlation_interval_s=correlation_interval_s
            ):
                create((first, second))
    return tuple(sorted(proposals, key=lambda proposal: proposal.action_id))


def deterministic_tie_key(
    proposal: CompetingActionProposal,
    *,
    global_hazard_seed: int,
    competition_event_index: int,
) -> str:
    payload = {
        "competition_event_index": int(competition_event_index),
        "global_hazard_seed": int(global_hazard_seed),
        "member_candidate_ids": list(proposal.member_candidate_ids),
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def select_temporal_or_degenerate_proposal(
    proposals: Iterable[CompetingActionProposal],
    *,
    global_hazard_seed: int,
    competition_event_index: int,
    event_time_tolerance_s: float = TIME_TOLERANCE_S,
) -> CompetingActionProposal:
    values = tuple(proposals)
    if not values:
        raise ValueError("cannot select from an empty proposal set")
    tolerance = _finite(event_time_tolerance_s, "event_time_tolerance_s")
    if tolerance < 0.0:
        raise ValueError("event-time tolerance must be nonnegative")
    earliest = min(min(value.completion_times_s) for value in values)
    degenerate = tuple(
        value
        for value in values
        if abs(min(value.completion_times_s) - earliest) <= tolerance
    )
    return min(
        degenerate,
        key=lambda proposal: (
            deterministic_tie_key(
                proposal,
                global_hazard_seed=global_hazard_seed,
                competition_event_index=competition_event_index,
            ),
            proposal.action_id,
        ),
    )


@dataclass(frozen=True)
class ActionEnergyReservation:
    action_id: str
    member_event_ids: tuple[str, ...]
    reserved_event_rewards_m: tuple[float, ...]
    status: str = "active"

    def __post_init__(self) -> None:
        if self.status not in {"active", "released", "accepted"}:
            raise ValueError("invalid reservation status")
        if not self.member_event_ids or len(self.member_event_ids) != len(self.reserved_event_rewards_m):
            raise ValueError("reservation event/reward fields are inconsistent")
        if len(set(self.member_event_ids)) != len(self.member_event_ids):
            raise ValueError("duplicate event in reservation")
        if any(not math.isfinite(value) or value < 0.0 for value in self.reserved_event_rewards_m):
            raise ValueError("reserved event rewards must be finite and nonnegative")


@dataclass(frozen=True)
class DirectionalCompetitionState:
    candidates: tuple[CleavageCandidate, ...]
    hazard_states: tuple[DirectionalHazardState, ...]
    global_hazard_seed: int
    competition_event_index: int = 0
    reservations: tuple[ActionEnergyReservation, ...] = ()
    consumed_event_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        candidates = canonical_candidate_inventory(self.candidates)
        hazards = tuple(sorted(self.hazard_states, key=lambda state: state.candidate_id))
        reservations = tuple(sorted(self.reservations, key=lambda item: item.action_id))
        consumed = tuple(sorted(str(value) for value in self.consumed_event_ids))
        object.__setattr__(self, "candidates", candidates)
        object.__setattr__(self, "hazard_states", hazards)
        object.__setattr__(self, "reservations", reservations)
        object.__setattr__(self, "consumed_event_ids", consumed)
        object.__setattr__(self, "global_hazard_seed", int(self.global_hazard_seed))
        object.__setattr__(self, "competition_event_index", int(self.competition_event_index))
        self.validate()

    @classmethod
    def initialize(
        cls, candidates: Iterable[CleavageCandidate], *, global_hazard_seed: int
    ) -> "DirectionalCompetitionState":
        inventory = canonical_candidate_inventory(candidates)
        return cls(
            candidates=inventory,
            hazard_states=tuple(
                DirectionalHazardState(candidate.candidate_id) for candidate in inventory
            ),
            global_hazard_seed=global_hazard_seed,
        )

    @property
    def pending_events(self) -> tuple[CompletedDirectionalEvent, ...]:
        return tuple(
            sorted(
                (event for state in self.hazard_states for event in state.pending_events),
                key=lambda event: event.event_id,
            )
        )

    def validate(self) -> None:
        candidate_ids = {candidate.candidate_id for candidate in self.candidates}
        hazard_ids = [state.candidate_id for state in self.hazard_states]
        if len(hazard_ids) != len(set(hazard_ids)) or set(hazard_ids) != candidate_ids:
            raise ValueError("directional hazard states must map one-to-one to candidates")
        if self.competition_event_index < 0:
            raise ValueError("competition_event_index must be nonnegative")
        pending = {event.event_id for event in self.pending_events}
        if len(pending) != len(self.pending_events):
            raise ValueError("duplicate pending event across candidates")
        if len(self.consumed_event_ids) != len(set(self.consumed_event_ids)):
            raise ValueError("duplicate consumed event")
        if pending.intersection(self.consumed_event_ids):
            raise ValueError("event cannot be both pending and consumed")
        action_ids = [reservation.action_id for reservation in self.reservations]
        if len(action_ids) != len(set(action_ids)):
            raise ValueError("duplicate reservation action_id")
        active_owned = []
        for reservation in self.reservations:
            if reservation.status == "active":
                if not set(reservation.member_event_ids).issubset(pending):
                    raise ValueError("active reservation references a nonpending event")
                active_owned.extend(reservation.member_event_ids)
            elif reservation.status == "accepted":
                if not set(reservation.member_event_ids).issubset(self.consumed_event_ids):
                    raise ValueError("accepted reservation events must be consumed")
        if len(active_owned) != len(set(active_owned)):
            raise ValueError("pending event is owned by multiple active reservations")


def reserve_action(
    state: DirectionalCompetitionState,
    proposal: CompetingActionProposal,
    *,
    event_rewards_m: Sequence[float],
) -> DirectionalCompetitionState:
    if len(event_rewards_m) != len(proposal.member_event_ids):
        raise ValueError("one event reward is required per proposal member")
    pending = {event.event_id for event in state.pending_events}
    if not set(proposal.member_event_ids).issubset(pending):
        raise ValueError("proposal references a nonpending event")
    active_owned = {
        event_id
        for reservation in state.reservations if reservation.status == "active"
        for event_id in reservation.member_event_ids
    }
    if active_owned.intersection(proposal.member_event_ids):
        raise ValueError("pending event is already reserved")
    if any(item.action_id == proposal.action_id for item in state.reservations):
        raise ValueError("action already has a reservation record")
    reservation = ActionEnergyReservation(
        action_id=proposal.action_id,
        member_event_ids=proposal.member_event_ids,
        reserved_event_rewards_m=tuple(float(value) for value in event_rewards_m),
    )
    return replace(state, reservations=state.reservations + (reservation,))


def release_reservation(
    state: DirectionalCompetitionState, action_id: str
) -> DirectionalCompetitionState:
    found = False
    reservations = []
    for reservation in state.reservations:
        if reservation.action_id == action_id:
            if reservation.status != "active":
                raise ValueError("only an active reservation can be released")
            reservation = replace(reservation, status="released")
            found = True
        reservations.append(reservation)
    if not found:
        raise KeyError(action_id)
    return replace(state, reservations=tuple(reservations))


def consume_completed_events(
    state: DirectionalCompetitionState, event_ids: Iterable[str]
) -> DirectionalCompetitionState:
    selected = tuple(sorted(str(value) for value in event_ids))
    if len(selected) != len(set(selected)):
        raise ValueError("event consumption list contains duplicates")
    pending = {event.event_id for event in state.pending_events}
    if not set(selected).issubset(pending):
        raise ValueError("cannot consume an event that is not pending")
    hazards = tuple(
        replace(
            hazard,
            pending_events=tuple(
                event for event in hazard.pending_events if event.event_id not in selected
            ),
        )
        for hazard in state.hazard_states
    )
    return replace(
        state,
        hazard_states=hazards,
        consumed_event_ids=state.consumed_event_ids + selected,
    )


def accept_reservation(
    state: DirectionalCompetitionState, action_id: str
) -> DirectionalCompetitionState:
    reservation = next(
        (item for item in state.reservations if item.action_id == action_id), None
    )
    if reservation is None:
        raise KeyError(action_id)
    if reservation.status != "active":
        raise ValueError("only an active reservation can be accepted")
    selected = set(reservation.member_event_ids)
    hazards = tuple(
        replace(
            hazard,
            pending_events=tuple(
                event for event in hazard.pending_events if event.event_id not in selected
            ),
        )
        for hazard in state.hazard_states
    )
    reservations = tuple(
        replace(item, status="accepted") if item.action_id == action_id else item
        for item in state.reservations
    )
    return DirectionalCompetitionState(
        candidates=state.candidates,
        hazard_states=hazards,
        global_hazard_seed=state.global_hazard_seed,
        reservations=reservations,
        consumed_event_ids=state.consumed_event_ids + tuple(sorted(selected)),
        competition_event_index=state.competition_event_index + 1,
    )


def _event_to_dict(event: CompletedDirectionalEvent) -> dict[str, Any]:
    return {
        "event_id": event.event_id,
        "candidate_id": event.candidate_id,
        "event_ordinal": event.event_ordinal,
        "completion_time_s": event.completion_time_s,
        "action_before": event.action_before,
        "action_increment": event.action_increment,
        "action_after": event.action_after,
    }


def competition_state_to_dict(state: DirectionalCompetitionState) -> dict[str, Any]:
    return {
        "schema": SCHEMA,
        "global_hazard_seed": state.global_hazard_seed,
        "competition_event_index": state.competition_event_index,
        "candidates": [candidate.to_dict() for candidate in state.candidates],
        "hazard_states": [
            {
                "candidate_id": hazard.candidate_id,
                "action": hazard.action,
                "previous_rate_per_s": hazard.previous_rate_per_s,
                "completed_event_count": hazard.completed_event_count,
                "residual_action": hazard.residual_action,
                "last_completion_time_s": hazard.last_completion_time_s,
                "pending_events": [_event_to_dict(event) for event in hazard.pending_events],
            }
            for hazard in state.hazard_states
        ],
        "reservations": [
            {
                "action_id": item.action_id,
                "member_event_ids": list(item.member_event_ids),
                "reserved_event_rewards_m": list(item.reserved_event_rewards_m),
                "status": item.status,
            }
            for item in state.reservations
        ],
        "consumed_event_ids": list(state.consumed_event_ids),
    }


def competition_state_to_json(state: DirectionalCompetitionState) -> str:
    return json.dumps(
        competition_state_to_dict(state), indent=2, sort_keys=True, allow_nan=False
    ) + "\n"


def competition_state_from_dict(payload: Mapping[str, Any]) -> DirectionalCompetitionState:
    if payload.get("schema") != SCHEMA:
        raise ValueError("unsupported directional-competition schema")
    candidates = tuple(
        CleavageCandidate(
            candidate_id=item["candidate_id"],
            plane_family=item["plane_family"],
            plane_variant=item["plane_variant"],
            direction_xy=tuple(item["direction_xy"]),
            normal_xy=tuple(item["normal_xy"]),
            angle_rad=item["angle_rad"],
            gamma_rel=item["gamma_rel"],
            orientation_convention=item["orientation_convention"],
        )
        for item in payload.get("candidates", [])
    )

    def load_event(item: Mapping[str, Any]) -> CompletedDirectionalEvent:
        event = CompletedDirectionalEvent(
            candidate_id=item["candidate_id"],
            event_ordinal=item["event_ordinal"],
            completion_time_s=item["completion_time_s"],
            action_before=item["action_before"],
            action_increment=item["action_increment"],
            action_after=item["action_after"],
        )
        if item.get("event_id") != event.event_id:
            raise ValueError("serialized event_id is inconsistent")
        return event

    hazards = tuple(
        DirectionalHazardState(
            candidate_id=item["candidate_id"],
            action=item["action"],
            previous_rate_per_s=item.get("previous_rate_per_s"),
            completed_event_count=item["completed_event_count"],
            residual_action=item["residual_action"],
            last_completion_time_s=item.get("last_completion_time_s"),
            pending_events=tuple(load_event(event) for event in item.get("pending_events", [])),
        )
        for item in payload.get("hazard_states", [])
    )
    reservations = tuple(
        ActionEnergyReservation(
            action_id=item["action_id"],
            member_event_ids=tuple(item["member_event_ids"]),
            reserved_event_rewards_m=tuple(item["reserved_event_rewards_m"]),
            status=item["status"],
        )
        for item in payload.get("reservations", [])
    )
    return DirectionalCompetitionState(
        candidates=candidates,
        hazard_states=hazards,
        global_hazard_seed=payload["global_hazard_seed"],
        competition_event_index=payload["competition_event_index"],
        reservations=reservations,
        consumed_event_ids=tuple(payload.get("consumed_event_ids", [])),
    )


def competition_state_from_json(text: str) -> DirectionalCompetitionState:
    return competition_state_from_dict(json.loads(text))


def preview_directional_interval(
    state: DirectionalHazardState,
    *,
    lambda_per_s: float,
    start_time_s: float,
    duration_s: float,
) -> DirectionalIntervalPreview:
    rate = _finite(lambda_per_s, "lambda_per_s")
    start_time = _finite(start_time_s, "start_time_s")
    duration = _finite(duration_s, "duration_s")
    if rate < 0.0 or duration < 0.0:
        raise ValueError("rate and duration must be nonnegative")
    increment = rate * duration
    end = state.action + increment
    events = []
    ordinal = state.completed_event_count + 1
    boundary = float(ordinal)
    limit = math.floor(end + 1.0e-13)
    while ordinal <= limit:
        crossing = start_time + (boundary - state.action) / rate if rate > 0.0 else math.inf
        if crossing >= start_time - TIME_TOLERANCE_S:
            events.append(
                CompletedDirectionalEvent(
                    candidate_id=state.candidate_id,
                    event_ordinal=ordinal,
                    completion_time_s=crossing,
                    action_before=state.action,
                    action_increment=boundary - state.action,
                    action_after=boundary,
                )
            )
        ordinal += 1
        boundary += 1.0
    return DirectionalIntervalPreview(
        candidate_id=state.candidate_id,
        start_action=state.action,
        end_action=end,
        rate_per_s=rate,
        start_time_s=start_time,
        duration_s=duration,
        completed_events=tuple(events),
    )


def commit_directional_interval(
    state: DirectionalHazardState,
    preview: DirectionalIntervalPreview,
) -> DirectionalHazardState:
    if preview.candidate_id != state.candidate_id or preview.start_action != state.action:
        raise ValueError("directional preview does not begin from accepted state")
    events = state.pending_events + preview.completed_events
    completed = state.completed_event_count + len(preview.completed_events)
    last_time = (
        preview.completed_events[-1].completion_time_s
        if preview.completed_events else state.last_completion_time_s
    )
    residual = preview.end_action - math.floor(preview.end_action + 1.0e-14)
    if residual < 0.0 and abs(residual) < 1.0e-12:
        residual = 0.0
    return DirectionalHazardState(
        candidate_id=state.candidate_id,
        action=preview.end_action,
        previous_rate_per_s=preview.rate_per_s,
        completed_event_count=completed,
        residual_action=residual,
        last_completion_time_s=last_time,
        pending_events=events,
    )


__all__ = [
    "ActionEnergyReservation", "CleavageCandidate", "CompletedDirectionalEvent",
    "CompetingActionProposal", "DirectionalCompetitionState",
    "DirectionalHazardState", "DirectionalIntervalPreview", "DirectionalRate", "SCHEMA",
    "accept_reservation",
    "canonical_candidate_id", "canonical_candidate_inventory",
    "competition_state_from_dict", "competition_state_from_json",
    "competition_state_to_dict", "competition_state_to_json",
    "commit_directional_interval", "directional_drive",
    "construct_action_proposals", "consume_completed_events",
    "deterministic_tie_key", "events_are_correlated",
    "preview_directional_interval", "preview_production_cleavage_rate",
    "release_reservation", "reserve_action",
    "select_temporal_or_degenerate_proposal",
    "tungsten_cleavage_candidates",
]
