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
    "CleavageCandidate", "CompletedDirectionalEvent", "DirectionalHazardState",
    "DirectionalIntervalPreview", "DirectionalRate", "SCHEMA",
    "canonical_candidate_id", "canonical_candidate_inventory",
    "commit_directional_interval", "directional_drive",
    "preview_directional_interval", "preview_production_cleavage_rate",
    "tungsten_cleavage_candidates",
]
