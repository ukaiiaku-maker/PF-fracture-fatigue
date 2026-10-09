"""Conditional morphology marks on already accepted canonical cleavage events.

No engine, directional-clock updater, baseline RNG, or physical-time integrator
is imported. Raw cooperative arrivals are sampled in a subgrid exposure window;
the window never advances the accepted physical endpoint.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from typing import Any, Callable

from scipy.integrate import quad
from scipy.special import gammainc, gammaincc, gammaincinv


BOUNDARY = "BRANCHING_KINETICS_MODEL_UNCALIBRATED"
MODEL_ID = "v13.conditional-raw-multihit-branch-mark/1"
KB_J_PER_K = 1.380649e-23


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def finite(value, name, minimum=0.0):
    if not math.isfinite(value) or value < minimum:
        raise ValueError(f"{name} must be finite and >= {minimum}")


@dataclass(frozen=True)
class ParentCleavageEvent:
    front_lineage: tuple[str, ...]
    process_owner_id: str
    primary_candidate_id: str
    primary_event_id: str
    primary_event_ordinal: int
    accepted_time_s: float
    accepted_opening_m: float
    accepted_state_id: str
    process_state_sha256: str
    cleavage_rng_sha256: str
    process_rng_sha256: str

    def __post_init__(self):
        if not self.front_lineage or not all(self.front_lineage) or self.primary_event_ordinal < 1:
            raise ValueError("accepted cleavage event needs an independent lineage and positive ordinal")
        for name in ("process_owner_id", "primary_candidate_id", "primary_event_id", "accepted_state_id",
                     "process_state_sha256", "cleavage_rng_sha256", "process_rng_sha256"):
            if not getattr(self, name):
                raise ValueError(f"missing accepted parent identity: {name}")
        finite(self.accepted_time_s, "accepted_time_s")
        finite(self.accepted_opening_m, "accepted_opening_m")


@dataclass(frozen=True)
class BranchOpportunityState:
    parent: ParentCleavageEvent
    opportunity_ordinal: int = 0

    def __post_init__(self):
        if self.opportunity_ordinal < 0:
            raise ValueError("negative opportunity ordinal")


@dataclass(frozen=True)
class BranchMarkParameters:
    enabled: bool = False
    exposure_time_s: float = 0.0
    junction_barrier_J: float = 0.0
    overlap_barrier_J: float = 0.0
    branch_seed: int = 0

    def __post_init__(self):
        for name in ("exposure_time_s", "junction_barrier_J", "overlap_barrier_J"):
            finite(getattr(self, name), name)
        if self.enabled and self.exposure_time_s <= 0.0:
            raise ValueError("enabled marks require an explicit positive subgrid exposure")


@dataclass(frozen=True)
class CompanionChannel:
    candidate_id: str
    raw_arrival_rate_per_s: float
    multihit_order: float
    raw_cleavage_barrier_J: float
    temperature_K: float
    junction_distance_m: float
    process_zone_length_m: float
    state_sha256: str
    geometrically_admissible: bool = True
    quenched_barrier_J: float = 0.0
    cooperative_work_J: float = 0.0

    def __post_init__(self):
        if not self.candidate_id or not self.state_sha256:
            raise ValueError("companion requires candidate and complete source-state identities")
        finite(self.raw_arrival_rate_per_s, "raw_arrival_rate_per_s")
        finite(self.multihit_order, "multihit_order", 1.0)
        finite(self.temperature_K, "temperature_K", 1e-300)
        finite(self.junction_distance_m, "junction_distance_m")
        finite(self.process_zone_length_m, "process_zone_length_m", 1e-300)
        for name in ("raw_cleavage_barrier_J", "quenched_barrier_J", "cooperative_work_J"):
            if not math.isfinite(getattr(self, name)):
                raise ValueError(f"nonfinite {name}")

    def arrival_rate(self, parameters):
        # No hard handoff-distance or shared-owner permission switch. The
        # overlap barrier decreases continuously with geometric separation.
        extra = (parameters.junction_barrier_J
                 + parameters.overlap_barrier_J * math.exp(-self.junction_distance_m / self.process_zone_length_m)
                 + self.quenched_barrier_J - self.cooperative_work_J)
        if self.raw_arrival_rate_per_s == 0.0:
            return 0.0
        if extra == 0.0:
            return self.raw_arrival_rate_per_s
        log_rate = math.log(self.raw_arrival_rate_per_s) - extra / (KB_J_PER_K * self.temperature_K)
        try:
            rate = math.exp(log_rate)
        except OverflowError as exc:
            raise ValueError("branch-only Arrhenius rate overflow; no rate cap applied") from exc
        finite(rate, "branch-only arrival rate")
        return rate


@dataclass(frozen=True)
class ConditionalProbabilities:
    single: float
    companions: tuple[tuple[str, float], ...]

    def partition_cleavage_intensity(self, existing_cleavage_rate):
        finite(existing_cleavage_rate, "existing cleavage intensity")
        return (existing_cleavage_rate * self.single,
                tuple((key, existing_cleavage_rate * p) for key, p in self.companions))


def conditional_probabilities(channels, parameters):
    channels = tuple(sorted((c for c in channels if c.geometrically_admissible), key=lambda c: c.candidate_id))
    if len({c.candidate_id for c in channels}) != len(channels):
        raise ValueError("duplicate companion identity")
    if not parameters.enabled or not channels:
        return ConditionalProbabilities(1.0, ())
    rates = [c.arrival_rate(parameters) for c in channels]
    exposure = [r * parameters.exposure_time_s for r in rates]
    if not all(math.isfinite(x) for x in exposure):
        raise ValueError("nonfinite branch exposure; no intensity cap applied")
    hit = [float(gammainc(c.multihit_order, x)) for c, x in zip(channels, exposure)]
    log_single = math.fsum(math.log1p(-p) for p in hit) if all(p < 1.0 for p in hit) else -math.inf
    single = math.exp(log_single)
    total_branch = -math.expm1(log_single)
    if len(channels) == 1:
        return ConditionalProbabilities(single, ((channels[0].candidate_id, hit[0]),))
    weights = []
    for j, channel in enumerate(channels):
        if rates[j] == 0.0 or hit[j] == 0.0:
            weights.append(0.0)
            continue
        # Integrate over the companion's own CDF. This keeps the integration
        # interval bounded even for huge raw rates and separated time scales.
        def survival_product(u):
            t = float(gammaincinv(channel.multihit_order, u)) / rates[j]
            return math.prod(float(gammaincc(c.multihit_order, rates[k] * t))
                             for k, c in enumerate(channels) if k != j)
        value, error = quad(survival_product, 0.0, hit[j], epsabs=1e-12, epsrel=1e-10)
        weights.append(value)
    mass = math.fsum(weights)
    if abs(mass - total_branch) > 1e-9:
        raise RuntimeError("conditional mark quadrature failed probability conservation")
    if mass:
        weights = [p * total_branch / mass for p in weights]
    return ConditionalProbabilities(single, tuple((c.candidate_id, p) for c, p in zip(channels, weights)))


def branch_uniform(opportunity, candidate_id, seed):
    parent = opportunity.parent
    key = {
        "model_id": MODEL_ID, "branch_seed": int(seed),
        "front_lineage": parent.front_lineage,
        "primary_event_id": parent.primary_event_id,
        "primary_event_ordinal": parent.primary_event_ordinal,
        "candidate_id": candidate_id,
        "opportunity_ordinal": opportunity.opportunity_ordinal,
    }
    identity = digest(key)
    integer = int(identity[:16], 16) >> 12
    return (integer + 1) / (2**52 + 1), identity


@dataclass(frozen=True)
class BranchMark:
    companion_id: str | None
    companion_completion_exposure_s: float | None
    branch_rng_identities: tuple[tuple[str, str], ...]
    baseline_physical_time_increment_s: float = 0.0


def draw_branch_mark(opportunity, channels, parameters):
    if not parameters.enabled:
        return BranchMark(None, None, ())
    channels = tuple(sorted(channels, key=lambda c: c.candidate_id))
    if len({c.candidate_id for c in channels}) != len(channels):
        raise ValueError("duplicate companion identity")
    completed, identities = [], []
    for channel in channels:
        if channel.candidate_id == opportunity.parent.primary_candidate_id:
            raise ValueError("primary cannot be its own companion")
        if channel.state_sha256 != opportunity.parent.process_state_sha256:
            raise ValueError("stale companion process state")
        if not channel.geometrically_admissible:
            continue
        rate = channel.arrival_rate(parameters)
        if rate == 0.0:
            continue
        u, identity = branch_uniform(opportunity, channel.candidate_id, parameters.branch_seed)
        identities.append((channel.candidate_id, identity))
        time = float(gammaincinv(channel.multihit_order, u)) / rate
        if time <= parameters.exposure_time_s:
            completed.append((time, channel.candidate_id))
    if not completed:
        return BranchMark(None, None, tuple(identities))
    time, candidate = min(completed)
    return BranchMark(candidate, time, tuple(identities))


@dataclass(frozen=True)
class MarkedCleavageResult:
    state: Any
    mark: BranchMark
    disposition: str
    companion_rejection_reason: str | None = None


def apply_conditional_branch_mark(*, parent, baseline_single_state, parameters,
                                  channels: Callable, exact_pair_trial: Callable,
                                  opportunity_ordinal=0):
    """Apply only to an existing accepted single-arm result.

    The caller supplies its immutable accepted state and an isolated exact
    pair trial. Disabled/no-event/no-companion/rejected-pair paths return the
    identical baseline object. A pair trial cannot cancel the primary event.
    """
    if parent is None or not parameters.enabled:
        return MarkedCleavageResult(baseline_single_state, BranchMark(None, None, ()),
            "BRANCH_DISABLED" if not parameters.enabled else "NO_CLEAVAGE_EVENT")
    opportunity = BranchOpportunityState(parent, opportunity_ordinal)
    mark = draw_branch_mark(opportunity, channels(), parameters)
    if mark.companion_id is None:
        return MarkedCleavageResult(baseline_single_state, mark, "SINGLE_NO_COMPANION")
    result = exact_pair_trial(parent, mark)
    if not result.accepted:
        reason = getattr(result, "rejection_reason", None) or getattr(result, "reason", None)
        if not reason:
            raise RuntimeError("rejected companion requires an exact rejection reason")
        return MarkedCleavageResult(baseline_single_state, mark, "SINGLE_COMPANION_REJECTED", reason)
    return MarkedCleavageResult(result.state, mark, "PAIR_ACCEPTED")
