"""Read-only frozen residual-clock race. No new clocks, RNG or process updates.

Actions and thresholds are cumulative and must share one observation time.
The exponential ensemble identity is not a probability for a frozen archive.
"""
from dataclasses import dataclass
import math
from itertools import groupby

from .conditional_branch_mark_v13 import BranchMark, MarkedCleavageResult, BOUNDARY

MODEL_ID = 'v13.inherited-companion-versus-renewed-primary/1'


@dataclass(frozen=True)
class ClockProjection:
    candidate_id: str
    action: float
    threshold: float
    event_ordinal: int
    threshold_rng_identity: str
    effective_rate_per_s: float
    observation_identity: str
    pending_event_id: str | None = None

    def __post_init__(self):
        if (not self.candidate_id or not self.threshold_rng_identity or
                not self.observation_identity or self.event_ordinal < 1 or
                not all(math.isfinite(x) for x in
                        (self.action, self.threshold, self.effective_rate_per_s)) or
                min(self.action, self.threshold) < 0 or self.effective_rate_per_s < 0 or
                (self.action > self.threshold and self.pending_event_id is None) or
                (self.pending_event_id is not None and self.action < self.threshold)):
            raise ValueError('inadmissible or already-pending clock projection')

    @property
    def completion_s(self):
        if self.pending_event_id is not None:
            return 0.
        residual = self.threshold - self.action
        if residual == 0:
            return 0.
        return residual / self.effective_rate_per_s if self.effective_rate_per_s else math.inf


def exponential_ensemble_probability(primary_rate, companion_rate, tau_c, *, pair_admissible=True):
    if (not all(math.isfinite(x) and x >= 0 for x in (primary_rate, companion_rate, tau_c))
            or tau_c == 0):
        raise ValueError('invalid exponential race scales')
    if not pair_admissible or companion_rate == 0:
        return 0.
    scale = max(primary_rate, companion_rate)
    normalized_sum = primary_rate / scale + companion_rate / scale
    return (companion_rate / scale / normalized_sum) * (-math.expm1(-scale * tau_c * normalized_sum))


def apply_primary_continuation_race(*, parent, baseline_single_state, primary,
                                  companions, tau_c, exact_pair_trial, enabled=False):
    if not enabled:
        return MarkedCleavageResult(baseline_single_state, BranchMark(None, None, ()), 'BRANCH_DISABLED')
    if (not math.isfinite(tau_c) or tau_c <= 0 or parent is None or
            primary.candidate_id != parent.primary_candidate_id or
            parent.primary_event_id not in baseline_single_state.competition.consumed_event_ids):
        raise ValueError('race requires the accepted canonical primary and positive tau_c')
    companions = tuple(companions)
    ids = [c.candidate_id for c in companions]
    if (len(ids) != len(set(ids)) or primary.candidate_id in ids or
            any(c.observation_identity != primary.observation_identity for c in companions)):
        raise ValueError('mixed clock identities/time origins or duplicate candidates')
    bound = min(primary.completion_s, tau_c)
    eligible = sorted((c for c in companions if c.completion_s < bound),
                      key=lambda c: (c.completion_s, c.candidate_id))
    for _, group in groupby(eligible, key=lambda c:c.completion_s):
        admissible = []
        for candidate in group:
            mark = BranchMark(candidate.candidate_id, candidate.completion_s, ())
            result = exact_pair_trial(parent, mark)
            if not result.accepted:
                if not result.rejection_reason or result.state is not baseline_single_state:
                    raise RuntimeError('pair veto must preserve exact single-arm fallback and reason')
                continue
            for name in ('competition', 'rng_state', 'tip_process_state', 'event_counters'):
                if getattr(result.state, name) is not getattr(baseline_single_state, name):
                    raise RuntimeError('pair changed canonical ' + name)
            admissible.append((result, mark))
        if len(admissible)>1:
            return MarkedCleavageResult(baseline_single_state, BranchMark(None, None, ()), 'SINGLE_UNRESOLVED_COMPANION_TIE')
        if admissible:
            result, mark = admissible[0]
            return MarkedCleavageResult(result.state, mark, 'PAIR_ACCEPTED')
    return MarkedCleavageResult(baseline_single_state, BranchMark(None, None, ()), 'SINGLE_PRIMARY_OR_EXPIRY_OR_PAIR_VETO')
