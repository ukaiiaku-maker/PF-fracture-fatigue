"""Default-off post-acceptance adapter; the canonical interval runs exactly once.

Not enabled in the PF campaign CLI. Parent/channel factories must use the
accepted process state. Pair factories must call the isolated exact topology
trial; they must not repeat process evolution or physical renewal.
"""
from dataclasses import dataclass

from .conditional_branch_mark_v13 import apply_conditional_branch_mark
from .production_step_loop_v11 import advance_accepted_step


@dataclass(frozen=True)
class MarkedAcceptedStep:
    canonical_result: object
    mark_result: object

    @property
    def state(self):
        return self.mark_result.state


def advance_marked_accepted_step(accepted, context, *, branch_parameters,
                               parent_factory, channels_factory,
                               exact_pair_trial_factory, **canonical_callbacks):
    canonical = advance_accepted_step(accepted, context, **canonical_callbacks)
    parent = None
    chosen = [trial for trial in canonical.trials if trial.selected]
    if branch_parameters.enabled and chosen:
        if len(chosen) != 1 or chosen[0].proposal.action_type != "one_arm":
            raise ValueError("V13 requires the canonical accepted single-arm parent, not legacy branching")
        parent = parent_factory(canonical, context)
        proposal = chosen[0].proposal
        if (parent.primary_candidate_id != proposal.member_candidate_ids[0]
                or parent.primary_event_id != proposal.member_event_ids[0]
                or parent.primary_event_ordinal != proposal.member_event_ordinals[0]
                or parent.accepted_time_s != context.physical_time_s + context.duration_s):
            raise ValueError("mark parent differs from the immutable canonical cleavage endpoint")
    result = apply_conditional_branch_mark(parent=parent, baseline_single_state=canonical.state,
        parameters=branch_parameters, channels=lambda: channels_factory(parent, canonical),
        exact_pair_trial=lambda p, m: exact_pair_trial_factory(accepted, canonical, p, m))
    # Immutable objects are deliberately retained by the V13 exact-pair
    # adapter. Reject an adapter that attempts a second physical renewal,
    # clock consumption or RNG replacement.
    for name in ("competition", "rng_state", "tip_process_state", "event_counters"):
        if getattr(result.state, name) is not getattr(canonical.state, name):
            raise RuntimeError(f"pair adapter replaced immutable canonical {name}")
    return MarkedAcceptedStep(canonical, result)
