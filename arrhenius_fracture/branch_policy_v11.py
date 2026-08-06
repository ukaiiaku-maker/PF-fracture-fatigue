"""Administrative safety limits for the bounded production branch state machine."""
from __future__ import annotations

from dataclasses import dataclass

from .directional_competition_v11 import CompetingActionProposal


MAX_BRANCH_BIRTHS = 8
BRANCH_CAP_VETO = "maximum_branch_birth_count_reached"


@dataclass(frozen=True)
class BranchPolicyDecision:
    permitted: bool
    veto_reason: str | None = None


def branch_birth_policy(
    proposal: CompetingActionProposal,
    *,
    committed_branch_birth_count: int,
    maximum_branch_births: int = MAX_BRANCH_BIRTHS,
) -> BranchPolicyDecision:
    count = int(committed_branch_birth_count)
    maximum = int(maximum_branch_births)
    if count < 0 or maximum < 0:
        raise ValueError("branch-birth counts must be nonnegative")
    if proposal.action_type == "two_arm" and count >= maximum:
        return BranchPolicyDecision(False, BRANCH_CAP_VETO)
    return BranchPolicyDecision(True)


__all__ = [
    "BRANCH_CAP_VETO", "MAX_BRANCH_BIRTHS", "BranchPolicyDecision",
    "branch_birth_policy",
]
