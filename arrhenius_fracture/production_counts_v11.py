"""Unambiguous bounded-network production counters."""
from __future__ import annotations

from collections import Counter

from .branch_policy_v11 import MAX_BRANCH_BIRTHS


def production_front_counts(state) -> dict[str, int]:
    statuses = Counter(branch.status for branch in state.crack_network.branches)
    return {
        "network_branch_object_count": len(state.crack_network.branches),
        "committed_branch_birth_count": int(state.event_counters.get("branch_birth_count", 0)),
        "maximum_branch_births": MAX_BRANCH_BIRTHS,
        "active_front_count": statuses["active"],
        "terminated_front_count": statuses["terminated"],
        "merged_front_count": statuses["merged"],
    }


__all__ = ["production_front_counts"]
