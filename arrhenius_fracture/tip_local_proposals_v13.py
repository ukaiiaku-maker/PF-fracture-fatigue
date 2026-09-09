"""Partition correlation by pre-event tip; never by shared process owner.

The legacy hazard keys and deterministic threshold stream are deliberately
unchanged. Multifront event identity adds (front, candidate, ordinal) at the
proposal boundary, without rekeying or redrawing accepted checkpoint clocks.
"""
from dataclasses import replace
import json

from .directional_competition_v11 import construct_action_proposals
from .tip_directional_observation_v11 import candidate_tip_owners


def construct_tip_local_proposals(state, *, correlation_interval_s):
    competition = state.competition
    owners = candidate_tip_owners(state.crack_network,
                                  (h.candidate_id for h in competition.hazard_states))
    groups = {}
    for hazard in competition.hazard_states:
        tip = owners[hazard.candidate_id]
        branch = state.crack_network.branch(tip)
        # A physical tip is born at one topology opportunity. Distinct tips
        # remain distinct even when they share the parent, junction, and owner.
        opportunity = json.dumps([tip, branch.parent_branch_id, branch.initiation_event],
                                 separators=(',', ':'))
        groups.setdefault((tip, opportunity), []).append(hazard)
    proposals = []
    for (tip, opportunity), hazards in sorted(groups.items()):
        proposals.extend(replace(p, event_owner_tip_id=tip, branch_opportunity_id=opportunity)
                         for p in construct_action_proposals(hazards,
                             correlation_interval_s=correlation_interval_s))
    return tuple(sorted(proposals, key=lambda p: p.action_id))
