"""Explicit semantics for legacy process-engine observations in v11.

The v11 directional clocks own cleavage topology.  The installed process engine
is retained as a conserved material-state observer, and its legacy ``fired``
field can also denote synchronization to an externally committed geometry
checkpoint.  That synchronization is not a time-resolution failure.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping


MODEL_ID = "v11.process_update_semantics/1"


@dataclass(frozen=True)
class ProcessUpdateDecision:
    event_semantics: str
    refinement_required: bool
    refinement_reason: str | None
    physical_hazard_action_step: float | None
    diagnostics: dict[str, Any]


def classify_process_update(
    info: Mapping[str, Any],
    *,
    directional_event_expected: bool,
    permitted_physical_hazard_action: float,
) -> ProcessUpdateDecision:
    fired = bool(info.get("fired", False))
    stochastic = bool(info.get("stochastic_hazard_enabled", False))
    raw_action = info.get("physical_hazard_action_step")
    physical_action = None if raw_action is None else max(float(raw_action), 0.0)
    checkpoint_synchronized = bool(info.get("avalanche_checkpoint_synchronized", False))
    checkpoint_only = (
        fired
        and not stochastic
        and (physical_action is None or physical_action == 0.0)
        and checkpoint_synchronized
    )
    n_fire = int(info.get("n_fire", 0))

    if checkpoint_only:
        semantics = "process_checkpoint_synchronization"
        refinement = False
        reason = None
    elif stochastic and physical_action is not None and physical_action > permitted_physical_hazard_action:
        semantics = "legacy_physical_hazard_crossing"
        refinement = True
        reason = "legacy_physical_hazard_action_exceeds_target"
    elif n_fire > 1:
        semantics = "legacy_physical_hazard_crossing" if stochastic else "ambiguous_process_event"
        refinement = True
        reason = "multiple_legacy_process_events_in_interval"
    elif fired != bool(directional_event_expected):
        semantics = "legacy_physical_hazard_crossing" if stochastic else "ambiguous_process_event"
        refinement = True
        reason = "legacy_process_event_mismatches_directional_topology_event"
    else:
        semantics = (
            "directional_cleavage_threshold_crossing"
            if directional_event_expected
            else "ordinary_process_state_advance"
        )
        refinement = False
        reason = None

    diagnostics = {
        "event_semantics": semantics,
        "refinement_required": refinement,
        "refinement_reason": reason,
        "directional_event_expected": bool(directional_event_expected),
        "fired": fired,
        "n_fire": n_fire,
        "stochastic_hazard_enabled": stochastic,
        "physical_hazard_action_step": physical_action,
        "permitted_physical_hazard_action": float(permitted_physical_hazard_action),
        "checkpoint_synchronized": checkpoint_synchronized,
    }
    return ProcessUpdateDecision(semantics, refinement, reason, physical_action, diagnostics)


__all__ = ["MODEL_ID", "ProcessUpdateDecision", "classify_process_update"]
