"""Pure fail-closed decisions for the bounded current-source V2 audit."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping


CLAIM_LABEL = "CAPABILITY_DEMONSTRATION_NOT_VALIDATED_BRANCHING_PHYSICS"


MORPHOLOGY_REQUIRED_GATES = (
    "run_reached_300um",
    "committed_daughter_birth",
    "daughter_non_stub_growth",
    "no_cross_wake_bridge_or_reconnection",
    "exact_length_topology_closure",
    "valid_cluster_bookkeeping",
    "independent_handoff_when_required",
    "no_branch_cap_clipping",
    "no_backward_growth",
    "birth_local_probes_reliable",
    "hazard_rng_state_geometry_provenance",
    "run_completed_without_fail_closed_exception",
)


def morphology_capability(gates: Mapping[str, bool]) -> bool:
    """Final local probes are not an unconditional morphology gate."""
    return all(bool(gates.get(name, False)) for name in MORPHOLOGY_REQUIRED_GATES)


def independent_tip_mechanics(gates: Mapping[str, bool]) -> bool:
    return bool(gates.get("final_local_probes_reliable", False))


def branch_birth_mechanics(gates: Mapping[str, bool]) -> bool:
    return bool(gates.get("committed_daughter_birth", False)) and bool(
        gates.get("birth_local_probes_reliable", False)
    )


@dataclass(frozen=True)
class TwoAxisDecision:
    morphology_capability_decision: str
    independent_tip_mechanics_decision: str
    cluster_handoff_decision: str
    predictive_branching_physics_validated: bool = False


def two_axis_decision(gates: Mapping[str, bool]) -> TwoAxisDecision:
    morphology = morphology_capability(gates)
    independent = independent_tip_mechanics(gates)
    handoff = bool(gates.get("independent_handoff_when_required", False))
    cluster_decision = (
        "NOT_TRIGGERED_UNRESOLVED_CLUSTER"
        if bool(gates.get("cluster_unresolved", False)) else
        "CONDITIONAL_GATE_SATISFIED" if handoff else
        "CONDITIONAL_GATE_NOT_SATISFIED"
    )
    return TwoAxisDecision(
        morphology_capability_decision=(
            "CURRENT_SOURCE_BRANCHING_MORPHOLOGY_CAPABILITY_DEMONSTRATED"
            if morphology else "CURRENT_SOURCE_BRANCHING_MORPHOLOGY_CAPABILITY_NOT_DEMONSTRATED"
        ),
        independent_tip_mechanics_decision=(
            "QUALIFIED" if independent else "UNQUALIFIED_FINAL_LOCAL_CONTOURS"
        ),
        cluster_handoff_decision=cluster_decision,
    )


__all__ = [
    "CLAIM_LABEL", "MORPHOLOGY_REQUIRED_GATES", "TwoAxisDecision",
    "branch_birth_mechanics", "independent_tip_mechanics",
    "morphology_capability", "two_axis_decision",
]
