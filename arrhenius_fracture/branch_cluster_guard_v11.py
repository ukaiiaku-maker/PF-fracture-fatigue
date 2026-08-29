"""Fail-closed validity guard for a shared unresolved v11 branch cluster."""
from __future__ import annotations

from dataclasses import dataclass
import math

from .branch_cluster_v11 import BranchClusterState
from .crack_network_v11 import CrackNetworkState


TERMINATION_STATUS = "branch_cluster_independent_tip_handoff_required"


@dataclass(frozen=True)
class BranchClusterGuardDiagnostic:
    cluster_id: str
    tip_separation_m: float
    arm_arclengths_from_junction_m: tuple[float, float]
    branch_handoff_length_m: float
    local_J_contour_radius_m: float
    local_contours_overlap: bool
    independently_valid_local_J: tuple[bool, bool]
    sufficient_post_junction_length: tuple[bool, bool]
    separation_reaches_process_zone: bool
    handoff_required: bool
    termination_status: str | None

    def to_dict(self) -> dict:
        return {
            "cluster_id": self.cluster_id,
            "tip_separation_m": self.tip_separation_m,
            "arm_arclengths_from_junction_m": list(self.arm_arclengths_from_junction_m),
            "branch_handoff_length_m": self.branch_handoff_length_m,
            "local_J_contour_radius_m": self.local_J_contour_radius_m,
            "local_contours_overlap": self.local_contours_overlap,
            "independently_valid_local_J": list(self.independently_valid_local_J),
            "sufficient_post_junction_length": list(self.sufficient_post_junction_length),
            "separation_reaches_process_zone": self.separation_reaches_process_zone,
            "handoff_required": self.handoff_required,
            "termination_status": self.termination_status,
            "guard_scale_contract": {
                "minimum_arm_length": "max(branch_handoff_length, local_J_contour_radius)",
                "minimum_tip_separation": "branch_handoff_length",
                "nonoverlap": "tip_separation >= 2*local_J_contour_radius",
                "fitted_parameters": False,
            },
        }


def evaluate_unresolved_cluster_guard(
    network: CrackNetworkState,
    cluster: BranchClusterState,
    *,
    branch_handoff_length_m: float,
    local_J_contour_radius_m: float,
    independently_valid_local_J: tuple[bool, bool],
) -> BranchClusterGuardDiagnostic:
    """Stop only when every inherited scale/contour test identifies two tips."""
    handoff_length = float(branch_handoff_length_m)
    radius = float(local_J_contour_radius_m)
    if not math.isfinite(handoff_length) or handoff_length <= 0.0:
        raise ValueError("branch handoff length must be finite and positive")
    if not math.isfinite(radius) or radius <= 0.0:
        raise ValueError("local J-contour radius must be finite and positive")
    if not cluster.unresolved:
        raise ValueError("the unresolved-cluster guard cannot evaluate a resolved cluster")
    arms = tuple(network.branch(branch_id) for branch_id in cluster.arm_branch_ids)
    if any(branch.status != "active" for branch in arms):
        # Coalesced/arrested arms cannot become two independently active tips.
        valid = tuple(False for _ in arms)
    else:
        valid = tuple(bool(value) for value in independently_valid_local_J)
    if len(valid) != 2:
        raise ValueError("local-J validity requires one value per cluster arm")
    junction = cluster.junction_xy_m
    lengths = tuple(
        math.fsum(
            math.dist(a, b)
            for a, b in zip(branch.path, branch.path[1:])
        )
        for branch in arms
    )
    if any(branch.root != junction for branch in arms):
        raise ValueError("cluster arm does not begin at the branch junction")
    separation = math.dist(arms[0].tip, arms[1].tip)
    minimum_length = max(handoff_length, radius)
    sufficient = tuple(length >= minimum_length for length in lengths)
    overlaps = separation < 2.0 * radius
    separation_ready = separation >= handoff_length
    required = all(sufficient) and separation_ready and all(valid) and not overlaps
    return BranchClusterGuardDiagnostic(
        cluster_id=cluster.cluster_id,
        tip_separation_m=separation,
        arm_arclengths_from_junction_m=lengths,
        branch_handoff_length_m=handoff_length,
        local_J_contour_radius_m=radius,
        local_contours_overlap=overlaps,
        independently_valid_local_J=valid,
        sufficient_post_junction_length=sufficient,
        separation_reaches_process_zone=separation_ready,
        handoff_required=required,
        termination_status=TERMINATION_STATUS if required else None,
    )


__all__ = [
    "BranchClusterGuardDiagnostic", "TERMINATION_STATUS",
    "evaluate_unresolved_cluster_guard",
]
