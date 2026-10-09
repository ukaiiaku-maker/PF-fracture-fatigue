"""Order-independent physical growth metrics for a branched crack graph."""
from __future__ import annotations

from dataclasses import dataclass
import math

from .crack_network_v11 import CrackNetworkState


@dataclass(frozen=True)
class CrackGrowthMetrics:
    network_total_new_crack_length_m: float
    max_root_to_tip_path_extension_m: float
    max_forward_projected_extension_m: float

    def to_dict_um(self) -> dict[str, float]:
        return {
            "network_total_new_crack_length_um": self.network_total_new_crack_length_m * 1e6,
            "max_root_to_tip_path_extension_um": self.max_root_to_tip_path_extension_m * 1e6,
            "max_forward_projected_extension_um": self.max_forward_projected_extension_m * 1e6,
        }


def crack_growth_metrics(
    network: CrackNetworkState, *, initial_crack_length_m: float,
) -> CrackGrowthMetrics:
    """Measure new crack without summing parallel arms for the stopping metric.

    A branch path owns only its geometrically realized segments.  The connected
    root-to-tip distance is consequently the sum of branch path lengths along
    the unique parent chain.  The initial root-crack length is subtracted once.
    """
    initial = float(initial_crack_length_m)
    if not math.isfinite(initial) or initial < 0.0:
        raise ValueError("initial_crack_length_m must be finite and nonnegative")
    by_id = {branch.branch_id: branch for branch in network.branches}
    path_length: dict[str, float] = {}

    def distance(branch_id: str) -> float:
        if branch_id in path_length:
            return path_length[branch_id]
        branch = by_id[branch_id]
        parent_distance = 0.0 if branch.parent_branch_id is None else distance(branch.parent_branch_id)
        value = parent_distance + branch.physical_path_length_m
        path_length[branch_id] = value
        return value

    root_to_tip = max(distance(branch.branch_id) for branch in network.branches)
    root_x = network.branch(network.primary_branch_id).path[0][0]
    furthest_x = max(branch.tip[0] for branch in network.branches)
    return CrackGrowthMetrics(
        network_total_new_crack_length_m=max(0.0, network.total_physical_crack_length_m - initial),
        max_root_to_tip_path_extension_m=max(0.0, root_to_tip - initial),
        max_forward_projected_extension_m=max(0.0, furthest_x - (root_x + initial)),
    )


__all__ = ["CrackGrowthMetrics", "crack_growth_metrics"]
