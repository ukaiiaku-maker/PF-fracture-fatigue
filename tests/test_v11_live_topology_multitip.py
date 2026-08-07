from dataclasses import replace
import math

import pytest

from arrhenius_fracture.crack_network_v11 import CrackBranchState
from arrhenius_fracture.live_topology_kernel_v11 import MAXIMUM_FRONTS_SUPPORTED, evaluate_exact_topology
from tests.test_live_topology_kernel_v11 import live_straight_request


def test_exact_provider_measures_three_active_tips_without_topology_interpolation():
    request = live_straight_request(0.0)
    root = request.crack_network.branch(request.crack_network.primary_branch_id)
    children = tuple(CrackBranchState(
        f"tip-{index}", root.branch_id, 1, 1,
        (root.tip, (root.tip[0] + 3e-6, root.tip[1] + offset)),
        (math.atan2(offset, 3e-6),),
    ) for index, offset in enumerate((-3e-6, 0.0, 3e-6)))
    network = replace(request.crack_network, branches=(replace(root, status="terminated"),) + children, branching_enabled=True)
    candidates = next(iter(request.candidates_by_tip.values()))
    result = evaluate_exact_topology(replace(
        request, crack_network=network,
        candidates_by_tip={branch.branch_id: candidates for branch in children},
        cluster_frame={"mode": "multi_tip_test"},
    ))
    assert MAXIMUM_FRONTS_SUPPORTED >= 3
    assert result["maximum_fronts_supported"] == MAXIMUM_FRONTS_SUPPORTED
    assert len(result["tips"]) == 3
    equilibrium = result["base_equilibrium"]
    assert equilibrium["applied_displacement"] > 0.0
    assert equilibrium["reaction_force"] != 0.0
    assert equilibrium["apparent_compliance"] == pytest.approx(
        equilibrium["applied_displacement"] / abs(equilibrium["reaction_force"])
    )
    assert all(math.isfinite(row["signed_J_J_per_m2"]) for tip in result["tips"] for row in tip["directional"])
    assert result["interpolation_permitted"] is False
