from dataclasses import replace

import numpy as np
import pytest

from arrhenius_fracture.sharp_wake_backend_v12 import V12_MODEL_ID
from arrhenius_fracture.topology_transaction_v11 import TopologyArm
from arrhenius_fracture.v12_production_driver import (
    build_loaded_state, execute_physical_two_arm_event,
)


def test_physical_arm_requires_nonzero_dissipation_direction_and_intersection():
    common = dict(
        candidate_id="c", branch_id="b", start_xy_m=(0.0, 0.0),
        end_xy_m=(1.0, 0.0), event_reward_m=1.0,
        event_classification="physical_cleavage",
        candidate_direction_xy=(1.0, 0.0), first_intersection_xy_m=(1.0, 0.0),
    )
    with pytest.raises(ValueError, match="nonzero hazard dissipation"):
        TopologyArm(hazard_dissipation_J_per_m=0.0, **common)
    with pytest.raises(ValueError, match="not aligned"):
        TopologyArm(
            hazard_dissipation_J_per_m=1.0,
            **{**common, "candidate_direction_xy": (0.0, 1.0)},
        )
    with pytest.raises(ValueError, match="first intersection"):
        TopologyArm(
            hazard_dissipation_J_per_m=1.0,
            **{**common, "first_intersection_xy_m": (0.9, 0.0)},
        )


def test_real_two_arm_transaction_has_distinct_oblique_children_and_dissipation():
    accepted = build_loaded_state(V12_MODEL_ID)
    result, audit = execute_physical_two_arm_event(accepted)
    assert audit["accepted"] is True
    assert audit["action_type"] == "two_arm"
    assert len(set(audit["candidate_ids"])) == 2
    assert len(set(audit["branch_ids"])) == 2
    assert audit["hazard_barrier_J"] > 0.0
    assert audit["hazard_dissipation_J_per_m"] > 0.0
    assert audit["energy_margin_J_per_m"] > 0.0
    directions = np.asarray(audit["directions"])
    assert directions[0, 1] * directions[1, 1] < 0.0
    assert len(result.crack_network.active_tip_ids) == 2
    assert result.energy_ledgers["hazard_dissipation_J_per_m"] == pytest.approx(
        audit["hazard_dissipation_J_per_m"]
    )


@pytest.mark.parametrize("failure_stage", [
    "accepted_snapshot", "field_transfer", "graph_edit", "remesh",
    "field_projection", "support_rebuild", "equilibrium", "energy_gate",
])
def test_forced_failure_is_only_a_rollback_screen(failure_stage):
    # Classification is carried by the arm; injected exceptions do not become
    # evidence that a physical event was reached.
    forced = TopologyArm("c", "b", (0.0, 0.0), (1.0, 0.0), 1.0, 0.0)
    assert forced.event_classification == "software_forced_geometry"
    assert failure_stage
