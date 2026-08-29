from dataclasses import replace

import pytest

from arrhenius_fracture.directional_competition_v11 import (
    CompletedDirectionalEvent, DirectionalCompetitionState,
)
from arrhenius_fracture.process_state_ownership_v11 import (
    ProcessStateOwner, ProcessStateOwnerRegistry,
)
from arrhenius_fracture.resolved_tip_state_v11 import resolve_unresolved_cluster
from tests.test_v11_resolved_tip_handoff import _cluster, _fresh


def _root():
    return ProcessStateOwnerRegistry(
        {"root": "root"},
        {"root": ProcessStateOwner("root", "single_tip_engine", process_engine_id="root")},
    )


def _validate(registry, engines, clusters=(), reservoirs=()):
    registry.validate(
        registry.owner_by_tip, engine_ids=engines, cluster_ids=clusters,
        reservoir_ids=reservoirs,
    )


def test_three_generation_ownership_transitions_are_reference_safe():
    registry = _root()
    _validate(registry, {"root"})
    registry = registry.branch("root", "C1", ("T1", "T2"))
    _validate(registry, {"C1"}, {"C1"})
    assert {registry.owner(tip).owner_type for tip in ("T1", "T2")} == {"unresolved_branch_cluster"}

    registry = registry.resolve("C1", ("T1", "T2"), "reservoir:C1")
    _validate(registry, {"T1", "T2"}, {"C1"}, {"reservoir:C1"})
    registry = registry.branch("T1", "C2", ("T1a", "T1b"))
    _validate(registry, {"C2", "T2"}, {"C1", "C2"}, {"reservoir:C1"})
    registry = registry.resolve("C2", ("T1a", "T1b"), "reservoir:C2")
    _validate(
        registry, {"T1a", "T1b", "T2"}, {"C1", "C2"},
        {"reservoir:C1", "reservoir:C2"},
    )
    registry = registry.branch("T1a", "C3", ("T1aa", "T1ab"))
    _validate(
        registry, {"C3", "T1b", "T2"}, {"C1", "C2", "C3"},
        {"reservoir:C1", "reservoir:C2"},
    )


def test_unresolved_arm_cannot_spawn_nested_cluster_and_sibling_owner_survives():
    registry = _root().branch("root", "jd5e346f22e115e5", ("b584", "b65"))
    with pytest.raises(ValueError, match="parent_process_zone_still_unresolved"):
        registry.branch("b584", "jaa8", ("child-a", "child-b"))
    assert registry.owner_by_tip == {
        "b584": "jd5e346f22e115e5", "b65": "jd5e346f22e115e5",
    }
    assert registry.reference_count("jd5e346f22e115e5") == 2
    _validate(registry, {"jd5e346f22e115e5"}, {"jd5e346f22e115e5"})


def test_handoff_preserves_pending_first_passage_events_for_revalidation():
    candidates, network, cluster = _cluster()
    base = DirectionalCompetitionState.initialize(candidates, global_hazard_seed=3621)
    event = CompletedDirectionalEvent(candidates[0].candidate_id, 1, 1.25, 0.0, 1.0, 1.0)
    hazards = list(base.hazard_states)
    hazards[0] = replace(hazards[0], pending_events=(event,))
    pending = replace(base, hazard_states=tuple(hazards))
    existing = {tip: pending for tip in cluster.arm_branch_ids}
    result = resolve_unresolved_cluster(
        network, cluster, candidates=candidates, global_hazard_seed=3621,
        fresh_tip_factory=_fresh, existing_competitions=existing,
    )
    for tip in cluster.arm_branch_ids:
        assert result.tips[tip].competition == pending
        assert result.tips[tip].competition.pending_events == (event,)
