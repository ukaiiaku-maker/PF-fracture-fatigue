from dataclasses import replace
import hashlib

import numpy as np
import pytest

from arrhenius_fracture.current_source_multifront_hooks_v12 import (
    CurrentSourceHookBindingError, build_current_source_production_hooks,
    current_source_hook_map, restore_complete_current_source_engine,
)
from arrhenius_fracture.directional_competition_v11 import (
    CleavageCandidate, DirectionalCompetitionState, DirectionalHazardState,
    competition_state_from_dict, competition_state_to_dict,
)
from arrhenius_fracture.general_multifront_v12 import (
    FrontRuntimeState, MultiFrontRuntimeState, ProcessEngineState,
    ResourcePolicy, TopologyProposal, commit_selected_proposal,
)
from arrhenius_fracture.multifront_checkpoint_v12 import (
    load_production_multifront_checkpoint, write_production_multifront_checkpoint,
)
from arrhenius_fracture.multifront_competition_v12 import (
    daughter_competitions_from_v11_lineage, finalize_competitions,
    preview_competitions,
)
from arrhenius_fracture.production_multifront_v12 import (
    ProposalTrialOutcome, accepted_fem_state_fingerprint,
    commit_exact_trial_outcome, rollover_accepted_state,
)
from arrhenius_fracture.sharp_front_v11_branching import _capture_shared_engine
from arrhenius_fracture.branch_checkpoint_v11 import restore_branch_checkpoint
from arrhenius_fracture.multifront_checkpoint_v12 import load_v11_checkpoint_as_v12
from arrhenius_fracture.restart_family_migration_v11 import process_state_digest, rng_digest
from arrhenius_fracture.topology_transaction_v11 import extend_network_arm, TopologyArm
from arrhenius_fracture.sharp_front_current_source_multifront_v12 import (
    v6_2_production_adapter_dry_run,
)


class MPZFixture:
    def __init__(self):
        self.mobile = np.array([[1.0, 2.0]])
        self.retained = np.array([[3.0, 4.0]])
        self.advance_total_m = 2.5e-6


class EngineFixture:
    MODEL_ID = "fixture.production-engine/1"

    def __init__(self):
        self.mpz = MPZFixture()
        self.B = 0.75
        self._hazard_rng_state = {"position": 17}


def candidate(name, direction, normal):
    return CleavageCandidate.create(
        plane_family="100", plane_variant=name, direction_xy=direction,
        normal_xy=normal, gamma_rel=1.0, orientation_convention="fixture",
    )


def runtime_fixture():
    candidates = (
        candidate("up", (1.0, 0.0), (0.0, 1.0)),
        candidate("down", (0.0, 1.0), (-1.0, 0.0)),
    )
    competition = DirectionalCompetitionState(
        candidates=candidates,
        hazard_states=tuple(
            DirectionalHazardState(item.candidate_id) for item in candidates
        ),
        global_hazard_seed=3621,
    )
    from arrhenius_fracture.crack_network_v11 import CrackNetworkState
    network = CrackNetworkState.one_tip(((0.0, 0.0),), initial_orientation_rad=0.0)
    front = network.active_tip_ids[0]
    payload = _capture_shared_engine(EngineFixture())
    engine = ProcessEngineState.from_v11_payload(
        engine_id="engine:root", source_state_id="source:v5.3", payload=payload,
        active_ledgers={"mobile": 3.0}, wake_ledgers={"retained": 4.0},
        signed_system_ledgers={"positive": 7.0}, update_count=0,
        event_renewal_count=0, local_process_coordinate_m=2.5e-6,
        family_identity="v5.3:745um:exact-prefix",
    )
    runtime = MultiFrontRuntimeState.one_front(
        network,
        FrontRuntimeState(
            front, competition_state_to_dict(competition),
            tuple(item.candidate_id for item in candidates), {"seed": 3621},
        ),
        engine, resource_policy=ResourcePolicy(
            "mechanistic", None, None, "experimental",
            "v11_correlated_proposal_compatibility",
        ), accepted_state_id="accepted:0", stress_field_state_id="stress:0",
    )
    return runtime, candidates


def fem_state(runtime):
    from types import SimpleNamespace
    return SimpleNamespace(
        crack_network=runtime.crack_network,
        displacement=np.array([0.0, 1.0]), ep_gp=np.array([[0.1]]),
        rho_gp=np.array([[2.0]]), damage=np.array([0.0]),
        elasticity_D=np.eye(3),
    )


def test_complete_engine_payload_restores_every_engine_and_mpz_field():
    runtime, _ = runtime_fixture()
    state = next(iter(runtime.process_engines.values()))
    restored = state.restore_into(EngineFixture())
    assert restored.B == 0.75
    assert restored._hazard_rng_state == {"position": 17}
    assert np.array_equal(restored.mpz.mobile, [[1.0, 2.0]])
    assert np.array_equal(restored.mpz.retained, [[3.0, 4.0]])
    assert state.checkpoint_payload_sha256 == state.opaque_state_fingerprint
    assert "mpz.mobile" in state.checkpoint_field_inventory


def test_two_phase_preview_is_pure_and_finalization_consumes_selected_only():
    runtime, candidates = runtime_fixture()
    front = runtime.active_front_ids[0]
    before = runtime.to_dict()
    endpoints = {front: {
        candidates[0].candidate_id: (1e-6, 0.0),
        candidates[1].candidate_id: (0.0, 1e-6),
    }}
    preview = preview_competitions(
        runtime, {front: {item.candidate_id: 1.1 for item in candidates}}, endpoints,
        start_time_s=0.0, duration_s=1.0, correlation_interval_s=1.0,
    )
    assert runtime.to_dict() == before
    pair = next(item for item in preview.proposals if item.action_type == "two_arm")
    finalized = finalize_competitions(
        runtime, preview, selected_proposal=pair, accepted=True,
    )
    assert set(finalized.selected_event_ids) == set(pair.member_event_ids)
    assert all(value == "selected_consumed" for value in finalized.disposition_by_event_id.values())
    assert finalized.runtime.front_runtimes[front].interval_count == 1
    rejected = finalize_competitions(
        runtime, preview, selected_proposal=pair, accepted=False,
    )
    assert rejected.runtime is runtime
    assert rejected.competition_hashes_before == rejected.competition_hashes_after


def test_unselected_completed_event_remains_pending():
    runtime, candidates = runtime_fixture()
    front = runtime.active_front_ids[0]
    endpoints = {front: {
        candidates[0].candidate_id: (1e-6, 0.0),
        candidates[1].candidate_id: (0.0, 1e-6),
    }}
    preview = preview_competitions(
        runtime,
        {front: {candidates[0].candidate_id: 1.1, candidates[1].candidate_id: 0.0}},
        endpoints, start_time_s=0.0, duration_s=1.0, correlation_interval_s=0.0,
    )
    selected = preview.proposals[0]
    result = finalize_competitions(
        runtime, preview, selected_proposal=selected, accepted=True,
    )
    assert result.selected_event_ids == selected.member_event_ids


def test_daughter_lineage_retains_complete_competition_not_marker():
    runtime, _ = runtime_fixture()
    parent = runtime.front_runtimes[runtime.active_front_ids[0]]
    daughters = daughter_competitions_from_v11_lineage(parent, ("d2", "d1"))
    assert daughters["d1"].competition_state == parent.competition_state
    assert "renewed_from_parent" not in daughters["d1"].competition_state
    assert daughters["d1"] is not daughters["d2"]


def test_binary_birth_keeps_full_clocks_but_activates_only_assigned_direction():
    runtime, candidates = runtime_fixture()
    front = runtime.active_front_ids[0]
    tip = runtime.crack_network.branch(front).tip
    proposal = TopologyProposal(
        "proposal:binary-active", "two_arm", front,
        runtime.owner_by_front[front],
        tuple(item.candidate_id for item in candidates), 0.5,
        ((tip[0] + 1e-6, tip[1]), (tip[0], tip[1] + 1e-6)),
    )
    born = commit_selected_proposal(runtime, proposal)
    assert len(born.active_front_ids) == 2
    for daughter in born.active_front_ids:
        state = born.front_runtimes[daughter]
        assigned = born.crack_network.branch(daughter).local_state["candidate_id"]
        assert state.candidate_ids == tuple(sorted(x.candidate_id for x in candidates))
        assert state.mechanically_active_candidate_ids == (assigned,)
        assert state.dormant_candidate_ids == tuple(
            candidate_id for candidate_id in state.candidate_ids
            if candidate_id != assigned
        )


def test_dormant_clock_is_serialized_unchanged_not_advanced_at_zero_rate():
    runtime, candidates = runtime_fixture()
    front = runtime.active_front_ids[0]
    active_id, dormant_id = (item.candidate_id for item in candidates)
    front_state = replace(
        runtime.front_runtimes[front],
        mechanically_active_candidate_ids=(active_id,),
    )
    runtime = replace(runtime, front_runtimes={front: front_state})
    before = competition_state_from_dict(front_state.competition_state)
    dormant_before = next(
        item for item in before.hazard_states if item.candidate_id == dormant_id
    )
    preview = preview_competitions(
        runtime, {front: {active_id: 1.1}},
        {front: {active_id: (1e-6, 0.0)}},
        start_time_s=0.0, duration_s=1.0, correlation_interval_s=0.0,
    )
    after = competition_state_from_dict(
        competition_state_to_dict(preview.fronts[front].previewed_competition)
    )
    dormant_after = next(
        item for item in after.hazard_states if item.candidate_id == dormant_id
    )
    assert dormant_after == dormant_before
    assert dormant_after.previous_rate_per_s == dormant_before.previous_rate_per_s


def test_exact_trial_commit_and_rollover_use_exact_network(tmp_path):
    runtime, candidates = runtime_fixture()
    front = runtime.active_front_ids[0]
    start = runtime.crack_network.branch(front).tip
    end = (1e-6, 0.0)
    proposal = TopologyProposal(
        "proposal:1", "one_arm", front, runtime.owner_by_front[front],
        (candidates[0].candidate_id,), 0.5, (end,),
        member_event_ids=("event:1",), member_event_ordinals=(1,),
    )
    arm = TopologyArm(candidates[0].candidate_id, front, start, end, 1e-6, 0.2)
    exact_network = extend_network_arm(runtime.crack_network, arm)
    accepted = fem_state(runtime)
    accepted.crack_network = exact_network
    topology = hashlib.sha256(exact_network.to_json().encode()).hexdigest()
    outcome = ProposalTrialOutcome(
        proposal, True, accepted, "accepted",
        exact_realized_crack_network=exact_network,
        exact_topology_transaction={"action_id": proposal.proposal_id},
        realized_endpoints_m=(end,), realized_arm_lengths_m=(1e-6,),
        wake_mutation={"changed_elements": 2},
        stored_energy_release_J_per_m=1.0,
        stored_energy_cost_J_per_m=0.2,
        topology_fingerprint=topology, geometry_fingerprint="geometry:1",
        accepted_state_fingerprint=accepted_fem_state_fingerprint(accepted),
    )
    committed = commit_exact_trial_outcome(runtime, outcome)
    assert committed.crack_network == exact_network
    assert committed.accepted_state_id != runtime.accepted_state_id
    record = committed.transaction_records[-1]
    assert record.proposal_id == "proposal:1"
    assert record.member_event_ids == ("event:1",)
    assert record.realized_endpoints_m == (end,)
    assert record.accepted_fem_topology_fingerprint == topology
    target = tmp_path / "production.json"
    write_production_multifront_checkpoint(
        committed, target, accepted_fem_state=accepted,
    )
    restored = load_production_multifront_checkpoint(target)
    assert restored.runtime.to_dict() == committed.to_dict()
    assert accepted_fem_state_fingerprint(restored.accepted_fem_state) == accepted_fem_state_fingerprint(accepted)


def test_exact_binary_trial_preserves_realized_v11_daughter_ids():
    runtime, candidates = runtime_fixture()
    front = runtime.active_front_ids[0]
    tip = runtime.crack_network.branch(front).tip
    ends = ((1e-6, -1e-7), (1e-6, 1e-7))
    proposal = TopologyProposal(
        "proposal:pair", "two_arm", front, runtime.owner_by_front[front],
        tuple(item.candidate_id for item in candidates), 0.5, ends,
        member_event_ids=("event:up", "event:down"), member_event_ordinals=(1, 1),
    )
    rebuilt = commit_selected_proposal(runtime, proposal).crack_network
    generated = rebuilt.active_tip_ids
    remap = dict(zip(generated, ("v11-daughter-A", "v11-daughter-B")))
    branches = tuple(
        replace(
            branch, branch_id=remap.get(branch.branch_id, branch.branch_id),
            parent_branch_id=remap.get(branch.parent_branch_id, branch.parent_branch_id),
        ) for branch in rebuilt.branches
    )
    from arrhenius_fracture.crack_network_v11 import CrackNetworkState
    exact = CrackNetworkState(
        branches=branches, primary_branch_id=rebuilt.primary_branch_id,
        geometry_generation=rebuilt.geometry_generation, branching_enabled=True,
    )
    accepted = fem_state(runtime); accepted.crack_network = exact
    topology = hashlib.sha256(exact.to_json().encode()).hexdigest()
    result = commit_exact_trial_outcome(runtime, ProposalTrialOutcome(
        proposal, True, accepted, "accepted",
        exact_realized_crack_network=exact,
        exact_topology_transaction={"action_id": "proposal:pair"},
        created_front_ids=exact.active_tip_ids, retired_front_ids=(front,),
        junction_id="v11-junction-exact", realized_endpoints_m=ends,
        realized_arm_lengths_m=tuple(np.linalg.norm(np.subtract(end, tip)) for end in ends),
        topology_fingerprint=topology,
    ))
    assert set(result.active_front_ids) == set(exact.active_tip_ids)
    assert set(result.front_runtimes) == set(exact.active_tip_ids)
    assert "v11-junction-exact" in result.junctions
    assert result.transaction_records[-1].created_front_ids == tuple(sorted(exact.active_tip_ids))


def test_rollover_rejects_stale_topology():
    runtime, _ = runtime_fixture()
    accepted = fem_state(runtime)
    rolled = rollover_accepted_state(runtime, accepted)
    assert rolled.accepted_state_id.startswith("accepted:")
    other, _ = runtime_fixture()
    branch = other.crack_network.branch(other.active_front_ids[0])
    changed = replace(branch, path=branch.path + ((1e-6, 0.0),), orientation_history_rad=(0.0,))
    from arrhenius_fracture.crack_network_v11 import CrackNetworkState
    accepted.crack_network = CrackNetworkState(
        (changed,), other.crack_network.primary_branch_id, 1, True,
    )
    with pytest.raises(RuntimeError, match="topology differ"):
        rollover_accepted_state(runtime, accepted)


def test_hook_factory_has_all_leaf_bindings_and_fails_closed_at_composition():
    mapping = current_source_hook_map()
    assert mapping["accepted_mesh_adaptation"]["status"] == "QUALIFIED"
    assert mapping["accepted_fem_solve"]["status"] == "QUALIFIED"
    assert all(row["status"] == "QUALIFIED" for row in mapping.values())
    with pytest.raises(CurrentSourceHookBindingError, match="composition"):
        build_current_source_production_hooks()


@pytest.mark.parametrize("relative", (
    "PF-fracture-fatigue_current_source_branching_integrated_envelope_v5_4_step1_migrations_20260901T130000Z_final/enabled_max2/step0000001_migrated_v5_4.json",
    "PF-fracture-fatigue_current_source_branching_completion_v5_4_2_20260901T204630Z/theta40_v5_4_2_enabled_max2_seed3621/checkpoint/transitions/step0000370_mesh_adaptation_g0099.json",
    "PF-fracture-fatigue_current_source_branching_completion_v5_4_2_20260901T204630Z/theta40_v5_4_2_enabled_max2_seed3621/checkpoint/latest.json",
))
def test_immutable_v5_4_2_checkpoints_restore_exact_physical_engine(relative):
    source = restore_branch_checkpoint(
        "/Volumes/Data/Data/Nanopillar_calculation/" + relative
    )
    runtime = load_v11_checkpoint_as_v12(
        source,
        resource_policy=ResourcePolicy(
            "mechanistic", None, None, "experimental",
            "v11_correlated_proposal_compatibility",
        ),
    )
    assert len(runtime.process_engines) == 1
    owner = next(iter(runtime.process_engines.values()))
    before = owner.complete_checkpoint_payload()
    restored = restore_complete_current_source_engine(owner)
    after = _capture_shared_engine(restored)
    assert process_state_digest(after) == process_state_digest(before)
    assert rng_digest(after) == rng_digest(before)
    assert getattr(restored, "_restart_family_migration_count", 0) == 1
    assert restored._state_kernel_family is restored.mpz._signed_kernel


def test_actual_v6_2_cli_dry_run_restores_n1_n2_and_starts_zero_workers(tmp_path):
    base = "/Volumes/Data/Data/Nanopillar_calculation/"
    result = v6_2_production_adapter_dry_run((
        "--v6-2-production-adapter-dry-run",
        "--branching-mode", "mechanistic",
        "--shared-region-branching", "experimental",
        "--v5-4-2-n1-checkpoint",
        base + "PF-fracture-fatigue_current_source_branching_integrated_envelope_v5_4_step1_migrations_20260901T130000Z_final/enabled_max2/step0000001_migrated_v5_4.json",
        "--v5-4-2-n2-checkpoint",
        base + "PF-fracture-fatigue_current_source_branching_completion_v5_4_2_20260901T204630Z/theta40_v5_4_2_enabled_max2_seed3621/checkpoint/latest.json",
        "--fresh-output-destination", str(tmp_path / "new-output"),
        "--fresh-cache-destination", str(tmp_path / "new-cache"),
    ))
    assert result["qualification"] == "FAIL_CLOSED"
    assert result["provider_lookup_count"] == 0
    assert result["mechanics_solve_count"] == 0
    assert not result["pf_worker_started"]
    assert set(result["inventories"]) == {"N1", "N2"}
    assert all(
        owner["process_state_sha256_before"] == owner["process_state_sha256_after"]
        and owner["rng_sha256_before"] == owner["rng_sha256_after"]
        for case in result["inventories"].values() for owner in case["owners"]
    )
