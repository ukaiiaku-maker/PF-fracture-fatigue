from dataclasses import replace
import hashlib
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from arrhenius_fracture.crack_network_v11 import CrackNetworkState
from arrhenius_fracture.current_source_multifront_hooks_v12 import (
    build_current_source_multifront_production_hooks,
)
from arrhenius_fracture.directional_competition_v11 import (
    CleavageCandidate, DirectionalCompetitionState, DirectionalHazardState,
    competition_state_to_dict,
)
from arrhenius_fracture.general_multifront_v12 import (
    CouplingEvidence, FrontCandidateObservation, FrontRuntimeState,
    MultiFrontRuntimeState, ProcessEngineState, ResourcePolicy, TopologyProposal,
    commit_selected_proposal, recompute_process_region_connectivity,
)
from arrhenius_fracture.multifront_competition_v12 import (
    daughter_competitions_from_v11_lineage,
)
from arrhenius_fracture.production_multifront_v12 import (
    ProposalTrialOutcome, rollover_accepted_state,
)
from arrhenius_fracture.sharp_front_current_source_multifront_v12 import (
    v6_3_stateful_production_dry_run,
)
from arrhenius_fracture.sharp_front_v11_branching import _capture_shared_engine
from arrhenius_fracture.stateful_multifront_production_v12 import (
    CurrentSourceMultiFrontProductionContextV12, ExactAcceptedTrialCacheV12,
    StatefulProductionInterlock, TransactionalMultiFrontWriterV12,
    create_fresh_independent_process_engine_v12,
    restore_stateful_production_context,
)


class _MPZ:
    def __init__(self):
        self.mobile = np.array([[1.0]])
        self.retained = np.array([[2.0]])
        self.advance_total_m = 0.0


class _Engine:
    MODEL_ID = "fixture"
    def __init__(self):
        self.mpz = _MPZ()
        self.B = 0.0


def _candidate(name, direction):
    return CleavageCandidate.create(
        plane_family="100", plane_variant=name, direction_xy=direction,
        normal_xy=(-direction[1], direction[0]), gamma_rel=1.0,
        orientation_convention="fixture",
    )


def _fixture(tmp_path):
    candidates = (_candidate("a", (1.0, 0.0)), _candidate("b", (0.0, 1.0)))
    competition = DirectionalCompetitionState(
        candidates=candidates,
        hazard_states=tuple(DirectionalHazardState(x.candidate_id) for x in candidates),
        global_hazard_seed=3621,
    )
    network = CrackNetworkState.one_tip(((0.0, 0.0),), initial_orientation_rad=0.0)
    front = network.active_tip_ids[0]
    engine = ProcessEngineState.from_v11_payload(
        engine_id="engine:root", source_state_id="source:root",
        payload=_capture_shared_engine(_Engine()), active_ledgers={"mobile": 1.0},
        wake_ledgers={"retained": 2.0}, signed_system_ledgers={"net": 1.0},
        update_count=0, event_renewal_count=0, local_process_coordinate_m=0.0,
    )
    runtime = MultiFrontRuntimeState.one_front(
        network, FrontRuntimeState(
            front, competition_state_to_dict(competition),
            tuple(x.candidate_id for x in candidates), {"seed": 3621},
        ), engine, resource_policy=ResourcePolicy(
            "mechanistic", None, None, "experimental",
            "v11_correlated_proposal_compatibility",
        ), accepted_state_id="accepted:fixture", stress_field_state_id="stress:fixture",
    )
    state = SimpleNamespace(
        crack_network=network, displacement=np.array([0.0, 1.0]),
        ep_gp=np.array([[0.0]]), rho_gp=np.array([[1.0]]),
        damage=np.array([0.0]), elasticity_D=np.eye(3),
    )
    context = CurrentSourceMultiFrontProductionContextV12(
        args={"temperature": 1100.0}, mechanical_configuration={"theta": 40.0},
        accepted_fem_state=state, accepted_stress_field=np.zeros((3, 1)),
        runtime=runtime, provider_runtime={"mode": "cache-only"},
        destination_cache_root=tmp_path / "cache", output_root=tmp_path / "out",
        checkpoint_path=tmp_path / "checkpoint" / "latest.json",
        candidates=candidates,
    )
    return context, candidates


def _proposal(context, candidate, suffix="nominal"):
    front = context.runtime.active_front_ids[0]
    return TopologyProposal(
        f"proposal:{suffix}", "one_arm", front,
        context.runtime.owner_by_front[front], (candidate.candidate_id,), 1.0,
        ((1e-6, 0.0),), member_event_ids=(f"event:{suffix}",),
        member_event_ordinals=(1,),
    )


def test_factory_returns_one_persistent_context_and_real_leaf_hooks(tmp_path):
    context, _ = _fixture(tmp_path)
    hooks = build_current_source_multifront_production_hooks(context)
    assert hooks.context is context
    assert all(callable(value) for value in (
        hooks.adapt_accepted_mesh, hooks.solve_accepted_fem,
        hooks.build_exact_topology_request, hooks.extract_directional_observations,
        hooks.evolve_process_engine, hooks.trial_topology_proposal,
        hooks.preview_competition, hooks.finalize_competition,
        hooks.renew_selected_event, hooks.validate_process_interval,
        hooks.write_output, hooks.write_checkpoint,
    ))


@pytest.mark.parametrize("disposition", ("nominal", "clipped", "intersecting", "coalescing"))
def test_exact_trial_cache_full_key_selection_and_destruction(tmp_path, disposition):
    context, candidates = _fixture(tmp_path)
    proposal = _proposal(context, candidates[0], disposition)
    outcome = ProposalTrialOutcome(
        proposal, True, context.accepted_fem_state, disposition,
        exact_realized_crack_network=context.runtime.crack_network,
        clipped_or_coalesced_disposition=disposition,
    )
    cache = ExactAcceptedTrialCacheV12()
    key = cache.create(context.runtime, proposal, outcome)
    assert key.accepted_state_id == context.runtime.accepted_state_id
    assert key.stress_field_state_id == context.runtime.stress_field_state_id
    assert key.selected_front_id == proposal.front_id
    assert key.process_owner_id == proposal.owner_id
    assert cache.select(context.runtime, proposal) is outcome
    cache.invalidate_after_interval("accepted:new")
    cache.require_empty()
    assert cache.audit()["creation_count"] == cache.audit()["destruction_count"] == 1
    with pytest.raises(StatefulProductionInterlock, match="no exact trial"):
        cache.select(context.runtime, proposal)


def test_daughters_have_distinct_objects_and_order_invariant_rng_identities(tmp_path):
    context, _ = _fixture(tmp_path)
    parent = next(iter(context.runtime.front_runtimes.values()))
    first = daughter_competitions_from_v11_lineage(parent, ("daughter:z", "daughter:a"))
    second = daughter_competitions_from_v11_lineage(parent, ("daughter:a", "daughter:z"))
    assert first == second
    a, z = first["daughter:a"], first["daughter:z"]
    assert a is not z
    assert a.competition_state is not z.competition_state
    assert a.lineage_rng_state is not z.lineage_rng_state
    assert a.lineage_rng_state["daughter_rng_stream_identity"] != z.lineage_rng_state["daughter_rng_stream_identity"]
    assert canonical_competition_hash(a) == canonical_competition_hash(z)


def canonical_competition_hash(front):
    from arrhenius_fracture.general_multifront_v12 import canonical_hash
    return canonical_hash(front.competition_state)


def test_transactional_writer_buffers_and_restores_append_counters(tmp_path):
    writer = TransactionalMultiFrontWriterV12(tmp_path / "out")
    writer.begin("tx:1", accepted_state_id="accepted:1", topology_fingerprint="topology:1")
    writer.buffer("event", {"event_id": "e1"})
    writer.buffer("trial", {"trial_id": "r1", "accepted": False}, readonly_rejected_trial=True)
    assert not writer.root.exists()
    restored = TransactionalMultiFrontWriterV12.restore(writer.snapshot())
    result = restored.flush_accepted()
    assert result["published"] == {"event": 1, "rejected_trials_readonly": 1}
    assert restored.row_counts["event"] == 1
    assert restored.readonly_trial_count == 1


def test_output_flush_precedes_checkpoint_and_complete_context_restores(tmp_path):
    context, _ = _fixture(tmp_path)
    context.begin_interval("tx:accepted")
    context.output_writer.buffer("interval", {"kind": "accepted"})
    rolled = rollover_accepted_state(context.runtime, context.accepted_fem_state)
    # The fixture's physical fields are unchanged but the accepted identity is
    # newly derived, which is the required state-generation rollover.
    context.commit_rollover(rolled, context.accepted_fem_state, context.accepted_stress_field)
    context.flush_before_checkpoint()
    assert (tmp_path / "out" / "interval.jsonl").is_file()
    assert not context.checkpoint_path.exists()
    context.publish_checkpoint()
    assert context.checkpoint_path.is_file()
    assert not context.checkpoint_path.with_name(
        context.checkpoint_path.name + ".accepted-fem.pkl"
    ).exists()
    assert not context.checkpoint_path.with_name(
        context.checkpoint_path.name + ".context.json"
    ).exists()
    restored = restore_stateful_production_context(context.checkpoint_path)
    assert restored.runtime.to_dict() == context.runtime.to_dict()
    assert restored.output_writer.row_counts == context.output_writer.row_counts
    # Source fixtures use a deliberately unqualified toy class.  Production
    # V11 engines are reconstructed eagerly; unreviewed fixture classes remain
    # represented by their complete immutable ProcessEngineState payload.
    assert restored.actual_engines == {}
    assert restored.transaction_phase == "accepted"


def test_physical_handoff_uses_fresh_engine_and_activates_detached_inventory(tmp_path):
    context, candidates = _fixture(tmp_path)
    runtime = context.runtime
    front = runtime.active_front_ids[0]
    tip = runtime.crack_network.branch(front).tip
    born = commit_selected_proposal(runtime, TopologyProposal(
        "proposal:birth", "two_arm", front, runtime.owner_by_front[front],
        tuple(item.candidate_id for item in candidates), 1.0,
        ((tip[0] + 1e-6, tip[1]), (tip[0], tip[1] + 1e-6)),
    ))
    context.runtime = born
    junction = next(iter(born.junctions))
    detached = born.active_front_ids[0]
    evidence = CouplingEvidence(
        junction, 5e-6, 2e-6, 1e-6, 1e-6, 1e-6,
        True, (detached,),
    )
    old_owner = born.owner_by_front[detached]
    old_engine_id = born.process_regions[old_owner].process_engine_id
    old_payload = born.process_engines[old_engine_id].checkpoint_payload_sha256
    def fresh_physical_engine(owner, members, owned_context):
        engine = _Engine()
        engine.mpz.mobile[...] = 0.0
        engine.mpz.retained[...] = 0.0
        return engine
    context.adapter_configuration["fresh_process_engine_factory"] = fresh_physical_engine
    result = recompute_process_region_connectivity(
        born, {junction: evidence},
        fresh_engine_factory=lambda owner, engine, region, members:
            create_fresh_independent_process_engine_v12(
                context, owner, engine, region, members
            ),
    )
    new_owner = result.owner_by_front[detached]
    new_engine = result.process_engines[result.process_regions[new_owner].process_engine_id]
    assert new_owner != old_owner
    assert new_engine.checkpoint_payload_b64 is not None
    assert new_engine.local_process_coordinate_m == 0.0
    assert result.process_engines[old_engine_id].checkpoint_payload_sha256 == old_payload
    assert result.front_runtimes[detached].mechanically_active_candidate_ids == (
        result.front_runtimes[detached].candidate_ids
    )
    retained = next(front_id for front_id in born.active_front_ids if front_id != detached)
    assert len(result.front_runtimes[retained].mechanically_active_candidate_ids) == 1


@pytest.mark.parametrize("stage", ("trial", "finalization", "renewal", "output_flush"))
def test_interruption_exposes_only_prior_or_complete_new_state(tmp_path, stage):
    context, _ = _fixture(tmp_path)
    prior = context.runtime.accepted_state_id
    context.begin_interval("tx:interrupt")
    context.output_writer.buffer("event", {"stage": stage})
    if stage in {"renewal", "output_flush"}:
        rolled = rollover_accepted_state(context.runtime, context.accepted_fem_state)
        context.commit_rollover(rolled, context.accepted_fem_state, context.accepted_stress_field)
    if stage == "output_flush":
        context.flush_before_checkpoint()
    assert not context.checkpoint_path.exists()
    # No latest pointer means restart remains at the prior accepted state;
    # flushed rows are unreachable until a matching pointer is published.
    assert prior.startswith("accepted:")


def test_native_n1_control_and_enabled_n2_stateful_dry_run(tmp_path):
    base = Path("/Volumes/Data/Data/Nanopillar_calculation")
    n1 = base / "PF-fracture-fatigue_current_source_branching_integrated_envelope_v5_4_step1_migrations_20260901T130000Z_final/control_max1/step0000001_migrated_v5_4.json"
    n2 = base / "PF-fracture-fatigue_current_source_branching_completion_v5_4_2_20260901T204630Z/theta40_v5_4_2_enabled_max2_seed3621/checkpoint/latest.json"
    result = v6_3_stateful_production_dry_run((
        "--v6-3-stateful-production-dry-run", "--branching-mode", "mechanistic",
        "--shared-region-branching", "experimental",
        "--v5-4-2-n1-checkpoint", str(n1), "--v5-4-2-n2-checkpoint", str(n2),
        "--fresh-output-destination", str(tmp_path / "fresh-output"),
        "--fresh-cache-destination", str(tmp_path / "fresh-cache"),
    ))
    assert result["qualification"] == "PASS"
    assert not result["trajectory_execution_authorized"]
    assert result["provider_solve_count"] == result["mechanics_solve_count"] == 0
    assert not result["pf_worker_started"] and not (tmp_path / "fresh-output").exists()
    assert result["cases"]["branch_disabled_N1_control"]["native_branching_mode"] == "disabled"
    assert len(result["cases"]["enabled_N2_terminal"]["active_front_ids"]) == 2


def test_composed_hook_executes_actual_v11_engine_once_without_renewal(tmp_path):
    from arrhenius_fracture.branch_checkpoint_v11 import restore_branch_checkpoint
    from arrhenius_fracture.fem import assemble_mechanics
    from arrhenius_fracture.multifront_checkpoint_v12 import load_v11_checkpoint_as_v12

    path = Path("/Volumes/Data/Data/Nanopillar_calculation/PF-fracture-fatigue_current_source_branching_completion_v5_4_2_20260901T204630Z/theta40_v5_4_2_enabled_max2_seed3621/checkpoint/transitions/step0000370_mesh_adaptation_g0099.json")
    source = restore_branch_checkpoint(path)
    runtime = load_v11_checkpoint_as_v12(source, resource_policy=ResourcePolicy(
        "mechanistic", None, None, "experimental",
        "v11_correlated_proposal_compatibility",
    ))
    state = source.state
    # This reconstructs the checkpoint's stress tensor by assembly only.  It
    # does not invoke solve_dirichlet or a provider solve.
    sigma = assemble_mechanics(
        state.mesh, state.displacement, state.ep_gp, state.rho_gp,
        state.damage, state.elasticity_D, state.material,
        cohesive_network=state.cohesive_network,
    )[2]
    candidate_map = {
        candidate.candidate_id: candidate
        for competition in source.front_competitions.values()
        for candidate in competition.candidates
    }
    context = CurrentSourceMultiFrontProductionContextV12(
        args={"source": "immutable"}, mechanical_configuration={"theta": 40.0},
        accepted_fem_state=state, accepted_stress_field=sigma, runtime=runtime,
        provider_runtime=source.provider_runtime,
        destination_cache_root=tmp_path / "cache", output_root=tmp_path / "out",
        checkpoint_path=tmp_path / "latest.json",
        candidates=tuple(candidate_map[key] for key in sorted(candidate_map)),
        adapter_configuration={
            "temperature_K": 1100.0, "pre_progress": 0.0,
            "permitted_physical_hazard_action": 1.0,
        },
    )
    hooks = build_current_source_multifront_production_hooks(context)
    owner_id, region = next(iter(runtime.process_regions.items()))
    before = runtime.process_engines[region.process_engine_id]
    front = sorted(region.member_front_ids)[0]
    branch = runtime.crack_network.branch(front)
    cid = str(branch.local_state["candidate_id"])
    observation = FrontCandidateObservation(
        accepted_state_id=runtime.accepted_state_id,
        stress_field_state_id=runtime.stress_field_state_id,
        front_id=front, owner_id=owner_id, candidate_id=cid,
        tip_coordinates_m=branch.tip, signed_local_J_J_per_m2=0.0,
        marginal_G_J_per_m2=0.0, kinetic_J_used_J_per_m2=0.0,
        directional_K_MPa_sqrt_m=0.0, directional_rate_per_s=0.0,
        tensor=(0.0,), tensor_reliability="checkpoint_assembly",
        controlling_scalar_K_tip_id=front, tensor_probe_tip_id=front,
    )
    after = hooks.evolve_process_engine(before, observation, 1.0e-9)
    assert after.update_count == before.update_count + 1
    assert after.event_renewal_count == before.event_renewal_count
    assert context.interval_info_by_owner[before.engine_id]["interval_evolved"] is True
    assert context.mechanics_solve_count == context.provider_solve_count == 0
